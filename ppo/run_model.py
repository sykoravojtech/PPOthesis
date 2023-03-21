import os
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from os import truncate
import gym
import numpy as np
import csv
from typing import Tuple, List

from my_parser import create_parser
from PPO import PPO
from wrappers import *
from utils import *
from model_paths import *

def make_single_env(args: argparse.Namespace) -> EvaluationEnv:
    """ Create a non-vectorized environment with the basic wrappers applied
    
    Args:
        args: arguments 
    """
    if args.render:
        env = gym.make(args.env, render_mode="human", continuous = True)
    else:
        env = gym.make(args.env, render_mode='rgb_array', continuous = True)
            
    env = NormalizeObservation(env)
    env = ClippedAction(env, low=0, high=1)
    env = EvaluationEnv(env, render_each=args.render_each, evaluate_for=args.evaluate_for, report_each=args.report_each)
    return env


def make_env(args: argparse.Namespace) -> Tuple[EvaluationEnv, gym.vector.SyncVectorEnv]:
    """ Create a vectorized environment with the correct wind type

    Args:
        args: arguments
    """
    single_env: EvaluationEnv = make_single_env(args)
    
    if args.wind_strength != None or args.wind_range != None or args.nowind_range != None:
        # add wind with specified parameters
        params = {}
        if args.wind_strength is not None:
            params["strength"] = args.wind_strength
        if args.wind_range is not None:
            params["wind_step_range"] = args.wind_range
        if args.nowind_range is not None:
            params["nonwind_step_range"] = args.nowind_range
        single_env = add_wind_wrapper(args.wind_wrapper, single_env, params)
    else:
        # add the default version of a wind
        single_env = add_wind_wrapper(args.wind_wrapper, single_env)
    
    # transform our single environment into a vectorized version
    env = gym.vector.SyncVectorEnv([lambda: single_env])
    
    return single_env, env
    
    
def run_single_model(args: argparse.Namespace, load_path: str) -> Tuple[float]:
    """ Run/evaluate a single model

    Args:
        args (argparse.Namespace): _description_
        load_path (str): _description_
    """
    # create a vectorized and non-vectorized environment
    single_env, env = make_env(args)
    
    # print_info(single_env, args)
    
    # create a PPO agent
    ppo: PPO = PPO(observation_space = env.observation_space, 
              action_space = env.action_space, 
              entropy_coeff = args.entropy_coeff,
              gamma = args.gamma,
              gae_lambda = args.gae_lambda,
              learning_rate = args.learning_rate,
              value_fun_coeff = args.vf_coeff)

    ppo.load_weights(load_path, verbose = True)
    
    ppo.run(
        env,
        single_env, 
        num_of_episodes = args.running_episodes,
        render = args.render,
        record = args.record
        )

    print()
    # print(single_env.get_mean_std(verbose=True))
    return single_env.get_mean_std()    
    
def run_multiple_models(args: argparse.Namespace, models_paths: List[str]) -> Tuple[List[float], List[float]]:
    """ Run all inputted models sequentially on the same environment

    Args:
        args: arguments
        models_paths: list of paths to the models
    """
    means = [] # mean/average
    stds = [] # standard deviation
    
    for path in models_paths:
        mean, std = run_single_model(args, path) # PATHS[PATHS_INDEX]
        means.append(mean)
        stds.append(std)
        print_divider()
        
    return means, stds  
        
# render models from one training which were after each other
def run_following_models(args: argparse.Namespace, first_episode: int, last_episode: int, base_dir: str):
    """ Run models saved after each other during training
    This is mainly to visualize how the model learned

    Args:
        args: arguments
        first_episode: from which episode weights should we start
        last_episode: at which episode weights should we end
        base_dir: the directory where all models are saved
    """
    episodes: List[int] = list(range(first_episode, last_episode + 1, 10))
    for ep in episodes:
        path: str = os.path.join(base_dir, f"ep{ep}", f"ep{ep}_weights")
        run_single_model(args, path) # PATHS[PATHS_INDEX]

# *************************************
# TODO CONTINUE HERE WITH CLEANUP
# *************************************
def write_to_csv(filename: str, rows):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def run_single_model_on_all_envs(args, load_path, save_csv = False):
    single_env, env = make_env(args)
    
    ppo = PPO(observation_space = env.observation_space, 
              action_space = env.action_space, 
              entropy_coeff = args.entropy_coeff,
              gamma = args.gamma,
              gae_lambda = args.gae_lambda,
              learning_rate = args.learning_rate,
              value_fun_coeff = args.vf_coeff)

    ppo.load_weights(load_path, verbose = True)
    
    means, stds = [], []
    
    for env in ALLENVS:
        args.wind_wrapper = env[0]
        args.wind_strength = env[1]
        single_env, env = make_env(args)
        
        ppo.run(
            env,
            single_env, 
            num_of_episodes = args.running_episodes,
            render = args.render,
            record = args.record
            )
        
        mean, std = single_env.get_mean_std()
        means.append(mean)
        stds.append(std)
        
        print_divider()
        print()
        
    if save_csv:
        print(f"{bcolors.GREEN} Saving to {load_path[5:].replace('/', '-')}.csv{bcolors.ENDC}")
        write_to_csv(f"{load_path[5:].replace('/', '-')}.csv", [ALLENVS_NAMES, round_list(means, 2), round_list(stds, 2)])

    return means, stds

def run_multiple_models_on_all_envs(args, model_paths, save_csv = None, csv_style = "separate"):
    if csv_style == "separate":
        for path in model_paths:
            print(f"\n{bcolors.RED}{'*' * 30}{bcolors.ENDC}")
            print_chapter_style(f"NEW MODEL: {path[5:]}")
            if save_csv is not None:
                run_single_model_on_all_envs(args, path, save_csv = True)
            else:
                run_single_model_on_all_envs(args, path)
    
    else: # together, combined, one large table
        table_means = [ALLENVS_NAMES]
        table_stds = [ALLENVS_NAMES]
        first_col = ["."]
        
        for path in model_paths:
            print(f"\n{bcolors.RED}{'*' * 30}{bcolors.ENDC}")
            print_chapter_style(f"NEW MODEL: {path[5:]}")
            means, stds = run_single_model_on_all_envs(args, path, save_csv = None)
            table_means.append(round_list(means, 2))
            table_stds.append(round_list(stds, 2))
            first_col.append(f"{path[5:]}")
        
        write_to_csv(f"means_{save_csv}", add_col_to_start(table_means, first_col))
        write_to_csv(f"stds_{save_csv}", add_col_to_start(table_stds, first_col))
 
    
if __name__ == '__main__':
    # show_terminal_colors()
    
    # print_tensorflow_version()
    # print_available_devices()
        
    args = create_parser().parse_args([] if "__file__" not in globals() else None)
    
    RUN_SINGLE_MODEL = True
    if RUN_SINGLE_MODEL:
        mean, std = run_single_model(args, NOWIND_MODEL)
        # print(f"mean={mean:.2f} std = {std:.2f}")
    
    RUN_FOLLOWING_MODELS = False
    if RUN_FOLLOWING_MODELS:
        run_following_models(
            args, 
            first_episode = 10, 
            last_episode = 260, 
            base_dir =  "archive/PRETgustySides/4to5")
    
    # run_single_model(args, "archive/left/3to4/ep80/ep80_weights")
    
    # means, stds = run_multiple_models(args, PATHS)
    # names = [get_name_of_last_dir(path) for path in PATHS]
    # means, stds = run_single_model_on_all_envs(args, args.load_model, save_csv = True)
    
    # run_multiple_models_on_all_envs(args, MODELS, save_csv = True)
    # run_multiple_models_on_all_envs(args, MODELS, save_csv = "TABLE.csv", csv_style = "combined")
    
    
    
    

    