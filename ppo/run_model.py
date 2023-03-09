import os
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from os import truncate
import gym
from gym.wrappers import RecordVideo
import numpy as np

from my_parser import create_parser
from PPO import PPO
from wrappers import *
from utils import *
import csv

import car_racing_environment # f"CarRacingFS{skip_frames}-v2"

RUN_FOR = 1


PUREENV_MODEL = 'BEST/pureEnv/projectBEST'
LEFT_MODELS = [
    'BEST/left/ep690_1to2',
    'BEST/left/ep580_2to3',
    'BEST/left/ep640_3to4',
    'BEST/left/ep560_4to5'
]
GUSTYLEFT_MODELS = [
    'BEST/gustyLeft/ep530_1to2',
    'BEST/gustyLeft/ep390_2to3',
    'BEST/gustyLeft/ep680_3to4',
    'BEST/gustyLeft/ep690_4to5'
]
RIGHT_MODELS = [
    'BEST/right/ep680_1to2',
    'BEST/right/ep680_2to3',
    'BEST/right/ep670_3to4',
    'BEST/right/ep680_4to5'
]
GUSTYRIGHT_MODELS = [
    'BEST/gustyRight/ep670_1to2',
    'BEST/gustyRight/ep650_2to3',
    'BEST/gustyRight/ep670_3to4',
    'BEST/gustyRight/ep670_4to5'
]
SIDES_MODELS = [
    'BEST/sides/ep400_1to2',
    'BEST/sides/ep420_2to3',
    'BEST/sides/ep820_3to4',
    'BEST/sides/ep400_4to5'
]
GUSTYSIDES_MODELS = [
    'BEST/gustySides/ep830_1to2',
    'BEST/gustySides/ep1240_2to3',
    'BEST/gustySides/ep760_3to4',
    'BEST/gustySides/ep780_4to5'
]
MODELS = [
    PUREENV_MODEL,
    *LEFT_MODELS,
    *GUSTYLEFT_MODELS,
    *RIGHT_MODELS,
    *GUSTYRIGHT_MODELS,
    *SIDES_MODELS,
    *GUSTYSIDES_MODELS,
    
    # 'BEST/left/ep810_0.1,0.2(358)',
    # 'BEST/gustyLeft/ep960_0.1,0.2(500)',
    # 'BEST/right/ep1220_0.1,0.2(435)',
    # 'BEST/gustyRight/ep780_0.1,0.2(275)',
]
MODELS_INDEX = 0

ALLENVS = [ # (wind_wrapper, strength, name_for_1st_column_of_csv)
    (None, None, "pureEnv"),
    ("left", [0.1, 0.2], "left1to2"),
    ("left", [0.2, 0.3], "left2to3"),
    ("left", [0.3, 0.4], "left3to4"),
    ("left", [0.4, 0.5], "left4to5"),
    ("gustyLeft", [0.1, 0.2], "gustyLeft1to2"),
    ("gustyLeft", [0.2, 0.3], "gustyLeft2to3"),
    ("gustyLeft", [0.3, 0.4], "gustyLeft3to4"),
    ("gustyLeft", [0.4, 0.5], "gustyLeft4to5"),
    ("right", [0.1, 0.2], "right1to2"),
    ("right", [0.2, 0.3], "right2to3"),
    ("right", [0.3, 0.4], "right3to4"),
    ("right", [0.4, 0.5], "right4to5"),
    ("gustyRight", [0.1, 0.2], "gustyRight1to2"),
    ("gustyRight", [0.2, 0.3], "gustyRight2to3"),
    ("gustyRight", [0.3, 0.4], "gustyRight3to4"),
    ("gustyRight", [0.4, 0.5], "gustyRight4to5"),
    ("sides", [0.1, 0.2], "sides1to2"),
    ("sides", [0.2, 0.3], "sides2to3"),
    ("sides", [0.3, 0.4], "sides3to4"),
    ("sides", [0.4, 0.5], "sides4to5"),
    ("gustySides", [0.1, 0.2], "gustySides1to2"),
    ("gustySides", [0.2, 0.3], "gustySides2to3"),
    ("gustySides", [0.3, 0.4], "gustySides3to4"),
    ("gustySides", [0.4, 0.5], "gustySides4to5"),
]
ALLENVS_NAMES = [env[2] for env in ALLENVS]


def make_single_env(args):
    if args.render:
        env = gym.make(args.env, render_mode="human", continuous = True)
    else:
        env = gym.make(args.env, render_mode='rgb_array', continuous = True)
    
    # print(env.action_space)
        
    env = NormalizeObservation(env)
    env = ClippedAction(env, low=0, high=1)
    env = EvaluationEnv(env, render_each=args.render_each, evaluate_for=args.evaluate_for, report_each=args.report_each)
    return env

def make_env(args):
    single_env = make_single_env(args)
    
    if args.wind_strength != None or args.wind_range != None or args.nowind_range != None:
        params = {}
        if args.wind_strength is not None:
            params["strength"] = args.wind_strength
        if args.wind_range is not None:
            params["wind_step_range"] = args.wind_range
        if args.nowind_range is not None:
            params["nonwind_step_range"] = args.nowind_range
        single_env = add_wind_wrapper(args.wind_wrapper, single_env, params)
    else:
        single_env = add_wind_wrapper(args.wind_wrapper, single_env)
    
    env = gym.vector.SyncVectorEnv([lambda: single_env])
    
    return single_env, env
    
def run_single_model(args, load_path):
    
    single_env, env = make_env(args)
    
    # print_info(single_env, args)
    
    # state, _ = env.reset()
    # print(f"{state[0].shape=} {state[0][50]=}")
    
    ppo = PPO(observation_space = env.observation_space, 
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
        num_of_episodes = RUN_FOR,
        render = args.render,
        record = args.record
        )

    print()
    # print(single_env.get_mean_std(verbose=True))
    return single_env.get_mean_std()    
    
def run_multiple_models(args, models_paths):
    means = []
    stds = []
    
    for path in models_paths:
        mean, std = run_single_model(args, path) # PATHS[PATHS_INDEX]
        means.append(mean)
        stds.append(std)
        print_divider()
        
    return means, stds  
        
# render models from one training which were after each other
def run_following_models(args, first_episode, last_episode, base_dir):
    episodes = list(range(first_episode, last_episode + 1, 10))
    for ep in episodes:
        path = os.path.join(base_dir, f"ep{ep}", f"ep{ep}_weights")
        # print(path)
        run_single_model(args, path) # PATHS[PATHS_INDEX]

def write_to_csv(filename, rows):
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
            num_of_episodes = RUN_FOR,
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
    
    # mean, std = run_single_model(args, PATHS[PATHS_INDEX])
    # print(f"mean={mean:.2f} std = {std:.2f}")
    
    RUN_FOLLOWING_MODELS = True
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
    
    
    
    

    