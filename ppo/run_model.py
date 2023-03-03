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

import car_racing_environment # f"CarRacingFS{skip_frames}-v2"

PATHS_INDEX = 0
PATHS = [
    'BEST/pureEnv/projectBEST',
    'BEST/left/ep810_0.1,0.2(358)',
    'BEST/gustyLeft/ep960_0.1,0.2(500)',
    'BEST/right/ep1220_0.1,0.2(435)',
    'BEST/gustyRight/ep780_0.1,0.2(275)',
    'BEST/sides/ep820_0.3,0.4',
    'BEST/sides/ep400_0.4,0.5',
    'BEST/sides/ep400_0.1,0.2',
    'BEST/sides/ep420_0.2,0.3'
    ]
MODEL_PATH = "BEST/pureEnv/projectBEST"
RUN_FOR = 2

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

    # print(single_env.get_mean_std(verbose=True))
    print()
    
    
def run_multiple_models(args, models_paths):
    for path in models_paths:
        run_single_model(args, path) # PATHS[PATHS_INDEX]
        
def run_following_models(args, epizodes, base_dir):
    for ep in epizodes:
        path = os.path.join(base_dir, f"ep{ep}", f"ep{ep}_weights")
        # print(path)
        run_single_model(args, path) # PATHS[PATHS_INDEX]

if __name__ == '__main__':
    # show_terminal_colors()
    
    # print_tensorflow_version()
    # print_available_devices()
        
    args = create_parser().parse_args([] if "__file__" not in globals() else None)
    
    # run_single_model(args, PATHS[PATHS_INDEX])
    
    # run_multiple_models(args, PATHS)
    
    eps = list(range(80,121,10))
    run_following_models(args, eps, "archive/left/3to4")
    
    # run_single_model(args, "archive/left/3to4/ep80/ep80_weights")
    
    
    

    