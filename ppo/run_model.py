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


MODEL_PATH = "BEST/projectBEST"
RUN_FOR = 5

if __name__ == '__main__':
    # show_terminal_colors()
    
    # print_tensorflow_version()
    # print_available_devices()
        
    args = create_parser().parse_args([] if "__file__" not in globals() else None)
    
    def make_env():
        if args.render:
            env = gym.make(args.env, render_mode="human", continuous = True)
        else:
            env = gym.make(args.env, render_mode='rgb_array', continuous = True)
        
        # print(env.action_space)
            
        env = NormalizeObservation(env)
        env = ClippedAction(env, low=0, high=1)
        env = EvaluationEnv(env, render_each=args.render_each, evaluate_for=args.evaluate_for, report_each=args.report_each)
        return env

    single_env = make_env()
    # single_env = GustyLeftWind(single_env)
    single_env = ContinuousLeftWind(single_env)
    # single_env = PrintAction(single_env)
    
    env = gym.vector.SyncVectorEnv([lambda: single_env])
    
    # print_info(single_env, args)
    
    # state, _ = env.reset()
    # print(f"{state[0].shape=} {state[0][50]=}")
    
    # exit()
    ppo = PPO(observation_space = env.observation_space, 
              action_space = env.action_space, 
              entropy_coeff = args.entropy_coeff,
              gamma = args.gamma,
              gae_lambda = args.gae_lambda,
              learning_rate = args.learning_rate,
              value_fun_coeff = args.vf_coeff)

    print_notification_style(f"Loading weights from '{MODEL_PATH}'")
    ppo.load_weights(MODEL_PATH) 
    
    ppo.run(
        env,
        single_env, 
        num_of_episodes = RUN_FOR,
        render = args.render,
        record = args.record
        )
