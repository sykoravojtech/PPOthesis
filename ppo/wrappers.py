from gym import ObservationWrapper, ActionWrapper, spaces
import gym
import numpy as np
import sys, os
from utils import *
from random import randint, uniform, choice

np.set_printoptions(precision=3) # number of decimal places numpy prints

PLUSMINUS = "\u00B1"

################ ACTION WRAPPERS #################

# easier calculation of wind
# easier working with numbers
class ClippedAction(ActionWrapper):
    def __init__(self, env:gym.Env, low=0, high=1):
        super().__init__(env)
        self.old_low = env.action_space.low
        self.old_high = env.action_space.high
        shape = env.action_space.shape
        self.action_space = spaces.Box(low=low, high=high, shape=shape)

    def action(self, action):
        l, h = self.action_space.low, self.action_space.high
        L, H = self.old_low, self.old_high

        return (action-l)/(h-l)*(H-L)+L
    
class ContinuousLeftWind(ActionWrapper):
    def __init__(self, env:gym.Env, strength=(0.1, 0.2)):
        super().__init__(env)
        self.strength = (1-strength[0], 1-strength[1]) # 1 - x because left means lowering the number. 10% strength means we get 90% of the action.
        self.index = 0 # action[0] is controlling left right movement, left is 0 right is 1
        self.parallel = len(env.action_space.shape) == 2
        print_notification_style(f"ContinuousLeftWind:\n\t{strength=}")

    def action(self, action):
        new_action = np.copy(action)
        curr_strength = uniform(*self.strength)
        
        if self.parallel: # vecenv, parallel environments
            new_action[:, self.index] *= curr_strength
        else:
            new_action[self.index] *= curr_strength
        # print(f"ContinuousLeftWind:\n\t{action}\n\t{new_action}")
        return new_action

class GustyLeftWind(ActionWrapper): # nárazový vítr
    """
    Args:
        strength (float): percentage change of action
        nonwind_step_range (float): how many steps will be WITHOUT wind
        wind_step_range (float): how many steps will be WITH wind
    """
    def __init__(self, env:gym.Env, strength=(0.1, 0.2), nonwind_step_range = (10, 50), wind_step_range = (20,50)):
        super().__init__(env)
        self.strength = (1-strength[0], 1-strength[1]) # 1 - x because left means lowering the number. 10% strength means we get 90% of the action. This is strength range
        self.index = 0 # action[0] is controlling left right movement, left is 0 right is 1
        self.nonwind_step_range = nonwind_step_range
        self.wind_step_range = wind_step_range
        self.wind = False
        self.curr_step = 0
        self.wind_step = 0
        self.max_wind = randint(*wind_step_range)
        self.max_nonwind = randint(*nonwind_step_range)
        self.parallel = len(env.action_space.shape) == 2
        print_notification_style(f"GustyLeftWind:\n\tstrength_range={strength}\n\t{nonwind_step_range=}\n\t{wind_step_range=}\n\t{self.parallel=}")

    def action(self, action):
        self.curr_step += 1
        if self.wind: # apply wind
            self.wind_step += 1
            
            new_action = np.copy(action)
            curr_strength = uniform(*self.strength)
            
            if self.parallel: # vecenv, parallel environments
                new_action[:, self.index] *= curr_strength
            else:
                new_action[self.index] *= curr_strength
            
            # print(f"GustyLeftWind({self.curr_step},{self.wind_step}):wind strength={curr_strength:.2f}\n\t{action}\n\t{new_action}")
            
            if self.wind_step >= self.max_wind:
                self.wind = False
                self.wind_step = 0
            
            return new_action
        else: # nonwind
            self.wind_step += 1
            # print(f"GustyLeftWind({self.curr_step},{self.wind_step}):NONwind {action}")
            
            if self.wind_step >= self.max_nonwind:
                self.wind = True
                self.wind_step = 0
            return action

class ContinuousRightWind(ActionWrapper):
    def __init__(self, env:gym.Env, strength=(0.1, 0.2)):
        super().__init__(env)
        self.strength = (1+strength[0], 1+strength[1])
        self.index = 0 # action[0] is controlling left right movement, left is 0 right is 1
        self.parallel = len(env.action_space.shape) == 2
        print_notification_style(f"ContinuousRightWind:\n\t{strength=}")

    def action(self, action):
        new_action = np.copy(action)
        curr_strength = uniform(*self.strength)
        
        if self.parallel: # vecenv, parallel environments
            new_action[:, self.index] *= curr_strength
            new_action[:, self.index][new_action[:, self.index] > 1.] = 1.
        else:
            new_action[self.index] *= curr_strength
            new_action[self.index] = new_action[self.index] if new_action[self.index] <= 1 else 1
            
        # print(f"ContinuousRightWind:\n\t{action}\n\t{new_action}")
        return new_action

class GustyRightWind(ActionWrapper): # nárazový vítr
    """
    Args:
        strength (float): percentage change of action
        nonwind_step_range (float): how many steps will be WITHOUT wind
        wind_step_range (float): how many steps will be WITH wind
    """
    def __init__(self, env:gym.Env, strength=(0.1, 0.2), nonwind_step_range = (10, 50), wind_step_range = (20,50)):
        super().__init__(env)
        self.strength = (1+strength[0], 1+strength[1])
        self.index = 0 # action[0] is controlling left right movement, (left,right)=(0,1) range
        self.nonwind_step_range = nonwind_step_range
        self.wind_step_range = wind_step_range
        self.wind = False
        self.curr_step = 0
        self.wind_step = 0
        self.max_wind = randint(*wind_step_range)
        self.max_nonwind = randint(*nonwind_step_range)
        self.parallel = len(env.action_space.shape) == 2
        print_notification_style(f"GustyRightWind:\n\tstrength_range={strength}\n\t{nonwind_step_range=}\n\t{wind_step_range=}\n\t{self.parallel=}")

    def action(self, action):
        self.curr_step += 1
        if self.wind: # apply wind
            self.wind_step += 1
            
            new_action = np.copy(action)
            curr_strength = uniform(*self.strength)
            
            if self.parallel: # vecenv, parallel environments
                new_action[:, self.index] *= curr_strength
                new_action[:, self.index][new_action[:, self.index] > 1.] = 1.
            else:
                new_action[self.index] *= curr_strength
                new_action[self.index] = new_action[self.index] if new_action[self.index] <= 1 else 1
            
            # print(f"GustyRightWind({self.curr_step},{self.wind_step}):wind strength={curr_strength:.2f}\n\t{action}\n\t{new_action}")
            
            if self.wind_step >= self.max_wind:
                self.wind = False
                self.wind_step = 0
            
            return new_action
        else: # nonwind
            self.wind_step += 1
            # print(f"GustyRightWind({self.curr_step},{self.wind_step}):NONwind {action}")
            
            if self.wind_step >= self.max_nonwind:
                self.wind = True
                self.wind_step = 0
            return action

RIGHT = +1
LEFT  = -1
class ContinuousSidesWind(ActionWrapper):
    
    def __init__(self, env:gym.Env, strength=(0.1, 0.2), block_range=(10,50), verbose=False):
        super().__init__(env)
        self.left_strength = (1-strength[0], 1-strength[1])
        self.right_strength = (1+strength[0], 1+strength[1])
        self.index = 0 # action[0] is controlling left right movement, left is 0 right is 1
        self.wind_step = 0
        self.max_wind = randint(*block_range)
        self.parallel = len(env.action_space.shape) == 2
        self.direction = choice([LEFT, RIGHT])
        self.block_range = block_range
        self.verbose = verbose
        print_notification_style(f"ContinuousSidesWind:\n\t{strength=} starting={self.direction} {block_range=}")

    def action(self, action):
        new_action = np.copy(action)
        
        if self.direction == RIGHT:
            self.wind_step += 1
            
            curr_strength = uniform(*self.right_strength)
            
            if self.parallel: # vecenv, parallel environments
                new_action[:, self.index] *= curr_strength
                new_action[:, self.index][new_action[:, self.index] > 1.] = 1.
            else:
                new_action[self.index] *= curr_strength
                new_action[self.index] = new_action[self.index] if new_action[self.index] <= 1 else 1
            
            if self.wind_step >= self.max_wind:
                self.direction = LEFT
                self.wind_step = 0
                self.max_wind = randint(*self.block_range)
        else: # LEFT
            self.wind_step += 1
            
            curr_strength = uniform(*self.left_strength)
            
            if self.parallel: # vecenv, parallel environments
                new_action[:, self.index] *= curr_strength
            else:
                new_action[self.index] *= curr_strength
            
            if self.wind_step >= self.max_wind:
                self.direction = RIGHT
                self.wind_step = 0
                self.max_wind = randint(*self.block_range)
            
        if self.verbose: print(f"ContinuousSidesWind({self.wind_step}/{self.max_wind}):dir={'RIGHT' if self.direction == RIGHT else 'LEFT'} {curr_strength=:.3f}\n\t{action}\n\t{new_action}")
        return new_action

class PrintAction(ActionWrapper):
    def __init__(self, env:gym.Env):
        super().__init__(env)
        self.x = 0

    def action(self, action):
        self.x += 1
        print(f"ActionWrapper({self.x}): {action}")
        return action

def add_wind_wrapper(name, env, params = {}):
    if name != None:
        name = name.lower()
        if name == "left":
            env = ContinuousLeftWind(env, **params)
        elif name == "gustyleft":
            env = GustyLeftWind(env, **params)
        elif name == "right":
            env = ContinuousRightWind(env, **params)
        elif name == "gustyright":
            env = GustyRightWind(env, **params)
        elif name == "sides":
            env = ContinuousSidesWind(env, **params)
        # elif name == "gustysides":
            # env = GustySidesWind(env, **params)
        else:
            print(f"{bcolors.RED} ** Wrapper name not found: '{name}' ** {bcolors.ENDC}")
    return env

############### OBSERVATION WRAPPERS ################

class NormalizeObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = np.mean(self.observation_space.low)
        high = np.mean(self.observation_space.high)
        # print(f"\tNormalizeObservation Wrapper: {low=} {high=}")
        shape = env.observation_space.shape
        self.observation_space = spaces.Box(low, high, shape=shape, dtype='float32')

    @staticmethod
    def convert_obs(obs):
        return ((obs-127.5)/127.5).astype('float32')

    def observation(self, obs):
        return NormalizeObservation.convert_obs(obs)

class expand_dim_obs(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = tuple([1] + list(self.observation_space.shape))
        low = self.observation_space.low
        high = self.observation_space.high
        self.observation_space = spaces.Box(low, high, shape=shape, dtype='float32')

    def observation(self, obs):
        return np.expand_dims(obs, axis=0)
    
class PrintObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        print(f"ObservationWrapper: {observation}")
        return observation
    
# class expand_dim_act(ActionWrapper):
#     def __init__(self, env:gym.Env):
#         super().__init__(env)
#         self.action_space.shape = tuple([1] + list(self.action_space.shape))
        
#     def action(self, action):
#         return np.expand_dims(action, axis=0)


# STRAKA WRAPPERS

class EvaluationEnv(gym.Wrapper):
    def __init__(self, env, seed=None, render_each=0, evaluate_for=100, report_each=10):
        super().__init__(env)
        self._render_each = render_each
        self._evaluate_for = evaluate_for
        self._report_each = report_each
        self._report_verbose = os.getenv("VERBOSE") not in [None, "", "0"]

        gym.Env.reset(self.unwrapped, seed=seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

        self._episode_running = False
        self._episode_returns = []
        self._mean_episode_returns = []
        self._evaluating_from = None
        self._original_render_mode = env.render_mode

    @property
    def episode(self):
        return len(self._episode_returns)

    def reset(self, *, start_evaluation=False, logging=True, seed=None, options=None):
        if seed is not None:
            raise RuntimeError("The EvaluationEnv cannot be reseeded")
        if self._evaluating_from is not None and self._episode_running:
            raise RuntimeError("Cannot reset a running episode after `start_evaluation=True`")
        if start_evaluation and self._evaluating_from is None:
            self._evaluating_from = self.episode

        if logging and self._render_each and (self.episode + 1) % self._render_each == 0:
            self.unwrapped.render_mode = "human"
        elif self._render_each:
            self.unwrapped.render_mode = self._original_render_mode
        self._episode_running = True
        self._episode_return = 0 if logging or self._evaluating_from is not None else None
        return super().reset(options=options)

    def step(self, action):
        if not self._episode_running:
            raise RuntimeError("Cannot run `step` on environments without an active episode, run `reset` first")

        observation, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated

        self._episode_running = not done
        if self._episode_return is not None:
            self._episode_return += reward
        if self._episode_return is not None and done:
            self._episode_returns.append(self._episode_return)

            if self._report_each and self.episode % self._report_each == 0:
                self._mean_episode_returns.append(np.mean(self._episode_returns[-self._evaluate_for:]))
                print(f"{bcolors.BLUE}Episode {self.episode}, mean {self._evaluate_for}-episode return "
                    f"{np.mean(self._episode_returns[-self._evaluate_for:]):.2f} "
                    f"{PLUSMINUS} {np.std(self._episode_returns[-self._evaluate_for:]):.2f}"
                    f"{'' if not self._report_verbose else ', returns ' + ' '.join(map('{:g}'.format, self._episode_returns[-self._report_each:]))}{bcolors.ENDC}",
                    file=sys.stderr, flush=True)
            if self._evaluating_from is not None and self.episode >= self._evaluating_from + self._evaluate_for:
                print("The mean {}-episode return after evaluation {:.2f} \u00B1 {:.2f}".format(
                    self._evaluate_for, np.mean(self._episode_returns[-self._evaluate_for:]),
                    np.std(self._episode_returns[-self._evaluate_for:])), flush=True)
                self.close()
                sys.exit(0)

        return observation, reward, terminated, truncated, info