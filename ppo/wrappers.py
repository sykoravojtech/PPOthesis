from gym import ObservationWrapper, ActionWrapper, Env, Wrapper
from gym.spaces.box import Box
import numpy as np
import sys, os
from random import randint, uniform, choice

from utils import *

np.set_printoptions(precision=3) # number of decimal places numpy prints

PLUSMINUS = "\u00B1"

################ ACTION WRAPPERS #################

class ClippedAction(ActionWrapper):
    """
    Clip the action between two numbers
    Used for easier calculation of wind & easier working with numbers
    """
    
    def __init__(self, env: Env, low: int = 0, high: int = 1):
        super().__init__(env)
        self.old_low: int = env.action_space.low
        self.old_high: int = env.action_space.high
        shape: tuple[int] = env.action_space.shape
        self.action_space: Box = Box(low=low, high=high, shape=shape)

    def action(self, action: tuple[float]) -> tuple[float]:
        l, h = self.action_space.low, self.action_space.high
        L, H = self.old_low, self.old_high

        return (action-l)/(h-l)*(H-L)+L
    
    
class ContinuousLeftWind(ActionWrapper):
    """ 
    Wrapper adding a continuous left wind
    
    Args:
        env: the environment to wrap around
        strength: strength of the wind / range for the percentage change of the action
    """
    
    def __init__(self, env: Env, strength: tuple[float, float] = (0.1, 0.2)):
        super().__init__(env)
        self.strength = (1-strength[0], 1-strength[1]) # 1 - x because left means lowering the number. 10% strength means we get 90% of the action.
        self.index = 0 # action[0] is controlling left right movement, left is 0 right is 1
        self.parallel = len(env.action_space.shape) == 2
        print_notification_style(f"ContinuousLeftWind:\n\t{strength=}")

    def action(self, action: tuple[float]) -> tuple[float]:
        new_action: tuple[float] = np.copy(action)
        curr_strength: float = uniform(*self.strength)
        
        if self.parallel: # vecenv, parallel environments
            new_action[:, self.index] *= curr_strength
        else:
            new_action[self.index] *= curr_strength
        # print(f"ContinuousLeftWind:\n\t{action}\n\t{new_action}")
        return new_action


class GustyLeftWind(ActionWrapper): # nárazový vítr
    """
    Wrapper adding a gusty left wind
    
    Args:
        strength: strength of the wind / range for the percentage change of the action
        nonwind_step_range: how many steps will be WITHOUT wind
        wind_step_range: how many steps will be WITH wind
    """
    def __init__(self, 
                 env: Env, 
                 strength: tuple[float, float] = (0.1, 0.2), 
                 nonwind_step_range: tuple[float, float] = (10, 50), 
                 wind_step_range: tuple[float, float] = (20,50)
                 ):
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

    def action(self, action: tuple[float]) -> tuple[float]:
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
    """ 
    Wrapper adding a continuous right wind
    
    Args:
        env: the environment to wrap around
        strength: strength of the wind / range for the percentage change of the action
    """
    
    def __init__(self, env: Env, strength: tuple[float, float] = (0.1, 0.2)):
        super().__init__(env)
        self.strength = (1+strength[0], 1+strength[1])
        self.index = 0 # action[0] is controlling left right movement, left is 0 right is 1
        self.parallel = len(env.action_space.shape) == 2
        print_notification_style(f"ContinuousRightWind:\n\t{strength=}")

    def action(self, action: tuple[float]) -> tuple[float]:
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
    Wrapper adding a gusty right wind
    
    Args:
        strength: strength of the wind / range for the percentage change of the action
        nonwind_step_range: how many steps will be WITHOUT wind
        wind_step_range: how many steps will be WITH wind
    """
    def __init__(self, 
                 env: Env, 
                 strength: tuple[float, float] = (0.1, 0.2), 
                 nonwind_step_range: tuple[float, float] = (10, 50), 
                 wind_step_range: tuple[float, float] = (20,50)
                 ):
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

    def action(self, action: tuple[float]) -> tuple[float]:
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
    """ 
    Wrapper adding a continuous wind from both sides
    
    Args:
        env: the environment to wrap around
        strength: strength of the wind / range for the percentage change of the action
        block_range: the range from which we choose the random number of steps during which the wind will be applied
        verbose: True if we want to print more information
    """
    
    def __init__(self, 
                 env: Env, 
                 strength: tuple[float, float] = (0.1, 0.2), 
                 block_range: tuple[float, float] = (10,50), 
                 verbose: bool = False
                 ):
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
        print_notification_style(f"ContinuousSidesWind:\n\tstarting = {'RIGHT' if self.direction == RIGHT else 'LEFT'}\n\t{strength=}\n\t{block_range=}")

    def action(self, action: tuple[float]) -> tuple[float]:
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


class GustySidesWind(ActionWrapper):
    """ 
    Wrapper adding a gusty wind from both sides
    
    Args:
        env: the environment to wrap around
        strength: strength of the wind / range for the percentage change of the action
        nonwind_step_range: how many steps will be WITHOUT wind
        wind_step_range: how many steps will be WITH wind
        verbose: True if we want to print more information
    """
    
    def __init__(self, 
                 env: Env, 
                 strength: tuple[float, float] = (0.1, 0.2), 
                 nonwind_step_range: tuple[float, float] = (10, 30), 
                 wind_step_range: tuple[float, float] = (50,100),
                 verbose: bool = False
                 ):
        super().__init__(env)
        self.left_strength = (1-strength[0], 1-strength[1])
        self.right_strength = (1+strength[0], 1+strength[1])
        self.index = 0 # action[0] is controlling left right movement, left is 0 right is 1
        self.wind_step = 0
        self.max_wind = randint(*wind_step_range)
        self.max_nonwind = randint(*nonwind_step_range)
        self.parallel = len(env.action_space.shape) == 2
        self.direction = choice([LEFT, RIGHT])
        self.nonwind_step_range = nonwind_step_range
        self.wind_step_range = wind_step_range
        self.verbose = verbose
        self.wind = False
        print_notification_style(f"GustySidesWind:\n\tstart = {'RIGHT' if self.direction == RIGHT else 'LEFT'}\n\tstrength_range={strength}\n\t{nonwind_step_range=}\n\t{wind_step_range=}\n\t{self.parallel=}")

    def action(self, action: tuple[float]) -> tuple[float]:        
        if self.wind: # apply wind
            self.wind_step += 1
            new_action = np.copy(action)
            
            if self.direction == RIGHT:
                curr_strength = uniform(*self.right_strength)
                
                if self.parallel: # vecenv, parallel environments
                    new_action[:, self.index] *= curr_strength
                    new_action[:, self.index][new_action[:, self.index] > 1.] = 1.
                else:
                    new_action[self.index] *= curr_strength
                    new_action[self.index] = new_action[self.index] if new_action[self.index] <= 1 else 1
                
            else: # LEFT
                curr_strength = uniform(*self.left_strength)
                
                if self.parallel: # vecenv, parallel environments
                    new_action[:, self.index] *= curr_strength
                else:
                    new_action[self.index] *= curr_strength
            
            # wind block ending condition
            if self.wind_step >= self.max_wind:
                self.wind_step = 0
                self.wind = False
                self.max_nonwind = randint(*self.nonwind_step_range)
                
            if self.verbose: print(f"GustySidesWind({self.wind_step}/{self.max_wind}):wind dir={'RIGHT' if self.direction == RIGHT else 'LEFT'} {curr_strength=:.3f}\n\t{action}\n\t{new_action}")
            
            return new_action
        
        else: # nonwind
            self.wind_step += 1
            if self.verbose: print(f"GustySidesWind({self.wind_step}/{self.max_nonwind}):NONwind {action}")
            
            if self.wind_step >= self.max_nonwind:
                self.wind_step = 0
                self.wind = True
                self.direction = choice([LEFT, RIGHT])
                self.max_wind = randint(*self.wind_step_range)
            return action
            
            
class PrintAction(ActionWrapper):
    """
    Wrapper that just prints every action
    """
    
    def __init__(self, env: Env):
        super().__init__(env)
        self.x = 0

    def action(self, action: tuple[float]) -> tuple[float]:
        self.x += 1
        print(f"ActionWrapper({self.x}): {action}")
        return action


def add_wind_wrapper(name: str, env: Env, params: dict = {}) -> Env:
    """ Add a wind wrapper of the correct wind with the specified parameters
    
    Args:
        name: name of the wind
        env: environment to wrap
        params: parameters for the wind wrappers 
    """

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
        elif name == "gustysides":
            env = GustySidesWind(env, **params)
        else:
            print(f"{bcolors.RED} ** Wrapper name not found: '{name}' ** {bcolors.ENDC}")
    else:
        print_notification_style("Running pureEnv")
    return env

############### OBSERVATION WRAPPERS ################

class NormalizeObservation(ObservationWrapper):
    """
    Normalize the observation (clip all values between 0 and 1)
    """
    def __init__(self, env: Env):
        super().__init__(env)
        low: float = np.mean(self.observation_space.low)
        high: float = np.mean(self.observation_space.high)
        # print(f"\tNormalizeObservation Wrapper: {low=} {high=}")
        shape: tuple[int] = env.observation_space.shape
        self.observation_space: Box = Box(low, high, shape=shape, dtype='float32')

    @staticmethod
    def convert_obs(obs):
        return ((obs-127.5)/127.5).astype('float32')

    def observation(self, obs):
        return NormalizeObservation.convert_obs(obs)


class expand_dim_obs(ObservationWrapper):
    """
    Expand the dimensions of the observation by 1
    """
    def __init__(self, env: Env):
        super().__init__(env)
        shape = tuple([1] + list(self.observation_space.shape))
        low = self.observation_space.low
        high = self.observation_space.high
        self.observation_space = Box(low, high, shape=shape, dtype='float32')

    def observation(self, obs: np.array):
        return np.expand_dims(obs, axis=0)
    
    
class PrintObservation(ObservationWrapper):
    """
    Prints every observation
    """
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        print(f"ObservationWrapper: {observation}")
        return observation
    
# class expand_dim_act(ActionWrapper):
#     def __init__(self, env: Env):
#         super().__init__(env)
#         self.action_space.shape = tuple([1] + list(self.action_space.shape))
        
#     def action(self, action):
#         return np.expand_dims(action, axis=0)


############### EVALUATION WRAPPERS ################

class EvaluationEnv(Wrapper):
    """
    Evaluation wrapper for calculating the mean scores
    inspired by Milan Straka's (MFF UK) implementation
    """
    
    def __init__(self, 
                 env: Env, 
                 seed: int = None, 
                 render_each: int= 0, 
                 evaluate_for: int = 100, 
                 report_each: int = 10
                 ):
        super().__init__(env)
        self._render_each = render_each
        self._evaluate_for = evaluate_for
        self._report_each = report_each
        self._report_verbose = os.getenv("VERBOSE") not in [None, "", "0"]

        Env.reset(self.unwrapped, seed=seed)
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
                print(f"The mean {self._evaluate_for}-episode return after evaluation {np.mean(self._episode_returns[-self._evaluate_for:]):.2f} \u00B1 {np.std(self._episode_returns[-self._evaluate_for:]):.2f}", flush=True)
                self.close()
                sys.exit(0)

        return observation, reward, terminated, truncated, info
    
    def get_mean_return(self) -> float:
        return np.mean(self._episode_returns[-self._evaluate_for:])
    
    def get_std(self) -> float:
        return np.std(self._episode_returns[-self._evaluate_for:])
    
    def get_mean_std(self, verbose: bool = False) -> tuple[float, float]:
        if verbose:
            print(f"{bcolors.GREEN}Episode {self.episode}, mean {self._evaluate_for}-episode return "
                    f"{np.mean(self._episode_returns[-self._evaluate_for:]):.2f} "
                    f"{PLUSMINUS} {np.std(self._episode_returns[-self._evaluate_for:]):.2f}"
                    f"{'' if not self._report_verbose else ', returns ' + ' '.join(map('{:g}'.format, self._episode_returns[-self._report_each:]))}{bcolors.ENDC}",
                    file=sys.stderr, flush=True)
        return self.get_mean_return(), self.get_std()