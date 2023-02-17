from gym import ObservationWrapper, spaces, ActionWrapper
import gym
import numpy as np
from collections import deque
import sys, os
# import cv2

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
        self._mean_episode_returns = [] # ADDED
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
                self._mean_episode_returns.append(np.mean(self._episode_returns[-self._evaluate_for:])) # ADDED
                print("Episode {}, mean {}-episode return {:.2f} +-{:.2f}{}".format(
                    self.episode, self._evaluate_for, np.mean(self._episode_returns[-self._evaluate_for:]),
                    np.std(self._episode_returns[-self._evaluate_for:]), "" if not self._report_verbose else
                    ", returns " + " ".join(map("{:g}".format, self._episode_returns[-self._report_each:]))),
                    file=sys.stderr, flush=True)
            if self._evaluating_from is not None and self.episode >= self._evaluating_from + self._evaluate_for:
                print("The mean {}-episode return after evaluation {:.2f} +-{:.2f}".format(
                    self._evaluate_for, np.mean(self._episode_returns[-self._evaluate_for:]),
                    np.std(self._episode_returns[-self._evaluate_for:])), flush=True)
                self.close()
                sys.exit(0)

        return observation, reward, terminated, truncated, info