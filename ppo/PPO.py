import os
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from collections import defaultdict
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
import tensorflow_probability as tfp
import numpy as np

from typing import Tuple, List, Dict
from gym.spaces.box import Box
from gym.vector.async_vector_env import AsyncVectorEnv
from tensorflow.python.ops.summary_ops_v2 import _ResourceSummaryWriter
from wrappers import EvaluationEnv

from actorcritic import get_ActorCritic_model
from utils import *

# import random

class PPO:
    """
    Deep Reinforcement Learning Agent acting using the Proximal Policy Optimization algorithm
    """
    
    def __init__(self, observation_space: Box, action_space: Box, entropy_coeff: float, gamma: float, gae_lambda: float, learning_rate: float, value_fun_coeff: float):
        self.gamma : float = gamma
        self.gae_lambda : float = gae_lambda
        self.model: Model = None
        self.action_space: Tuple[float] = action_space
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.entropy_coeff : float = entropy_coeff
        self.get_probdist = self.get_beta_probdist
        self.vf_coeff : float = value_fun_coeff
        
        # check if we have the correct dimensions (env has 3 and vectorized has 4)
        if len(observation_space.shape) == 4:
            # print_notification_style("Building CNN ActorCritic Model")
            # passing just the pure one observation shape
            self.model: Model = get_ActorCritic_model(
                observation_space.shape[1:], action_space)
        else:
            raise Exception(
                f'Unsupported observation space shape {observation_space.shape} ... only 4 is supported')


    def save_weights(self, filename: str = 'model'):
        self.model.save_weights(filename, save_format="h5")


    def load_weights(self, filename: str = 'model', verbose: bool = False):
        if verbose: print_notification_style(f"Loading weights from {filename}")
        self.model.load_weights(filename)
        
        
    def get_model(self) -> Model:
        return self.model


    def model_summary(self):
        self.model.summary()
        self.model.get_layer("CNN_model").summary()


    @tf.function
    def choose_action(self, state: tf.Tensor) -> Tuple[tf.Tensor]:
        """ Generates a probability distribution from the Actor output. Then samples it to find the best action

        Args:
            state: the current state representation of the environment (RGB image)
        """
        prob_dist, values = self.get_probdist(state)
        action = prob_dist.sample()
        return action, values, prob_dist.log_prob(action)


    def get_values(self, state: np.array) -> tf.Tensor: # EagerTensor
        """ Returns the values generated by the critic

        Args:
            state: the current state representation of the environment (RGB image)
        """
        return self.get_probdist(state)[1]


    def get_beta_probdist(self, state: tf.Tensor) -> Tuple[tfp.distributions.Independent, tf.Tensor]:
        """
        Returns the probability distribution from which the action will be chosen. 
        Also returns the value from the Critic.

        Args:
            state: the current state representation of the environment (RGB image)
        """
        value, alpha, beta = self.model(state)
        prob_dist = tfp.distributions.Independent(tfp.distributions.Beta(alpha, beta), reinterpreted_batch_ndims=1)
        return prob_dist, tf.squeeze(value, axis=-1)


    @tf.function
    def get_loss_policy_clipped(self, advantage: tf.Tensor, logprobs: tf.Tensor, old_logprobs: tf.Tensor, clip: float) -> tf.Tensor:
        """ Returns the clipped loss
        loss / surrogate objective / conservative policy iteration / TRPO loss 

        Args:
            advantage: Tensor of estimates of the advantage function
            logprobs: logarithm of the probability ratio
            old_logprobs: logarithm of the probability ratio from the previous episode
            clip: clipping range [0,1]
        """
        ratio = tf.math.exp(logprobs - old_logprobs)
        clipped_adv = tf.clip_by_value(ratio, 1 - clip, 1 + clip) * advantage
        loss_policy = -tf.reduce_mean(tf.minimum(ratio*advantage, clipped_adv))
        return loss_policy


    @tf.function
    def get_loss_critic_value(self, pred_value: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
        """ Calculates the value function

        Args:
            pred_value: predicted value
            returns: calculated returns from the episode
        """
        return tf.reduce_mean((pred_value - returns)**2)


    # @tf.function # gives an error
    def gradient(self, states: tf.Tensor, 
                 actions: tf.Tensor, 
                 returns: tf.Tensor, 
                 values: tf.Tensor, 
                 clip: float, 
                 old_logprobs: tf.Tensor
                 ) -> Tuple[zip, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """ Does the gradient descent

        Args:
            states: List of the images/states seen in the current episode
            actions: List of the actions used in the current episode
            returns: List of the returns obtained in the current episode
            values: List of the values generated in the current episode
            clip: clipping range 
            old_logprobs: logarithm of the probability ratio from the previous episode
        """
        eps = 1e-8
        advantages = returns - values
        advantages: tf.Tensor = (advantages 
                      - tf.reduce_mean(advantages) / (tf.keras.backend.std(advantages) + eps))
        
        with tf.GradientTape() as tape:
            prob_dist, predicted_vals = self.get_probdist(states)
            logprobs: tf.Tensor = prob_dist.log_prob(actions)
            loss_policy_clipped: tf.Tensor = self.get_loss_policy_clipped(advantages, logprobs, old_logprobs, clip)
            loss_critic_value: tf.Tensor = self.get_loss_critic_value(predicted_vals, returns)
            entropy: tf.Tensor = prob_dist.entropy()

            # main loss function of PPO
            loss: tf.Tensor = loss_policy_clipped + loss_critic_value*self.vf_coeff - entropy * \
                self.entropy_coeff  
                
        # # Printing values inside Tensor
        # with tf.compat.v1.Session() as sess:
        #     a = tf.print(loss[0], loss_policy_clipped, loss_critic_value*self.vf_coeff, entropy[0] *self.entropy_coeff)
        #     print(f"\tLOSS = {a}")

        approx_kldiv: tf.Tensor = 0.5 * tf.reduce_mean(tf.square(old_logprobs-logprobs))

        vars: Tuple = tape.watched_variables()
        grads: Tuple[tf.Tensor] = tape.gradient(loss, vars)
                
        return zip(grads, vars), loss_critic_value, loss_policy_clipped, entropy, approx_kldiv


    @tf.function
    def learn_on_single_batch(self, clip: tf.Tensor, lr: tf.Tensor, states: tf.Tensor, returns: tf.Tensor, actions: tf.Tensor, values: tf.Tensor, old_logprobs: tf.Tensor) -> Tuple[tf.Tensor]:
        """ Does gradient descent learning on single batch of experiences

        Args:
            clip: clipping range
            lr: learning rate
            states: List of the images/states seen in the current episode
            returns: List of the returns obtained in the current episode
            actions: List of the actions used in the current episode
            values: List of the values generated in the current episode
            old_logprobs: logarithm of the probability ratio from the previous episode
        """
        states = tf.keras.backend.cast_to_floatx(states)
        grads_and_vars, loss_v, loss_policy_clipped, entropy, approx_kldiv = self.gradient(
            states, actions, returns, values, clip, old_logprobs)
        self.optimizer.learning_rate = lr
        self.optimizer.apply_gradients(grads_and_vars)

        return loss_policy_clipped, loss_v, entropy, approx_kldiv


    def get_returns(self, 
                    rewards: List[np.array], 
                    values: List[np.array], 
                    dones: List[np.array], 
                    last_values: np.array, 
                    last_dones: tf.Tensor, 
                    dtype: str
                    ) -> np.array:
        
        timesteps: int = len(rewards)
        num_of_envs: int = rewards[0].shape[0]
        advantages: np.array = np.zeros((timesteps, num_of_envs))
        curr_adv: float = 0.
                
        for step in reversed(range(timesteps)):
            if step != timesteps-1:
                next_values = values[step+1]
                next_non_terminal = 1.0 - dones[step+1]
            else:
                next_values = last_values
                next_non_terminal = 1.0 - last_dones

            delta = rewards[step] + self.gamma * \
                next_values * next_non_terminal - values[step]
                
            curr_adv = delta + self.gamma * self.gae_lambda * next_non_terminal * curr_adv
            advantages[step] = curr_adv
            
        returns: np.array = (advantages + values).astype(dtype)
        return returns


    def learn(self, 
              state: np.array, 
              returns: np.array, 
              actions: np.array, 
              values: np.array, 
              old_logprobs: np.array, 
              clip: float, 
              lr: float, 
              batch_size: int, 
              epochs: int
              ) -> defaultdict:
        """ Learn from the obtained batch of experiences

        Args:
            state: List of the images/states seen in the current episode
            returns: List of the returns obtained in the current episode
            actions: List of the actions used in the current episode
            values: List of the values generated in the current episode
            old_logprobs: logarithm of the probability ratio from the previous episode
            clip: clipping range
            lr: learning rate
            batch_size: mini batch size hyperparameter 
            epochs: number of epochs to execute
        """
        
        losses = defaultdict(list)
        num_batches: int = state.shape[0]
        indxs = np.arange(num_batches)
        for _ in range(epochs):
            np.random.shuffle(indxs)
            for start in range(0, num_batches, batch_size):
                end = start + batch_size
                batch_i = indxs[start:end]
                slices = (tf.constant(sravo[batch_i]) for sravo in (
                    state, returns, actions, values, old_logprobs))
                loss = self.learn_on_single_batch(
                    clip, lr, *slices)
                for k, v in zip(['policy_loss', 'value_loss', 'entropy', 'kl'], loss):
                    losses[k].append(v)

        return losses


    def train(
        self, 
        env: AsyncVectorEnv, 
        args: argparse.Namespace, 
        num_of_episodes: int, 
        steps_per_ep: int, 
        epochs_per_ep: int,
        batch_size: int, 
        clip_range: float, 
        lr: float, 
        save_interval: int,
        models_dir: str, 
        starting_episode: int, 
        print_freq: int, 
        logger: _ResourceSummaryWriter
    ):
        """ 
        The main training function which iterates over episodes and steps in each episode 
        and executes everything neccessary for the training of our Reinforcement Deep Learning Agent

        Args:
            env: vectorized environment
            args: arguments, mainly for hyperparameters
            num_of_episodes: number of episodes to train on 
            steps_per_ep: number of steps in each episode
            epochs_per_ep: number of epochs to learn on each episode
            batch_size: mini-batch size
            clip_range: clipping range for the loss function 
            lr: learning rate of the optimizer
            save_interval: save a backup of the model each Nth episode
            models_dir: directory where we want to save the model weights
            starting_episode: the episode number to start on (mainly for clearer logging)
            print_freq: print information about the training each Nth episode
            logger: for us this is the TensorBoard logger
        """
        print_chapter_style(f"TRAINING for {num_of_episodes-starting_episode+1} episodes")
        dtype = tf.keras.backend.floatx()
        dones: np.array = np.zeros((env.num_envs,), dtype=dtype)
        overall_scores: np.array = np.zeros_like(dones)
        
        state, _ = env.reset()
        
        score_history: List[float] = []
        avg_score_history: List[float] = []

        step = 0
        for ep in range(starting_episode, num_of_episodes + 1):
            
            # initialize batches
            batch_states, batch_rewards, batch_actions, batch_values, batch_dones, batch_logprobs = [], [], [], [], [], []
            for _ in range(steps_per_ep):
                step += 1
                
                # choose action
                actions, values, logprobs = self.choose_action(state)
                actions: np.array = actions.numpy()

                # add newly obtained data to batches
                batch_states.append(state)
                batch_actions.append(actions)
                batch_values.append(values.numpy())
                batch_dones.append(dones)
                batch_logprobs.append(logprobs.numpy())

                # use the chosen action
                state, rewards, terminated, truncated, _ = env.step(actions)
                
                dones = terminated | truncated
                # if np.any(dones):
                #     print(f"{dones = }")
                dones = tf.cast(dones, tf.float32)

                # sort out rewards
                batch_rewards.append(rewards)
                overall_scores += rewards
                for i in range(rewards.shape[0]):  # for each env in vector env
                    if dones[i]:
                        score_history.append(overall_scores[i])
                        overall_scores[i] = 0

            final_values = self.get_values(state).numpy()

            # calculates the returns from this episode using advantage and values
            returns = self.get_returns(
                batch_rewards, batch_values, batch_dones, final_values, dones, dtype=dtype)

            # reshaping
            batch_states = np.concatenate(batch_states, axis=0)
            returns = np.concatenate(returns, axis=0)
            batch_actions = np.concatenate(batch_actions, axis=0)
            batch_values = np.concatenate(batch_values, axis=0)
            batch_logprobs = np.concatenate(batch_logprobs, axis=0)

            # choice between constant and decaying learning rate
            if args.constant_lr:
                lr_now = lr(1.0)
            else:
                lr_now = lr(1.0 - ep/num_of_episodes)

            # learn from the obtained batch of experiences
            self.learn(batch_states, returns, batch_actions, batch_values, batch_logprobs,
                       clip_range, lr_now, batch_size, epochs=epochs_per_ep)
            
            # calculate the mean score for a smooth graph for comparing
            avg_score: float = np.mean(score_history[-300:] if len(score_history) > 300 else score_history)
            avg_score_history.append(avg_score)

            # print information about the episode
            if ep % print_freq == 0:
                print(f'==> {bcolors.BLUE}episode: {ep}/{num_of_episodes} step={step}, avg score: {avg_score:.3f}{bcolors.ENDC}')

            # save weights to be able to load the model later
            if ep % save_interval == 0:
                chkpt_dir = os.path.join(models_dir, f"ep{ep}")
                os.makedirs(chkpt_dir)
                weights_dir = os.path.join(chkpt_dir, f"ep{ep}_weights")
                print_notification_style(f"Saving model ep={ep}")
                self.save_weights(weights_dir)
                save_pltgraph(avg_score_history, chkpt_dir, ep, starting_episode)

            # tensorboard for clear overview of average score over all episodes
            if args.tensorboard:
                with logger.as_default():
                    tf.summary.scalar('average score', avg_score, step=step)
                    tf.summary.scalar('learning rate', lr_now, step=step)
                    tf.summary.scalar('episode', ep, step=step)

        # end of all episodes
        self.save_weights(os.path.join(models_dir, "FINAL"))


    def run(self, 
            env: AsyncVectorEnv, 
            single_env: EvaluationEnv, 
            render: bool, 
            record: bool, 
            num_of_episodes: int = 10
            ):
        """ The main function for testing our pretrained models. 

        Args:
            env: vectorized environment
            single_env: non-vectorized environment
            render: True if we want to render the states as an image instead of just a list of numbers
            record: True if we want to show the pygame window before starting the evaluation, to be able to set the window in out recording software.
            num_of_episodes: how many episodes to evaluate the model on
        """                       
        if record:
            env.reset()
            single_env.render()    
            input("==> Press ENTER to begin running")
        
        print_chapter_style(f"Running for {num_of_episodes} episodes")
        
        for episode in range(1, num_of_episodes + 1):
            done = False
            state, _ = env.reset()
            step = 0
            score = 0
            
            # run until the episode is finished
            while not done:
                if render:
                    single_env.render()
                action, _, _ = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action.numpy())
                done = terminated or truncated
                score += reward[0]
                state = next_state
                step += 1
            
            # print is executed as part of the EvaluationEnv wrapper
            # print(f"episode {ep}: {score = :.0f}")
