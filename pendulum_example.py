"""
Script for running the pendulum task (known as Pendulum-v0 in OpenAI gym).
"""

import gym
import numpy as np
from matplotlib import pyplot as plt

from custom_environments import CustomPendulum
from experiment_utils import gen_initialization_trajectories


if __name__ == '__main__':

    # initialize the Custom Pendulum environment

    env = CustomPendulum()

    # set experiment hyperparameters

    num_init_episodes = 4
    num_init_timesteps = 40

    # Generate initial dataset of trajectories to train PILCO using random rollouts.

    X, Y, episode_returns = gen_initialization_trajectories(env, num_init_episodes, num_init_timesteps)
