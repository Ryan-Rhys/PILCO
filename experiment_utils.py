"""
Module containing utility functions for running PILCO experiments.
"""

import numpy as np


def gen_initialization_trajectories(env, num_episodes, timesteps):
    """
    Generate state-control trajectories to be used as data to initialize PILCO with.

    :param env: OpenAI gym environment object.
    :param num_episodes: The number of episodes to collect trajectories for.
    :param timesteps: The maximum number of episodes per timestep.
    :return: state_control_trajectory_list: a numpy array of states and controls used as features for PILCO.
             delta_trajectory_list: a numpy array of deltas (new_state - old_state) values used as targets for PILCO
             episode_returns: a list of individual episode returns.
    """

    # Store the state-control trajectories for all episodes
    state_control_trajectory_list = []

    # Store the delta values for all episodes (new_state - old_state)
    delta_trajectory_list = []

    # store the returns for all episodes
    episode_returns = []

    for episode in range(num_episodes):
        state_control_trajectory = []
        delta_trajectory = []
        ep_return = 0
        previous_state = env.reset()
        for timestep in range(timesteps):

            # Sample a random action
            action = env.action_space.sample()

            new_state, reward, done, _ = env.step(action)
            ep_return += reward
            state_control_trajectory.append(np.hstack((new_state, action)))
            delta = new_state - previous_state
            delta_trajectory.append(delta)
            previous_state = new_state
            if done:
                break

        state_control_trajectory_list.append(np.array(state_control_trajectory))
        delta_trajectory_list.append(np.array(delta_trajectory))
        episode_returns.append(ep_return)

    # convert the state-control and delta trajectories into numpy arrays of appropriate shape.

    state_control_trajectory_list = np.array(state_control_trajectory_list).\
        reshape(-1, np.shape(np.array(state_control_trajectory_list))[-1])
    delta_trajectory_list = np.array(delta_trajectory_list).reshape(-1, np.shape(np.array(delta_trajectory_list))[-1])

    return state_control_trajectory_list, delta_trajectory_list, episode_returns
