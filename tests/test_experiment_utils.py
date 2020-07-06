"""
Tests for the experiment_utils.py module
"""

import numpy as np
import pytest

from custom_environments import CustomPendulum
from experiment_utils import gen_initialization_trajectories


@pytest.mark.parametrize("num_episodes, timesteps, expected_state_control_shape, expected_delta_shape",
                         [(4, 40, 4, 3), (1, 50, 4, 3)])
def test_gen_initialization_trajectories(num_episodes, timesteps, expected_state_control_shape, expected_delta_shape):
    """
    Test for the gen_intialization_trajectories function to assert that the returned state trajectories are the
    expected shape in the second dimension (first dimension is stochastic because the number of timesteps depends
    on when the episode ends).

    :param num_episodes: Number of episodes to rollout environment for.
    :param timesteps: Maximum number of timesteps per episode.
    :param expected_shape: Shape of the returned episode state trajectories.
    """
    env = CustomPendulum()  # We always use the same environment
    state_control_trajectory_list, delta_trajectory_list, episode_returns = \
        gen_initialization_trajectories(env, num_episodes, timesteps)
    assert np.shape(state_control_trajectory_list)[1] == expected_state_control_shape
    assert np.shape(delta_trajectory_list)[1] == expected_delta_shape
    assert len(episode_returns) == num_episodes
