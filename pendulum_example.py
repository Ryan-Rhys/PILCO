"""
Script for running the pendulum task (known as Pendulum-v0 in OpenAI gym).
"""

import argparse

from gpflow.utilities import set_trainable
import numpy as np
from matplotlib import pyplot as plt

from custom_environments import CustomPendulum
from experiment_utils import gen_initialization_trajectories, gen_pilco_trajectory
from pilco.controller.det_gp_controller import GPController
from pilco.cost.saturating_cost import SaturatingCost
from pilco.model.pilco_model import PILCO


def main(num_init_episodes, num_init_timesteps, num_rbf_functions, u_max, horizon, pilco_steps, num_episodes, restarts,
         maxiter, seed_range, jitter):
    """

    :param num_init_episodes: Number of episodes to generate intialization data
    :param num_init_timesteps: Maximum number of timesteps per episode to generate intialization data
    :param num_rbf_functions: The number of radial basis functions to use in the determinstic GP controller.
    :param u_max: Bound for squashing function applied to action distribution
    :param horizon: Controller horizon
    :param pilco_steps: Number of timesteps to run PILCO trajectories for
    :param num_episodes: Number of episodes following initialization to run PILCO for
    :param restarts: Number of random restarts for optimizers
    :param maxiter: Max iterations for optimizers
    :param seed_range: Random seed range
    :param jitter: Jitter level for GP models. Default is 1e-4
    """

    # set additional pendulum experiment hyperparameters

    target_state = np.array([1.0, 0.0, 0.0])  # The target state
    weight_matrix = np.diag([2.0, 2.0, 0.3])  # Weight matrix to capture the sensitivity of the cost to different
                                              # dimensions

    # initial state mean and variance taken from:
    # https://github.com/nrontsis/PILCO/blob/master/examples/Hyperparameter%20setting%20and%20troubleshooting%20tips.ipynb
    init_mean = np.reshape([-1.0, 0.0, 0.0], (1, 3))  # initial state mean.
    init_var = np.diag([0.01, 0.05, 0.01])  # initial state variance

    # Main loop

    returns_from_seeds = []

    for seed in range(seed_range):

        # initialize the Custom Pendulum environment and fix the random seed

        env = CustomPendulum()
        env.action_space.np_random.seed(seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # Generate initial dataset of trajectories to train PILCO using random rollouts.

        X, Y, episode_returns = gen_initialization_trajectories(env, num_init_episodes, num_init_timesteps)

        controller = GPController(state_dim=state_dim, action_dim=action_dim, num_rbf_functions=num_rbf_functions,
                                  u_max=u_max)

        cost = SaturatingCost(state_dim, weight_matrix, target_state)

        pilco = PILCO((X, Y), state_dim=state_dim, action_dim=action_dim, controller=controller, horizon=horizon,
                      cost=cost, m_init=init_mean, S_init=init_var)

        # set custom jitter value for GPs

        for model in pilco.mgpr.models:
            model.likelihood.variance.assign(jitter)
            set_trainable(model.likelihood.variance, False)

        # Start learning

        returns_list = []
        for episode in range(num_episodes):

            print(f"Episode {episode}")
            pilco.optimize_models(maxiter=maxiter, restarts=restarts)
            pilco.optimize_policy(maxiter=maxiter, restarts=restarts)
            X_new, Y_new, ep_return = gen_pilco_trajectory(env, pilco, timesteps=pilco_steps, render=False)

            print(f'Return is {ep_return}')

            # Add data to PILCO

            X = np.vstack((X, X_new))
            Y = np.vstack((Y, Y_new))
            pilco.mgpr.set_data((X, Y))

            # Collect returns for plotting
            returns_list.append(ep_return)

        returns_from_seeds.append(returns_list)

    # Compute the mean and standard error of the returns averaged over different environment random seeds.

    returns_list = np.array(returns_from_seeds)
    mean_seed_returns = np.mean(returns_list, axis=0)
    std_seed_returns = np.std(returns_list, axis=0)/np.sqrt(num_episodes)  # standard error of returns

    upper = mean_seed_returns + std_seed_returns
    lower = mean_seed_returns - std_seed_returns

    episode_axis_vals = np.arange(num_episodes)

    plt.plot(episode_axis_vals, mean_seed_returns)
    plt.fill_between(episode_axis_vals, lower, upper, alpha=0.2)
    plt.title('PILCO Performance on the Pendulum Environment')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.savefig('plots/pendulum_returns.png')
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-nie', '--num_init_episodes', type=int, default=4,
                        help='Number of episodes to generate intialization data')
    parser.add_argument('-nit', '--num_init_timesteps', type=int, default=40,
                        help='Maximum number of timesteps per episode to generate intialization data')
    parser.add_argument('-nrf', '--num_rbf_functions', type=int, default=30,
                        help='The number of radial basis functions to use in the determinstic GP controller.')
    parser.add_argument('-u', '--u_max', type=float, default=2.0,
                        help='Bound for squashing function applied to action distribution')
    parser.add_argument('-h', '--horizon', type=int, default=30,
                        help='Controller horizon')
    parser.add_argument('-ps', '--pilco_steps', type=int, default=40,
                        help='Number of timesteps to run PILCO trajectories for')
    parser.add_argument('-ne', '--num_episodes', type=int, default=8,
                        help='Number of episodes following initialization to run PILCO for')
    parser.add_argument('-r', '--restarts', type=int, default=2,
                        help='Number of random restarts for optimizers')
    parser.add_argument('-mx', '--maxiter', type=int, default=50,
                        help='Max number of iterations for optimizers')
    parser.add_argument('-sr', '--seed_range', type=int, default=5,
                        help='number of random seeds to average over')
    parser.add_argument('-j', '--jitter', type=float, default=1e-4,
                        help='Jitter level for GP models')

    args = parser.parse_args()

    main(args.num_init_episodes, args.num_init_timesteps, args.num_rbf_functions, args.u_max, args.horizon,
         args.pilco_steps, args.num_episodes, args.restarts, args.maxiter, args.seed_range, args.jitter)
