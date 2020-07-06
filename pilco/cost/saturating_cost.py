"""
Module implementing the saturating cost given in "Gaussian Processes for Data-Efficient Learning in Robotics and Control"
"""

import gpflow
from gpflow import Parameter, Module
import numpy as np
import tensorflow as tf


class SaturatingCost(Module):
    """
    Saturating cost function together with additional weight matrix used by:
    https://github.com/nrontsis/PILCO/blob/master/examples/Hyperparameter%20setting%20and%20troubleshooting%20tips.ipynb
    """
    def __init__(self, state_dim, weight_matrix, target_state):
        """
        :param state_dim: dimensionality of observation.
        :param weight_matrix: numpy array giving the pre-defined weight matrix which sets the sensitivity of the reward
                              to the different dimensions.
        :param target_state: numpy array giving the target state.
        """
        self.state_dim = state_dim
        self.weight_matrix = Parameter(np.reshape(weight_matrix, (state_dim, state_dim)), trainable=False)
        self.target_state = Parameter(np.reshape(target_state, (1, state_dim)), trainable=False)

    def compute_cost(self, m, s):
        """
        Compute the cost of a given state distribution specified by its mean and variance.

        :param m: The mean of the state distribution
        :param s: The variance of the state distribution
        :return: cost: float specifying the cost
        """

        SW = s @ self.weight_matrix

        iSpW = tf.transpose(tf.linalg.solve((tf.eye(self.state_dim, dtype=gpflow.config.default_float()) + SW),
                                             tf.transpose(self.weight_matrix), adjoint=True))

        cost = tf.exp(-(m - self.target_state) @  iSpW @ tf.transpose(m - self.target_state)/2) / \
                 tf.sqrt(tf.linalg.det(tf.eye(self.state_dim, dtype=gpflow.config.default_float()) + SW))

        cost.set_shape([1, 1])
        return cost
