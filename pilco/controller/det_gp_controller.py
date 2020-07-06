"""
Module for the deterministic GP controller detailed in section 5 of:
"Gaussian Processes for Data-Efficient Learning in Robotics and Control"
https://ieeexplore.ieee.org/abstract/document/6654139
"""

import gpflow
from gpflow import Parameter, set_trainable, Module
from gpflow.utilities import positive, to_default_float
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd


class GPController(Module):
    """
    Class to implement a deterministic GP controller, equivalent to a radial basis function network.
    """
    def __init__(self, state_dim, action_dim, num_rbf_functions, u_max):
        """

        :param state_dim: observation dimensionality
        :param action_dim: action dimensionality
        :param num_rbf_functions: int specifying the number of radial basis functions to use
        :param u_max: Parameter for the sin squashing function.
        """

        self.num_inputs = state_dim
        self.num_outputs = action_dim
        self.num_rbf_functions = num_rbf_functions
        self.u_max = u_max
        self.optimizers = []

        # sample the basis function centre values and target noise
        np.random.seed(10)
        rbf_basis_centres = np.random.randn(num_rbf_functions, state_dim)
        target_noise = 0.1 * np.random.randn(num_rbf_functions, action_dim)
        data = [rbf_basis_centres, target_noise]

        # Initialize the DetGP model
        self.models = [DetGP(data)]

    def calculate_factorizations(self):
        """
        from https://github.com/nrontsis/PILCO
        """
        K = self.K(self.X)
        identity_matrix = tf.eye(tf.shape(self.X)[0], batch_shape=[self.num_outputs],
                                 dtype=gpflow.config.default_float())
        L = tf.linalg.cholesky(K + self.noise[:, None, None]*identity_matrix)
        iK = tf.linalg.cholesky_solve(L, identity_matrix, name='chol1_calc_fact')
        Y_ = tf.transpose(self.Y)[:, :, None]
        beta = tf.linalg.cholesky_solve(L, Y_, name="chol2_calc_fact")[:, :, 0]
        return iK, beta

    def predict_given_factorizations(self, m, s, iK, beta):
        """
        from https://github.com/nrontsis/PILCO

        Approximate GP regression at noisy inputs via moment matching
        IN: mean (m) (row vector) and (s) variance of the state
        OUT: mean (M) (row vector), variance (S) of the action
             and inv(s)*input-ouputcovariance
        """

        s = tf.tile(s[None, None, :, :], [self.num_outputs, self.num_outputs, 1, 1])
        inp = tf.tile((self.X - m)[None, :, :], [self.num_outputs, 1, 1])

        # Calculate M and V: mean and inv(s) times input-output covariance
        iL = tf.linalg.diag(1/self.lengthscales)
        iN = inp @ iL
        B = iL @ s[0, ...] @ iL + tf.eye(self.num_inputs, dtype=gpflow.config.default_float())

        # Redefine iN as in^T and t --> t^T
        # B is symmetric so its the same
        t = tf.linalg.matrix_transpose(tf.linalg.solve(B, tf.linalg.matrix_transpose(iN), adjoint=True, name='predict_gf_t_calc'),)
        lb = tf.exp(-tf.reduce_sum(iN * t, -1)/2) * beta
        tiL = t @ iL
        c = self.variance / tf.sqrt(tf.linalg.det(B))

        M = (tf.reduce_sum(lb, -1) * c)[:, None]
        V = tf.matmul(tiL, lb[:, :, None], adjoint_a=True)[..., 0] * c[:, None]

        # Calculate S: Predictive Covariance
        R = s @ tf.linalg.diag(
                1/tf.square(self.lengthscales[None, :, :]) +
                1/tf.square(self.lengthscales[:, None, :])
            ) + tf.eye(self.num_inputs, dtype=gpflow.config.default_float())

        X = inp[None, :, :, :]/tf.square(self.lengthscales[:, None, None, :])
        X2 = -inp[:, None, :, :]/tf.square(self.lengthscales[None, :, None, :])
        Q = tf.linalg.solve(R, s, name='Q_solve')/2
        Xs = tf.reduce_sum(X @ Q * X, -1)
        X2s = tf.reduce_sum(X2 @ Q * X2, -1)
        maha = -2 * tf.matmul(X @ Q, X2, adjoint_b=True) + \
            Xs[:, :, :, None] + X2s[:, :, None, :]

        k = tf.math.log(self.variance)[:, None] - \
            tf.reduce_sum(tf.square(iN), -1)/2
        L = tf.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
        S = (tf.tile(beta[:, None, None, :], [1, self.num_outputs, 1, 1])
                @ L @
                tf.tile(beta[None, :, :, None], [self.num_outputs, 1, 1, 1])
            )[:, :, 0, 0]

        diagL = tf.transpose(tf.linalg.diag_part(tf.transpose(L)))
        S = S - tf.linalg.diag(tf.reduce_sum(tf.multiply(iK, diagL), [1, 2]))
        S = S / tf.sqrt(tf.linalg.det(R))
        S = S + tf.linalg.diag(self.variance)
        S = S - M @ tf.transpose(M)

        return tf.transpose(M), S, tf.transpose(V)

    def compute_action(self, state_mean, state_var):
        """
        Compute the mean and variance of the action distribution given the mean and variance of the state distribution.
        :param state_mean: mean of the state distribution.
        :param state_var: variance of the state distribution.
        :return: action_mean, action_var: characterise the action distribution.
        """

        with tf.name_scope("controller") as scope:
            iK, beta = self.calculate_factorizations()
            action_mean, action_var, V = self.predict_given_factorizations(state_mean, state_var, 0.0 * iK, beta)
            action_var = action_var - tf.linalg.diag(self.variance - 1e-6)
            action_mean, action_var, V2 = squashing_function(action_mean, action_var, self.u_max)
            V = V @ V2

        return action_mean, action_var, V

    def randomize(self):
        """
        Utility function for the purpose of optimizing the controller parameters via random restarts. Resets the
        parameter values upon each restart.
        """

        np.random.seed(20)
        print("Randomizing the RBF controller parameters")
        for model in self.models:
            model.X.assign(np.random.normal(size=model.data[0].shape))
            model.Y.assign(self.u_max / 10 * np.random.normal(size=model.data[1].shape))

            # Draw the kernel lengthscale values of the RBF controller from a Gaussian with mean 1 and std 0.1
            mean = 1
            sigma = 0.1
            model.kernel.lengthscales.assign(mean + sigma*np.random.normal(size=model.kernel.lengthscales.shape))

    @property
    def X(self):
        return self.models[0].data[0]

    @property
    def Y(self):
        return tf.concat([model.data[1] for model in self.models], axis=1)

    @property
    def data(self):
        return self.X, self.Y

    def K(self, X1, X2=None):
        return tf.stack([model.kernel.K(X1, X2) for model in self.models])

    @property
    def lengthscales(self):
        return tf.stack([model.kernel.lengthscales.value() for model in self.models])

    @property
    def variance(self):
        return tf.stack([model.kernel.variance.value() for model in self.models])

    @property
    def noise(self):
        return tf.stack([model.likelihood.variance.value() for model in self.models])


class DetGP(gpflow.Module):
    """
    Deterministic GP
    """
    def __init__(self, data):
        """
        :param data:
        """
        gpflow.Module.__init__(self)
        self.X = Parameter(data[0], name="X", dtype=gpflow.default_float())
        self.Y = Parameter(data[1], name="Y", dtype=gpflow.default_float())
        self.data = [self.X, self.Y]
        self.kernel = gpflow.kernels.SquaredExponential(lengthscales=tf.ones([data[0].shape[1], ],
                                                                             dtype=gpflow.config.default_float()))

        # Set bounds on the optimiser to constrain the lengthscales to be positive and greater than 0.001 cf.
        # https://www.prowler.io/blog/mixture-density-networks-in-gpflow-a-tutorial
        # hyperparameters taken from:
        # https://github.com/nrontsis/PILCO/blob/master/examples/Hyperparameter%20setting%20and%20troubleshooting%20tips.ipynb

        bounded_lengthscales = Parameter(self.kernel.lengthscales, transform=positive(lower=1e-3))
        self.kernel.lengthscales = bounded_lengthscales
        self.kernel.lengthscales.prior = tfd.Gamma(to_default_float(1.1), to_default_float(0.1))
        self.kernel.variance.assign(1.0)
        set_trainable(self.kernel.variance, False)

        self.likelihood = gpflow.likelihoods.Gaussian()
        self.likelihood.variance.assign(1e-4)
        set_trainable(self.likelihood.variance, False)


def squashing_function(mean, var, u_max):
    """
    See section 5 of https://ieeexplore.ieee.org/abstract/document/6654139
    "We squash the preliminary policy π~ through a bounded and differentiable squashing function
    which is the third-order Fourier series expansion of a trapezoidal wave, normalized to the interval [-u_max, u_max]"
    :param mean: Mean of the input action
    :param var: Variance of the input action
    :param u_max: float specifying the bounds of the squashing function.
    :return: mean_out: mean of the squashed input action, var_out: variance of the squashed input action, covar_out:
             covariance of the squashed input action.
    """

    action_dim = tf.shape(mean)[1]
    u_max = u_max * tf.ones((1, action_dim), dtype=gpflow.config.default_float())
    mean_out = u_max * tf.sin(mean) * tf.exp(-tf.linalg.diag_part(var) / 2)
    lq = -(tf.linalg.diag_part(var)[:, None] + tf.linalg.diag_part(var)[None, :]) / 2
    var_out = (tf.exp(lq + var) - tf.exp(lq)) * tf.cos(tf.transpose(mean) - mean) - \
              (tf.exp(lq - var) - tf.exp(lq)) * tf.cos(tf.transpose(mean) + mean)
    var_out = u_max * tf.transpose(u_max) * var_out / 2
    covar_out = u_max * tf.linalg.diag(tf.cos(mean)* tf.exp(-tf.linalg.diag_part(var)/2))

    return mean_out, var_out, tf.reshape(covar_out, shape=[action_dim, action_dim])
