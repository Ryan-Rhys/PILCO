"""
Module containing the definition of PILCO. The code is heavily influenced by the following implementation:
https://github.com/nrontsis/PILCO

"""

import gpflow
from gpflow.utilities import set_trainable, to_default_float
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_probability import distributions as tfd


class PILCO(gpflow.models.BayesianModel):
    """
    Main class for PILCO.
    """
    def __init__(self, data, state_dim, action_dim, horizon, controller, cost, m_init=None, S_init=None, name=None):
        """

        :param data: (X, Y) trajectories. X = concatenated observation/action vectors. Y is a vector of state deltas.
        :param state_dim: dimensionality of observation space
        :param action_dim: dimensionality of action space
        :param horizon: Controller horizon
        :param controller: Controller object
        :param cost: Cost object
        :param m_init: initial state mean
        :param S_init: initial state variance
        :param name: passed into BayesianModel superclass
        """
        super(PILCO, self).__init__(name)
        self.mgpr = MGPR(data)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.controller = controller
        self.cost = cost
        self.m_init = m_init
        self.S_init = S_init
        self.optimizer = None

    def training_loss(self):
        """
        Loss function for the controller parameters.
        """
        reward = self.predict(self.m_init, self.S_init, self.horizon)[2]
        return -reward

    def optimize_models(self, maxiter=200, restarts=1):
        """
        Optimize the GP models.
        :param maxiter: max number of iterations (unused)
        :param restarts: Number of random restarts
        :return:
        """
        self.mgpr.optimize(restarts=restarts)
        # Print the resulting model parameters. Useful for debugging cf.
        # https://github.com/nrontsis/PILCO/blob/master/examples/Hyperparameter%20setting%20and%20troubleshooting%20tips.ipynb
        lengthscales = {}
        variances = {}
        noises = {}
        i = 0
        for model in self.mgpr.models:
            lengthscales['GP' + str(i)] = model.kernel.lengthscales.numpy()
            variances['GP' + str(i)] = np.array([model.kernel.variance.numpy()])
            noises['GP' + str(i)] = np.array([model.likelihood.variance.numpy()])
            i += 1
        print('-----Learned models------')
        pd.set_option('precision', 3)
        print('---Lengthscales---')
        print(pd.DataFrame(data=lengthscales))
        print('---Variances---')
        print(pd.DataFrame(data=variances))
        print('---Noises---')
        print(pd.DataFrame(data=noises))

    def optimize_policy(self, maxiter=50, restarts=1):
        """
        Optimize policy parameters. adapted from from https://github.com/nrontsis/PILCO
        :param maxiter: max optimizer iterations
        :param restarts: number of random restarts
        """
        mgpr_trainable_params = self.mgpr.trainable_parameters
        for param in mgpr_trainable_params:
            set_trainable(param, False)

        if not self.optimizer:
            self.optimizer = gpflow.optimizers.Scipy()
            self.optimizer.minimize(self.training_loss, self.trainable_variables, options=dict(maxiter=maxiter))
        else:
            self.optimizer.minimize(self.training_loss, self.trainable_variables, options=dict(maxiter=maxiter))
        restarts -= 1

        best_parameter_values = [param.numpy() for param in self.trainable_parameters]
        best_reward = self.compute_reward()
        for restart in range(restarts):
            self.controller.randomize()
            self.optimizer.minimize(self.training_loss, self.trainable_variables, options=dict(maxiter=maxiter))
            reward = self.compute_reward()
            if reward > best_reward:
                best_parameter_values = [param.numpy() for param in self.trainable_parameters]
                best_reward = reward

        for i, param in enumerate(self.trainable_parameters):
            param.assign(best_parameter_values[i])
        for param in mgpr_trainable_params:
            set_trainable(param, True)

    def compute_action(self, x_m):
        return self.controller.compute_action(x_m, tf.zeros([self.state_dim, self.state_dim],
                                                            gpflow.config.default_float()))[0]

    def predict(self, m_x, s_x, n):
        """
        from https://github.com/nrontsis/PILCO
        """
        loop_vars = [
            tf.constant(0, tf.int32),
            m_x,
            s_x,
            tf.constant([[0]], gpflow.config.default_float())
        ]

        _, m_x, s_x, cost = tf.while_loop(
            # Termination condition
            lambda j, m_x, s_x, cost: j < n,
            # Body function
            lambda j, m_x, s_x, cost: (
                j + 1,
                *self.propagate(m_x, s_x),
                tf.add(cost, self.cost.compute_cost(m_x, s_x))
            ), loop_vars
        )
        return m_x, s_x, cost

    def propagate(self, m_x, s_x):
        """
        from https://github.com/nrontsis/PILCO
        """
        m_u, s_u, c_xu = self.controller.compute_action(m_x, s_x)

        m = tf.concat([m_x, m_u], axis=1)
        s1 = tf.concat([s_x, s_x@c_xu], axis=1)
        s2 = tf.concat([tf.transpose(s_x@c_xu), s_u], axis=1)
        s = tf.concat([s1, s2], axis=0)

        M_dx, S_dx, C_dx = self.mgpr.predict_on_noisy_inputs(m, s)
        M_x = M_dx + m_x
        S_x = S_dx + s_x + s1@C_dx + tf.matmul(C_dx, s1, transpose_a=True, transpose_b=True)

        # While-loop requires the shapes of the outputs to be fixed
        M_x.set_shape([1, self.state_dim])
        S_x.set_shape([self.state_dim, self.state_dim])
        return M_x, S_x

    def compute_reward(self):
        """
        Utility method for changing the sign of the return.
        """
        return -self.training_loss()

    @property
    def maximum_log_likelihood_objective(self):
        """
        Necessary for class definition although unused in the example.
        """
        return -self.training_loss()


class MGPR(gpflow.Module):
    """
    Class for multiple GP Regression. Taken from from https://github.com/nrontsis/PILCO
    """
    def __init__(self, data, name=None):
        super(MGPR, self).__init__(name)

        self.num_outputs = data[1].shape[1]
        self.num_dims = data[0].shape[1]
        self.num_datapoints = data[0].shape[0]
        self.create_models(data)
        self.optimizers = []

    def create_models(self, data):
        self.models = []
        for i in range(self.num_outputs):
            kern = gpflow.kernels.SquaredExponential(lengthscales=tf.ones([data[0].shape[1],], dtype=gpflow.config.default_float()))
            kern.lengthscales.prior = tfd.Gamma(to_default_float(1.1), to_default_float(1/10.0)) # priors have to be included before
            kern.variance.prior = tfd.Gamma(to_default_float(1.5), to_default_float(1/2.0))    # before the model gets compiled
            self.models.append(gpflow.models.GPR((data[0], data[1][:, i:i+1]), kernel=kern))
            self.models[-1].likelihood.prior = tfd.Gamma(to_default_float(1.2), to_default_float(1/0.05))

    def set_data(self, data):
        for i in range(len(self.models)):
            if isinstance(self.models[i].data[0], gpflow.Parameter):
                self.models[i].X.assign(data[0])
                self.models[i].Y.assign(data[1][:, i:i+1])
                self.models[i].data = [self.models[i].X, self.models[i].Y]
            else:
                self.models[i].data = (data[0], data[1][:, i:i+1])

    def optimize(self, restarts=1):
        if len(self.optimizers) == 0:  # This is the first call to optimize();
            for model in self.models:
                # Create an gpflow.train.ScipyOptimizer object for every model embedded in mgpr
                optimizer = gpflow.optimizers.Scipy()
                optimizer.minimize(model.training_loss, model.trainable_variables)
                self.optimizers.append(optimizer)
        else:
            for model, optimizer in zip(self.models, self.optimizers):
                optimizer.minimize(model.training_loss, model.trainable_variables)

        for model, optimizer in zip(self.models, self.optimizers):
            best_params = {
                "lengthscales" : model.kernel.lengthscales.value(),
                "k_variance" : model.kernel.variance.value(),
                "l_variance" : model.likelihood.variance.value()}
            best_loss = model.training_loss()
            for restart in range(restarts):
                randomize(model)
                optimizer.minimize(model.training_loss, model.trainable_variables)
                loss = model.training_loss()
                if loss < best_loss:
                    best_params["k_lengthscales"] = model.kernel.lengthscales.value()
                    best_params["k_variance"] = model.kernel.variance.value()
                    best_params["l_variance"] = model.likelihood.variance.value()
                    best_loss = model.training_loss()
            model.kernel.lengthscales.assign(best_params["lengthscales"])
            model.kernel.variance.assign(best_params["k_variance"])
            model.likelihood.variance.assign(best_params["l_variance"])

    def predict_on_noisy_inputs(self, m, s):
        iK, beta = self.calculate_factorizations()
        return self.predict_given_factorizations(m, s, iK, beta)

    def calculate_factorizations(self):
        K = self.K(self.X)
        batched_eye = tf.eye(tf.shape(self.X)[0], batch_shape=[self.num_outputs], dtype=gpflow.config.default_float())
        L = tf.linalg.cholesky(K + self.noise[:, None, None]*batched_eye)
        iK = tf.linalg.cholesky_solve(L, batched_eye, name='chol1_calc_fact')
        Y_ = tf.transpose(self.Y)[:, :, None]
        beta = tf.linalg.cholesky_solve(L, Y_, name="chol2_calc_fact")[:, :, 0]
        return iK, beta

    def predict_given_factorizations(self, m, s, iK, beta):
        """
        Approximate GP regression at noisy inputs via moment matching
        IN: mean (m) (row vector) and (s) variance of the state
        OUT: mean (M) (row vector), variance (S) of the action
             and inv(s)*input-ouputcovariance
        """

        s = tf.tile(s[None, None, :, :], [self.num_outputs, self.num_outputs, 1, 1])
        inp = tf.tile(self.centralized_input(m)[None, :, :], [self.num_outputs, 1, 1])

        # Calculate M and V: mean and inv(s) times input-output covariance
        iL = tf.linalg.diag(1/self.lengthscales)
        iN = inp @ iL
        B = iL @ s[0, ...] @ iL + tf.eye(self.num_dims, dtype=gpflow.config.default_float())

        # Redefine iN as in^T and t --> t^T
        # B is symmetric so its the same
        t = tf.linalg.matrix_transpose(
                tf.linalg.solve(B, tf.linalg.matrix_transpose(iN), adjoint=True, name='predict_gf_t_calc'),
            )

        lb = tf.exp(-tf.reduce_sum(iN * t, -1)/2) * beta
        tiL = t @ iL
        c = self.variance / tf.sqrt(tf.linalg.det(B))

        M = (tf.reduce_sum(lb, -1) * c)[:, None]
        V = tf.matmul(tiL, lb[:, :, None], adjoint_a=True)[..., 0] * c[:, None]

        # Calculate S: Predictive Covariance
        R = s @ tf.linalg.diag(
                1/tf.square(self.lengthscales[None, :, :]) +
                1/tf.square(self.lengthscales[:, None, :])
            ) + tf.eye(self.num_dims, dtype=gpflow.config.default_float())

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

    def centralized_input(self, m):
        return self.X - m

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


def randomize(model, mean=1, sigma=0.01):
    """
    Randomize the GP model hyperparameters
    :param model: GPflow model object
    :param mean: mean of the noise distribution
    :param sigma: std of the nosie distribution
    """
    np.random.seed(3)

    model.kernel.lengthscales.assign(mean + sigma*np.random.normal(size=model.kernel.lengthscales.shape))
    model.kernel.variance.assign(mean + sigma*np.random.normal(size=model.kernel.variance.shape))

    # If we don't fix the noise value.
    if model.likelihood.variance.trainable:
        model.likelihood.variance.assign(mean + sigma*np.random.normal())
