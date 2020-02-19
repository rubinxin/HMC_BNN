import numpy as np
from utilities.utilities import get_init_data
from exps_tasks.math_functions import get_function
from models.mcdrop import MCDROPWarp
from utilities.utilities import load_uci_data
from pybnn.hmcbnn1L import HMCBNN
import argparse
import os
import pickle

class hyper_learning_for_hmcnn(object):
    def __init__(self, task='yacht', n_units= 50, activation = 'tanh', seed=0):

        # ------------ saving paths created ------------
        self.saving_model_path = None
        # ------------ generate train and test data ------------
        np.random.seed(seed)
        # ------------ generate data ------------
        if task in ['yacht', 'concrete', 'wine-quality-red', 'bostonHousing']:
            X_train, y_train, X_test, y_test = load_uci_data(data_directory=f'exps_tasks/datasets/{task}',split_number=seed)
        else:
            maths_f, x_bounds, _, true_fmin = get_function(task)
            n_train = 5000
            n_test  = 2000

            X_train, y_train = get_init_data(obj_func=maths_f, noise_var=1e-6, n_init=n_train, bounds=x_bounds)
            X_test, y_test = get_init_data(obj_func=maths_f, noise_var=1e-6, n_init=n_test, bounds=x_bounds)
        # X_train, y_train, X_test, y_test = load_uci_data(data_directory=f'./datasets/{task}', split_number=seed)

        self.X_train = X_train
        self.y_train = y_train.flatten()
        self.X_test = X_test
        self.y_test = y_test.flatten()

        # ------------  bnn hyperparameters ------------
        self.num_samples = 300
        self.keep_every = 3
        self.activation = activation
        self.seed = seed
        # bnds for hyperparameters to be tuned
        self.true_bnds = np.array([[2, 50.0],         # num_steps_per_sample
                                   [100, 5000],         # tau_out
                                   [1e-3, 1e-1],       # length_scale
                                   [1e-4, 5e-2]]       # step_size
                                    )
        self.d = len(self.true_bnds)

    def evaluate(self, log_theta):
        '''
        :param theta: 1 x 2 [log_tau, log_dropout_p]
        :return: test_ll
        '''
        log_theta = np.atleast_2d(log_theta)
        log_bnds = np.log(self.true_bnds)
        # ----- rescale from [-1, 1] to the true search range ------------
        r = log_bnds[:,1]-log_bnds[:, 0]
        log_theta = r * (log_theta + 1) / 2 + log_bnds[:, 0]
        theta = np.exp(log_theta)
        print(f'hyper={theta}')

        # ----- hyperparameters to be tuned ------
        num_steps_per_sample = int(theta[:, 0])
        tau_out = float(theta[:, 1])
        length_scale = float(theta[:, 2])
        step_size = float(theta[:, 3])

        self.num_burn_in_steps = int(self.num_samples * num_steps_per_sample - self.num_samples)
        if self.num_burn_in_steps < 100:
            self.num_burn_in_steps = 100

        model = HMCBNN(num_samples=self.num_samples, step_size=step_size, num_steps_per_sample=num_steps_per_sample,
                       tau_out=tau_out, keep_every=self.keep_every, lengthscale=length_scale, burnin=self.num_burn_in_steps,
                       normalize_input=True, normalize_output=True, seed=self.seed, actv=self.activation)

        # ------------ train and validate model ------------
        model.train(self.X_train, self.y_train)
        test_nll, rmse = model.validate(self.X_test, self.y_test)
        nll_rmse_sum = test_nll + rmse
        # if np.isnan(test_nll):
        #     y = np.atleast_2d([50])
        # else:
        #     y = np.atleast_2d(test_nll)
        if np.isnan(test_nll):
            sum_fail = 100 + np.random.rand(1)
            y = np.atleast_2d(sum_fail)
        else:
            y = np.atleast_2d(nll_rmse_sum)

        return y

if __name__ == '__main__':
    hplearn = hyper_learning_for_hmcnn(task='yacht')
    # num_steps_per_sample, tau_out, length_scale, step_size   --> seems (step_size x num_steps_per_sample <=1e-1)
    theta_init = np.array([[2, 100, 1e-3, 5e-2],
                           [2, 200, 1e-3, 3e-3],
                           [5, 2000, 2e-3, 1e-2],
                           [20, 500, 1e-1, 5e-3],
                           [30, 5000, 1e-2, 1e-4],
                           [50, 3000, 1e-2, 3e-4]])
    log_theta_init = np.log(theta_init)
    log_bnds = np.log(hplearn.true_bnds)
    log_theta_scaled = 2 * (log_theta_init - log_bnds[:, 0]) / (log_bnds[:, 1] - log_bnds[:, 0]) - 1

    for i in range(6):
        x = log_theta_scaled[i,:]
        print(x)
        y = hplearn.evaluate(x)
