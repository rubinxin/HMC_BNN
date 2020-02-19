import hamiltorch
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utilities.base_model import BaseModel
from utilities.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization


class Net(nn.Module):
    def __init__(self, n_inputs, n_units=[50, 50, 50]):
        super(Net, self).__init__()
        self.bias = True
        self.fc1 = nn.Linear(n_inputs, n_units[0], bias=self.bias)
        self.fc2 = nn.Linear(n_units[0], n_units[1], bias=self.bias)
        self.fc3 = nn.Linear(n_units[1], n_units[2], bias=self.bias)
        self.out = nn.Linear(n_units[2], 1, bias=self.bias)

        self.activation = nn.Tanh()

    def forward(self, x):

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))

        return self.out(x)


class HMCBNN(BaseModel):

    def __init__(self, num_samples=3000, step_size=0.1, num_steps_per_sample=1, tau_out=1,
                 n_units_1=50, n_units_2=50, n_units_3=50, keep_every=2, normalize_input=True, normalize_output=True,
                 seed=0, RM=False, burnin=500, prior_tau=1.0):
        """
        This module performs HMC approximation for a Bayesian neural network.
        """

        # Set the random seeds
        self.seed = seed
        torch.manual_seed(seed)
        hamiltorch.set_random_seed(seed)
        # Check availability of GPU
        self.device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

        self.X = None
        self.y = None
        # Network params
        self.network = None
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.n_units_1 = n_units_1
        self.n_units_2 = n_units_2
        self.n_units_3 = n_units_3

        # HMC params
        self.num_steps_per_sample = num_steps_per_sample
        self.step_size = step_size
        self.num_samples = num_samples
        self.RM = RM
        self.burnin = burnin
        self.tau_out = tau_out
        self.prior_tau = prior_tau
        self.keep_every = keep_every

    @BaseModel._check_shapes_train
    def train(self, X, y):
        """
        Train the model on the training data
        """

        # Normalize inputs
        if self.normalize_input:
            self.X, self.X_mean, self.X_std = zero_mean_unit_var_normalization(X)
        else:
            self.X = X

        # Normalize ouputs
        if self.normalize_output:
            self.y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y)
        else:
            self.y = y
        self.y = self.y[:, None]

        # Create the neural network
        features = X.shape[1]
        N_tr = X.shape[0]
        network = Net(n_inputs=features, n_units=[self.n_units_1, self.n_units_2, self.n_units_3])
        network = network.to(self.device)

        # Set up hmc
        X_train = Variable(torch.Tensor(self.X)).to(self.device)
        y_train = Variable(torch.Tensor(self.y)).to(self.device)
        if self.RM:
            sampler = hamiltorch.Sampler.RMHMC
        else:
            sampler = hamiltorch.Sampler.HMC
            # sampler = hamiltorch.Sampler.HMC_NUTS
            self.num_samples = self.num_samples+self.burnin

        # Tau list is the prior precision (or weight decay)
        self.tau_list = torch.FloatTensor([self.prior_tau] * len(list(network.parameters())))
        params_init = hamiltorch.util.flatten(network).to(self.device).clone()
        params_hmc = hamiltorch.sample_model(network, X_train, y_train, model_loss='regression', normalizing_const=N_tr,
                                             params_init=params_init, num_samples=self.num_samples, step_size=self.step_size,
                                             num_steps_per_sample=self.num_steps_per_sample, sampler=sampler,
                                             tau_out=self.tau_out, tau_list=self.tau_list, burn=self.burnin)
        self.params_hmc = params_hmc
        self.model = network

    @BaseModel._check_shapes_predict
    def predict(self, X_test):
        """
        Returns the predictive mean and variance of at test data
        """

        # Normalize inputs
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_ = X_test

        X_test_tensor = Variable(torch.Tensor(X_)).to(self.device)

        # Perform MC approximation for marginalisation
        parm_hmc_final = self.params_hmc.copy()[::self.keep_every]
        y = torch.ones(X_.shape[0], 1)
        print(f'n_sample_used={len(parm_hmc_final)}')
        pred_list, _ = hamiltorch.predict_model(self.model, X_test_tensor, y, samples=parm_hmc_final,
                                                model_loss='regression', tau_out=self.tau_out, tau_list=self.tau_list)
        pred_list_np = pred_list.cpu().numpy().squeeze()
        m = np.mean(pred_list_np, 0)
        v = np.var(pred_list_np, 0)

        # Denormalise the predictive mean and variance
        if self.normalize_output:
            m = zero_mean_unit_var_denormalization(m, self.y_mean, self.y_std)
            v *= self.y_std ** 2
        m = m.flatten()
        v = v.flatten()

        return m, v
