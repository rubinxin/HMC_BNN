import numpy as np
from hmcbnn import HMCBNN
from bohamiann import Bohamiann
import pickle
import argparse
import os
from scipy.stats import norm

def validate(m, v, y_test):
    total_nll = - np.mean(norm.logpdf(y_test, loc=m, scale=np.sqrt(v)))
    total_mse = np.mean((y_test - m) ** 2)
    return total_nll, total_mse

def load_uci_data(data_directory, split_number = 2):
    # data_directory = 'yacht'
    # split number (up to 20 except protein data )
    # this number decide on the random split on the data but split/train ratio remain the same for all split number
    split = split_number
    _DATA_DIRECTORY_PATH = data_directory + "/data/"
    # _DATA_DIRECTORY_PATH = "./datasets/" + data_directory + "/data/"

    _DATA_FILE = _DATA_DIRECTORY_PATH + "data.txt"
    _INDEX_FEATURES_FILE = _DATA_DIRECTORY_PATH + "index_features.txt"
    _INDEX_TARGET_FILE = _DATA_DIRECTORY_PATH + "index_target.txt"
    _N_SPLITS_FILE = _DATA_DIRECTORY_PATH + "n_splits.txt"

    def _get_index_train_test_path(split_num, train = True):
        """
           Method to generate the path containing the training/test split for the given
           split number (generally from 1 to 20).
           @param split_num      Split number for which the data has to be generated
           @param train          Is true if the data is training data. Else false.
           @return path          Path of the file containing the requried data
        """
        if train:
            return _DATA_DIRECTORY_PATH + "index_train_" + str(split_num) + ".txt"
        else:
            return _DATA_DIRECTORY_PATH + "index_test_" + str(split_num) + ".txt"

    # We load the data
    data = np.loadtxt(_DATA_FILE)

    # We load the indexes for the features and for the target
    index_features = np.loadtxt(_INDEX_FEATURES_FILE)
    index_target = np.loadtxt(_INDEX_TARGET_FILE)

    # X: (N, d),  y: (N,)
    X = data[ : , [int(i) for i in index_features.tolist()] ]
    y = data[ : , int(index_target.tolist()) ]

    # We iterate over the training test splits
    n_splits = np.loadtxt(_N_SPLITS_FILE)

    # We load the indexes of the training and test sets
    # print ('Loading file: ' + _get_index_train_test_path(split, train=True))
    # print ('Loading file: ' + _get_index_train_test_path(split, train=False))
    index_train = np.loadtxt(_get_index_train_test_path(split, train=True))
    index_test = np.loadtxt(_get_index_train_test_path(split, train=False))

    X_train = X[ [int(i) for i in index_train.tolist()] ]
    y_train = y[ [int(i) for i in index_train.tolist()] ]

    X_test = X[ [int(i) for i in index_test.tolist()] ]
    y_test = y[ [int(i) for i in index_test.tolist()] ]

    X_train_original = X_train
    y_train_original = y_train
    num_training_examples = int(0.8 * X_train.shape[0])
    X_validation = X_train[num_training_examples:, :]
    y_validation = y_train[num_training_examples:]
    X_train = X_train[0:num_training_examples, :]
    y_train = y_train[0:num_training_examples]


    return X_train, y_train, X_test, y_test


def regression_test(method='SGHMC', task_name = 'yacht', seed_range=10):

    test_nll_s = []
    rmse_list_s = []
    for s in range(seed_range):

        # ------------ generate data ------------
        if task in ['yacht', 'concrete', 'wine-quality-red', 'bostonHousing']:
            X_train, y_train, X_test, y_test = load_uci_data(data_directory=f'uci_datasets/{task}',split_number=s)

        print(f'data loaded: train/test ratio={X_train.shape[0]/X_test.shape[0]} and run{method} at seed={s}')
        np.random.seed(s)
        normalize_input = True
        normalize_output = True

        if method == 'HMC':
            # ------------ define and train hmc BNN model ------------
            num_samples = 300
            num_steps_per_sample = 5
            tau_out = 1   # tau_out is the likelihood precision (1/aleatoric_uncertainty)
                            # --> need to remove this by redefining the loss function in hamiltorch later
            init_tau = 1    # init_tau is the prior precision (or weight decay)
            step_size = 0.03
            print(f'orgin_step_size ={step_size}')
            num_burn_in_steps = 100
            model = HMCBNN(num_samples= num_samples, step_size = step_size, num_steps_per_sample = num_steps_per_sample,
                           tau_out=tau_out, prior_tau=init_tau, burnin=num_burn_in_steps,
                           normalize_input=normalize_input, normalize_output=normalize_output, seed=s)
            model.train(X_train, y_train.flatten())

        else:
            # ------------ define and train sghmc BNN model ------------
            num_steps = 5000
            num_burn_in_steps = 2000
            keep_every = 30
            model = Bohamiann(normalize_input=normalize_input, normalize_output=normalize_output, seed=s)
            model.train(X_train, y_train, num_steps=num_steps, keep_every=keep_every,
                        num_burn_in_steps=num_burn_in_steps, verbose=False)

        # ------------ validate BNN model ------------
        m_test, v_test = model.predict(X_test)
        test_nll, rmse = validate(m_test, v_test, y_test.flatten())

        test_nll_s.append(test_nll)
        rmse_list_s.append(rmse)
        print(f'{method} on {task_name}: test_nll={test_nll}, rmse={rmse}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run UCI regression Experiments")
    parser.add_argument('-m', '--method', help='Model type: BOHAM',
                        default='HMC', type=str)
    parser.add_argument('-s', '--nseed', help='Total random seeds',
                        default=3, type=int)
    parser.add_argument('-t', '--task', help='Task name',
                        default='bostonHousing', type=str)

    args = parser.parse_args()
    print(f"Got arguments: \n{args}")

    nseed = args.nseed
    method = args.method
    task = args.task
    regression_test(method=method, task_name=task, seed_range=nseed)