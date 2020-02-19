import numpy as np
import torch
import torch.nn as nn

def utility(util_type='recent', Y_train=0):
    '''Inputs:
    y_true: true values (N,D)
    y_pred: predicted values (N,D)
    utility_type: the type of utility function to be used for maximisation
    y_ob: training data
    '''

    def util(y_pred_samples, H_x, y_true_batch):

        threshold = np.mean(Y_train)
        M = np.percentile(Y_train, 90)
        scale = 0

        if util_type == 'se_ysample_clip':
            l = ((y_pred_samples - H_x) ** 2)
            u = torch.exp(-l / M)
            G = torch.mean(u, 0) + 1e-8
            log_G_unclipped = torch.log(G)
            log_G_clip = scale * torch.ones_like(log_G_unclipped)
            log_G = torch.where(y_true_batch < threshold, log_G_unclipped, log_G_clip)


        elif util_type == 'linear_se_ysample_clip':
            l = ((y_pred_samples - H_x) ** 2)
            l_mean = torch.mean(l, 0)
            log_G_unclipped = - l_mean/M
            log_G_clip = scale * torch.ones_like(log_G_unclipped)
            log_G = torch.where(y_true_batch < threshold, log_G_unclipped, log_G_clip)

        elif util_type == 'se_ytrue_clip':
            l = ((y_true_batch - H_x) ** 2)
            u = torch.exp(-l / M)
            G = u  # 1 sample at y_true
            log_G_unclipped = torch.log(G)
            log_G_clip = scale * torch.ones_like(log_G_unclipped)
            log_G = torch.where(y_true_batch < threshold, log_G_unclipped, log_G_clip)

        elif util_type == 'linear_se_ytrue_clip':
            l = ((y_true_batch - H_x) ** 2)
            # l_mean = torch.mean(l, 0)
            log_G_unclipped = - l/M
            log_G_clip = scale * torch.ones_like(log_G_unclipped)
            log_G = torch.where(y_true_batch < threshold, log_G_unclipped, log_G_clip)


        return log_G

    return util

def cal_loss(y_true, output, util, H_x, y_pred_samples, log_var, regularization=None):
    a = 1.0
    if regularization is None:
        mse_loss = heteroscedastic_loss(y_true, output, log_var)
    else:
        mse_loss_1 = heteroscedastic_loss(y_true, output, log_var)
        mse_loss = mse_loss_1 + regularization

    log_condi_gain = util(y_pred_samples, H_x, y_true)

    utility_value = a * log_condi_gain.mean()
    calibrated_loss = mse_loss - utility_value

    return calibrated_loss, mse_loss, log_condi_gain

def heteroscedastic_loss(true, mean, log_var):
    precision = torch.exp(-log_var)
    return torch.mean(torch.sum(precision * (true - mean)**2 + log_var, 1), 0)

