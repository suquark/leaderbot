# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import numpy as np
from ..models.util import kl_divergence

__all__ = ['float_to_str', 'evaluate_cel', 'evaluate_kld', 'evaluate_jsd',
           'evaluate_mae']


# ============
# float to str
# ============

def float_to_str(x):
    """
    Convert a float number to string with fixed length.
    """

    if x == float('inf'):
        x_str = ' ' * (len(f'{0:0.1f}')) + 'inf'
    elif x == float('-inf'):
        x_str = '-' + ' ' * (len(f'{0:0.1f}') - 1) + 'inf'
    elif np.isnan(x):
        x_str = ' ' * (len(f'{0:0.1f}')) + 'nan'
    else:
        x_str = f'{x:>0.4f}'

    return x_str


# ============
# evaluate mae
# ============

def evaluate_mae(model, data=None, tie=True, density=False):
    """
    Mean absolute error.
    """

    if data is not None:
        x = data['X']
        y = data['Y']
    else:
        x = model.x
        y = model.y

    p_pred = model.infer(x)

    if not tie:
        # Remove the third column that corresponds to the tie data
        y = y[:, :-1]
        p_pred = p_pred[:, :-1]

    n_obs = y
    n_obs_sum = n_obs.sum(axis=1, keepdims=True)
    n_obs_sum = np.tile(n_obs_sum, (1, n_obs.shape[1]))
    p_obs = n_obs / n_obs_sum

    n_pred = n_obs_sum * p_pred

    if density:
        abs_diff = np.abs(p_obs - p_pred)
    else:
        abs_diff = np.abs(n_obs - n_pred)

    mae = np.mean(abs_diff, axis=0)
    mae_all = np.mean(abs_diff)

    if not tie:
        mae = np.r_[mae, np.nan]

    return mae, mae_all


# ============
# evaluate cel
# ============

def evaluate_cel(model, data=None, tie=True):
    """
    Cross-Entropy Loss.


    Notes
    -----

    When CEL is not inf, its value is exactly identical to NLL.

    The Bradley-Terry models (where no tie is included), the CEL value is non.
    For all other models, CEL and NLL are identical.
    """

    if data is not None:
        x = data['X']
        y = data['Y']
    else:
        x = model.x
        y = model.y

    p_pred = model.infer(x)

    if not tie:
        # Remove the third column that corresponds to the tie data
        y = y[:, :-1]
        p_pred = p_pred[:, :-1]

    cel = y * np.log(p_pred)
    cel[y == 0] = 0.0
    cel = -cel.sum() / y.sum()

    return cel


# ============
# evaluate kld
# ============

def evaluate_kld(model, data=None, tie=True):
    """
    Kullback-Leibler divergence.
    """

    if data is not None:
        x = data['X']
        y = data['Y']
    else:
        x = model.x
        y = model.y

    p_pred = model.infer(x)

    if not tie:
        # Remove the third column that corresponds to the tie data
        y = y[:, :-1]
        p_pred = p_pred[:, :-1]

    y_sum = y.sum(axis=1, keepdims=True)
    y_sum[y_sum == 0] = 1.0
    p_obs = y / y_sum

    kld = kl_divergence(p_obs, p_pred)
    kld = kld.sum(axis=1)

    kld = np.mean(kld)
    # kld_mean = np.mean(kld)
    # kld_std = np.std(kld)

    return kld


# ============
# evaluate jsd
# ============

def evaluate_jsd(model, data=None, tie=True):
    """
    Jensen-Shannon divergence.
    """

    if data is not None:
        x = data['X']
        y = data['Y']
    else:
        x = model.x
        y = model.y

    p_pred = model.infer(x)

    if not tie:
        # Remove the third column that corresponds to the tie data
        y = y[:, :-1]
        p_pred = p_pred[:, :-1]

    y_sum = y.sum(axis=1, keepdims=True)
    y_sum[y_sum == 0] = 1.0
    p_obs = y / y_sum
    p_mean = (p_pred + p_obs) / 2.0

    jsd = 0.5 * (kl_divergence(p_obs, p_mean) + kl_divergence(p_pred, p_mean))
    jsd = jsd.sum(axis=1)

    jsd = np.mean(jsd)
    # jsd_mean = np.mean(jsd)
    # jsd_std = np.std(jsd)

    return jsd
