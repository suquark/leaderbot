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


# ========
# evaluate
# ========

def evaluate(
        models: list,
        train: bool = False,
        print: bool = True):
    """
    Evaluate models for goodness of fit with respect to training data.

    Parameters
    ----------

    models : list[leaderbot.models.BaseModel]
        A single or a list of models to be evaluated.

        .. note::

            All models should be created using the same dataset to make proper
            comparison.

    train : bool, default=False
        If `True`, the models will be trained. If `False`, it is assumed that
        the models are pre-trained.

    print : bool, default=False
        If `True`, a table of the analysis is printed.

    Returns
    -------

    metrics : dict
        A dictionary containing the following keys and values:

        * ``'name'``: list of names of models
        * ``'n_param'``: list of number of parameters of each model
        * ``'loss'``: list of loss values of each model.
        * ``'jsd'``: list of Jensen-Shannon divergences of each model.
        * ``'kld'``: list of Kullback-Leiber divergences of each model.
        * ``'aic'``: list of Akaike information criterion of each model.
        * ``'bic'``: list of Bayesian information criterion of each model.

    Raises
    ------

    RuntimeError
        if ``train`` is `False` but at least one of the models are not
        pre-trained.

    Examples
    --------

    .. code-block:: python

        >>> import leaderbot

        >>> # Create a list of models
        >>> model1 = leaderbot.BradleyTerryScaled(data)
        >>> model2 = leaderbot.RaoKupperScaled(data)
        >>> model3 = leaderbot.DavidsonScaled(data)
        >>> models = [model1, model2, model3]

        >>> # Evaluate models
        >>> metrics = leaderbot.evaluate(models, train=True, print=True)
    """

    # Convert a single model to a singleton list
    if not isinstance(models, list):
        models = list(models)

    if train:
        for model in models:
            model.train()
    else:
        # Check a model is trained
        for model in models:
            if model.param is None:
                raise RuntimeError('Models are not trained. Set "train" to'
                                   '"True", or pre-train models in advance.')

    # Outputs
    name = []
    n_param = []
    jsd = []   # Jensen-Shannon divergence
    kld = []   # Kullback-Leibler divergence
    aic = []   # Akaike information criterion
    bic = []   # Bayesian information criterion
    loss = []  # loss function

    for model in models:

        # Model attributes
        name.append(model.__module__.split('.')[-1])
        n_param.append(model.n_param)

        # Divergences
        jsd.append(_jsd(model))
        kld.append(_kld(model))

        # Loss with no constraint (just likelihood)
        loss_ = model.loss(return_jac=False, constraint=False)
        loss.append(loss_)

        # Information criteria
        aic.append(_aic(model, loss_))
        bic.append(_bic(model, loss_))

    # Output
    metrics = {
        'name': name,
        'n_param': n_param,
        'loss': loss,
        'jsd': jsd,
        'kld': kld,
        'aic': aic,
        'bic': bic,
    }

    if print:
        pass

    return metrics


# ===
# jsd
# ===

def _jsd(model):
    """
    Jensen-Shannon divergence.
    """

    y = model.y
    p_pred = model.infer()

    y_sum = y.sum(axis=1, keepdims=True)
    y_sum[y_sum == 0] = 1.0
    p_obs = y / y_sum
    p_mean = (p_pred + p_obs) / 2.0
    jsd_ = 0.5 * (kl_divergence(p_obs, p_mean) + kl_divergence(p_pred, p_mean))
    jsd_ = jsd_.sum(axis=1)

    jsd_ = np.mean(jsd_)
    # jsd_mean = np.mean(jsd_)
    # jsd_std = np.std(jsd_)

    return jsd_


# ===
# kld
# ===

def _kld(model):
    """
    Kullback-Leibler divergence
    """

    y = model.y
    p_pred = model.infer()

    y_sum = y.sum(axis=1, keepdims=True)
    y_sum[y_sum == 0] = 1.0
    p_obs = y / y_sum
    kld_ = kl_divergence(p_obs, p_pred)
    kld_ = kld_.sum(axis=1)

    kld_ = np.mean(kld_)
    # kld_mean = np.mean(kld_)
    # kld_std = np.std(kld_)

    return kld_


# ===
# aic
# ===

def _aic(model, loss_):
    """
    Akaike information criterion
    """

    aic_ = 2.0 * model.n_param - 2.0 * loss_
    return aic_


# ===
# bic
# ===

def _bic(model, loss_):
    """
    Bayesian information criterion
    """

    y = model.y
    n_data = y.shape[0]

    bic_ = model.n_param * np.log(n_data) - 2.0 * loss_
    return bic_
