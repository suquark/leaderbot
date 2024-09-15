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
        report: bool = True):
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

    report : bool, default=False
        If `True`, a table of the analysis is printed.

    Returns
    -------

    metrics : dict
        A dictionary containing the following keys and values:

        * ``'name'``: list of names of the models.
        * ``'n_param'``: list of number of parameters of the models.
        * ``'loss'``: list of loss values of the models.
        * ``'jsd'``: list of Jensen-Shannon divergences of the models.
        * ``'kld'``: list of Kullback-Leiber divergences of the models.
        * ``'aic'``: list of Akaike information criterion of the models.
        * ``'bic'``: list of Bayesian information criterion of the models.

    Raises
    ------

    RuntimeError
        if ``train`` is `False` but at least one of the models are not
        pre-trained.

    Examples
    --------

    .. code-block:: python

        >>> import leaderbot as lb

        >>> # Obtain data
        >>> data = lb.data.load()

        >>> # Create models to compare
        >>> model_01 = lb.models.BradleyTerry(data)
        >>> model_02 = lb.models.BradleyTerryScaled(data)
        >>> model_03 = lb.models.BradleyTerryScaledR(data)
        >>> model_04 = lb.models.RaoKupper(data)
        >>> model_05 = lb.models.RaoKupperScaled(data)
        >>> model_06 = lb.models.RaoKupperScaledR(data)
        >>> model_07 = lb.models.Davidson(data)
        >>> model_08 = lb.models.DavidsonScaled(data)
        >>> model_09 = lb.models.DavidsonScaledR(data)

        >>> # Create a list of models
        >>> models = [model_01, model_02, model_03,
        ...           model_04, model_05, model_06,
        ...           model_07, model_08, model_09]

        >>> # Evaluate models
        >>> metrics = lb.evaluate(models, train=True, report=True)

        The above code outputs the following table

        .. literalinclude:: ../_static/data/evaluate.txt
            :language: none
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
        name.append(model.__class__.__name__)
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

    if report:
        print('+-----------------------+---------+--------+--------+--------' +
              '+--------+---------+')
        print('| name                  | # param | loss   | KLD    | JSD %  ' +
              '| AIC    | BIC     |')
        print('+-----------------------+---------+--------+--------+--------' +
              '+--------+---------+')

        for i in range(len(name)):

            name_length = 21
            name_str = name[i]
            if len(name_str) > name_length:
                name_str = name_str[:(name_length - 3)] + '...'
            name_str = name_str.ljust(name_length)

            kld_f = kld[i]
            if kld_f == float('inf'):
                kld_str = ' ' * (len(f'{0:0.1f}')) + 'inf'
            elif kld_f == float('-inf'):
                kld_str = '-' + ' ' * (len(f'{0:0.1f}') - 1) + 'inf'
            else:
                kld_str = f'{kld_f:>0.4f}'

            print(f'| {name_str:<21s} '
                  f'| {n_param[i]:>7} '
                  f'| {loss[i]:>0.4f} '
                  f'| {kld_str} '
                  f'| {100.0 * jsd[i]:>0.4f} '
                  f'| {aic[i]:>0.2f} '
                  f'| {bic[i]:>0.2f} |')

        print('+-----------------------+---------+--------+--------+--------' +
              '+-------+----------+')

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
