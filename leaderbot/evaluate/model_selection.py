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
from ._util import float_to_str, evaluate_cel

__all__ = 'model_selection'


# ===============
# model_selection
# ===============

def model_selection(
        models: list,
        train: bool = False,
        tie: bool = False,
        report: bool = True):
    """
    Evaluate model selection.

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

    tie : bool, default=False
        If `False`, ties in the data are not counted toward model evaluation.
        This option is only effective on
        :class:`leaderbot.models.BradleyTerry` model, and has no effect on the
        other models.

    report : bool, default=False
        If `True`, a table of the analysis is printed.

    Returns
    -------

    metrics : dict
        A dictionary containing the following keys and values:

        * ``'name'``: list of names of the models.
        * ``'n_param'``: list of number of parameters of the models.
        * ``'nll'``: list of negative log-likelihood values of the models.
        * ``'aic'``: list of Akaike information criterion of the models.
        * ``'bic'``: list of Bayesian information criterion of the models.
        * ``'cel_win'``: list of cross entropies for win outcomes.
        * ``'cel_loss'``: list of cross entropies for loss outcomes.
        * ``'cel_tie'``: list of cross entropies for tie outcomes.
        * ``'cel_all'``: list of cross entropies for all outcomes.

    Raises
    ------

    RuntimeError
        if ``train`` is `False` but at least one of the models are not
        pre-trained.

    Examples
    --------

    .. code-block:: python
        :emphasize-lines: 23, 24

        >>> import leaderbot as lb
        >>> from leaderbot.models import BradleyTerry as BT
        >>> from leaderbot.models import RaoKupper as RK
        >>> from leaderbot.models import Davidson as DV

        >>> # Obtain data
        >>> data = lb.data.load()

        >>> # Create a list of models to compare
        >>> models = [
        ...    BT(data, k_cov=None),
        ...    BT(data, k_cov=0),
        ...    BT(data, k_cov=1),
        ...    RK(data, k_cov=None, k_tie=0),
        ...    RK(data, k_cov=0, k_tie=0),
        ...    RK(data, k_cov=1, k_tie=1),
        ...    DV(data, k_cov=None, k_tie=0),
        ...    DV(data, k_cov=0, k_tie=0),
        ...    DV(data, k_cov=0, k_tie=1)
        ... ]

        >>> # Evaluate models
        >>> metrics = lb.evaluate.model_selection(models, train=True,
        ...                                       report=True)

    The above code outputs the following table

    .. literalinclude:: ../_static/data/model_selection.txt
        :language: none
    """

    # Convert a single model to a singleton list
    if not isinstance(models, list):
        models = [models]

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
    nll = []  # Negative log-likelihood
    cel_win = []  # Cross-entropy loss for win
    cel_loss = []  # Cross-entropy loss for loss
    cel_tie = []  # Cross-entropy loss for tie
    cel_all = []  # Cross-entropy loss for all win, loss, and tie
    kld = []  # Kullback-Leibler divergence
    jsd = []  # Jensen-Shannon divergence
    aic = []  # Akaike information criterion
    bic = []  # Bayesian information criterion

    for model in models:

        # Model attributes
        class_name = model.__class__.__name__
        name.append(class_name)
        n_param.append(model.n_param)

        # For any model other than BradleyTerry, always consider tie.
        if not class_name.startswith('BradleyTerry'):
            tie_ = True
        else:
            tie_ = tie

        # Cross entropy loss
        cel_, cel_all_ = evaluate_cel(model, data=None, tie=tie_)

        cel_win.append(cel_[0])
        cel_loss.append(cel_[1])
        cel_tie.append(cel_[2])
        cel_all.append(cel_all_)

        # Loss with no constraint (just likelihood)
        nll_ = model.loss(return_jac=False, constraint=False)
        nll.append(nll_)

        # Information criteria
        aic.append(_evaluate_aic(model, nll_))
        bic.append(_evaluate_bic(model, nll_))

    # Output
    metrics = {
        'name': name,
        'n_param': n_param,
        'nll': nll,
        'cel_win': cel_win,
        'cel_loss': cel_loss,
        'cel_tie': cel_tie,
        'cel_all': cel_all,
        'jsd': jsd,
        'kld': kld,
        'aic': aic,
        'bic': bic,
    }

    if report:
        print('+----+--------------+---------+--------+---------------------' +
              '-----------+---------+---------+')
        print('|    |              |         |        |               CEL   ' +
              '           |         |         |')
        print('| id | model        | # param |    NLL |    all     win    lo' +
              'ss     tie |     AIC |     BIC |')
        print('+----+--------------+---------+--------+---------------------' +
              '-----------+---------+---------+')

        for i in range(len(name)):

            name_length = 12
            name_str = name[i]
            if len(name_str) > name_length:
                name_str = name_str[:(name_length - 3)] + '...'
            name_str = name_str.ljust(name_length)

            cel_win_str = float_to_str(cel_win[i])
            cel_loss_str = float_to_str(cel_loss[i])
            cel_tie_str = float_to_str(cel_tie[i])
            cel_all_str = float_to_str(cel_all[i])

            print(f'| {i+1:>2d} '
                  f'| {name_str:<12s} '
                  f'| {n_param[i]:>7} '
                  f'| {nll[i]:>0.4f} '
                  f'| {cel_all_str}'
                  f'  {cel_win_str}'
                  f'  {cel_loss_str}'
                  f'  {cel_tie_str} '
                  f'| {aic[i]:>7.1f} '
                  f'| {bic[i]:>7.1f} |')

        print('+----+--------------+---------+--------+---------------------' +
              '-----------+---------+---------+')

    return metrics


# ============
# evaluate aic
# ============

def _evaluate_aic(model, nll):
    """
    Akaike information criterion.
    """

    aic = 2.0 * model.n_param - 2.0 * nll
    return aic


# ============
# evaluate bic
# ============

def _evaluate_bic(model, nll):
    """
    Bayesian information criterion
    """

    x = model.x
    n_samples = x.shape[0]

    bic = model.n_param * np.log(n_samples) - 2.0 * nll
    return bic
