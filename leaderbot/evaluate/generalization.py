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
from ._util import evaluate_kld, evaluate_jsd, evaluate_error
from ..data import DataType

__all__ = ['generalization']


# ==============
# generalization
# ==============

def generalization(
        models: list,
        test_data: DataType = None,
        train: bool = False,
        tie: bool = False,
        density: bool = False,
        metric: str = 'MAE',
        report: bool = True):
    """
    Evaluate metrics for generalization performance.

    Parameters
    ----------

    models : list[leaderbot.models.BaseModel]
        A single or a list of models to be evaluated.

        .. note::

            All models should be created using the same dataset to make proper
            comparison.

    test_data : DataType
        Test data to evaluate model prediction. If `None`, the model's
        training data is used. See :func:`leaderbot.evaluate.goodness_of_fit`.

    train : bool, default=False
        If `True`, the models will be trained. If `False`, it is assumed that
        the models are pre-trained.

    tie : bool, default=False
        If `False`, ties in the data are not counted toward model evaluation.
        This option is only effective on
        :class:`leaderbot.models.BradleyTerry` model, and has no effect on the
        other models.

    density : bool, default=False
        If `False`, the frequency (count) of events are evaluated. If `True`,
        the probability density of the events are evaluated.

        .. note::
            When ``density`` is set to `True`, the probability density values
            are multiplied by ``100.0``, and the results of errors should be
            interpreted in percent.

    metric : {``'MAE'``, ``'MAPE'``, ``'SMAPE'``, ``'RMSE'``}, \
            default= ``'MAE'``
        The metric of comparison:

        * ``'MAE'``: Mean absolute error.
        * ``'MAPE'``: Mean absolute percentage error.
        * ``'SMAPE'``: Symmetric mean absolute percentage error.
        * ``'RMSE'``: Root mean square error.

    report : bool, default=False
        If `True`, a table of the analysis is printed.

    Returns
    -------

    metrics : dict
        A dictionary containing the following keys and values:

        * ``'name'``: list of names of the models.
        * ``'kld'``: list of Kullback-Leiber divergences of the models.
        * ``'jsd'``: list of Jensen-Shannon divergences of the models.
        * ``'err_win'``: list of errors for win predictions.
        * ``'err_loss'``: list of errors for loss predictions.
        * ``'err_tie'``: list of errors for tie predictions.
        * ``'err_all'``: list of errors for overall predictions.

    Raises
    ------

    RuntimeError
        if ``train`` is `False` but at least one of the models are not
        pre-trained.

    Examples
    --------

    .. code-block:: python
        :emphasize-lines: 26, 27

        >>> import leaderbot as lb
        >>> from leaderbot.models import BradleyTerry as BT
        >>> from leaderbot.models import RaoKupper as RK
        >>> from leaderbot.models import Davidson as DV

        >>> # Obtain data
        >>> data = lb.data.load()

        >>> # Split data to training and test data
        >>> training_data, test_data = lb.data.split(data, test_ratio=0.2)

        >>> # Create a list of models to compare
        >>> models = [
        ...    BT(training_data, k_cov=None),
        ...    BT(training_data, k_cov=0),
        ...    BT(training_data, k_cov=1),
        ...    RK(training_data, k_cov=None, k_tie=0),
        ...    RK(training_data, k_cov=0, k_tie=0),
        ...    RK(training_data, k_cov=1, k_tie=1),
        ...    DV(training_data, k_cov=None, k_tie=0),
        ...    DV(training_data, k_cov=0, k_tie=0),
        ...    DV(training_data, k_cov=0, k_tie=1)
        ... ]

        >>> # Evaluate generalization on test data
        >>> metrics = lb.evaluate.generalization(models, test_data, train=True,
        ...                                      report=True)

    The above code outputs the following table

    .. literalinclude:: ../_static/data/generalization.txt
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
    kld = []  # Kullback-Leiber divergence
    jsd = []  # Jensen-Shannon divergence
    err_win = []  # Mean absolute error, win
    err_loss = []  # Mean absolute error, loss
    err_tie = []  # Mean absolute error, tie
    err_all = []  # Mean absolute error, all

    for model in models:

        # Model attributes
        class_name = model.__class__.__name__
        name.append(class_name)

        # For any model other than BradleyTerry, always consider tie.
        if not class_name.startswith('BradleyTerry'):
            tie_ = True
        else:
            tie_ = tie

        # Divergences
        jsd.append(evaluate_jsd(model, data=test_data, tie=tie_))
        kld.append(evaluate_kld(model, data=test_data, tie=tie_))

        # Mean absolute error of the prediction on test data
        err_, err_all_ = evaluate_error(model, data=test_data, metric=metric,
                                        tie=tie_, density=density)

        err_win.append(err_[0])
        err_loss.append(err_[1])
        err_tie.append(err_[2])
        err_all.append(err_all_)

    # Output
    metrics = {
        'name': name,
        'kld': kld,
        'jsd': jsd,
        'err_win': err_win,
        'err_loss': err_loss,
        'err_tie': err_tie,
        'err_all': err_all,
    }

    if report:
        print('+----+--------------+----------------------------+------+----' +
              '--+')
        print(f'|    |              |           {metric:>5s}            |   ' +
              '   |      |')
        print('| id | model        |   win   loss    tie    all | KLD% | JSD' +
              '% |')
        print('+----+--------------+----------------------------+------+----' +
              '--+')

        for i in range(len(name)):

            name_length = 12
            name_str = name[i]
            if len(name_str) > name_length:
                name_str = name_str[:(name_length - 3)] + '...'
            name_str = name_str.ljust(name_length)

            if np.isnan(err_tie[i]):
                tie_str = '-----'
            else:
                if density:
                    tie_str = f'{err_tie[i]:>5.2f}'
                else:
                    tie_str = f'{err_tie[i]:>5.1f}'

            if density:
                print(f'| {i+1:>2d} '
                      f'| {name_str:<12s} '
                      f'| {err_win[i]:>5.2f} '
                      f' {err_loss[i]:>5.2f} '
                      f' {tie_str} '
                      f' {err_all[i]:>5.2f} '
                      f'| {100.0 * kld[i]:>4.2f} '
                      f'| {100.0 * jsd[i]:>4.2f} |')
            else:
                print(f'| {i+1:>2d} '
                      f'| {name_str:<12s} '
                      f'| {err_win[i]:>5.1f} '
                      f' {err_loss[i]:>5.1f} '
                      f' {tie_str} '
                      f' {err_all[i]:>5.1f} '
                      f'| {100.0 * kld[i]:>4.2f} '
                      f'| {100.0 * jsd[i]:>4.2f} |')

        print('+----+--------------+----------------------------+------+----' +
              '--+')

    return metrics
