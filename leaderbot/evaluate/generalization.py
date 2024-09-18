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
from ._util import float_to_str, evaluate_kld, evaluate_jsd, evaluate_mae
from ..data import DataType

__all__ = ['generalization']


# ==============
# generalization
# ==============

def generalization(
        models: list,
        test_data: DataType = None,
        train: bool = False,
        tie: bool = True,
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

    tie : bool, default=True
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
        * ``'kld'``: list of Kullback-Leiber divergences of the models.
        * ``'jsd'``: list of Jensen-Shannon divergences of the models.
        * ``'mae_win'``: list of mean absolute error for win predictions.
        * ``'mae_loss'``: list of mean absolute error for loss predictions.
        * ``'mae_tie'``: list of mean absolute error for tie predictions.
        * ``'mae_all'``: list of mean absolute error for overall predictions.

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

        >>> # Obtain data
        >>> data = lb.data.load()

        >>> # Split data to training and test data
        >>> training_data, test_data = lb.data.split(data, test_ratio=0.2)

        >>> # Create a list of models to compare
        >>> models = [
        ...    lb.models.BradleyTerry(training_data),
        ...    lb.models.BradleyTerryScaled(training_data),
        ...    lb.models.BradleyTerryScaledR(training_data),
        ...    lb.models.RaoKupper(training_data),
        ...    lb.models.RaoKupperScaled(training_data),
        ...    lb.models.RaoKupperScaledR(training_data),
        ...    lb.models.Davidson(training_data),
        ...    lb.models.DavidsonScaled(training_data),
        ...    lb.models.DavidsonScaledR(training_data)
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
    mae_win = []  # Mean absolute error, win
    mae_loss = []  # Mean absolute error, loss
    mae_tie = []  # Mean absolute error, tie
    mae_all = []  # Mean absolute error, all

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
        mae_, mae_all_ = evaluate_mae(model, data=test_data, tie=tie_,
                                      density=False)

        mae_win.append(mae_[0])
        mae_loss.append(mae_[1])
        mae_tie.append(mae_[2])
        mae_all.append(mae_all_)

    # Output
    metrics = {
        'name': name,
        'kld': kld,
        'jsd': jsd,
        'mae_win': mae_win,
        'mae_loss': mae_loss,
        'mae_tie': mae_tie,
        'mae_all': mae_all,
    }

    if report:
        print('+-----------------------+----------------------------+-------' +
              '-+--------+')
        print('|                       |    Mean Absolute Error     |       ' +
              ' |        |')
        print('| model                 |   win   loss    tie    all | KLD   ' +
              ' | JSD %  |')
        print('+-----------------------+----------------------------+-------' +
              '-+--------+')

        for i in range(len(name)):

            name_length = 21
            name_str = name[i]
            if len(name_str) > name_length:
                name_str = name_str[:(name_length - 3)] + '...'
            name_str = name_str.ljust(name_length)

            kld_str = float_to_str(kld[i])

            if np.isnan(mae_tie[i]):
                tie_str = '-----'
            else:
                tie_str = f'{mae_tie[i]:>5.2f}'

            print(f'| {name_str:<21s} '
                  f'| {mae_win[i]:>5.2f} '
                  f' {mae_loss[i]:>5.2f} '
                  f' {tie_str} '
                  f' {mae_all[i]:>5.2f} '
                  f'| {kld_str} '
                  f'| {100.0 * jsd[i]:>0.4f} |')

        print('+-----------------------+----------------------------+-------' +
              '-+--------+')

    return metrics
