# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

from .generalization import generalization

__all__ = ['goodness_of_fit']


# ===============
# goodness of fit
# ===============

def goodness_of_fit(
        models: list,
        train: bool = False,
        tie: bool = True,
        density: bool = False,
        metric: str = 'MAE',
        report: bool = True):
    """
    Evaluate metrics for goodness of fit.

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


    tie : bool, default=True
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

    metric : {``'err'``, ``'MAPE'``, ``'SMAPE'``, ``'RMSE'``}, \
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
        :emphasize-lines: 20, 21

        >>> import leaderbot as lb

        >>> # Obtain data
        >>> data = lb.data.load()

        >>> # Create a list of models to compare
        >>> models = [
        ...    lb.models.BradleyTerry(data),
        ...    lb.models.BradleyTerryScaled(data),
        ...    lb.models.BradleyTerryScaledR(data),
        ...    lb.models.RaoKupper(data),
        ...    lb.models.RaoKupperScaled(data),
        ...    lb.models.RaoKupperScaledR(data),
        ...    lb.models.Davidson(data),
        ...    lb.models.DavidsonScaled(data),
        ...    lb.models.DavidsonScaledR(data)
        ... ]

        >>> # Evaluate models
        >>> metrics = lb.evaluate.goodness_of_fit(models, train=True,
        ...                                       report=True)

    The above code outputs the following table

    .. literalinclude:: ../_static/data/goodness_of_fit.txt
        :language: none
    """

    # When setting data to None, the model's data is used.
    metrics = generalization(models, test_data=None, train=train, tie=tie,
                             density=density, metric=metric, report=report)

    return metrics
