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
from ..data import DataType


# ========
# evaluate
# ========

def evaluate(
        data: DataType,
        models: list,
        train: bool = False,
        print: bool = True):
    """
    Evaluate models.

    Parameters
    ----------

    models : list or leaderbot.baseModel
        A single model or List of models to be evaluated.

    train : bool, default=False
        If `True`, the models will be trained. If `False`, it is assumed that
        the models are pre-trained.

    print : bool, default=False
        If `True`, a table of the analysis is printed.
    """

    # Convert a single model to a singleton list
    if not isinstance(models, list):
        models = list(models)

    if train:
        for model in models:
            model.train()

    # Outputs
    name = []
    n_param = []
    jsd = []  # Jensen-Shannon divergence
    kld = []  # Kullback-Leibler divergence
    aic = []  # Akaike information criterion
    bic = []  # bayesian infrmation criterion

    for model in models:
        # Not implemented
        pass


# ===
# jsd
# ===

def _jsd():
    """
    Jensen-Shannon divergence.
    """


