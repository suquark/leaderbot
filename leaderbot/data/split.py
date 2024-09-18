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
from ._util import DataType

__all__ = ['split']


# =====
# split
# =====

def split(
        data: DataType,
        test_ratio: float = 0.2,
        seed: int = 0):
    """
    Split data to training and test data.

    Parameters
    ----------

    data : DataType
        Input dataset.

    test_ratio : float, default=0.2
        The ratio of the number of samples from test data with respect to the
        input dataset. The ration should be a number between zero and one.

    seed : int, default=0
        Random state to initialize random generation. If `None`, no seed will
        be set.

    Returns
    -------

    training_data : DataType
        Training data with the ratio of  ``1-test_ratio`` number of samples.

    test_data : DataType
        Test data with the ratio of  ``test_ratio`` number of samples.

    See Also
    --------

    leaderbot.data.load

    Examples
    --------

    .. code-block:: python
        :emphasize-lines: 7

        >>> import leaderbot as lb

        >>> # Load default dataset
        >>> data = lb.data.load()

        >>> # Split data to training and test data
        >>> training_data, test_data = lb.data.split(data)
    """

    if seed is not None:
        np.random.seed(seed)

    # Extract data
    x = data['X']
    y = data['Y']
    models = data['models']

    # Shuffle indices
    n = x.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)

    # Split point
    split_idx = int(n * (1.0 - test_ratio))

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    x_train = x[train_indices]
    y_train = y[train_indices]

    x_test = x[test_indices]
    y_test = y[test_indices]

    training_data = {
        'X': x_train,
        'Y': y_train,
        'models': models
    }

    test_data = {
        'X': x_test,
        'Y': y_test,
        'models': models
    }

    return training_data, test_data
