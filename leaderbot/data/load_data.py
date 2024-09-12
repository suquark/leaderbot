# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import json
import os
import numpy as np
from typing import TypedDict, List, Union

__all__ = ['load_data', 'DataType']


# =========
# Data Type
# =========

class DataType(TypedDict):
    """
    Standard data type for input data.
    """

    X: Union[List[List[int]], np.ndarray[np.integer]]
    Y: Union[List[List[int]], np.ndarray[np.integer]]
    models: Union[List[str], np.ndarray[np.str_]]


# ==========
# clean data
# ==========

def _clean_data(
        x: Union[List[List[int]], np.ndarray[np.integer]],
        y: Union[List[List[int]], np.ndarray[np.integer]]):
    """
    Removes data pairs with no matches (where the entire row of self.y is
    zero).
    """

    y_sum = np.sum(y, axis=-1)
    ind_valid = y_sum != 0
    x = x[ind_valid]
    y = y[ind_valid]

    return x, y


# ====================
# check data duplicate
# ====================

def _check_data_duplicacy(
        x: Union[List[List[int]], np.ndarray[np.integer]],
        verbose: bool = False) -> int:
    """
    Checks duplicate matches. Calling this is optional.
    """

    duplicacy_count = 0

    for i in range(x.shape[0]):
        for j in range(i+1, x.shape[0]):
            if (((x[i, 0], x[i, 1]) == (x[j, 0], x[j, 1])) or
                    ((x[i, 0], x[i, 1]) == (x[j, 1], x[j, 0]))):

                duplicacy_count += 1

                if verbose:
                    print(f'{i}, {j}, '
                          f'i-th: ({x[i, 0]}, {x[i, 1]}), '
                          f'j-th: ({x[j, 0]}, {x[j, 1]})')

    return duplicacy_count


# =========
# load data
# =========

def load_data(
        clean: bool = True,
        check_duplicacy: bool = False):
    """
    Load the latest chatbot arena data.

    Parameters
    ----------

    clean : bool, default=True
        If `True`, the pairs with zero win, loss, and tie counts are deleted
        from the list of data.

    check_duplicacy : bool, default=False
        If `True`, all pairs in the data are checked for duplicacy.

        .. note::

            Performing this check may consume time.

    Returns
    -------

    data : DataType
        A dictionary containing the following key/values:

        * ``'X'``:
            A list of tuple of two indices ``(i, j)`` representing a match
            between a pair of agents with the indices ``i`` and ``j``.
        * ``'Y'``:
            A list of tuples of three integers ``(n_win, n_loss, n_ties)``
            representing the frequencies of win, loss, and ties between agents
            ``i`` and ``j`` given by the corresponding tuple in ``X``.
        * ``'models'``: a list of thre name of agents in the match.

    Raises
    ------

    if ``check_duplicacy`` is ` True`:

        Warning
            If duplicacy were found in the data.
    """

    filename = 'chatbotarena_20240814.json'
    data_dir = os.path.dirname(__file__)

    with open(os.path.join(data_dir, filename)) as f:
        data = json.load(f)

    if clean:
        x = np.array(data['X'])
        y = np.array(data['Y'])

        x, y = _clean_data(x, y)

        data['X'] = x
        data['Y'] = y

    if check_duplicacy:
        x = np.array(data['X'])
        duplicacy_count = _check_data_duplicacy(x)
        if duplicacy_count:
            raise Warning('Found %d delicacies in data!' % duplicacy_count)

    return data
