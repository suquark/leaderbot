# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import requests
import json
import os
import numpy as np
from typing import List, Union
from ._util import is_url_or_local

__all__ = ['load_data']


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


# ================
# whitelist filter
# ================

def _whitelist_filter(
        data,
        whitelist):
    """
    Select a subset of data using a whitelist.
    """

    if is_url_or_local(whitelist) == 'local':
        with open(whitelist) as f:
            whitelist = set(json.load(f))

    elif not isinstance(whitelist, list):
        raise ValueError('"whitelist" is neither list nor a file.')

    selected_agents = []
    mapping = {}
    for i, agent in enumerate(data["models"]):
        if agent in whitelist:
            mapping[i] = len(selected_agents)
            selected_agents.append(agent)
        else:
            mapping[i] = None

    new_X = []
    new_Y = []

    for x, y in zip(data["X"], data["Y"], strict=True):
        if mapping[x[0]] is None or mapping[x[1]] is None:
            continue
        new_X.append([mapping[x[0]], mapping[x[1]]])
        new_Y.append(y)

    selected_data = {
        "models": selected_agents,
        "X": new_X,
        "Y": new_Y,
    }

    return selected_data


# =========
# load data
# =========

def load_data(
        filename: str = None,
        whitelist: Union[List[str], str] = None,
        clean: bool = True,
        check_duplicacy: bool = False):
    """
    Load data from JSON file or URL.

    Parameters
    ----------

    filename : str, default=None
        A ``.json`` filename of the data. The filename can be the location on
        the local machine or a URL of a file on a remote server accessible via
        the HTTP or HTTPS protocol. If `None`, a default file that is shipped
        with the package will be used.

    whitelist : list or str, default=None
        A list of agent names to be selected from the full set of agent names
        in the data. Alternatively, a ``.json`` filename can be provided, which
        should contain a list of names to be used.

    clean : bool, default=True
        If `True`, the pairs with zero win, loss, and tie counts are deleted
        from the list of data.

    check_duplicacy : bool, default=False
        If `True`, all pairs in the data are checked for duplicacy.

        .. note::

            Performing this check is time consuming.

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
        * ``'models'``: a list of the name of agents in the match.

    Raises
    ------

    If ``check_duplicacy`` is ` True`:

        Warning
            If duplicacy were found in the data.

    See Also
    --------

    leaderbot.data.convert

    Examples
    --------

    .. code-block:: python

        >>> from leaderbot.data import load_data

        >>> # Load default data provided by the package
        >>> data = load_data()

        >>> # Load from a file
        >>> filename = '/scratch/user/my-data.json'
        >>> data = load_data(filename)

        >>> # Load default data, but only select a custom whitelist of names
        >>> whitelist = [
        ...     "chatgpt-4o-latest",
        ...     "gemini-1.5-pro-exp-0801",
        ...     "gpt-4o-2024-05-13",
        ...     "gpt-4o-mini-2024-07-18",
        ... ]
        >>> data = load_data(whitelist=whitelist)

        >>> # Use a sample whitelist provided by the package
        >>> from leaderbot.data import sample_whitelist
        >>> data = load_data(whitelist=sample_whitelist)
    """

    if filename is None:
        base_filename = 'chatbotarena_20240814.json'
        data_dir = os.path.dirname(__file__)
        filename = os.path.join(data_dir, base_filename)

    status = is_url_or_local(filename)

    if status == 'url':
        # Read a remote file
        response = requests.get(filename)
        response.raise_for_status()
        data = response.json()

    elif status == 'local':
        with open(filename) as f:
            data = json.load(f)

    else:
        raise ValueError(f'{filename} is neither a URL nor a local file.')

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
            raise Warning('Found %d duplicacy in data!' % duplicacy_count)

    return data
