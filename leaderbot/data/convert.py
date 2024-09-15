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
import pandas as pd
import tqdm

__all__ = ['convert']


# =======
# convert
# =======

def convert(
        input_file: str,
        output_file: str):
    """
    Convert raw data from Chatbot Arena into a processed JSON format to be
    compatible as input to :func:`leaderbot.data.load`.

    Parameters
    ----------

    input_file : str
        A ``.json`` filename of the raw data. The filename can be the location
        on the local machine or a URL of a file on a remote server accessible
        via the HTTP or HTTPS protocol.

    output_file : str
        The output ``.json`` file to write the converted data.

    Notes
    -----

    This function converts the raw JSON file to another JSON file containing
    the following key and values:

    * ``'X'``:
        A list of tuple of two indices ``(i, j)`` representing a match
        between a pair of agents with the indices ``i`` and ``j``.
    * ``'Y'``:
        A list of tuples of three integers ``(n_win, n_loss, n_ties)``
        representing the frequencies of win, loss, and ties between agents
        ``i`` and ``j`` given by the corresponding tuple in ``X``.
    * ``'models'``: a list of the name of agents in the match.

    See Also
    --------

    leaderbot.data.load

    Examples
    --------

    .. code-block:: python
        :emphasize-lines: 9

        >>> from leaderbot.data import convert, load

        >>> # Input data
        >>> input_file = 'https://storage.googleapis.com/arena_external_' + \\
        ...              'data/public/clean_battle_20240814_public.json'

        >>> # Convert data
        >>> output_file = 'converted_data.json'
        >>> convert(input_file, output_file)

        >>> # Load the converted data
        >>> data = load(output_file)
    """

    # load the JSON data from the local file
    with open(input_file, 'r') as file:
        battles = pd.read_json(file).sort_values(ascending=True, by=["tstamp"])

    # we use anony battles only for leaderboard
    battles = battles[battles["anony"]]

    # we de-duplicate top 0.1% redundant prompts
    # see https://lmsys.org/blog/2024-05-17-category-hard/#note-enhancing-quali
    # ty-through-de-duplication
    print("Before dedup: ", len(battles))
    battles = battles[
            battles["dedup_tag"].apply(lambda x: x.get("sampled", False))]
    print("After dedup: ", len(battles))

    # get unique model names from "model_a" and "model_b" columns
    combined_series = pd.concat(
            [battles["model_a"], battles["model_b"]]).drop_duplicates()

    # Convert to sorted list
    model_list = sorted(combined_series.tolist())

    data_dict = {}
    reverse_dict = {m: i for i, m in enumerate(model_list)}

    for _, row in tqdm.tqdm(battles.iterrows(), total=len(battles)):
        model_a, model_b = row["model_a"], row["model_b"]
        model_a_id = reverse_dict[model_a]
        model_b_id = reverse_dict[model_b]

        if row["winner"] == "model_a":
            index = 0
        elif row["winner"] == "model_b":
            index = 1
        elif row["winner"] == "tie":
            index = 2
        else:
            # both bad
            index = 3

        if model_a_id < model_b_id:
            pair = (model_a_id, model_b_id)
        else:
            pair = (model_b_id, model_a_id)
            # flip winner
            index = [1, 0, 2, 3][index]
        if pair not in data_dict:
            data_dict[pair] = [0, 0, 0, 0]

        data_dict[pair][index] += 1

    data = list(data_dict.items())
    data.sort(key=lambda x: x[0])

    dataset = {
        "models": model_list,
        "X": [x[0] for x in data],
        "Y": [x[1][:3] for x in data]
    }

    with open(output_file, "w") as f:
        json.dump(dataset, f)
