# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.

# =======
# imports
# =======

import numpy as np
from ..data import DataType
from typing import List, Union

from .base_model import BaseModel

__all__ = ['FactorModel']

# ==========
# Base Model
# ==========


class FactorModel(BaseModel):
    """
    Base class for all factor models.

    .. note::

        This class should not be instantiated.

    See Also
    --------

    BradleyTerryFactor
    RaoKupperFactor
    DavidsonFactor

    Attributes
    ----------

    x : np.ndarray
        A 2D array of integers with the shape ``(n_pairs, 2)`` where each row
        consists of indices ``[i, j]`` representing a match between a pair of
        agents with the indices ``i`` and ``j``.

    y : np.ndarray
        A 2D array of integers with the shape ``(n_pairs, 3)`` where each row
        consists of three counts ``[n_win, n_loss, n_ties]`` representing the
        frequencies of win, loss, and ties between agents ``i`` and ``j`` given
        by the corresponding row of the input array ``x``.

    agents : list
        A list of the length ``n_agents`` representing the name of agents.

    n_agents : int
        Number of agents.

    param : np.array, default=None
        The model parameters. This array is set once the model is trained.

    n_factors : int
        Number of factors for matrix factorization.

    Methods
    -------

    train
        Train model parameters.

    infer
        Makes inference of match probabilities.

    predict
        Predict the output of a match between agents.

    rank
        Print leaderboard table and plot prediction for agents.

    visualize
        Visualize correlation and score of the agents.
    """

    # ====
    # init
    # ====

    def __init__(self, data: DataType, n_factors: int):
        """
        Constructor.
        """
        super().__init__(data)
        self.n_factors = n_factors

        self.n_param = (2 + n_factors) * self.n_agents

        # Approximate bound for parameters (only needed for shgo optimization
        # method). Note that these bounds are not enforced, rather, only used
        # for seeding multi-initial points in global optimization methods.
        self._param_bounds = [(-1.0, 1.0) for _ in range(self.n_agents)] + \
                             [(0.01, 1.0) for _ in range(self.n_agents)] + \
                             [(-1.0, 1.0) for _ in range(self.n_agents * n_factors)]

    # ================
    # initialize param
    # ================

    def _initialize_param(self):
        """
        Initialize parameters.
        """

        # Initial parameters
        init_param = np.zeros(self.n_param)
        init_param[:self.n_agents] = self._initialize_scores()
        init_param[self.n_agents:self.n_agents * 2] = \
            np.full(self.n_agents, np.sqrt(1.0 / self.n_agents))
        init_param[self.n_agents * 2:self.n_agents * (2 + self.n_factors)] = \
            np.random.rand(self.n_agents * self.n_factors)

        return init_param

    # =====
    # infer
    # =====

    def infer(self, x: Union[List[int], np.ndarray[np.integer]] = None):
        """
        Make inference on probabilities of outcome of win, loss, or tie.

        Parameters
        ----------

        x : np.ndarray
            A 2D array of integers with the shape ``(n_pairs, 2)`` where each
            row consists of indices ``[i, j]`` representing a match between a
            pair of agents with the indices ``i`` and ``j``. If `None`, the
            ``X`` variable from the input data is used.

        Returns
        -------

        prob : np.array
            An array of the shape ``(n_pairs, 3)`` where the columns
            represent the win, loss, and tie probabilities for the model `i`
            against model `j` in order that appears in the input `x`.

        Raises
        ------

        RuntimeError
            If the model is not trained before calling this method.

        See Also
        --------

        train : train model parameters.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12

            >>> from leaderbot.data import load
            >>> from leaderbot.models import DavidsonScaled

            >>> # Create a model
            >>> data = load()
            >>> model = DavidsonScaled(data)

            >>> # Train the model
            >>> model.train()

            >>> # Make inference
            >>> prob = model.infer()
        """

        if self.param is None:
            raise RuntimeError('train model first.')

        if x is None:
            x = self.x

        # Call sample loss to only compute probabilities, but not loss itself
        # _, _, probs = self._sample_loss(self.param, x, None, self.n_agents,
        _, _, probs = self._sample_loss(self.param,
                                        x,
                                        self.y,
                                        self.n_agents,
                                        n_factors=self.n_factors,
                                        return_jac=False,
                                        inference_only=True)

        return np.column_stack(probs)
