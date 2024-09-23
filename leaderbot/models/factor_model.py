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

    n_cov_factors : int
        Number of factors for matrix factorization.

    Methods
    -------

    loss
        Loss function of the model.

    train
        Train model parameters.

    infer
        Makes inference of match probabilities.

    predict
        Predict the output of a match between agents.

    rank
        Return rank of the agents based on their score.

    leaderboard
        Print leaderboard table and plot prediction for agents.

    visualize
        Visualize correlation and score of the agents.

    plot_scores
        Plots scores versus rank

    match_matrix
        Plot match matrices of win and tie counts of mutual matches.
    """

    # ====
    # init
    # ====

    def __init__(self, data: DataType, n_cov_factors: int):
        """
        Constructor.
        """
        super().__init__(data)

        self.n_cov_factors = n_cov_factors

        # Total number of parameters
        self.n_param = (2 + n_cov_factors) * self.n_agents

        # Indices of parameters
        self._scale_idx = slice(self.n_agents, self.n_agents * 2)
        self._cov_factor_idx = slice(self.n_agents * 2,
                                     self.n_agents * (2 + self.n_cov_factors))

        # Containing which features
        self._has_scale = True
        self._has_cov_factor = True

        # Approximate bound for parameters (only needed for shgo optimization
        # method). Note that these bounds are not enforced, rather, only used
        # for seeding multi-initial points in global optimization methods.
        self._param_bounds = [(-1.0, 1.0) for _ in range(self.n_agents)] + \
                             [(0.01, 1.0) for _ in range(self.n_agents)] + \
                             [(-1.0, 1.0) for _ in range(
                                 self.n_agents * n_cov_factors)]

    # ================
    # initialize param
    # ================

    def _initialize_param(self):
        """
        Initialize parameters.
        """

        # Initial parameters
        init_param = np.zeros(self.n_param)
        init_param[self._score_idx] = self._initialize_scores()

        init_param[self._scale_idx] = \
            np.full(self.n_agents, np.sqrt(1.0 / (self.n_agents - 1.0)))

        # Using a deterministic matrix instead of random. Here we use DCT basis
        # functions, not it could also be anything else and it does not have to
        # be orthogonal (as DCT is.). However, its columns has to be centered.
        if self.n_cov_factors > 0:
            M = self._generate_basis(self.n_agents, self.n_cov_factors)
            M = M - np.tile(np.mean(M, axis=0), (self.n_agents, 1))
            init_param[self._cov_factor_idx] = M.ravel()

        return init_param

    # ==============
    # get covariance
    # ==============

    def _get_covariance(
            self,
            param: np.ndarray = None,
            centered: bool = False):
        """
        Covariance matrix.

        Parameters
        ----------

        param : np.ndarray, default=None
            Model parameters. If `None`, the trained model parameters are used.

        centered : bool, default = False
            If `True`, the doubly-centered operator is applied to the
            covariance matrix, making it doubly-stochastic Gramian matrix
            with null space of dim 1 and zero sum rows and columns.
        """

        if param is None:
            if self.param is None:
                raise RuntimeError('train model first.')
            param = self.param

        if param.size < 2 * self.n_agents:
            # The model does not have Thurstonian covariance.
            return None

        if not self._has_scale:
            raise RuntimeError('model does have scale parameters.')

        # Diagonals of covariance matrix
        t = param[self._scale_idx]

        # Constructing covariance
        D = np.diag(t**2)
        if self.n_cov_factors > 0:
            M = param[self._cov_factor_idx].reshape(
                self.n_agents, self.n_cov_factors)
            S = D + M @ M.T
        else:
            S = D

        # Centering covariance
        if centered:
            Id = np.eye(self.n_agents, dtype=float)
            J = np.ones((self.n_agents, self.n_agents), dtype=float)
            C = Id - J / self.n_agents  # centering matrix
            S = C @ S @ C   # centered cov

        return S

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
        if hasattr(self, "basis") and hasattr(self, "n_tie_factors"):
            _, _, probs = self._sample_loss(self.param,
                                            x,
                                            self.y,
                                            self.n_agents,
                                            self.n_cov_factors,
                                            self.n_tie_factors,
                                            self.basis,
                                            return_jac=False,
                                            inference_only=True)
        else:
            _, _, probs = self._sample_loss(self.param,
                                            x,
                                            self.y,
                                            self.n_agents,
                                            self.n_cov_factors,
                                            return_jac=False,
                                            inference_only=True)

        return np.column_stack(probs)
