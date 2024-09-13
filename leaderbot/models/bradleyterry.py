# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# imports
# =======

import numba
import numpy as np
from .base_model import BaseModel
from ..data import DataType
from typing import List, Union

__all__ = ['BradleyTerry']


# =============
# Bradley Terry
# =============

class BradleyTerry(BaseModel):
    """
    Paired comparison based on Bradley-Terry model.

    Parameters
    ----------

    data : dict
        A dictionary of data that is provided by :func:`leaderbot.load_data`.

    Notes
    -----

    The Bradley-Terry model of paired comparison is based on [1]_. This
    model does not include ties in the data.

    References
    ----------

    .. [1] Bradley, R., Terry, M. (1952). Rank Analysis of Incomplete Block
           Designs: I. The Method of Paired Comparisons. `Biometrika`, 39
           (3/4), 324-345. https://doi.org/10.2307/2334029

    See Also
    --------

    BradleyTerryScaled
    BradleyTerryScaledR
    BradleyTerryScaledRIJ

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

    n_param : int
        Number of parameters

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

    Examples
    --------

    .. code-block:: python

        >>> from leaderbot.data import load_data
        >>> from leaderbot.models import BradleyTerry

        >>> # Create a model
        >>> data = load_data()
        >>> model = BradleyTerry(data)

        >>> # Train the model
        >>> model.train()

        >>> # Make inference
        >>> p_win, p_loss, p_tie = model.infer()
    """

    # ====
    # init
    # ====

    def __init__(
            self,
            data: DataType):
        """
        Constructor.
        """

        super().__init__(data)
        self.n_param = self.n_agents

    # ================
    # iterative solver
    # ================

    @staticmethod
    @numba.jit(nopython=True)
    def _iterative_solver(
            n_agents: int,
            x: np.ndarray[np.integer],
            y: np.ndarray[np.integer],
            max_iter: int = 100,
            tol: float = 1e-7):
        """
        Iterative solver for the Bradley-Terry model.

        Parameters
        ----------

        n_agents : in
            Number of models.

        x : np.ndarray[np.integer]
            Array of shape (n_pairs, 3) where each row represents a pair of
            models.

        y : np.ndarray[np.integer]
            Array of shape (n_pairs, 3) where each row represents the outcome
            of a pair of models.

        max_iter : int, default=100
            Maximum number of iterations.

        tol : float, default=1e-7
            Tolerance for convergence.

        Returns
        -------

        param : np.array
            Array of parameters

        References
        ----------

        * Newman, M. E. J. (2023), Efficient Computation of Rankings from
          Pairwise Comparisons. Journal of Machine Learning Research. 24-238,
          pp 1--25m. http://jmlr.org/papers/v24/22-1086.html
        """

        w = np.zeros(n_agents)
        i, j = x.T
        win_count, loss_count, tie_count = y.T
        w_ij = win_count + 0.5 * tie_count
        w_ji = loss_count + 0.5 * tie_count

        for _ in range(max_iter):
            last_w = w.copy()
            p = np.exp(w)
            a = np.zeros_like(p)
            b = np.zeros_like(p)
            pi, pj = p[i], p[j]

            s = pi + pj
            p_win_ij = pi / s
            a_i = w_ij * (1 - p_win_ij)
            b_i = w_ji / s
            a_j = w_ji * p_win_ij
            b_j = w_ij / s

            for k in range(x.shape[0]):
                a[i[k]] += a_i[k]
                b[i[k]] += b_i[k]
                a[j[k]] += a_j[k]
                b[j[k]] += b_j[k]

            # Use MAP prior to avoid instability with sparse data
            a += 1 / (1 + p)
            b += 1 / (1 + p)

            w = np.log(a / b)
            w -= w.mean()
            if np.allclose(w, last_w, atol=tol):
                return w

        return w

    # =====
    # train
    # =====

    def train(
            self,
            init_param: Union[List[float], np.ndarray[np.floating]] = None,
            method: str = None,
            max_iter: int = 100,
            tol: float = 1e-8):
        """
        Tune model parameters with maximum likelihood estimation method.

        .. note::

            This function overwrites the base class's method.

        Parameters
        ----------

        init_param : array_like, default=None
            Initial parameters.

            .. note::

                This argument is not used, and is only kept for consistency
                with the corresponding base method.

        method : str, default=None
            Method of optimization.

            .. note::

                This argument is not used, and is only kept for consistency
                with the corresponding base method.

        max_iter : int, default=100
            Maximum number of iterations.


        tol : float, default=1e-8
            Tolerance of optimization.

        See Also
        --------

        predict : predict probabilities based on given parameters.

        Notes
        -----

        The trained parameters are available as ``param`` attribute.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 9

            >>> from leaderbot.data import load_data
            >>> from leaderbot.models import BradleyTerry

            >>> # Create a model
            >>> data = load_data()
            >>> model = BradleyTerry(data)

            >>> # Train the model
            >>> model.train()

            >>> # Make inference
            >>> prob = model.infer()
        """

        self.param = self._iterative_solver(self.n_agents, self.x, self.y,
                                            max_iter=max_iter, tol=tol)

    # =====
    # infer
    # =====

    def infer(
            self,
            x: Union[List[int], np.ndarray[np.integer]] = None):
        """
        Make inference on probabilities of outcome of win, loss, or tie.

        .. note::

            This function overwrites the base class's method.

        Parameters
        ----------

        x : np.ndarray
            A 2D array of integers with the shape ``(n_pairs, 2)`` where each
            row consists of indices ``[i, j]`` representing a match between a
            pair of agents with the indices ``i`` and ``j``. If `None`, the
            ``X`` variable from the input data is used.

        Returns
        -------

        prob : np.array[np.float]
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

            >>> from leaderbot.data import load_data
            >>> from leaderbot.models import BradleyTerry

            >>> # Create a model
            >>> data = load_data()
            >>> model = BradleyTerry(data)

            >>> # Train the model
            >>> model.train()

            >>> # Make prediction
            >>> pred = model.predict()
        """

        if self.param is None:
            raise RuntimeError('train model first.')

        if x is None:
            x = self.x

        i, j = x.T
        z = self.param[i] - self.param[j]

        # Probabilities
        p_win = 1 / (1 + np.exp(-z))  # sigmoid
        p_loss = 1 - p_win
        p_tie = 1 - p_win - p_loss

        return np.column_stack([p_win, p_loss, p_tie])
