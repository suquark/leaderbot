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
from .util import sigmoid, cross_entropy
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
        A dictionary of data that is provided by
        :func:`leaderbot.data.load`.

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
        Plots scores versus rank.

    match_matrix
        Plot match matrices of win and tie counts of mutual matches.

    Examples
    --------

    .. code-block:: python

        >>> from leaderbot.data import load
        >>> from leaderbot.models import BradleyTerry

        >>> # Create a model
        >>> data = load()
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

    # ===========
    # sample loss
    # ===========

    @staticmethod
    @numba.jit(nopython=True)
    def _sample_loss(
            w: Union[List[float], np.ndarray[np.floating]],
            x: np.ndarray[np.integer],
            y: np.ndarray[np.integer],
            n_agents: int,
            return_jac: bool = True,
            inference_only: bool = False):
        """
        Loss per each sample data (instance).
        """

        # Initialize outputs so numba does not complain
        loss_ = None
        grads = None
        probs = None

        i, j = x.T
        xi, xj = w[i], w[j]
        z = xi - xj

        # Probabilities
        p_win = sigmoid(z)
        p_loss = 1.0 - p_win

        if inference_only:
            p_tie = np.zeros_like(p_loss)
            probs = (p_win, p_loss, p_tie)
            return loss_, grads, probs

        # loss for each sample ij
        win_ij, loss_ij, tie_ij = y.T
        w_ij = win_ij + 0.5 * tie_ij
        l_ij = loss_ij + 0.5 * tie_ij
        loss_ = - cross_entropy(w_ij, p_win) - cross_entropy(l_ij, p_loss)

        if return_jac:
            # grad_z = w_ij * (p_win - 1) + l_ij * p_win
            grad_z = (w_ij + l_ij) * p_win - w_ij

            grad_xi = grad_z
            grad_xj = -grad_xi

            grads = (grad_xi, grad_xj)

        return loss_, grads, probs

    # ====
    # loss
    # ====

    def loss(
            self,
            w: Union[List[float], np.ndarray[np.floating]] = None,
            return_jac: bool = True,
            constraint: bool = True):
        """
        Total loss for all data instances.

        Parameters
        ----------

        w : array_like, default=None
            Parameters. If `None`, the pre-trained parameters are used,
            provided is already trained.

        return_jac : bool, default=True
            if `True`, the Jacobian of loss with respect to the parameters is
            also returned.

        constraint : bool, default=True
            If `True`, the constrain on the parameters is also added to the
            loss.

        Returns
        -------

        loss : float
            Total loss for all data points.

        if return_jac is `True`:

            jac : np.array
                An array of the size of the number of parameters, representing
                the Jacobian of loss.

        Raises
        ------

        RuntimeWarning
            If loss is ``nan``.

        RuntimeError
            If the model is not trained and the input ``w`` is set to `None`.

        See Also
        --------

        train : train model parameters.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 13

            >>> from leaderbot.data import load
            >>> from leaderbot.models import BradleyTerryScaled

            >>> # Create a model
            >>> data = load()
            >>> model = BradleyTerryScaled(data)

            >>> # Generate an array of parameters
            >>> import numpy as np
            >>> w = np.random.randn(model.n_param)

            >>> # Compute loss and its gradient with respect to parameters
            >>> loss, jac = model.loss(w, return_jac=True, constraint=False)
        """

        if w is None:
            if self.param is None:
                raise RuntimeError('train model first.')
            w = self.param

        loss_, grads, _ = self._sample_loss(w, self.x, self.y, self.n_agents,
                                            return_jac=return_jac,
                                            inference_only=False)

        loss_ = loss_.sum() / self._count
        if np.isnan(loss_):
            raise RuntimeWarning("loss is nan")

        if return_jac:
            grad_xi, grad_xj = grads
            i, j = self.x.T
            n = self.x.shape[0]
            ax = np.arange(n)
            jac = np.zeros((n, w.shape[0]))
            jac[ax, i] += grad_xi
            jac[ax, j] += grad_xj

            jac = jac.sum(axis=0) / self._count

        if constraint:
            # constraining score parameters
            # constraint_diff = np.sum(np.exp(w[:n_agents])) - 1
            constraint_diff = np.sum(w[:self.n_agents])
            constraint_loss = constraint_diff ** 2
            loss_ += constraint_loss

            if return_jac:
                # constraining score parameters
                # constraint_jac = 2 * constraint_diff * np.exp(w[:n_agents])
                constraint_jac = 2.0 * constraint_diff
                jac[:self.n_agents] += constraint_jac

        if return_jac:
            return loss_, jac
        else:
            return loss_

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

            >>> from leaderbot.data import load
            >>> from leaderbot.models import BradleyTerry

            >>> # Create a model
            >>> data = load()
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

            >>> from leaderbot.data import load
            >>> from leaderbot.models import BradleyTerry

            >>> # Create a model
            >>> data = load()
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
