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
from .util import double_sigmoid, cross_entropy
from ..data import DataType
from typing import List, Union

__all__ = ['DavidsonGen']


# ============
# Davidson Gen
# ============

class DavidsonGen(BaseModel):
    """
    Paired comparison based on generalized Davidson model.

    Parameters
    ----------

    data : dict
        A dictionary of data that is provided by
        :func:`leaderbot.data.load`.

    Notes
    -----

    The Davidson model of paired comparison is based on [1]_.

    References
    ----------

    .. [1] Davidson, R. R. (1970). On Extending the Bradley-Terry Model to
           Accommodate Ties in Paired Comparison Experiments. `Journal of the
           American Statistical Association`, 65(329), 317–328.
           https://doi.org/10.2307/2283595

    See Also
    --------

    Davidson
    DavidsonScaled
    DavidsonScaledR
    DavidsonScaledRIJ

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
        >>> from leaderbot.models import DavidsonGen

        >>> # Create a model
        >>> data = load()
        >>> model = DavidsonGen(data)

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
            data: DataType,
            n_factors: int = 1):
        """
        Constructor.
        """

        super().__init__(data)
        self.n_factors = n_factors

        self.n_param = self.n_agents + self.n_agents * self.n_factors

        # TEST
        A = np.random.randn(self.n_agents, self.n_factors)
        self.basis = np.linalg.qr(A)[0]
        self.basis[:, 0] = 1.0

        # Approximate bound for parameters (only needed for shgo optimization
        # method). Note that these bounds are not enforced, rather, only used
        # for seeding multi-initial points in global optimization methods.
        self._param_bounds = \
            [(-1.0, 1.0) for _ in range(self.n_agents)] + \
            [(-1.0, 1.0) for _ in range(self.n_agents)] * self.n_factors

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

        return init_param

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
            n_factors: int,
            basis: np.ndarray[np.floating],
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
        # mu_i, mu_j = w[n_agents + i], w[n_agents + j]
        # mu = mu_i + mu_j

        m = w[n_agents:].reshape(n_agents, n_factors)
        ri = m[i]
        rj = m[j]
        bi = basis[i]
        bj = basis[j]
        mu = np.sum(ri * bj, axis=1) + np.sum(bi * rj, axis=1)

        z = xi - xj
        u = 0.5 * z - mu
        v = -0.5 * z - mu

        # Probabilities
        p_win = double_sigmoid(z, u)
        p_loss = double_sigmoid(-z, v)
        p_tie = double_sigmoid(-u, -v)

        if inference_only:
            probs = (p_win, p_loss, p_tie)
            return loss_, grads, probs

        # loss for each sample ij
        win_ij, loss_ij, tie_ij = y.T
        loss_ = - cross_entropy(win_ij, p_win) \
                - cross_entropy(loss_ij, p_loss) \
                - cross_entropy(tie_ij, p_tie)

        if return_jac:
            grad_p_win_z = -(np.exp(-z) + 0.5 * np.exp(-u)) * p_win
            grad_p_loss_z = (np.exp(z) + 0.5 * np.exp(-v)) * p_loss
            grad_p_tie_z = 0.5 * (np.exp(u) - np.exp(v)) * p_tie
            grad_z = \
                win_ij * grad_p_win_z + \
                loss_ij * grad_p_loss_z + \
                tie_ij * grad_p_tie_z

            grad_p_win_u = np.exp(-u) * p_win
            grad_p_loss_u = np.exp(-v) * p_loss
            grad_p_tie_u = -(np.exp(u) + np.exp(v)) * p_tie
            grad_mu = \
                win_ij * grad_p_win_u + \
                loss_ij * grad_p_loss_u + \
                tie_ij * grad_p_tie_u

            grad_xi = grad_z
            grad_xj = -grad_z

            # grad_mu_i = grad_mu
            # grad_mu_j = grad_mu
            # grad_mu_i = grad_mu * bj
            # grad_mu_j = grad_mu * bi

            grad_mu_i = grad_mu[:, None] * bj
            grad_mu_j = grad_mu[:, None] * bi

            grads = (grad_xi, grad_xj, grad_mu_i, grad_mu_j)

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
            >>> from leaderbot.models import DavidsonGen

            >>> # Create a model
            >>> data = load()
            >>> model = DavidsonGen(data)

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
                                            self.n_factors, self.basis,
                                            return_jac=return_jac,
                                            inference_only=False)

        loss_ = loss_.sum() / self._count
        if np.isnan(loss_):
            raise RuntimeWarning("loss is nan")

        if return_jac:
            grad_xi, grad_xj, grad_mu_i, grad_mu_j = grads
            i, j = self.x.T
            n = self.x.shape[0]
            ax = np.arange(n)
            jac = np.zeros((n, w.shape[0]))
            jac[ax, i] += grad_xi
            jac[ax, j] += grad_xj

            # TEST
            # print('-----------')
            # print()
            # print(jac.shape)
            # print(grad_mu_i.shape)

            # jac[ax, self.n_agents + i] += grad_mu_i
            # jac[ax, self.n_agents + j] += grad_mu_j

            dm = np.zeros((n, self.n_agents, self.n_factors))
            dm[ax, i] += grad_mu_i
            dm[ax, j] += grad_mu_j
            # jac[ax, self.n_agents:self.n_agents * self.n_factors] = \
            jac[ax, self.n_agents:] = \
                dm.reshape(n, self.n_agents * self.n_factors)

            jac = jac.sum(axis=0) / self._count

        if constraint:
            # constraining score parameters
            # constraint_diff = np.sum(np.exp(w[:n_agents])) - 1
            constraint_diff = np.sum(w[:self.n_agents])
            constraint_loss = constraint_diff ** 2
            loss_ += constraint_loss

            if return_jac:
                # constraint_jac = 2 * constraint_diff * np.exp(w[:n_agents])
                constraint_jac = 2.0 * constraint_diff
                jac[:self.n_agents] += constraint_jac

        if return_jac:
            return loss_, jac
        else:
            return loss_

    # =====
    # infer
    # =====

    def infer(
            self,
            x: Union[List[int], np.ndarray[np.integer]] = None):
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
        _, _, probs = self._sample_loss(self.param, x, self.y, self.n_agents,
                                        self.n_factors, self.basis,
                                        return_jac=False, inference_only=True)

        return np.column_stack(probs)
