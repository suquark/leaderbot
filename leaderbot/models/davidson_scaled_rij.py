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

__all__ = ['DavidsonScaledRIJ']


# ===================
# Davidson Scaled RIJ
# ===================

class DavidsonScaledRIJ(BaseModel):
    """
    Paired comparison based on Davidson model and Thurstonian model with
    full-rank covariance.

    Parameters
    ----------

    data : dict
        A dictionary of data that is provided by
        :func:`leaderbot.data.load`.

    n_tie_factor : int, default=1
        Number of factors to be used in tie parameters. If ``0``, no factor is
        used and the model falls back to the original Davidson formulation.

    Notes
    -----

    The Davidson model of paired comparison is based on [1]_.

    .. note::

        When training this model, for faster convergence, set
        ``method='L-BFGS-B'`` method in :func:`train` function instead of the
        default ``'BFGS'`` method.

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

    n_param : int
        Number of parameters

    n_tie_factor : int
        Number of factors used in tie parameters.

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

    scores
        Get scores

    plot_scores
        Plots scores versus rank.

    match_matrix
        Plot match matrices of win and tie counts of mutual matches.

    Examples
    --------

    .. code-block:: python

        >>> from leaderbot.data import load
        >>> from leaderbot.models import DavidsonScaledRIJ

        >>> # Create a model
        >>> data = load()
        >>> model = DavidsonScaledRIJ(data)

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
            n_tie_factors: int = 1):
        """
        Constructor.
        """

        super().__init__(data)

        # Number of rij parameters
        self._n_pairs = self.n_agents * (self.n_agents - 1) // 2

        # Number of factors for tie (rank of matrix in factor analysis)
        self.n_tie_factors = n_tie_factors

        # Total number of parameters for modeling tie
        self._n_tie_param = max(1, self.n_agents * self.n_tie_factors)

        # Total number of parameters
        self.n_param = 2 * self.n_agents + self._n_pairs + self._n_tie_param

        # Indices of parameters
        self._scale_idx = slice(self.n_agents, self.n_agents * 2)
        self._tie_factor_idx = slice(
            self._n_pairs + self.n_agents * 2,
            self._n_pairs + self.n_agents * (2 + self.n_tie_factors))

        # Basis functions for tie factor model
        self.basis = self._generate_basis(self.n_agents, self.n_tie_factors)

        # Containing which features
        self._has_scale = True
        self._has_tie_factor = True

        # Approximate bound for parameters (only needed for shgo optimization
        # method). Note that these bounds are not enforced, rather, only used
        # for seeding multi-initial points in global optimization methods.
        self._param_bounds = [(-1.0, 1.0) for _ in range(self.n_agents)] + \
                             [(0.01, 1.0) for _ in range(self.n_agents)] + \
                             [(-1.0, 1.0) for _ in range(self._n_pairs)] + \
                             [(-1.0, 1.0) for _ in range(self._n_tie_param)]

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
            n_tie_factors: int,
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
        grad_gi = None
        grad_gj = None

        i, j = x.T
        k = j * (j - 1) // 2 + i  # pair index

        xi, xj = w[i], w[j]
        ti, tj = w[i + n_agents], w[j + n_agents]
        rij = w[k + n_agents * 2]

        if n_tie_factors == 0:
            mu = np.full_like(xi, w[-1])
        else:
            n_pairs = n_agents * (n_agents - 1) // 2
            g = w[2 * n_agents + n_pairs:].reshape(n_agents, n_tie_factors)
            gi = g[i]
            gj = g[j]
            phi_i = basis[i]
            phi_j = basis[j]
            mu = np.sum(gi * phi_j, axis=1) + np.sum(gj * phi_i, axis=1)

        s_rij = np.tanh(rij)
        scale = 1.0 / np.sqrt(ti ** 2 + tj ** 2 - 2 * s_rij * np.abs(ti * tj))
        z = (xi - xj) * scale
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

            grad_xi = grad_z * scale
            grad_xj = -grad_xi

            grad_scale = -0.5 * grad_z * z * scale ** 2
            grad_ti = grad_scale * 2 * (ti - s_rij * np.sign(ti) * np.abs(tj))
            grad_tj = grad_scale * 2 * (tj - s_rij * np.sign(tj) * np.abs(ti))
            grad_rij = -grad_scale * 2 * np.abs(ti * tj) * (1.0 - s_rij ** 2)

            if n_tie_factors > 0:
                grad_gi = grad_mu[:, None] * phi_j
                grad_gj = grad_mu[:, None] * phi_i

            grads = (grad_xi, grad_xj, grad_ti, grad_tj, grad_rij, grad_mu,
                     grad_gi, grad_gj)

        return loss_, grads, probs

    # ====
    # loss
    # ====

    def loss(
            self,
            w: Union[List[float], np.ndarray[np.floating]] = None,
            return_jac: bool = False,
            constraint: bool = False):
        """
        Total loss for all data instances.

        Parameters
        ----------

        w : array_like, default=None
            Parameters. If `None`, the pre-trained parameters are used,
            provided is already trained.

        return_jac : bool, default=False
            if `True`, the Jacobian of loss with respect to the parameters is
            also returned.

        constraint : bool, default=False
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
            >>> from leaderbot.models import DavidsonScaledRIJ

            >>> # Create a model
            >>> data = load()
            >>> model = DavidsonScaledRIJ(data)

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

        loss_, grads, _ = self._sample_loss(w,
                                            self.x,
                                            self.y,
                                            self.n_agents,
                                            self.n_tie_factors,
                                            self.basis,
                                            return_jac=return_jac,
                                            inference_only=False)

        loss_ = loss_.sum() / self._count
        if np.isnan(loss_):
            raise RuntimeWarning("loss is nan")

        if return_jac:
            grad_xi, grad_xj, grad_ti, grad_tj, grad_rij, grad_mu, grad_gi, \
                grad_gj = grads
            i, j = self.x.T
            k = j * (j - 1) // 2 + i  # pair index
            n_samples = self.x.shape[0]
            ax = np.arange(n_samples)
            jac = np.zeros((n_samples, w.shape[0]))
            jac[ax, i] += grad_xi
            jac[ax, j] += grad_xj
            jac[ax, i + self.n_agents] += grad_ti
            jac[ax, j + self.n_agents] += grad_tj
            jac[ax, k + self.n_agents * 2] += grad_rij

            if self.n_tie_factors == 0:
                jac[ax, -1] += grad_mu
            else:
                dg = np.zeros((n_samples, self.n_agents, self.n_tie_factors))
                dg[ax, i] += grad_gi
                dg[ax, j] += grad_gj
                jac[ax, self._tie_factor_idx] = \
                    dg.reshape(n_samples, self.n_agents * self.n_tie_factors)

            jac = jac.sum(axis=0) / self._count

        if constraint:
            # constraining score parameters
            constraint_diff = np.sum(w[self._score_idx])
            constraint_loss = constraint_diff ** 2
            loss_ += constraint_loss

            # Constraining scale parameters
            constraint_scale = np.sum(w[self._scale_idx]**2) - 1.0
            constraint_scale_loss = constraint_scale**2
            loss_ += constraint_scale_loss

            if return_jac:
                # constraining score parameters
                constraint_jac = 2.0 * constraint_diff
                jac[self._score_idx] += constraint_jac

                # Constraining scale parameters
                constraint_scale_jac = \
                    4.0 * constraint_scale * w[self._scale_idx]
                jac[self._scale_idx] += constraint_scale_jac

        if return_jac:
            return loss_, jac
        else:
            return loss_
