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
from ._base_model import BaseModel
from ._math_util import sigmoid, cross_entropy
from ..data import DataType
from typing import List, Union

__all__ = ['RaoKupperScaledR']


# ===================
# Rao Kupper Scaled R
# ===================

class RaoKupperScaledR(BaseModel):
    """
    Paired comparison based on Rao-Kupper model and Thurstonian model with
    diagonal plus rank-1 covariance.

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

    The Rao-Kupper model of paired comparison is based on [1]_.

    References
    ----------

    .. [1] Rao, P. V., Kupper, L. L. (1967). Ties in Paired-Comparison
           Experiments: A Generalization of the Bradley-Terry Model. `Journal
           of the American Statistical Association`, 62(317), 194–204.
           https://doi.org/10.1080/01621459.1967.10482901

    See Also
    --------

    RaoKupper
    RaoKupperScaled
    RaoKupperScaledRIJ
    RaoKupperFactor

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

    fisher
        Observed Fisher information matrix.

    rank
        Return rank of the agents based on their score.

    leaderboard
        Print leaderboard table.

    marginal_outcomes
        Plot marginal probabilities and frequencies of win, loss, and tie.

    map_distance
        Visualize distance between agents using manifold learning projection.

    cluster
        Cluster competitors to performance tiers.

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
        >>> from leaderbot.models import RaoKupperScaledR

        >>> # Create a model
        >>> data = load()
        >>> model = RaoKupperScaledR(data)

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

        # Number of factors for tie (rank of matrix in factor analysis)
        self.n_tie_factors = n_tie_factors

        # Total number of parameters for modeling tie
        self._n_tie_param = max(1, self.n_agents * self.n_tie_factors)

        # Total number of parameters
        self.n_param = 2 * self.n_agents + 1 + self._n_tie_param

        # Indices of parameters
        self._scale_idx = slice(self.n_agents, self.n_agents * 2)
        self._tie_factor_idx = slice(
            1 + self.n_agents * 2,
            1 + self.n_agents * (2 + self.n_tie_factors))

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
                             [(-1.0, 1.0)] + \
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

        Note: in Rao-Kupper model, eta should be no-negative so that p_tie be
        non-negative. To enforce this, we add absolute value and adjust its
        gradient with sign of eta.
        """

        # Initialize outputs so numba does not complain
        loss_ = None
        grads = None
        probs = None
        grad_gi = None
        grad_gj = None

        i, j = x.T
        xi, xj = w[i], w[j]
        ti, tj = w[i + n_agents], w[j + n_agents]
        r = w[2 * n_agents]
        min_eta = 1e-2

        if n_tie_factors == 0:
            eta_scalar = np.maximum(w[-1], min_eta)  # clip eta
            eta = np.full_like(xi, eta_scalar)
        else:
            g = w[2 * n_agents + 1:].reshape(n_agents, n_tie_factors)
            gi = g[i]
            gj = g[j]
            phi_i = basis[i]
            phi_j = basis[j]
            eta = np.sum(gi * phi_j, axis=1) + np.sum(gj * phi_i, axis=1)
            eta[np.abs(eta) < min_eta] = min_eta   # clip eta

        s_r = np.tanh(r)
        scale = 1.0 / np.sqrt(ti ** 2 + tj ** 2 - 2 * s_r * np.abs(ti * tj))
        z = (xi - xj) * scale

        d_win = z - np.abs(eta)
        d_loss = -z - np.abs(eta)

        # Probabilities
        p_win = sigmoid(d_win)
        p_loss = sigmoid(d_loss)

        if inference_only:
            # p_tie = (np.exp(np.abs(eta) * 2) - 1) * p_win * p_loss
            p_tie = 1.0 - p_win - p_loss
            probs = (p_win, p_loss, p_tie)
            return loss_, grads, probs

        # loss for each sample ij
        win_ij, loss_ij, tie_ij = y.T
        loss_ = - cross_entropy((win_ij + tie_ij), p_win) \
                - cross_entropy((loss_ij + tie_ij), p_loss) \
                - cross_entropy(tie_ij, (np.exp(2.0 * np.abs(eta)) - 1.0))

        if return_jac:
            grad_dwin = (win_ij + tie_ij) * (1.0 - p_win)
            grad_dloss = (loss_ij + tie_ij) * (1.0 - p_loss)
            grad_z = -grad_dwin + grad_dloss

            grad_xi = grad_z * scale
            grad_xj = -grad_xi

            grad_scale = -0.5 * grad_z * z * scale ** 2
            grad_ti = grad_scale * 2 * ti
            grad_tj = grad_scale * 2 * tj
            grad_scale = -0.5 * grad_z * z * scale ** 2
            grad_ti = grad_scale * 2 * (ti - s_r * np.sign(ti) * np.abs(tj))
            grad_tj = grad_scale * 2 * (tj - s_r * np.sign(tj) * np.abs(ti))
            grad_r = -grad_scale * 2 * np.abs(ti * tj) * (1.0 - s_r ** 2)

            grad_eta = (grad_dwin + grad_dloss - (2.0 * tie_ij) /
                        (1.0 - np.exp(-2.0 * np.abs(eta)))) * np.sign(eta)

            # At eta=0, the gradient w.r.t eta is "-np.inf * np.sign(eta)".
            # However, for stability of the optimizer, we set its gradient to
            # zero in the clipped interval where it is assumed to be constant.
            grad_eta[np.abs(eta) < min_eta] = 0.0

            if n_tie_factors > 0:
                grad_gi = grad_eta[:, None] * phi_j
                grad_gj = grad_eta[:, None] * phi_i

            grads = (grad_xi, grad_xj, grad_ti, grad_tj, grad_r, grad_eta,
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
        fisher : Observed Fisher information matrix.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 13

            >>> from leaderbot.data import load
            >>> from leaderbot.models import RaoKupperScaledR

            >>> # Create a model
            >>> data = load()
            >>> model = RaoKupperScaledR(data)

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
            grad_xi, grad_xj, grad_ti, grad_tj, grad_r, grad_eta, grad_gi, \
                grad_gj = grads
            i, j = self.x.T
            n_samples = self.x.shape[0]
            ax = np.arange(n_samples)
            jac = np.zeros((n_samples, w.shape[0]))
            jac[ax, i] += grad_xi
            jac[ax, j] += grad_xj
            jac[ax, i + self.n_agents] += grad_ti
            jac[ax, j + self.n_agents] += grad_tj
            jac[ax, 2 * self.n_agents] += grad_r

            if self.n_tie_factors == 0:
                jac[ax, -1] += grad_eta
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
