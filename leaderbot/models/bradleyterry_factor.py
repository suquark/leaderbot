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
from .factor_model import FactorModel
from .util import sigmoid, cross_entropy
from ..data import DataType
from typing import List, Union

__all__ = ['BradleyTerryFactor']


# ===================
# Bradleyterry Factor
# ===================

class BradleyTerryFactor(FactorModel):
    """
    Paired comparison based on Bradley-Terry and Thurstonian model with
    factored covariance.

    Parameters
    ----------

    data : dict
        A dictionary of data that is provided by
        :func:`leaderbot.data.load`.

    n_cov_factors : int, default=3
        Number of factors for matrix factorization.

    Notes
    -----

    The Bradley-Terry model of paired comparison is based on [1]_. This
    model does not include ties in the data.

    .. note::

        When training this model, for faster convergence, set
        ``method='L-BFGS-B'`` method in :func:`train` function instead of the
        default ``'BFGS'`` method.

    References
    ----------

    .. [1] Bradley, R., Terry, M. (1952). Rank Analysis of Incomplete Block
           Designs: I. The Method of Paired Comparisons. `Biometrika`, 39
           (3/4), 324-345. https://doi.org/10.2307/2334029

    See Also
    --------

    BradleyTerry
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

    fisher
        Observed Fisher information matrix.

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
        >>> from leaderbot.models import BradleyTerryFactor

        >>> # Create a model
        >>> data = load()
        >>> model = BradleyTerryFactor(data, n_cov_factors=3)

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
            n_cov_factors: int = 3):
        """
        Constructor.
        """

        super().__init__(data, n_cov_factors)

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
            n_cov_factors: int,
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
        ti, tj = w[i + n_agents], w[j + n_agents]
        m = w[n_agents * 2:n_agents * (2 + n_cov_factors)].reshape(
            n_agents, n_cov_factors)
        ri, rj = m[i], m[j]

        s_ij = np.sum(ri * rj, axis=1)
        s_ii = np.sum(ri * ri, axis=1) + ti ** 2
        s_jj = np.sum(rj * rj, axis=1) + tj ** 2

        scale = 1.0 / np.sqrt(s_ii + s_jj - 2.0 * s_ij)
        z = (xi - xj) * scale

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
        loss_ = -cross_entropy(w_ij, p_win) - cross_entropy(l_ij, p_loss)

        if return_jac:
            # grad_z = w_ij * (p_win - 1) + l_ij * p_win
            grad_z = (w_ij + l_ij) * p_win - w_ij

            grad_xi = grad_z * scale
            grad_xj = -grad_xi

            grad_scale = -0.5 * grad_z * z * scale ** 2
            grad_ti = grad_scale * 2.0 * ti
            grad_tj = grad_scale * 2.0 * tj

            grad_ri = grad_scale[:, None] * 2.0 * (ri - rj)
            grad_rj = -grad_ri

            grads = grad_xi, grad_xj, grad_ti, grad_tj, grad_ri, grad_rj

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
            >>> from leaderbot.models import BradleyTerryFactor

            >>> # Create a model
            >>> data = load()
            >>> model = BradleyTerryFactor(data)

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
                                            self.n_cov_factors,
                                            return_jac=return_jac,
                                            inference_only=False)

        loss_ = loss_.sum() / self._count
        if np.isnan(loss_):
            raise RuntimeWarning("loss is nan")

        if return_jac:
            grad_xi, grad_xj, grad_ti, grad_tj, grad_ri, grad_rj = grads
            i, j = self.x.T
            n_samples = self.x.shape[0]
            ax = np.arange(n_samples)
            jac = np.zeros((n_samples, w.shape[0]))
            jac[ax, i] += grad_xi
            jac[ax, j] += grad_xj
            jac[ax, i + self.n_agents] += grad_ti
            jac[ax, j + self.n_agents] += grad_tj

            if self.n_cov_factors > 0:
                dm = np.zeros((n_samples, self.n_agents, self.n_cov_factors))
                dm[ax, i] += grad_ri
                dm[ax, j] += grad_rj
                jac[ax, self._cov_factor_idx] = \
                    dm.reshape(n_samples, self.n_agents * self.n_cov_factors)

            jac = jac.sum(axis=0) / self._count

        if constraint:

            # Extract parameters
            x = w[self._score_idx]
            t = w[self._scale_idx]

            # constraining score parameters
            constraint_score = np.sum(x)
            constraint_score_loss = constraint_score ** 2
            loss_ += constraint_score_loss

            # Constructing covariance
            D = np.diag(t**2)
            if self.n_cov_factors > 0:
                M = w[self._cov_factor_idx].reshape(self.n_agents,
                                                    self.n_cov_factors)
                S = D + M @ M.T
            else:
                S = D

            # Centering covariance
            Id = np.eye(self.n_agents, dtype=float)
            J = np.ones((self.n_agents, self.n_agents), dtype=float)
            C = Id - J / self.n_agents  # centering matrix
            Sc = C @ S @ C   # centered cov

            # Constraining cov (scale and factor) parameters
            constraint_cov = np.trace(Sc) - 1.0
            constraint_cov_loss = constraint_cov ** 2
            loss_ += constraint_cov_loss

            if self.n_cov_factors > 0:
                constraint_factors = np.sum(M @ M.T)  # / self.n_agents
                loss_ += constraint_factors

            if return_jac:
                # constraining score parameters
                constraint_jac = 2.0 * constraint_score
                jac[self._score_idx] += constraint_jac

                # Constraining scale parameters
                constraint_scale_jac = 2.0 * constraint_cov * \
                    (1.0 - 1.0 / self.n_agents) * 2.0 * t
                jac[self._scale_idx] += constraint_scale_jac

                # Constraining factor parameters
                if self.n_cov_factors > 0:
                    constraint_cov_factor_jac = 2.0 * constraint_cov * \
                        2.0 * np.ravel(C @ M)
                    jac[self._cov_factor_idx] += constraint_cov_factor_jac

                    constraint_factor_jac = \
                        np.ravel(2.0 * J @ M)  # / self.n_agents
                    jac[self._cov_factor_idx] += constraint_factor_jac

        if return_jac:
            return loss_, jac
        else:
            return loss_
