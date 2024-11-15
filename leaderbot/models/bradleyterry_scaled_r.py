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

__all__ = ['BradleyTerryScaledR']


# =====================
# Bradleyterry Scaled R
# =====================

class BradleyTerryScaledR(BaseModel):
    """
    Paired comparison based on Bradley-Terry and Thurstonian model with
    diagonal plus rank-1 covariance.

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

    BradleyTerry
    BradleyTerryScaled
    BradleyTerryScaledRIJ
    BradleyTerryFactor

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
        >>> from leaderbot.models import BradleyTerryScaledR

        >>> # Create a model
        >>> data = load()
        >>> model = BradleyTerryScaledR(data)

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

        # Total number of parameters
        self.n_param = 2 * self.n_agents + 1

        # Indices of parameters
        self._scale_idx = slice(self.n_agents, self.n_agents * 2)

        # Containing which features
        self._has_scale = True

        # Approximate bound for parameters (only needed for shgo optimization
        # method). Note that these bounds are not enforced, rather, only used
        # for seeding multi-initial points in global optimization methods.
        self._param_bounds = [(-1.0, 1.0) for _ in range(self.n_agents)] + \
                             [(0.01, 1.0) for _ in range(self.n_agents)] + \
                             [(-1.0, 1.0)]

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
        r = w[-1]

        s_r = np.tanh(r)
        scale = 1.0 / np.sqrt(ti ** 2 + tj ** 2 - 2 * s_r * np.abs(ti * tj))
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
        loss_ = - cross_entropy(w_ij, p_win) - cross_entropy(l_ij, p_loss)

        if return_jac:
            # grad_z = w_ij * (p_win - 1) + l_ij * p_win
            grad_z = (w_ij + l_ij) * p_win - w_ij

            grad_xi = grad_z * scale
            grad_xj = -grad_xi

            grad_scale = -0.5 * grad_z * z * scale ** 2
            grad_ti = grad_scale * 2 * (ti - s_r * np.sign(ti) * np.abs(tj))
            grad_tj = grad_scale * 2 * (tj - s_r * np.sign(tj) * np.abs(ti))
            grad_r = -grad_scale * 2 * np.abs(ti * tj) * (1 - s_r ** 2)

        grads = (grad_xi, grad_xj, grad_ti, grad_tj, grad_r)

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
            >>> from leaderbot.models import BradleyTerryScaledR

            >>> # Create a model
            >>> data = load()
            >>> model = BradleyTerryScaledR(data)

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
                                            return_jac=return_jac,
                                            inference_only=False)

        loss_ = loss_.sum() / self._count
        if np.isnan(loss_):
            raise RuntimeWarning("loss is nan")

        if return_jac:
            grad_xi, grad_xj, grad_ti, grad_tj, grad_r = grads
            i, j = self.x.T
            n_samples = self.x.shape[0]
            ax = np.arange(n_samples)
            jac = np.zeros((n_samples, w.shape[0]))
            jac[ax, i] += grad_xi
            jac[ax, j] += grad_xj
            jac[ax, i + self.n_agents] += grad_ti
            jac[ax, j + self.n_agents] += grad_tj
            jac[ax, -1] += grad_r

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
