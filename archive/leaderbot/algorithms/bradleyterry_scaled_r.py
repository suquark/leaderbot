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
        A dictionary of data that is provided by :func:`leaderbot.load_data`.

    Notes
    -----

    The Bradley-Terry model of paired comparison is based on [1]_. This
    algorithm does not include ties in the data.

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

    Attributes
    ----------

    x : np.ndarray
        A 2D array with shape ``(n_data, 2)`` where each row represents a game
        between the players with the indices ``(i, j)``.

    y : np.ndarray
        A 2D array with the shape ``(n_data, 3)`` where each row represents the
        counts of win, loss, and ties between player ``i`` and ``j``.

    models : list
        A list of the length ``n_models`` representing the name of models.

    n_models : int
        Number of models that play game with each other.

    weights : np.array, default=None
        The algorithm weights (parameters). This array is set once the model
        is trained.

    n_weights : int
        Number of weight parameters

    Methods
    -------

    train
        Train model weights (parameters).

    infer
        Make inference of game probabilities based on a trained model weights.

    predict
        Predict output of a game between players.

    loss
        Loss function of the model.

    rank
        Print leader-board table and plot model prediction counts.

    visualize
        Visualize correlation and score of the models.

    Examples
    --------

    .. code-block:: python

        >>> from leaderbot.data import load_data
        >>> from leaderbot.algorithms import BradleyTerryScaledR

        >>> # Create a model
        >>> data = load_data()
        >>> alg = BradleyTerryScaledR(data)

        >>> # Train the model
        >>> alg.train()

        >>> # Make inference
        >>> p_win, p_loss, p_tie = alg.inference()
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

        self.n_weights = 2 * self.n_models + 1

        # Approximate bound for weights (only needed for shgo optimization
        # method). Note that these bounds are not enforced, rather, only used
        # for seeding multi-initial points in global optimization methods.
        self._weight_bounds = [(-1.0, 1.0) for _ in range(self.n_models)] + \
                              [(0.01, 1.0) for _ in range(self.n_models)] + \
                              [(-1.0, 1.0)]

    # ==================
    # initialize weights
    # ==================

    def _initialize_weights(self):
        """
        Initialize weight parameters.
        """

        # Initial weight
        init_weights = np.zeros(self.n_weights)
        init_weights[:self.n_models] = self._initialize_scores()
        init_weights[self.n_models:self.n_models * 2] = \
            np.full(self.n_models, np.sqrt(1.0 / self.n_models))

        return init_weights

    # ===========
    # sample loss
    # ===========

    @staticmethod
    @numba.jit(nopython=True)
    def _sample_loss(
            w: Union[List[float], np.ndarray[np.floating]],
            x: np.ndarray[np.integer],
            y: np.ndarray[np.integer],
            n_models: int,
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
        ti, tj = w[i + n_models], w[j + n_models]
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
            w: Union[List[float], np.ndarray[np.floating]],
            return_jac: bool = True,
            constraint: bool = True):
        """
        Total loss for all data instances.

        Parameters
        ----------

        w : array_like
            Weight parameters.

        return_jac : bool, default=True
            if `True`, the Jacobian of loss with respect to the weights is
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
                An array of the size of the number of weights, representing the
                Jacobian of loss.

        Raises
        ------

        RuntimeError
            If loss is ``nan``.

        See Also
        --------

        train : train model parameters.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 13

            >>> from leaderbot.data import load_data
            >>> from leaderbot.algorithms import BradleyTerryScaledR

            >>> # Create a model
            >>> data = load_data()
            >>> alg = BradleyTerryScaledR(data)

            >>> # Generate a weight array
            >>> import numpy as np
            >>> w = np.random.randn(alg.n_weights)

            >>> # Compute loss and its gradient with respect to weights
            >>> loss, jac = alg.loss(w, return_jac=True, constraint=False)
        """

        loss_, grads, _ = self._sample_loss(w, self.x, self.y, self.n_models,
                                            return_jac=return_jac,
                                            inference_only=False)

        loss_ = loss_.sum() / self._count
        if np.isnan(loss_):
            raise RuntimeWarning("loss is nan")

        if return_jac:
            grad_xi, grad_xj, grad_ti, grad_tj, grad_r = grads
            i, j = self.x.T
            n = self.x.shape[0]
            ax = np.arange(n)
            jac = np.zeros((n, w.shape[0]))
            jac[ax, i] += grad_xi
            jac[ax, j] += grad_xj
            jac[ax, i + self.n_models] += grad_ti
            jac[ax, j + self.n_models] += grad_tj
            jac[ax, -1] += grad_r

            jac = jac.sum(axis=0) / self._count

        if constraint:
            # constraining score parameters
            # constraint_diff = np.sum(np.exp(w[:n_models])) - 1
            constraint_diff = np.sum(w[:self.n_models])
            constraint_loss = constraint_diff ** 2
            loss_ += constraint_loss

            # Constraining scale parameters
            constraint_scale = \
                np.sum(w[self.n_models:2*self.n_models]**2) - 1.0
            constraint_scale_loss = constraint_scale**2
            loss_ += constraint_scale_loss

            if return_jac:
                # constraining score parameters
                # constraint_jac = 2 * constraint_diff * np.exp(w[:n_models])
                constraint_jac = 2.0 * constraint_diff
                jac[:self.n_models] += constraint_jac

                # Constraining scale parameters
                constraint_scale_jac = \
                    4.0 * constraint_scale * w[self.n_models:2*self.n_models]
                jac[self.n_models:2*self.n_models] += constraint_scale_jac

        if return_jac:
            return loss_, jac
        else:
            return loss_
