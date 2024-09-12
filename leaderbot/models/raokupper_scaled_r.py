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
        A dictionary of data that is provided by :func:`leaderbot.load_data`.

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
        >>> from leaderbot.models import RaoKupperScaledR

        >>> # Create a model
        >>> data = load_data()
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
            data: DataType):
        """
        Constructor.
        """

        super().__init__(data)

        self.n_param = 2 * self.n_agents + 2

        # Approximate bound for param (only needed for shgo optimization
        # method). Note that these bounds are not enforced, rather, only used
        # for seeding multi-initial points in global optimization methods.
        self._param_bounds = [(-1.0, 1.0) for _ in range(self.n_agents)] + \
                             [(0.01, 1.0) for _ in range(self.n_agents)] + \
                             [(-1.0, 1.0)] + \
                             [(0.0, 1.0)]

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

        Note: in Rao-Kupper model, eta should be no-negative so that p_tie be
        non-begative. To enforce this, we add absolute value and adjust its
        gradient with sign of eta.
        """

        # Initialize outputs so numba does not complain
        loss_ = None
        grads = None
        probs = None

        i, j = x.T
        xi, xj = w[i], w[j]
        ti, tj = w[i + n_agents], w[j + n_agents]
        r = w[-2]
        min_eta = 1e-2
        eta = np.maximum(w[-1], min_eta)  # clip eta

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
            p_tie = 1 - p_win - p_loss
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
            grad_r = -grad_scale * 2 * np.abs(ti * tj) * (1 - s_r ** 2)

            # At eta=0, the gradient w.r.t eta is "-np.inf * np.sign(eta)".
            # However, for stability of the optimizer, we set its gradient to
            # zero in the clipped interval where it is assumed to be constant.
            if np.abs(eta) < min_eta:
                grad_eta = np.zeros((xi.shape[0]), dtype=float)
            else:
                grad_eta = (grad_dwin + grad_dloss - (2.0 * tie_ij) /
                            (1.0 - np.exp(-2.0 * np.abs(eta)))) * np.sign(eta)

            grads = (grad_xi, grad_xj, grad_ti, grad_tj, grad_r, grad_eta)

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
            Parameters.

        return_jac : bool, default=True
            if `True`, the Jacobian of loss with respect to the param is
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
                An array of the size of the number of param, representing the
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
            >>> from leaderbot.models import RaoKupperScaledR

            >>> # Create a model
            >>> data = load_data()
            >>> model = RaoKupperScaledR(data)

            >>> # Generate an array of parameters
            >>> import numpy as np
            >>> w = np.random.randn(model.n_param)

            >>> # Compute loss and its gradient with respect to param
            >>> loss, jac = model.loss(w, return_jac=True, constraint=False)
        """

        loss_, grads, _ = self._sample_loss(w, self.x, self.y, self.n_agents,
                                            return_jac=return_jac,
                                            inference_only=False)

        loss_ = loss_.sum() / self._count
        if np.isnan(loss_):
            raise RuntimeWarning("loss is nan")

        if return_jac:
            grad_xi, grad_xj, grad_ti, grad_tj, grad_r, grad_eta = grads
            i, j = self.x.T
            n = self.x.shape[0]
            ax = np.arange(n)
            jac = np.zeros((n, w.shape[0]))
            jac[ax, i] += grad_xi
            jac[ax, j] += grad_xj
            jac[ax, i + self.n_agents] += grad_ti
            jac[ax, j + self.n_agents] += grad_tj
            jac[ax, -2] += grad_r
            jac[ax, -1] += grad_eta

            jac = jac.sum(axis=0) / self._count

        if constraint:
            # constraining score parameters
            # constraint_diff = np.sum(np.exp(w[:n_agents])) - 1
            constraint_diff = np.sum(w[:self.n_agents])
            constraint_loss = constraint_diff ** 2
            loss_ += constraint_loss

            # Constraining scale parameters
            constraint_scale = \
                np.sum(w[self.n_agents:2*self.n_agents]**2) - 1.0
            constraint_scale_loss = constraint_scale**2
            loss_ += constraint_scale_loss

            if return_jac:
                # constraining score parameters
                # constraint_jac = 2 * constraint_diff * np.exp(w[:n_agents])
                constraint_jac = 2.0 * constraint_diff
                jac[:self.n_agents] += constraint_jac

                # Constraining scale parameters
                constraint_scale_jac = \
                    4.0 * constraint_scale * w[self.n_agents:2*self.n_agents]
                jac[self.n_agents:2*self.n_agents] += constraint_scale_jac

        if return_jac:
            return loss_, jac
        else:
            return loss_
