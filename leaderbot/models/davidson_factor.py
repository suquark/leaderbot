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
from .util import double_sigmoid, cross_entropy
from ..data import DataType
from typing import List, Union

__all__ = ['DavidsonFactor']

# ===================
# Davidson Scaled RIJ
# ===================


class DavidsonFactor(FactorModel):
    """
    Paired comparison based on Davidson model and Thurstonian model with
    full-rank covariance.

    Parameters
    ----------

    data : dict
        A dictionary of data that is provided by
        :func:`leaderbot.data.load`.

    n_factors : int, default=3
        Number of factors for matrix factorization.

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

    n_factors : int
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
        Print leaderboard table and plot prediction for agents.

    visualize
        Visualize correlation and score of the agents.

    Examples
    --------

    .. code-block:: python

        >>> from leaderbot.data import load
        >>> from leaderbot.models import DavidsonFactor

        >>> # Create a model
        >>> data = load()
        >>> model = DavidsonFactor(data)

        >>> # Train the model
        >>> model.train()

        >>> # Make inference
        >>> p_win, p_loss, p_tie = model.infer()
    """

    # ====
    # init
    # ====

    def __init__(self, data: DataType, n_factors: int = 3):
        """
        Constructor.
        """
        super().__init__(data, n_factors)
        self.n_param += 1
        self._param_bounds.append((-1.0, 1.0))

    # ===========
    # sample loss
    # ===========

    @staticmethod
    @numba.jit(nopython=True)
    def _sample_loss(w: Union[List[float], np.ndarray[np.floating]],
                     x: np.ndarray[np.integer],
                     y: np.ndarray[np.integer],
                     n_agents: int,
                     n_factors: int,
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
        i, j = x.T

        xi, xj = w[i], w[j]
        ti, tj = w[i + n_agents], w[j + n_agents]
        m = w[n_agents * 2:n_agents * (2 + n_factors)].reshape(
            n_agents, n_factors)
        ri, rj = m[i], m[j]

        s_ij = np.sum(ri * rj, axis=1)
        s_ii = np.sum(ri * ri, axis=1) + ti ** 2
        s_jj = np.sum(rj * rj, axis=1) + tj ** 2

        scale = 1 / np.sqrt(s_ii + s_jj - 2 * s_ij)
        z = (xi - xj) * scale

        mu = w[-1]
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
            grad_ti = grad_scale * 2 * ti
            grad_tj = grad_scale * 2 * tj

            grad_ri = grad_scale[:, None] * 2 * (ri - rj)
            grad_rj = -grad_ri

            grads = (grad_xi, grad_xj, grad_ti, grad_tj, grad_ri, grad_rj,
                     grad_mu)

        return loss_, grads, probs

    # ====
    # loss
    # ====

    def loss(self,
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
            >>> from leaderbot.models import DavidsonFactor

            >>> # Create a model
            >>> data = load()
            >>> model = DavidsonFactor(data)

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
                                            n_factors=self.n_factors,
                                            return_jac=return_jac,
                                            inference_only=False)

        loss_ = loss_.sum() / self._count
        if np.isnan(loss_):
            raise RuntimeWarning("loss is nan")

        if return_jac:
            grad_xi, grad_xj, grad_ti, grad_tj, grad_ri, grad_rj, grad_mu = grads
            i, j = self.x.T
            n = self.x.shape[0]
            ax = np.arange(n)
            jac = np.zeros((n, w.shape[0]))
            jac[ax, i] += grad_xi
            jac[ax, j] += grad_xj
            jac[ax, i + self.n_agents] += grad_ti
            jac[ax, j + self.n_agents] += grad_tj
            dm = np.zeros((n, self.n_agents, self.n_factors))
            dm[ax, i] += grad_ri
            dm[ax, j] += grad_rj
            jac[ax, self.n_agents * 2:self.n_agents * (2 + self.n_factors)] = \
                dm.reshape(n, self.n_agents * self.n_factors)
            jac[ax, -1] += grad_mu

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
            constraint_scale_loss = constraint_scale ** 2
            loss_ += constraint_scale_loss

            if return_jac:
                # constraining score parameters
                # constraint_jac = 2 * constraint_diff * np.exp(w[:n_agents])
                constraint_jac = 2.0 * constraint_diff
                jac[:self.n_agents] += constraint_jac

                # Constraining scale parameters
                constraint_scale_jac = \
                    4.0 * constraint_scale * w[self.n_agents:2*self.n_agents]
                jac[self.n_agents:2 * self.n_agents] += constraint_scale_jac

        if return_jac:
            return loss_, jac
        else:
            return loss_
