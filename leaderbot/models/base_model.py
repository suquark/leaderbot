# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# imports
# =======

import numpy as np
import scipy.optimize
from ..data import DataType
from ._plot_utils import plot_match_matrices
from typing import List, Union
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import texplot
from sklearn.manifold import MDS
from sklearn.decomposition import KernelPCA

__all__ = ['BaseModel']


# ==========
# Base Model
# ==========

class BaseModel(object):
    """
    Base class for all models.

    .. note::

        This class should not be instantiated.

    See Also
    --------

    BradleyTerry
    BradleyTerryScaled
    BradleyTerryScaledR
    BradleyTerryScaledRIJ
    RaoKupper
    RaoKupperScaled
    RaoKupperScaledR
    RaoKupperScaledRIJ
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

    Methods
    -------

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
        Plots scores versus rank

    match_matrix
        Plot match matrices of win and tie counts of mutual matches.
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

        # Public members
        self.x = np.array(data['X'])
        self.y = np.array(data['Y'])
        self.agents = data['models']
        self.n_agents = len(self.agents)
        self.param = None

        # Protected members
        self._score_idx = slice(None, self.n_agents)
        self._scale_idx = slice(None, None)
        self._cov_factor_idx = slice(None, None)
        self._tie_factor_idx = slice(None, None)
        self._has_scale = False
        self._has_cov_factor = False
        self._has_tie_factor = False
        self._count = np.sum(self.y)
        self._result = None

    # =================
    # cumulative counts
    # =================

    def _cumulative_counts(
            self,
            x: Union[List[int], np.ndarray] = None,
            y: Union[List[int], np.ndarray] = None,
            density: bool = False):
        """
        Cumulative number of wins, losses, and ties for each agent with
        respect to all its agents against all other agents.

        This function returns either the frequencies or probabilities
        (density).

        Parameters
        ----------

        x : np.ndarray
            A 2D array of integers with the shape ``(n_pairs, 2)`` where each
            row consists of indices ``[i, j]`` representing a match between a
            pair of agents with the indices ``i`` and ``j``.

        y : np.ndarray
            A 2D array of integers with the shape ``(n_pairs, 3)`` where each
            row consists of three counts ``[n_win, n_loss, n_ties]``
            representing the frequencies of win, loss, and ties between agents
            ``i`` and ``j`` given by the corresponding row of the input array
            ``x``.

        density : bool, default=False
            If `True`, the output is given as probabilities. If `False`, the
            output is given as frequencies.
        """

        if x is None:
            x = self.x

        if y is None:
            y = self.y

        if x.shape[0] != y.shape[0]:
            raise ValueError('"x" and "y" data are not consistent.')

        wins, losses, ties = y[:, 0], y[:, 1], y[:, 2]

        n_agents = self.n_agents
        n_wins = np.zeros((n_agents,), dtype=int)
        n_losses = np.zeros((n_agents,), dtype=int)
        n_ties = np.zeros((n_agents,), dtype=int)
        n_matches = np.zeros((n_agents,), dtype=int)

        for i in range(n_agents):

            # Find all matches with the i-th agent
            ind_0 = (x[:, 0] == i)
            ind_1 = (x[:, 1] == i)

            # All counts where i won j plus all counts where j lost i
            n_wins[i] = np.sum(wins[ind_0]) + np.sum(losses[ind_1])

            # All counts where i lost j plus all counts where j won i
            n_losses[i] = np.sum(losses[ind_0]) + np.sum(wins[ind_1])

            # All counts where both i and j tied
            n_ties[i] = np.sum(ties[ind_0]) + np.sum(ties[ind_1])

        if density:
            # Normalize by the total counts of each agent
            n_matches = n_wins + n_losses + n_ties
            p_wins = n_wins / n_matches
            p_losses = n_losses / n_matches
            p_ties = n_ties / n_matches

            return p_wins, p_losses, p_ties

        else:
            return n_wins, n_losses, n_ties

    # =================
    # initialize scores
    # =================

    def _initialize_scores(self):
        """
        Initialize scores based on cumulative win and tie counts.
        """

        # Initialize scores based on cumulative probability of wins and ties
        p_wins, _, p_ties = self._cumulative_counts(density=True)
        init_scores = p_wins + p_ties / 2.0
        init_scores[np.isnan(init_scores)] = 0.0
        init_scores = init_scores - np.mean(init_scores)

        return init_scores

    # ==============
    # generate basis
    # ==============

    def _generate_basis(
            self,
            n_rows: int,
            n_cols: int):
        """
        Generate an orthonormal matrix for tie factor model.

        The basis is generated using discrete cosine transform of type four.

        This function is called in those models that have tie incorporated in
        their formulation.
        """

        basis = np.zeros((n_rows, n_cols), dtype=float)
        if n_cols > 0:

            # Discrete cosine transform basis of type four (DCT-IV)
            for i in range(1, n_rows + 1):
                for j in range(1, n_cols + 1):
                    basis[i-1, j-1] = np.sqrt(2.0 / n_rows) * \
                        np.cos((np.pi / n_rows) * (i-0.5) * (j-0.5))

        return basis

    # ==============
    # get covariance
    # ==============

    def _get_covariance(
            self,
            param: np.ndarray = None,
            centered: bool = False):
        """
        Covariance matrix

        Parameters
        ----------

        param : np.ndarray, default=None
            Model parameters. If `None`, the trained model parameters are used.

        centered : bool, default = False
            If `True`, the doubly-centered operator is applied to the
            covariance matrix, making it doubly-stochastic Gramian matrix
            with null space of dim 1 and zero sum rows and columns.
        """

        if param is None:
            if self.param is None:
                raise RuntimeError('train model first.')
            param = self.param

        if param.size < 2 * self.n_agents:
            # The model does not have Thurstonian covariance.
            return None

        if not self._has_scale:
            raise RuntimeError('model does have scale parameters.')

        # Diagonals of covariance matrix
        ti = np.abs(param[self._scale_idx])

        # Off-diagonals of correlation matrix
        if not self.has_cov_factor:
            n_pairs = self.n_agents * (self.n_agents - 1) // 2
            if param.size >= 2*self.n_agents + n_pairs:
                rij = param[2*self.n_agents:2*self.n_agents + n_pairs]
            else:
                rij = None

        # S is Covariance
        S = np.zeros((self.n_agents, self.n_agents), dtype=float)

        for j in range(self.n_agents):
            for i in range(j, self.n_agents):

                if i == j:
                    S[i, i] = ti[i]**2

                else:
                    if rij is None:
                        S[i, j] = 0.0
                    else:
                        k = j * (j-1) // 2 + i
                        S[i, j] = np.abs(ti[i] * ti[j]) * np.tanh(rij[k])

                    # Symmetry
                    S[j, i] = S[i, j]

        if centered:
            # double-centering operator C
            v = np.ones((self.n_agents, 1))
            J = v @ v.T  # matrix of all ones
            Id = np.eye(self.n_agents)
            C = Id - J / self.n_agents

            # double-centering Gram matrix
            S = C @ S @ C

        return S

    # ===============
    # distance matrix
    # ===============

    def _distance_matrix(self):
        """
        Distance matrix between mutual agents.
        """

        if self.param is None:
            raise RuntimeError('train model first.')

        # Get covariance matrix
        S = self._get_covariance(param=self.param, centered=True)

        # Scores
        xi = self.param[:self.n_agents]

        # Distance matrix
        D = np.zeros((self.n_agents, self.n_agents), dtype=float)

        for i in range(self.n_agents):
            for j in range(i, self.n_agents):

                if i == j:
                    D[i, j] = 0.0
                else:
                    if S is None:
                        # For non-Thurstonian models
                        D[i, j] = np.abs(xi[i] - xi[j])
                    else:
                        D[i, j] = np.abs(xi[i] - xi[j]) / \
                            np.sqrt(S[i, i] + S[j, j] - 2.0 * S[i, j])

                    # Symmetry
                    D[j, i] = D[i, j]

        return D

    # =====
    # train
    # =====

    def train(
            self,
            init_param: Union[List[float], np.ndarray[np.floating]] = None,
            method: str = 'BFGS',
            max_iter: int = 1500,
            tol: float = 1e-8):
        """
        Tune model parameters with maximum likelihood estimation method.

        Parameters
        ----------

        init_param : array_like, default=None
            Initial parameters. If `None`, an initial guess is used based on
            the cumulative counts between agent matches.

        method : str, default= ``'BFGS'``
            Optimization method.

            * ``'BFGS'``: local optimization (best method overall)
            * ``'L-BFGS-B'``: local optimization (best method for all
              ``ScaledRIJ`` models)
            * ``'CG'``: local optimization
            * ``'Newton-CG'``: local optimization (most accurate method, but
              slow)
            * ``'TNC'``: local optimization (least accurate method)
            * ``'Nelder-Mead'``: local optimization (slow)
            * ``'Powell'``: local optimization (often does not converge)
            * ``'shgo'``: Hybrid global and local optimization (slow)
            * ``'basinhopping'``: Hybrid global and local optimization (slow)

            See `scipy.optimize` for further details on each of the above
            methods.

        max_iter : int, default=1500
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
            >>> from leaderbot.models import DavidsonScaled

            >>> # Create a model
            >>> data = load()
            >>> model = DavidsonScaled(data)

            >>> # Train the model
            >>> model.train()

            >>> # Make inference
            >>> prob = model.infer()
        """

        # Check or initialize param
        if init_param is None:
            init_param = self._initialize_param()

        # Check method
        if method in ['BFGS', 'L-BFGS-B', 'CG', 'Newton-CG', 'TNC', 'shgo',
                      'basinhopping']:
            jac = True
        elif method in ['Nelder-Mead', 'Powell']:
            jac = False
        else:
            # Other methods need Hessian, which is not implemented.
            raise ValueError('"method" is not supported.')

        # Apply constrain on the loss function
        constraint = True

        if method in ['BFGS', 'L-BFGS-B', 'CG', 'Newton-CG', 'TNC',
                      'Nelder-Mead', 'Powell']:

            # Local optimization methods
            self._result = scipy.optimize.minimize(
                    self.loss,
                    init_param,
                    args=(jac, constraint),
                    jac=jac,
                    method=method,
                    options={'maxiter': max_iter},
                    tol=tol)

        elif method in ['sho', 'basinhopping']:

            # Global optimization methods with BFGS local optimization
            if method == 'shgo':

                minimizer_kwargs = {
                    'method': 'BFGS',
                    'jac': jac,
                }

                self._result = scipy.optimize.shgo(
                        self.loss,
                        self._param_bounds, n=20,
                        minimizer_kwargs=minimizer_kwargs,
                        args=(jac, constraint),
                        worders=-1)

            elif method == 'basinhopping':

                # basinhopping requires args to be passed through local
                # optimizer
                minimizer_kwargs = {
                    'method': 'BFGS',
                    'jac': jac,
                    'args': (jac, constraint),
                }

                self._result = scipy.optimize.basinhopping(
                        self.loss,
                        init_param, niter=20,
                        minimizer_kwargs=minimizer_kwargs)

        self.param = self._result.x

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
        y = np.empty_like(self.y)  # not used

        if hasattr(self, "basis") and hasattr(self, "n_tie_factors"):
            # For those models that have factored tie
            _, _, probs = self._sample_loss(self.param, x, self.y,
                                            self.n_agents, self.n_tie_factors,
                                            self.basis, return_jac=False,
                                            inference_only=True)
        else:
            # For those models that do not have factored tie
            _, _, probs = self._sample_loss(self.param, x, y, self.n_agents,
                                            return_jac=False,
                                            inference_only=True)

        return np.column_stack(probs)

    # =======
    # predict
    # =======

    def predict(
            self,
            x: Union[List[int], np.ndarray[np.integer]] = None):
        """
        Predict win, loss, or tie between agents.

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
            An array of the shape ``(n_agents, )`` where each entry represents
            the following codes:

            * ``+1``: agent ``i`` wins ``j``.
            * ``0``: agent ``i`` ties ``j``.
            * ``-1``: agent ``i`` losses to ``j``.

        Raises
        ------

        RuntimeError
            If the model is not trained before calling this method.

        See Also
        --------

        train : train model parameters.
        infer : make inference of the probabilities of win, loss, and tie.

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

            >>> # Make prediction
            >>> x = zip((0, 1, 2), (1, 2, 0))
            >>> pred = model.predict(x)
        """

        if self.param is None:
            raise RuntimeError('train model first.')

        if x is None:
            x = self.x

        probs = self.infer(x)
        max_ind = np.argmax(probs, axis=1)

        pred = np.zeros((probs.shape[0], ), dtype=int)
        for i in range(pred.shape[0]):
            if max_ind[i] == 0:
                pred[0] = +1
            elif max_ind[i] == 1:
                pred[0] = -1

        return pred

    # ====
    # rank
    # ====

    def rank(self):
        """
        Rank agents based on their scores.

        Returns
        -------

        rnk : np.ndarray
            A 1D array of the size ``n_agents``, containing the zero-based
            indices that rank agents from the higher to lowest scores.

        Raises
        ------

        RuntimeError
            If the model is not trained before calling this method.

        See Also
        --------

        leaderboard
        visualize

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

            >>> # Leaderboard rank and plot
            >>> rnk = model.rank()
        """

        if self.param is None:
            raise RuntimeError('train model first.')

        # Scores are the x_i, x_j parameters across all models
        score = self.param[:self.n_agents]
        rnk = np.argsort(score)[::-1]

        return rnk

    # ===========
    # plot scores
    # ===========

    def plot_scores(
            self,
            max_rank: bool = None,
            horizontal: bool = True,
            save: bool = False,
            latex: bool = False):
        """
        Plots agents scores by rank.

        Parameters
        ----------

        max_rank : int, default=None
            The maximum number of agents to be displayed. If `None`, all
            agents in the input dataset will be ranked and shown.

        horizontal : bool, default=True
            If `True`, horizontal bars will be plotted, otherwise, vertical
            bars will be plotted.

        save : bool, default=False
            If `True`, the plot will be saved. This argument is effective only
            if ``plot`` is `True`.

        latex : bool, default=False
            If `True`, the plot is rendered with LaTeX engine, assuming the
            ``latex`` executable is available on the ``PATH``. Enabling this
            option will slow the plot generation.

        Raises
        ------

        RuntimeError
            If the model is not trained before calling this method.

        See Also
        --------

        rank
        visualize
        leaderboard

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

            >>> # Plot scores by rank
            >>> model.plot_scores(max_rank=30)

        The above code provides the text output and plot below.

        .. literalinclude:: ../_static/data/leaderboard.txt
            :language: none

        .. image:: ../_static/images/plots/scores.png
            :align: center
            :class: custom-dark
        """

        if self.param is None:
            raise RuntimeError('train model first.')

        # Check input arguments
        if max_rank is None:
            max_rank = self.n_agents
        elif max_rank > self.n_agents:
            raise ValueError('"max_rank" can be at most equal to the number ' +
                             ' of agents.')

        # Scores are the x_i, x_j parameters across all models
        scores = self.param[:self.n_agents]
        rank_ = np.argsort(scores)[::-1]
        rank_ = rank_[:max_rank]
        scores_ranked = scores[rank_]
        agents_ranked = np.array(self.agents)[rank_]

        with texplot.theme(rc={'font.family': 'sans-serif'}, use_latex=latex):

            num_bars = len(agents_ranked)
            fig_length = num_bars * 0.20
            fig_width = 5

            if horizontal:
                figsize = (fig_width, fig_length)
            else:
                figsize = (fig_length, fig_width)

            fig, ax = plt.subplots(figsize=figsize)

            if horizontal:
                # Horizontal bars
                ax.barh(agents_ranked[::-1], scores_ranked[::-1],
                        color='firebrick')
                ax.set_xlabel('Score', fontsize=10)
                ax.set_ylim([-0.75, len(agents_ranked) - 0.25])
                ax.tick_params(axis='y', which='both', length=0)
                ax.grid(True, axis='x', linestyle='--', alpha=0.6)

            else:
                # Vertical bars
                ax.bar(agents_ranked, scores_ranked, color='firebrick')
                ax.set_ylabel('Score', fontsize=10)
                ax.set_xlim([-0.75, len(agents_ranked) - 0.25])
                ax.tick_params(axis='x', which='both', length=0)
                ax.grid(True, axis='y', linestyle='--', alpha=0.6)
                ax.tick_params(axis='x', rotation=90, labelsize=9,
                               labelright=False)

            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # ax.spines['left'].set_visible(False)

            ax.tick_params(axis='y', labelsize=9)
            ax.tick_params(axis='x', labelsize=9)

            plt.tight_layout()

            plt.show()

            texplot.show_or_save_plot(plt, default_filename='scores',
                                      transparent_background=False,
                                      dpi=200, show_and_save=save,
                                      verbose=True)

    # ===========
    # leaderboard
    # ===========

    def leaderboard(
            self,
            max_rank: bool = None,
            plot: bool = False,
            save: bool = False,
            latex: bool = False):
        """
        Print leaderboard of the agent matches.

        Parameters
        ----------

        max_rank : int, default=None
            The maximum number of agents to be displayed. If `None`, all
            agents in the input dataset will be ranked and shown.

        plot : bool, default=False
            If `True`, the observed and predicted frequencies of matches will
            be plotted against rank.

        save : bool, default=False
            If `True`, the plot will be saved. This argument is effective only
            if ``plot`` is `True`.

        latex : bool, default=False
            If `True`, the plot is rendered with LaTeX engine, assuming the
            ``latex`` executable is available on the ``PATH``. Enabling this
            option will slow the plot generation.

        Raises
        ------

        RuntimeError
            If the model is not trained before calling this method.

        See Also
        --------

        visualize

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

            >>> # Leaderboard report and plot
            >>> model.leaderboard(max_rank=30, plot=True)

        The above code provides the text output and plot below.

        .. literalinclude:: ../_static/data/leaderboard.txt
            :language: none

        .. image:: ../_static/images/plots/rank.png
            :align: center
            :class: custom-dark
        """

        if self.param is None:
            raise RuntimeError('train model first.')

        # Check input arguments
        if max_rank is None:
            max_rank = self.n_agents
        elif max_rank > self.n_agents:
            raise ValueError('"max_rank" can be at most equal to the number ' +
                             ' of agents.')

        # Scores are the x_i, x_j parameters across all models
        score = self.param[:self.n_agents]
        rank_ = np.argsort(score)[::-1]
        rank_ = rank_[:max_rank]

        # Cumulative count of observed data
        p_wins, p_losses, p_ties = self._cumulative_counts(density=True)
        n_wins, n_losses, n_ties = self._cumulative_counts(density=False)
        n_matches = n_wins + n_losses + n_ties

        # Predicted outcome
        prob = self.infer()

        n_pred = np.sum(self.y, axis=1, keepdims=True) * prob
        p_wins_pred, p_losses_pred, p_ties_pred = \
            self._cumulative_counts(self.x, n_pred, density=True)

        print('+---------------------------+--------+--------+--------------' +
              '-+---------------+')
        print('|                           |        |    num |   observed   ' +
              ' |   predicted   |')
        print('| rnk  agent                |  score |  match | win loss  tie' +
              ' | win loss  tie |')
        print('+---------------------------+--------+--------+--------------' +
              '-+---------------+')

        for i in range(max_rank):

            name_length = 20
            name = self.agents[rank_[i]]
            if len(name) > name_length:
                name = name[:(name_length - 3)] + '...'

            print(f'| {i+1:>3}. {name:<20} | '
                  f'{score[rank_[i]]:>+0.3f} | '
                  f'{n_matches[rank_[i]]:>6} | '
                  f'{100*p_wins[rank_[i]]:>2.0f}%  '
                  f'{100*p_losses[rank_[i]]:>2.0f}%  '
                  f'{100*p_ties[rank_[i]]:>2.0f}% | '
                  f'{100*p_wins_pred[rank_[i]]:>2.0f}%  '
                  f'{100*p_losses_pred[rank_[i]]:>2.0f}%  '
                  f'{100*p_ties_pred[rank_[i]]:>2.0f}% | ')

        print('+---------------------------+--------+--------+--------------' +
              '-+---------------+')

        if plot:

            n_wins_pred, n_losses_pred, n_ties_pred = \
                self._cumulative_counts(self.x, n_pred, density=False)

            with texplot.theme(rc={'font.family': 'serif'}, use_latex=latex):

                rng = np.arange(1, 1+max_rank)

                fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 5.5))

                # First plot row: frequencies
                ax[0, 0].plot(rng, n_wins[rank_], color='maroon',
                              label='Observed')
                ax[0, 0].plot(rng, n_wins_pred[rank_], color='black',
                              label='Predicted')
                ax[0, 1].plot(rng, n_losses[rank_], color='maroon',
                              label='Observed')
                ax[0, 1].plot(rng, n_losses_pred[rank_], color='black',
                              label='Predicted')
                ax[0, 2].plot(rng, n_ties[rank_], color='maroon',
                              label='Observed')
                ax[0, 2].plot(rng, n_ties_pred[rank_], color='black',
                              label='Predicted')

                # Second plot row: probabilities
                ax[1, 0].plot(rng, 100.0 * p_wins[rank_], color='maroon',
                              label='Observed')
                ax[1, 0].plot(rng, 100.0 * p_wins_pred[rank_], color='black',
                              label='Predicted')
                ax[1, 1].plot(rng, 100.0 * p_losses[rank_], color='maroon',
                              label='Observed')
                ax[1, 1].plot(rng, 100.0 * p_losses_pred[rank_], color='black',
                              label='Predicted')
                ax[1, 2].plot(rng, 100.0 * p_ties[rank_], color='maroon',
                              label='Observed')
                ax[1, 2].plot(rng, 100.0 * p_ties_pred[rank_], color='black',
                              label='Predicted')

                for j in range(3):
                    ax[1, j].set_ylim(top=100)
                    ax[0, j].set_ylabel('Frequency')
                    ax[1, j].set_ylabel('Probability')

                    # Format y axis to use 10k labels instead of 10000
                    ax[0, j].yaxis.set_major_formatter(mticker.FuncFormatter(
                        lambda x, _: f'{int(x/1000)}k'))

                    # Format y axis to use percent
                    ax[1, j].yaxis.set_major_formatter(
                        mticker.PercentFormatter(decimals=0))

                    for i in range(2):
                        ax[i, j].legend(fontsize='small')
                        ax[i, j].set_xlim([rng[0], rng[-1]])
                        ax[i, j].set_ylim(bottom=0)
                        ax[i, j].set_xlabel('Rank')

                for i in range(2):
                    ax[i, 0].set_title('Wins')
                    ax[i, 1].set_title('Losses')
                    ax[i, 2].set_title('Ties')

                plt.tight_layout()

                texplot.show_or_save_plot(plt, default_filename='rank',
                                          transparent_background=False,
                                          dpi=200, show_and_save=save,
                                          verbose=True)

    # ============
    # match matrix
    # ============

    def match_matrix(
            self,
            max_rank: bool = None,
            density: bool = True,
            source: str = 'both',
            win_range: tuple = None,
            tie_range: tuple = None,
            horizontal: bool = True,
            save: bool = False,
            latex: bool = False):
        """
        Plot match matrices of win and tie counts of mutual matches.

        Parameters
        ----------

        max_rank : int, default=None
            The maximum number of agents to be displayed. If `None`, all
            agents in the input dataset will be ranked and shown.

        density : bool, default=True
            If `False`, the frequency (count) of win and tie are plotted. If
            `True`, the probability of the win and tie are plotted.

        source : {``'observed'``, ``'predicted'``, ``'both'``},\
                default= ``'both'``

            The source of data to be used:

            * ``'observed'``: The observed win and tie counts based on the
              input training data to the model.
            * ``'predicted'``: The prediction of win and tie counts by the
              trained model.
            * ``'both'``: Plots both of the observed and predicted data.

        win_range : tuple, default=None
            The tuple of two float numbers ``(vmin, vmax)`` determining the
            range of the heatmap plot for win matrix. If `None`, the minimum
            and maximum range of data is used.

        tie_range : tuple, default=None
            The tuple of two float numbers ``(vmin, vmax)`` determining the
            range of the heatmap plot for tie matrix. If `None`, the minimum
            and maximum range of data is used.

        horizontal : bool, default=True
            If `True`, the subplots for win and tie are placed row-wise. If
            `False`, they are plotted in column-wise.

        save : bool, default=False
            If `True`, the plot will be saved. This argument is effective only
            if ``plot`` is `True`.

        latex : bool, default=False
            If `True`, the plot is rendered with LaTeX engine, assuming the
            ``latex`` executable is available on the ``PATH``. Enabling this
            option will slow the plot generation.

        Raises
        ------

        RuntimeError
            If the model is not trained before calling this method.

        See Also
        --------

        visualize

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12, 13, 14

            >>> from leaderbot.data import load
            >>> from leaderbot.models import Davidson

            >>> # Create a model
            >>> data = load()
            >>> model = Davidson(data)

            >>> # Train the model
            >>> model.train()

            >>> # Plot match matrices for win and tie
            >>> model.match_matrix(max_rank=20, density=True, source='both',
            ...                    latex=True, save=True, horizontal=True,
            ...                    win_range=[0.2, 0.6], tie_range=[0.15, 0.4])

        The above code provides the text output and plot below.

        .. image:: ../_static/images/plots/match_matrix_density_true.png
            :align: center
            :class: custom-dark

        Similarly, plots for win and tie frequencies can be obtained as
        follows:

        .. code-block:: python

            >>> # Plot match matrices for win and tie
            >>> model.match_matrix(max_rank=20, density=False, source='both',
            ...                    latex=True, save=True, horizontal=True,
            ...                    win_range=[0, 3000], tie_range=[0, 1500])

        .. image:: ../_static/images/plots/match_matrix_density_false.png
            :align: center
            :class: custom-dark
        """

        scores = self.param[:self.n_agents]
        rank_ = np.argsort(scores)[::-1]
        rank_ = rank_[:max_rank]

        x = self.x
        y = self.y

        y_sum = y.sum(axis=1, keepdims=True)
        y_sum = np.tile(y_sum, (1, y.shape[1]))
        p_obs = y / y_sum

        # Find which rows of X has (i, j) indices both from rank_
        mask = np.isin(x[:, 0], rank_) & np.isin(x[:, 1], rank_)
        row_indices = np.where(mask)[0]

        # The map j = rank_[i] indicates the rank of i of j. Conversely, the
        # inverse map i = inverse_rank_[j] indicates the one element with ran
        # j is the i-th data
        inverse_rank = {value: idx for idx, value in enumerate(rank_)}

        # Check arguments
        if source not in ['observed', 'predicted', 'both']:
            raise ValueError('Invalid "source" argument.')

        # Generate match matrices for observed data
        if source in ['observed', 'both']:

            # Initialize matrices
            if density:
                # probability of observations
                p_obs_win = np.ma.masked_all((max_rank, max_rank), dtype=float)
                p_obs_tie = np.ma.masked_all((max_rank, max_rank), dtype=float)
            else:
                # Count (frequency) of observations
                n_obs_win = np.ma.masked_all((max_rank, max_rank), dtype=int)
                n_obs_tie = np.ma.masked_all((max_rank, max_rank), dtype=int)

            # Iterate over all rows of input data x containing rank_ indices
            for row in row_indices:

                # Get the actual indices
                i, j = x[row, :]

                # Get the rank of these indices
                rank_i = inverse_rank.get(i, None)
                rank_j = inverse_rank.get(j, None)

                if density:
                    # Probability of observations
                    p_obs_win[rank_i, rank_j] = p_obs[row, 0]
                    p_obs_win[rank_j, rank_i] = p_obs[row, 1]  # use loss
                    p_obs_tie[rank_i, rank_j] = p_obs[row, 2]
                    p_obs_tie[rank_j, rank_i] = p_obs[row, 2]  # symmetry
                else:
                    # Count (frequency) of observations
                    n_obs_win[rank_i, rank_j] = y[row, 0]
                    n_obs_win[rank_j, rank_i] = y[row, 1]  # use loss
                    n_obs_tie[rank_i, rank_j] = y[row, 2]
                    n_obs_tie[rank_j, rank_i] = y[row, 2]  # symmetry

        # Generate match matrices for predicted data
        if source in ['predicted', 'both']:

            if density:
                # Construct the list of all pairs between elements in the rank_
                # array, even though they might not have had real match. We
                # will make prediction for these pairs.
                x_all = []

                for i in range(max_rank-1):
                    for j in range(i+1, max_rank):
                        x_all.append([rank_[i], rank_[j]])

                x_all = np.array(x_all)

                # Make prediction for all matches
                p_pred = self.infer(x_all)

                # Initialize matrices
                p_pred_win = np.ma.masked_all((max_rank, max_rank),
                                              dtype=float)
                p_pred_tie = np.ma.masked_all((max_rank, max_rank),
                                              dtype=float)

                for row in range(x_all.shape[0]):
                    i, j = x_all[row, :]
                    rank_i = inverse_rank.get(i, None)
                    rank_j = inverse_rank.get(j, None)

                    # Probabilities of predictions
                    p_pred_win[rank_i, rank_j] = p_pred[row, 0]
                    p_pred_win[rank_j, rank_i] = p_pred[row, 1]
                    p_pred_tie[rank_i, rank_j] = p_pred[row, 2]
                    p_pred_tie[rank_j, rank_i] = p_pred[row, 2]

            else:
                # Make prediction only on those pairs that had actual match.
                p_pred = self.infer(x)
                n_pred = p_pred * y_sum

                # Initialize matrices
                n_pred_win = np.ma.masked_all((max_rank, max_rank),
                                              dtype=float)
                n_pred_tie = np.ma.masked_all((max_rank, max_rank),
                                              dtype=float)

                for row in row_indices:
                    i, j = x[row, :]
                    rank_i = inverse_rank.get(i, None)
                    rank_j = inverse_rank.get(j, None)

                    # Count (frequency) of predictions
                    n_pred_win[rank_i, rank_j] = n_pred[row, 0]
                    n_pred_win[rank_j, rank_i] = n_pred[row, 1]
                    n_pred_tie[rank_i, rank_j] = n_pred[row, 2]
                    n_pred_tie[rank_j, rank_i] = n_pred[row, 2]

        with texplot.theme(rc={'font.family': 'sans-serif'}, use_latex=latex):

            # Determine figure size and number of rows and column in the figure
            size = 4.5
            if source == 'both':
                figsize = (2*size, 2*size)
                nrows = 2
                ncols = 2
            else:
                if horizontal:
                    figsize = (2*size, size)
                    nrows = 1
                    ncols = 2
                else:
                    figsize = (size, 2*size)
                    nrows = 2
                    ncols = 1

            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

            if source != 'both':
                ax = np.atleast_2d(ax)
                if not horizontal:
                    ax = ax.T

            win_idx = np.array([0, 0])
            if horizontal:
                tie_idx = np.array([0, 1])
                nxt_idx = np.array([1, 0])
            else:
                tie_idx = np.array([1, 0])
                nxt_idx = np.array([0, 1])

            if source in ['observed', 'both']:
                ax_win = ax[tuple(win_idx)]
                ax_tie = ax[tuple(tie_idx)]

                if density:
                    plot_match_matrices(fig, ax_win, ax_tie, p_obs_win,
                                        p_obs_tie, density=density,
                                        win_range=win_range,
                                        tie_range=tie_range,
                                        horizontal=horizontal,
                                        extra_title=' (Observed Data)',
                                        cbar_label='Probability')
                else:
                    plot_match_matrices(fig, ax_win, ax_tie, n_obs_win,
                                        n_obs_tie, density=density,
                                        win_range=win_range,
                                        tie_range=tie_range,
                                        horizontal=horizontal,
                                        extra_title=' (Observed Data)',
                                        cbar_label='Frequency')

                # Move to the next row or column for the next plots (if 'both')
                win_idx = win_idx + nxt_idx
                tie_idx = tie_idx + nxt_idx

            if source in ['predicted', 'both']:
                ax_win = ax[tuple(win_idx)]
                ax_tie = ax[tuple(tie_idx)]

                if density:
                    plot_match_matrices(fig, ax_win, ax_tie, p_pred_win,
                                        p_pred_tie, density=density,
                                        win_range=win_range,
                                        tie_range=tie_range,
                                        horizontal=horizontal,
                                        extra_title=' (Model Prediction)',
                                        cbar_label='Probability')
                else:
                    plot_match_matrices(fig, ax_win, ax_tie, n_pred_win,
                                        n_pred_tie, density=density,
                                        win_range=win_range,
                                        tie_range=tie_range,
                                        horizontal=horizontal,
                                        extra_title=' (Model Prediction)',
                                        cbar_label='Frequency')

            plt.tight_layout()

            if (not horizontal) or (source == 'both'):
                plt.subplots_adjust(hspace=-0.1)

            texplot.show_or_save_plot(plt, default_filename='match_matrix',
                                      transparent_background=False,
                                      dpi=200, show_and_save=save,
                                      verbose=True)

    # =========
    # visualize
    # =========

    def visualize(
            self,
            max_rank: int = None,
            method: str = 'kpca',
            dim: str = '3d',
            save: bool = False,
            latex: bool = False):
        """
        Visualize correlation between agents using manifold learning
        projection.

        Parameters
        ----------

        max_rank : int, default=None
            The maximum number of agents to be displayed. If `None`, all
            agents in the input dataset will be ranked and shown.

        method : {``'kpca'``, ``'mds'``}
            Method of visualization:

            * ``'kpca'``: Kernel-PCA
            * ``'mds'``: Multi-Dimensional Scaling

        dim : {``'2d'``, ``'3d'``}
            Dimension of visualization

        save : bool, default=False
            If `True`, the plot will be saved. This argument is effective only
            if ``plot`` is `True`.

        latex : bool, default=False
            If `True`, the plot is rendered with LaTeX engine, assuming the
            ``latex`` executable is available on the ``PATH``. Enabling this
            option will slow the plot generation.

        Raises
        ------

        RuntimeError
            If the model is not trained before calling this method.

        See Also
        --------

        rank

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

            >>> # Plot kernel PCA
            >>> model.visualize(max_rank=50)

        The above code produces plot below.

        .. image:: ../_static/images/plots/kpca.png
            :align: center
            :class: custom-dark
        """

        if self.param is None:
            raise RuntimeError('train model first.')

        # Check input arguments
        if max_rank is None:
            max_rank = self.n_agents
        elif max_rank > self.n_agents:
            raise ValueError('"max_rank" can be at most equal to the number ' +
                             ' of agents.')

        if method not in ['kpca', 'mds']:
            raise ValueError('Invalid method.')

        if dim not in ['2d', '3d']:
            raise ValueError('Invalid dimension.')

        # Compute distance matrix
        D = self._distance_matrix()

        # Compute visualization points
        if method == 'mds':

            # MDS method
            mds = MDS(n_components=3, dissimilarity='precomputed',
                      random_state=42)
            points = mds.fit_transform(D)

            # 3D visualization settings
            if dim == '3d':
                elev, azim, roll = 8, 115, 0

        elif method == 'kpca':

            # Kernel PCA
            gamma = 1e-4  # You might need to adjust gamma based on your data
            kernel_matrix = np.exp(-gamma * D ** 2)
            kpca = KernelPCA(n_components=3, kernel='precomputed')
            points = kpca.fit_transform(kernel_matrix)

            # 3D visualization settings
            if dim == '3d':
                elev, azim, roll = 8, 145, 0

        # Scores are the x_i, x_j parameters across all models
        score = self.param[:self.n_agents]
        rank_ = np.argsort(score)[::-1]
        rank_ = rank_[:max_rank]

        # Reorder variables based on rank
        points_ranked = points[rank_]
        agents_ranked = np.array(self.agents)[rank_]
        score_ranked = score[rank_]
        colors = np.linspace(0.0, 1.0, max_rank)

        # Plot titles
        # if method == 'mds':
        #     method_name = 'Multi-Dimensional Projection'
        # elif method == 'kpca':
        #     method_name = 'Kernel PCA'
        # if dim == '2d':
        #     dim_name = '2D'
        # elif dim == '3d':
        #     dim_name = '3D'

        # Visualize
        with texplot.theme(rc={'font.family': 'sans-serif'}, use_latex=latex):

            fontsize = 9

            if dim == '2d':

                sizes = [5000 * (score_ranked[i] - score_ranked[max_rank-1]) +
                         100 * score_ranked[max_rank-1]
                         for i in range(max_rank)]

                x_ = -points_ranked[:, 0]
                y_ = points_ranked[:, 1]

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(x_, y_, s=sizes, c=colors, cmap='turbo_r',
                           alpha=0.55)

                for i, name in enumerate(agents_ranked[:]):
                    ax.text(x_[i], y_[i], name, fontsize=fontsize, ha='center',
                            va='center')

                # ax.set_aspect('equal', adjustable='box')
                # ax.set_xlim([-0.021, 0])
                # ax.set_ylim([-0.005, 0.0125])
                ax.set_xlabel(r'$x_1$')
                ax.set_ylabel(r'$x_2$')
                # ax.set_title(f'{method_name} in {dim_name}')

            elif dim == '3d':

                c_map = 'turbo_r'
                # c_map = 'nipy_spectral_r'
                # c_map = 'gnuplot2_r'

                sizes = [5000 * (score_ranked[i] - score_ranked[max_rank-1]) +
                         100 * score_ranked[max_rank-1]
                         for i in range(max_rank)]

                x_ = points_ranked[:, 0]
                y_ = -points_ranked[:, 1]
                z_ = -points_ranked[:, 2]

                fig = plt.figure(figsize=(8, 7))
                ax = fig.add_subplot(projection='3d')
                ax.set_proj_type('persp', focal_length=0.2)

                ax.scatter(x_, y_, z_, s=sizes, c=colors, cmap=c_map,
                           alpha=0.6)

                for i, name in enumerate(agents_ranked[:]):
                    ax.text(x_[i], y_[i], z_[i], name, fontsize=fontsize,
                            ha='center', va='center')

                ax.view_init(elev=elev, azim=azim, roll=roll)

                # Remove tick labels
                # ax.set_xticks([])
                # ax.set_yticks([])
                # ax.set_zticks([])

                # ax.grid(False)

                ax.set_xlabel(r'$x_1$')
                ax.set_ylabel(r'$x_2$')
                ax.set_zlabel(r'$x_3$')
                # ax.set_title(f'{method_name} in {dim_name}')

                # Remove axis panes (background of the plot)
                # ax.xaxis.pane.fill = False
                # ax.yaxis.pane.fill = False
                # ax.zaxis.pane.fill = False

                # Set edge color
                ax.xaxis.pane.set_edgecolor('black')
                ax.yaxis.pane.set_edgecolor('black')
                ax.zaxis.pane.set_edgecolor('black')

                # Set edge line width
                ax.xaxis.pane.set_linewidth(1)
                ax.yaxis.pane.set_linewidth(1)
                ax.zaxis.pane.set_linewidth(1)

                x_min = np.min(x_)
                x_max = np.max(x_)
                y_min = np.min(y_)
                y_max = np.max(y_)
                z_min = np.min(z_)
                z_max = np.max(z_)

                eps = 0.05
                dx = x_max - x_min
                dy = y_max - y_min
                dz = z_max - z_min

                ax.axes.set_xlim3d(left=x_min - dx * eps,
                                   right=x_max + dx * eps)
                ax.axes.set_ylim3d(bottom=y_min - dy * eps,
                                   top=y_max + dy * eps)
                ax.axes.set_zlim3d(bottom=z_min - dz * eps,
                                   top=z_max + dz * eps)

                ax.set_box_aspect(aspect=None, zoom=1)
                plt.subplots_adjust(left=-0.05, right=0.9, top=1.1,
                                    bottom=0.05)

            plt.tight_layout()

            texplot.show_or_save_plot(plt, default_filename='visualization',
                                      transparent_background=False,
                                      dpi=200, show_and_save=save,
                                      verbose=True)
