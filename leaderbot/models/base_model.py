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
from typing import List, Union
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
        Print leaderboard table and plot prediction for agents.

    visualize
        Visualize correlation and score of the agents.
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

    # ==========
    # covariance
    # ==========

    def _covariance(
            self,
            centered: bool = False):
        """
        Covariance matrix

        Parameters
        ----------

        centered : bool, default = False
            If `True`, the doubly-centered operator is applied to the
            covariance matrix, making it doubly-stochastic Gramian matrix
            with null space of dim 1 and zero sum rows and columns.
        """

        if self.param is None:
            raise RuntimeError('train model first.')

        if self.param.size < 2 * self.n_agents:
            # The model does not have Thurstonian covariance.
            return None

        # Diagonals of covariance matrix
        ti = np.abs(self.param[self.n_agents:2*self.n_agents])

        # Off-diagonals of correlation matrix
        n_pairs = self.n_agents * (self.n_agents - 1) // 2
        if self.param.size >= 2*self.n_agents + n_pairs:
            rij = self.param[2*self.n_agents:2*self.n_agents + n_pairs]
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
            C = np.eye(self.n_agents) - v @ v.T / self.n_agents

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
        S = self._covariance(centered=True)

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

        predict : predict probabilities based on given param.

        Notes
        -----

        The trained parameters are available as ``param`` attribute.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 9

            >>> from leaderbot.data import load_data
            >>> from leaderbot.models import DavidsonScaled

            >>> # Create a model
            >>> data = load_data()
            >>> model = DavidsonScaled(data)

            >>> # Train the model
            >>> model.train()

            >>> # Make prediction
            >>> p_win, p_loss, p_tie = model.infer()
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
            An array of the shape ``(n_samples, 3)`` where the columns
            represent the win, loss, and tie probabilities for the model `i`
            against model `j` in order that appears in the member `x`.

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

            >>> from leaderbot.data import load_data
            >>> from leaderbot.models import DavidsonScaled

            >>> # Create a model
            >>> data = load_data()
            >>> model = DavidsonScaled(data)

            >>> # Train the model
            >>> model.train()

            >>> # Make inference
            >>> p_win, p_loss, p_tie = model.infer()
        """

        if self.param is None:
            raise RuntimeError('train model first.')

        if x is None:
            x = self.x

        # Call sample loss to only compute probabilities, but not loss itself
        # _, _, probs = self._sample_loss(self.param, x, None, self.n_agents,
        _, _, probs = self._sample_loss(self.param, x, self.y, self.n_agents,
                                        return_jac=False, inference_only=True)

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
        inference : make inference of the probabilities of win, loss, and tie.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12

            >>> from leaderbot.data import load_data
            >>> from leaderbot.models import DavidsonScaled

            >>> # Create a model
            >>> data = load_data()
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

    def rank(
            self,
            max_rank: bool = None,
            plot: bool = False,
            save: bool = False):
        """
        Rank agents based on their scores.

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

            >>> from leaderbot.data import load_data
            >>> from leaderbot.models import DavidsonScaled

            >>> # Create a model
            >>> data = load_data()
            >>> model = DavidsonScaled(data)

            >>> # Train the model
            >>> model.train()

            >>> # Leaderboard rank and plot
            >>> model.rank(max_rank=30, plot=True)

        The above code provides the text output and plot below.

        .. literalinclude:: ../_static/data/rank.txt
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

        # Cumulative count of observed data
        p_wins, p_losses, p_ties = self._cumulative_counts(density=True)
        n_wins, n_losses, n_ties = self._cumulative_counts(density=False)
        n_matches = n_wins + n_losses + n_ties

        # Predicted outcome
        prob = self.infer()

        n_pred = np.sum(self.y, axis=1, keepdims=True) * prob
        p_wins_pred, p_losses_pred, p_ties_pred = \
            self._cumulative_counts(self.x, n_pred, density=True)

        # Scores are the x_i, x_j parameters across all models
        score = self.param[:self.n_agents]
        rank_ = np.argsort(score)[::-1]
        rank_ = rank_[:max_rank]

        print('+---------------------------+--------+--------+--------------' +
              '-+---------------+')
        print('|                           |        |    num |   observed   ' +
              ' |   predicted   |')
        print('| rnk  agent                |  score |  match | win loss  tie' +
              ' | win loss  tie |')
        print('+---------------------------+--------+--------+--------------' +
              '-+---------------+')

        for i in range(max_rank):

            name = self.agents[rank_[i]]
            if len(name) > 20:
                name = name[:17] + '...'

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

            with texplot.theme(rc={'font.family': 'serif'}):

                rng = np.arange(1, 1+max_rank)

                fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 5.5))

                # First plot row: frequencies
                ax[0, 0].plot(rng, n_wins[rank_], color='maroon',
                              label='observed')
                ax[0, 0].plot(rng, n_wins_pred[rank_], color='black',
                              label='predicted')
                ax[0, 1].plot(rng, n_losses[rank_], color='maroon',
                              label='observed')
                ax[0, 1].plot(rng, n_losses_pred[rank_], color='black',
                              label='predicted')
                ax[0, 2].plot(rng, n_ties[rank_], color='maroon',
                              label='observed')
                ax[0, 2].plot(rng, n_ties_pred[rank_], color='black',
                              label='predicted')

                # Second plot row: probabilities
                ax[1, 0].plot(rng, p_wins[rank_], color='maroon',
                              label='observed')
                ax[1, 0].plot(rng, p_wins_pred[rank_], color='black',
                              label='predicted')
                ax[1, 1].plot(rng, p_losses[rank_], color='maroon',
                              label='observed')
                ax[1, 1].plot(rng, p_losses_pred[rank_], color='black',
                              label='predicted')
                ax[1, 2].plot(rng, p_ties[rank_], color='maroon',
                              label='observed')
                ax[1, 2].plot(rng, p_ties_pred[rank_], color='black',
                              label='predicted')

                for j in range(3):
                    ax[1, j].set_ylim(top=1)
                    ax[0, j].set_ylabel('Frequency')
                    ax[1, j].set_ylabel('Probability')

                    for i in range(2):
                        ax[i, j].legend(fontsize='small')
                        ax[i, j].set_xlim([rng[0], rng[-1]])
                        ax[i, j].set_ylim(bottom=0)
                        ax[i, j].set_xlabel('Model Rank')

                for i in range(2):
                    ax[i, 0].set_title('Wins')
                    ax[i, 1].set_title('Losses')
                    ax[i, 2].set_title('Ties')

                plt.tight_layout()

                texplot.show_or_save_plot(plt, default_filename='rank',
                                          transparent_background=True, dpi=200,
                                          show_and_save=save, verbose=True)

    # =========
    # visualize
    # =========

    def visualize(
            self,
            max_rank: int = None,
            method: str = 'kpca',
            dim: str = '3d',
            save: bool = False):
        """

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

            >>> from leaderbot.data import load_data
            >>> from leaderbot.models import DavidsonScaled

            >>> # Create a model
            >>> data = load_data()
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
        with texplot.theme(rc={'font.family': 'sans-serif'}):

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
                    ax.text(x_[i], y_[i], name, fontsize=8, ha='center',
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
                    ax.text(x_[i], y_[i], z_[i], name, fontsize=8, ha='center',
                            va='center')

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

                ax.xaxis.pane.set_edgecolor('black')  # Set edge color
                ax.yaxis.pane.set_edgecolor('black')
                ax.zaxis.pane.set_edgecolor('black')

                ax.xaxis.pane.set_linewidth(1)  # Set edge line width
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
                                      transparent_background=True, dpi=200,
                                      show_and_save=save, verbose=True)
