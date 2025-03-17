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
from ..data import DataType
from typing import List, Union

__all__ = ['BaseInterface']


# ==============
# Base Interface
# ==============

class BaseInterface(object):
    """
    Base for interface classes.

    Interface classes include:

    * ``BradleyTerry``
    * ``RaoKupper``
    * ``Davidson``

    which all derived from this class.
    """

    # ====
    # init
    # ====

    def __init__(self):
        """
        Constructor.
        """

        self._delegated_model = None
        self._delegated_model_name = ''

    # ========
    # get attr
    # ========

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying model instance.

        This function delegates only those member methods and attributes that
        do not exists in this class. However, for methods and attributes that
        do exists, this method will not be used.

        For example, the method ``train`` which is defined below here will not
        be delegated using this function.
        """

        return getattr(self._delegated_model, name)

    # ====
    # repr
    # ====

    def __repr__(self):
        """
        Return a string representation of the delegated model.
        """

        model_name = self._delegated_model_name
        k_cov = getattr(self._delegated_model, "k_cov", None)
        k_tie = self._delegated_model.k_tie

        if k_cov is None:
            repres = f'{model_name}(data, k_tie={k_tie})'
        else:
            repres = f'{model_name}(data, k_cov={k_cov}, k_tie={k_tie})'

        return repres

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
            >>> from leaderbot.models import Davidson

            >>> # Create a model
            >>> data = load()
            >>> model = Davidson(data)

            >>> # Generate an array of parameters
            >>> import numpy as np
            >>> w = np.random.randn(model.n_param)

            >>> # Compute loss and its gradient with respect to parameters
            >>> loss, jac = model.loss(w, return_jac=True, constraint=False)
        """

        return self._delegated_model.loss(w=w, return_jac=return_jac,
                                          constraint=constraint)

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
            >>> from leaderbot.models import Davidson

            >>> # Create a model
            >>> data = load()
            >>> model = Davidson(data)

            >>> # Train the model
            >>> model.train()

            >>> # Make inference
            >>> prob = model.infer()
        """

        return self._delegated_model.train(
            init_param=init_param, method=method, max_iter=max_iter, tol=tol)

    # =====
    # infer
    # =====

    def infer(
            self,
            x: Union[List[int], np.ndarray[np.integer], DataType] = None):
        """
        Infer the probabilities of win, loss, and tie outcomes.

        Parameters
        ----------

        x : np.ndarray, list, zip, or leaderbot.data.DataType
            A 2D array (or equivalent list to zip) of integers with the shape
            ``(n_pairs, 2)`` where each row consists of indices ``[i, j]``
            representing a match between a pair of agents with the indice
            ``i`` and ``j``. Alternatively, a dictionary of the type
            :class:`leaderbot.data.DataType` can be provided. If `None`, the
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
        predict: Predict win, loss, or tie between agents.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12

            >>> from leaderbot.data import load
            >>> from leaderbot.models import Davidson

            >>> # Create a model
            >>> data = load()
            >>> model = Davidson(data)

            >>> # Train the model
            >>> model.train()

            >>> # Make inference
            >>> prob = model.infer()
        """

        return self._delegated_model.infer(x=x)

    # =======
    # predict
    # =======

    def predict(
            self,
            x: Union[List[int], np.ndarray[np.integer]] = None):
        """
        Predict outcome between competitors.

        Parameters
        ----------

        x : np.ndarray, list, zip, or leaderbot.data.DataType
            A 2D array (or equivalent list or zip) of integers with the shape
            ``(n_pairs, 2)`` where each row consists of indices ``[i, j]``
            representing a match between a pair of agents with the indices
            ``i`` and ``j``. Alternatively, a dictionary of the type
            :class:`leaderbot.data.DataType` can be provided. If `None`, the
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
            :emphasize-lines: 13

            >>> from leaderbot.data import load
            >>> from leaderbot.models import Davidson

            >>> # Create a model
            >>> data = load()
            >>> model = Davidson(data)

            >>> # Train the model
            >>> model.train()

            >>> # Make prediction
            >>> x = list(zip((0, 1, 2), (1, 2, 0)))
            >>> pred = model.predict(x)
        """

        return self._delegated_model.predict(x=x)

    # ======
    # fisher
    # ======

    def fisher(
            self,
            w: Union[List[float], np.ndarray[np.floating]] = None,
            epsilon: float = 1e-8,
            order: int = 2):
        """
        Observed Fisher information matrix.

        Parameters
        ----------

        w : array_like, default=None
            Parameters :math:`\\boldsymbol{\\theta}`. If `None`, the
            pre-trained parameters are used, provided is already trained.

        epsilon : float, default=1e-8
            The step size in finite differencing method in estimating
            derivative.

        order : {2, 4}, default=2
            Order of Finite differencing:

            * `2`: Second order central difference.
            * `4`: Fourth order central difference.

        Returns
        -------

        J : numpy.ndarray
            The observed Fisher information matrix of size :math:`m \\times m`
            where :math:`m` is the number of parameters.

        Raises
        ------

        RuntimeWarning
            If loss is ``nan``.

        RuntimeError
            If the model is not trained and the input ``w`` is set to `None`.

        See Also
        --------

        loss : Log-likelihood (loss) function.

        Notes
        -----

        The observed Fisher information matrix is the negative of the Hessian
        of the log likelihood function. Namely, if
        :math:`\\boldsymbol{\\theta}` is the array of all parameters of the
        size :math:`m`, then the observed Fisher information is the matrix
        :math:`\\mathcal{J}` of size :math:`m \\times m`

        .. math::

            \\mathcal{J}(\\boldsymbol{\\theta}) =
            - \\nabla \\nabla^{\\intercal} \\ell(\\boldsymbol{\\theta}),

        where :math:`\\ell(\\boldsymbol{\\theta})` is the log-likelihood
        function (see :func:`loss`).

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 13, 17

            >>> from leaderbot.data import load
            >>> from leaderbot.models import Davidson

            >>> # Create a model
            >>> data = load()
            >>> model = Davidson(data)

            >>> # Generate an array of parameters
            >>> import numpy as np
            >>> w = np.random.randn(model.n_param)

            >>> # Fisher information for the given input parameters
            >>> J = model.fisher(w)

            >>> # Fisher information for the trained parameters
            >>> model.train()
            >>> J = model.fisher()
        """

        return self._delegated_model.fisher(w=w, epsilon=epsilon, order=order)

    # ====
    # rank
    # ====

    def rank(self):
        """
        Rank competitors based on their scores.

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
        map_distance

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12

            >>> from leaderbot.data import load
            >>> from leaderbot.models import Davidson

            >>> # Create a model
            >>> data = load()
            >>> model = Davidson(data)

            >>> # Train the model
            >>> model.train()

            >>> # Leaderboard rank and plot
            >>> rnk = model.rank()
        """

        return self._delegated_model.rank()

    # ======
    # scores
    # ======

    def scores(self):
        """
        Get scores.

        Raises
        ------

        RuntimeError
            If the model is not trained before calling this method.

        See Also
        --------

        plot_scores
        rank

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12

            >>> from leaderbot.data import load
            >>> from leaderbot.models import Davidson

            >>> # Create a model
            >>> data = load()
            >>> model = Davidson(data)

            >>> # Train the model
            >>> model.train()

            >>> # Plot scores by rank
            >>> scores = model.scores()
        """

        return self._delegated_model.scores()

    # ===========
    # plot scores
    # ===========

    def plot_scores(
            self,
            max_rank: bool = None,
            horizontal: bool = False,
            plot_range: tuple = None,
            bg_color: tuple = 'none',
            fg_color: tuple = 'black',
            save: bool = False,
            latex: bool = False):
        """
        Plots competitors' scores by rank.

        Parameters
        ----------

        max_rank : int, default=None
            The maximum number of agents to be displayed. If `None`, all
            agents in the input dataset will be ranked and shown.

        horizontal : bool, default=False
            If `True`, horizontal bars will be plotted, otherwise, vertical
            bars will be plotted.

        plot_range : tuple or list, default=None
            A tuple or list of minimum and maximum range of the plot limits.

        bg_color : str or tuple, default='none'
            Color of the background canvas. The default value of ``'none'``
            means transparent.

        fg_color : str or tuple, default='black'
            Color of the axes and text.

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
        map_distance
        leaderboard

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12

            >>> from leaderbot.data import load
            >>> from leaderbot.models import Davidson

            >>> # Create a model
            >>> data = load()
            >>> model = Davidson(data)

            >>> # Train the model
            >>> model.train()

            >>> # Plot scores by rank
            >>> model.plot_scores(max_rank=50)

        The above code provides the text output and plot below.

        .. literalinclude:: ../_static/data/leaderboard.txt
            :language: none

        .. image:: ../_static/images/plots/scores.png
            :align: center
            :class: custom-dark
        """

        self._delegated_model.plot_scores(
            max_rank=max_rank, horizontal=horizontal, plot_range=plot_range,
            bg_color=bg_color, fg_color=fg_color, save=save, latex=latex)

    # ===========
    # leaderboard
    # ===========

    def leaderboard(
            self,
            max_rank: bool = None):
        """
        Print leaderboard of the agent matches.

        Parameters
        ----------

        max_rank : int, default=None
            The maximum number of agents to be displayed. If `None`, all
            agents in the input dataset will be ranked and shown.

        Raises
        ------

        RuntimeError
            If the model is not trained before calling this method.

        See Also
        --------

        map_distance
        marginal_outcomes

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12

            >>> from leaderbot.data import load
            >>> from leaderbot.models import Davidson

            >>> # Create a model
            >>> data = load()
            >>> model = Davidson(data)

            >>> # Train the model
            >>> model.train()

            >>> # Leaderboard report and plot
            >>> model.leaderboard(max_rank=30)

        The above code provides the text output and plot below.

        .. literalinclude:: ../_static/data/leaderboard.txt
            :language: none
        """

        self._delegated_model.leaderboard(max_rank=max_rank)

    # =================
    # marginal outcomes
    # =================

    def marginal_outcomes(
            self,
            max_rank: bool = None,
            bg_color: tuple = 'none',
            fg_color: tuple = 'black',
            save: bool = False,
            latex: bool = False):
        """
        Plot marginal probabilities and frequencies of win, loss, and tie.

        Parameters
        ----------

        max_rank : int, default=None
            The maximum number of agents to be displayed. If `None`, all
            agents in the input dataset will be ranked and shown.

        bg_color : str or tuple, default='none'
            Color of the background canvas. The default value of ``'none'``
            means transparent.

        fg_color : str or tuple, default='black'
            Color of the axes and text.


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

        map_distance
        leaderboard

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12

            >>> from leaderbot.data import load
            >>> from leaderbot.models import Davidson

            >>> # Create a model
            >>> data = load()
            >>> model = Davidson(data)

            >>> # Train the model
            >>> model.train()

            >>> # Leaderboard report and plot
            >>> model.marginal_outcomes(max_rank=30)

        .. image:: ../_static/images/plots/rank.png
            :align: center
            :class: custom-dark
        """

        self._delegated_model.marginal_outcomes(
            max_rank=max_rank, bg_color=bg_color, fg_color=fg_color, save=save,
            latex=latex)

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
            horizontal: bool = False,
            bg_color: tuple = 'none',
            fg_color: tuple = 'black',
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

        horizontal : bool, default=False
            If `True`, the subplots for win and tie are placed row-wise. If
            `False`, they are plotted in column-wise.

        bg_color : str or tuple, default='none'
            Color of the background canvas. The default value of ``'none'``
            means transparent.

        fg_color : str or tuple, default='black'
            Color of the axes and text.

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

        map_distance

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

        self._delegated_model.match_matrix(
            max_rank=max_rank, density=density, source=source,
            win_range=win_range, tie_range=tie_range, horizontal=horizontal,
            bg_color=bg_color, fg_color=fg_color, save=save, latex=latex)

    # ============
    # map distance
    # ============

    def map_distance(
            self,
            ax=None,
            cmap=None,
            max_rank: int = None,
            method: str = 'kpca',
            dim: str = '3d',
            sign: tuple = None,
            bg_color: tuple = 'none',
            fg_color: tuple = 'black',
            save: bool = False,
            latex: bool = False):
        """
        Visualize distance between agents using manifold learning projection.

        Parameters
        ----------

        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, default=None
            Axis object for plotting. If `None`, a 3D axis is created.

        cmap : matplotlib.colors.LinearSegmentedColormap, default=None
            Colormap for the plot. If `None`, a default colormap is used.

        max_rank : int, default=None
            The maximum number of agents to be displayed. If `None`, all
            agents in the input dataset will be ranked and shown.

        method : {``'kpca'``, ``'mds'``}
            Method of visualization:

            * ``'kpca'``: Kernel-PCA
            * ``'mds'``: Multi-Dimensional Scaling

        dim : tuple or {``'2d'``, ``'3d'``}
            Dimension of visualization. If a tuple is given, the specific axes
            indices in the tuple is plotted. For example, ``(2, 0)`` plots
            principal axes :math:`(x_2, x_0)`.

        sign : tuple, default=None
            A tuple consisting `-1` and `1`, representing the sign each axes.
            For example, ``sign=(1, -1)`` together with ``dim=(0, 2)`` plots
            the principal axes :math:`(x_0, -x_2)`. If `None`, all signs are
            assumed to be positive.

        bg_color : str or tuple, default='none'
            Color of the background canvas. The default value of ``'none'``
            means transparent.

        fg_color : str or tuple, default='black'
            Color of the axes and text.

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

        cluster
        rank

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12

            >>> from leaderbot.data import load
            >>> from leaderbot.models import Davidson

            >>> # Create a model
            >>> data = load()
            >>> model = Davidson(data)

            >>> # Train the model
            >>> model.train()

            >>> # Plot kernel PCA
            >>> model.map_distance(max_rank=50)

        The above code produces plot below.

        .. image:: ../_static/images/plots/kpca.png
            :align: center
            :class: custom-dark
        """

        self._delegated_model.map_distance(
            ax=ax, cmap=cmap, max_rank=max_rank, method=method, dim=dim,
            sign=sign, bg_color=bg_color, fg_color=fg_color, save=save,
            latex=latex)

    # =======
    # cluster
    # =======

    def cluster(
            self,
            ax=None,
            max_rank: int = None,
            tier_label: bool = False,
            method: str = 'complete',
            color_threshold: float = 0.15,
            bg_color: tuple = 'none',
            fg_color: tuple = 'black',
            save: bool = False,
            latex: bool = False):
        """
        Cluster competitors to performance tiers.

        Parameters
        ----------

        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, default=None
            Axis object for plotting. If `None`, a 3D axis is created.

        max_rank : int, default=None
            The maximum number of agents to be displayed. If `None`, all
            agents in the input dataset will be ranked and shown.

        tier_label : bool, default=False,
            If `True`, the branch lines up to the first three hierarchies are
            labeled.

        method : str, default='complete'
            Clustering algorithm. See scipy.cluster.hierarchy.linkage methods.

        color_threshold : float, default=0.15
            A threshold between 0 and 1 where linkage distance above the
            threshold is rendered in black and below the threshold is rendered
            in colors.

        bg_color : str or tuple, default='none'
            Color of the background canvas. The default value of ``'none'``
            means transparent.

        fg_color : str or tuple, default='black'
            Color of the axes and text.

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

        map_distance
        rank

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12

            >>> from leaderbot.data import load
            >>> from leaderbot.models import RaoKupperFactor

            >>> # Create a model
            >>> data = load()
            >>> model = RaoKupperFactor(data, n_cov_factor=3, n_tie_factor=20)

            >>> # Train the model
            >>> model.train()

            >>> # Plot kernel PCA
            >>> model.cluster(max_rank=100, tier_label=True, latex=True)

        The above code produces plot below.

        .. image:: ../_static/images/plots/cluster.png
            :align: center
            :class: custom-dark
            :width: 50%
        """

        self._delegated_model.cluster(
            ax=ax, max_rank=max_rank, tier_label=tier_label, method=method,
            color_threshold=color_threshold, bg_color=bg_color,
            fg_color=fg_color, save=save, latex=latex)
