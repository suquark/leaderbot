# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import numpy as np
import matplotlib                                                  # noqa: F401
import matplotlib.pyplot as plt
import texplot


__all__ = 'compare_ranks'


# =============
# compare ranks
# =============

def compare_ranks(
        models,
        order=None,
        rank_range=None,
        ax=None,
        save=False,
        latex=False):
    """
    Compare ranking of various models with bump chart.

    Parameters
    ----------

    models : list
        A list of models, with at least two models.

    order : list or np.array_like, default=None
        A list of indices to re-order or select from the list of models. If
        `None`, the models appear are given in the ``models`` list.

    rank_range : list, default=None
        A range of ranks ``[rank_min, rank_max]`` to be used in the comparison.
        The rank indices in ``rank_range`` are 1-based indexing and inclusive
        on both start and end of the range. The ranking range is with respect
        to the rank that is provided by the first model in the ``models`` list.
        If `None`, the full range of ranks is used.

    ax : matplotlib.axes._axes.Axes
        If a `matplotlib` axis object is given, this plot is shown in the
        provided axis. Otherwise, a new plot is generated.

    save : bool, default=False
        If `True`, the plot will be saved. This argument is effective only if
        ``plot`` is `True`.

    latex : bool, default=False
        If `True`, the plot is rendered with LaTeX engine, assuming the
        ``latex`` executable is available on the ``PATH``. Enabling this
        option will slow the plot generation.

    Raises
    ------

    RuntimeError
        If any of the models are not trained before calling this method.

    See Also
    --------

    leaderbot.evaluate.model_selection
    leaderbot.evaluate.goodness_of_fit
    leaderbot.evaluate.generalization

    Examples
    --------

    .. code-block:: python
        :emphasize-lines: 25

        >>> import leaderbot as lb
        >>> from leaderbot.models import BradleyTerryFactor as BTF
        >>> from leaderbot.models import RaoKupperFactor as RKF
        >>> from leaderbot.models import DavidsonFactor as DVF

        >>> # Load data
        >>> data = lb.data.load()

        >>> # Create a list of models to compare
        >>> models = [
        ...     BTF(data, n_cov_factors=0),
        ...     BTF(data, n_cov_factors=3),
        ...     RKF(data, n_cov_factors=0, n_tie_factors=0),
        ...     RKF(data, n_cov_factors=0, n_tie_factors=1),
        ...     RKF(data, n_cov_factors=0, n_tie_factors=3),
        ...     DVF(data, n_cov_factors=0, n_tie_factors=0),
        ...     DVF(data, n_cov_factors=0, n_tie_factors=1),
        ...     DVF(data, n_cov_factors=0, n_tie_factors=3)
        ... ]

        >>> # Train the models
        >>> for model in models: model.train()

        >>> # Compare ranking of the models
        >>> lb.evaluate.compare_ranks(models, rank_range=[40, 70])

    The above code produces plot below.

    .. image:: ../_static/images/plots/bump_chart.png
        :align: center
        :class: custom-dark
    """

    if not isinstance(models, list):
        raise ValueError('"models" should be a list of models.')

    if len(models) < 2:
        raise ValueError('At least, two models should be given.')

    for model in models:
        if model.param is None:
            raise RuntimeError('train model first.')

    if order is not None:
        selected_models = [models[i] for i in order]
    else:
        selected_models = models
        order = np.arange(len(selected_models))

    # Get the ranking by each model
    ranks = []
    for model in selected_models:
        ranks.append(model.rank())
    ranks = np.array(ranks)

    if rank_range is None:
        rank_range = [0, ranks.shape[1]]

    # Convert rank to zero-based indexing, but inclusive, meaning both first
    # and last index are included.
    rank_range_ = [rank_range[0]-1, rank_range[1]]

    with texplot.theme(rc={'font.family': 'sans-serif'}, use_latex=latex):

        if ax is None:
            figsize = (
                3.5 + len(selected_models) * (4.4 - 3.5) / 12.0,
                1.0 + (rank_range_[1] - rank_range_[0]) * (12.0 - 1.0) / 60.0)
            fig, ax_ = plt.subplots(figsize=figsize)
        else:
            ax_ = ax

        n_agents = ranks.shape[1]
        n_models = ranks.shape[0]

        colors = ['black', 'firebrick', 'orangered', 'darkgoldenrod',
                  'royalblue', 'mediumblue']
        fontsize = 9

        for i in range(n_agents):
            # Entity ranked i-th in the first model
            entity = ranks[0, i]

            # Find where this entity is ranked in other models
            ranks_over_models = []
            for j in range(n_models):
                # Find its rank (convert to 1-based index)
                rank_in_j = \
                    np.where(ranks[j] == entity)[0][0] + 1
                ranks_over_models.append(rank_in_j)

            color = colors[i % len(colors)]
            ax_.plot(range(n_models),
                     np.array(ranks_over_models) - rank_range_[0],
                     marker='o',
                     markerfacecolor='white', markeredgecolor=color,
                     markeredgewidth=1.75, markersize=4.5,
                     label=f'Entity {entity}',
                     color=color)

        agent_names = selected_models[0].agents

        for i in range(rank_range_[0], rank_range_[1]):
            entity = ranks[0, i]

            name = agent_names[ranks[0, i]]
            name_length = 17
            if len(name) > name_length:
                name = name[:(name_length-3)] + '...'

            ax_.text(-0.13, i - rank_range_[0] + 1, f'{name:>20s}',
                     ha='right',  va='center',
                     transform=ax_.get_yaxis_transform(), fontsize=fontsize)

            ax_.text(0.0, i - rank_range_[0] + 1, f'{i+1:>3d}', ha='right',
                     va='center', transform=ax_.get_yaxis_transform(),
                     fontsize=fontsize)

        for i in range(n_models):
            ax_.text(i, 0, f'Model {order[i]+1}', ha='center', va='bottom',
                     rotation=90, fontsize=fontsize)

        ax_.set_ylim([0.5, rank_range_[1] - rank_range_[0] + 0.5])
        ax_.invert_yaxis()

        ax_.xaxis.set_ticks([])
        ax_.set_xticklabels([])
        ax_.yaxis.set_ticks([])
        ax_.set_yticklabels([])

        ax_.yaxis.set_ticks_position('right')
        ax_.yaxis.set_label_position('right')
        ax_.xaxis.set_ticks_position('top')
        ax_.xaxis.set_label_position('top')

        ax_.spines['top'].set_visible(False)
        ax_.spines['right'].set_visible(False)
        ax_.spines['bottom'].set_visible(False)
        ax_.spines['left'].set_visible(False)

        if ax is None:
            plt.tight_layout()

            texplot.show_or_save_plot(plt, default_filename='bump_chart',
                                      transparent_background=True, dpi=200,
                                      show_and_save=save, verbose=True)
