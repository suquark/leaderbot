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
import matplotlib.pyplot as plt
import texplot

__all__ = ['_plot_scores']


# ===========
# plot scores
# ===========


def _plot_scores(
        model,
        max_rank: bool = None,
        horizontal: bool = False,
        plot_range: tuple = None,
        bg_color: tuple = 'none',
        fg_color: tuple = 'black',
        save: bool = False,
        latex: bool = False):
    """
    Implementation function for :func:`leaderbot.BaseModel.plot_scores`.
    """

    if model.param is None:
        raise RuntimeError('train model first.')

    # Check input arguments
    if max_rank is None:
        max_rank = model.n_agents
    elif max_rank > model.n_agents:
        raise ValueError('"max_rank" can be at most equal to the number ' +
                         ' of agents.')

    # Scores are the x_i, x_j parameters across all models
    scores = model.param[:model.n_agents]
    rank_ = np.argsort(scores)[::-1]
    rank_ = rank_[:max_rank]
    scores_ranked = scores[rank_]
    agents_ranked = np.array(model.agents)[rank_]

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
                    facecolor='firebrick', edgecolor='none')
            ax.set_xlabel('Score', fontsize=10)
            ax.set_ylim([-0.75, len(agents_ranked) - 0.25])
            ax.tick_params(axis='y', which='both', length=0)
            ax.grid(True, axis='x', linestyle='--', alpha=0.6)

            if plot_range is not None:
                ax.set_xlim(plot_range)

        else:
            # Vertical bars
            ax.bar(agents_ranked, scores_ranked, facecolor='firebrick',
                   edgecolor='none')
            ax.set_ylabel('Score', fontsize=10)
            ax.set_xlim([-0.75, len(agents_ranked) - 0.25])
            ax.tick_params(axis='x', which='both', length=0)
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)
            ax.tick_params(axis='x', rotation=90, labelsize=9,
                           labelright=False)

            if plot_range is not None:
                ax.set_ylim(plot_range)

        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_visible(False)

        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=9)

        # Foreground color
        if fg_color != 'black':

            # Change axis spine colors
            ax.spines['bottom'].set_color(fg_color)
            ax.spines['top'].set_color(fg_color)
            ax.spines['left'].set_color(fg_color)
            ax.spines['right'].set_color(fg_color)

            # Change tick color
            ax.tick_params(axis='x', colors=fg_color)
            ax.tick_params(axis='y', colors=fg_color)

            # Change label color
            ax.xaxis.label.set_color(fg_color)
            ax.yaxis.label.set_color(fg_color)

            # Change title color
            ax.title.set_color(fg_color)

        # Background color
        if bg_color == 'none':
            transparent_bg = True
        else:
            fig.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
            transparent_bg = False

        plt.tight_layout()

        texplot.show_or_save_plot(plt, default_filename='scores',
                                  transparent_background=transparent_bg,
                                  dpi=200, show_and_save=save,
                                  verbose=True)
