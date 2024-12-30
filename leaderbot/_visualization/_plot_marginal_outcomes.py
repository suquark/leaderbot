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
import matplotlib.ticker as mticker
import texplot

__all__ = ['plot_marginal_outcomes']


# ======================
# plot marginal outcomes
# ======================

def plot_marginal_outcomes(
        model,
        max_rank: bool = None,
        bg_color: tuple = 'none',
        fg_color: tuple = 'black',
        save: bool = False,
        latex: bool = False):
    """
    Implementation function for :func:`leaderbot.BaseModel.marginal_outcomes`.
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
    score = model.param[:model.n_agents]
    rank_ = np.argsort(score)[::-1]
    rank_ = rank_[:max_rank]

    # Cumulative count of observed data
    p_wins, p_losses, p_ties = model._cumulative_counts(density=True)
    n_wins, n_losses, n_ties = model._cumulative_counts(density=False)

    # Predicted outcome
    prob = model.infer()

    n_pred = np.sum(model.y, axis=1, keepdims=True) * prob
    p_wins_pred, p_losses_pred, p_ties_pred = \
        model._cumulative_counts(model.x, n_pred, density=True)

    n_wins_pred, n_losses_pred, n_ties_pred = \
        model._cumulative_counts(model.x, n_pred, density=False)

    with texplot.theme(rc={'font.family': 'serif'}, use_latex=latex):

        rng = np.arange(1, 1+max_rank)

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 5.5))

        obs_color = fg_color
        pred_color = 'maroon'

        obs_label = 'Observed Data'
        pred_label = 'Model Prediction'

        # First plot row: frequencies
        ax[0, 0].plot(rng, n_wins[rank_], color=obs_color,
                      label=obs_label)
        ax[0, 0].plot(rng, n_wins_pred[rank_], color=pred_color,
                      label=pred_label)
        ax[0, 1].plot(rng, n_losses[rank_], color=obs_color,
                      label=obs_label)
        ax[0, 1].plot(rng, n_losses_pred[rank_], color=pred_color,
                      label=pred_label)
        ax[0, 2].plot(rng, n_ties[rank_], color=obs_color,
                      label=obs_label)
        ax[0, 2].plot(rng, n_ties_pred[rank_], color=pred_color,
                      label=pred_label)

        # Second plot row: probabilities
        ax[1, 0].plot(rng, 100.0 * p_wins[rank_], color=obs_color,
                      label=obs_label)
        ax[1, 0].plot(rng, 100.0 * p_wins_pred[rank_],
                      color=pred_color, label=pred_label)
        ax[1, 1].plot(rng, 100.0 * p_losses[rank_], color=obs_color,
                      label=obs_label)
        ax[1, 1].plot(rng, 100.0 * p_losses_pred[rank_],
                      color=pred_color, label=pred_label)
        ax[1, 2].plot(rng, 100.0 * p_ties[rank_], color=obs_color,
                      label=obs_label)
        ax[1, 2].plot(rng, 100.0 * p_ties_pred[rank_],
                      color=pred_color, label=pred_label)

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

            if bg_color == 'none':
                legend_bg_color = 'white'
            else:
                legend_bg_color = bg_color

            for i in range(2):
                ax[i, j].legend(fontsize='x-small', facecolor=legend_bg_color,
                                labelcolor=fg_color)
                ax[i, j].set_xlim([rng[0], rng[-1]])
                ax[i, j].set_ylim(bottom=0)
                ax[i, j].set_xlabel('Rank')

        for i in range(2):
            ax[i, 0].set_title('Win')
            ax[i, 1].set_title('Loss')
            ax[i, 2].set_title('Tie')

        # Foreground color
        if fg_color != 'black':

            for ax_ in ax.ravel():

                # Change axis spine colors
                ax_.spines['bottom'].set_color(fg_color)
                ax_.spines['top'].set_color(fg_color)
                ax_.spines['left'].set_color(fg_color)
                ax_.spines['right'].set_color(fg_color)

                # Change tick color
                ax_.tick_params(axis='x', colors=fg_color)
                ax_.tick_params(axis='y', colors=fg_color)

                # Change label color
                ax_.xaxis.label.set_color(fg_color)
                ax_.yaxis.label.set_color(fg_color)

                # Change title color
                ax_.title.set_color(fg_color)

        # Background color
        if bg_color == 'none':
            transparent_bg = True
        else:
            fig.set_facecolor(bg_color)
            for ax_ in ax.ravel():
                ax_.set_facecolor(bg_color)
            transparent_bg = False

        plt.tight_layout()

        texplot.show_or_save_plot(plt, default_filename='rank',
                                  transparent_background=transparent_bg,
                                  dpi=200, show_and_save=save,
                                  verbose=True)
