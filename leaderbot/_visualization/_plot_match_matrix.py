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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import texplot

__all__ = ['plot_match_matrix']


# ========================
# plot single match matrix
# ========================

def _plot_single_match_matrix(
        fig,
        win_ax,
        tie_ax,
        win_matrix: np.ma.core.MaskedArray,
        tie_matrix: np.ma.core.MaskedArray,
        win_range: tuple = None,
        tie_range: tuple = None,
        density: bool = True,
        horizontal: bool = True,
        extra_title: str = '',
        cbar_label: str = '',
        bg_color: tuple = 'none',
        fg_color: tuple = 'black'):
    """
    Helper function for `plot_match_matrix`. This function plots a single
    plot.
    """

    if win_range is None:
        win_vmin = np.ma.min(win_matrix)
        win_vmax = np.ma.max(win_matrix)
    else:
        win_vmin, win_vmax = win_range

    if tie_range is None:
        tie_vmin = np.ma.min(tie_matrix)
        tie_vmax = np.ma.max(tie_matrix)
    else:
        tie_vmin, tie_vmax = tie_range

    # snap = 0.03
    # vmin = round(vmin / snap) * snap
    # vmax = round(vmax / snap) * snap

    step_size = 5
    labels = [1] + \
        [i for i in range(step_size, win_matrix.shape[0] + 1, step_size)]

    cmap = 'gist_heat_r'

    # Heatmap for wins
    if density:
        im1 = win_ax.imshow(100.0 * win_matrix, cmap=cmap, vmin=100.0*win_vmin,
                            vmax=100.0*win_vmax)
    else:
        im1 = win_ax.imshow(win_matrix, cmap=cmap, vmin=win_vmin,
                            vmax=win_vmax)

    win_ax.set_xticks(
        [0] + list(np.arange(step_size-1, win_matrix.shape[0], step_size)))
    win_ax.set_yticks(
        [0] + list(np.arange(step_size-1, win_matrix.shape[0], step_size)))
    win_ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
    win_ax.set_yticklabels(labels, fontsize=10)
    win_ax.tick_params(axis='both', length=0)
    win_ax.xaxis.set_ticks_position('top')
    win_ax.xaxis.set_label_position('top')
    win_ax.set_xlabel('Rank')
    win_ax.set_ylabel('Rank')
    win_ax.text(0.5, -0.02, 'Win Matrix' + extra_title, ha='center', va='top',
                transform=win_ax.transAxes)

    divider1 = make_axes_locatable(win_ax)
    cax1 = divider1.append_axes("right", size="5%", pad=0.06)

    # Diagonals (which are masked) have value of zero. If vmin is larger than
    # zero, ensure to use a colormap that reaches white color and make colorbar
    # to show the extend mark.
    if win_vmin > 0.0:
        extend = 'min'
    else:
        extend = 'neither'

    cbar1 = fig.colorbar(im1, cax=cax1, extend=extend)
    cbar1.set_label(cbar_label, color=fg_color)

    if density:
        cbar1.ax.yaxis.set_major_formatter(
            mticker.PercentFormatter(decimals=0))

    # Foreground and background colors
    cbar1.ax.yaxis.set_tick_params(color=fg_color)
    cbar1.outline.set_edgecolor(fg_color)
    for tick_label in cbar1.ax.get_yticklabels():
        tick_label.set_color(fg_color)
    cbar1.ax.set_facecolor(bg_color)

    # Heatmap for ties
    if density:
        im2 = tie_ax.imshow(100.0*tie_matrix, cmap=cmap, vmin=100.0*tie_vmin,
                            vmax=100.0*tie_vmax)
    else:
        im2 = tie_ax.imshow(tie_matrix, cmap=cmap, vmin=win_vmin,
                            vmax=win_vmax)

    tie_ax.set_xticks(
        [0] + list(np.arange(step_size-1, win_matrix.shape[0], step_size)))
    tie_ax.set_yticks(
        [0] + list(np.arange(step_size-1, win_matrix.shape[0], step_size)))
    tie_ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
    tie_ax.set_yticklabels(labels, fontsize=10)
    tie_ax.tick_params(axis='both', length=0)
    tie_ax.xaxis.set_ticks_position('top')
    tie_ax.xaxis.set_label_position('top')
    tie_ax.set_xlabel('Rank')
    tie_ax.set_ylabel('Rank')
    tie_ax.text(0.5, -0.02, 'Tie Matrix' + extra_title, ha='center', va='top',
                transform=tie_ax.transAxes)

    divider2 = make_axes_locatable(tie_ax)
    cax2 = divider2.append_axes("right", size="5%", pad=0.06)

    # Diagonals (which are masked) have value of zero. If vmin is larger than
    # zero, ensure to use a colormap that reaches white color and make colorbar
    # to show the extend mark.
    if tie_vmin > 0.0:
        extend = 'min'
    else:
        extend = 'neither'

    cbar2 = fig.colorbar(im2, cax=cax2, extend=extend)
    cbar2.set_label(cbar_label, color=fg_color)

    if density:
        cbar2.ax.yaxis.set_major_formatter(
            mticker.PercentFormatter(decimals=0))

    # Foreground and background colors
    cbar2.ax.yaxis.set_tick_params(color=fg_color)
    cbar2.outline.set_edgecolor(fg_color)
    for tick_label in cbar2.ax.get_yticklabels():
        tick_label.set_color(fg_color)
    cbar2.ax.set_facecolor(bg_color)


# =================
# plot match matrix
# =================

def plot_match_matrix(
        model,
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
    Implementation function for
    :func:`leaderbot.models.BaseModel.match_matrix`.
    """

    scores = model.param[:model.n_agents]
    rank_ = np.argsort(scores)[::-1]
    rank_ = rank_[:max_rank]

    x = model.x
    y = model.y

    y_sum = y.sum(axis=1, keepdims=True)
    y_sum = np.tile(y_sum, (1, y.shape[1]))
    p_obs = y / y_sum

    # Find which rows of X has (i, j) indices both from rank_
    mask = np.isin(x[:, 0], rank_) & np.isin(x[:, 1], rank_)
    row_indices = np.where(mask)[0]

    # The map j = rank_[i] indicates the rank of i of j. Conversely, the
    # inverse map i = inverse_rank_[j] indicates the one element with ran j is
    # the i-th data
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
            # array, even though they might not have had real match. We will
            # make prediction for these pairs.
            x_all = []

            for i in range(max_rank-1):
                for j in range(i+1, max_rank):
                    x_all.append([rank_[i], rank_[j]])

            x_all = np.array(x_all)

            # Make prediction for all matches
            p_pred = model.infer(x_all)

            # Initialize matrices
            p_pred_win = np.ma.masked_all((max_rank, max_rank), dtype=float)
            p_pred_tie = np.ma.masked_all((max_rank, max_rank), dtype=float)

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
            p_pred = model.infer(x)
            n_pred = p_pred * y_sum

            # Initialize matrices
            n_pred_win = np.ma.masked_all((max_rank, max_rank), dtype=float)
            n_pred_tie = np.ma.masked_all((max_rank, max_rank), dtype=float)

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
                _plot_single_match_matrix(
                        fig, ax_win, ax_tie, p_obs_win, p_obs_tie,
                        density=density, win_range=win_range,
                        tie_range=tie_range, horizontal=horizontal,
                        extra_title=' (Observed Data)',
                        cbar_label='Probability', bg_color=bg_color,
                        fg_color=fg_color)
            else:
                _plot_single_match_matrix(
                        fig, ax_win, ax_tie, n_obs_win, n_obs_tie,
                        density=density, win_range=win_range,
                        tie_range=tie_range, horizontal=horizontal,
                        extra_title=' (Observed Data)',
                        cbar_label='Frequency', bg_color=bg_color,
                        fg_color=fg_color)

            # Move to the next row or column for the next plots (if 'both')
            win_idx = win_idx + nxt_idx
            tie_idx = tie_idx + nxt_idx

        if source in ['predicted', 'both']:
            ax_win = ax[tuple(win_idx)]
            ax_tie = ax[tuple(tie_idx)]

            if density:
                _plot_single_match_matrix(
                        fig, ax_win, ax_tie, p_pred_win, p_pred_tie,
                        density=density, win_range=win_range,
                        tie_range=tie_range, horizontal=horizontal,
                        extra_title=' (Model Prediction)',
                        cbar_label='Probability', bg_color=bg_color,
                        fg_color=fg_color)
            else:
                _plot_single_match_matrix(
                        fig, ax_win, ax_tie, n_pred_win, n_pred_tie,
                        density=density, win_range=win_range,
                        tie_range=tie_range, horizontal=horizontal,
                        extra_title=' (Model Prediction)',
                        cbar_label='Frequency', bg_color=bg_color,
                        fg_color=fg_color)

        plt.tight_layout()

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

        if (not horizontal) or (source == 'both'):
            plt.subplots_adjust(hspace=-0.1)

        texplot.show_or_save_plot(plt, default_filename='match_matrix',
                                  transparent_background=transparent_bg,
                                  dpi=200, show_and_save=save,
                                  verbose=True)
