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

__all__ = ['plot_match_matrices']


# ===================
# plot match matrices
# ===================

def plot_match_matrices(
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
        cbar_label: str = ''):
    """
    Helper function for :func:`leaderbot.models.BaseModel.match_matrix`.
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
    # zero, ensure to use a colomap that reaches white color and make colorbar
    # to show the extend mark.
    if win_vmin > 0.0:
        extend = 'min'
    else:
        extend = 'neither'

    cbar1 = fig.colorbar(im1, cax=cax1, extend=extend)
    cbar1.set_label(cbar_label)

    if density:
        cbar1.ax.yaxis.set_major_formatter(
            mticker.PercentFormatter(decimals=0))

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
    # zero, ensure to use a colomap that reaches white color and make colorbar
    # to show the extend mark.
    if tie_vmin > 0.0:
        extend = 'min'
    else:
        extend = 'neither'

    cbar2 = fig.colorbar(im2, cax=cax2, extend=extend)
    cbar2.set_label(cbar_label)

    if density:
        cbar2.ax.yaxis.set_major_formatter(
            mticker.PercentFormatter(decimals=0))
