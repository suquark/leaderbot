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
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import optimal_leaf_ordering
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
import matplotlib.colors as mcolors
import texplot

__all__ = ['plot_cluster']


def hls_to_hex(h_, l_, s_):
    """
    Convert HLS (Hue, Lightness, Saturation) to Hex color.
    """

    # Convert HLS to RGB (0–1 range)
    rgb = hls_to_rgb(h_, l_, s_)

    # Convert RGB to Hex
    return mcolors.to_hex(rgb)


# ===================
# get distance matrix
# ===================

def _get_distance_matrix(model, scores, rnk):
    """
    Dissimilarity matrix is used for hierarchical clustering.
    """

    # Get dissimilarity matrix
    S = model._get_covariance(centered=False)
    S = S[rnk, :][:, rnk]

    n = S.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            D[i, j] = (scores[i] - scores[j]) / \
                    np.sqrt(S[i, i] + S[j, j] - 2.0 * S[i, j])
            D[j, i] = D[i, j]

    return D


# ===================
# plot linear cluster
# ===================

def _plot_linear_cluster(ax, linkage, labels, color_threshold, colors,
                         bg_color, fg_color, rc, save, latex):
    """
    Plot cluster in Cartesian coordinates.
    """

    linkage[:, 2] = linkage[:, 2] / (np.max(linkage[:, 2]) * 1.02)

    with texplot.theme(use_latex=latex, font_scale=0.85):

        plt.rcParams.update(rc)
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 15))
        else:
            fig = ax.get_figure()

        sch.set_link_color_palette(colors)
        dendro = sch.dendrogram(linkage, no_plot=False, labels=labels,
                                orientation='left', no_labels=False,
                                color_threshold=color_threshold,
                                show_contracted=True,
                                above_threshold_color=fg_color, ax=ax,
                                leaf_rotation=0)

        label_colors = dendro['leaves_color_list']
        # Get y-axis labels for orientation 'left'
        x_labels = ax.get_ymajorticklabels()
        for lbl, color in zip(x_labels, label_colors):
            lbl.set_color(color)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(False)  # Keep only left spine (y-axis)
        # ax.get_xaxis().set_visible(False)

        # ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        # ax.set_xlim([1e-3, 1])
        # ax.set_xscale('log')
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel('Linkage Dist.')
        # ax.set_ylabel('Linkage Dist.')
        # ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=9)
        # ax.xaxis.labelpad = 8
        ax.yaxis.labelpad = 8

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

        filename = 'cluster'
        texplot.show_or_save_plot(plt, default_filename=filename,
                                  transparent_background=transparent_bg,
                                  dpi=200, bbox_inches='tight', pad_inches=0,
                                  show_and_save=save, verbose=True)


# ============
# plot cluster
# ============

def plot_cluster(
        model,
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
    Implementation function for :func:`leaderbot.BaseModel.cluster`.
    """

    rnk = model.rank()
    scores = model.scores()
    scores = scores[rnk]

    names = np.array(model.agents)
    names = names[rnk]

    # Max rank
    if max_rank is None:
        m = model.n_agents
    else:
        m = max_rank

    # distance matrix
    D = _get_distance_matrix(model, scores, rnk)
    dist_matrix = D[:m, :][:, :m]

    # Competitor names
    name_max_length = 100

    labels = []
    for i in range(m):
        name = names[i]
        if len(name) > name_max_length:
            name = name[:name_max_length-3] + '...'
            labels.append(f'{name:>20}')
        else:
            labels.append(f'{name}')
    labels = np.array(labels)

    dist_matrix_condensed = squareform(dist_matrix, checks=True)
    linkage = sch.linkage(dist_matrix_condensed, method=method,
                          optimal_ordering=True)
    linkage = optimal_leaf_ordering(linkage, dist_matrix_condensed)

    colors = [fg_color, 'darkolivegreen', 'olive', 'chocolate',
              'firebrick']

    # Set up figure and polar axis
    rc = {
        'font.family': 'sans-serif',
        'text.usetex': latex,
        'text.latex.preamble':
            r'\usepackage{fixltx2e} ' +
            r'\usepackage[utf8]{inputenc} ' +
            r'\usepackage{amsmath} ' +
            r'\usepackage{amsfonts} ' +
            r'\usepackage{pifont}',
    }

    _plot_linear_cluster(ax, linkage, labels, color_threshold, colors,
                         bg_color, fg_color, rc, save, latex)
