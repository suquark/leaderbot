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
import texplot

__all__ = ['plot_cluster']


# ===========
# curved text
# ===========

def _curved_text(ax, center_angle, radius, text, color, spacing=5, flip=False):
    """
    Print text along a circular path
    """

    if not isinstance(text, str):
        text_ = list(text)
    else:
        text_ = text

    tot_angle = float(spacing) * float(len(text_))

    if flip:
        start_angle = center_angle + tot_angle / 2.0
    else:
        start_angle = center_angle - tot_angle / 2.0

    for i, char in enumerate(text_):
        if flip:
            angle = start_angle - float(i) * spacing
            rot = angle - 90
        else:
            angle = start_angle + float(i) * spacing
            rot = angle + 90

        ax.text(np.radians(angle), radius, rf'\textbf{{{char}}}', ha='center',
                va='center',  rotation=rot, rotation_mode='anchor',
                color=color)


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

def _plot_linear_cluster(ax, linkage, labels, color_threshold, colors, rc,
                         save, latex):
    """
    Plot cluster in Cartesian coordinates.
    """

    linkage[:, 2] = linkage[:, 2] / (np.max(linkage[:, 2]) * 1.02)

    with texplot.theme(use_latex=latex, font_scale=0.85):

        plt.rcParams.update(rc)
        if ax is None:
            _, ax = plt.subplots(figsize=(3, 15))

        sch.set_link_color_palette(colors)
        dendro = sch.dendrogram(linkage, no_plot=False, labels=labels,
                                orientation='left', no_labels=False,
                                color_threshold=color_threshold,
                                show_contracted=True,
                                above_threshold_color='black', ax=ax,
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

        filename = 'cluster'
        texplot.show_or_save_plot(plt, default_filename=filename,
                                  transparent_background=True,
                                  dpi=200, bbox_inches='tight', pad_inches=0,
                                  show_and_save=save, verbose=True)


# =====================
# plot circular cluster
# =====================

def _plot_circular_cluster(ax, dist_matrix, linkage, labels, color_threshold,
                           colors, tier_label, rc, link_distance_pow, save,
                           latex):
    """
    Plots cluster in polar coordinates.
    """

    linkage[:, 2] = linkage[:, 2] / (np.max(linkage[:, 2]))

    sch.set_link_color_palette(colors)
    dendro = sch.dendrogram(linkage, no_plot=True, labels=labels,
                            orientation='top', no_labels=False,
                            color_threshold=color_threshold,
                            show_contracted=True,
                            above_threshold_color='black', leaf_rotation=0)

    indices = dendro['leaves']

    with texplot.theme(use_latex=latex, font_scale=1):
        plt.rcParams.update(rc)
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 7),
                                 subplot_kw={'projection': 'polar'})
        else:
            if ax.name != 'polar':
                raise ValueError('"ax" is not a polar Axes. Please pass an ' +
                                 'Axes created with projection="polar".')

        # Calculate angles and distances for each branch
        num_leaves = len(dendro['ivl'])
        angles = np.linspace(0, 2 * np.pi, num_leaves, endpoint=False)

        min_x, max_x = np.inf, -np.inf
        min_y, max_y = np.inf, -np.inf

        for i, (xs, ys) in enumerate(zip(dendro['icoord'], dendro['dcoord'])):
            xs = np.array(xs) / 10.0
            ys = np.array(ys)
            if min(xs) < min_x:
                min_x = min(xs)
            if max(xs) > max_x:
                max_x = max(xs)
            if min(ys) < min_y:
                min_y = min(ys)
            if max(ys) > max_y:
                max_y = max(ys)

        n_data = dist_matrix.shape[0]

        for i, (xs, ys) in enumerate(zip(dendro['icoord'], dendro['dcoord'])):
            xs = np.array(xs) / 10.0
            ys = (np.array(ys))**link_distance_pow

            t = (n_data / (n_data + 1)) * 2.0 * np.pi * (xs - min_x) / \
                (max_x - min_x)
            r = (max_y - ys) / (max_y - min_y)

            # Draw the radial line segments
            t_rng = np.linspace(t[1], t[2], 40)
            r_rng = np.ones_like(t_rng) * r[1]
            ax.plot([t[0], t[1]], [r[0], r[1]], color=dendro['color_list'][i])
            ax.plot([t[2], t[3]], [r[2], r[3]], color=dendro['color_list'][i])
            ax.plot(t_rng, r_rng, color=dendro['color_list'][i])

        # Place labels around the circumference
        for index, angle, label, color in zip(indices, angles, dendro['ivl'],
                                              dendro['leaves_color_list']):
            if index + 1 == 100:
                ax.text(angle, 1.035, f'{index+1}', ha='left', va='center',
                        rotation=np.degrees(angle), color=color,
                        rotation_mode='anchor', fontsize=10)
            else:
                ax.text(angle, 1.035, f'{index+1}', ha='left', va='center',
                        rotation=np.degrees(angle), color=color,
                        rotation_mode='anchor')
            ax.text(angle, 1.14, label, ha='left', va='center',
                    rotation=np.degrees(angle), color=color,
                    rotation_mode='anchor')

        # Add text for tier of branches
        if tier_label:
            ax.text(5 * 2*np.pi/360, 0.36, r'\textbf{Tier I', rotation=2)
            _curved_text(ax, -144, 0.22, list('Tier ') + ['II'],
                         color='black', spacing=9)
            _curved_text(ax, 108, 0.37,
                         list('Tier ') + ['II'] + [r'\textsubscript{A}'],
                         color='black', spacing=5, flip=True)
            _curved_text(ax, -56, 0.58,
                         list('Tier ') + ['II'] + [r'\textsubscript{B}'],
                         color='black', spacing=3.5, flip=False)
            _curved_text(ax, 55, 0.62,
                         list('Tier ') + ['II'] + [r'\textsubscript{A}'] +
                         [r'\textsubscript{1}'],
                         color=colors[1], spacing=3.2, flip=True)
            _curved_text(ax, 158, 0.66,
                         list('Tier ') + ['II'] + [r'\textsubscript{A}'] +
                         [r'\textsubscript{2}'],
                         color=colors[2], spacing=3.2, flip=True)
            _curved_text(ax, -96, 0.71,
                         list('Tier ') + ['II'] + [r'\textsubscript{B}'] +
                         [r'\textsubscript{1}'],
                         color=colors[3], spacing=3.2, flip=False)
            _curved_text(ax, -21, 0.7,
                         list('Tier ') + ['II'] + [r' \textsubscript{B}'] +
                         [r'\textsubscript{2}'],
                         color=colors[4], spacing=3.2, flip=False)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.spines['polar'].set_visible(False)

        filename = 'cluster'
        texplot.show_or_save_plot(plt, default_filename=filename,
                                  transparent_background=True,
                                  dpi=200, bbox_inches='tight', pad_inches=0,
                                  show_and_save=save, verbose=True)


# ============
# plot cluster
# ============

def plot_cluster(
        model,
        ax=None,
        max_rank: int = None,
        layout: str = 'circular',
        tier_label: bool = False,
        method: str = 'complete',
        color_threshold: float = 0.15,
        link_distance_pow: float = 0.4,
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
    if layout == 'circular':
        name_max_length = 23
    else:
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

    if layout == 'circular':
        colors = ['black', 'darkgreen', 'limegreen', 'darkorange', 'red']
    else:
        colors = ['black', 'darkolivegreen', 'olive', 'chocolate', 'firebrick']

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

    if layout == 'circular':
        _plot_circular_cluster(ax, dist_matrix, linkage, labels,
                               color_threshold, colors, tier_label, rc,
                               link_distance_pow, save, latex)
    else:
        _plot_linear_cluster(ax, linkage, labels, color_threshold, colors, rc,
                             save, latex)
