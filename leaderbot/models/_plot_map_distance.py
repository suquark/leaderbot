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
from ._plot_util import snap_limits_to_ticks
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import texplot
from sklearn.manifold import MDS
from sklearn.decomposition import KernelPCA
from adjustText import adjust_text

__all__ = ['plot_map_distance']


# =================
# plot map distance
# =================

def plot_map_distance(
        model,
        ax=None,
        cmap=None,
        max_rank: int = None,
        method: str = 'kpca',
        dim: str = '3d',
        sign: tuple = None,
        save: bool = False,
        latex: bool = False):
    """
    Implementation function for :func:`leaderbot.BaseModel.map_distance`.
    """

    if model.param is None:
        raise RuntimeError('train model first.')

    # Check input arguments
    if max_rank is None:
        max_rank = model.n_agents
    elif max_rank > model.n_agents:
        raise ValueError('"max_rank" can be at most equal to the number ' +
                         ' of agents.')

    if method not in ['kpca', 'mds']:
        raise ValueError('Invalid method.')

    if (dim not in ['2d', '3d']) and (not isinstance(dim, tuple)):
        raise ValueError('Invalid dimension.')

    if sign is not None:
        if not isinstance(sign, tuple):
            raise ValueError('"sign" should be a tuple.')

        if isinstance(dim, tuple) and (len(dim) != len(sign)):
            raise ValueError('Length of "dim" and "sign" tuples should '
                             'be the same.')
        elif (dim == '2d') and (len(sign) != 2):
            raise ValueError('Length of "sign" should be 2.')
        elif (dim == '3d') and (len(sign) != 3):
            raise ValueError('Length of "sign" should be 3.')

    # Compute distance matrix
    D = model._distance_matrix()

    # Compute visualization points
    if method == 'mds':

        # MDS method
        mds = MDS(n_components=3, dissimilarity='precomputed',
                  random_state=42)
        points = mds.fit_transform(D)

        # 3D visualization settings
        if (dim == '3d') or (isinstance(dim, tuple) and len(dim) == 3):
            elev, azim, roll = 8, 115, 0

    elif method == 'kpca':

        # Kernel PCA
        gamma = 1e-4  # You might need to adjust gamma based on your data
        kernel_matrix = np.exp(-gamma * D ** 2)
        kpca = KernelPCA(n_components=3, kernel='precomputed')
        points = kpca.fit_transform(kernel_matrix)

        # 3D visualization settings
        if (dim == '3d') or (isinstance(dim, tuple) and len(dim) == 3):
            elev, azim, roll = 8, 145, 0

    # Scores are the x_i, x_j parameters across all models
    score = model.param[:model.n_agents]
    rank_ = np.argsort(score)[::-1]
    rank_ = rank_[:max_rank]

    # Reorder variables based on rank
    points_ranked = points[rank_]
    agents_ranked = np.array(model.agents)[rank_]
    score_ranked = score[rank_]
    colors = np.linspace(0.0, 1.0, max_rank)

    # Cut long names
    max_length = 20
    for i in range(agents_ranked.size):
        if len(agents_ranked[i]) > max_length:
            agents_ranked[i] = agents_ranked[i][:max_length-3] + '...'

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

        fontsize = 10

        if cmap is None:
            cmap = 'turbo_r'

        if (dim == '2d') or (isinstance(dim, tuple) and len(dim) == 2):

            sizes = [5000 * (score_ranked[i] - score_ranked[max_rank-1]) +
                     100 * score_ranked[max_rank-1]
                     for i in range(max_rank)]

            if isinstance(dim, tuple):
                x_ = points_ranked[:, dim[0]]
                y_ = points_ranked[:, dim[1]]
            else:
                x_ = points_ranked[:, 0]
                y_ = points_ranked[:, 1]

            if sign is not None:
                x_ = x_ * np.sign(sign[0])
                y_ = y_ * np.sign(sign[1])

            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 8))
            else:
                fig = ax.get_figure()

            sc = ax.scatter(x_, y_, s=sizes, c=colors, cmap=cmap,
                            alpha=0.8, edgecolor=(0.0, 0.0, 0.0, 0.0),
                            linewidth=0.5)

            texts = [ax.text(x_[i], y_[i], agents_ranked[i],
                             fontsize=fontsize, ha='center', va='center')
                     for i in range(len(agents_ranked))]

            # Adjust text to avoid overlaps
            adjust_text(texts, objects=sc, time_lim=10,
                        ensure_inside_axes=True,
                        arrowprops=dict(arrowstyle='->',
                                        color=(0.2, 0.2, 0.2),
                                        lw=0.7))

            # ax.set_aspect('equal', adjustable='box')
            # ax.set_xlim([-0.021, 0])
            # ax.set_ylim([-0.005, 0.0125])
            ax.set_xlabel(r'$\xi_1$')
            ax.set_ylabel(r'$\xi_2$')
            # ax.set_title(f'{method_name} in {dim_name}')

            ax.set_title('Multidimensional Scaling')

            norm = Normalize(vmin=score_ranked.min(),
                             vmax=score_ranked.max())
            sm = ScalarMappable(cmap=cmap.reversed(), norm=norm)
            sm.set_array(score_ranked)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2.2%", pad=0.1)
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label('Score')

        elif (dim == '3d') or (isinstance(dim, tuple) and len(dim) == 3):

            sizes = [5000 * (score_ranked[i] - score_ranked[max_rank-1]) +
                     100 * score_ranked[max_rank-1]
                     for i in range(max_rank)]

            if isinstance(dim, tuple):
                x_ = points_ranked[:, dim[0]]
                y_ = points_ranked[:, dim[1]]
                z_ = points_ranked[:, dim[2]]
            else:
                x_ = points_ranked[:, 0]
                y_ = points_ranked[:, 1]
                z_ = points_ranked[:, 2]

            if sign is not None:
                x_ = x_ * np.sign(sign[0])
                y_ = y_ * np.sign(sign[1])
                z_ = z_ * np.sign(sign[2])

            if ax is None:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(projection='3d')

            ax.set_proj_type('persp', focal_length=0.17)

            ax.scatter(x_, y_, z_, s=sizes, c=colors, cmap=cmap,
                       alpha=0.8, edgecolor=(0.0, 0.0, 0.0, 0.0),
                       linewidth=0.5)

            for i, name in enumerate(agents_ranked[:]):
                ax.text(x_[i], y_[i], z_[i], name, fontsize=fontsize,
                        ha='center', va='center')

            ax.view_init(elev=elev, azim=azim, roll=roll)

            # Remove tick labels
            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.set_zticks([])

            # ax.grid(False)

            ax.set_xlabel(r'$\xi_1$')
            ax.set_ylabel(r'$\xi_2$')
            ax.set_zlabel(r'$\xi_3$')
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
            ax.xaxis.pane.set_linewidth(1.5)
            ax.yaxis.pane.set_linewidth(1.5)
            ax.zaxis.pane.set_linewidth(1.5)

            x_min = np.min(x_)
            x_max = np.max(x_)
            y_min = np.min(y_)
            y_max = np.max(y_)
            z_min = np.min(z_)
            z_max = np.max(z_)

            snap_limits_to_ticks(ax, x_min, x_max, y_min, y_max, z_min,
                                 z_max, eps=0.05, tol=0.02)

            ax.set_box_aspect(aspect=None, zoom=1)
            plt.subplots_adjust(left=0.0, right=0.9, top=1.0, bottom=0.05)

        # plt.tight_layout()

        if method == 'mds':
            filename = 'mds'
        elif method == 'kpca':
            filename = 'kpca'

        texplot.show_or_save_plot(plt, default_filename=filename,
                                  transparent_background=False,
                                  dpi=200, bbox_inches=None,
                                  show_and_save=save, verbose=True)
