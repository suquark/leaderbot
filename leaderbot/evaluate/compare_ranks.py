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


# =====================
# compute marker radius
# =====================

def _compute_marker_radius(ax, markersize, edgewidth=0):
    """
    Compute the visual marker radius in data units dynamically.

    Parameters
    ----------

    ax
        The Axes object containing the plot.

    markersize
        Marker size in points (for `plot`).

    edgewidth
        Edge width in points (optional).

    Returns
    -------

    radius
        Marker radius in data units.
    """

    # Marker size is diameter in points
    points_radius = markersize / 2

    # Convert to inches
    inches_radius = points_radius / 72

    # Add edge width to the radius
    inches_radius += edgewidth / 72

    # Get axes dimensions in inches
    bbox = ax.get_position()  # Axes bounding box in figure coordinates
    fig = ax.figure
    fig_width, fig_height = fig.get_size_inches()
    axes_width = bbox.width * fig_width
    axes_height = bbox.height * fig_height

    # Get axis limits and data range
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_data_range = np.diff(xlim)[0]
    y_data_range = np.diff(ylim)[0]

    # Calculate data units per inch for x and y
    x_data_per_inch = x_data_range / axes_width
    y_data_per_inch = y_data_range / axes_height

    return inches_radius, x_data_per_inch, y_data_per_inch


# =================
# trim line segment
# =================

def _trim_line_segment(x1, y1, x2, y2, inches_radius, x_data_per_inch,
                       y_data_per_inch):
    """
    Trims the endpoints of a line segment by a given radius.

    Parameters
    ----------

    x1, y1
        Coordinates of the start point.

    x2, y2
        Coordinates of the end point.

    inches_radius
        Marker radius in inches.

    x_data_per_inch
        Data units per inch for the x-axis.

    y_data_per_inch
        Data units per inch for the y-axis.

    Returns
    -------

    xt1, yt1, xt2, yt2
        Coordinates of the trimmed line segment.

    Notes
    -----

    The purpose of this function is to plot lines that its start and end is
    trimmed. This is because, if we want to make circle markers to be hollow
    inside, by trimming and plotting each segment separately (as in one plot),
    the lines wont show up inside the marker's circles.

    One option to achieve this is to plot the whole lines at once and use
    ``markercolor='none'``, but this shows the line inside the marker. To
    resolve this, another solution is to set ``markercolor='white'``, which
    hides the line inside the marker. But this won't make the hollow
    transparent. To make it transparent while not showing the line inside the
    hollow marker, the only option is to not have any line passing inside the
    marker area. As such, each segment of piecewise-line should be plotted
    separately while its start and end being trimmed by a specific amount.
    """

    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)

    # Avoid division by zero for degenerate segments
    if length == 0:
        return x1, y1, x2, y2

    # Scale trimming radius based on the line's slope and axis aspect ratio
    trim_radius_x = inches_radius * x_data_per_inch
    trim_radius_y = inches_radius * y_data_per_inch

    # Adjust trimming radii dynamically to respect the slope
    scale_factor = np.sqrt((dy / trim_radius_y)**2 + (dx / trim_radius_x)**2)
    trim_dx = dx / scale_factor
    trim_dy = dy / scale_factor

    # Apply trimming to both ends of the line
    return x1 + trim_dx, y1 + trim_dy, x2 - trim_dx, y2 - trim_dy


# =============
# compare ranks
# =============

def compare_ranks(
        models,
        order: list = None,
        rank_range: list = None,
        ax=None,
        bg_color: tuple = 'none',
        fg_color: tuple = 'black',
        mf_color: tuple = 'white',
        save: bool = False,
        latex: bool = False):
    """
    Compare ranking of various models.

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

    bg_color : str or tuple, default='none'
        Color of the background canvas. The default value of ``'none'`` means
        transparent.

    fg_color : str or tuple, default='black'
        Color of the axes and text.

    mf_color : str or tuple, default=None
        Marker face color. If `None, the ``bg_color`` is used

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
        >>> from leaderbot.models import BradleyTerry as BT
        >>> from leaderbot.models import RaoKupper as RK
        >>> from leaderbot.models import Davidson as DV

        >>> # Load data
        >>> data = lb.data.load()

        >>> # Create a list of models to compare
        >>> models = [
        ...     BT(data, k_cov=0),
        ...     BT(data, k_cov=3),
        ...     RK(data, k_cov=0, k_tie=0),
        ...     RK(data, k_cov=0, k_tie=1),
        ...     RK(data, k_cov=0, k_tie=3),
        ...     DV(data, k_cov=0, k_tie=0),
        ...     DV(data, k_cov=0, k_tie=1),
        ...     DV(data, k_cov=0, k_tie=3)
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
            fig = ax.figure

        n_agents = ranks.shape[1]
        n_models = ranks.shape[0]

        colors = [fg_color, 'firebrick', 'orangered', 'darkgoldenrod',
                  'royalblue', 'mediumblue']
        fontsize = 9

        # Settings for marker (all units are in "points", not data unit)
        markersize = 4.5
        markeredgewidth = 1.75

        # Compute the marker radius in data units
        # inches_radius, x_data_per_inch, y_data_per_inch = \
        #     _compute_marker_radius(ax_, markersize-markeredgewidth,
        #                            edgewidth=markeredgewidth)

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

            x = np.arange(n_models)
            y = np.array(ranks_over_models) - rank_range_[0]

            # Plot lines and markers both together (commended out in favor of
            # the following lines instead using trimming)
            if mf_color is None:
                if bg_color == 'none':
                    markerfacecolor = 'white'
                else:
                    markerfacecolor = bg_color
            else:
                markerfacecolor = mf_color
            ax_.plot(x, y, marker='o', markerfacecolor=markerfacecolor,
                     markeredgecolor=color, markeredgewidth=markeredgewidth,
                     markersize=markersize, label=f'Entity {entity}',
                     color=color)

            # Plot markers only
            # ax_.plot(x, y, 'o', markerfacecolor=bg_color,
            #          markeredgecolor=color, markeredgewidth=markeredgewidth,
            #          markersize=markersize, label=f'Entity {entity}',
            #          color=color)

            # Plot lines only, but segment by segment to apply line trims
            # Here, the start and end of each line segment is trimmed with the
            # amount of markersize, or that lines do not show inside marker's
            # hollow circle.
            # for i in range(x.size - 1):
            #     # Trim start and end of line segment
            #     xt1, yt1, xt2, yt2 = _trim_line_segment(
            #         x[i], y[i], x[i + 1], y[i + 1], inches_radius,
            #         x_data_per_inch, y_data_per_inch)
            #
            #     # Plot trimmed line segment
            #     ax_.plot([xt1, xt2], [yt1, yt2], color=color)

        agent_names = selected_models[0].agents

        for i in range(rank_range_[0], rank_range_[1]):
            entity = ranks[0, i]

            name = agent_names[ranks[0, i]]
            name_length = 17
            if len(name) > name_length:
                name = name[:(name_length-3)] + '...'

            ax_.text(-0.13, i - rank_range_[0] + 1, f'{name:>20s}',
                     ha='right',  va='center',
                     transform=ax_.get_yaxis_transform(), fontsize=fontsize,
                     color=fg_color)

            ax_.text(0.0, i - rank_range_[0] + 1, f'{i+1:>3d}', ha='right',
                     va='center', transform=ax_.get_yaxis_transform(),
                     fontsize=fontsize, color=fg_color)

        for i in range(n_models):
            ax_.text(i, 0, f'Model {order[i]+1}', ha='center', va='bottom',
                     rotation=90, fontsize=fontsize, color=fg_color)

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

        if ax is None:
            plt.tight_layout()

            texplot.show_or_save_plot(plt, default_filename='bump_chart',
                                      transparent_background=transparent_bg,
                                      dpi=200, show_and_save=save,
                                      verbose=True)
