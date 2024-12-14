# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# imports
# =======

__all__ = ['snap_limits_to_ticks']


# ===================
# snap limit to ticks
# ===================

def snap_limits_to_ticks(
        ax,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
        eps: float = 0.05,
        tol: float = 0.02):
    """
    Snaps x, y, and z limits to grid lines for 3D plots.
    """

    # Calculate axis ranges with padding
    dx = x_max - x_min
    dy = y_max - y_min
    dz = z_max - z_min

    x_min = x_min - dx * eps
    x_max = x_max + dx * eps
    y_min = y_min - dy * eps
    y_max = y_max + dy * eps
    z_min = z_min - dz * eps
    z_max = z_max + dz * eps

    # Get current tick values
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    z_ticks = ax.get_zticks()

    # Snap limits to the nearest tick if within tolerance
    x_min = min(x_ticks, key=lambda tick: abs(tick - x_min)) \
        if any(abs(tick - x_min) < tol * dx for tick in x_ticks) \
        else x_min

    x_max = min(x_ticks, key=lambda tick: abs(tick - x_max)) \
        if any(abs(tick - x_max) < tol * dx for tick in x_ticks) \
        else x_max

    y_min = min(y_ticks, key=lambda tick: abs(tick - y_min)) \
        if any(abs(tick - y_min) < tol * dy for tick in y_ticks) \
        else y_min

    y_max = min(y_ticks, key=lambda tick: abs(tick - y_max)) \
        if any(abs(tick - y_max) < tol * dy for tick in y_ticks) \
        else y_max

    z_min = min(z_ticks, key=lambda tick: abs(tick - z_min)) \
        if any(abs(tick - z_min) < tol * dz for tick in z_ticks) \
        else z_min

    z_max = min(z_ticks, key=lambda tick: abs(tick - z_max)) \
        if any(abs(tick - z_max) < tol * dz for tick in z_ticks) \
        else z_max

    # Apply the limits
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(y_min, y_max)
    ax.set_zlim3d(z_min, z_max)
