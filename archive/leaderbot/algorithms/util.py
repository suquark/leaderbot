# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import numba
import numpy as np

__all__ = ['sigmoid', 'double_sigmoid', 'cross_entropy']


# =======
# sigmoid
# =======

@numba.jit(nopython=True)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ==============
# double sigmoid
# ==============

@numba.jit(nopython=True)
def double_sigmoid(a, b):
    return 1.0 / (1.0 + np.exp(-a) + np.exp(-b))


# =============
# cross entropy
# =============

@numba.jit(nopython=True)
def cross_entropy(p, q):
    """
    Cross entropy between two distributions or two frequencies, especially for
    the case when p is exactly zero integer and q is near zero floating point.

    Since p is integer, p == 0 comparison is fine.
    """

    h = p * np.log(q)
    h[p == 0] = 0
    return h
