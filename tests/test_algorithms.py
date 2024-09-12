#! /usr/bin/env python

# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import sys
import time
from leaderbot.data import load_data
from leaderbot.algorithms import BradleyTerry, BradleyTerryScaled, \
    BradleyTerryScaledR, BradleyTerryScaledRIJ, RaoKupper, RaoKupperScaled, \
    RaoKupperScaledR, RaoKupperScaledRIJ, Davidson, DavidsonScaled, \
    DavidsonScaledR, DavidsonScaledRIJ


# ===============
# test algorithms
# ===============

def test_algorithms():
    """
    A test for :mod:`leaderbot.algorithms` module.
    """

    data = load_data()

    algorithms = [
        BradleyTerry,
        BradleyTerryScaled,
        BradleyTerryScaledR,
        BradleyTerryScaledRIJ,
        RaoKupper,
        RaoKupperScaled,
        RaoKupperScaledR,
        RaoKupperScaledRIJ,
        Davidson,
        DavidsonScaled,
        DavidsonScaledR,
        DavidsonScaledRIJ
    ]

    for algorithm in algorithms:

        print(f'{algorithm.__name__:<21s} ...', end='', flush=True)

        t0 = time.time()
        alg = algorithm(data)

        if algorithm.__name__.endswith('ScaledRIJ'):
            method = 'L-BFGS-B'
        else:
            method = 'BFGS'

        alg.train(method=method)
        alg.inference()
        t1 = time.time() - t0

        print(f' passed in {t1:>5.1f} sec.', flush=True)


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(test_algorithms())
