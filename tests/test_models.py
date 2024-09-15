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
from leaderbot.data import load
from leaderbot.models import BradleyTerry, BradleyTerryScaled, \
    BradleyTerryScaledR, BradleyTerryScaledRIJ, RaoKupper, RaoKupperScaled, \
    RaoKupperScaledR, RaoKupperScaledRIJ, Davidson, DavidsonScaled, \
    DavidsonScaledR, DavidsonScaledRIJ


# ===========
# test models
# ===========

def test_models():
    """
    A test for :mod:`leaderbot.models` module.
    """

    data = load()

    Models = [
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

    for Model in Models:

        print(f'{Model.__name__:<21s} ...', end='', flush=True)

        t0 = time.time()
        model = Model(data)

        if Model.__class__.__name__.endswith('ScaledRIJ'):
            method = 'L-BFGS-B'
        else:
            method = 'BFGS'

        model.train(method=method)
        model.infer()
        model.predict()
        t1 = time.time() - t0

        print(f' passed in {t1:>5.1f} sec.', flush=True)


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(test_models())
