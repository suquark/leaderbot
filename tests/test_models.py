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
from leaderbot.models import BradleyTerry, RaoKupper, Davidson


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
        RaoKupper,
        Davidson,
    ]

    # k_covs = [None, 0, 1, 'full']
    k_covs = [None, 0, 1]
    k_ties = [0, 1]

    for Model in Models:
        for k_cov in k_covs:

            if Model.__name__.startswith('BradleyTerry'):
                k_ties_ = [None]
            else:
                k_ties_ = k_ties

            for k_tie in k_ties_:

                print(f'{Model.__name__:<12s} | ' +
                      f'k_cov: {str(k_cov):>4s} | ' +
                      f'k_tie: {str(k_tie):>4s} | ...',
                      end='', flush=True)

                if Model.__name__.startswith('BradleyTerry'):
                    model = Model(data, k_cov=k_cov)
                else:
                    model = Model(data, k_cov=k_cov, k_tie=k_tie)

                if k_cov == 'full':
                    method = 'L-BFGS-B'
                else:
                    method = 'BFGS'

                t0 = time.time()
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
