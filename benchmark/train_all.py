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

import os
import sys
import pickle
import leaderbot as lb
from copy import deepcopy
import time
import platform


# ==================
# get allocated cpus
# ==================

def _get_allocated_cpus():
    """
    Get num processors.
    """

    slurm_cpus = os.getenv('SLURM_CPUS_PER_TASK')
    slurm_ntasks = os.getenv('SLURM_NTASKS')

    if slurm_cpus:
        # SLURM_CPUS_PER_TASK is set, return that value
        return int(slurm_cpus)
    elif slurm_ntasks:
        # SLURM_NTASKS is set, use that value for number of tasks
        return int(slurm_ntasks)
    else:
        # Fall back to total number of available CPUs on the machine
        return os.cpu_count()


# =========
# train all
# =========

def train_all():
    """
    """

    tie = 'tie'
    # tie = 'both'
    data = lb.data.load(tie=tie)

    # use_whole_data = True
    use_whole_data = False

    if use_whole_data:
        # Do not split data. Use the whole data as training
        training_data = data
        test_data = data

    else:
        # Split data to training and test
        training_data, test_data = lb.data.split(data, test_ratio=0.1,
                                                 seed=20)

    # Create a copy of the data with no tie (for Bradley-Terry)
    training_data_no_tie = deepcopy(training_data)
    test_data_no_tie = deepcopy(test_data)

    training_data_no_tie['Y'][:, -1] = 0
    test_data_no_tie['Y'][:, -1] = 0

    models = [
        lb.models.BradleyTerry(training_data, k_cov=None),
        lb.models.BradleyTerry(training_data, k_cov=0),
        lb.models.BradleyTerry(training_data, k_cov=3),

        lb.models.BradleyTerry(training_data_no_tie, k_cov=None),
        lb.models.BradleyTerry(training_data_no_tie, k_cov=0),
        lb.models.BradleyTerry(training_data_no_tie, k_cov=3),

        lb.models.RaoKupper(training_data, k_cov=None, k_tie=0),
        lb.models.RaoKupper(training_data, k_cov=None, k_tie=1),
        lb.models.RaoKupper(training_data, k_cov=None, k_tie=10),
        lb.models.RaoKupper(training_data, k_cov=None, k_tie=20),

        lb.models.RaoKupper(training_data, k_cov=0, k_tie=0),
        lb.models.RaoKupper(training_data, k_cov=0, k_tie=1),
        lb.models.RaoKupper(training_data, k_cov=0, k_tie=10),
        lb.models.RaoKupper(training_data, k_cov=0, k_tie=20),

        lb.models.RaoKupper(training_data, k_cov=3, k_tie=0),
        lb.models.RaoKupper(training_data, k_cov=3, k_tie=1),
        lb.models.RaoKupper(training_data, k_cov=3, k_tie=10),
        lb.models.RaoKupper(training_data, k_cov=3, k_tie=20),

        lb.models.Davidson(training_data, k_cov=None, k_tie=0),
        lb.models.Davidson(training_data, k_cov=None, k_tie=1),
        lb.models.Davidson(training_data, k_cov=None, k_tie=10),
        lb.models.Davidson(training_data, k_cov=None, k_tie=20),

        lb.models.Davidson(training_data, k_cov=0, k_tie=0),
        lb.models.Davidson(training_data, k_cov=0, k_tie=1),
        lb.models.Davidson(training_data, k_cov=0, k_tie=10),
        lb.models.Davidson(training_data, k_cov=0, k_tie=20),

        lb.models.Davidson(training_data, k_cov=3, k_tie=0),
        lb.models.Davidson(training_data, k_cov=3, k_tie=1),
        lb.models.Davidson(training_data, k_cov=3, k_tie=10),
        lb.models.Davidson(training_data, k_cov=3, k_tie=20),
    ]

    wall_time = []
    proc_time = []

    # Train all. If model is ScaledRIJ, use l-BFGS-B, otherwise, use BFGS.
    method = 'BFGS'
    n = len(models)
    for i, model in enumerate(models):

        name = model.__class__.__name__
        print(f'[{i+1:>2d} / {n:>2d}] training {name:<21} ... ', end='',
              flush=True)

        wall_t0 = time.time()
        proc_t0 = time.process_time()

        model.train(method=method)

        wall_t1 = time.time()
        proc_t1 = time.process_time()

        wall_time.append(wall_t1 - wall_t0)
        proc_time.append(proc_t1 - proc_t0)

        # Hessian takes a significant amount of memory
        if (hasattr(model, "_result")) and (model._result is not None) and \
                ("hess_inv" in model._result):
            del model._result["hess_inv"]

        print('done.', flush=True)

    results = {
            'data': data,
            'training_data': training_data,
            'training_data_no_tie': training_data_no_tie,
            'test_data': test_data,
            'test_data_no_tie': test_data_no_tie,
            'models': models,
            'wall_time': wall_time,
            'proc_time': proc_time,
            'device': platform.processor(),
            'num_proc': _get_allocated_cpus(),
    }

    if use_whole_data:
        filename = 'models_train_full'
    else:
        filename = 'models_train_split'

    if tie == 'both':
        filename += '_both_ties'

    filename += '.pkl'

    with open(filename, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved to {filename}.')


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(train_all())
