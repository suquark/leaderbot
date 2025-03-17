# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# imports
# =======

from ._base_interface import BaseInterface
from ._bradleyterry_no_cov import BradleyTerryNoCov
from ._bradleyterry_factor_cov import BradleyTerryFactorCov
from ..data import DataType

__all__ = ['BradleyTerry']


# =============
# Bradley Terry
# =============

class BradleyTerry(BaseInterface):
    """
    Generalized Bradley-Terry model.

    Parameters
    ----------

    data : dict
        A dictionary of data that is provided by
        :func:`leaderbot.data.load`.

    k_cov : int, default=0
        Determines the structure of covariance in the model based on the
        following values:

        * ``None``: this means no covariance is used in the model, retrieving
          the original Bradley Terry model.
        * ``0``: this assumes covariance is a diagonal matrix.
        * positive integer: this assumes covariance is a diagonal plus
          low-rank matrix where the rank of low-rank approximation is
          ``k_cov``.

        See `Notes` below for further details.

    See Also
    --------

    RaoKupper
    Davidson

    Notes
    -----

    This class implements a generalization of the Bradley Terry model based on
    [1]_, incorporating covariance in the model.

    **Covariance Model:**

    This model utilizes a covariance matrix with diagonal plus low-rank
    structure of the form

    .. math::

        \\mathbf{\\Sigma} = \\mathbf{D} +
        \\mathbf{\\Lambda} \\mathbf{\\Lambda}^{\\intercal},

    where

    * :math:`\\mathbf{\\Sigma}` is an :math:`m \\times m` symmetric positive
      semi-definite covariance matrix where :math:`m` is the number of
      agents (competitors).
    * :math:`\\mathbf{D}`: is an :math:`m \\times m` diagonal matrix with
      non-negative diagonals.
    * :math:`\\mathbf{\\Lambda}`: is a full-rank
      :math:`m \\times k_{\\mathrm{cov}}` matrix where
      :math:`k_{\\mathrm{cov}}` is given by the input parameter ``k_cov``.

    If ``k_cov=None``, the covariance matrix is not used in the model,
    retrieving the original Bradley-Terry model [2]_. If ``k_cov=0``, the
    covariance model reduces to a diagonal matrix :math:`\\mathbf{D}`.

    **Tie Model:**

    The Bradley Terry model does not include the `tie` outcomes in the data.
    To consider `tie` outcomes, use :class:`leaderbot.models.RaoKupper` or
    :class:`leaderbot.models.Davidson` models instead.

    References
    ----------

    .. [1] Siavash Ameli, Siyuan Zhuang, Ion Stoica, and Michael W. Mahoney.
           `A Statistical Framework for Ranking LLM-Based Chatbots
           <https://openreview.net/pdf?id=rAoEub6Nw2>`__. *The Thirteenth
           International Conference on Learning Representations*, 2025.

    .. [2] Ralph A. Bradley and Milton E. Terry. `Rank Analysis of Incomplete
           Block Designs: I. The Method of Paired Comparisons.
           <https://doi.org/10.2307/2334029>`__. `Biometrika`, 39 (3/4),
           324-345, 1952.

    Attributes
    ----------

    x : np.ndarray
        A 2D array of integers with the shape ``(n_pairs, 2)`` where each row
        consists of indices ``[i, j]`` representing a match between a pair of
        agents with the indices ``i`` and ``j``.

    y : np.ndarray
        A 2D array of integers with the shape ``(n_pairs, 3)`` where each row
        consists of three counts ``[n_win, n_loss, n_ties]`` representing the
        frequencies of win, loss, and ties between agents ``i`` and ``j`` given
        by the corresponding row of the input array ``x``.

    agents : list
        A list of the length ``n_agents`` representing the name of agents
        (competitors).

    n_agents : int
        Number of agents (competitors).

    param : np.array, default=None
        The model parameters. This array is set once the model is trained.

    n_param : int
        Number of parameters

    k_cov : int
        Number of factors for matrix factorization.

    Methods
    -------

    loss
        Loss function of the model.

    train
        Train model parameters.

    infer
        Makes inference of match probabilities.

    predict
        Predict the output of a match between agents.

    fisher
        Observed Fisher information matrix.

    rank
        Return rank of the agents based on their score.

    leaderboard
        Print leaderboard table.

    marginal_outcomes
        Plot marginal probabilities and frequencies of win, loss, and tie.

    map_distance
        Visualize distance between agents using manifold learning projection.

    cluster
        Cluster competitors to performance tiers.

    scores
        Get scores

    plot_scores
        Plots scores versus rank.

    match_matrix
        Plot match matrices of win and tie counts of mutual matches.

    Examples
    --------

    .. code-block:: python

        >>> from leaderbot.data import load
        >>> from leaderbot.models import BradleyTerry

        >>> # Create a model
        >>> data = load()
        >>> model = BradleyTerry(data)

        >>> # Train the model
        >>> model.train()

        >>> # Make inference
        >>> prob = model.infer()
    """

    # ====
    # init
    # ====

    def __init__(
            self,
            data: DataType,
            k_cov: int = 0):
        """
        Constructor.
        """

        super().__init__()

        self._delegated_model_name = 'BradleyTerry'

        if k_cov is None:
            self._delegated_model = BradleyTerryNoCov(data)
        else:
            self._delegated_model = BradleyTerryFactorCov(
                data, n_cov_factors=k_cov)
