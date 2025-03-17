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
from ._davidson_no_cov import DavidsonNoCov
from ._davidson_factor_cov import DavidsonFactorCov
from ..data import DataType

__all__ = ['Davidson']


# ========
# Davidson
# ========

class Davidson(BaseInterface):
    """
    Generalized Davidson model.

    Parameters
    ----------

    data : dict
        A dictionary of data that is provided by
        :func:`leaderbot.data.load`.

    k_cov : int, default=0
        Determines the structure of covariance in the model based on the
        following values:

        * ``None``: this means no covariance is used in the model. Together
          with setting ``k_tie=0``, the original Davidson model is retrieved.
        * ``0``: this assumes covariance is a diagonal matrix.
        * positive integer: this assumes covariance is a diagonal plus
          low-rank matrix where the rank of low-rank approximation is
          ``k_cov``.

        See `Notes` below for further details.

    n_tie_factor : int, default=1
        Determines the rank of low-rank factor structure for modeling tie
        outcomes based on the following values:

        * ``0``: this assumes no low-rank factor model. Together with setting
          ``k_tie=0``, the original Davidson model is retrieved.
        * positive integer: this employs a low-rank structure for modeling tie
          outcomes with rank ``k_tie``.

        See `Notes` below for further details.

    See Also
    --------

    BradleyTerry
    RaoKupper

    Notes
    -----

    This class implements a generalization of the Davidson model based on [1]_,
    incorporating covariance and tie factor models.

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
    retrieving the original Bradley-Terry model [2]_ (along with setting
    ``k_tie=0``). If ``k_cov=0``, the covariance model reduces to a diagonal
    matrix :math:`\\mathbf{D}`.

    **Tie Model:**

    Modeling tie in Davidson model introduces a threshold parameter
    :math:`\\eta`. In generalized Davidson model, threshold parameter is
    instead modeled by the additive low-rank structure of the form

    .. math::

        \\mathbf{H} = \\begin{cases}
            \\mathbf{G} \\boldsymbol{\\Phi}^{\\intercal} +
            \\boldsymbol{\\Phi} \\mathbf{G}^{\\intercal},
            & 0< k_{\\mathrm{tie}} \\leq m \\\\
            \\eta \\mathbf{J} & k_{\\mathrm{tie}} = 0,
        \\end{cases}

    where

    * :math:`\\mathbf{H}` is an :math:`m \\times m` symmetric matrix where its
      elements represent pair-specific thresholds and :math:`m` is the number
      of agents (competitors).
    * :math:`\\mathbf{G}` is an :math:`m \\times k_{\\mathrm{tie}}` matrix of
      parameters of the full rank :math:`k_{\\mathrm{tie}}` given by the input
      argument ``k_tie``.
    * :math:`\\boldsymbol{\\Phi}` is an :math:`m \\times k_{\\mathrm{tie}}`
      orthonormal matrix of basis functions.
    * :math:`\\mathbf{J}` is an :math:`m \\times m` matrix of all ones.

    Setting ``k_tie = 0`` leads to a model with single tie threshold,
    retrieving the original Davidson model (along with setting
    ``k_cov=None``).

    A similar approach that also models tie outcomes is
    :class:`leaderbot.models.RaoKupper` model.

    **Best Practices for Setting Parameters:**

    The number of model parameters and training time scale with
    :math:`k_{\\mathrm{cov}}` and :math:`k_{\\mathrm{tie}}`. Depending on the
    dataset size, choosing too small or too large a value for these parameters
    can lead to under- or over-parameterization. In practice, moderate values
    of :math:`1 \\sim 10` often balance model fit, test accuracy, and training
    runtime efficiency.

    References
    ----------

    .. [1] Siavash Ameli, Siyuan Zhuang, Ion Stoica, and Michael W. Mahoney.
           `A Statistical Framework for Ranking LLM-Based Chatbots
           <https://openreview.net/pdf?id=rAoEub6Nw2>`__. *The Thirteenth
           International Conference on Learning Representations*, 2025.

    .. [2] Roger R. Davidson. `On Extending the Bradley-Terry Model to
           Accommodate Ties in Paired Comparison Experiments.
           <https://doi.org/10.2307/2283595>`__. `Journal of the American
           Statistical Association`, 65(329), 317–328, 1970.

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
        A list of the length ``n_agents`` representing the name of agents.

    n_agents : int
        Number of agents.

    param : np.array, default=None
        The model parameters. This array is set once the model is trained.

    n_param : int
        Number of parameters

    k_cov : int
        Number of factors for matrix factorization.

    n_tie_factor : int
        Number of factors used in tie parameters.

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
        >>> from leaderbot.models import Davidson

        >>> # Create a model
        >>> data = load()
        >>> model = Davidson(data)

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
            k_cov: int = 0,
            k_tie: int = 1):
        """
        Constructor.
        """

        super().__init__()

        self._delegated_model_name = 'Davidson'

        if k_cov is None:
            self._delegated_model = DavidsonNoCov(
                    data, n_tie_factors=k_tie)
        else:
            self._delegated_model = DavidsonFactorCov(
                data, n_cov_factors=k_cov, n_tie_factors=k_tie)
