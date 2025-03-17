.. _api:

API Reference
*************

Ranking Models
--------------

Classes to define statistical ranking models to train on data.

.. autosummary::
    :toctree: generated
    :caption: Models
    :recursive:
    :template: autosummary/class.rst

    leaderbot.models.BradleyTerry
    leaderbot.models.RaoKupper
    leaderbot.models.Davidson

Model Evaluation
----------------

Evaluate metrics for model selection, goodness of fit, and generalization performance.

.. autosummary::
    :toctree: generated
    :caption: Model Evaluation
    :recursive:
    :template: autosummary/member.rst

    leaderbot.evaluate.model_selection
    leaderbot.evaluate.goodness_of_fit
    leaderbot.evaluate.generalization
    leaderbot.evaluate.compare_ranks

Data
----

Load, convert, and split data.

.. autosummary::
    :toctree: generated
    :caption: Data
    :recursive:
    :template: autosummary/member.rst

    leaderbot.data.load
    leaderbot.data.convert
    leaderbot.data.split
    leaderbot.data.sample_whitelist

Types
-----

Data types (for internal docstring only).

.. autosummary::
    :toctree: generated
    :caption: Types
    :recursive:
    :template: autosummary/class.rst

    leaderbot.data.DataType
