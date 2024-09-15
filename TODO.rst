TODOs
=====

Interface Classes
-----------------

Combine similar classes to one class and instead use class constructor to
dispatch to different methods. For example, these classes:

* ``Davidson``
* ``DavidsonScaled``
* ``DavidsonScaledR``
* ``DavidsonScaledRIJ``

Instead of these four classes, just create one interface class, ``Davidson``,
but in its instructor define the argument ``cov`` where:

* ``cov=None`` refers to Vanilla ``Davidson`` with no covariance
* ``cov='diag'`` refers to ``DavidsonScaled`` (this is diagonal covariance)
* ``cov=1`` refers to ``DavidsonScaledR`` (this is factor analysis of rank 1)
* ``cov='full'`` refers to ``DavidsonScaledRIJ`` (this is full covariance)

Also, ``cov=k`` will be factor analysis of rank k. With this, we will only
have three classes:

* ``BradleyTerry(data, cov)``
* ``RaoKupper(data, cov)``
* ``Davidson(data, cov)``

Documentation
-------------

* Font size in ``kpca.png`` plot is too small.

* In ``README.rst``, why the link to ``CONTRIBUTING.rst`` is not properly shown
  in GitHub?

Visualization
-------------

* Why in Jupyter notebook, 3D plot is not centered? Use ``%matplotlib tk``
  to temporarily fix the plot canvas.

* Add win-table from archive codes.

Model Evaluation
----------------

Prototype function: ``evaluate(models: list, print: bool, train: bool)``. Not
fully implemented yet.

``ScaledRIJ`` Optimization
--------------------------

All ``*ScaledRIJ`` classes can not be optimized with ``BFGS``, rather, they
only work with ``L-BFGS-B``. Needs work. It might need constraint with the
trace of full ``S``, rather than its diagonal constraint.

Package Naming
--------------

* leaderbot
* arena (taken)
* botarena
* chatbotarena
* chatbot_arena

Notes
=====

Terminologies
-------------

* Changed the name ``model`` to ``agent`` as this might conflate the LLM models
  with the statistical model (likelihood model itself). The word ``agent`` also
  represents chatbots.

* ``algorithm`` was changed to ``model``.

* ``weight`` to ``param``.

* Better to use ``match`` instead of ``battle`` or ``game`` in the
  documentation.

Models
------

* In ``RaoKupper``, the parameter ``eta`` should be constrained to be positive
  to make sure ``p_tie`` remains positive. This can be done using its
  absolute value.

  Also, we avoid overwriting ``w[-1]`` with the cutoff max value of ``1e-3``
  (now changed to ``1e-2``). The overwrite confuses optimizer. Rather, this
  should be done on a local copy of ``w[-1]`` (called ``eta``). This fixed its
  stability.

* In rare cases, in ``RaoKupper``, when instead of ``np.abs(eta)``, if we use
  ``eta**2`` (to force it be positive) the following issue emerges:

  The values of ``eta`` become so large that ``double_sigmoid`` becomes exactly
  zero, hence, ``cross_entropy(p, q)`` becomes ``np.inf``. Since there are
  three cross entropy terms being added, and two of them become ``-np.inf``
  while the other one becomes ``+np.inf``. As such, adding minus and plus
  ``inf`` raises to ``nan`` issue.

  The solution is to write another function where in case if one of these three
  terms gives ``inf`` (no matter plus or minus), always the sum of the tree
  terms return ``inf`` (since loss function does not go to ``-inf``).

  We do not need this solution at the moment as ``np.abs(eta)`` just works
  fine.

* In ``BradleyTerry``, ``train`` and ``infere`` overwrites the base class as a
  different iterative solution is available.
