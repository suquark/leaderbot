TODOs
=====

Interface Classes
-----------------

Combine similar classes to one and instead use class constructor to dispatch to
different methods. For example, these classes:

* Davidson
* DavidsonScaled
* DavidsonScaledR
* DavidsonScaledRIJ

Instead of these four classes, just create one interface class, "Davidson",
but in its instructor define the argument "cov" where:

* ``cov=None`` refers to Vanilla Davidson with no covariance
* ``cov='diag'`` refers to DavidsonScaled (as this is diagonal covariance)
* ``cov=1`` refers to DavidsonScaledR (as this is factor analysis of rank 1)
* ``cov='full'`` refers to DavidsonScaledRIJ (as this is full covariance)

Also, ``cov=k`` will be factor analysis of rank k. With this, we will only
have three classes:

* BradleyTerry(data, cov)
* RaoKupper(data, cov)
* Davidson(data, cov)

Documentation
-------------

* Add fig for rank and vis. (done).

Visualization
-------------

* Why in jupyter notebook, 3D plot is not centered? Use ``%matplotlib tk``
  to temporarily fix the plot canvas.

Model Evaluation
----------------

Prototype function: ``evaluate(models: list, print: bool, train: bool)``.

Data
----

Merge archived code to ``load_data`` by adding option to to get recent data.

ScaledRIJ Optimization
----------------------

All ScaledRIJ classes can not be optimized with ``BFGS``, rather, they only
work with ``L-BFGS-B``. Needs work. It might need constraint with the trace of
full ``S``, rather than its diagonal constraint.

Notes
=====

Terminologies
-------------

* Changed the name ``model`` to ``agent`` as this might conflate with the
  LLM models with the statistical model (likelihood model itself). ``agent``
  also represents chatbots.

* ``algorithm`` was changed to ``model``.

* ``weight`` to ``param``.

* Better to use ``match`` instead of ``battle`` or ``game`` in doc.

* 

Models
------

* In ``RaoKupper``, the parameter ``eta`` is contained to be positive using its
  absolute value. Also, we avoid overwriting ``w[-1]`` with the cutoff max
  value of 1e-3 (now 1e-2). The overwrite confuses optimizer. Rather, this
  should be done on a local copy of ``w[-1]`` (called eta). This fixed its
  stability.

* In rare cases, in ``RaoKupper``, when instead of ``np.abs(eta)``, if we use
  ``eta**2`` (to force it be positive) the following issue emerges: The values
  of eta become so large that ``double_sigmoid`` becomes exactly zero, hence,
  ``cross_entropy(p, q)`` becomes ``np.inf``. Since there are three cross
  entropy terms are added, two of them become ``-np.inf`` and one becomes
  ``+np.inf``. As such, adding minus and plus inf gives nan. The solution is
  to write another function where in case if one of these three terms gives inf
  (no matter plus or minus), always the sum of the tree term return inf (as
  loss function does not go to -inf). We do not need this solution at the
  moment as ``np.abs(eta)`` just works fine.

* In ``BradleyTerry``, ``train`` and ``infere`` overwrites the base class as a
  different iterative solution is available.
