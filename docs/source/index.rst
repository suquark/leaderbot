.. module:: leaderbot

|project| Documentation
***********************

|project| is a python package that provides a **leader**\ board for
chat\ **bot**\ s based on `Chatbot Arena <https://lmarena.ai/>`_ project.

.. .. grid:: 4
..
..     .. grid-item-card:: Install
..         :link: install
..         :link-type: ref
..         :text-align: center
..         :class-card: custom-card-link
..
..     .. grid-item-card:: User Guide
..         :link: user_guide
..         :link-type: ref
..         :text-align: center
..         :class-card: custom-card-link
..
..     .. grid-item-card:: API reference
..         :link: api
..         :link-type: ref
..         :text-align: center
..         :class-card: custom-card-link
..
..     .. grid-item-card:: Publications
..         :link: index_publications
..         :link-type: ref
..         :text-align: center
..         :class-card: custom-card-link

Install
=======

Install with ``pip`` from `PyPI <https://pypi.org/project/leaderbot/>`_:

.. prompt:: bash
    
    pip install leaderbot

Alternatively, clone the source code and install with

.. prompt:: bash
   
    cd source_dir
    pip install .

Quick Usage
===========

The package provides several statistical models (see
:ref:`API reference <api>` for details). In the example below, we use
:class:`leaderbot.Davidson` class to build a model. However, working with
other models is similar.

Create and Train a Model
------------------------

.. code-block:: python

    >>> from leaderbot.data import load_data
    >>> from leaderbot.models import Davidson

    >>> # Create a model
    >>> data = load_data()
    >>> model = Davidson(data)

    >>> # Train the model
    >>> model.train()

Leaderboard Table
-----------------

To print leaderboard table of the chatbot agents, use
:func:`leaderbot.Davidson.rank` function:

.. code-block:: python

    >>> # Leaderboard rank and plot
    >>> model.rank(max_rank=30, plot=True)

The above code prints the table below:

.. literalinclude:: _static/data/rank.txt
    :language: none

.. image:: _static/images/plots/rank.png
    :align: center
    :class: custom-dark

Visualize Correlation
---------------------

The correlation of the chatbot performances can be visualized with
:func:`leaderbot.Davidson.visualize` using various methods. Here is an
example with the Kernel PCA method:

.. code-block:: python

    >>> # Plot kernel PCA
    >>> model.visualize(max_rank=50)

The above code produces plot below demonstrating the Kernel PCA projection on
three principal axes:

.. image:: _static/images/plots/kpca.png
    :align: center
    :class: custom-dark

Make Inference and Prediction
-----------------------------

Once a model is trained, you can make inference on the probabilities of win,
loss, or tie for a pair of agents using :func:`leaderbot.Davidson.infer`
method:

.. code-block:: python

    >>> # Create a list of three matches using pairs of indices of agents
    >>> matches = zip((0, 1, 2), (1, 2, 0))

    >>> # Make inference
    >>> prob = model.infer(matches)

    >>> # Make prediction
    >>> pred = model.predict(mathces)

Model Evaluation
----------------

Compare the performance of multiple models using :func:`leaderbot.evaluate`
function:

.. code-block:: python

    >>> # Create a list of models
    >>> model1 = leaderbot.BradleyTerryScaled(data)
    >>> model2 = leaderbot.RaoKupperScaled(data)
    >>> model3 = leaderbot.DavidsonScaled(data)
    >>> models = [model1, model2, model3]

    >>> # Train models
    >>> for model in models:
    ...     model.train()

    >>> # Evaluate models
    >>> leaderbot.evaluate(models)

The above model evaluation performs the analysis of the goodness of fit using
Bayesian and Akaike information criteria, and KL and Jensen-Shannon
divergences.

API Reference
=============

Check the list of functions, classes, and modules of |project| with their
usage, options, and examples.

.. toctree::
    :maxdepth: 2
   
    API Reference <api>

Test
====

You may test the package with `tox <https://tox.wiki/>`__:

.. prompt:: bash

    cd source_dir
    tox

Alternatively, test with `pytest <https://pytest.org>`__:

.. prompt:: bash

    cd source_dir
    pytest

How to Contribute
=================

We welcome contributions via GitHub's pull request. Developers should review
our :ref:`Contributing Guidelines <contribute>` before submitting their code.
If you do not feel comfortable modifying the code, we also welcome feature
requests and bug reports.

.. _index_publications:

.. Publications
.. ============
..
.. For information on how to cite |project|, publications, and software
.. packages that used |project|, see:

License
=======

This project uses a BSD 3-clause license in hopes that it will be accessible to
most projects. If you require a different license, please raise an issue and we
will consider a dual license.

.. |pypi| image:: https://img.shields.io/pypi/v/leaderbot
.. |traceflows-light| image:: _static/images/icons/logo-leaderbot-light.svg
   :height: 23
   :class: only-light
.. |traceflows-dark| image:: _static/images/icons/logo-leaderbot-dark.svg
   :height: 23
   :class: only-dark
