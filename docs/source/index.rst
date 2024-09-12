.. module:: leaderbot

|project| Documentation
***********************

|project| is a python package that provides **leader**\ board for
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

API Reference
=============

Check the list of functions, classes, and modules of |project| with their
usage, options, and examples.

.. toctree::
    :maxdepth: 2
   
    API Reference <api>

Quick Usage
===========

In the example below, we use :class:`leaderbot.DavidsonSaled` class to build a
model. Users can choose from many other models provided in the
:ref:`API <api>`.

Create and Train a Model
------------------------

.. code-block:: python

    >>> from leaderbot.data import load_data
    >>> from leaderbot.models import DavidsonScaled

    >>> # Create a model
    >>> data = load_data()
    >>> model = DavidsonScaled(data)

    >>> # Train the model
    >>> model.train()

Leaderboard Table
-----------------

To print leaderboard table of the LLM agents, use
:func:`leaderbot.BaseModel.rank` function:

.. code-block:: python

    >>> # Leaderboard rank and plot
    >>> model.rank(max_rank=30, plot=True)

The above code provides the text output and plot below.

.. literalinclude:: _static/data/rank.txt
    :language: none

.. image:: _static/images/plots/rank.png
    :align: center
    :class: custom-dark

Visualize Results
-----------------

The results can be visualized with :func:`leaderboard.BaseModel.visualize`
using various methods. Here is an example using Kernel PCA method:

.. code-block:: python

    >>> # Plot kernel PCA
    >>> model.visualize(max_rank=50)

The above code produces plot below.

.. image:: _static/images/plots/kpca.png
    :align: center
    :class: custom-dark

Make Inference and Prediction
-----------------------------

Once a model is trained, you can make inference on the probabilities of win,
loss, or tie for a pair of agents using :func:`leaderbot.BaseModel.infer`
method:

.. code-block:: python

    >>> # Create a list of matches using indices of agents
    >>> matches = zip((0, 1, 2), (1, 2, 0))

    >>> # Make inference
    >>> p_win, p_loss, p_tie = model.infer(matches)

    >>> # Make prediction
    >>> pred = model.predict(mathces)

Model Evaluation
----------------

If you created multiple models, you can compare their performance using
:func:`leaderbot.evaluate` function:


.. code-block:: python

    >>> # Create a list of models
    >>> model1 = leaderbot.BradleyTerry(data)
    >>> model2 = leaderbot.RaoKupper(data)
    >>> model3 = leaderbot.Davidson(data)
    >>> models = [model1, model2, model3]

    >>> # Train models
    >>> for model in models:
    ...     model.train()

    >>> # Evaluate models
    >>> leaderbot.evaluate(models)

The above provides several analysis of the goodness of fit of the modes,
including loss, Bayesian and Akaike information criteria, KL and
Jensen-Shannon divergences and


Test
====

You may test the package with `tox <https://tox.wiki/>`__:

.. prompt:: bash

    cd source_dir
    tox

How to Contribute
=================

We welcome contributions via GitHub's pull request. If you do not feel
comfortable modifying the code, we also welcome feature requests and bug
reports.

.. _index_publications:

.. Publications
.. ============
..
.. For information on how to cite |project|, publications, and software
.. packages that used |project|, see:

License
=======

This project uses a BSD 3-clause license in hopes that it will be accessible
to most projects. If you require a different license, please raise an issue and
we will consider a dual license.

.. |pypi| image:: https://img.shields.io/pypi/v/leaderbot
.. |traceflows-light| image:: _static/images/icons/logo-leaderbot-light.svg
   :height: 23
   :class: only-light
.. |traceflows-dark| image:: _static/images/icons/logo-leaderbot-dark.svg
   :height: 23
   :class: only-dark
