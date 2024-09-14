.. image:: docs/source/_static/images/icons/logo-leaderbot-light.png
    :align: left
    :width: 230
    :class: custom-dark

*leaderbot* is a python package that provides a **leader**\ board for
chat\ **bot**\ s based on `Chatbot Arena <https://lmarena.ai/>`_ project.

Install
=======

Install with ``pip`` from `PyPI <https://pypi.org/project/leaderbot/>`_:

.. code-block::

    pip install leaderbot

Alternatively, clone the source code and install with

.. code-block::

    cd source_dir
    pip install .

Build Documentation
===================

.. code-block::

    cd docs
    make clean html

The documentation can be viewed at ``/docs/build/html/index.html``, which
includes the `API` reference of classes and functions with their usage.

Quick Usage
===========

The package provides several statistical models (see API reference for
details). In the example below, we use ``leaderbot.Davidson`` class to build a
model. However, working with other models is similar.

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
``leaderbot.Davidson.rank`` function:

.. code-block:: python

    >>> # Leaderboard rank and plot
    >>> model.rank(max_rank=30, plot=True)

The above code prints the table below:

::

    +---------------------------+--------+--------+---------------+---------------+
    |                           |        |    num |   observed    |   predicted   |
    | rnk  agent                |  score |  match | win loss  tie | win loss  tie |
    +---------------------------+--------+--------+---------------+---------------+
    |   1. chatgpt-4o-latest    | +0.221 |  11798 | 53%  23%  24% | 55%  25%  20% |
    |   2. gemini-1.5-pro-ex... | +0.200 |  16700 | 51%  26%  23% | 52%  27%  20% |
    |   3. gpt-4o-2024-05-13    | +0.181 |  66560 | 51%  26%  23% | 52%  28%  20% |
    |   4. gpt-4o-mini-2024-... | +0.171 |  15929 | 46%  29%  25% | 48%  31%  21% |
    |   5. claude-3-5-sonnet... | +0.170 |  40587 | 47%  31%  22% | 48%  32%  21% |
    |   6. gemini-advanced-0514 | +0.167 |  44319 | 49%  29%  22% | 50%  30%  21% |
    |   7. llama-3.1-405b-in... | +0.161 |  15680 | 44%  32%  24% | 45%  34%  21% |
    |   8. gpt-4o-2024-08-06    | +0.159 |   7796 | 43%  32%  25% | 45%  34%  21% |
    |   9. gemini-1.5-pro-ap... | +0.159 |  57941 | 47%  31%  22% | 48%  32%  21% |
    |  10. gemini-1.5-pro-ap... | +0.156 |  48381 | 52%  28%  20% | 52%  28%  20% |
    |  11. athene-70b-0725      | +0.149 |   9125 | 43%  35%  22% | 43%  36%  21% |
    |  12. gpt-4-turbo-2024-... | +0.148 |  73106 | 47%  29%  24% | 49%  31%  21% |
    |  13. mistral-large-2407   | +0.147 |   9309 | 41%  35%  25% | 43%  37%  21% |
    |  14. llama-3.1-70b-ins... | +0.143 |  10946 | 41%  36%  22% | 42%  37%  21% |
    |  15. claude-3-opus-202... | +0.141 | 134831 | 49%  29%  21% | 50%  30%  20% |
    |  16. gpt-4-1106-preview   | +0.141 |  81545 | 53%  25%  22% | 54%  26%  20% |
    |  17. yi-large-preview     | +0.134 |  42947 | 46%  32%  22% | 47%  33%  21% |
    |  18. gpt-4-0125-preview   | +0.134 |  74890 | 49%  28%  23% | 50%  29%  20% |
    |  19. gemini-1.5-flash-... | +0.125 |  45312 | 43%  35%  22% | 43%  36%  21% |
    |  20. reka-core-20240722   | +0.125 |   5518 | 39%  39%  22% | 40%  39%  21% |
    |  21. deepseek-v2-api-0628 | +0.115 |  13075 | 37%  39%  24% | 39%  40%  21% |
    |  22. gemma-2-27b-it       | +0.114 |  22252 | 38%  38%  24% | 40%  39%  21% |
    |  23. deepseek-coder-v2... | +0.114 |   3162 | 35%  42%  24% | 36%  43%  21% |
    |  24. yi-large             | +0.109 |  13563 | 40%  37%  24% | 41%  38%  21% |
    |  25. bard-jan-24-gemin... | +0.106 |  10499 | 53%  31%  15% | 51%  29%  20% |
    |  26. nemotron-4-340b-i... | +0.106 |  16979 | 40%  37%  23% | 41%  38%  21% |
    |  27. llama-3-70b-instruct | +0.104 | 133374 | 42%  36%  22% | 43%  37%  21% |
    |  28. glm-4-0520           | +0.102 |   8271 | 39%  38%  23% | 40%  39%  21% |
    |  29. reka-flash-20240722  | +0.100 |   5397 | 34%  44%  22% | 34%  45%  21% |
    |  30. reka-core-20240501   | +0.097 |  51460 | 38%  39%  23% | 39%  40%  21% |
    +---------------------------+--------+--------+---------------+---------------+

The above code also produces the following plot of the frequencies and

.. image:: docs/source/_static/images/plots/rank.png
    :align: center
    :class: custom-dark

Visualize Correlation
---------------------

The correlation of the chatbot performances can be visualized with
``leaderbot.Davidson.visualize`` using various methods. Here is an example
with the Kernel PCA method:

.. code-block:: python

    >>> # Plot kernel PCA
    >>> model.visualize(max_rank=50)

The above code produces plot below demonstrating the Kernel PCA projection on
three principal axes:

.. image:: docs/source/_static/images/plots/kpca.png
    :align: center
    :class: custom-dark

Make Inference and Prediction
-----------------------------

Once a model is trained, you can make inference on the probabilities of win,
loss, or tie for a pair of agents using ``leaderbot.Davidson.infer`` method:

.. code-block:: python

    >>> # Create a list of three matches using pairs of indices of agents
    >>> matches = zip((0, 1, 2), (1, 2, 0))

    >>> # Make inference
    >>> prob = model.infer(matches)

    >>> # Make prediction
    >>> pred = model.predict(mathces)

Model Evaluation
----------------

Compare the performance of multiple models using ``leaderbot.evaluate``
function:

.. code-block:: python

    >>> import leaderbot as lb

    >>> # Obtain data
    >>> data = lb.data.load_data()

    >>> # Create models to compare
    >>> model_01 = lb.BradleyTerry(data)
    >>> model_02 = lb.BradleyTerryScaled(data)
    >>> model_03 = lb.BradleyTerryScaledR(data)
    >>> model_04 = lb.RaoKupper(data)
    >>> model_05 = lb.RaoKupperScaled(data)
    >>> model_06 = lb.RaoKupperScaledR(data)
    >>> model_07 = lb.Davidson(data)
    >>> model_08 = lb.DavidsonScaled(data)
    >>> model_09 = lb.DavidsonScaledR(data)

    >>> # Create a list of models
    >>> models = [model_01, model_02, model_03,
    ...           model_04, model_05, model_06,
    ...           model_07, model_08, model_09]

    >>> # Evaluate models
    >>> metrics = lb.evaluate(models, train=True, print=True)

The above model evaluation performs the analysis of the goodness of fit using
the value of loss function, KL divergence (KLD), Jensen-Shannon divergence
(JSD), Bayesian information criterion (BIC), and Akaike information criterion
(AIC), and prints a report these metrics the following table:

::

    +-----------------------+---------+--------+--------+--------+----------+-----------+
    | name                  | # param | loss   | KLD    | JSD    | AIC      | BIC       |
    +-----------------------+---------+--------+--------+--------+----------+-----------+
    | BradleyTerry          |     129 | 0.6554 |    inf | 0.0724 | 256.6892 | 1049.7267 |
    | BradleyTerryScaled    |     258 | 0.6552 |    inf | 0.0722 | 514.6896 | 2100.7646 |
    | BradleyTerryScaledR   |     259 | 0.6552 |    inf | 0.0722 | 516.6896 | 2108.9122 |
    | RaoKupper             |     130 | 1.0095 | 0.0332 | 0.0092 | 257.9810 | 1057.1661 |
    | RaoKupperScaled       |     259 | 1.0092 | 0.0323 | 0.0090 | 515.9815 | 2108.2042 |
    | RaoKupperScaledR      |     260 | 1.0092 | 0.0323 | 0.0090 | 517.9816 | 2116.3518 |
    | Davidson              |     130 | 1.0100 | 0.0341 | 0.0094 | 257.9800 | 1057.1651 |
    | DavidsonScaled        |     259 | 1.0098 | 0.0332 | 0.0092 | 515.9805 | 2108.2031 |
    | DavidsonScaledR       |     260 | 1.0098 | 0.0332 | 0.0092 | 517.9805 | 2116.3507 |
    +-----------------------+---------+--------+--------+--------+----------+-----------+

Test
====

You may test the package with `tox <https://tox.wiki/>`__:

.. code-block::

    cd source_dir
    tox

Alternatively, test with `pytest <https://pytest.org>`__:

.. code-block::

    cd source_dir
    pytest

How to Contribute
=================

We welcome contributions via GitHub's pull request. Developers should review
our [Contributing Guidelines](CONTRIBUTING.rst) before submitting their code.
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
