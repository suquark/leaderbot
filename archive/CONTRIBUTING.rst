Contributing Guidelines
=======================

Thank you for contributing to this project! We follow strict coding standards
to ensure code quality and maintainability. Please follow these guidelines when
submitting code:

Coding Standards
----------------

1. **PEP8 Compliance**

   * All code must follow `PEP8 <https://peps.python.org/pep-0008/>`__
     standards.
   * Use linters (such as `flake8 <https://flake8.pycqa.org/>`__ or
     `ruff <https://docs.astral.sh/ruff/>`__) to check your code before
     submitting.

2. **Line Length**

   The maximum line length is **80 characters**.

3. **Naming Conventions**

   Follow standard naming conventions:

   * Variables: ``snake_case``
   * Classes: ``CamelCase``
   * Constants: ``ALL_CAPS``

4. **Type Hints**

   * All functions should have
     `type hints <https://docs.python.org/3/library/typing.html>`__ for their
     *input* parameters.
   * Type hints for *return* values are not required (optional).

   Example:

   .. code:: python

       def add(x: int, y: int) -> int:
           return x + y

5. **Interface Functions**:

   * Functions **without an underscore** at the beginning of their name (e.g.,
     ``loss``) are considered **interface functions**. These are the
     public-facing functions that users will import and interact with.
   * These functions should have clear, concise docstrings, using the
     **numpydoc** format (see Documentation section below).

6. **Non-Interface Functions**

   * Functions that start with an underscore (e.g., ``_sample_loss``) are
     considered non-interface or internal functions. These are not meant to be
     directly accessed by the user.
   * These functions still require proper type hints and concise commenting for
     internal logic, but a comprehensive docstrings is not required.

Documentation
-------------

1. Use `reStructuredText <https://docutils.sourceforge.io/rst.html>`_ (``.rst``
   files) for documentation and README files.

2. **Docstrings**

   * Interface functions and classes must include docstrings.
   * Non-interface functions are not required to have docstring.
   * Use the `numpydoc format <https://numpydoc.readthedocs.io/>`__ for all
     docstrings.

     .. code:: python

         def add(x: int, y: int) -> int:
             """
             Add two integers.

             Parameters
             ----------

             x : int
                 First number to add.

             y : int
                 Second number to add.

             Returns
             -------

             z : int
                 The sum of `x` and `y`.
             """

             return x + y

3. **Build Sphinx Documentation**

   * Upon adding new class/module, include the corresponding class/module name
     to the documentation in ``/docs/source/api.rst``.
   * Build the Sphinx documentation with:

     .. code:: bash

         cd /docs
         make clean html

     You can view the documentation at ``/docs/build/html/index.html``.

Tests
-----

1. **Test Scripts**

   * Upon adding new class/module, include a test script for it in ``/tests``
     directory.
   * Test the package with `tox <https://tox.wiki/>`__ or
     `pytest <https://docs.pytest.org/>`__.

Package Structure
-----------------

1. **Adding New Algorithm**

   To add a new algorithm, such as as an algorithm named ``foo_bar``:

   1. Create a new file in ``/leaderbot/algorithms/foo_bar.py``.
   2. Define a class therein called ``FooBar`` that is inherited from
      ``BaseModel`` base class.
   3. In ``/leaderbot/algorithms/__init__.py`` import your new class and add
      its name to ``__all__`` variable.
