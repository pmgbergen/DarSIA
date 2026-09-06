#################
Contributor guide
#################

.. note::

   This section is being expanded. See ``DEVELOPER_NOTES.md`` and
   ``CONTRIBUTORS.md`` in the repository for the current details.

Development environment
=======================

::

   git clone https://github.com/pmgbergen/darsia.git
   cd darsia
   uv sync --extra dev --extra docs

Coding standards
================

All code must pass ``black`` (version 24.10.0, line length 88), ``isort``
(``profile = black``) and ``flake8`` before it is merged. Type hints are
expected on new public functions.

Docstrings are being migrated to **NumPy style** (rendered via ``numpydoc``).
New or edited docstrings should use NumPy-style sections
(``Parameters`` / ``Returns`` / ``Raises`` / ``Examples``).

Running the tests
=================

::

   uv run pytest tests/unit -v
   uv run pytest tests/integration

GUI tests use ``pytest-qt``.

Building the documentation
==========================

::

   uv run sphinx-build -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` in a browser. Add ``-W --keep-going`` to
reproduce the CI build, which treats warnings as errors.
