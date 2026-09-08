#################
Contributor guide
#################

.. toctree::
   :hidden:

   writing_docs

Getting the source
==================

.. code-block:: bash

   git clone https://github.com/pmgbergen/darsia.git
   cd darsia
   uv sync --extra dev --extra docs

Development happens on the ``dev`` branch; ``main`` tracks releases. Open pull
requests against ``dev``.

Coding standards
================

All code under ``src/`` must pass, and CI enforces:

============  ==================================================================
``black``     version **24.10.0** (pinned via ``required-version`` in
              ``pyproject.toml``), line length 88
``isort``     ``profile = "black"``
``flake8``    line length 95; ignore list in ``setup.cfg``
============  ==================================================================

.. code-block:: bash

   uv run black src
   uv run isort src
   uv run flake8 src

Type hints are expected on new public functions; ``mypy`` is available as a dev
dependency (the CI ``mypy`` step is currently disabled).

Docstrings feed the API reference and must be **NumPy style** (``Parameters`` /
``Returns`` with underlines), rendered by ``numpydoc``. ``numpydoc validate``
can lint a docstring against the NumPy conventions.

Running the tests
=================

.. code-block:: bash

   uv run pytest                 # everything
   uv run pytest tests/unit -q   # fast unit tests
   uv run pytest tests/integration

GUI tests use ``pytest-qt``. The example scripts are exercised by
``tests/integration/test_examples.py`` (and, more thoroughly, by the docs
build).

Continuous integration
======================

The *Build test* workflow (``.github/workflows/ci.yml``) runs on every push and
pull request to ``main`` and ``dev``:

* **static-checks** -- ``black`` / ``flake8`` / ``isort`` on ``src``;
* **build-without-petsc** -- ``pytest`` on Python 3.12, 3.13 and 3.14;
* **pytest-with-petsc** -- full suite with PETSc, on ``main`` only.

The *Documentation* workflow builds this site and deploys it to GitHub Pages on
every push to ``dev``.

Building the documentation
==========================

.. code-block:: bash

   uv run sphinx-build -b html docs docs/_build/html

Open ``docs/_build/html/index.html``. CI builds with ``-W --keep-going``
(warnings are errors), so run it that way locally too. See :doc:`writing_docs`
for how the docs are organised and how to add an example or a page.
