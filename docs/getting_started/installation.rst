############
Installation
############

DarSIA requires **Python 3.12 or newer**. It is not on PyPI yet, so install it
from a clone of the repository.

With uv (recommended)
=====================

`uv <https://github.com/astral-sh/uv>`_ is a fast Python package/environment
manager.

.. code-block:: bash

   # install uv once
   curl -LsSf https://astral.sh/uv/install.sh | sh

   git clone https://github.com/pmgbergen/darsia.git
   cd darsia
   uv python install 3.12
   uv sync --extra dev

Add ``--extra docs`` to also install the documentation toolchain
(``uv sync --extra dev --extra docs``).

With pip
========

.. code-block:: bash

   git clone https://github.com/pmgbergen/darsia.git
   cd darsia
   pip install -e ".[dev]"        # or ".[dev,docs]"

Optional: ``petsc4py``
======================

``petsc4py`` accelerates the performance-critical linear solvers (notably the
Wasserstein / Beckmann solvers). It is **not** installed by default.

**Linux (Ubuntu/Debian):**

.. code-block:: bash

   sudo apt-get install -y libhypre-dev libmumps-seq-dev build-essential \
     gcc gfortran mpich cmake
   pip install numpy mpi4py
   PETSC_CONFIGURE_OPTIONS="--download-hypre --download-mumps --download-parmetis \
     --download-ml --download-metis --download-scalapack" pip install petsc petsc4py

**macOS / conda:**

.. code-block:: bash

   conda install -c conda-forge petsc petsc4py

The bundled ``conda_env.yaml`` creates a complete conda environment (named
``darcia``) with PETSc included: ``conda env create --file conda_env.yaml``.

The graphical interface
=======================

The GUI ships with DarSIA (its Qt bindings, ``PySide6``, are a normal
dependency). Launch it with::

   uv run darsia

Optionally register a desktop / Start-menu entry::

   uv run darsia-install-desktop            # add
   uv run darsia-install-desktop --uninstall  # remove

On Windows this needs ``pywin32``, which is installed automatically on Windows.
See :doc:`../user_guide/gui/index` for a full GUI tutorial.

Verifying the installation
==========================

.. code-block:: bash

   uv run python -c "import darsia; print(darsia.__name__)"
   uv run pytest tests/unit -q

Building the documentation
==========================

With the ``docs`` extra installed::

   uv run sphinx-build -b html docs docs/_build/html

then open ``docs/_build/html/index.html``. See :doc:`../development/writing_docs`
for how the documentation is organised.
