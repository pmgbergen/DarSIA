![build](https://github.com/pmgbergen/DarSIA/workflows/Build%20test/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: Apache v2](https://img.shields.io/hexpm/l/apa)](https://opensource.org/licenses/Apache-2.0)

# DarSIA
Darcy scale image analysis toolbox

# Documentation
Visit pmgbergen.github.io/DarSIA

# Citing

If you use DarSIA in your research, we ask you to cite the following publication:

Nordbotten, J. M., Benali, B., Both, J. W., Brattekås, B., Storvik, E., & Fernø, M. A. (2023).
DarSIA: An open-source Python toolbox for two-scale image processing of dynamics in porous media.
Transport in Porous Media, https://doi.org/10.1007/s11242-023-02000-9

The first release can be also found on Zenodo:
10.5281/zenodo.7515016

## Installation

DarSIA is developed under Python 3.12+. Clone the repository from GitHub and enter the DarSIA folder.

### Using uv (recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. Install it once, then use it for all DarSIA installs:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone & install (editable, with dev dependencies)
git clone https://github.com/pmgbergen/DarSIA.git
cd DarSIA
uv sync --extra dev
```

### Using pip

```bash
git clone https://github.com/pmgbergen/DarSIA.git
cd DarSIA
pip install -e .[dev]
```

### Optional: `petsc4py` (recommended for performance-critical solvers)

`petsc4py` is an optional but recommended dependency for performance-critical solvers such as Wasserstein distance computation. It is not installed by default.

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install -y libhypre-dev libmumps-seq-dev build-essential gcc gfortran mpich cmake
pip install numpy mpi4py
PETSC_CONFIGURE_OPTIONS="--download-hypre --download-mumps --download-parmetis --download-ml --download-metis --download-scalapack" pip install petsc petsc4py

# Then install DarSIA:
# E.g. with uv (editable environment):
uv sync --extra dev
```

**macOS / conda:**
```bash
conda install -c conda-forge petsc petsc4py
```
See also `conda_env.yaml` for a complete conda environment.


## GUI

The DarSIA GUI provides an interactive interface for image analysis workflows.

### Running the GUI

```bash
uv run darsia
```

### Desktop Integration (Optional)

To make DarSIA appear in your Linux application menu or Windows Start Menu:

```bash
uv run darsia-install-desktop
```

To remove the desktop entry:

```bash
uv run darsia-install-desktop --uninstall
```

**Note:** On Windows, this feature requires `pywin32`, which is automatically installed when syncing the `darsia` package on Windows systems.


## Developing DarSIA

Use black (version 22.3.0), flake8 and isort formatting.
See [DEVELOPER_NOTES.md](./DEVELOPER_NOTES.md) for workflow documentation maintenance guidance, including risks, acceptance criteria, and update conventions.
