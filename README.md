![build](https://github.com/pmgbergen/DarSIA/workflows/Build%20test/badge.svg)
[![docs](https://github.com/pmgbergen/DarSIA/workflows/Documentation/badge.svg)](https://pmgbergen.github.io/DarSIA)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: Apache v2](https://img.shields.io/hexpm/l/apa)](https://opensource.org/licenses/Apache-2.0)

# DarSIA

**Darcy scale image analysis toolbox** — an open-source Python library for
turning images of porous-media experiments into quantitative physical data.

DarSIA represents an image as an array that also knows its physical extent,
acquisition time and coordinate system, and provides:

- **I/O** for optical photographs, DICOM stacks and simulation output (vtu);
- **corrections** — colour, illumination, curvature, perspective, drift,
  deformation;
- **restoration** — total-variation and H1 denoising;
- **segmentation and registration** of multi-layered media;
- **concentration analysis** — tracer / CO2 / mass maps, with pluggable signal
  models and calibration;
- **transport-based distances** (Earth Mover's / Wasserstein);
- a **configuration-driven workflow system** and a **Qt GUI** that runs it
  without scripting.

## Documentation

<https://pmgbergen.github.io/DarSIA> — user guide, runnable example gallery, and full API reference.

See the [example gallery](https://pmgbergen.github.io/DarSIA/auto_examples/) for
more, including CO2 concentration analysis and Wasserstein distances.

## Installation

DarSIA needs **Python 3.12+** and is installed from a clone (not on PyPI yet, and here using the recommended package manager [uv](https://docs.astral.sh/uv/getting-started/installation/)).

```bash
git clone https://github.com/pmgbergen/DarSIA.git
cd DarSIA
uv python install 3.12         # installs Python 3.12
uv sync --extra dev            # or: pip install -e ".[dev]"
```

The [installation guide](https://pmgbergen.github.io/DarSIA/getting_started/installation.html)
covers the optional `petsc4py` solvers, a conda environment, the GUI, and
building the documentation.

## GUI

A Qt desktop application that runs the configuration-driven analysis workflow
from a TOML file — set up, calibrate, and batch-process an experiment without
writing a driver script.

```bash
uv run darsia-install-desktop   # optional: adds a desktop / Start-menu launcher
uv run darsia                   # launch the GUI after desktop installation
```

See the [GUI guide](https://pmgbergen.github.io/DarSIA/user_guide/gui/index.html)
for a full walkthrough.

## Citing

If you use DarSIA in your research, we ask you to cite the following publication:

Nordbotten, J. M., Benali, B., Both, J. W., Brattekås, B., Storvik, E., & Fernø, M. A. (2023).
DarSIA: An open-source Python toolbox for two-scale image processing of dynamics in porous media.
Transport in Porous Media, https://doi.org/10.1007/s11242-023-02000-9

The first release can be also found on Zenodo:
10.5281/zenodo.7515016

## Developing DarSIA

DarSIA is an open+source project, and contributions are most welcomed! Code under `src/` must pass `black` (24.10.0), `isort` and `flake8`; run the
tests with `uv run pytest`. See the
[contributor guide](https://pmgbergen.github.io/DarSIA/development/index.html)
and [DEVELOPER_NOTES.md](./DEVELOPER_NOTES.md).
