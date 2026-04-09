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
DarSIA is developed under Python 3.10. Clone the repository from github and enter the DarSIA folder. Then, run the following command to install:

```bash
pip install .
```
To install DarSIA as editable (recommended), along with the tools to develop and run tests, run the following in your virtual environment:
```bash
$ pip install -e .[dev]
```

### Optional: `petsc4py` (recommended for performance-critical solvers)

`petsc4py` is an optional but recommended dependency for performance-critical solvers such as Wasserstein distance computation. It is not installed by default.

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install -y libhypre-dev libmumps-seq-dev build-essential gcc gfortran mpich cmake
pip install numpy mpi4py
PETSC_CONFIGURE_OPTIONS="--download-hypre --download-mumps --download-parmetis --download-ml --download-metis --download-scalapack" pip install petsc petsc4py
# Then install DarSIA with petsc extra
pip install .[petsc]
# Or combined with dev dependencies:
pip install .[dev,petsc]
```

**macOS / conda:**
```bash
conda install -c conda-forge petsc petsc4py
```
See also `conda_env.yaml` for a complete conda environment.
## Usage

The following Python script can be applied to the test image in the examples/images folder.

```python
import numpy as np

# Create a darsia Image: An image that also contains information of physical entities
image = darsia.imread("images/baseline.jpg", width=2.8, height=1.5)

# Use the show method to take a look at the imported image.
image.show()

# Copy the image and adds a grid on top of it.
grid_image = image.add_grid(dx=0.1, dy=0.1)
grid_image.show()

# Extract region of interest (ROI) from image (box defined by two corners):
ROI_image = image.subregion(coordinates=np.array([[1.5, 0], [2.8, 0.7]]))
ROI_image.show()
```

Furthermore, we encourage any user to checkout the examples in the examples folder and the jupyter notebooks in the examples/notebooks folder.

## Developing DarSIA

Use black (version 22.3.0), flake8 and isort formatting.
See [DEVELOPER_NOTES.md](./DEVELOPER_NOTES.md) for workflow documentation maintenance guidance, including risks, acceptance criteria, and update conventions.
