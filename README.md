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
arXiv preprint arXiv:2301.05455.

The first release can be also found on Zenodo:
10.5281/zenodo.7515016

## Installation
Clone the repository from github and enter the DarSIA folder. Then, run the following command to install:

```bash
pip install .
```

## Usage

The following Python script can be applied to the test image in the examples/images folder.

```python
import darsia as da

# Create a darsia Image: An image that also contains information of physical entities
image = da.Image("examples/images/baseline.jpg", origin = [5, 2], width = 280, height = 150)

# Use the show method to take a look at the imported image (push any button to close the window)
image.show()

# Copies the image and adds a grid on top of it.
grid_image = image.add_grid(origin = [5, 2], dx = 10, dy = 10)
grid_image.show()

# Extract region of interest (ROI) from image:
ROI_image = da.extractROI(image, [[150, 0], [280, 70]])
ROI_image.show()
```

Furthermore, we encourage any user to checkout the jupyter notebooks in the examples/notebooks folder.

## Developing DarSIA
To install darsia, along with the tools to develop and run tests, run the following in your virtual environment:
```bash
$ pip install -e .[dev]
```

Use black (version 22.3.0), flake8 and isort formatting.

