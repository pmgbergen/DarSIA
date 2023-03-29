"""Example demonstrating the I/O routines for numpy files."""

import os
from pathlib import Path

import numpy as np

import darsia

# Provide folder with numpy image
folder = f"{os.path.dirname(__file__)}/images"
path = Path(folder) / Path("random_distribution.npy")

# Read numpy image
np_image = darsia.imread(path, dim=2, width=2, height=1)
np_image.show("random distribution", 3)
