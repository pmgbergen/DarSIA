"""Example of reading and resampling vtu images corresponding
to mixed-dimensional data. For this, embed the one-dimensional data
and superpose both, weighted by a dimensionally relevant quantity.

"""

import os
from pathlib import Path

import darsia

# Read two-dimensional vtu image (standard)
folder = Path(f"{os.path.dirname(__file__)}/images")
vtu_2d_path = folder / Path("fracture_flow_2.vtu")
vtu_image_2d = darsia.imread(vtu_2d_path, key="c", shape=(200, 200), vtu_dim=2)

# Read one-dimensional vtu image (two-dimensional reconstruction
# through conservative embedding)
vtu_1d_path = folder / Path("fracture_flow_1.vtu")
fracture_aperture = 0.1 * 0.01  # in m
vtu_image_1d = darsia.imread(
    vtu_1d_path,
    key="c",
    shape=(1001, 51),
    vtu_dim=1,
    width=fracture_aperture,
)

# Equidimensional reconstrctions. Superpose 2d and 1d images.
porosity_2d = 0.211
porosity_1d = 1.0
vtu_image = darsia.superpose(
    [darsia.weight(vtu_image_2d, porosity_2d), darsia.weight(vtu_image_1d, porosity_1d)]
)
vtu_image.show("equi-dimensionsional reconstruction", 5)
