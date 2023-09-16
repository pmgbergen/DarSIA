"""Grid utilities."""

from typing import Union

import numpy as np
import scipy.sparse as sps

import darsia


class Grid:
    """Tensor grid.

    Attributes:
        shape: Shape of grid.
        ndim: Number of dimensions.
        size: Number of grid points.

    """

    def __init__(self, shape: tuple, voxel_size: Union[float, list] = 1.0):
        """Initialize grid."""

        self.shape = shape
        self.dim = len(shape)
        self.size = np.prod(shape)
        self.voxel_size = (
            np.array(voxel_size)
            if isinstance(voxel_size, list)
            else voxel_size * np.ones(self.dim)
        )
        assert len(self.voxel_size) == self.dim


def generate_grid(image: darsia.Image) -> Grid:
    """Get grid object."""
    shape = image.num_voxels
    voxel_size = image.voxel_size
    return Grid(shape, voxel_size)
