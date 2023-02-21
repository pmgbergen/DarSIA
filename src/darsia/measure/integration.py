"""
Module collecting tools for geometric integration,
taking into account width, height and depth of pixels.

"""

from typing import Optional, Union

import cv2
import numpy as np

import darsia

# TODO 3d?


class Geometry:
    """
    Class containing information of the geometry.

    Also allows for geometrical integration.

    Example:

    dimensions = {"width": 1., "height": 2., "depth": 0.1}
    shape = (20,10)
    geometry = darsia.Geometry(shape, **dimensions)

    """

    def __init__(self, shape: tuple[int], **kwargs) -> None:

        # Determine number of voxels in each dimension
        Ny, Nx = shape[:2]
        Nz = 1 if len(shape) < 3 else shape[2]

        # Cache
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz

        # Define width, height and depth of each voxel
        self.voxel_width = kwargs.get("voxel width", None)
        if self.voxel_width is None:
            self.voxel_width = kwargs.get("width") / Nx

        self.voxel_height = kwargs.get("voxel height", None)
        if self.voxel_height is None:
            self.voxel_height = kwargs.get("height") / Ny

        self.voxel_depth = kwargs.get("voxel depth", None)
        if self.voxel_depth is None:
            self.voxel_depth = kwargs.get("depth") / Nz

        # Determine effective pixel and voxel measures
        self.voxel_area = np.multiply(self.voxel_width, self.voxel_height)
        self.voxel_volume = (
            self.voxel_area
            if self.voxel_depth is None
            else np.multiply(self.voxel_area, self.voxel_depth)
        )

        # Cache voxel volume
        self.cached_voxel_volume = self.voxel_volume.copy()

    def integrate(
        self, data: Union[darsia.Image, np.ndarray], mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Integrate data over the entire geometry.
        """

        # Check compatibility of data formats
        if isinstance(self.voxel_volume, np.ndarray):
            if data.shape[:2] != self.cached_voxel_volume.shape[:2]:
                scaling = np.prod(self.voxel_volume.shape[:2]) / np.prod(data.shape[:2])
                self.cached_voxel_volume = (
                    cv2.resize(
                        self.voxel_volume,
                        tuple(reversed(data.shape[:2])),
                        interpolation=cv2.INTER_AREA,
                    )
                    * scaling
                )
            return np.sum(np.multiply(self.cached_voxel_volume, data))
        else:
            # TODO 3d
            Ny_data, Nx_data = data.shape[:2]
            rescaled_voxel_volume = (
                self.voxel_volume * self.Nx / Nx_data * self.Ny / Ny_data
            )
            return rescaled_voxel_volume * np.sum(data)


class PorousGeometry(Geometry):
    """
    Class containing information of a porous geometry.

    Also allows for geometrical integration over pore space.

    Example:

    asset = {"porosity": 0.2, "width": 1., "height": 2., "depth": 0.1}
    shape = (20,10)
    geometry = darsia.PorousGeometry(shape, **asset)

    """

    def __init__(self, shape: np.ndarray, **kwargs) -> None:

        super().__init__(shape, **kwargs)
        self.porosity = kwargs.get("porosity")

        # Determine effective pixel and voxel measures
        self.voxel_area = np.multiply(self.voxel_area, self.porosity)
        self.voxel_volume = np.multiply(self.voxel_volume, self.porosity)

        # Update cached voxel volume due to involvement of the porosity
        self.cached_voxel_volume = self.voxel_volume.copy()
