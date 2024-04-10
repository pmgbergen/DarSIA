"""Module collecting tools for geometric integration.

Various versions of the general algorithm are provided, taking into account width,
height and depth of pixels.

"""

from __future__ import annotations

from typing import Optional, Union

import cv2
import numpy as np

import darsia


class Geometry:
    """
    Class containing information of the geometry.

    Also allows for geometrical integration.

    Example:

    dimensions = {"width": 1., "height": 2., "depth": 0.1}
    shape = (20,10)
    geometry = darsia.Geometry(shape, **dimensions)

    """

    def __init__(
        self,
        space_dim: int,
        num_voxels: Union[tuple[int], list[int]],
        dimensions: Optional[list] = None,
        voxel_size: Optional[list] = None,
        **kwargs,
    ) -> None:
        """
        Constructor.

        Args:
            space_dim (int): spatial dimensions of the geometry.
            shape (tuple of int): shape of voxelization of the geometry.
            dimensions (list, optional): dimensions of the entire geometry.
            voxel_size (list, optional): dimensions of single voxel.

        """

        self.space_dim = space_dim
        """Spatial dimension of geometry."""

        self.num_voxels = list(num_voxels[:space_dim])
        """Number of voxels in each spatial direction."""

        # NOTE: dimensions overrules voxel_size for retrieving dimensions
        # of geometry and voxels.
        if dimensions is None:
            assert voxel_size is not None
            self.voxel_size = voxel_size
            """Dimensions of a single voxel."""
            self.dimensions = [
                self.num_voxels[i] * self.voxel_size[i] for i in range(self.space_dim)
            ]
            """Dimensions of geometry."""
        else:
            self.dimensions = dimensions
            self.voxel_size = [
                self.dimensions[i] / self.num_voxels[i] for i in range(self.space_dim)
            ]

        self.voxel_volume = np.prod(self.voxel_size)
        """Volume (area in 2d) of a single voxel."""
        self.cached_voxel_volume = self.voxel_volume.copy()
        """Internal copy of the voxel volume for efficient integration."""

    def integrate(
        self, data: Union[darsia.Image, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Integrate data over the entire geometry.

        Args:
            data (np.ndarray): data attached to voxels.

        Returns:
            float or array: integral of data over geometry, array if time series and/or
                non-scalar data is provided.

        Raises:
            ValueError: In dimensions other than 2, if data and geometry incompatible
                and reshape is needed.

        """
        # ! ---- Make sure that the geometry is compatible with the provided data

        # Fetch data
        fetched_data = data if isinstance(data, np.ndarray) else data.img
        fetched_shape = list(fetched_data.shape[: self.space_dim])
        scaling = np.prod(np.divide(self.num_voxels, fetched_shape))

        # Resize the voxel volumes using conservative resizing.
        if isinstance(self.voxel_volume, np.ndarray):

            cached_shape = list(self.cached_voxel_volume.shape)
            if not all([i == j for i, j in zip(fetched_shape, cached_shape)]):

                if not self.space_dim == 2:
                    raise ValueError("Incompatible data format only supported in 2d.")

                # To cover the most general case, a weighted sum is required. The weight is
                # essentially provided by the voxel volume. Due to a possible spatial
                # variability however, reshaping and rescaling is required. In ordert to
                # increase efficiency use a cache, assuming frequently similar data as input.
                # In the case, a fixed (scalar) depth had been provided, the base class can be
                # utilized. Otherwise, a more involved reshape of the effective volume is
                # required.
                self.cached_voxel_volume = (
                    cv2.resize(
                        self.voxel_volume,
                        tuple(reversed(fetched_data.shape[:2])),
                        interpolation=cv2.INTER_AREA,  # conservative.
                    )
                    * scaling
                )

        else:
            # Scalar case.
            if not all([i == j for i, j in zip(fetched_shape, self.num_voxels)]):
                self.cached_voxel_volume = self.voxel_volume * scaling

        # ! ---- Perform spatial integration
        if isinstance(data, np.ndarray):
            weighted_sum = np.multiply(self.cached_voxel_volume, data)
        elif isinstance(data, darsia.Image):
            weighted_sum = np.multiply(self.cached_voxel_volume, data.img)
        else:
            raise ValueError("Data type not supported.")
        for i in range(self.space_dim):
            weighted_sum = np.sum(weighted_sum, axis=0)
        return weighted_sum

    def normalize(
        self, img: darsia.Image, img_ref: darsia.Image, return_ratio: bool = False
    ) -> Union[darsia.Image, tuple[darsia.Image, np.ndarray]]:
        """Normalize image with respect to another one, such that both have the same
        integral.

        Args:
            img (darsia.Image): image to be rescaled
            img_ref (darsia.Image): reference image
            return_ratio (bool): flag controlling whether the ratio between reference
                and original integrals is returned

        Returns:
            darsia.Image: rescaled image
            np.ndarray, optional: ratio between reference and original integrals

        """
        integral_ref = self.integrate(img_ref)
        integral = self.integrate(img)
        ratio = np.divide(integral_ref, integral)
        rescaled_img = darsia.weight(img, ratio)

        if return_ratio:
            return rescaled_img, ratio
        else:
            return rescaled_img


class WeightedGeometry(Geometry):
    """Geometry with weighted volume."""

    def __init__(
        self,
        weight: Union[float, np.ndarray],
        space_dim: int,
        num_voxels: Union[tuple[int], list[int]],
        dimensions: Optional[list] = None,
        voxel_size: Optional[list] = None,
        **kwargs,
    ) -> None:
        """
        Constructor for extruded two-dimensional geometry.

        Args:
            weight (float or array): weight compatible with geometry.
            space_dim (int): see Geometry.
            num_voxels (tuple): see Geometry.
            dimensions (list): see Geometry.
            voxel_size (list): see Geometry.

        Raises:
            ValueError: if weight has wrong dimensions.

        """
        super().__init__(space_dim, num_voxels, dimensions, voxel_size)

        # Sanity check
        if isinstance(weight, np.ndarray) and len(weight.shape) != self.space_dim:
            raise ValueError

        # Add weight
        self.voxel_volume = np.multiply(self.voxel_volume, weight)
        """Effective voxel volume in 3d."""
        self.cached_voxel_volume = self.voxel_volume.copy()
        """Internal copy of the voxel volume for efficient integration."""


class ExtrudedGeometry(WeightedGeometry):
    """One or two-dimensional geometry extruded to three dimensions."""

    def __init__(
        self,
        expansion: Union[float, np.ndarray],
        space_dim: int,
        num_voxels: Union[tuple[int], list[int]],
        dimensions: Optional[list] = None,
        voxel_size: Optional[list] = None,
        **kwargs,
    ) -> None:
        """
        Constructor for extruded two-dimensional geometry.

        Args:
            expansion (float or array): effective depth/area of 1d/2d geometry.
            space_dim (int): see Geometry.
            num_voxels (tuple): see Geometry.
            dimensions (list): see Geometry.
            voxel_size (list): see Geometry.

        Raises:
            ValueError: if spatial dimension not 2.

        """
        super().__init__(expansion, space_dim, num_voxels, dimensions, voxel_size)


class PorousGeometry(WeightedGeometry):
    """Class containing information of a porous geometry."""

    def __init__(
        self,
        porosity: Union[float, np.ndarray],
        space_dim: int,
        num_voxels: Union[tuple[int], list[int]],
        dimensions: Optional[list] = None,
        voxel_size: Optional[list] = None,
        **kwargs,
    ) -> None:
        """
        Constructor for extruded two-dimensional geometry.

        Args:
            porosity (float or array): porosity.
            space_dim (int): see Geometry.
            num_voxels (tuple): see Geometry.
            dimensions (list): see Geometry.
            voxel_size (list): see Geometry.

        """
        super().__init__(porosity, space_dim, num_voxels, dimensions, voxel_size)


class ExtrudedPorousGeometry(ExtrudedGeometry):
    """Class containing information of a porous geometry."""

    def __init__(
        self,
        porosity: Union[float, np.ndarray],
        depth: Union[float, np.ndarray],
        space_dim: int,
        num_voxels: Union[tuple[int], list[int]],
        dimensions: Optional[list] = None,
        voxel_size: Optional[list] = None,
        **kwargs,
    ) -> None:
        """
        Constructor for extruded, porous, two-dimensional geometry.

        Args:
            porosity (float or array): porosity.
            depth (float or array): effective depth.
            space_dim (int): see Geometry.
            num_voxels (tuple): see Geometry.
            dimensions (list): see Geometry.
            voxel_size (list): see Geometry.

        """
        integrated_porosity = np.multiply(porosity, depth)
        super().__init__(
            integrated_porosity, space_dim, num_voxels, dimensions, voxel_size
        )
