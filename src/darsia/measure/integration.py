"""Module collecting tools for geometric integration.

Various versions of the general algorithm are provided, taking into account width,
height and depth of pixels.

"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import cv2
import numpy as np

import darsia


def _without_nan(weight: float | np.ndarray) -> float | np.ndarray:
    """Weight with NaN replaced by zero (a new array; scalars as they are)."""
    return np.nan_to_num(weight, nan=0.0) if isinstance(weight, np.ndarray) else weight


def _porosity_times_depth(
    porosity: float | np.ndarray | darsia.Image,
    depth: float | np.ndarray | darsia.Image,
) -> float | np.ndarray:
    """Product of porosity and depth (values of images)."""
    return np.multiply(
        porosity.img if isinstance(porosity, darsia.Image) else porosity,
        depth.img if isinstance(depth, darsia.Image) else depth,
    )


def _porous_weight(
    porosity: float | np.ndarray | darsia.Image,
    depth: float | np.ndarray | darsia.Image,
) -> float | np.ndarray:
    """Weight of a porous geometry: porosity times depth, NaN replaced by zero."""
    return _without_nan(_porosity_times_depth(porosity, depth))


class Geometry:
    """Geometry of a physical domain, supporting geometric integration.

    Examples
    --------
    .. code-block:: python

        dimensions = {"width": 1.0, "height": 2.0, "depth": 0.1}
        shape = (20, 10)
        geometry = darsia.Geometry(shape, **dimensions)
    """

    def __init__(
        self,
        space_dim: int,
        num_voxels: tuple[int] | list[int],
        dimensions: Optional[list] = None,
        voxel_size: Optional[list] = None,
        **kwargs,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        space_dim : int
            Spatial dimensions of the geometry.
        num_voxels : tuple of int
            Shape of the voxelisation of the geometry.
        dimensions : list, optional
            Dimensions of the entire geometry.
        voxel_size : list, optional
            Dimensions of single voxel.
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

    def voxel_volumes(self, shape: tuple[int, ...]) -> np.ndarray | float:
        """Voxel volumes with which `integrate` weights data of the given shape.

        Parameters
        ----------
        shape : tuple
            Spatial shape of the data.

        Returns
        -------
        np.ndarray or float
            Array of that shape (resized if the geometry has another resolution), or a
            scalar for a uniform geometry. Not to be modified.
        """
        self._prepare_cached_voxel_volume(list(shape[: self.space_dim]))
        return self.cached_voxel_volume

    def _prepare_cached_voxel_volume(self, fetched_shape: list[int]) -> None:
        """Prepare cached voxel volume for data with a specific spatial shape."""
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
                        tuple(reversed(fetched_shape[:2])),
                        interpolation=cv2.INTER_AREA,  # conservative.
                    )
                    * scaling
                )

        else:
            # Scalar case.
            if all([i == j for i, j in zip(fetched_shape, self.num_voxels)]):
                self.cached_voxel_volume = self.voxel_volume.copy()
            else:
                if not self.space_dim == 2:
                    raise ValueError("Incompatible data format only supported in 2d.")
                self.cached_voxel_volume = self.voxel_volume * scaling

    def integrate(self, data: darsia.Image | np.ndarray) -> float | np.ndarray:
        """
        Integrate data over the entire geometry.

        Parameters
        ----------
        data : np.ndarray
            Data attached to voxels.

        Returns
        -------
        float or array
            Integral of data over geometry, array if time series and/or
            non-scalar data is provided.

        Raises
        ------
        ValueError
            In dimensions other than 2, if data and geometry incompatible
            and reshape is needed.
        """
        # ! ---- Make sure that the geometry is compatible with the provided data

        # Fetch data
        fetched_data = data if isinstance(data, np.ndarray) else data.img
        fetched_shape = list(fetched_data.shape[: self.space_dim])
        if isinstance(data, darsia.ExtensiveImage):
            extensive_sum = data.img
            for _ in range(self.space_dim):
                extensive_sum = np.sum(extensive_sum, axis=0)
            return extensive_sum

        self._prepare_cached_voxel_volume(fetched_shape)

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

    def make_extensive(self, data: darsia.Image) -> darsia.ExtensiveImage:
        """Convert intensive values to integrated per-cell values on geometry grid.

        Parameters
        ----------
        data : darsia.Image
            Image of intensive values.

        Returns
        -------
        darsia.ExtensiveImage
            Extensive image represented on geometry voxelization.

        Raises
        ------
        ValueError
            If data is not an image, dimensions are incompatible, or
            incompatible reshaping is requested in dimensions other than 2.
        """
        if not isinstance(data, darsia.Image):
            raise ValueError("Data type not supported.")
        if not data.space_dim == self.space_dim:
            raise ValueError("Incompatible space dimensions for image and geometry.")
        if not np.allclose(data.dimensions, self.dimensions):
            raise ValueError(
                "Image and geometry need to cover the same physical dimensions."
            )

        fetched_shape = list(data.img.shape[: self.space_dim])
        self._prepare_cached_voxel_volume(fetched_shape)

        weighted_sum = np.multiply(self.cached_voxel_volume, data.img)
        metadata = data.metadata()
        # Remove "type" as it is informational only and not an __init__ argument.
        metadata.pop("type", None)
        weighted_sum_image = darsia.ExtensiveImage(weighted_sum, **metadata)

        # If required, map weighted values conservatively to the geometry grid.
        if not all([i == j for i, j in zip(fetched_shape, self.num_voxels)]):
            if not self.space_dim == 2:
                raise ValueError("Incompatible data format only supported in 2d.")

            weighted_sum_image = darsia.resize(
                weighted_sum_image,
                shape=tuple(self.num_voxels[:2]),
            )

        return weighted_sum_image

    def normalize(
        self, img: darsia.Image, img_ref: darsia.Image, return_ratio: bool = False
    ) -> darsia.Image | tuple[darsia.Image, np.ndarray]:
        """Normalize image with respect to another one, such that both have the same
        integral.

        Parameters
        ----------
        img : darsia.Image
            Image to be rescaled.
        img_ref : darsia.Image
            Reference image.
        return_ratio : bool
            Flag controlling whether the ratio between reference
            and original integrals is returned.

        Returns
        -------
        darsia.Image
            Rescaled image.
        np.ndarray, optional
            Ratio between reference and original integrals.
        """
        integral_ref = self.integrate(img_ref)
        integral = self.integrate(img)
        ratio = np.divide(integral_ref, integral)
        rescaled_img = darsia.weight(img, ratio)

        if return_ratio:
            return rescaled_img, ratio
        else:
            return rescaled_img

    def subregion(self, roi: darsia.CoordinateArray) -> "Geometry":
        """Extract subregion of the geometry.

        Parameters
        ----------
        roi : darsia.CoordinateArray
            Region of interest.

        Returns
        -------
        Geometry
            Subregion geometry.
        """
        # Compute new dimensions
        new_dimensions = []
        for i in range(self.space_dim):
            length = np.max(roi, axis=0)[i] - np.min(roi, axis=0)[i]
            new_dimensions.append(length)

        # Compute new number of voxels
        new_num_voxels = []
        for i in range(self.space_dim):
            length = np.max(roi, axis=0)[i] - np.min(roi, axis=0)[i]
            voxel_size = self.voxel_size[i]
            new_num_voxels.append(int(np.ceil(length / voxel_size)))

        # Create sub-geometry
        return Geometry(
            self.space_dim,
            new_num_voxels,
            new_dimensions,
        )


class WeightedGeometry(Geometry):
    """Geometry with weighted volume."""

    def __init__(
        self,
        weight: float | np.ndarray,
        space_dim: int,
        num_voxels: tuple[int] | list[int],
        dimensions: Optional[list] = None,
        voxel_size: Optional[list] = None,
        weight_source: Callable[[], float | np.ndarray] | None = None,
        **kwargs,
    ) -> None:
        """
        Constructor for extruded two-dimensional geometry.

        Parameters
        ----------
        weight : float or array
            Weight compatible with geometry.
        space_dim : int
            See Geometry.
        num_voxels : tuple
            See Geometry.
        dimensions : list
            See Geometry.
        voxel_size : list
            See Geometry.
        weight_source : Callable, optional
            Gives the weight (NaN replaced by zero) whenever :attr:`weight` is asked
            for; then the weight is not kept. Otherwise a copy of `weight` is kept.

        Raises
        ------
        ValueError
            If weight has wrong dimensions.
        """
        super().__init__(space_dim, num_voxels, dimensions, voxel_size)

        # Sanity check
        if isinstance(weight, np.ndarray) and len(weight.shape) != self.space_dim:
            raise ValueError(
                "Weight must have the same number of dimensions as the geometry."
            )

        # Convert nans to zeros, as they typically arise due to ROIs that remove
        # signals.
        self._weight_source = weight_source
        self._weight = None if weight_source is not None else _without_nan(weight)
        self.voxel_volume = np.multiply(self.voxel_volume, weight)
        """Effective voxel volume in 3d."""
        self.cached_voxel_volume = self.voxel_volume
        """Voxel volume for data of the current shape (the voxel volume itself, or a
        resized one, see `_prepare_cached_voxel_volume`). It may be the same array as
        `voxel_volume`: neither of them may be modified in place."""

    @property
    def weight(self) -> float | np.ndarray:
        """Weight of the geometry, a scalar or an array (NaN replaced by zero)."""
        if self._weight_source is not None:
            return self._weight_source()
        return self._weight

    def _subregion_weight(self, roi: darsia.CoordinateArray) -> float | np.ndarray:
        """Weight restricted to a subregion (a copy for arrays)."""
        weight = self.weight
        if not isinstance(weight, np.ndarray):
            return weight
        weight_image = darsia.Image(
            weight, dimensions=self.dimensions, space_dim=self.space_dim
        )
        return weight_image.subregion(roi).img.copy()

    def subregion(self, roi: darsia.CoordinateArray) -> "WeightedGeometry":
        """Extract subregion of the geometry.

        Parameters
        ----------
        roi : darsia.CoordinateArray
            Region of interest.

        Returns
        -------
        WeightedGeometry
            Subregion geometry. Its weight is taken from this geometry when asked
            for, not kept.
        """
        # Extract sub-geometry using base class method
        sub_geometry = super().subregion(roi)

        # Create sub-weighted geometry
        sub_weight = self._subregion_weight(roi)
        return WeightedGeometry(
            sub_weight,
            sub_geometry.space_dim,
            list(sub_weight.shape),
            sub_geometry.dimensions,
            sub_geometry.voxel_size,
            weight_source=partial(self._subregion_weight, roi),
        )


class ExtrudedGeometry(WeightedGeometry):
    """One or two-dimensional geometry extruded to three dimensions."""

    def __init__(
        self,
        expansion: float | np.ndarray,
        space_dim: int,
        num_voxels: tuple[int] | list[int],
        dimensions: Optional[list] = None,
        voxel_size: Optional[list] = None,
        **kwargs,
    ) -> None:
        """
        Constructor for extruded two-dimensional geometry.

        Parameters
        ----------
        expansion : float or array
            Effective depth/area of 1d/2d geometry.
        space_dim : int
            See Geometry.
        num_voxels : tuple
            See Geometry.
        dimensions : list
            See Geometry.
        voxel_size : list
            See Geometry.

        Raises
        ------
        ValueError
            If spatial dimension not 2.
        """
        self.depth = expansion
        super().__init__(expansion, space_dim, num_voxels, dimensions, voxel_size)


class PorousGeometry(WeightedGeometry):
    """Class containing information of a porous geometry."""

    def __init__(
        self,
        porosity: float | np.ndarray | darsia.Image,
        space_dim: int,
        num_voxels: tuple[int] | list[int],
        dimensions: Optional[list] = None,
        voxel_size: Optional[list] = None,
        **kwargs,
    ) -> None:
        """
        Constructor for extruded two-dimensional geometry.

        Parameters
        ----------
        porosity : float or array
            Porosity.
        space_dim : int
            See Geometry.
        num_voxels : tuple
            See Geometry.
        dimensions : list
            See Geometry.
        voxel_size : list
            See Geometry.
        """
        self.porosity = porosity
        super().__init__(porosity, space_dim, num_voxels, dimensions, voxel_size)


class ExtrudedPorousGeometry(WeightedGeometry):
    """Class containing information of a porous geometry.

    The pore volume of a voxel is its area times porosity times depth. Porosity and
    depth are referenced, not copied, and must not be changed afterwards: the weight
    (porosity times depth) is computed from them whenever it is asked for, while the
    voxel volumes are computed once.

    """

    def __init__(
        self,
        porosity: float | np.ndarray | darsia.Image,
        depth: float | np.ndarray | darsia.Image,
        space_dim: int,
        num_voxels: tuple[int] | list[int],
        dimensions: Optional[list] = None,
        voxel_size: Optional[list] = None,
        **kwargs,
    ) -> None:
        """
        Constructor for extruded, porous, two-dimensional geometry.

        Parameters
        ----------
        porosity : float or array
            Porosity.
        depth : float or array
            Effective depth.
        space_dim : int
            See Geometry.
        num_voxels : tuple
            See Geometry.
        dimensions : list
            See Geometry.
        voxel_size : list
            See Geometry.
        """
        self.porosity = porosity
        """Porosity (referenced)."""
        self.depth = depth
        """Depth (referenced)."""
        super().__init__(
            _porosity_times_depth(porosity, depth),
            space_dim,
            num_voxels,
            dimensions,
            voxel_size,
            weight_source=partial(_porous_weight, porosity, depth),
        )

    def update(self, depth: float | np.ndarray | darsia.Image) -> None:
        """Change the depth (not available).

        A new depth would have to reach everything derived from the geometry: the
        voxel volumes (area times porosity times depth, with NaN as in the
        constructor, not by dividing by the old weight, which is zero where the
        porosity is), the voxel volumes resized for data of other shapes, region
        geometries (:meth:`subregion`) and reduced copies (e.g. of the calibration).
        For a calibration of the depth, note that a global factor cannot be told
        apart from the scaling of the signal; a depth model with a few parameters
        can.

        Parameters
        ----------
        depth : float or array
            Effective depth.

        Raises
        ------
        NotImplementedError
            Always; build a new geometry instead.
        """
        raise NotImplementedError(
            "ExtrudedPorousGeometry.update is not available: build a new geometry "
            "(see the docstring for what a depth update would have to refresh)."
        )
