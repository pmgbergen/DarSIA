"""Image class.

Images contain the image array, and in addition metadata about origin and dimensions.

"""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, cast
from warnings import warn

from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia


class Image:
    """General image class."""

    # ! ---- Constructors

    def __init__(
        self,
        img: np.ndarray,
        transformations: Optional[list] = None,
        **kwargs,
    ) -> None:
        """Initalization of a physical space-time image.

        Allows for scalar and vector-values  2d, 3d, 4d, images, including
        time-slices as well as time-series. The boolean flag 'scalar' stores
        whether the data is stored in an additional dimension of the Image or not
        (note scalar data can be in general also encoded as multichromatic image
        with 1d data). Furthermore, 'series' holds this information, while

        Args:
            img (array): space_dim+time_dim+range_dim space-time data array
            transformations (list of callable): transformations as reduction
                and correction routines. Called in order.
            kwargs:
                keyword arguments controlling many of the attributes, mostly
                having default values targeting conventional optical images.

        Attributes:
            dim (int): dimensionality of the physical space
            scalar (boolean): flag storing whether data is scalar-valued and does
                effectivley does not use any extra axis.
            series (boolean): flag storing whether the array is a space-time array
            indexing (str): axis indexing of the first dim entries
            img (array): (space-time) image array
            date (list): absolute times for all slices
            time (list): relative times for all slices

        Example:
            multichromatic_3d_image_series = np.array((Nx, Ny, Nz, Nt, Nd), dtype=float)
            image = darsia.Image(
                multichromatic_3d_image_series,
                scalar = False,
                series = True,
                dim = 3
            )

        """

        # ! ---- Cache data
        self.img = img
        """Data array."""

        self.shape = img.shape  # TODO - not attribute of self.
        """Shape of the image."""

        self.original_dtype = img.dtype
        """Original dtype at construction of the object."""

        # ! ---- Spatial meta information
        self.space_dim: int = kwargs.get("dim", 2)
        """Dimension of the spatial domain."""

        self.space_num: int = np.prod(self.shape[: self.space_dim])
        """Spatial resolution, i.e., number of voxels."""

        self.indexing = kwargs.get("indexing", "ijk"[: self.space_dim])
        """Indexing of each axis in context of matrix indexing (ijk)
        or Cartesian coordinates (xyz)."""

        # NOTE: For now, only anticipate matrix indexing.
        assert self.indexing == "ijk"[: self.space_dim]

        self.dimensions: list[float] = kwargs.get("dimensions", self.space_dim * [1])
        """Dimension in the directions corresponding to the indexings."""

        # Accept keywords 'dimensions' and 'height', 'width', 'depth', with the latter
        # over-writing the former. In both 2d and 3d, the three keywords address the
        # first, seconds, and third (if applicable) dimension.
        if "height" in kwargs:
            self.dimensions[0] = kwargs.get("height")
        if "width" in kwargs:
            self.dimensions[1] = kwargs.get("width")
        if "depth" in kwargs:
            self.dimensions[2] = kwargs.get("depth")

        self.num_voxels: int = self.img.shape[: self.space_dim]
        """Number of voxels in each dimension."""

        self.voxel_size: list[float] = [
            self.dimensions[i] / self.num_voxels[i] for i in range(self.space_dim)
        ]
        """Size of each voxel in each direction, ordered as indexing."""

        default_origin = self.space_dim * [0]
        for index_counter, index in enumerate(self.indexing):
            axis, reverse_axis = darsia.interpret_indexing(
                index, "xyz"[: self.space_dim]
            )
            if reverse_axis:
                default_origin[axis] = self.dimensions[index_counter]
        self.origin = np.array(kwargs.pop("origin", default_origin))
        """Cartesian coordinates associated to the [0,0,0] voxel (after
        applying transformations), using Cartesian indexing."""

        self.coordinatesystem: darsia.CoordinateSystem = darsia.CoordinateSystem(self)
        """Physical coordinate system with equipped transformation from voxel to
        Cartesian space."""

        # ! ---- Temporal meta information
        self.series = kwargs.get("series", False)
        """Flag controlling whether the data array corresponds to a time series."""

        if self.series:
            self.time_dim = 1
            """Dimensionality of the image in temporal sense."""

            self.time_num = self.img.shape[self.space_dim]
            """Number of time points."""

        else:
            self.time_dim = 0
            self.time_num = 1

        # ! ---- Add absolute time data in datetime format

        default_date = self.time_num * [None] if self.series else None
        date: Union[Optional[datetime], list[Optional[datetime]]] = kwargs.get(
            "date", default_date
        )
        self.date = date
        """Time in datetime format."""

        # ! ---- Retrieve relative time from absolute date

        self.time = None
        """Relative time in scalar format (in seconds)."""

        time: Optional[Union[float, int, list]] = kwargs.pop("time", None)
        reference_date: Optional[datetime] = kwargs.pop("reference_date", None)
        self.set_time(time, reference_date)

        # ! ---- Time related safety check

        # Require definition of some time.
        if self.series:
            assert not (self._is_none(self.date) and self._is_none(self.time))

        # ! ---- Data meta information
        self.scalar = kwargs.get("scalar", False)
        """Flag controlling whether the data array is scalar, i.e., it does not
        use an extra axis to encode the range."""

        if self.scalar:
            self.range_dim: int = 0
            """Dimensionality of the image in data sense."""

            self.range_num: int = 1
            """Number of entries for each data entry."""

        else:
            self.range_dim = len(self.shape[self.space_dim + self.time_dim :])
            self.range_num = np.prod(self.shape[self.space_dim + self.time_dim :])

        # ! ---- Apply transformations

        # NOTE: Require mapping format: darsia.Image -> darsia.Image
        if transformations is not None:
            for transformation in transformations:
                if transformation is not None and hasattr(transformation, "__call__"):
                    transformation(self)

        # ! ---- Update spatial metadata, after shape-altering transformations
        self.space_num: int = np.prod(self.shape[: self.space_dim])
        """Spatial resolution, i.e., number of voxels."""

        self.num_voxels: list[int] = list(self.img.shape[: self.space_dim])
        """Number of voxels in each dimension."""

        self.voxel_size: list[float] = [
            self.dimensions[i] / self.num_voxels[i] for i in range(self.space_dim)
        ]
        """Size of each voxel in each direction, ordered as indexing."""

        self.coordinatesystem: darsia.CoordinateSystem = darsia.CoordinateSystem(self)
        """Physical coordinate system with equipped transformation from voxel to
        Cartesian space."""

        self.opposite_corner = self.coordinatesystem.coordinate(
            self.shape[: self.space_dim]
        )
        """Cartesian coordinate of the corner opposite to origin."""

        # ! ---- Safety check on dimensionality and resolution of image.
        assert len(self.shape) == self.space_dim + self.time_dim + self.range_dim
        assert np.prod(self.shape) == self.space_num * self.time_num * self.range_num

    def set_time(
        self,
        time: Optional[Union[float, int, list]] = None,
        reference_date: Optional[datetime] = None,
        reference_time: Optional[float] = None,
    ) -> None:
        """Setter for time array.

        Args:
            time (scalar or list, optional): time to be set; if None, time is retrieved
                from date.
            reference_date (datetime, optional): reference date.
            reference_time (float, optional): reference_time in seconds.

        """
        # ! ---- Safety check

        # Only allow to use one of the two input arguments.
        if reference_date is not None and reference_time is not None:
            raise ValueError("Choose only one reference.")

        if time is None:
            # From date
            if self.series:
                if self._is_none(self.date):
                    self.time = None
                else:
                    self.time = [
                        (self.date[i] - self.date[0]).total_seconds()
                        for i in range(self.time_num)
                    ]
            else:
                if reference_date is None:
                    self.time = None
                else:
                    self.time = (self.date - reference_date).total_seconds()

        else:
            # From argument
            self.time = time

        # Correct for reference time
        if reference_time is not None:
            if isinstance(self.time, list):
                self.time = [time - reference_time for time in self.time]
            elif self.time is not None:
                self.time -= reference_time

    def copy(self) -> Image:
        """Copy constructor.

        Returns:
            Image: Copy of the image object.

        """
        return copy.deepcopy(self)

    def append(self, image: Image, offset: Optional[Union[float, int]] = None) -> None:
        """Append other image to current image. Makes in particular
        a non-space-time image to a space-time image.

        Args:
            image (Image): image to be appended.
            offset (float or int, optional): time increment between last and next slice.

        """

        # ! ---- Safety checks
        assert self.space_dim == image.space_dim
        assert self.scalar == image.scalar
        assert np.allclose(np.array(self.num_voxels), np.array(image.num_voxels))
        if not self._is_none(self.date) and not self._is_none(image.date):
            self_last_date = self.date[-1] if self.series else self.date
            image_first_date = image.date[0] if image.series else image.date
            assert self_last_date < image_first_date
        assert np.allclose(np.array(self.dimensions), np.array(image.dimensions))
        assert np.allclose(self.origin, image.origin)

        # ! ---- Update data

        # Auxiliary routine for slicing images
        def slice_image(im: Image) -> list[np.ndarray]:
            if im.series:
                slices = [
                    im.img[..., i] if im.scalar else im.img[..., i, :]
                    for i in range(im.time_num)
                ]
            else:
                slices = [im.img]
            return slices

        # Slice the current and input images
        slices = slice_image(self) + slice_image(image)

        # Stack images together
        self.img = np.stack(slices, axis=self.space_dim)
        self.series = True

        # ! ---- Update time

        # Time in datetime format
        if not isinstance(self.date, list):
            self.date = [self.date]
        if isinstance(image.date, list):
            self.date = self.date + image.date
        else:
            self.date.append(image.date)

        # Relative time - combine internal stored times
        if self._is_none(self.time) or self._is_none(image.time) or offset is None:
            time = None
        else:
            # Append relative times, plus offset
            time = self.time if isinstance(self.time, list) else [self.time]
            if isinstance(image.time, list):
                time = time + [t + offset for t in image.time]
            else:
                time.append(image.time + offset)

        # Specs
        self.time_dim = 1
        self.time_num = len(time)

        # Update relative time
        self.set_time(time)

    # ! ---- Transformations

    def resize(self, cx: float, cy: Optional[float] = None) -> None:
        raise NotImplementedError

    def astype(self, data_type) -> Any:
        """For scalar data types, change the data type of the data array.
        For Image data types, cast the entire image.

        Args:
            data_type: target data type

        Returns:
            Image: image with transformed data type

        """
        copy_image = self.copy()
        if data_type in [
            int,
            float,
            np.uint8,
            np.uint16,
            np.float16,
            np.float32,
            np.float64,
            bool,
        ]:
            copy_image.img = copy_image.img.astype(data_type)
        else:
            # TODO test
            copy_image = cast(data_type, copy_image)
            raise NotImplementedError

        return copy_image

    # ! ---- Extraction routines

    def metadata(self) -> dict:
        """Return all metadata required to initiate an image via keyword
        arguments.

        Returns:
            dict: metadata with keys equal to all keywords arguments.

        """
        metadata = {
            "dim": self.space_dim,
            "indexing": self.indexing,
            "dimensions": self.dimensions,
            "origin": self.origin,
            "series": self.series,
            "scalar": self.scalar,
            "date": self.date,
            "time": self.time,
        }
        return copy.copy(metadata)

    def shape_metadata(self) -> dict:
        """Return all metadata connected to dimensions of image and voxels.

        Useful to define darsia.Geometry from darsia.Image.

        Returns:
            dict: metadata with keys to instantiate a Geometry.

        """
        metadata = {
            "space_dim": self.space_dim,
            "dimensions": self.dimensions,
            "num_voxels": self.num_voxels,
            "voxel_size": self.voxel_size,
        }
        return metadata

    def time_slice(self, time_index: int) -> Image:
        """Extraction of single time slice.

        Args:
            time_index (int): time index in interval [0, ..., time_num-1].

        Returns:
            Image: single-timed image.

        """
        if not self.series:
            raise ValueError

        # Fetch data and return corresponding datatype
        if self.scalar:
            img = self.img[..., time_index]

        else:
            img = self.img[..., time_index, :]

        # Fetch and update metadata
        metadata = self.metadata()
        metadata["series"] = False
        metadata["date"] = self.date[time_index]
        if self.time is None:
            metadata["time"] = None
        else:
            metadata["time"] = self.time[time_index]

        # Create image with same data type but updates image data and metadata
        return type(self)(img=img, **metadata)

    def time_interval(
        self,
        time_indices: Optional[slice] = None,
        reset_time: bool = False,
    ) -> Image:
        """Extraction of temporal subregion.

        Args:
            time_indices (slice, optional): time interval; only for space-time images.
            reset_time (bool): flag controlling whether relative time is reset.

        Returns:
            Image: image with restricted temporal domain.

        Raises:
            ValueError: if image is not a time series.

        """
        # ! ---- Safety checks

        if time_indices is not None and not self.series:
            raise ValueError("Image is not a time-series.")

        # ! ---- Fetch data
        if self.scalar:
            img = self.img[..., time_indices]
        else:
            img = self.img[..., time_indices, :]

        # ! ---- Update metadata

        # ! ---- Fetch and adapt metadata
        metadata = self.metadata()
        metadata["date"] = self.date[time_indices]
        times = self.time[time_indices]
        if reset_time:
            metadata["time"] = [time - times[0] for time in times]
        else:
            metadata["time"] = times

        return type(self)(img=img, **metadata)

    def subregion(
        self,
        voxels: Optional[tuple[slice]] = None,
        coordinates: Optional[np.ndarray] = None,
    ) -> Image:
        """Extraction of spatial subregion.

        Args:
            voxels (tuple of slices, optional): voxel intervals in all dimensions.
            coordinates (array, optional): points in space, in Cartesian coordinates,
                uniquely defining a box.

        Returns:
            Image: image with restricted spatial domain.

        Raises:
            ValueError: if both voxels and coordinates are not None, or if neither
                voxels nor coordinates are provide

        """
        # ! ---- Safety checks

        if (voxels is not None) == (coordinates is not None):
            raise ValueError("Use (only) one way of specifying the subregion.")

        # ! ---- Translate coordinates to voxels

        if coordinates is not None:
            # Translate coordinates to voxels
            voxels_box = self.coordinatesystem.voxel(coordinates)

            # Extract slices
            voxels = [
                slice(
                    max(0, np.min(voxels_box[:, d])),
                    min(np.max(voxels_box[:, d]), self.num_voxels[d]),
                )
                for d in range(self.space_dim)
            ]

            # Convert to tuple
            voxels = tuple(voxels)

        assert len(voxels) == self.space_dim

        # ! ---- Extract dimensions and new origin from voxels

        origin_voxel = [0 if sl.start is None else sl.start for sl in voxels]
        origin = self.coordinatesystem.coordinate(origin_voxel)

        opposite_voxel = [
            self.num_voxels[i] if sl.stop is None else sl.stop
            for i, sl in enumerate(voxels)
        ]
        opposite = self.coordinatesystem.coordinate(opposite_voxel)

        cartesian_dimensions = np.absolute(opposite - origin)
        dimensions = []
        for matrix_index in range(self.space_dim):
            axis = "ijk"[matrix_index]
            indexing = "xyz"[: self.space_dim]
            cartesian_index, _ = darsia.interpret_indexing(axis, indexing)
            dimensions.append(cartesian_dimensions[cartesian_index])

        # ! ---- Fetch data
        img = self.img[voxels]

        # ! ---- Fetch and adapt metadata
        metadata = self.metadata()
        metadata["dimensions"] = dimensions
        metadata["origin"] = origin

        return type(self)(img=img, **metadata)

    # ! ---- Arithmetics

    def __add__(self, other: Image) -> Image:
        """Add two images of same size.

        Arguments:
            other (Image): image to subtract from self

        Returns:
            Image: sum of images

        """
        if self.img.shape != other.img.shape:
            raise ValueError("Images have different shapes.")
        else:
            metadata = self.metadata()
            return type(self)(self.img + other.img, **metadata)

    def __sub__(self, other: Image) -> Image:
        """Subtract two images of same size.

        Arguments:
            other (Image): image to subtract from self

        Returns:
            Image: difference image

        """
        if self.img.shape != other.img.shape:
            raise ValueError("Images have different shapes.")
        else:
            metadata = self.metadata()
            return type(self)(self.img - other.img, **metadata)

    def __mul__(self, scalar: Union[float, int]) -> Image:
        """Scaling of image.

        Arguments:
            scalar (float or int): scaling parameter

        Returns:
            Image: scaled image

        """
        if not isinstance(scalar, float) or isinstance(scalar, int):
            raise ValueError

        result_image = self.copy()
        result_image.img *= scalar
        return result_image

    __rmul__ = __mul__

    # TODO add more standard routines with NotImplementedError
    # TODO try TVD.

    # ! ---- Display methods and I/O

    def show(
        self,
        name: str = "",
        duration: Optional[int] = None,
        show: bool = True,
        **kwargs,
    ) -> None:
        """Show image using matplotlib.pyplots built-in methods.

        Args:
            name (str): name in the displayed window.
            duration (int, optional): display duration in seconds.
            show (bool): flag controlling whether the plot is forced to be displayed.
            **kwargs: additional arguments passed to matplotlib.pyplot.
                thresh (float): threshold for displaying the image; only for 3d images.
                relative (bool): flag controlling whether the threshold is relative.
                side_view (str): side view type of 3d image; only for 3d images;
                    either "scatter" or "integrated".

        """

        for time_index in range(self.time_num):
            if self.series:
                # Fetch time for series (not used otherwise)
                abs_time = (
                    ""
                    if self.date[time_index] is None
                    else " - " + str(self.date[time_index])
                )
                rel_time = (
                    ""
                    if self.time[time_index] is None
                    else " - " + str(self.time[time_index])
                )

                # Append name with time
                _name = name
                if not _name == "":
                    _name += " - "
                _name += f"{time_index} - {abs_time} -  {rel_time} sec."

            # Plot the entire 2d image in plain mode
            if self.space_dim == 2:
                # Only works for scalar and optical images.
                assert self.scalar or self.range_num in [1, 3]

                # Fetch data array
                if self.series:
                    if self.scalar:
                        c = self.img[..., time_index]
                    else:
                        c = self.img[..., time_index, :]
                else:
                    c = self.img

                # Plot
                plt.figure(name)
                plt.imshow(skimage.img_as_float(c))

            elif self.space_dim == 3:
                # Only works for scalar images.
                assert self.scalar

                # Fetch bounding box
                corners = np.vstack((self.origin, self.opposite_corner))
                bbox = np.array([np.min(corners, axis=0), np.max(corners, axis=0)])

                # Extract physical coordinates and flatten
                matrix_indices = np.transpose(
                    np.indices(self.img.shape[:3]).reshape((3, -1))
                )
                coordinates = self.coordinatesystem.coordinate(matrix_indices)

                # Extract values
                if self.series:
                    c = self.img[..., time_index].reshape((1, -1))[0]
                    time_slice = self.time_slice(time_index)
                else:
                    c = self.img.reshape((1, -1))[0]
                    time_slice = self

                # Restrict to active voxels
                thresh = kwargs.get("thresh", np.min(self.img))
                relative = kwargs.get("relative", False)
                if relative:
                    thresh = thresh * np.max(self.img)
                active = c > thresh

                # Signal strength
                alpha = 0.9 * (c - np.min(c)) / (np.max(c) - np.min(c)) + 0.1
                scaling = kwargs.get("scaling", 1)
                s = scaling * alpha

                # 3d view
                fig_3d = plt.figure(name + " - 3d view")
                ax_3d = Axes3D(fig_3d)
                ax_3d.scatter(
                    xs=coordinates[active, 0],
                    ys=coordinates[active, 1],
                    zs=coordinates[active, 2],
                    s=s[active],
                    alpha=np.power(alpha[active], 2),
                    c=c[active],
                    cmap="viridis",
                )
                ax_3d.set_xlabel("x-axis")
                ax_3d.set_ylabel("y-axis")
                ax_3d.set_zlabel("z-axis")
                ax_3d.set_xlim(bbox[0, 0], bbox[1, 0])
                ax_3d.set_ylim(bbox[0, 1], bbox[1, 1])
                ax_3d.set_zlim(bbox[0, 2], bbox[1, 2])

                # 2d side views. Offer two possibilities. Either a scatter plot or an
                # integrated view. The latter uses integration over the axis "into" the
                # screen.
                side_view = kwargs.get("side_view", "scatter").lower()
                assert side_view in ["scatter", "integrated"]

                fig_2d_xy = plt.figure(name + " - 2d side view - x-y")
                if side_view == "scatter":
                    plt.scatter(
                        coordinates[active, 0],
                        coordinates[active, 1],
                        s=s[active],
                        alpha=alpha[active],
                        c=c[active],
                        cmap="viridis",
                    )
                    plt.xlim(bbox[0, 0], bbox[1, 0])
                    plt.ylim(bbox[0, 1], bbox[1, 1])
                elif side_view == "integrated":
                    reduction = darsia.AxisAveraging(axis="z", dim=3)
                    reduced_image = reduction(time_slice)
                    plt.imshow(skimage.img_as_float(reduced_image.img))
                plt.xlabel("x-axis")
                plt.ylabel("y-axis")

                fig_2d_xz = plt.figure(name + " - 2d side view - x-z")
                if side_view == "scatter":
                    plt.scatter(
                        coordinates[active, 0],
                        coordinates[active, 2],
                        s=s[active],
                        alpha=alpha[active],
                        c=c[active],
                        cmap="viridis",
                    )
                    plt.xlim(bbox[0, 0], bbox[1, 0])
                    plt.ylim(bbox[0, 2], bbox[1, 2])
                elif side_view == "integrated":
                    reduction = darsia.AxisAveraging(axis="y", dim=3)
                    reduced_image = reduction(time_slice)
                    plt.imshow(skimage.img_as_float(reduced_image.img))
                plt.xlabel("x-axis")
                plt.ylabel("z-axis")

                plt.figure(name + " - 2d side view - y-z")
                if side_view == "scatter":
                    plt.scatter(
                        coordinates[active, 1],
                        coordinates[active, 2],
                        s=s[active],
                        alpha=alpha[active],
                        c=c[active],
                        cmap="viridis",
                    )
                    plt.xlim(bbox[0, 1], bbox[1, 1])
                    plt.ylim(bbox[0, 2], bbox[1, 2])
                elif side_view == "integrated":
                    reduction = darsia.AxisAveraging(axis="x", dim=3)
                    reduced_image = reduction(time_slice)
                    plt.imshow(skimage.img_as_float(reduced_image.img))
                plt.xlabel("y-axis")
                plt.ylabel("z-axis")

            if duration is None:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(int(duration))
                plt.close()

    def write_metadata(self, path: Union[str, Path]) -> None:
        """
        Writes the metadata dictionary to a json-file.

        Arguments:
            path (str): path to the json file

        """
        metadata = self.extract_metadata()
        with open(Path(path), "w") as outfile:
            json.dump(metadata, outfile, indent=4)

    def write_array(
        self,
        path: Union[str, Path],
        indexing: str = "matrix",
        allow_pickle: bool = True,
    ) -> None:
        """Auxiliary routine for storing the current image array as npy array.

        Args:
            path (Path): path to file.
            indexing (str): If "matrix", the array is stored using matrix indexing,
                if "Cartesian", the array is stored using Cartesian indexing.
            allow_pickle (bool): Flag controlling whether pickle is allowed.

        """
        assert indexing.lower() in ["matrix", "cartesian"]

        Path(path).parents[0].mkdir(parents=True, exist_ok=True)

        plain_path = Path(path).with_suffix("")

        np.save(
            str(plain_path) + ".npy",
            darsia.matrixToCartesianIndexing(self.img)
            if indexing.lower() == "cartesian"
            else self.img,
            allow_pickle=allow_pickle,
        )

    # ! ---- Auxiliary routines

    def _is_none(self, item) -> bool:
        """Repeated routine used to check the status of time and date
        attributes.

        Returns:
            bool: True if item is None or a list containing None.

        """
        if isinstance(item, list):
            return None in item
        else:
            return item is None


class ScalarImage(Image):
    """Special case of a space-time image, with 1d data, e.g., monochromatic
    photographs.

    """

    # ! ---- Constructors.

    def __init__(
        self,
        img: np.ndarray,
        transformations: Optional[list] = None,
        **kwargs,
    ) -> None:
        """Specialized constructor for optical images.

        In addition to the constructor of general images,
        the color space is required to be specified through
        the keyword arguments.

        """
        # Define metadata specific for optical images
        scalar_metadata = {
            "scalar": True,
        }
        kwargs.pop("scalar", None)

        self.name = kwargs.get("name", None)
        """Name of image, e.g., used to describe the origin."""

        # Construct a general image with the specs of an optical image
        super().__init__(img, transformations, **scalar_metadata, **kwargs)

        assert self.range_dim == 0

    def copy(self) -> ScalarImage:
        """Copy constructor.

        Returns:
            ScalarImage: Copy of the image object.

        """
        return copy.deepcopy(self)

    # ! ---- Fast access.

    def metadata(self) -> dict:
        """Generator of metadata; can be used to init a new optical image with same
        specs.

        Returns:
            dict: metadata with keys equal to all keywords agurments.

        """
        # Start with generic metadata.
        metadata = super().metadata()

        # Add specs specific to optical images.
        metadata["name"] = self.name

        return copy.copy(metadata)

    # ! ---- I/O

    def write(self, path: Path, **kwargs) -> None:
        """Write image to file.

        Arguments:
            path (Path): full path to image.
            keyword arguments:
                quality (int): number between 0 and 100, indicating
                    the resolution used to store a jpg image
                compression (int): number between 0 and 9, indicating
                    the level of compression used for storing in
                    png format.

        """
        # Write image, using the conventional matrix indexing
        ubyte_image = skimage.img_as_ubyte(self.img)
        suffix = Path(path).suffix.lower()

        if suffix in [".jpg", ".jpeg"]:
            quality = kwargs.get("quality", 90)
            cv2.imwrite(
                str(Path(path)), ubyte_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
        elif suffix == ".png":
            compression = kwargs.get("compression", 6)
            cv2.imwrite(
                str(Path(path)), ubyte_image, [cv2.IMWRITE_PNG_COMPRESSION, compression]
            )
        else:
            cv2.imwrite(str(Path(path)), ubyte_image)

        print("Image saved as: " + str(Path(path)))


class OpticalImage(Image):
    """Special case of 2d trichromatic optical images, typically originating from
    photographs.

    """

    # ! ---- Constructors.

    def __init__(
        self,
        img: np.ndarray,
        transformations: Optional[list] = None,
        **kwargs,
    ) -> None:
        """Specialized constructor for optical images.

        In addition to the constructor of general images,
        the color space is required to be specified through
        the keyword arguments.

        """
        # Define metadata specific for optical images
        optical_metadata = {
            "dim": 2,
            "indexing": "ij",
            "scalar": False,
        }
        kwargs.pop("dim", None)
        kwargs.pop("indexing", None)
        kwargs.pop("scalar", None)

        # Construct a general image with the specs of an optical image
        super().__init__(img, transformations, **optical_metadata, **kwargs)

        assert self.range_dim == 1 and self.range_num == 3

        # Add info on color space
        self.color_space = kwargs.get("color_space", "RGB").upper()
        """Color space of the trichromatic data space."""

        if self.color_space not in ["RGB", "BGR", "HSV"]:
            raise NotImplementedError

        if "color_space" not in kwargs:
            warn("No color space provided. The color space RGB is implicitly assumed.")

    def copy(self) -> OpticalImage:
        """Copy constructor.

        Returns:
            OpticalImage: Copy of the image object.

        """
        return copy.deepcopy(self)

    # ! ---- Fast access.

    def metadata(self) -> dict:
        """Generator of metadata; can be used to init a new optical image with same
        specs.

        Returns:
            dict: metadata with keys equal to all keywords agurments.

        """
        # Start with generic metadata.
        metadata = super().metadata()

        # Add specs specific to optical images.
        metadata["color_space"] = self.color_space

        return copy.copy(metadata)

    # ! ---- I/O

    def write(self, path: Path, **kwargs) -> None:
        """Write image to file.

        Arguments:
            path (Path): full path to image.
            keyword arguments:
                quality (int): number between 0 and 100, indicating
                    the resolution used to store a jpg image
                compression (int): number between 0 and 9, indicating
                    the level of compression used for storing in
                    png format.

        """
        Path(path).parents[0].mkdir(parents=True, exist_ok=True)

        # To prepare for the use of cv2.imwrite, convert to BGR color space.
        bgr_image = self.to_bgr(return_image=True)
        bgr_array = bgr_image.img

        # Write image, using the conventional matrix indexing
        if self.original_dtype == np.uint8:
            ubyte_image = skimage.img_as_ubyte(bgr_array)
            suffix = Path(path).suffix.lower()

            if suffix in [".jpg", ".jpeg"]:
                quality = kwargs.get("quality", 90)
                cv2.imwrite(
                    str(Path(path)),
                    ubyte_image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), quality],
                )
            elif suffix == ".png":
                compression = kwargs.get("compression", 6)
                cv2.imwrite(
                    str(Path(path)),
                    ubyte_image,
                    [cv2.IMWRITE_PNG_COMPRESSION, compression],
                )
            else:
                cv2.imwrite(str(Path(path)), ubyte_image)

        elif self.original_dtype == np.uint16:
            ubyte_image = skimage.img_as_uint(bgr_array)
            cv2.imwrite(
                str(Path(path)),
                ubyte_image,
                [
                    cv2.IMWRITE_TIFF_COMPRESSION,
                    1,
                    cv2.IMWRITE_TIFF_XDPI,
                    350,
                    cv2.IMWRITE_TIFF_YDPI,
                    350,
                ],
            )

        else:
            raise NotImplementedError

        print("Image saved as: " + str(Path(path)))

    # ! ---- Color transformations

    def to_trichromatic(self, color_space: str, return_image: bool = False) -> None:
        """Transforms image to another trichromatic color space.

        Args:
            color_space: target color space.
            return_image (bool): flag controlling whether the converted image
                is returned, or converted internally.

        Returns:
            OpticalImage (optional): converted image, if requested via 'return_image'.

        """

        if color_space.upper() not in ["RGB", "BGR", "HSV"]:
            raise NotImplementedError

        if color_space.upper() == self.color_space:
            img: np.ndarray = self.img.copy()
        else:
            option = eval("cv2.COLOR_" + self.color_space + "2" + color_space.upper())
            if self.series:
                slices = []
                for time_index in range(self.time_num):
                    slices.append(cv2.cvtColor(self.img[..., time_index, :], option))
                img = np.stack(slices, axis=self.space_dim)

            else:
                img = cv2.cvtColor(self.img, option)

        # If an image is returned, do not change the image attribute.
        if return_image:
            image = self.copy()
            image.img = img
            image.color_space = color_space.upper()
            return image
        else:
            self.img = img
            self.color_space = color_space.upper()

    def to_monochromatic(self, key: str) -> ScalarImage:
        """Returns monochromatic version of the image.

        Returns:
            ScalarImage: monochromatic image.

        """
        # Do not alter underlying image, as this operation cannot be reversed.
        image = self.copy()

        # Add robustness.
        key = key.lower()

        # Adapt data array
        if key == "gray":
            option = eval("cv2.COLOR_" + self.color_space + "2GRAY")
            if self.series:
                slices = []
                for time_index in range(self.time_num):
                    slices.append(cv2.cvtColor(self.img[..., time_index, :], option))
                img = np.stack(slices, axis=self.space_dim)

            else:
                img = cv2.cvtColor(self.img, option)

        elif key in ["red", "green", "blue"]:
            image.to_trichromatic("rgb")
            if key == "red":
                img = image.img[..., 0]
            elif key == "green":
                img = image.img[..., 1]
            elif key == "blue":
                img = image.img[..., 2]

        elif key in ["hue", "saturation", "value"]:
            image.to_trichromatic("hsv")
            if key == "hue":
                img = image.img[..., 0]
            elif key == "saturation":
                img = image.img[..., 1]
            elif key == "value":
                img = image.img[..., 2]

        # Adapt specs.
        metadata = image.metadata()
        del metadata["color_space"]
        metadata["name"] = key
        metadata["scalar"] = True

        # Return scalar image
        return ScalarImage(img, **metadata)

    # ! ---- Utilities

    def add_grid(
        self,
        origin: Optional[Union[np.ndarray, list[float]]] = None,
        dx: float = 1,
        dy: float = 1,
        color: tuple = (0, 0, 125),
        thickness: int = 9,
    ) -> OpticalImage:
        """
        Adds a grid on the image and returns new image.

        Arguments:
            origin (np.ndarray): origin of the grid, in physical units - the reference
                coordinate system is provided by the corresponding attribute
                coordinatesystem
            dx (float): grid size in x-direction, in physical units
            dy (float): grid size in y-direction, in physical units
            color (tuple of int): BGR color of the grid
            thickness (int): thickness of the grid lines

        Returns:
            OpticalImage: original image with grid on top

        """
        # Set origin if it was not provided
        if origin is None:
            origin = self.origin
        origin = np.array(origin)

        # Determine the number of grid lines required
        num_horizontal_lines: int = math.ceil(self.dimensions[0] / dy) + 1
        num_vertical_lines: int = math.ceil(self.dimensions[1] / dx) + 1

        # Start from original image
        gridimg: np.array = self.img.copy()
        metadata = self.metadata()

        # Add horizontal grid lines (line by line)
        for i in range(num_horizontal_lines):
            # Determine the outer boundaries in x directions
            xmin = self.coordinatesystem.domain["xmin"]
            xmax = self.coordinatesystem.domain["xmax"]

            # Determine the y coordinate of the line
            y = origin[1] + i * dy

            # Determine the pixels corresponding to the end points of the horizontal
            # line (xmin,y) - (xmax,y), in (row,col) format.
            start = self.coordinatesystem.voxel([xmin, y])
            end = self.coordinatesystem.voxel([xmax, y])

            # Add single line. NOTE: cv2.line takes pixels as inputs with the reversed
            # matrix indexing, i.e., (col,row) instead of (row,col). Furthermore,
            # it requires tuples.
            gridimg = cv2.line(
                gridimg, tuple(reversed(start)), tuple(reversed(end)), color, thickness
            )

        # Add vertical grid lines (line by line)
        for j in range(num_vertical_lines):
            # Determine the outer boundaries in x directions
            ymin = self.coordinatesystem.domain["ymin"]
            ymax = self.coordinatesystem.domain["ymax"]

            # Determine the y coordinate of the line
            x = origin[0] + j * dx

            # Determine the pixels corresponding to the end points of the vertical line
            # (x, ymin) - (x, ymax), in (row,col) format.
            start = self.coordinatesystem.voxel([x, ymin])
            end = self.coordinatesystem.voxel([x, ymax])

            # Add single line. NOTE: cv2.line takes pixels as inputs with the reversed
            # matrix indexing, i.e., (col,row) instead of (row,col). Furthermore,
            # it requires tuples.
            gridimg = cv2.line(
                gridimg, tuple(reversed(start)), tuple(reversed(end)), color, thickness
            )

        # Return image with grid as Image object
        return OpticalImage(img=gridimg, **metadata)
