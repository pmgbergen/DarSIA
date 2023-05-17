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

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import skimage
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots

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

        self.num_voxels: tuple[int] = self.img.shape[: self.space_dim]
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

        # NOTE: Ideally, we would like to have a time array for each slice, but this
        # turns out to be a bit cumbersome. Depending on the operating system,
        # obtaining the time stamp of a file is not always possible (in particular on
        # Windows OS). Therefore, the assert is deactivated. Errors may occur later on.
        if self.series and (self._is_none(self.date) and self._is_none(self.time)):
            warn("No time information provided for the image.")

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
                    transformation(self, overwrite=True)

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
        self.time_num += image.time_num

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
            # Do nothing
            return self

        # ! ---- Translate coordinates to voxels

        if coordinates is not None:
            # Translate coordinates to voxels
            voxels_box = self.coordinatesystem.voxel(coordinates)

            # Extract slices
            voxels: tuple[slice] = tuple(
                slice(
                    max(0, np.min(voxels_box[:, d])),
                    min(np.max(voxels_box[:, d]), self.num_voxels[d]),
                )
                for d in range(self.space_dim)
            )

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
        title: str = "",
        duration: Optional[int] = None,
        mode: str = "matplotlib",
        **kwargs,
    ) -> None:
        """Show image using matplotlib.pyplots or plotly built-in methods. The latter
        often is faster.

        Args:
            title (str): title in the displayed window.
            duration (int, optional): display duration in seconds.
            mode (str): display mode; either "matplotlib" or "plotly".
            **kwargs: additional arguments passed to show_matplotlib or show_plotly.

        """
        assert mode in ["matplotlib", "plotly"], "Unknown mode."

        if mode == "matplotlib":
            self.show_matplotlib(title, duration, **kwargs)
        elif mode == "plotly":
            self.show_plotly(title, duration, **kwargs)

    def show_matplotlib(
        self,
        title: str = "",
        duration: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Show routine using matplotlib.pyplot built-in methods.

        Args:
            title (str): title in the displayed window.
            duration (int, optional): display duration in seconds.
            **kwargs: additional arguments passed to matplotlib.pyplot.
                threshold (float): threshold for displaying 3d images.
                relative (bool): flag controlling whether the threshold is relative.
                view (str): view type; either "scatter" or "voxel"; only for 3d images.
                    NOTE: "voxel" plots are more time consuming.
                side_view (str): side view type of 3d image; only for 3d images;
                    either "scatter" or "voxel".
                surpress_2d (bool): flag controlling whether 2d images are displayed.
                surpress_3d (bool): flag controlling whether 3d images are displayed.
                delay (bool): flag controlling whether the display is delayed; can be
                    used to display multiple images at the same time.

        """

        # Use different plotting styles for different spatial dimensions. In 1d, time
        # series can be visualized in a single plot, and thus receive special treatment.
        if self.space_dim == 1:

            # Extract physical coordinates and flatten
            matrix_indices = np.transpose(
                np.indices(self.img.shape[:1]).reshape((1, -1))
            )
            coordinates = self.coordinatesystem.coordinate(matrix_indices)

            # Fetch data and make sure to expand dimensions to have explicit access to
            # the range components - simplifies code generalization.
            if self.scalar:
                array = self.img[..., np.newaxis]
            else:
                array = self.img

            # Generate the plot(s)
            if self.series:
                # Plot evolution of each component in separate plot.
                for comp in range(self.range_num):
                    fig, ax = plt.subplots(1)
                    fig.suptitle(title + f" component {comp}")
                    # Put time on y axis and start time 0 on the bottom.
                    plt.imshow(np.flip(np.transpose(array[..., comp]), 0))
                    # TODO replace indices in ticks with actual coordinates and time.
                    ax.set_xlabel("x-axis")
                    ax.set_ylabel("time-axis")
            else:
                # Plot each component in same plot
                fig = plt.figure(title)
                for comp in range(self.range_num):
                    plt.plot(coordinates, array[..., comp])
                plt.legend([f"Comp. {i}" for i in range(self.range_num)])

            # Show the plot(s)
            delay = kwargs.get("delay", False)
            if not delay:
                if duration is None:
                    plt.show()
                else:
                    plt.show(block=False)
                    plt.pause(int(duration))
                    plt.close()

        else:
            # In 2d and 3d, plot each time slice separately.
            for time_index in range(self.time_num):
                if self.series:
                    # Fetch time for series (not used otherwise)

                    assert self.date is None or (
                        isinstance(self.date, list)
                        and all(
                            [d is None or isinstance(d, datetime) for d in self.date]
                        )
                    )  # mypy
                    abs_time = (
                        ""
                        if not isinstance(self.date, list)
                        or self.date[time_index] is None
                        else " - " + str(self.date[time_index])
                    )

                    assert self.time is None or isinstance(self.time, list)
                    rel_time = (
                        ""
                        if self.time is None or self.time[time_index] is None
                        else " - " + str(self.time[time_index])
                    )

                    # Append title with time
                    _title = title
                    if not _title == "":
                        _title += " - "
                    _title += f"{time_index} - {abs_time} -  {rel_time} sec."
                else:
                    _title = title

                if self.space_dim == 2:
                    # Plot the entire 2d image in plain mode
                    # Only works for scalar and optical images.
                    assert self.scalar or self.range_num in [1, 3]

                    # Fetch data array
                    if self.series:
                        if self.scalar:
                            array = self.img[..., time_index]
                        else:
                            array = self.img[..., time_index, :]
                    else:
                        array = self.img

                    # Plot
                    fig = plt.figure(_title)
                    plt.imshow(skimage.img_as_float(array))

                elif self.space_dim == 3:
                    # ! --- Preliminaries

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
                        array = self.img[..., time_index]
                        time_slice = self.time_slice(time_index)
                    else:
                        array = self.img
                        time_slice = self
                    flat_array = array.reshape((1, -1))[0]

                    # Restrict to active voxels
                    threshold = kwargs.get("threshold", np.min(self.img))
                    relative = kwargs.get("relative", False)
                    if relative:
                        threshold = threshold * np.max(self.img)
                    active = flat_array > threshold

                    # Signal strength
                    alpha_min = 0.1
                    alpha = alpha_min + (
                        (1.0 - alpha_min)
                        * (flat_array - np.min(array))
                        / (np.max(array) - np.min(array))
                    )
                    scaling = kwargs.get("scaling", 1)
                    s = scaling * alpha

                    # ! ---- 3d view

                    # Offer two possibilities. Either a scatter plot or a voxel plot.

                    surpress_3d = kwargs.get("surpress_3d", False)
                    if not surpress_3d:
                        fig_3d = plt.figure(_title + " - 3d view")
                        ax_3d = Axes3D(fig_3d)

                        view = kwargs.get("view", "scatter").lower()
                        assert view in ["scatter", "voxel"]
                        if view == "scatter":
                            ax_3d.scatter(
                                xs=coordinates[active, 0],
                                ys=coordinates[active, 1],
                                zs=coordinates[active, 2],
                                s=s[active],
                                alpha=np.power(alpha[active], 2),
                                c=flat_array[active],
                                cmap="viridis",
                            )

                        elif view == "voxel":
                            # Convert coordinates into np.indices format, listing all voxel
                            # corners.
                            voxel_corners = np.indices(np.array(self.img[:3].shape) + 1)
                            reshaped_voxel_corners = np.transpose(
                                voxel_corners.reshape((3, -1))
                            )
                            reshaped_voxel_coordinates = (
                                self.coordinatesystem.coordinate(reshaped_voxel_corners)
                            )
                            voxel_coordinates = np.transpose(
                                reshaped_voxel_coordinates
                            ).reshape(voxel_corners.shape)

                            # Convert array values to colors and transfer signal strength
                            facecolors = plt.cm.viridis(array)
                            alpha_voxels = alpha_min + (1.0 - alpha_min) * (
                                array - np.min(array)
                            ) / (np.max(array) - np.min(array))
                            facecolors[..., -1] = alpha_voxels
                            active_voxels = array > threshold

                            ax_3d.voxels(
                                voxel_coordinates[0],
                                voxel_coordinates[1],
                                voxel_coordinates[2],
                                active_voxels,
                                facecolors=facecolors,
                            )

                        ax_3d.set_xlabel("x-axis")
                        ax_3d.set_ylabel("y-axis")
                        ax_3d.set_zlabel("z-axis")
                        ax_3d.set_xlim(bbox[0, 0], bbox[1, 0])
                        ax_3d.set_ylim(bbox[0, 1], bbox[1, 1])
                        ax_3d.set_zlim(bbox[0, 2], bbox[1, 2])

                    # ! ---- 2d side views

                    # Offer two possibilities. Either a scatter plot or an
                    # integrated view. The latter uses integration over the axis "into" the
                    # screen.

                    surpress_2d = kwargs.get("surpress_2d", False)
                    if not surpress_2d:
                        side_view = kwargs.get("side_view", "scatter").lower()
                        assert side_view in ["scatter", "voxel"]
                        fig_2d, axs = plt.subplots(1, 3)
                        fig_2d.suptitle("2d side views")

                        # xy-plane
                        axs[0].set_title(_title + " - x-y plane")
                        if side_view == "scatter":
                            axs[0].scatter(
                                coordinates[active, 0],
                                coordinates[active, 1],
                                s=s[active],
                                alpha=alpha[active],
                                c=flat_array[active],
                                cmap="viridis",
                            )
                            axs[0].set_xlim(bbox[0, 0], bbox[1, 0])
                            axs[0].set_ylim(bbox[0, 1], bbox[1, 1])
                        elif side_view == "voxel":
                            reduction = darsia.AxisAveraging(axis="z", dim=3)
                            reduced_image = reduction(time_slice)
                            axs[0].imshow(skimage.img_as_float(reduced_image.img))
                        axs[0].set_xlabel("x-axis")
                        axs[0].set_ylabel("y-axis")
                        axs[0].set_aspect("equal")

                        # xz-plane
                        axs[1].set_title(_title + " - x-z plane")
                        if side_view == "scatter":
                            axs[1].scatter(
                                coordinates[active, 0],
                                coordinates[active, 2],
                                s=s[active],
                                alpha=alpha[active],
                                c=flat_array[active],
                                cmap="viridis",
                            )
                            axs[1].set_xlim(bbox[0, 0], bbox[1, 0])
                            axs[1].set_ylim(bbox[0, 2], bbox[1, 2])
                        elif side_view == "voxel":
                            reduction = darsia.AxisAveraging(axis="y", dim=3)
                            reduced_image = reduction(time_slice)
                            axs[1].imshow(skimage.img_as_float(reduced_image.img))
                        axs[1].set_xlabel("x-axis")
                        axs[1].set_ylabel("z-axis")
                        axs[1].set_aspect("equal")

                        # yz-plane
                        axs[2].set_title(_title + " - y-z plane")
                        if side_view == "scatter":
                            axs[2].scatter(
                                coordinates[active, 1],
                                coordinates[active, 2],
                                s=s[active],
                                alpha=alpha[active],
                                c=flat_array[active],
                                cmap="viridis",
                            )
                            axs[2].set_xlim(bbox[0, 1], bbox[1, 1])
                            axs[2].set_ylim(bbox[0, 2], bbox[1, 2])
                        elif side_view == "voxel":
                            reduction = darsia.AxisAveraging(axis="x", dim=3)
                            reduced_image = reduction(time_slice)
                            axs[2].imshow(skimage.img_as_float(reduced_image.img))
                        axs[2].set_xlabel("y-axis")
                        axs[2].set_ylabel("z-axis")
                        axs[2].set_aspect("equal")

                    # Make sure that any plot is shown
                    assert not (surpress_2d and surpress_3d)

                delay = kwargs.get("delay", False)
                if not delay or time_index == self.time_num - 1:
                    if duration is None:
                        plt.show()
                    else:
                        plt.show(block=False)
                        plt.pause(int(duration))
                        plt.close()

    def show_plotly(
        self,
        title: str = "",
        duration: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Show routine using plotly built-in methods.

        Args:
            title (str): title in the displayed window.
            duration (int, optional): display duration in seconds.
            **kwargs: additional arguments passed to matplotlib.pyplot.
                threshold (float): threshold for displaying 3d images.
                relative (bool): flag controlling whether the threshold is relative.
                view (str): view type; either "scatter" or "voxel"; only for 3d images.
                    NOTE: "voxel" plots are more time consuming.
                side_view (str): side view type of 3d image; only for 3d images;
                    either "scatter" or "voxel".
                surpress_2d (bool): flag controlling whether 2d images are displayed.
                surpress_3d (bool): flag controlling whether 3d images are displayed.

        """
        for time_index in range(self.time_num):
            if self.series:
                # Make mypy happy
                assert isinstance(self.date, list) and all(
                    [isinstance(d, datetime) for d in self.date]
                )
                assert isinstance(self.time, list)

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

                # Append title with time
                _title = title
                if not _title == "":
                    _title += " - "
                _title += f"{time_index} - {abs_time} -  {rel_time} sec."
            else:
                _title = title

            # Plot the entire 2d image in plain mode
            if self.space_dim == 2:
                # Only works for scalar and optical images.
                assert self.scalar or self.range_num in [1, 3]

                # Fetch data array
                if self.series:
                    if self.scalar:
                        array = self.img[..., time_index]
                    else:
                        array = self.img[..., time_index, :]
                else:
                    array = self.img

                # Fetch x and y labels
                axes = []
                for i in range(2):
                    axis, revert = darsia.interpret_indexing("xy"[i], "ij")
                    relative_axis = np.arange(1, self.num_voxels[axis] + 1)
                    orientation = -1 if revert else 1
                    axes.append(
                        self.origin[i]
                        + orientation * self.voxel_size[axis] * relative_axis
                    )

                # Plot
                fig_2d = px.imshow(
                    skimage.img_as_float(array),
                    title=_title,
                    x=axes[0],
                    y=axes[1],
                    aspect="equal",
                )
                # fig.update_yaxes(autorange="reversed")

                fig_2d.show()

            elif self.space_dim == 3:
                # ! ---- Preliminaries

                # Only works for scalar images.
                assert self.scalar

                # Extract physical coordinates and flatten
                matrix_indices = np.indices(self.img.shape[:3])
                reshaped_matrix_indices = np.transpose(matrix_indices.reshape((3, -1)))
                reshaped_coordinates = self.coordinatesystem.coordinate(
                    reshaped_matrix_indices
                )
                coordinates = np.transpose(reshaped_coordinates).reshape(
                    matrix_indices.shape
                )

                # Extract values
                if self.series:
                    array = self.img[..., time_index]
                    time_slice = self.time_slice(time_index)
                else:
                    array = self.img
                    time_slice = self

                # Restrict to active voxels
                threshold = kwargs.get("threshold", np.min(self.img))
                relative = kwargs.get("relative", False)
                if relative:
                    threshold = np.min(self.img) + threshold * (
                        np.max(self.img) - np.min(self.img)
                    )
                active = array > threshold
                if np.count_nonzero(active) > 1e5:
                    warn(
                        """Too many active voxels. Plotting may take a while """
                        """or even fail."""
                    )

                # ! ---- 3d view

                # Offer two possibilities. Either a scatter plot or a voxel plot.

                surpress_3d = kwargs.get("surpress_3d", False)
                if not surpress_3d:
                    view = kwargs.get("view", "scatter").lower()
                    assert view in ["scatter", "voxel"]
                    if view == "scatter":
                        fig_3d = go.Figure(
                            data=go.Scatter3d(
                                x=coordinates[0, active].flatten(),
                                y=coordinates[1, active].flatten(),
                                z=coordinates[2, active].flatten(),
                                mode="markers",
                                marker=dict(
                                    size=3,
                                    color=array[active].flatten(),
                                    colorscale="Viridis",
                                    opacity=0.5,
                                ),
                                # title=_title + " - 3d view",
                            )
                        )
                    elif view == "voxel":
                        # Convert coordinates into np.indices format.
                        voxel_corners = np.indices(self.img.shape[:3])
                        reshaped_voxel_corners = np.transpose(
                            voxel_corners.reshape((3, -1))
                        )
                        reshaped_voxel_coordinates = self.coordinatesystem.coordinate(
                            reshaped_voxel_corners
                        )
                        voxel_coordinates = np.transpose(
                            reshaped_voxel_coordinates
                        ).reshape(voxel_corners.shape)

                        fig_3d = go.Figure(
                            data=go.Volume(
                                x=voxel_coordinates[0].flatten(),
                                y=voxel_coordinates[1].flatten(),
                                z=voxel_coordinates[2].flatten(),
                                value=array.flatten(),
                                isomin=threshold,
                                isomax=np.max(array),
                                opacity=0.5,
                                surface_count=10,
                                # title=_title + " - 3d view",
                            )
                        )

                    fig_3d.show()

                # ! ---- 2d side views

                # Offer two possibilities. Either a scatter plot or an
                # integrated view. The latter uses integration over the axis "into" the
                # screen.

                surpress_2d = kwargs.get("surpress_2d", False)
                if not surpress_2d:
                    side_view = kwargs.get("side_view", "scatter").lower()
                    assert side_view in ["scatter", "voxel"]

                    fig_2d = make_subplots(
                        rows=1,
                        cols=3,
                        subplot_titles=["x-y plane", "x-z plane", "y-z plane"],
                    )
                    fig_2d.update_layout(title_text="2d side views")

                    # xy-plane
                    if side_view == "scatter":
                        fig_2d.add_trace(
                            go.Scatter(
                                x=coordinates[0, active].flatten(),
                                y=coordinates[1, active].flatten(),
                                mode="markers",
                                marker=dict(
                                    size=3,
                                    color=array[active].flatten(),
                                    colorscale="Viridis",
                                    opacity=0.5,
                                ),
                                # labels={"x": "x", "y": "y"},
                            ),
                            row=1,
                            col=1,
                        )

                        fig_2d.add_trace(
                            go.Scatter(
                                x=coordinates[0, active].flatten(),
                                y=coordinates[2, active].flatten(),
                                mode="markers",
                                marker=dict(
                                    size=3,
                                    color=array[active].flatten(),
                                    colorscale="Viridis",
                                    opacity=0.5,
                                ),
                                # labels={"x": "x", "y": "z"},
                            ),
                            row=1,
                            col=2,
                        )

                        fig_2d.add_trace(
                            go.Scatter(
                                x=coordinates[1, active].flatten(),
                                y=coordinates[2, active].flatten(),
                                mode="markers",
                                marker=dict(
                                    size=3,
                                    color=array[active].flatten(),
                                    colorscale="Viridis",
                                    opacity=0.5,
                                ),
                                # labels={"x": "y", "y": "z"},
                            ),
                            row=1,
                            col=3,
                        )

                    elif side_view == "voxel":
                        # Fetch x, y, z labels
                        axes = []
                        for i in range(3):
                            axis, revert = darsia.interpret_indexing("xyz"[i], "ijk")
                            relative_axis = np.arange(1, self.num_voxels[axis] + 1)
                            orientation = -1 if revert else 1
                            axes.append(
                                self.origin[i]
                                + orientation * self.voxel_size[axis] * relative_axis
                            )

                        # xy-plane
                        reduction = darsia.AxisAveraging(axis="z", dim=3)
                        reduced_image = reduction(time_slice)
                        fig_2d.add_trace(
                            go.Heatmap(
                                z=skimage.img_as_float(reduced_image.img),
                                x=axes[0],
                                y=axes[1],
                            ),
                            row=1,
                            col=1,
                        )

                        # xz-plane
                        reduction = darsia.AxisAveraging(axis="y", dim=3)
                        reduced_image = reduction(time_slice)
                        fig_2d.add_trace(
                            go.Heatmap(
                                z=skimage.img_as_float(reduced_image.img),
                                x=axes[0],
                                y=axes[2],
                            ),
                            row=1,
                            col=2,
                        )

                        # yz-plane
                        reduction = darsia.AxisAveraging(axis="x", dim=3)
                        reduced_image = reduction(time_slice)
                        fig_2d.add_trace(
                            go.Heatmap(
                                z=skimage.img_as_float(reduced_image.img),
                                x=axes[1],
                                y=axes[2],
                            ),
                            row=1,
                            col=3,
                        )

                    fig_2d.show()

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
            y = origin[1] - i * dy

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
