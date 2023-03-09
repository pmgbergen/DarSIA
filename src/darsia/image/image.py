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
import skimage
from PIL import Image as PIL_Image

import darsia


class Image:
    """Base image class.

    Contains the actual image, as well as meta data, i.e., base properties such as
    position in global coordinates, width and height. One can either pass in metadata
    (origin, width and height, amongst other entities) by passing them directly to
    the constructor or through a metadata-file (default is to pass metadata diretly
    to the constructor. If a metadatafile is required, the "read_metadata_from_file"
    variable has to be passed as True). Image can either be passed in as numpy array
    or a path to a file (this is automatically detected through the input). Base
    functionality such as saving to image-file, and drawing a grid is provided in the class.

    Attributes:
        img (np.ndarray): image array
        img (str): path to image, alternative way to feed the actual image.
        metadata (dict):
            width (float): physical width of the image
            height (float): physical height of the image
            origin (np.ndarray): physical coordinates of the lower left corner, i.e.,
                of the (img.shape[0],0) pixel
            color_space (str): Color space (RGB/BGR/RED/GREEN/BLUE/GRAY)
            timestamp (datetime.datetime): timestamp of the image in (default) datetime format.
        shape (np.ndarray): num_pixels, as well number of color channels (typically 3 for RGB)
        dx (float): pixel size in x-direction
        dy (float): pixel size in y-direction
    """

    def __init__(
        self,
        img: Union[np.ndarray, str, Path],
        metadata: Optional[dict] = None,
        drift_correction: Optional[darsia.DriftCorrection] = None,
        translation_correction: Optional[darsia.TranslationCorrection] = None,
        deformation_correction: Optional[darsia.DeformationCorrection] = None,
        color_correction: Optional[darsia.ColorCorrection] = None,
        curvature_correction: Optional[darsia.CurvatureCorrection] = None,
        **kwargs,
    ) -> None:
        """Constructor of Image object.

        The input image can either be a path to the image file or a numpy array with
        conventional matrix indexing. Some metadata must be provided. If several
        versions are provided a metadata dictionary will used over a path to a
        metadata json-file over keyword arguments.

        Arguments:
            img (Union[np.ndarray, str, Path]): image array with matrix indexing
            metadata (dict, Optional): metadata dictionary, default is None.
            drift_correction (darsia.DriftCorrection, Optional): Drift correction object.
            translation_correction (darsia.TranslationCorrection, Optional): Translation
                correction object.
            deformation_correction (darsia.DeformationCorrection, Optional): Deformation
                correction object.
            color_correction (darsia.ColorCorrection, Optional): Color correction object.
                Default is none, but should be included if the image is to be color
                corrected at initialization.
            curvature_correction (darsia.CurvatureCorrection, Optional):
                Curvature correction object. Default is none, but should be included
                if the image is to be curvature corrected at initialization
            kwargs (Optional arguments)
                metadata_source (str): Path to a metadata json-file that provides
                    metadata such as physical width, height and origin of image
                    as well as  color space
                origin (np.ndarray): physical coordinates of the lower left corner
                width (float): physical width of the image
                height (float): physical height of the image
                color_space (str): the color space of the image. So far only BGR
                    and RGB are "valid", but more should be added at a later time.
                timestamp (datetime.datetime): timestamp of the image. If it is not
                    provided, and available in the image file it will be read by pillow.
        """
        self.space_dim = 2

        # Read metadata.
        no_colorspace_given = False
        if metadata is not None:
            self.metadata = metadata

        elif "metadata_source" in kwargs:
            metadata_source = kwargs.get("metadata_source")
            with open(str(Path(metadata_source)), "r") as openfile:
                self.metadata = json.load(openfile)

        else:
            self.metadata = {}
            if "width" in kwargs:
                self.metadata["width"] = kwargs["width"]
            elif curvature_correction is not None:
                self.metadata["width"] = curvature_correction.config["crop"]["width"]
            else:
                raise Exception("image width not specified")

            if "height" in kwargs:
                self.metadata["height"] = kwargs["height"]
            elif curvature_correction is not None:
                self.metadata["height"] = curvature_correction.config["crop"]["height"]
            else:
                raise Exception("image height not specified")

            self.metadata["origin"] = np.array(kwargs.pop("origin", np.array([0, 0])))

            no_colorspace_given = "color_space" not in kwargs
            self.metadata["color_space"] = kwargs.get("color_space", "BGR")
            self.metadata["timestamp"] = kwargs.get("timestamp", None)

        # Fetch image
        if isinstance(img, np.ndarray):
            self.img: np.ndarray = img

            # Come up with default metadata
            self.name = "Unnamed image"

            if no_colorspace_given:
                warn("Please provide a color space. Now it is assumed to be BGR.")

            self.metadata["original_dtype"] = self.img.dtype

        elif isinstance(img, str) or isinstance(img, Path):

            self.img = cv2.imread(str(Path(img)), cv2.IMREAD_UNCHANGED)
            self.metadata["color_space"] = "BGR"
            self.metadata["original_dtype"] = self.img.dtype

            # Convert to RGB
            self.toRGB()

            if self.metadata["timestamp"] is None:
                pil_img = PIL_Image.open(Path(img))

                # Read timestamp from exif metadata if existent
                self.exif = pil_img.getexif()
                if self.exif.get(306) is not None:
                    self.metadata["timestamp"] = datetime.strptime(
                        self.exif.get(306), "%Y:%m:%d %H:%M:%S"
                    )

            self.imgpath = img
            self.name = cast(str, img)

        else:
            raise Exception(
                "Invalid image data. Provide either a path to an image or an image array."
            )

        # Apply corrections
        if drift_correction is not None:
            self.img = drift_correction(self.img)

        # Move after shape correction objects.
        if color_correction is not None:
            self.img = color_correction(self.img)

        if translation_correction is not None:
            self.img = translation_correction(self.img)

        if curvature_correction is not None:
            self.img = curvature_correction(self.img)
            assert (
                self.metadata["width"] == curvature_correction.config["crop"]["width"]
            )
            assert (
                self.metadata["height"] == curvature_correction.config["crop"]["height"]
            )

        # FIXME: Deformation correction is defined on the corrected baseline...
        # Would make more sense to move this up.
        if deformation_correction is not None:
            self.img = deformation_correction(self.img)

        # Determine numbers of cells in each dimension and cell size
        self.num_pixels_height: int = self.img.shape[:2][0]
        self.num_pixels_width: int = self.img.shape[:2][1]
        self.dx: float = self.metadata["width"] / self.num_pixels_width
        self.dy: float = self.metadata["height"] / self.num_pixels_height

        # Define the pixels in the corners of the image
        self.corners = {
            "upperleft": np.array([0, 0]),
            "lowerleft": np.array([self.num_pixels_height, 0]),
            "lowerright": np.array([self.num_pixels_height, self.num_pixels_width]),
            "upperright": np.array([0, self.num_pixels_width]),
        }

        # Establish a coordinate system based on the metadata
        self.coordinatesystem: darsia.CoordinateSystem = darsia.CoordinateSystem(self)

    def copy(self) -> darsia.Image:
        """
        Copy constructor.

        Returns:
            darsia.Image: Copy of the image object.

        """
        return Image(np.copy(self.img), copy.copy(self.metadata))

    # ! ---- Fast-access getter functions for metadata

    @property
    def origin(self) -> np.ndarray:
        return self.metadata["origin"]

    @property
    def width(self) -> float:
        return self.metadata["width"]

    @property
    def height(self) -> float:
        return self.metadata["height"]

    @property
    def colorspace(self) -> str:
        return self.metadata["color_space"]

    @property
    def timestamp(self) -> datetime:
        return self.metadata["timestamp"]

    @timestamp.setter
    def timestamp(self, time: datetime) -> None:
        self.metadata["timestamp"] = time

    @property
    def original_dtype(self) -> np.dtype:
        return self.metadata["original_dtype"]

    # ! ---- Operation overloaders
    def __sub__(self, other: darsia.Image):
        """Subtract two images.

        Arguments:
            other (Image): image to subtract from self

        Returns:
            Image: difference image
        """
        if self.img.shape != other.img.shape:
            warn("Images have different shapes. Resizing second argument to match.")
            return darsia.Image(
                self.img - cv2.resize(other.img, tuple(reversed(self.img.shape[:2]))),
                copy.copy(self.metadata),
            )
        else:
            return darsia.Image(self.img - other.img, copy.copy(self.metadata))

    # ! ---- I/O

    def write(
        self,
        path,
    ) -> None:
        """Write image to file.

        Here, the BGR-format is used. Image path, name and format
        can be changed by passing them as strings to the method.

        Arguments:
            path (str): path to image, including image name and file format
        """
        # cv2 requires BGR format
        write_image = cast(darsia.Image, self.toBGR(return_image=True)).img

        # Write image, using the conventional matrix indexing
        if self.original_dtype == np.uint8:
            write_image = skimage.img_as_ubyte(write_image)
            cv2.imwrite(str(Path(path)), write_image)
        elif self.original_dtype == np.uint16:
            write_image = skimage.img_as_uint(write_image)
            cv2.imwrite(
                str(Path(path)),
                write_image,
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
            raise Exception(f"Cannot write the datatype {self.original_dtype}")

        print("Image saved as: " + str(Path(path)))

    def write_metadata_to_file(self, path: str) -> None:
        """
        Writes the metadata dictionary to a json-file.

        Arguments:
            path (str): path to the json file
        """

        with open(str(Path(path)), "w") as outfile:
            json.dump(self.metadata, outfile, indent=4)

    def show(self, name: Optional[str] = None, wait: Optional[int] = 0) -> None:
        """Show image.

        Arguments:
            name (str, optional): name addressing the window for visualization
            wait (int, optional): waiting time in milliseconds, if not set
                the window is open until any key is pressed.
        """
        if name is None:
            name = self.name

        if self.metadata["color_space"] == "RGB":
            bgrim = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        else:
            bgrim = self.img

        # Display image
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, bgrim)
        cv2.waitKey(wait)
        cv2.destroyAllWindows()

    # Not completely satisfied with this solution of timing
    def plt_show(self, time: Optional[int] = None) -> None:
        """Show image using matplotlib.pyplots built-in imshow"""

        if self.metadata["color_space"] == "BGR":
            rgbim = cv2.cvtColor(skimage.img_as_ubyte(self.img), cv2.COLOR_BGR2RGB)
        else:
            rgbim = skimage.img_as_ubyte(self.img)
        if time is not None:
            plt.imshow(rgbim)
            plt.pause(time)
            plt.close()
        else:
            plt.imshow(rgbim)
            plt.show()

    # ! ---- Utilities

    # Seems like something that should read an image and return a new one with grid.
    def add_grid(
        self,
        origin: Optional[Union[np.ndarray, list[float]]] = None,
        dx: float = 1,
        dy: float = 1,
        color: tuple = (0, 0, 125),
        thickness: int = 9,
    ) -> "Image":
        """
        Adds a grid on the image and returns new image.

        Arguments:
            origin (np.ndarray): origin of the grid, in physical units - the reference
                coordinate system is provided by the corresponding attribute coordinatesystem
            dx (float): grid size in x-direction, in physical units
            dy (float): grid size in y-direction, in physical units
            color (tuple of int): BGR color of the grid
            thickness (int): thickness of the grid lines

        Returns:
            Image: original image with grid on top
        """
        # Set origin if it was not provided
        if origin is None:
            origin = self.metadata["origin"]
        elif isinstance(origin, list):
            origin = np.array(origin)

        # Determine the number of grid lines required
        num_horizontal_lines: int = math.ceil(self.metadata["height"] / dy) + 1
        num_vertical_lines: int = math.ceil(self.metadata["width"] / dx) + 1

        # Start from original image
        gridimg = self.img.copy()

        # Add horizontal grid lines (line by line)
        for i in range(num_horizontal_lines):

            # Determine the outer boundaries in x directions
            xmin = self.coordinatesystem.domain["xmin"]
            xmax = self.coordinatesystem.domain["xmax"]

            # Determine the y coordinate of the line
            y = origin[1] + i * dy

            # Determine the pixels corresponding to the end points of the horizontal line
            # (xmin,y) - (xmax,y), in (row,col) format.
            start = self.coordinatesystem.coordinateToPixel(np.array([xmin, y]))
            end = self.coordinatesystem.coordinateToPixel(np.array([xmax, y]))

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
            start = self.coordinatesystem.coordinateToPixel(np.array([x, ymin]))
            end = self.coordinatesystem.coordinateToPixel(np.array([x, ymax]))

            # Add single line. NOTE: cv2.line takes pixels as inputs with the reversed
            # matrix indexing, i.e., (col,row) instead of (row,col). Furthermore,
            # it requires tuples.
            gridimg = cv2.line(
                gridimg, tuple(reversed(start)), tuple(reversed(end)), color, thickness
            )

        # Return image with grid as Image object
        return Image(img=gridimg, metadata=self.metadata)

    def resize(self, cx: float, cy: Optional[float] = None) -> None:
        """ "
        Resize routine.

        Arguments:
            cx (float): the amount of which to scale in x direction
            cy (float, optional): the amount of which to scale in y direction;
                default value is cx
        """
        if cy is None:
            cy = cx

        # Coarsen image
        self.img = cv2.resize(
            self.img,
            None,
            fx=cx,
            fy=cy,
            interpolation=cv2.INTER_AREA if min(cx, cy) < 1 else cv2.INTER_CUBIC,
        )

        # Update parameters and coordinate system
        self.dx *= 1 / cx
        self.dy *= 1 / cy
        self.coordinatesystem = darsia.CoordinateSystem(self)

    # ! ---- Color transformations

    def toBGR(self, return_image: bool = False) -> Optional[darsia.Image]:
        """
        Transforms image to BGR if it is in RGB

        Args:
            return_image (bool): flag controlling whether the converted image
                is returned, or converted internally.

        Returns:
            darsia.Image (optional): converted image, if requested
        """
        if self.metadata["color_space"] == "RGB":
            if not return_image:
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                self.metadata["color_space"] = "BGR"
                return None
            else:
                image = self.copy()
                image.img = cv2.cvtColor(image.img, cv2.COLOR_BGR2RGB)
                image.metadata["color_space"] = "BGR"
                return image
        else:
            return None

    def toRGB(self, return_image: bool = False) -> Optional[darsia.Image]:
        """
        Transforms image to RGB if it is in BGR.

        Args:
            return_image (bool): flag controlling whether the converted image
                is returned, or converted internally.

        Returns:
            darsia.Image (optional): converted image, if requested
        """
        if self.metadata["color_space"] == "BGR":
            if not return_image:
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                self.metadata["color_space"] = "RGB"
                self.metadata["color_space"] = "RGB"
                return None
            else:
                image = self.copy()
                image.img = cv2.cvtColor(image.img, cv2.COLOR_BGR2RGB)
                image.metadata["color_space"] = "RGB"
                return image
        else:
            return None

    def toGray(self, return_image: bool = False) -> Optional[darsia.Image]:
        """
        Returns a greyscale version of the darsia image.

        Args:
            return_image (bool): flag controlling whether the converted image
                is returned, or converted internally.

        Returns:
            darsia.Image (optional): converted image, if requested
        """

        if return_image:
            gray_img = self.copy()
            if self.metadata["color_space"] == "RGB":
                gray_img.img = cv2.cvtColor(gray_img.img, cv2.COLOR_RGB2GRAY)
            elif self.metadata["color_space"] == "BGR":
                gray_img.img = cv2.cvtColor(gray_img.img, cv2.COLOR_BGR2GRAY)
            elif self.metadata["color_space"] == "GRAY":
                pass
            else:
                raise Exception(
                    "Only RGB or BGR images can be converted to Gray at the moment"
                )
            gray_img.metadata["color_space"] = "GRAY"
            return gray_img
        else:
            if self.metadata["color_space"] == "RGB":
                self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
            elif self.metadata["color_space"] == "BGR":
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            elif self.metadata["color_space"] == "GRAY":
                pass
            else:
                raise Exception(
                    "Only RGB or BGR images can be converted to Gray at the moment"
                )
            self.metadata["color_space"] = "GRAY"
            return None

    def toRed(self, return_image: bool = False) -> Optional[darsia.Image]:
        """
        Returns a red channel version of the darsia image.

        Args:
            return_image (bool): flag controlling whether the converted image
                is returned, or converted internally.

        Returns:
            darsia.Image (optional): converted image, if requested
        """
        if return_image:
            red_img = self.copy()
            if self.metadata["color_space"] == "RGB":
                red_img.img = red_img.img[:, :, 0]
            elif self.metadata["color_space"] == "BGR":
                red_img.img = red_img.img[:, :, 2]
            elif self.metadata["color_space"] == "RED":
                pass
            else:
                raise Exception(
                    "Only RGB or BGR images can be converted to Red at the moment"
                )
            red_img.metadata["color_space"] = "RED"
            return red_img
        else:
            if self.metadata["color_space"] == "RGB":
                self.img = self.img[:, :, 0]
            elif self.metadata["color_space"] == "BGR":
                self.img = self.img[:, :, 2]
            elif self.metadata["color_space"] == "RED":
                pass
            else:
                raise Exception(
                    "Only RGB or BGR images can be converted to Red at the moment"
                )
            self.metadata["color_space"] = "RED"
            return None

    def toBlue(self, return_image: bool = False) -> Optional[darsia.Image]:
        """
        Returns a blue channel version of the darsia image.

        Args:
            return_image (bool): flag controlling whether the converted image
                is returned, or converted internally.

        Returns:
            darsia.Image (optional): converted image, if requested
        """
        if return_image:
            blue_img = self.copy()
            if self.metadata["color_space"] == "RGB":
                blue_img.img = blue_img.img[:, :, 2]
            elif self.metadata["color_space"] == "BGR":
                blue_img.img = blue_img.img[:, :, 0]
            elif self.metadata["color_space"] == "BLUE":
                pass
            else:
                raise Exception(
                    "Only RGB or BGR images can be converted to blue at the moment"
                )
            blue_img.metadata["color_space"] = "BLUE"
            return blue_img
        else:
            if self.metadata["color_space"] == "RGB":
                self.img = self.img[:, :, 2]
            elif self.metadata["color_space"] == "BGR":
                self.img = self.img[:, :, 0]
            elif self.metadata["color_space"] == "BLUE":
                pass
            else:
                raise Exception(
                    "Only RGB or BGR images can be converted to blue at the moment"
                )
            self.metadata["color_space"] = "BLUE"
            return None

    def toGreen(self, return_image: bool = False) -> Optional[darsia.Image]:
        """
        Returns a green channel version of the darsia image.

        Args:
            return_image (bool): flag controlling whether the converted image
                is returned, or converted internally.

        Returns:
            darsia.Image (optional): converted image, if requested
        """
        if return_image:
            green_img = self.copy()
            if self.metadata["color_space"] == "RGB":
                green_img.img = green_img.img[:, :, 1]
            elif self.metadata["color_space"] == "BGR":
                green_img.img = green_img.img[:, :, 1]
            elif self.metadata["color_space"] == "GREEN":
                pass
            else:
                raise Exception(
                    "Only RGB or BGR images can be converted to green at the moment"
                )
            green_img.metadata["color_space"] = "GREEN"
            return green_img
        else:
            if self.metadata["color_space"] == "RGB":
                self.img = self.img[:, :, 1]
            elif self.metadata["color_space"] == "BGR":
                self.img = self.img[:, :, 1]
            elif self.metadata["color_space"] == "GREEN":
                pass
            else:
                raise Exception(
                    "Only RGB or BGR images can be converted to green at the moment"
                )
            self.metadata["color_space"] = "GREEN"
            return None

    def toHue(self, return_image: bool = False) -> Optional[darsia.Image]:
        """
        Returns a hue channel version of the darsia image.

        Args:
            return_image (bool): flag controlling whether the converted image
                is returned, or converted internally.

        Returns:
            darsia.Image (optional): converted image, if requested
        """
        # Only for RGB images for now.
        assert self.metadata["color_space"] == "RGB"

        if return_image:
            hue_img = self.copy()
            hue_img.img = cv2.cvtColor(hue_img.img, cv2.COLOR_RGB2HSV)[:, :, 0]
            hue_img.metadata["color_space"] = "HUE"
            return hue_img
        else:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)[:, :, 0]
            self.metadata["color_space"] = "HUE"
            return None

    def toSaturation(self, return_image: bool = False) -> Optional[darsia.Image]:
        """
        Returns a saturation channel version of the darsia image.

        Args:
            return_image (bool): flag controlling whether the converted image
                is returned, or converted internally.

        Returns:
            darsia.Image (optional): converted image, if requested
        """
        # Only for RGB images for now.
        assert self.metadata["color_space"] == "RGB"

        if return_image:
            sat_img = self.copy()
            sat_img.img = cv2.cvtColor(sat_img.img, cv2.COLOR_RGB2HSV)[:, :, 1]
            sat_img.metadata["color_space"] = "SATURATION"
            return sat_img
        else:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)[:, :, 1]
            self.metadata["color_space"] = "SATURATION"
            return None

    def toValue(self, return_image: bool = False) -> Optional[darsia.Image]:
        """
        Returns a value channel version of the darsia image

        Args:
            return_image (bool): flag controlling whether the converted image
                is returned, or converted internally.

        Returns:
            darsia.Image (optional): converted image, if requested
        """
        # Only for RGB images for now.
        assert self.metadata["color_space"] == "RGB"

        if return_image:
            val_img = self.copy()
            val_img.img = cv2.cvtColor(val_img.img, cv2.COLOR_RGB2HSV)[:, :, 2]
            val_img.metadata["color_space"] = "VALUE"
            return val_img
        else:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)[:, :, 2]
            self.metadata["color_space"] = "VALUE"
            return None


class GeneralImage:
    """
    Develop version of a general, potential space-time Image class.

    """

    # ! ---- Constructors

    def __init__(
        self,
        img: np.ndarray,
        time: Optional[Union[datetime, float, list[datetime], list[float]]] = None,
        transformations: Optional[list] = None,
        **kwargs,
    ) -> None:
        """
        Initalization of a physical space-time image.

        Allows for scalar and vector-values  2d, 3d, 4d, images, including
        time-slices as well as time-series. The boolean flag 'scalar' stores
        whether the data is stored in an additional dimension of the Image or not
        (note scalar data can be in general also encoded as multichromatic image
        with 1d data). Furthermore, 'series' holds this information, while

        Args:
            img (array): space_dim+time_dim+range_dim space-time data array
            time (float, list or array): times corresponding to image or slices.
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
        self.shape = img.shape
        self.original_dtype = img.dtype

        # ! ---- Spatial meta information
        self.space_dim: int = kwargs.get("dim", 2)
        """Dimension of the spatial domain."""

        self.space_num: int = np.prod(self.shape[: self.space_dim])
        """Spatial resolution, i.e., number of voxels."""

        self.indexing = kwargs.get("indexing", "ij")
        """Indexing of each axis in context of matrix indexing (ijk)
        or Cartesian coordinates (xyz)."""

        # For now, only anticipate matrix indexing.
        assert self.indexing == "ijk"[: self.space_dim]

        self.dimensions: list[float] = kwargs.get("dimensions", self.space_dim * [1])
        """Dimension in the directions corresponding to the indexings."""

        self.num_voxels: int = self.img.shape[: self.space_dim]
        """Number of voxels in each dimension."""

        self.voxel_size: list[float] = [
            self.dimensions[i] / self.num_voxels[i] for i in range(self.space_dim)
        ]
        """Size of each voxel in each direction, ordered as indexing."""

        self.origin = np.array(kwargs.pop("origin", self.space_dim * [0]))
        """Cartesian coordinates associated to the [0,0,0] voxel (after
        applying transformations)."""

        self.coordinatesystem: darsia.GeneralCoordinateSystem = (
            darsia.GeneralCoordinateSystem(self)
        )
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

            if time is None:
                self.time: list = self.time_num * [None]
                """Time data."""
            else:
                self.time = time

        else:
            self.time_dim = 0
            self.time_num = 1
            self.time = time

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

        # TODO rm this list.

        # NOTE: Recommended order of transformations:
        # 1. Reduction from 3d to 2d

        # For 2d images:
        # 1. drift correction to some baseline image
        # 2. color correction
        # 3. translation correction to some baseline image
        # 4. curvature correction calibrated based on a baseline image
        # 5. deformation correction wrt. corrected baseline image

        # NOTE: Require mapping format:
        # darsia.SpaceTimeImage -> darsia.SpaceTimeImage
        if transformations is not None:
            for transformation in transformations:
                transformation(self)

        # ! ---- Update spatial metadata, after shape-altering transformations
        self.space_num: int = np.prod(self.shape[: self.space_dim])
        """Spatial resolution, i.e., number of voxels."""

        self.num_voxels: int = self.img.shape[: self.space_dim]
        """Number of voxels in each dimension."""

        self.voxel_size: list[float] = [
            self.dimensions[i] / self.num_voxels[i] for i in range(self.space_dim)
        ]
        """Size of each voxel in each direction, ordered as indexing."""

        self.coordinatesystem: darsia.GeneralCoordinateSystem = (
            darsia.GeneralCoordinateSystem(self)
        )
        """Physical coordinate system with equipped transformation from voxel to
        Cartesian space."""

        # ! ---- Safety check on dimensionality and resolution of image.
        assert len(self.shape) == self.space_dim + self.time_dim + self.range_dim
        assert np.prod(self.shape) == self.space_num * self.time_num * self.range_num

    def copy(self) -> GeneralImage:
        """
        Copy constructor.

        Returns:
            GeneralImage: Copy of the image object.

        """
        return copy.deepcopy(self)

    def append(self, image: GeneralImage) -> None:
        """Append other image to current image. Makes in particular
        a non-space-time image to a space-time image.

        Args:
            image (GeneralImage): image to be appended.

        """

        # ! ---- Safety checks

        assert self.space_dim == image.space_dim
        assert self.scalar == image.scalar
        assert np.all(np.isclose(np.array(self.num_voxels), np.array(image.num_voxels)))
        assert self.time[-1] < image.time[0]
        assert np.all(np.isclose(np.array(self.dimensions), np.array(image.dimensions)))
        assert np.all(np.isclose(np.array(self.origin), np.array(image.origin)))

        # ! ---- Update data

        # Auxiliary routine for slicing images
        def slice_image(im: GeneralImage) -> list[np.ndarray]:
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
        self.time = self.time + image.time
        self.time
        self.time_dim = 1
        self.time_num = len(self.time)

    # ! ---- Transformations

    def resize(self, cx: float, cy: Optional[float] = None) -> None:
        raise NotImplementedError

    def astype(self, data_type) -> Any:
        """For scalar data types, change the data type of the data array.
        For Image data types, cast the entire image.

        Args:
            data_type: target data type

        Returns:
            GeneralImage: image with transformed data type

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
            dict: metadata with keys equal to all keywords agurments.

        """
        metadata = {
            "dim": self.space_dim,
            "indexing": self.indexing,
            "dimensions": self.dimensions,
            "origin": self.origin,
            "series": self.series,
            "scalar": self.scalar,
            # TODO color_space in optical image
        }
        return copy.copy(metadata)

    def time_slice(self, time_index: int) -> GeneralImage:
        """Extraction of single time slice.

        Args:
            time_index (int): time index in interval [0, ..., time_num-1].

        Returns:
            GeneralImage: single-timed image.

        """
        if not self.series:
            raise ValueError

        # Fetch data and return corresponding datatype
        if self.scalar:
            img = self.img[..., time_index]

        else:
            img = self.img[..., time_index, :]

        # Fetch time
        time = self.time[time_index]

        # Fetch and update metadata
        metadata = self.metadata()
        metadata["series"] = False

        # Create image with same data type but updates image data and metadata
        return type(self)(img=img, time=time, **metadata)

    def subregion(
        self,
        voxels: Optional[tuple[slice]] = None,
        coordinates: Optional[np.ndarray] = None,
    ) -> GeneralImage:
        """Extraction of spatial subregion.

        Args:
            voxels (tuple of slices, optional): voxel intervals in all dimensions.
            coordinates (array, optional): points in space, in Cartesian coordinates,
                uniquely defining a box.

        Returns:
            GeneralImage: image with restricted spatial domain.

        Raises:
            ValueError: if both voxels and coordinates are not None, or if neither voxels
                nor coordinates are provide

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
                    max(0, voxels_coordinates.min(axis=d)),
                    min(voxels_coordinates.max(axis=d), self.num_voxels[d]),
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

        # ! ---- Fetch data and time
        img = self.img[voxels]
        time = copy.copy(self.time)

        # ! ---- Fetch and adapt metadata
        metadata = self.metadata()
        metadata["dimensions"] = dimensions
        metadata["origin"] = origin

        return type(self)(img=img, time=time, **metadata)

    # ! ---- I/O

    def write_to_numpy(
        self,
        path: Union[str, Path],
    ) -> None:
        """Write image to file in numpy format.

        Here, the BGR-format is used. Image path, name and format
        can be changed by passing them as strings to the method.

        Arguments:
            path (str): path to image, including image name and file format

        """
        np.save(path, self.img)

    def write_metadata_to_json(self, path: Union[str, Path]) -> None:
        """
        Writes the metadata dictionary to a json-file.

        Arguments:
            path (str): path to the json file

        """
        metadata = self.extract_metadata()
        with open(str(Path(path)), "w") as outfile:
            json.dump(metadata, outfile, indent=4)

    # ! ---- Arithmetics

    def __add__(self, other: GeneralImage) -> GeneralImage:
        """Add two images of same size.

        Arguments:
            other (GeneralImage): image to subtract from self

        Returns:
            GeneralImage: sum of images

        """
        if self.img.shape != other.img.shape:
            raise ValueError("Images have different shapes.")
        else:
            metadata = self.metadata()
            time = self.time
            return type(self)(self.img + other.img, time=time, **metadata)

    def __sub__(self, other: GeneralImage) -> GeneralImage:
        """Subtract two images of same size.

        Arguments:
            other (GeneralImage): image to subtract from self

        Returns:
            GeneralImage: difference image

        """
        if self.img.shape != other.img.shape:
            raise ValueError("Images have different shapes.")
        else:
            metadata = self.metadata()
            time = self.time
            return type(self)(self.img - other.img, time=time, **metadata)

    def __mul__(self, scalar: Union[float, int]) -> GeneralImage:
        """Scaling of image.

        Arguments:
            scalar (float or int): scaling parameter

        Returns:
            GeneralImage: scaled image

        """
        if not isinstance(scalar, float) or isinstance(scalar, int):
            raise ValueError

        result_image = self.copy()
        result_image.img *= scalar
        return result_image

    __rmul__ = __mul__

    # TODO add more standard routines with NotImplementedError
    # TODO try TVD.

    # ! ---- Display methods

    def show(
        self,
        name: Optional[str] = None,
        duration: Optional[int] = None,
        show: bool = True,
    ) -> None:
        """Show image using matplotlib.pyplots built-in methods.

        Args:
            name (str, optional): name in the displayed window.
            duration (int, optional): display duration in seconds.
            show (bool): flag controlling whether the plot is forced to be
                displayed.

        """
        # Only works for optical and scalar images.
        assert self.scalar or self.range_num in [1, 3]
        assert self.space_dim == 2

        if self.series:

            if name is None:
                name = ""

            for time_index in range(self.time_num):
                time = (
                    ""
                    if self.time[time_index] is None
                    else " - " + str(self.time[time_index])
                )
                plt.figure(name + f" - {time_index}" + time)
                if self.scalar:
                    plt.imshow(self.img[..., time_index])
                else:
                    plt.imshow(self.img[..., time_index, :])
                if duration is None:
                    duration = 3
                plt.show(block=False)
                plt.pause(int(duration))
                plt.close()

        else:
            if name is None:
                name = ""

            plt.figure(name)
            plt.imshow(self.img)
            if show:
                if duration is None:
                    plt.show()
                else:
                    plt.show(block=False)
                    plt.pause(int(duration))
                    plt.close()


class ScalarImage(GeneralImage):
    """Special case of a space-time image, with 1d data, e.g., monochromatic photograps."""

    # ! ---- Constructors.

    def __init__(
        self,
        img: np.ndarray,
        time: Optional[Union[float, list[float], np.ndarray]] = None,
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

        # Construct a general image with the specs of an optical image
        super().__init__(img, time, transformations, **scalar_metadata, **kwargs)

        assert self.range_dim == 0

        self.name = kwargs.get("name", None)
        """Name of image, e.g., used to describe the origin."""

    def copy(self) -> ScalarImage:
        """Copy constructor.

        Returns:
            ScalarImage: Copy of the image object.

        """
        return copy.deepcopy(self)

    # ! ---- Fast access.

    def metadata(self) -> dict:
        """Generator of metadata; can be used to init a new optical image with same specs.

        Returns:
            dict: metadata with keys equal to all keywords agurments.

        """
        # Start with generic metadata.
        metadata = super().metadata()

        # Add specs specific to optical images.
        metadata["name"] = self.name

        return copy.copy(metadata)


class OpticalImage(GeneralImage):
    """
    Special case of 2d trichromatic optical images, typically originating from photographs.

    """

    # ! ---- Constructors.

    def __init__(
        self,
        img: np.ndarray,
        time: Optional[Union[datetime, float, list[float], np.ndarray]] = None,
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
        super().__init__(img, time, transformations, **optical_metadata, **kwargs)

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
        """Generator of metadata; can be used to init a new optical image with same specs.

        Returns:
            dict: metadata with keys equal to all keywords agurments.

        """
        # Start with generic metadata.
        metadata = super().metadata()

        # Add specs specific to optical images.
        metadata["color_space"] = self.color_space

        return copy.copy(metadata)

    # ! ---- I/O

    def write(
        self,
        path,
    ) -> None:
        """Write image to file.

        Arguments:
            path (str): full path to image.

        """
        # To prepare for the use of cv2.imwrite, convert to BGR color space.
        bgr_image = self.to_bgr(return_image=True)
        bgr_array = bgr_image.img

        # Write image, using the conventional matrix indexing
        if self.original_dtype == np.uint8:

            write_image = skimage.img_as_ubyte(bgr_array)
            cv2.imwrite(str(Path(path)), write_image)

        elif self.original_dtype == np.uint16:

            write_image = skimage.img_as_uint(bgr_array)
            cv2.imwrite(
                str(Path(path)),
                write_image,
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
        time = self.time
        metadata = image.metadata()
        del metadata["color_space"]
        metadata["name"] = key
        metadata["scalar"] = True

        # Return scalar image
        return ScalarImage(img, time=time, **metadata)

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
                coordinate system is provided by the corresponding attribute coordinatesystem
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
        time = self.time
        metadata = self.metadata()

        # Add horizontal grid lines (line by line)
        for i in range(num_horizontal_lines):

            # Determine the outer boundaries in x directions
            xmin = self.coordinatesystem.domain["xmin"]
            xmax = self.coordinatesystem.domain["xmax"]

            # Determine the y coordinate of the line
            y = origin[1] + i * dy

            # Determine the pixels corresponding to the end points of the horizontal line
            # (xmin,y) - (xmax,y), in (row,col) format.
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
        return OpticalImage(img=gridimg, time=time, **metadata)
