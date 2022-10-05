"""Image class.

Images contain the image array, and in addition metadata about origo and dimensions.
"""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from warnings import warn

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PIL_Image

import daria as da


class Image:
    """Base image class.

    Contains the actual image, as well as meta data, i.e., base properties such as
    position in global coordinates, width and height. One can either pass in metadata
    (origo, width and height, amongst other entities) by passing them directly to
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
            origo (np.ndarray): physical coordinates of the lower left corner, i.e.,
                of the (img.shape[0],0) pixel
            color_space (str): Color space (RGB/BGR/RED/GREEN/BLUE/GRAY)
        shape (np.ndarray): num_pixels, as well number of color channels (typically 3 for RGB)
        dx (float): pixel size in x-direction
        dy (float): pixel size in y-direction
    """

    def __init__(
        self,
        img: Union[np.ndarray, str, Path],
        metadata: Optional[dict] = None,
        curvature_correction: Optional[da.CurvatureCorrection] = None,
        color_correction: Optional[da.ColorCorrection] = None,
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
            curvature_correction (daria.CurvatureCorrection, Optional):
                Curvature correction object. Default is none, but should be included
                if the image is to be curvature corrected at initialization
            color_correction (daria.ColorCorrection, Optional): Color correction object.
                Default is none, but should be included if the image is to be color
                corrected at initialization.
            kwargs (Optional arguments)
                metadata_source (str): Path to a metadata json-file that provides
                    metadata such as physical width, height and origo of image
                    as well as  color space
                origo (np.ndarray): physical coordinates of the lower left corner
                width (float): physical width of the image
                height (float): physical height of the image
                color_space (str): the color space of the image. So far only BGR
                    and RGB are "valid", but more should be added at a later time.
        """

        # Read metadata.
        no_colorspace_given = False
        if metadata is not None:
            self.metadata = metadata

        elif "metadata_source" in kwargs:
            metadata_source = kwargs.pop("metadata_source")
            with open(str(Path(metadata_source)), "r") as openfile:
                self.metadata = json.load(openfile)

        else:
            self.metadata: dict = {}
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

            self.metadata["origo"]: np.ndarray[float] = kwargs.pop(
                "origo", np.array([0, 0])
            )

            no_colorspace_given = "color_space" not in kwargs
            self.metadata["color_space"] = kwargs.pop("color_space", "BGR")

        # Fetch image
        if isinstance(img, np.ndarray):
            self.img = img

            # Come up with default metadata
            self.name = "Unnamed image"
            self.timestamp = None

            if no_colorspace_given:
                warn("Please provide a color space. Now it is assumed to be BGR.")

        elif isinstance(img, str) or isinstance(img, Path):
            pil_img = PIL_Image.open(Path(img))
            self.img = np.array(pil_img)

            # PIL reads in RGB format
            self.metadata["color_space"] = "RGB"

            # Read exif metadata
            self.exif = pil_img.getexif()
            if self.exif.get(306) is not None:
                self.timestamp: datetime = datetime.strptime(
                    self.exif.get(306), "%Y:%m:%d %H:%M:%S"
                )
            else:
                self.timestamp = None

            self.imgpath = img
            self.name = img
        else:
            raise Exception(
                "Invalid image data. Provide either a path to an image or an image array."
            )

        # Apply corrections
        if color_correction is not None:
            self.toRGB()
            self.img = color_correction(self.img)

        if curvature_correction is not None:
            self.img = curvature_correction(self.img)
            assert (
                self.metadata["width"] == curvature_correction.config["crop"]["width"]
            )
            assert (
                self.metadata["height"] == curvature_correction.config["crop"]["height"]
            )

        # Determine numbers of cells in each dimension and cell size
        self.num_pixels_height, self.num_pixels_width = self.img.shape[:2]
        self.dx = self.metadata["width"] / self.num_pixels_width
        self.dy = self.metadata["height"] / self.num_pixels_height

        # Define the pixels in the corners of the image
        self.corners = {
            "upperleft": np.array([0, 0]),
            "lowerleft": np.array([self.num_pixels_height, 0]),
            "lowerright": np.array([self.num_pixels_height, self.num_pixels_width]),
            "upperright": np.array([0, self.num_pixels_width]),
        }

        # Establish a coordinate system based on the metadata
        self.coordinatesystem: da.CoordinateSystem = da.CoordinateSystem(self)

    @property
    def origo(self) -> list:
        return self.metadata["origo"]

    @property
    def width(self) -> float:
        return self.metadata["width"]

    @property
    def height(self) -> float:
        return self.metadata["height"]

    @property
    def colorspace(self) -> str:
        return self.metadata["color_space"]

    def write(
        self,
        name: str = "img",
        path: str = str(Path("images/modified/")),
        file_format: str = ".jpg",
    ) -> None:
        """Write image to file.

        Here, the BGR-format is used. Image path, name and format
        can be changed by passing them as strings to the method.

        Arguments:
            name (str): name of image
            path (str): tpath to image (only folders)
            file_format (str): file ending, deifning the image format
        """
        # cv2 requires BGR format
        self.toBGR()
        assert self.metadata["color_space"] == "BGR"

        # Write image, using the conventional matrix indexing
        cv2.imwrite(str(Path(path + name + file_format)), self.img)
        print("Image saved as: " + str(Path(path + name + file_format)))

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
            rgbim = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        else:
            rgbim = self.img
        if time is not None:
            plt.imshow(rgbim)
            plt.pause(time)
            plt.close()
        else:
            plt.imshow(rgbim)
            plt.show()

    # Seems like something that should read an image and return a new one with grid.
    def add_grid(
        self,
        origo: Optional[Union[np.ndarray, list[float]]] = None,
        dx: float = 1,
        dy: float = 1,
        color: tuple = (0, 0, 125),
        thickness: int = 9,
    ) -> "Image":
        """
        Adds a grid on the image and returns new image.

        Arguments:
            origo (np.ndarray): origo of the grid, in physical units - the reference
                coordinate system is provided by the corresponding attribute coordinatesystem
            dx (float): grid size in x-direction, in physical units
            dy (float): grid size in y-direction, in physical units
            color (tuple of int): BGR color of the grid
            thickness (int): thickness of the grid lines

        Returns:
            Image: original image with grid on top
        """
        # Set origo if it was not provided
        if origo is None:
            origo = self.metadata["origo"]
        elif isinstance(origo, list):
            origo = np.array(origo)

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
            y = origo[1] + i * dy

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
            x = origo[0] + j * dx

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

    # resize image by using cv2's resize command
    def resize(self, cx: float, cy: Optional[float] = None) -> None:
        """ "
        Coarsen the image object

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
        self.coordinatesystem: da.CoordinateSystem = da.CoordinateSystem(self)

    def copy(self) -> "Image":
        """
        Returns a copy of the image object.
        """
        return Image(np.copy(self.img), copy.copy(self.metadata))

    def toBGR(self, return_image: bool = False) -> Optional[da.Image]:
        """
        Transforms image to BGR if it is in RGB
        """
        if self.metadata["color_space"] == "RGB":
            if not return_image:
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                self.metadata["color_space"] = "BGR"
            else:
                return_image = self.copy()
                return_image.img = cv2.cvtColor(return_image.img, cv2.COLOR_BGR2RGB)
                return_image.metadata["color_space"] = "BGR"
                return return_image

    def toRGB(self, return_image: bool = False) -> Optional[da.Image]:
        """
        Transforms image to RGB if it is in BGR
        """
        if self.metadata["color_space"] == "BGR":
            if not return_image:
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                self.metadata["color_space"] = "RGB"
                self.metadata["color_space"] = "RGB"
            else:
                return_image = self.copy()
                return_image.img = cv2.cvtColor(return_image.img, cv2.COLOR_BGR2RGB)
                return_image.metadata["color_space"] = "RGB"
                return return_image

    def toGray(self, return_image: bool = False) -> Optional[da.Image]:
        """
        Returns a greyscale version of the daria image
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

    def toRed(self, return_image: bool = False) -> Optional[da.Image]:
        """
        Returns a red channel version of the daria image
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

    def toBlue(self, return_image: bool = False) -> Optional[da.Image]:
        """
        Returns a blue channel version of the daria image
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

    def toGreen(self, return_image: bool = False) -> Optional[da.Image]:
        """
        Returns a green channel version of the daria image
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
