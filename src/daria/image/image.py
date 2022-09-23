"""Image class.

Images contain the image array, and in addition metadata about origo and dimensions.
"""

from __future__ import annotations

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
        imgpath (str): path to image, also used to define source for metadata
        width (float): physical width of the image
        height (float): physical height of the image
        depth (float): physical depth of the image (only relevant in 3d)
        origo (np.ndarray): physical coordinates of the lower left corner, i.e.,
            of the (img.shape[0],0) pixel
        dim (int): dimension of image, could be 2, 3, 4 (incl. the possibility for time)
        shape (np.ndarray): num_pixels, as well number of color channels (typically 3 for RGB)
        dx (float): pixel size in x-direction
        dy (float): pixel size in y-direction
        dz (float): pixel size in z-direction
    """

    def __init__(
        self,
        img: Union[np.ndarray, str],
        metadata: Optional[dict] = None,
        curvature_correction: Optional[da.CurvatureCorrection] = None,
        color_correction: Optional[da.ColorCorrection] = None,
        **kwargs,
    ) -> None:
        """Constructor of Image object.

        The input image can either be a path to the image file or a numpy array with
        conventional matrix indexing.

        Arguments:
            img (Union[np.ndarray, str]): image array with matrix indexing
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
                    as well as colorspace
                origo (np.ndarray): physical coordinates of the lower left corner
                width (float): physical width of the image
                height (float): physical height of the image
                color_space (str): the colorspace of the image. So far only BGR
                    and RGB are "valid", but more should be added at a later time.
        """

        # Read metadata
        if metadata is not None:
            self.metadata = metadata
            self.width: float = self.metadata["width"]
            self.height: float = self.metadata["height"]
            self.origo: list[float] = self.metadata["origo"]
            self.colorspace: str = self.metadata["color_space"]

        elif "metadata_source" in kwargs:
            metadata_source = kwargs.pop("metadata_source")
            with open(str(Path(metadata_source)), "r") as openfile:
                self.metadata = json.load(openfile)

            self.width: float = self.metadata["width"]
            self.height: float = self.metadata["height"]
            self.origo = self.metadata["origo"]
            self.colorspace: str = self.metadata["color_space"]

        else:
            self.width: float = kwargs.pop("width", 1)
            self.height: float = kwargs.pop("height", 1)
            self.origo: list[float] = kwargs.pop("origo", [0, 0])
            self.colorspace: str = kwargs.pop("color_space", "BGR")
            self.update_metadata()

        # Fetch image
        if isinstance(img, np.ndarray):
            self.img = img

            # Come up with default metadata
            self.name = "Unnamed image"
            self.timestamp = None

            if (
                (metadata is None)
                and ("color_space" not in kwargs)
                and ("metadata_source" not in kwargs)
            ):
                warn("Please provide a colorspace. Now it is assumed to be BGR.")

        elif isinstance(img, str):
            pil_img = PIL_Image.open(Path(img))
            self.img = np.array(pil_img)

            # PIL reads in RGB format
            self.colorspace: str = "RGB"
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

        if curvature_correction is not None:
            self.img = curvature_correction(self.img)
            self.width = curvature_correction.config["crop"]["width"]
            self.height = curvature_correction.config["crop"]["height"]
            self.origo = [0, 0]
            self.update_metadata()

        if color_correction is not None:
            self.toRGB()
            self.img = color_correction(self.img)

        # Determine numbers of cells in each dimension and cell size
        self.num_pixels_height, self.num_pixels_width = self.img.shape[:2]
        self.dx = self.width / self.num_pixels_width
        self.dy = self.height / self.num_pixels_height

        # Define the pixels in the corners of the image
        self.corners = {
            "upperleft": np.array([0, 0]),
            "lowerleft": np.array([self.num_pixels_height, 0]),
            "lowerright": np.array([self.num_pixels_height, self.num_pixels_width]),
            "upperright": np.array([0, self.num_pixels_width]),
        }

        # Establish a coordinate system based on the metadata
        self.coordinatesystem: da.CoordinateSystem = da.CoordinateSystem(self)

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
        assert self.colorspace == "BGR"

        # Write image, using the conventional matrix indexing
        cv2.imwrite(str(Path(path + name + file_format)), self.img)
        print("Image saved as: " + str(Path(path + name + file_format)))

    def update_metadata(self) -> None:
        self.metadata: dict = {
            "width": self.width,
            "height": self.height,
            "origo": self.origo,
            "color_space": self.colorspace,
        }

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

        if self.colorspace == "RGB":
            bgrim = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        else:
            bgrim = self.img

        # Display image
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, bgrim)
        cv2.waitKey(wait)
        cv2.destroyAllWindows()

    def plt_show(self, time: Optional(int) = None) -> None:
        """Show image using matplotlib.pyplots built-in imshow"""

        if self.colorspace == "BGR":
            rgbim = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        else:
            rgbim = self.img
        if time is not None:
            plt.imshow(rgbim, block = False)
            plt.pause(time)
            plt.close
        else:
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
            origo = self.origo
        elif isinstance(origo, list):
            origo = np.array(origo)

        # Determine the number of grid lines required
        num_horizontal_lines: int = math.ceil(self.height / dy) + 1
        num_vertical_lines: int = math.ceil(self.width / dx) + 1

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
        return Image(
            img=gridimg, origo=self.origo, width=self.width, height=self.height
        )

    # resize image by using cv2's resize command
    def resize(self, cx: float, cy: float) -> None:
        """ "
        Coarsen the image object

        Arguments:
            cx: the amount of which to scale in x direction
            cy: the amount of which to scale in y direction
        """

        # Coarsen image
        self.img = cv2.resize(self.img, None, fx=cx, fy=cy)

        # Update parameters and coordinate system
        self.dx *= 1 / cx
        self.dy *= 1 / cy
        self.coordinatesystem: da.CoordinateSystem = da.CoordinateSystem(self)

    def copy(self) -> "Image":
        """
        Returns a copy of the image object.
        """
        return Image(self.img, self.metadata, color_space=self.colorspace)

    def toBGR(self) -> None:
        """
        Transforms image to BGR if it is in RGB
        """
        if self.colorspace == "RGB":
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.colorspace = "BGR"
            self.metadata["color_space"] = "BGR"

    def toRGB(self) -> None:
        """
        Transforms image to RGB if it is in BGR
        """
        if self.colorspace == "BGR":
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.colorspace = "RGB"
            self.metadata["color_space"] = "RGB"

    def toGray(self) -> da.Image:
        """
        Returns a greyscale version of the daria image
        """
        gray_img = self.copy()
        if self.colorspace == "BGR":
            gray_img.img = cv2.cvtColor(gray_img.img, cv2.COLOR_BGR2GRAY)
            gray_img.colorspace = "GRAY"
            gray_img.update_metadata()
        return gray_img

