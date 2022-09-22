"""Image class.

Images contain the image array, and in addition metadata about origo and dimensions.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

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
        img: np.ndarray,
        origo: Union[np.ndarray, list[float]] = np.array([0, 0]),
        width: float = 1,
        height: float = 1,
        depth: float = 0,
        dim: int = 2,
        read_metadata_from_file: bool = False,
        metadata_path: Optional[str] = None,
        # TODO USE **kwargs in contructor
    ) -> None:
        """Constructor of Image object.

        The input image can either be a path to the image file or a numpy array with
        conventional matrix indexing.

        Arguments:
            img (np.ndarray): image array with matrix indexing
            origo (np.ndarray): physical coordinates of the lower left corner
            width (float): physical width of the image
            height (float): physical height of the image
            depth (float): physical depth of the image, only relevant for 3d images
            dim (int): dimension of image, could be 2, 3, 4 (incl. the possibility for time)
            read_metadata_from_file (bool): controlling whether meta data is read from file
            metadata_path (str): path to metadata; should be changed if metadata is not placed
                in the "images/metadata"-folder and taking the same path as the image. If
                image is not passed as a path one also needs to pass it as a path.
        """

        # Fetch image
        if isinstance(img, np.ndarray):
            self.img = img
            self.name = "Unnamed image"
        elif isinstance(img, str):
            self.img = cv2.imread(str(Path(img)))
            self.imgpath = img
            self.name = img
        else:
            raise Exception(
                "Invalid image data. Provide either a path to an image or an image array."
            )

        # Read metadata from file or create from input arguments
        if read_metadata_from_file:
            self.create_metadata_from_file(metadata_path)
        else:
            self.origo = origo if isinstance(origo, np.ndarray) else np.array(origo)
            self.width = width
            self.height = height
            self.dim = dim
        self.shape = self.img.shape

        # Determine numbers of cells in each dimension and cell size
        self.num_pixels_height, self.num_pixels_width = self.shape[:2]
        self.dx = self.width / self.num_pixels_width
        self.dy = self.height / self.num_pixels_height
        # ... in 3d
        if self.dim == 3:
            self.depth = depth
            self.num_pixels_depth = self.shape[2]
            self.dz = self.depth / self.num_pixels_depth

        # Define the pixels in the corners of the image
        self.corners = {
            "upperleft": np.array([0, 0]),
            "lowerleft": np.array([self.num_pixels_height, 0]),
            "lowerright": np.array([self.num_pixels_height, self.num_pixels_width]),
            "upperright": np.array([0, self.num_pixels_width]),
        }

        # Establish a coordinate system based on the metadata
        self.coordinatesystem: da.CoordinateSystem = da.CoordinateSystem(self)

        self.colorspace: str = "bgr"

    # There might be a cleaner way to do this. Then again, it works.
    def create_metadata_from_file(self, path: Optional[str]) -> None:
        """Reading routine for metadata.

        Arguments:
            path (str, optional): path to metadata
        """
        # If path is None, expect metadata in subfolder at same location
        # as the orginal image. Otherwise use given path.
        if path is None:
            assert isinstance(self.imgpath, str)
            # TODO This may not be OS independent.
            pl = self.imgpath.split("/")
            name = pl[2].split(".")[0]
            path = pl[0] + "/metadata/" + name + ".txt"
        try:
            f = open(str(Path(path)), "r")
        except OSError:
            print(f"Could not open/read file: {path}")
            sys.exit()

        # Read in meta data: origo, width, height, dim
        md_dict = {}
        for line in f:
            key, value = line.split(":")
            md_dict[key] = value
        # FIXME The origo numbers are hardcoded, might be a better solution
        origo_nums = md_dict["Origo"].replace("[", "").replace("]", "").split(",")
        self.origo = np.array([float(origo_nums[0]), float(origo_nums[1])])
        self.width = float(md_dict["Width"])
        self.height = float(md_dict["Height"])
        self.dim = int(md_dict["Dimension"])

    def write(
        self,
        name: str = "img",
        path: str = str(Path("images/modified/")),
        file_format: str = ".jpg",
        save_metadata: bool = True,
        metadata_path: str = str(Path("images/metadata/")),
    ) -> None:
        """Write image to file.

        Here, the BGR-format is used. Image path, name and format
        can be changed by passing them as strings to the method.

        Arguments:
            name (str): name of image
            path (str): tpath to image (only folders)
            file_format (str): file ending, deifning the image format
            save_metadata (bool): controlling whether metadata is stored
            metadata_path (str): path to metadata (only folders); the metadata file
                has the same name as the image (and a .txt ending)
        """
        # Write image, using the conventional matrix indexing
        cv2.imwrite(str(Path(path + name + file_format)), self.img)
        print("Image saved as: " + str(Path(path + name + file_format)))

        # Write meta data
        if save_metadata:
            f = open(str(Path(metadata_path + name + ".txt")), "w")
            f.write("Origo: " + str(self.origo) + "\n")
            f.write("Width: " + str(self.width) + "\n")
            f.write("Height: " + str(self.height) + "\n")
            f.write("Dimension: " + str(self.dim) + "\n")
            f.close()

    def show(self, name: Optional[str] = None, wait: Optional[int] = 0) -> None:
        """Show image.

        Arguments:
            name (str, optional): name addressing the window for visualization
            wait (int, optional): waiting time in milliseconds, if not set
                the window is open until any key is pressed.
        """
        if name is None:
            name = self.name

        if self.colorspace == "rgb":
            bgrim = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        else:
            bgrim = self.img

        # Display image
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, bgrim)
        cv2.waitKey(wait)
        cv2.destroyAllWindows()

    def plt_show(self) -> None:
        """Show image using matplotlib.pyplots built-in imshow"""

        if self.colorspace == "bgr":
            rgbim = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        else:
            rgbim = self.img
        plt.imshow(rgbim)

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
        return Image(self.img, self.origo, self.width, self.height)

    def toBGR(self) -> None:
        if self.colorspace == "rgb":
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.colorspace = "bgr"

    def toRGB(self) -> None:
        if self.colorspace == "bgr":
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.colorspace = "rgb"
