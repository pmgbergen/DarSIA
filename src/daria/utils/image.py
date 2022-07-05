"""Image class.

Images contain the image array, and in addition metadata about origo and dimensions.
"""

import sys
from typing import Optional

import cv2
import daria as da
import numpy as np


class Image:
    """Base image class.

    Contains the actual image, as well as meta data, i.e., base properties such as
    position in global coordinates, width and height. One can either pass in metadata
    (origo, width and height, amongst other entities) by passing them directly to
    the constructor or through a metadata-file (default is to pass metadata diretly
    to the constructor, and if metadatafile is requires the "read_metadata_from_file"
    variable has to be passed as True). Image can either be passed in as numpy array
    or a path to a file (this is automatically detected through the input). Base
    functionality such as saving to image-file, and drawing a grid is provided in the class.

    Attributes:
        img (np.ndarray): image array, with pixel access, using x,y ordering, such that the
            lower left corner of the image corresponds to the (0,0) pixel.
        imgpath (str): path to image, also used to define source for metadata
        width (float): physical width of the image
        height (float): physical height of the image
        depth (float): physical depth of the image (only relevant in 3d)
        origo (list): physical coordinates of the lower left corner, i.e., (0,0) pixel
        dim (int): dimension of image, could be 2, 3, 4 (incl. the possibility for time)
        shape (np.ndarray): num_pixels, as well number of color channels (typically 3 for RGB)
        dx (float): pixel size in x-direction
        dy (float): pixel size in y-direction
        dz (float): pixel size in z-direction
    """

    def __init__(
        self,
        img: np.ndarray,
        origo: list = [0, 0],
        width: float = 1,
        height: float = 1,
        depth: float = 0,
        dim: int = 2,
        read_metadata_from_file: bool = False,
        metadata_path: Optional[str] = None,
        # USE **kwargs in contructor
    ) -> None:
        """Constructor of Image object.

        The input image can either be a path to the image file or a numpy array with
        standard pixel ordering.

        Arguments:
            img (np.ndarray): image array, with standard pixel access, using y,x, such that
                (0,0) corresponds to the top left corner
            origo (list): physical coordinates of the lower left corner, i.e., (0,0) pixel
            width (float): physical width of the image
            height (float): physical height of the image
            depth (float): physical depth of the image, only relevant for 3d images
            dim (int): dimension of image, could be 2, 3, 4 (incl. the possibility for time)
            read_metadata_from_file (bool): controlling whether meta data is read from file
            metadata_path (str): path to metadata; should be changed if metadata is not placed
                in the "images/metadata"-folder and taking the same path as the image. If
                image is not passed as a path one also needs to pass it as a path.
        """

        # Fetch image and convert to physical pixel ordering
        if type(img) == np.ndarray:
            self.img = da.utils.conversions.standardToPhysicalPixelOrdering(img)
        elif type(img) == str:
            self.img = da.utils.conversions.standardToPhysicalPixelOrdering(
                cv2.imread(img)
            )
            self.imgpath = img
        else:
            raise Exception(
                "Invalid image data. Provide either a path to an image or an image array."
            )

        # Read metadata from file or create from input arguments
        if read_metadata_from_file:
            self.create_metadata_from_file(metadata_path)
        else:
            self.origo = origo
            self.width = width
            self.height = height
            self.dim = dim
        self.shape = self.img.shape
        self.dx = self.width / self.shape[1]
        self.dy = self.height / self.shape[0]
        if self.dim == 3:
            self.depth = depth
            self.dz = self.depth / self.shape[2]

    # There might be a cleaner way to do this. Then again, it works.
    def create_metadata_from_file(self, path: Optional[str]) -> None:
        """Reading routine for metadata.

        Arguments:
            path (str): path to metadata
        """
        # If path is None, expect metadata in subfolder at same location
        # as the orginal image. Otherwise use given path.
        if path is None:
            pl = self.imgpath.split("/")
            name = pl[2].split(".")[0]
            path = pl[0] + "/metadata/" + name + ".txt"
        try:
            f = open(path, "r")
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
        self.origo = [float(origo_nums[0]), float(origo_nums[1])]
        self.width = float(md_dict["Width"])
        self.height = float(md_dict["Height"])
        self.dim = float(md_dict["Dimension"])

    def write(
        self,
        name: str = "img",
        path: str = "images/modified/",
        format: str = ".jpg",
        save_metadata: bool = True,
        metadata_path: str = "images/metadata/",
    ) -> None:
        """Write image to file.

        Here, the BGR-format is used. Image path, name and format
        can be changed by passing them as strings to the method.

        Arguments:
            name (str): name of image
            path (str): tpath to image (only folders)
            fomat (str): file ending, deifning the image format
            save_metadata (bool): controlling whether metadata is stored
            metadata_path (str): path to metadata (only folders); the metadata file
                has the same name as the image (and a .txt ending)
        """
        # Write image
        # TODO conversion of pixels required
        cv2.imwrite(path + name + format, self.img)
        print("Image saved as: " + path + name + format)

        # Write meta data
        if save_metadata:
            f = open(metadata_path + name + ".txt", "w")
            f.write("Origo: " + str(self.origo) + "\n")
            f.write("Width: " + str(self.width) + "\n")
            f.write("Height: " + str(self.height) + "\n")
            f.write("Dimension: " + str(self.dim) + "\n")
            f.close()

    # Draws a grid on the image and writes it.
    def draw_grid(
        self,
        DX: float = 100,
        DY: float = 100,
        color: tuple[int] = (0, 0, 125),
        thickness: int = 9,
        name: str = "img-grid",
        path: str = "images/modified/",
        format: str = ".jpg",
    ) -> None:
        # TODO add origin for the grid to allow for translating the grid arbitrarily
        """
        Draws a grid on the image and writes it to file.

        Arguments:
            DX (float): grid size in x-direction, in physical units
            DY (float): grid size in y-direction, in physical units
            color (tuple of int): RGB color of the grid
            thickness (int): thickness of the grid lines
            name (str): name of the image file for exporting
            path (str): path to file
            format (str): file ending, defining the image format
        """
        # TODO conversion of pixels
        # Determine the number of grid lines required for
        num_h_lines: int = round((self.shape[0] + self.origo[1] / self.dy) / DY)
        num_v_lines: int = round((self.shape[1] + self.origo[0] / self.dx) / DX)

        # Start from original image
        gridimg = self.img.copy()

        # Add horizontal grid lines (line by line)
        for row in range(num_h_lines):
            gridimg = cv2.line(
                gridimg,
                (
                    # TODO revisit: is the order correct here? conversion of pixels
                    round(-self.origo[0] / self.dx),
                    round(row * DY),
                ),
                (round(self.shape[1]), round(row * DY)),
                color,
                thickness,
            )

        # Add vertical grid lines (line by line)
        for col in range(num_v_lines):
            gridimg = cv2.line(
                gridimg,
                (
                    # TODO revisit: is the order correct here? conversion of pixels
                    round(col * DX - self.origo[0] / self.dx),
                    round(0),
                ),
                (
                    round(col * DX - self.origo[0] / self.dx),
                    round(self.shape[0] + self.origo[1] / self.dy),
                ),
                color,
                thickness,
            )

        # Write image with grid to file
        cv2.imwrite(path + name + format, gridimg)
