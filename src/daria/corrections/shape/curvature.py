"""
Curvature correction class.

A class for setup and application of curvature correction.
"""

from __future__ import annotations

import copy
from email.mime import image
import json
import math
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import skimage
from PIL import Image as PIL_Image
from scipy.ndimage import map_coordinates

import daria as da


class CurvatureCorrection:
    """Class for curvature correction of curved images.

    Contains routines for setting up the curvature correction, as well as applying
    it to images.

    Attributes:
        config (dict): config dictionary for curvture correction.

        Circumstantial attributes:
            reference_image (np.ndarray): image matrix of the reference image.
            current_image (np.ndarray): image matrix of the updated reference image.
            width (float): physical width of reference image.
            height (float): physical height of reference image.
            in_meters (bool): True if width/height is in meters.
            Ny (int): number of pixels in vertical direction in reference image.
            Nx (int): number of pixels in horizontal direction in reference image.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        """
        Constructor of curvature correction class.

        NOTE: CurvatureCorrection should be mostly initialized with a config file
        which controls the correction routine. The possibility to define a curvature
        correction using a path to an image (not a daria.Image) should be however
        only used for setting up the config file via CurvatureCorrection as
        showcased in examples/notebooks/curvature_correction_walkthrough.ipynb

        Arguments:
            kwargs (Optional keyword arguments):
                config (dict, optional): config dictionary; default is None
                image_source (Union[Path, np.ndarray]): image source that either can
                            be provided as a path to an image or an image matrix.
                            Either this or the config_source must be provided.
                config_source (Path): path to the config source. Either this or the
                            image_source must be provided.
                width (float): physical width of the image. Only relevant if
                            image_source is provided.
                height (float): physical height of the image. Only relevant if
                            image_source is provided.
                in_meters (bool): returns True if width and height are given
                            in terms of meters. Only relevant if image_source
                            is provided.
        """

        if config is not None:
            # Read config directly from argument list
            self.config = copy.deepcopy(config)

        elif "image_source" in kwargs:
            self.config: dict() = {}
            im_source = kwargs.pop("image_source")
            if isinstance(im_source, np.ndarray):
                self.reference_image = im_source

            elif isinstance(im_source, str):
                self.reference_image = np.array(PIL_Image.open(Path(im_source)))

            else:
                raise Exception(
                    "Invalid image data. Provide either a path to an image or an image array."
                )
            self.current_image = np.copy(self.reference_image)
            self.dtype = self.current_image.dtype
            self.Ny, self.Nx = self.reference_image.shape[:2]
            self.in_meters = kwargs.pop("in_meters", True)
            self.width = kwargs.pop("width", 1.0)
            self.height = kwargs.pop("height", 1.0)

        elif "config_source" in kwargs:

            config_source = kwargs.pop("config_source")
            assert isinstance(config_source, str)
            with open(str(Path(config_source)), "r") as openfile:
                self.config = json.load(openfile)

        else:
            raise Exception(
                "Please provide either an image as 'image_source' \
                    or a config file as 'config_source'."
            )

        # The internally stored config file is tailored to when resize_factor is equal to 1.
        # For other values, it has to be adapted.
        self.resize_factor = kwargs.pop("resize_factor", 1.0)
        if not math.isclose(self.resize_factor, 1.0):
            self._adapt_config()

        # Initialize cache for precomputed transformed coordinates
        self.cache = {}

        # Hardcode the interpolation order, used when mapping pixels to transformed
        # coordinates
        self.interpolation_order: int = kwargs.pop("interpolation_order", 1)

    # ! ---- I/O routines

    def write_config_to_file(self, path: Path) -> None:
        """
        Writes the config dictionary to a json-file.

        Arguments:
            path (Path): path to the json file
        """

        with open(str(path), "w") as outfile:
            json.dump(self.config, outfile, indent=4)

    def read_config_from_file(self, path: Path) -> None:
        """
        Reads a json-file to the config disctionary.

        Arguments:
            path (Path): path to the json-file.
        """
        with open(str(path), "r") as openfile:

            self.config = json.load(openfile)

    def return_image(self) -> da.Image:
        """
        Returns the current image as a daria image width provided width and height.
        """
        return da.Image(self.temporary_image, width=self.width, height=self.height)

    def show_image(self) -> None:
        """
        Shows the current image using matplotlib.pyplot
        """
        plt.imshow(self.temporary_image)
        plt.show()

    @property
    def temporary_image(self):
        return skimage.util.img_as_ubyte(self.current_image)

    # ! ---- Wrappers for single transformations

    def pre_bulge_correction(self, **kwargs) -> None:
        """
        Initialize the curvature correction by forcing all stright lines
        to curve inwards and not outwards.

        Arguments:
            kwargs (optional keyword arguments):
                "horizontal_bulge" (float): parameter for the curvature correction related to
                    the horizontal bulge of the image.
                "horizontal_center_offset" (int): offset in terms of pixel of the image center
                    in x-direction, as compared to the numerical center
                vertical_bulge (float): parameter for the curvature correction related to the
                    vertical bulge of the image.
                "vertical_center_offset" (int): offset in terms of pixel of the image center in
                    y-direction, as compared to the numerical center
        """
        self.config["init"] = {
            "horizontal_bulge": kwargs.pop("horizontal_bulge", 0),
            "horizontal_center_offset": kwargs.pop("horizontal_center_offset", 0),
            "vertical_bulge": kwargs.pop("verical_bulge", 0),
            "vertical_center_offset": kwargs.pop("vertical_center_offset", 0),
        }
        self.current_image = self.simple_curvature_correction(
            self.current_image, **self.config["init"]
        )

    def crop(self, corner_points: list) -> None:
        """
        Crop the image along the corners of the image.

        The four corner points of the image should be provided, and this method
        will update the config file and modify the current image.

        Arguments:
            corner_points (list): list of the corner points. Preferably the list
                        should be ordered starting from the upper left corner
                        and going counter clockwise.
        """

        self.config["crop"] = {
            "pts_src": corner_points,
            "width": self.width,
            "height": self.height,
            "in meters": self.in_meters,
        }

        self.current_image = da.extract_quadrilateral_ROI(
            self.current_image, **self.config["crop"]
        )

    def bulge_corection(
        self, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0
    ) -> None:
        """
        Bulge correction

        Corrects bulging of image, depending on the amount of pixels that the
        image is bulged inwards on each side.

        Arguments:
            left (int): amount of bulged pixels on the left side of the image.
            right (int): amount of bulged pixels on the right side of the image.
            top (int): amount of bulged pixels on the top of the image.
            bottom (int): amount of bulged pixels on the bottom of the image.
        """
        (
            horizontal_bulge,
            horizontal_bulge_center_offset,
            vertical_bulge,
            vertical_bulge_center_offset,
        ) = self.compute_bulge(left=left, right=right, top=top, bottom=bottom)
        print(self.compute_bulge(left=left, right=right, top=top, bottom=bottom))
        self.config["bulge"] = {
            "horizontal_bulge": horizontal_bulge,
            "horizontal_center_offset": horizontal_bulge_center_offset,
            "vertical_bulge": vertical_bulge,
            "vertical_center_offset": vertical_bulge_center_offset,
        }

        self.current_image = self.simple_curvature_correction(
            self.current_image, **self.config["bulge"]
        )

    def stretch_correction(
        self,
        point_source: list,
        point_destination: list,
        stretch_center: list,
    ) -> None:
        """
        Stretch correction.

        Stretches the image depending on the displacement of a
        single point (point source <--> point_destination) and
        an undisplaced point (stretch_center)

        Arguments:
            "point_source" (list): point that has been translated.
            "point_destination" (list): the ought to be position.
            "stretch_center" (list): the stretch center.
        """
        (
            horizontal_stretch,
            horizontal_stretch_center_offset,
            vertical_stretch,
            vertical_stretch_center_offset,
        ) = self.compute_stretch(
            point_source=point_source,
            point_destination=point_destination,
            stretch_center=stretch_center,
        )

        self.config["stretch"] = {
            "horizontal_stretch": horizontal_stretch,
            "horizontal_center_offset": horizontal_stretch_center_offset,
            "vertical_stretch": vertical_stretch,
            "vertical_center_offset": vertical_stretch_center_offset,
        }

        self.current_image = self.simple_curvature_correction(
            self.current_image, **self.config["stretch"]
        )

    # ! ---- Auxiliary routines for computing tuning parameters in the correction.

    def compute_bulge(self, **kwargs):
        """
        Compute the bulge parameters depending on the maximum number of pixels
        that the image has been displaced on each side.

        Arguments:
            kwargs (optional keyword arguments):
                "left" (int): the maximum number of pixels that the image
                              has been displaced on the left side
                "right" (int): the maximum number of pixels that the image
                              has been displaced on the right side
                "top" (int): the maximum number of pixels that the image
                              has been displaced on the top
                "bottom" (int): the maximum number of pixels that the image
                              has been displaced on the bottom
        """

        left = kwargs.pop("left", 0)
        right = kwargs.pop("right", 0)
        top = kwargs.pop("top", 0)
        bottom = kwargs.pop("bottom", 0)

        # Determine the center of the image
        if (left + right == 0) and (top + bottom == 0):
            image_center = [round(self.Nx/2), round(self.Ny/2)]
        elif (left + right == 0):
            image_center = [round(self.Nx/2), round(self.Ny * (top) / (top + bottom))]
        elif (top + bottom == 0):
            image_center = [round(self.Nx * (left) / (left + right)), round(self.Ny/2)]
        else:
            image_center = [
                round(self.Nx * (left) / (left + right)),
                round(self.Ny * (top) / (top + bottom)),
            ]

        print(image_center)
        print(self.Nx)
        print(self.Ny)
        # Determine the offset of the numerical center of the image
        horizontal_bulge_center_offset = image_center[0] - round(self.Nx / 2)
        vertical_bulge_center_offset = image_center[1] - round(self.Ny / 2)

        # Determine the bulge tuning coefficients as explained in the daria notes
        # Assume here that the maximum impressions are applied at the image center
        horizontal_bulge = left / (
            (left - image_center[0]) * image_center[1] * (self.Ny - image_center[1])
        )
        vertical_bulge = top / (
            (top - image_center[1]) * image_center[0] * (self.Nx - image_center[0])
        )

        return (
            horizontal_bulge,
            horizontal_bulge_center_offset,
            vertical_bulge,
            vertical_bulge_center_offset,
        )

    def compute_stretch(self, **kwargs):
        """
        Compute the stretch parameters depending on the stretch center,
        and a known translation.

        Arguments:
            kwargs (optional keyword arguments):
                "point_source" (list): point that has been translated.
                "point_destination" (list): the ought to be position.
                "stretch_center" (list): the stretch center.
        """

        pt_src = kwargs.pop("point_source", [self.Ny, self.Nx])
        pt_dst = kwargs.pop("point_destination", [self.Ny, self.Nx])
        stretch_center = kwargs.pop(
            "stretch_center", [round(self.Ny / 2), round(self.Nx / 2)]
        )

        # Update the offset to the center
        horizontal_stretch_center_offset = stretch_center[0] - round(self.Nx / 2)
        vertical_stretch_center_offset = stretch_center[1] - round(self.Ny / 2)

        # Compute the tuning parameter as explained in the notes
        horizontal_stretch = -(pt_dst[0] - pt_src[0]) / (
            (pt_src[0] - stretch_center[0]) * pt_src[0] * (self.Nx - pt_src[0])
        )
        vertical_stretch = -(pt_dst[1] - pt_src[1]) / (
            (pt_src[1] - stretch_center[1]) * pt_src[1] * (self.Ny - pt_src[1])
        )

        return (
            horizontal_stretch,
            horizontal_stretch_center_offset,
            vertical_stretch,
            vertical_stretch_center_offset,
        )

    # ! ---- Main correction routines

    def __call__(
        self,
        img: Union[str, Path, np.ndarray],
        update_cache: bool = False,
    ) -> np.ndarray:
        """
        Call method of the curvature correction.

        Applies the curvature correction to a provided image, and returns the
        corrected image as an array. If set in the constructor, the image
        will be resized in the first step.

        Arguments:
            img (np.ndarray): image array
            update_cache (bool): flag controlling whether the transformed coordinates are
                precomputed even if already existent; default is False.

        Returns:
            np.ndarray: curvature corrected image.
        """
        # FIXME: I suggest to remove the possibility of calling images via str outside
        # of daria.Image. Opening an image comes with a pre-defined choice of the
        # colorspace. This control is lost if we do not further pass this information.
        # Also, modular thinking has had its success in software development for a reason.
        # I would also say, either the user is loading the image him/herself but then
        # is required to provide all other necessary information. Other than that, I am
        # in favor of centralizing opening images from file. This be either in daria.Image
        # or if need in utils (in addition). But I think, only daria.Image is the right
        # place for it.

        # Read image
        if isinstance(img, str) or isinstance(img, Path):
            img = np.array(PIL_Image.open(Path(img)))
        assert isinstance(img, np.ndarray)

        # Precompute transformed coordinates based on self.config, if required.
        if update_cache or "grid" not in self.cache:
            self._precompute_transformed_coordinates(img)

        # Fetch precomputed transformed coordinates and the shape of the transformed image.
        grid = self.cache["grid"]
        shape = self.cache["shape"]

        # Determine the corrected image
        corrected_img = self._transform_image(img, grid, shape)

        return corrected_img

    # TODO: Add an automatic way (using e.g, gradient decent) to choose the parameters.
    # OR determine manual tuning rules.
    def simple_curvature_correction(
        self,
        img: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        General routine for applying stretch and bulge transforms. This routine
        in contrast to __call__ does always use the keyword arguments and constructs
        the transformation instead of using cached values.

        Args:
            img (np.ndarray): image array
            kwargs (optional keyword arguments): see _transform_coordinates for more details.

        Returns:
            np.ndarray: corrected image
        """
        # Read size of image
        Ny, Nx = img.shape[:2]

        # Define coordinates
        x = np.arange(Nx)
        y = np.arange(Ny)

        # Construct associated meshgrid with Cartesian indexing
        X, Y = np.meshgrid(x, y)

        # Transform coordinates accoring to input
        X, Y = self._transform_coordinates(X, Y, **kwargs)

        # Create out grid as the corrected grid, use (row,col) format
        grid = np.array([Y.ravel(), X.ravel()])
        shape = X.shape[:2]

        # Determine the corrected image
        corrected_img = self._transform_image(img, grid, shape)

        return corrected_img

    # ! ---- Main auxiliary correction routines - dirctly called in the main routines

    def _precompute_transformed_coordinates(self, img: np.ndarray) -> None:
        """
        Definition of the standard coordinate transformation routine and the
        order of transformation. Furthermore, this routine implicitly defines
        hardcoded keywords addressing the single transformation.

        The final result is stored in cache.

        Args:
            img (np.ndarray)
        """
        # Read size of image
        Ny, Nx = img.shape[:2]

        # Define coordinates
        x = np.arange(Nx)
        y = np.arange(Ny)

        # Construct associated meshgrid with Cartesian indexing
        X, Y = np.meshgrid(x, y)

        if "init" in self.config:
            X, Y = self._transform_coordinates(X, Y, **self.config["init"])

        if "crop" in self.config:
            X = da.extract_quadrilateral_ROI(X, **self.config["crop"])
            Y = da.extract_quadrilateral_ROI(Y, **self.config["crop"])

        if "bulge" in self.config:
            X, Y = self._transform_coordinates(X, Y, **self.config["bulge"])

        if "stretch" in self.config:
            X, Y = self._transform_coordinates(X, Y, **self.config["stretch"])

        # Create out grid as the corrected grid, use (row,col) format
        grid = np.array([Y.ravel(), X.ravel()])

        # Store grid and shape
        self.cache["grid"] = grid
        self.cache["shape"] = X.shape[:2]

    def _adapt_config(self) -> None:
        """
        Adapt config file for resized images, assuming config is correct
        for resize_factor = 1.
        """
        for mainkey in ["init", "bulge"]:
            if mainkey in self.config:
                for key in [
                    "horizontal_bulge",
                    "vertical_bulge",
                    "horizontal_center_offset",
                    "vertical_center_offset",
                ]:
                    if key in self.config[mainkey]:
                        self.config[mainkey][key] *= self.resize_factor

        if "crop" in self.config:
            self.config["crop"]["pts_src"] = (
                self.resize_factor * np.array(self.config["crop"]["pts_src"])
            ).tolist()

        if "stretch" in self.config:
            for key in [
                "horizontal_stretch",
                "vertical_stretch",
                "horizontal_center_offset",
                "vertical_center_offset",
            ]:
                self.config["stretch"][key] *= self.resize_factor

    def _transform_coordinates(
        self, X: np.ndarray, Y: np.ndarray, **kwargs
    ) -> tuple[np.ndarray]:
        """
        Routine for applying stretch and bulge transformation of coordinates.

        Args:
            img (np.ndarray): image array
            kwargs (optional keyword arguments): see _transform_coordinates for more details.
                "horizontal_bulge" (float): parameter for the curvature correction related
                    to the horizontal bulge of the image.
                "horizontal_stretch" (float): parameter for the curvature correction related
                    to the horizontal stretch of the image
                "horizontal_center_offset" (int): offset in terms of pixel of the image
                    center in x-direction, as compared to the numerical center
                vertical_bulge (float): parameter for the curvature correction related to
                    the vertical bulge of the image.
                "vertical_stretch" (float): parameter for the curvature correction related
                    to the vertical stretch of the image
                "vertical_center_offset" (int): offset in terms of pixel of the image center
                    in y-direction, as compared to the numerical center

        Returns:
            tuple of arrays: the transformed coordinates; first x and second y.
        """
        # Read in tuning parameters
        horizontal_bulge: float = kwargs.pop("horizontal_bulge", 0.0)
        horizontal_stretch: float = kwargs.pop("horizontal_stretch", 0.0)
        horizontal_center_offset: int = kwargs.pop("horizontal_center_offset", 0)
        vertical_bulge: float = kwargs.pop("vertical_bulge", 0.0)
        vertical_stretch: float = kwargs.pop("vertical_stretch", 0.0)
        vertical_center_offset: int = kwargs.pop("vertical_center_offset", 0)

        Ny, Nx = X.shape[:2]

        # Image center in pixels, but in (col, row) order
        image_center = [
            round(Nx / 2) + horizontal_center_offset,
            round(Ny / 2) + vertical_center_offset,
        ]

        # Define coordinate system relative to image center, in terms of pixels
        X -= image_center[0]
        Y -= image_center[1]

        # Warp the coordinate system nonlinearly, correcting for bulge and stretch effects.
        Xmod = (
            X
            + horizontal_bulge * np.multiply(X, (np.max(Y) - Y) * (Y - np.min(Y)))
            + horizontal_stretch * X * (np.max(X) - X) * (X - np.min(X))
        )
        Ymod = (
            Y
            + vertical_bulge * np.multiply(Y, (np.max(X) - X) * (X - np.min(X)))
            + vertical_stretch * Y * (np.max(Y) - Y) * (Y - np.min(Y))
        )

        # Map corrected grid back to positional arguments, i.e. invert the definition
        # of the local coordinate system
        Xmod += image_center[0]
        Ymod += image_center[1]

        return Xmod, Ymod

    def _transform_image(
        self, img: np.ndarray, grid: tuple[np.ndarray], shape: tuple[int]
    ) -> np.ndarray:
        """
        Routine to transform an image based on transformed coordinates.

        Args:
            img (np.ndarray): image array
            grid (tuple of arrays): x and y components of the transformed coordinates
            shape (tuple): shape of the final image

        Returns:
            np.ndarray: transformed image
        """
        # Initialize the corrected image.
        corrected_img = np.zeros((*shape, img.shape[2]), dtype=img.dtype)

        # Detemine the corrected image using interpolation based on the transformed
        # coordinates.
        for i in range(img.shape[2]):
            # Consider each color channel separately
            in_data = img[:, :, i]
            # Map image to new coordinates
            im_array_as_vector = map_coordinates(
                in_data, grid, order=self.interpolation_order
            )
            # Convert to correct shape and data type
            corrected_img[:, :, i] = im_array_as_vector.reshape(shape).astype(img.dtype)

        return corrected_img
