"""
Curvature correction class.

A class for setup and application of curvature correction.
"""

from __future__ import annotations

from pathlib import Path

import json
from pathlib import Path
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
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

    def __init__(self, **kwargs) -> None:
        """
        Constructor of curvature correction class.

        Arguments:
            kwargs (Optional keyword arguments):
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

        if "image_source" in kwargs:
            self.config: dict() = {}
            im_source = kwargs.pop("image_source")
            if isinstance(im_source, np.ndarray):
                self.reference_image = im_source

            elif isinstance(im_source, Path):
                self.reference_image = cv2.imread(str(im_source))

            else:
                raise Exception(
                    "Invalid image data. Provide either a path to an image or an image array."
                )
            self.current_image = np.copy(self.reference_image)
            self.Ny, self.Nx = self.reference_image.shape[:2]
            self.in_meters = kwargs.pop("in_meters", True)
            self.width = kwargs.pop("width", 1.0)
            self.height = kwargs.pop("height", 1.0)

        elif "config_source" in kwargs:

            config_source = kwargs.pop("config_source")
            assert isinstance(config_source, Path)
            with open(str(config_source), "r") as openfile:
                self.config = json.load(openfile)

        else:
            raise Exception(
                "Please provide either an image as 'image_source' \
                    or a config file as 'config_source'."
            )

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

    def return_current_image(self) -> da.Image:
        """
        Returns the current image as a daria image width provided width and height.
        """
        return da.Image(self.current_image, width=self.width, height=self.height)

    def __call__(self, image_source: Union[str, np.ndarray]) -> np.ndarray:
        """
        Call method of the curvature correction.

        Applies the curvature correction to a provided image, and returns the image
        as an array.

        Arguments:
            image_source (Union[str, np.ndarray]): either the path to an image or an
                            image matrix

        Returns:
            np.ndarray: curvature corrected image.
        """
        if isinstance(image_source, np.ndarray):
            image_tmp = image_source
        elif isinstance(image_source, str):
            image_tmp = cv2.imread(str(Path(image_source)))
        else:
            raise Exception(
                "Invalid image data. Provide either a path to an image or an image array."
            )
        image_tmp = self.simple_curvature_correction(image_tmp, **self.config["init"])
        image_tmp = da.extract_quadrilateral_ROI(image_tmp, **self.config["crop"])
        image_tmp = self.simple_curvature_correction(image_tmp, **self.config["bulge"])
        image_tmp = self.simple_curvature_correction(
            image_tmp, **self.config["stretch"]
        )
        return image_tmp

    def write_config_to_file(self, path: Path) -> None:
        """
        Writes the config dictionary to a json-file.

        Arguments:
            path (Path): path to the json file
        """

        with open(str(path), "w") as outfile:
            json.dump(self.config, outfile, indent=4)

    def show_current(self) -> None:
        """
        Shows the current image using matplotlib.pyplot
        """
        plt.imshow(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))

    def read_config_from_file(self, Path: path) -> None:
        """
        Reads a json-file to the config disctionary.

        Arguments:
            path (Path): path to the json-file.
        """
        with open(str(path), "r") as openfile:

            self.config = json.load(openfile)

    # TODO: Add an automatic way (using e.g, gradient decent) to choose the parameters.
    # OR determine manual tuning rules.
    def simple_curvature_correction(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Correction of bulge and stretch effects.

        Args:
            img (np.ndarray): image array
            kwargs (optional keyword arguments):
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
                "interpolation_order (int)": interpolation order to map back transformed
                    image to Cartesian pixel grid

        Returns:
            np.ndarray: corrected image

        # NOTE: The same image size is used, i.e., the aspect ratio of the image
        # is taken the same as the input and it is therefore implicitly assumed
        # that the input image already is warped such that the aspect ratio of
        # the image is correct. Also it i
        """
        # Read in tuning parameters
        horizontal_bulge: float = kwargs.pop("horizontal_bulge", 0.0)
        horizontal_stretch: float = kwargs.pop("horizontal_stretch", 0.0)
        horizontal_center_offset: int = kwargs.pop("horizontal_center_offset", 0)
        vertical_bulge: float = kwargs.pop("vertical_bulge", 0.0)
        vertical_stretch: float = kwargs.pop("vertical_stretch", 0.0)
        vertical_center_offset: int = kwargs.pop("vertical_center_offset", 0)
        interpolation_order: int = kwargs.pop("interpolation_order", 1)

        # Assume a true image in the form of an array is provided
        if not isinstance(img, np.ndarray):
            raise Exception(
                "Invalid image data. Provide either a path to an image or an image array."
            )

        # Read size of image
        Ny, Nx = img.shape[:2]

        # NOTE: Finding the true centre of the image actually depends on many factors
        # including lense properties. Thus, the task is actually quite hard. Here, a
        # simple approach is used, simply choosing the numerical centre of the image
        # corrected by the user.

        # Image center in pixels, but in (col, row) order
        image_center = [
            round(Nx / 2) + horizontal_center_offset,
            round(Ny / 2) + vertical_center_offset,
        ]

        # Define coordinate system relative to image center, in terms of pixels
        x = np.arange(Nx) - image_center[0]
        y = np.arange(Ny) - image_center[1]

        # Construct associated meshgrid with Cartesian indexing
        X, Y = np.meshgrid(x, y)

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

        # Create out grid as the corrected grid, use (row,col) format
        out_grid = np.array([Ymod.ravel(), Xmod.ravel()])

        # Define the shape corrected image.
        img_mod = np.zeros_like(img, dtype=img.dtype)

        # Do interpolate original image on the new grid
        for i in range(img.shape[2]):
            in_data = img[:, :, i]
            im_array = map_coordinates(in_data, out_grid, order=interpolation_order)
            img_mod[:, :, i] = im_array.reshape(img.shape[:2]).astype(img.dtype)

        return img_mod

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
        image_center = [
            int(self.Nx * (left + 1e-6) / (left + right + 2e-6)),
            int(self.Ny * (top + 1e-6) / (top + bottom + 2e-6)),
        ]

        # Determine the offset of the numerical center of the image
        horizontal_bulge_center_offset = image_center[0] - int(self.Nx / 2)
        vertical_bulge_center_offset = image_center[1] - int(self.Ny / 2)

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
            "stretch_center", [int(self.Ny / 2), int(self.Nx / 2)]
        )

        # Update the offset to the center
        horizontal_stretch_center_offset = stretch_center[0] - int(self.Nx / 2)
        vertical_stretch_center_offset = stretch_center[1] - int(self.Ny / 2)

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
