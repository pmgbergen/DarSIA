"""Curvature correction class.

A class for setup and application of curvature correction.
"""

from __future__ import annotations

import copy
import json
import math
import tomllib
from pathlib import Path
from typing import Optional, Union
from warnings import warn

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy.ndimage import map_coordinates

import darsia


def load_curvature_correction_config_from_toml(path: Path) -> dict:
    data = tomllib.loads(path.read_text())
    config = {}
    try:
        sec = data["curvature"]
    except KeyError:
        raise UserWarning(f"No 'curvature' section found in {path}.")
        return config
    try:
        sec_init = sec["init"]
        if sec_init is not None:
            config["init"] = {
                "horizontal_bulge": sec_init.get("horizontal_bulge", 0.0),
                "vertical_bulge": sec_init.get("vertical_bulge", 0.0),
            }
    except KeyError:
        raise UserWarning(f"No 'curvature.init' section found in {path}.")
    try:
        sec_crop = sec["crop"]
        if sec_crop is not None:
            config["crop"] = {
                "pts_src": darsia.make_voxel(sec_crop.get("pts_src", [])),
                "width": sec_crop.get("width", 1.0),
                "height": sec_crop.get("height", 1.0),
                "in meters": sec_crop.get("in meters", True),
            }
    except KeyError:
        raise UserWarning(f"No 'curvature.crop' section found in {path}.")
    try:
        sec_bulge = sec["bulge"]
        if sec_bulge is not None:
            config["bulge"] = {
                "horizontal_bulge": sec_bulge.get("horizontal_bulge", 0.0),
                "horizontal_center_offset": sec_bulge.get(
                    "horizontal_center_offset", 0
                ),
                "vertical_bulge": sec_bulge.get("vertical_bulge", 0.0),
                "vertical_center_offset": sec_bulge.get("vertical_center_offset", 0),
            }
    except KeyError:
        raise UserWarning(f"No 'curvature.bulge' section found in {path}.")
    try:
        sec_stretch = sec["stretch"]
        if sec_stretch is not None:
            config["stretch"] = {
                "horizontal_stretch": sec_stretch.get("horizontal_stretch", 0.0),
                "horizontal_center_offset": sec_stretch.get(
                    "horizontal_center_offset", 0
                ),
                "vertical_stretch": sec_stretch.get("vertical_stretch", 0.0),
                "vertical_center_offset": sec_stretch.get("vertical_center_offset", 0),
            }
    except KeyError:
        raise UserWarning(f"No 'curvature.stretch' section found in {path}.")
    return config


class CurvatureCorrection(darsia.BaseCorrection):
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

    def __init__(
        self, config: Optional[Union[dict, str, Path]] = None, **kwargs
    ) -> None:
        """
        Constructor of curvature correction class.

        NOTE: CurvatureCorrection should be mostly initialized with a config file
        which controls the correction routine. The possibility to define a curvature
        correction using a path to an image (not a darsia.Image) should be however
        only used for setting up the config file via CurvatureCorrection as
        showcased in examples/notebooks/curvature_correction_walkthrough.ipynb

        Arguments:
            kwargs (Optional keyword arguments):
                config (dict, str, Path): config dictionary; default is None. Either this
                            or the image must be provided.
                image (Union[Path, np.ndarray]): image source that either can
                            be provided as a path to an image or an image matrix.
                            Either this or the config must be provided.
                width (float): physical width of the image. Only relevant if
                            image is provided.
                height (float): physical height of the image. Only relevant if
                            image is provided.
                in_meters (bool): returns True if width and height are given
                            in terms of meters. Only relevant if image
                            is provided.
        """

        if config is not None:
            # Read config directly from argument list
            if isinstance(config, dict):
                self.config = copy.deepcopy(config)
            elif isinstance(config, (str, Path)):
                path = Path(config)
                if path.suffix == ".json":
                    with open(path, "r") as openfile:
                        self.config = json.load(openfile)
                elif path.suffix == ".toml":
                    self.config = load_curvature_correction_config_from_toml(path)
        else:
            self.config = {}

        if "image" in kwargs:
            im_source = kwargs.get("image")
            if isinstance(im_source, np.ndarray):
                self.reference_image = im_source

            elif isinstance(im_source, str):
                self.reference_image = cv2.imread(im_source, cv2.IMREAD_UNCHANGED)
                self.reference_image = cv2.cvtColor(
                    self.reference_image, cv2.COLOR_BGR2RGB
                )

            else:
                raise Exception(
                    "Invalid image data. Provide either a path to an image or an image array."
                )
            self.current_image = np.copy(self.reference_image)
            self.dtype = self.current_image.dtype
            self.in_meters = kwargs.get("in_meters", True)
            self.width = kwargs.get("width", 1.0)
            self.height = kwargs.get("height", 1.0)

        else:
            warn("No image provided. Please provide an image or a config file.")

        # The internally stored config file is tailored to when resize_factor is equal to 1.
        # For other values, it has to be adapted.
        self.resize_factor = kwargs.get("resize_factor", 1.0)
        if not math.isclose(self.resize_factor, 1.0):
            self._adapt_config()

        # Initialize cache for precomputed transformed coordinates
        self.cache: dict = {}
        if hasattr(self, "config"):
            self.use_cache = self.config.get("use_cache", False)
            if self.use_cache:
                self.cache_path = Path(
                    self.config.get("cache", "./cache/curvature_transformation.npy")
                )
                self.cache_path.parents[0].mkdir(parents=True, exist_ok=True)
        else:
            self.use_cache = False

        # Hardcode the interpolation order, used when mapping pixels to transformed
        # coordinates
        self.interpolation_order: int = kwargs.get("interpolation_order", 1)

    # ! ---- I/O routines

    def write_config_to_file(self, path: Union[Path, str]) -> None:
        """
        Writes the config dictionary to a json-file.

        Arguments:
            path (Path): path to the json file
        """

        with open(Path(path), "w") as outfile:
            json.dump(self.config, outfile, indent=4)

    def read_config_from_file(self, path: Path) -> None:
        """
        Reads a json-file to the config disctionary.

        Arguments:
            path (Path): path to the json-file.
        """
        with open(str(path), "r") as openfile:
            self.config = json.load(openfile)

    def save(self, path: Path) -> None:
        """Save the curvature correction to a file.

        Arguments:
            path (Path): path to the file

        """
        # Make sure the parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Store color space and local scaling images as npz files
        np.savez(
            path,
            class_name=type(self).__name__,
            config=self.config,
            cache=self.cache if hasattr(self, "cache") else None,
        )

        print(f"Curvature correction saved to {path}.")

    def load(self, path: Path) -> None:
        """Load the curvature correction from a file.

        Arguments:
            path (Path): path to the file

        """
        # Make sure the file exists
        if not path.is_file():
            raise FileNotFoundError(f"File {path} not found.")

        # Load color space and local scaling images from npz file
        data = np.load(path, allow_pickle=True)
        if "config" not in data:
            raise ValueError("Invalid file format.")
        self.config = data["config"].item()
        pre_cache = data.get("cache", None)
        if pre_cache is not None:
            self.cache = pre_cache.item()

    def return_image(self) -> darsia.Image:
        """
        Returns the current image as a darsia image width provided width and height.
        """
        return darsia.Image(self.temporary_image, width=self.width, height=self.height)

    def show_image(self) -> None:
        """
        Shows the current image using matplotlib.pyplot
        """
        plt.imshow(skimage.img_as_ubyte(self.temporary_image))
        plt.show()

    @property
    def temporary_image(self):
        if self.dtype == np.uint16:
            return skimage.util.img_as_uint(self.current_image)
        else:
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
            "horizontal_bulge": kwargs.get("horizontal_bulge", 0),
            "horizontal_center_offset": kwargs.get("horizontal_center_offset", 0),
            "vertical_bulge": kwargs.get("verical_bulge", 0),
            "vertical_center_offset": kwargs.get("vertical_center_offset", 0),
        }
        self.current_image = self.simple_curvature_correction(
            self.current_image, **self.config["init"]
        )

    def crop(self, corner_points: darsia.VoxelArray) -> None:
        """
        Crop the image along the corners of the image.

        The four corner points of the image should be provided, and this method
        will update the config file and modify the current image.

        Arguments:
            corner_points (VoxelArray): list of the corner points. Preferably the list
                        should be ordered starting from the upper left corner
                        and going counter clockwise.
        """

        if not isinstance(corner_points, darsia.VoxelArray):
            corner_points = darsia.make_voxels(corner_points)

        self.config["crop"] = {
            "pts_src": corner_points,
            "width": self.width,
            "height": self.height,
            "in meters": self.in_meters,
        }

        self.current_image = darsia.extract_quadrilateral_ROI(
            self.current_image, **self.config["crop"]
        )

    def bulge_correction(
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

    # ! ---- Auxiliary routines for computing tuning parameters in the correction.

    def compute_bulge(self, img: Optional[np.ndarray] = None, **kwargs):
        """
        Compute the bulge parameters depending on the maximum number of pixels
        that the image has been displaced on each side.

        Arguments:
            img (np.ndarray, optional): image array, basis for the computation.
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

        left = kwargs.get("left", 0)
        right = kwargs.get("right", 0)
        top = kwargs.get("top", 0)
        bottom = kwargs.get("bottom", 0)

        if img is None:
            Ny, Nx = self.current_image.shape[:2]
        else:
            Ny, Nx = img.shape[:2]

        # Determine the center of the image
        if (left + right == 0) and (top + bottom == 0):
            image_center = [round(Nx / 2), round(Ny / 2)]
        elif left + right == 0:
            image_center = [round(Nx / 2), round(Ny * (top) / (top + bottom))]
        elif top + bottom == 0:
            image_center = [
                round(Nx * (left) / (left + right)),
                round(Ny / 2),
            ]
        else:
            image_center = [
                round(Nx * (left) / (left + right)),
                round(Ny * (top) / (top + bottom)),
            ]

        # Determine the offset of the numerical center of the image
        horizontal_bulge_center_offset = image_center[0] - round(Nx / 2)
        vertical_bulge_center_offset = image_center[1] - round(Ny / 2)

        # Determine the bulge tuning coefficients as explained in the darsia notes
        # Assume here that the maximum impressions are applied at the image center
        horizontal_bulge = left / (
            (left - image_center[0]) * image_center[1] * (Ny - image_center[1])
        )
        vertical_bulge = top / (
            (top - image_center[1]) * image_center[0] * (Nx - image_center[0])
        )

        return (
            horizontal_bulge,
            horizontal_bulge_center_offset,
            vertical_bulge,
            vertical_bulge_center_offset,
        )

    def compute_stretch(self, img: Optional[np.ndarray] = None, **kwargs):
        """
        Compute the stretch parameters depending on the stretch center,
        and a known translation.

        Arguments:
            img (np.ndarray, optional): image array, basis for the computation.
            kwargs (optional keyword arguments):
                "point_source" (list): point that has been translated.
                "point_destination" (list): the ought to be position.
                "stretch_center" (list): the stretch center.
        """

        if img is None:
            Ny, Nx = self.current_image.shape[:2]
        else:
            Ny, Nx = img.shape[:2]

        pt_src = kwargs.get("point_source", [Ny, Nx])
        pt_dst = kwargs.get("point_destination", [Ny, Nx])
        stretch_center = kwargs.get("stretch_center", [round(Ny / 2), round(Nx / 2)])

        # Update the offset to the center
        horizontal_stretch_center_offset = stretch_center[0] - round(Nx / 2)
        vertical_stretch_center_offset = stretch_center[1] - round(Ny / 2)

        # Check whether zero horizontal stretch should be applied
        if (pt_dst[0] - pt_src[0]) == 0:
            horizontal_stretch = 0
        # Check whether point is chosen too close to the center
        # (within 5% of total pixels), and make warning
        elif abs(pt_src[0] - stretch_center[0]) < round(0.05 * Nx):
            horizontal_stretch = 0
            warn(
                "point_source chosen too close to stretch center for correction"
                " in horizontal direction (within 5"
                "%"
                " of pixels in horizontal"
                " direction). Please choose points that are approximately 1/4th"
                " away from the top or the left or right of the image. Horizontal stretch"
                " is set to zero now."
            )
        elif abs(pt_src[0] - Nx) < round(0.05 * Nx):
            horizontal_stretch = 0
            warn(
                "point_source chosen too close to the right edge of the image for correction"
                " in horizontal direction (within 5"
                "%"
                " of pixels in horizontal"
                " direction). Please choose points that are approximately 1/4th"
                " away from the top or the left or right of the image. Horizontal stretch"
                " is set to zero now."
            )
        elif pt_src[0] < round(0.05 * Nx):
            horizontal_stretch = 0
            warn(
                "point_source chosen too close to the left edge of the image for correction"
                " in horizontal direction (within 5"
                "%"
                " of pixels in horizontal"
                " direction). Please choose points that are approximately 1/4th"
                " away from the top or the left or right of the image. Horizontal stretch"
                " is set to zero now."
            )
        else:
            # Compute the tuning parameter as explained in the notes.
            horizontal_stretch = -(pt_dst[0] - pt_src[0]) / (
                (pt_src[0] - stretch_center[0]) * pt_src[0] * (Nx - pt_src[0])
            )

        # Check whether zero vertical stretch should be applied
        if (pt_dst[1] - pt_src[1]) == 0:
            vertical_stretch = 0
        # Check whether point is chosen too close to the center
        # (within 5% of total pixels), and make warning
        elif abs(pt_src[1] - stretch_center[1]) < round(0.05 * Ny):
            vertical_stretch = 0
            warn(
                "point_source chosen too close to stretch center for correction"
                " in vertical direction (within 5"
                "%"
                " of pixels in vertical"
                " direction). Please choose points that are approximately 1/4th"
                " away from the top or the bottom of the image. Vertical stretch"
                " is set to zero now."
            )
        elif pt_src[0] < round(0.05 * Ny):
            vertical_stretch = 0
            warn(
                "point_source chosen too close to the top of the image for correction"
                " in vertical direction (within 5"
                "%"
                " of pixels in vertical"
                " direction). Please choose points that are approximately 1/4th"
                " away from the top or the bottom of the image. Vertical stretch"
                " is set to zero now."
            )
        elif abs(Ny - pt_src[1]) < round(0.05 * Ny):
            vertical_stretch = 0
            warn(
                "point_source chosen too close to the bottom of the image for correction"
                " in vertical direction (within 5"
                "%"
                " of pixels in vertical"
                " direction). Please choose points that are approximately 1/4th"
                " away from the top or the bottom of the image. Vertical stretch"
                " is set to zero now."
            )
        else:
            # Compute the tuning parameter as explained in the notes.
            vertical_stretch = -(pt_dst[1] - pt_src[1]) / (
                (pt_src[1] - stretch_center[1]) * pt_src[1] * (Ny - pt_src[1])
            )

        return (
            horizontal_stretch,
            horizontal_stretch_center_offset,
            vertical_stretch,
            vertical_stretch_center_offset,
        )

    # ! ---- Main correction routines

    # TODO add __call__ and change coordinate system?

    def correct_array(
        self,
        img: np.ndarray,
    ) -> np.ndarray:
        """
        Call method of the curvature correction.

        Applies the curvature correction to a provided image, and returns the
        corrected image as an array. If set in the constructor, the image
        will be resized in the first step.

        Arguments:
            img (np.ndarray): image array

        Returns:
            np.ndarray: curvature corrected image.

        """
        # Precompute transformed coordinates based on self.config, if required.
        if (
            not (self.use_cache and self.cache_path.exists())
            and "grid" not in self.cache
        ):
            self._precompute_transformed_coordinates(img)

            # Store in cache
            if self.use_cache:
                np.save(self.cache_path, self.cache)

        elif self.use_cache and self.cache_path.exists():
            # Reache cache from file
            self.cache = np.load(self.cache_path, allow_pickle=True).item()

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
        x = np.arange(Nx, dtype=np.float32)
        y = np.arange(Ny, dtype=np.float32)

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
        # Define the current pixel mesh before any transformation
        Ny, Nx = img.shape[:2]
        X, Y = np.meshgrid(
            np.arange(Nx, dtype=np.float32), np.arange(Ny, dtype=np.float32)
        )

        # Store references of the pixel coordinates in dict to easily iterate over both
        coords = {
            "X": X,
            "Y": Y,
        }

        for key, pixels in coords.items():
            # Apply transformation in the (only) expected order
            if "init" in self.config:
                pixels = self.simple_curvature_correction(pixels, **self.config["init"])
            if "crop" in self.config:
                pixels = darsia.extract_quadrilateral_ROI(pixels, **self.config["crop"])
            if "bulge" in self.config:
                pixels = self.simple_curvature_correction(
                    pixels, **self.config["bulge"]
                )
            if "stretch" in self.config:
                pixels = self.simple_curvature_correction(
                    pixels, **self.config["stretch"]
                )

            # Store the updated values
            coords[key] = pixels

        # Fetch the updated X, Y
        X = coords["X"]
        Y = coords["Y"]

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
    ) -> tuple[np.ndarray, np.ndarray]:
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
        horizontal_bulge: float = kwargs.get("horizontal_bulge", 0.0)
        horizontal_stretch: float = kwargs.get("horizontal_stretch", 0.0)
        horizontal_center_offset: int = kwargs.get("horizontal_center_offset", 0)
        vertical_bulge: float = kwargs.get("vertical_bulge", 0.0)
        vertical_stretch: float = kwargs.get("vertical_stretch", 0.0)
        vertical_center_offset: int = kwargs.get("vertical_center_offset", 0)

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
        self, img: np.ndarray, grid: np.ndarray, shape: tuple[int, ...]
    ) -> np.ndarray:
        """
        Routine to transform an image based on transformed coordinates.

        Args:
            img (np.ndarray): image array
            grid (np.ndarray): array of x and y components of the transformed coordinates
            shape (tuple): shape of the final image

        Returns:
            np.ndarray: transformed image
        """
        # Initialize the corrected image. To unify code, transform to 3d arrays and
        # transform back in the end if needed.
        img = np.atleast_3d(img)
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
            # Convert to correct shape and data type (if necessary)
            if im_array_as_vector.dtype == img.dtype:
                corrected_img[:, :, i] = im_array_as_vector.reshape(shape)
            else:
                corrected_img[:, :, i] = im_array_as_vector.reshape(shape).astype(
                    img.dtype
                )

        return np.squeeze(corrected_img)

    def correct_metadata(self, metadata: dict = {}) -> dict:
        """Extract metadata from the config file.

        Args:
            metadata (dict, optional): metadata dictionary to be updated. Defaults to {}.

        Returns:
            dict: metadata

        """
        # Initialize metadata
        meta = {}

        # Read physical dimensions from config file
        if "crop" in self.config:
            # Update the metadata
            if all([key in self.config["crop"] for key in ["width", "height"]]):
                # NOTE: Dimensions of Image uses matrix convention, i.e. (rows, cols).
                dimensions = [
                    self.config["crop"]["height"],
                    self.config["crop"]["width"],
                ]
                meta["dimensions"] = dimensions
                meta["origin"] = darsia.CoordinateArray(
                    [0, self.config["crop"]["height"]]
                )

        return meta
