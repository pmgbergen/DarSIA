"""Module containing a Machine learning free color correction/calibration
based on the Classic Color Checker from calibrite / x-rite.
"""

import copy
import json
from abc import ABC
from pathlib import Path
from typing import Literal, Optional, Union
from warnings import warn

import colour
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia


class ColorChecker(ABC):
    """Base class for color checkers."""

    _reference_swatches_rgb: np.ndarray
    """Reference colors in RGB format, column by column"""

    @property
    def swatches_rgb(self):
        return self._reference_swatches_rgb

    @property
    def swatches_RGB(self):
        return (self._reference_swatches_rgb * 255).astype(np.uint8)

    def plot(self):
        """Plot color checker."""
        # Plot
        _, ax = plt.subplots()
        ax.imshow(self._reference_swatches_rgb)
        ax.set_xlabel("horizontal pixel")
        ax.set_ylabel("vertical pixel")
        ax.set_title("Color checker")
        plt.show()

    def save(self, path: Path) -> None:
        path.parents[0].mkdir(parents=True, exist_ok=True)
        np.save(path, self._reference_swatches_rgb)
        print(f"Color checker saved to {path}.")


class ColorCheckerAfter2014(ColorChecker):
    """Definition of the classic color checker under (hardcoded) default
    illumination conditions.
    """

    def __init__(self):
        # Store reference colors in Lab format.
        # From: https://www.xrite.com/service-support/...
        # ...new_color_specifications_for_colorchecker_sg_and_classic_charts
        reference_swatches_lab = np.array(
            [
                [[37.54, 14.37, 14.92]],
                [[62.73, 35.83, 56.5]],
                [[28.37, 15.42, -49.8]],
                [[95.19, -1.03, 2.93]],
                [[64.66, 19.27, 17.5]],
                [[39.43, 10.75, -45.17]],
                [[54.38, -39.72, 32.27]],
                [[81.29, -0.57, 0.44]],
                [[49.32, -3.82, -22.54]],
                [[50.57, 48.64, 16.67]],
                [[42.43, 51.05, 28.62]],
                [[66.89, -0.75, -0.06]],
                [[43.46, -12.74, 22.72]],
                [[30.1, 22.54, -20.87]],
                [[81.8, 2.67, 80.41]],
                [[50.76, -0.13, 0.14]],
                [[54.94, 9.61, -24.79]],
                [[71.77, -24.13, 58.19]],
                [[50.63, 51.28, -14.12]],
                [[35.63, -0.46, -0.48]],
                [[70.48, -32.26, -0.37]],
                [[71.51, 18.24, 67.37]],
                [[49.57, -29.71, -28.32]],
                [[20.64, 0.07, -0.46]],
            ]
        ).astype(np.float32)

        # Resort from column-by-column to a row-by-row format
        reference_swatches_lab = reference_swatches_lab.reshape((4, 6, 3), order="F")

        # Convert reference swatch colors to RGB
        self._reference_swatches_rgb = cv2.cvtColor(
            np.atleast_3d(reference_swatches_lab), cv2.COLOR_Lab2RGB
        )
        """Reference colors in RGB format, column by column"""


class CustomColorChecker(ColorChecker):
    """Swatch colors determined from user-prescribed input image."""

    def __init__(
        self,
        reference_colors: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        path: Optional[Path] = None,
    ) -> None:
        """Define custom color checker.

        Three possible sources are available for the reference colors:
        1. Read from array in (4,6,3) shaped array
        2. Extract from image
        3. Read from file
        The priority is as above.

        Args:
            colors (np.ndarray, optional): reference RGB colors
            image (np.ndarray, optional): image restricted to color checker
            path (Path, optional): path for storing and fetching reference colors

        """
        # Assert only one argument is provided
        assert (
            np.count_nonzero(
                [reference_colors is not None, image is not None, path is not None]
            )
            == 1
        ), "Provide exactly one of the following: colors, image, path."

        self._reference_swatches_rgb: np.ndarray
        """Reference colors in RGB format, column by column."""

        if reference_colors is not None:
            self._reference_swatches_rgb = reference_colors.copy()

        if image is not None:
            self._reference_swatches_rgb = self._extract_from_image(image)

        if path is not None:
            self._reference_swatches_rgb = np.load(path)

    def _extract_from_image(self, img: np.ndarray) -> np.ndarray:
        """
        ML-free variant of detect_colour_checkers_segmentation from colour.
        Exepcts images to be restricted to the ROI such that the landmarks in the
        corners are also the corners of the image.

        Args:
            img: image of color checker.

        Returns:
            np.ndarray: 4 x 6 array with colour checker colors.
        """
        # Assert that the image is in uint8 or uint16 format
        if img.dtype in [np.uint8, np.uint16]:
            img = skimage.img_as_float(img)

        # Resize to fixed size
        img = darsia.extract_quadrilateral_ROI(
            img, pts_src=None, width=27.3, height=17.8
        )
        Ny, Nx = img.shape[:2]
        fixed_width = 500
        fixed_height = int(Ny / Nx * fixed_width)
        resized_img = cv2.resize(img, (fixed_width, fixed_height))

        # Upper left corners of swatches
        swatch_pos_row, swatch_pos_col = np.meshgrid(
            [12, 93, 175, 255],
            [12, 95, 177, 260, 344, 427],
            indexing="ij",
        )
        swatch_size = 50

        # Extract colors of all swatches in RGB by taking averages
        swatches = np.zeros((4, 6, 3), dtype=np.float32)
        for row in range(4):
            for col in range(6):
                pos_row = swatch_pos_row[row, col]
                pos_col = swatch_pos_col[row, col]
                roi = (
                    slice(pos_row, pos_row + swatch_size),
                    slice(pos_col, pos_col + swatch_size),
                )

                pixels = np.float32(resized_img[roi].reshape(-1, 3))
                n_colors = 5
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    200,
                    0.1,
                )
                flags = cv2.KMEANS_RANDOM_CENTERS
                _, labels, palette = cv2.kmeans(
                    pixels, n_colors, None, criteria, 10, flags
                )
                _, counts = np.unique(labels, return_counts=True)
                dominant_color = palette[np.argmax(counts)]
                swatches[row, col] = dominant_color

        return swatches


class ColorCorrection(darsia.BaseCorrection):
    """ML-free color correction.

    Precise user-input is required for detecting the color checker. The Calibrite/
    X-rite color checker has four corner landmarks on the colorchecker. The pixel
    coordinates of these have to be provided as input parameters, starting with the
    corner close to the brown swatch, and continuing in counter-clockwise direction.
    The color correction is applied prior to curvature correction in
    the darsia.Image initialization, and therefore, the roi should be with respect to
    the uncorrected image.
    """

    def __init__(
        self,
        base: Optional[Union[darsia.Image, ColorChecker]] = None,
        config: Optional[dict] = None,
    ):
        """
        Constructor of converter, setting up a priori all data needed for fast conversion.

        Args:
            base (Image or ColorChecker, optional): reference defining a color checker; if
                None provided, use CustomColorChecker.
            config (dict, str, Path): config file for initialization of images; keys:
                "roi" (tuple of slices, np.ndarray, or None): ROI containing a colour
                    checker, provided either as intervals, corner points, or nothing. The
                    recommended choice is to provide an array of coordinates.
                "whitebalancing" (bool): apply white balancing based on the third bottom left
                    swatch if True, default is True
                "verbosity" (bool): flag controlling whether extracted ROIs of the colorchecker
                    as well as the extracted swatch colors are displayed. Useful for debugging.
        """

        # Define config
        if config is not None:
            self.config: dict = copy.deepcopy(config)
            """Config dictionary for initialization of color correction."""
            self._init_from_config(base)
        else:
            self.config = {}

    def _init_from_config(
        self, base: Optional[Union[darsia.Image, ColorChecker]]
    ) -> None:
        """Auxiliary function for initialization from config.

        Args:
            base (Image or ColorChecker, optional): reference defining a color checker; if
                None provided, use CustomColorChecker.

        """
        self.active: bool = self.config.get("active", True)
        """Flag controlling whether correction is active"""

        self.whitebalancing: bool = self.config.get("whitebalancing", True)
        """Flag controlling whether whitebalance is applied as part of correction"""

        self.colorbalancing: Literal["affine", "linear"] = self.config.get(
            "colorbalancing", "affine"
        )
        """Mode for color balancing applied as part of correction"""

        self.verbosity: bool = self.config.get("verbosity", False)
        """Flag controlling whether intermediate output is printed to screen"""

        roi = self.config["roi"]
        assert roi is not None, "Provide ROI for color correction."
        self.roi: darsia.VoxelArray = darsia.make_voxel(roi)
        """ROI - anti-clockwise oriented markers starting at the brown swatch"""

        self.balancing: Literal["colour", "darsia"] = self.config.get(
            "balancing", "darsia"
        )

        self.clip: bool = self.config.get("clip", False)
        """Flag controlling whether values outside the feasible range [0., 1.] are clipped"""

        # Construct color checker
        if base is None:
            base = self.config.get("colorchecker", None)
        self._setup_colorchecker(base)

    def correct_array(
        self,
        img: np.ndarray,
    ) -> np.ndarray:
        """
        Similar workflow as by colour-science to match the colors of the color checker with
        the corresponding reference values, but tailored and simplified, based on the precise
        user-input on the location of the color checker. Reference to the general workflow:
        https://github.com/colour-science/colour-checker-detection/blob/master/colour_checker_detection/examples/examples_detection.ipynb

        Args:
            img (np.ndarray): image in RGB space, with values in uint8,
                uint16, float32, or float64.

        Returns:
            np.ndarray: corrected image
        """
        if not self.active:
            return skimage.img_as_float(img).astype(np.float32)

        # Make sure that the image is in uint8 or uint16 format
        if img.dtype in [np.uint8, np.uint16]:
            img = skimage.img_as_float(img)
        if img.dtype not in [np.float32, np.float64]:
            raise ValueError(
                "Provide image in np.uint8, np.uint16, np.float32, or np.float64 format."
            )

        # Extract part of the image containing a color checker.
        colorchecker_img: np.ndarray = self._restrict_to_roi(img)

        # Determine swatch colors
        colorchecker = CustomColorChecker(image=colorchecker_img)
        swatches = colorchecker.swatches_rgb
        reference_swatches = self.colorchecker.swatches_rgb

        if self.balancing == "colour":
            # Use methods from colour for balancing color and white balancing

            # Flatten column-by-column
            reference_swatches = np.squeeze(
                reference_swatches.reshape((24, 1, 3), order="F")
            )
            swatches = np.squeeze(swatches.reshape((24, 1, 3), order="F"))

            if self.colorbalancing == "affine":
                warn(
                    "Affine color balancing not implemented for balacong via 'colour'."
                )

            # Apply color correction onto full image based on the swatch colors in
            # comparison with the standard colors
            corrected_img = colour.colour_correction(
                skimage.img_as_float(img),
                swatches,
                reference_swatches,
                method="Cheung 2004",
            )

            # Apply white balancing, such that the third bottom left swatch of the color
            # checker is exact. As swatch colors are stored column-by-column, this particular
            # swatch is at position 11.
            if self.whitebalancing:
                corrected_colorchecker_img: np.ndarray = self._restrict_to_roi(
                    corrected_img
                )
                swatches = CustomColorChecker(
                    image=corrected_colorchecker_img
                ).swatches_rgb
                swatches = np.squeeze(swatches.reshape((24, 1, 3), order="F"))
                pos = 11
                corrected_img *= reference_swatches[pos, :] / swatches[pos, :]

        elif self.balancing == "darsia":

            # DarSIA implementation fully affine, both for WB and CB
            balance = darsia.AdaptiveBalance()
            img = skimage.img_as_float(img)
            if self.whitebalancing:
                balance.find_balance(
                    swatches[-1], reference_swatches[-1], mode="diagonal"
                )
            balance.find_balance(
                swatches[:-1],
                reference_swatches[:-1],
                mode="affine" if self.colorbalancing == "affine" else "linear",
            )
            corrected_img = balance.apply_balance(img)

        else:
            raise ValueError(
                f"balancing {self.balancing} not supported, choose 'colour' or 'darsia'"
            )

        # Error analysis
        # colorchecker_img_cb = self._restrict_to_roi(corrected_img)
        # corrected_swatches = CustomColorChecker(image=colorchecker_img_cb).swatches_rgb
        # reference_swatches = self.colorchecker.swatches_rgb
        # print(np.min(corrected_img), np.max(corrected_img))
        # print(np.mean(np.abs(corrected_swatches - reference_swatches)))

        # The correction may result in values outside the feasible range [0., 1.].
        # Thus, simply clip the values for consistency.
        if self.clip:
            corrected_img = np.clip(corrected_img, 0, 1)

        # Ensure a data format which can be used by cv2 etc.
        return corrected_img.astype(np.float32)

    def write_config_to_file(self, path: Union[Path, str]) -> None:
        """
        Writes the config dictionary to a json-file.

        Arguments:
            path (Path): path to the json file
        """
        with open(Path(path), "w") as outfile:
            json.dump(self.config, outfile, indent=4)

    def save(self, path: Path) -> None:
        """Save the color correction to a file.

        Args:
            path (Path): path to the file

        """
        # Make sure that the path exists
        path.parents[0].mkdir(parents=True, exist_ok=True)

        # Save the color correction
        np.savez(
            path,
            class_name=type(self).__name__,
            base=self.colorchecker._reference_swatches_rgb,
            config=self.config,
        )
        print(f"Color correction saved to {path}.")

    def load(self, path: Path) -> None:
        """Load the color correction from a file.

        Args:
            path (Path): path to the file

        """
        assert path.exists(), f"File {path} does not exist."
        data = np.load(path, allow_pickle=True)
        base = CustomColorChecker(reference_colors=data["base"])
        self.config = data["config"].item()
        self._init_from_config(base=base)

    # ! ---- Auxiliary files

    def _setup_colorchecker(
        self, base: Optional[Union[darsia.Image, ColorChecker]]
    ) -> None:
        """Auxiliary setup routine for setting up the custom color checker.

        Defines self.colorchecker.

        Args:
            base (Image or ColorChecker, optional): reference for color checker

        """
        if base is None:
            # Choose a classic color checker
            self.colorchecker = ColorCheckerAfter2014()

        elif isinstance(base, ColorChecker):
            self.colorchecker = base

        elif isinstance(base, darsia.Image):
            # Extract part of the image containing a color checker.
            colorchecker_img = self._restrict_to_roi(base.img)
            self.colorchecker = CustomColorChecker(image=colorchecker_img)

    def _restrict_to_roi(self, img: np.ndarray) -> np.ndarray:
        """
        Auxiliary function to extract roi from image.

        Args:
            img (np.ndarray): input image

        Returns:
            np.ndarray: restricted image

        """
        row_pixels = np.sort(self.roi[:, 0])
        col_pixels = np.sort(self.roi[:, 1])
        row_diff = max(row_pixels[1] - row_pixels[0], row_pixels[3] - row_pixels[2])
        col_diff = max(col_pixels[1] - col_pixels[0], col_pixels[3] - col_pixels[2])
        roi_is_box = row_diff < 0.01 * img.shape[0] and col_diff < 0.01 * img.shape[1]
        atol = max(0.01 * img.shape[0], 0.01 * img.shape[1])
        if roi_is_box:
            # self.roi is more or less a box
            roi_slices = (
                slice(row_pixels[0], row_pixels[3]),
                slice(col_pixels[0], col_pixels[3]),
            )
            box_img = img[roi_slices]
            # need to extract a box with the brown sample first - assume the first
            # voxel in self.roi is the brown sample
            if np.allclose([row_pixels[0], col_pixels[0]], [self.roi[0]], atol=atol):
                # brown sample is in the upper left corner
                return box_img
            elif np.allclose([row_pixels[0], col_pixels[3]], [self.roi[0]], atol=atol):
                # brown sample is in the upper right corner - rotate 90 degrees clockwise
                return np.rot90(box_img, 1)
            elif np.allclose([row_pixels[3], col_pixels[3]], [self.roi[0]], atol=atol):
                # brown sample is in the lower right corner - rotate 180 degrees
                return np.rot90(box_img, -2)
            elif np.allclose([row_pixels[3], col_pixels[0]], [self.roi[0]], atol=atol):
                # brown sample is in the lower left corner - rotate 90 degrees counterclockwise
                return np.rot90(box_img, -1)
            else:
                raise ValueError("The brown sample is not in the corner of the ROI.")

        else:
            # Use width and height (in cm - irrelevant) as provided by the manufacturer Xrite.
            return darsia.extract_quadrilateral_ROI(
                img, pts_src=self.roi, width=27.3, height=17.8, indexing="matrix"
            )
