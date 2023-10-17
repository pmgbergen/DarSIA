"""Module containing a Machine learning free color correction/calibration
based on the Classic Color Checker from calibrite / x-rite.
"""

import copy
import json
from pathlib import Path
from typing import Optional, Union
from warnings import warn

import colour
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia


class ColorCheckerAfter2014:
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

        # Convert reference swatch colors to RGB
        self.reference_swatches_rgb = np.squeeze(
            cv2.cvtColor(np.atleast_3d(reference_swatches_lab), cv2.COLOR_LAB2RGB)
        )


class CustomColorChecker:
    """Swatch colors determined from user-prescribed input image."""

    def __init__(
        self, path: Optional[Path] = None, reference_colors: Optional[np.ndarray] = None
    ) -> None:
        """
        Either read reference colors from file or from provided array.
        If colors are provided, these will be prioritized and also written to file.

        Args:
            path (Path): path for storing and fetching reference colors
            colors (np.ndarray, optional): reference RGB colors
        """

        # Prioritize colors if provided.
        if reference_colors is None:
            self.reference_swatches_rgb = np.load(path)

        else:
            self.reference_swatches_rgb = reference_colors
            if path is not None:
                path.parents[0].mkdir(parents=True, exist_ok=True)
                np.save(path, reference_colors)


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
        base: Optional[darsia.Image] = None,
        config: Optional[dict] = None,
        roi: Optional[Union[tuple, np.ndarray, list]] = None,
        verbosity: bool = False,
        whitebalancing: bool = True,
    ):
        """
        Constructor of converter, setting up a priori all data needed for fast conversion.

        Args:
            base (Image, optional): reference image defining a custom color checker.
            config (dict, str, Path): config file for initialization of images. Can be
                used instead of roi, but roi is always prefered if it is present.
            roi (tuple of slices, np.ndarray, or None): ROI containing a colour checker,
                provided either as intervals, corner points, or nothing. The recommended
                choice is to provide an array of coordinates. Can also be provided as
                part of config; roi in config is prioritized.
            verbosity (bool): flag controlling whether extracted ROIs of the colorchecker
                as well as the extracted swatch colors are displayed. Useful for debugging.
            whitebalancing (bool): apply white balancing based on the third bottom left swatch
                if True, default is True
        """

        # Define config
        if config is not None:
            self.config = copy.deepcopy(config)
        else:
            self.config = {}

        # Check whether colorchecker active
        self.active = self.config.get("active", True)

        # Define ROI
        self.roi: Optional[Union[tuple, np.ndarray]] = None
        if "roi" in self.config:
            self.roi = np.array(self.config["roi"])
        elif isinstance(roi, np.ndarray):
            self.roi = roi
            self.config["roi"] = self.roi.tolist()
        elif isinstance(roi, list):
            self.roi = np.array(roi)
            self.config["roi"] = roi
        elif isinstance(roi, tuple):
            warn(
                "An array of corner points is preferred.\
                The tuple will not be stored in the config file."
            )
            self.roi = roi

        # Store flags
        self.verbosity = verbosity
        self.whitebalancing = whitebalancing

        # Reference of the class color checker - allow for the classic color checker and
        # a custom one - assist in creating the latter with tools from ColorCorrection.
        # Use the input image to define a custom color checker if provided.
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
        swatches = self._detect_colour_checkers_segmentation(colorchecker_img)

        # Apply color correction onto full image based on the swatch colors in comparison with
        # the standard colors
        corrected_img = colour.colour_correction(
            skimage.img_as_float(img),
            swatches,
            self.colorchecker.reference_swatches_rgb,
            method="Cheung 2004",
        )

        # Apply white balancing, such that the third bottom left swatch of the color checker
        # is exact. As swatch colors are stored column-by-column, this particular swatch is
        # at position 11.
        if self.whitebalancing:
            corrected_colorchecker_img: np.ndarray = self._restrict_to_roi(
                corrected_img
            )
            swatches = self._detect_colour_checkers_segmentation(
                corrected_colorchecker_img
            )
            pos = 11
            corrected_img *= (
                self.colorchecker.reference_swatches_rgb[pos, :] / swatches[pos, :]
            )

        # For debugging, double-check once more the swatches.
        if self.verbosity:
            corrected_colorchecker_img = self._restrict_to_roi(corrected_img)
            swatches = self._detect_colour_checkers_segmentation(
                corrected_colorchecker_img
            )

        # The correction may result in values outside the feasible range [0., 1.].
        # Thus, simply clip the values for consistency.
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

    # ! ---- Auxiliary files

    def _setup_colorchecker(self, base: Optional[darsia.Image]) -> None:
        """Auxiliary setup routine for setting up the custom color checker.

        Defines self.colorchecker.

        Args:
            base (Image, optional): reference image if provided

        """
        # Fetch type of colorchecker
        reference = self.config.get(
            "reference", "classic" if base is None else "custom"
        )
        assert reference in ["classic", "custom"]

        # Set up either classic or custom color checker
        if reference == "classic":

            # Choose a classic color checker
            self.colorchecker = ColorCheckerAfter2014()

        elif reference == "custom":

            # Allow to use provided image or some path to color patches. Prioritize image.
            if base is None:

                # Fetch path and flag on updating
                colorchecker_path: Path = Path(
                    self.config.get(
                        "custom_colorchecker_path", "custom_colorchecker.npy"
                    )
                )
                update_colorchecker: bool = self.config.get(
                    "custom_colorchecker_update", False
                )

                # If required, extract swatches from a provided baseline image and use these
                # as reference colors; otherwise fetch info form file.
                if update_colorchecker or not colorchecker_path.exists():

                    # Fetch info on baseline image and roi
                    baseline_path: str = self.config.get("baseline")
                    baseline = cv2.cvtColor(
                        cv2.imread(str(Path(baseline_path)), cv2.IMREAD_UNCHANGED),
                        cv2.COLOR_BGR2RGB,
                    )
                    baseline_roi: np.ndarray = np.array(
                        self.config.get("custom_colorchecker_reference_roi", self.roi)
                    )

                    # Extract part of the image containing a color checker.
                    # Use width and height (in cm - irrelevant) as provided by the
                    # manufacturer Xrite.
                    assert self.roi.shape == (4, 2)
                    colorchecker_img = darsia.extract_quadrilateral_ROI(
                        baseline, pts_src=baseline_roi, width=27.3, height=17.8
                    )

                    # Determine swatch colors
                    swatches = self._detect_colour_checkers_segmentation(
                        colorchecker_img
                    )

                    # Scale to interval [0,1], to not loose any information, perform the
                    # scaling by hand
                    assert baseline.dtype in [np.uint8, np.uint16]
                    scaling = 255.0 if baseline.dtype == np.uint8 else 255.0**2
                    swatches /= scaling

                    self.colorchecker = CustomColorChecker(colorchecker_path, swatches)

                else:
                    self.colorchecker = CustomColorChecker(colorchecker_path)

            else:
                # Extract part of the image containing a color checker.
                # Use width and height (in cm - irrelevant) as provided by the
                # manufacturer Xrite.
                assert self.roi.shape == (4, 2)
                colorchecker_img = darsia.extract_quadrilateral_ROI(
                    base.img, pts_src=self.roi, width=27.3, height=17.8
                )

                # Determine swatch colors
                swatches = self._detect_colour_checkers_segmentation(colorchecker_img)

                # Scale to interval [0,1], to not loose any information, perform the
                # scaling by hand
                assert base.img.dtype in [np.uint8, np.uint16]
                scaling = 255.0 if base.img.dtype == np.uint8 else 255.0**2
                swatches /= scaling

                self.colorchecker = CustomColorChecker(reference_colors=swatches)

    def _restrict_to_roi(self, img: np.ndarray) -> np.ndarray:
        """
        Auxiliary function to extract roi from image.

        Args:
            img (np.ndarray): input image

        Returns:
            np.ndarray: restricted image
        """
        if self.roi is None:
            return_img: np.ndarray = img
        elif isinstance(self.roi, tuple):
            return_img = img[self.roi]
        elif isinstance(self.roi, np.ndarray):
            assert self.roi.shape == (4, 2)
            # Use width and height (in cm - irrelevant) as provided by the manufacturer Xrite.
            return_img = darsia.extract_quadrilateral_ROI(
                img, pts_src=self.roi, width=27.3, height=17.8
            )

        # For debugging puposes provide the possibility to plot the extracted image.
        if self.verbosity:
            plt.figure()
            plt.imshow(return_img)
            plt.show()

        return return_img

    def _detect_colour_checkers_segmentation(self, img: np.ndarray) -> np.ndarray:
        """
        ML-free variant of detect_colour_checkers_segmentation from colour-science.
        Exepcts images to be restricted to the ROI such that the landmarks in the
        corners are also the corners of the image. Does

        Args:
            img: image of color checker.

        Returns:
            np.ndarray: 4 x 6 array with colour checker colors.
        """
        # Resize to fixed size
        Ny, Nx = img.shape[:2]
        fixed_width = 500
        fixed_height = int(Ny / Nx * fixed_width)
        resized_img = cv2.resize(img, (fixed_width, fixed_height))

        # Upper left corners of swatches
        swatch_pos_X, swatch_pos_Y = np.meshgrid(
            [12, 95, 177, 260, 344, 427],
            [12, 93, 175, 255],
        )
        swatch_size = 50

        # Extract colors of all swatches in RGB by taking averages
        swatches = np.zeros((4, 6, 3), dtype=np.float32)
        for row in range(4):
            for col in range(6):
                pos_x = swatch_pos_X[row, col]
                pos_y = swatch_pos_Y[row, col]
                roi = (
                    slice(pos_y, pos_y + swatch_size),
                    slice(pos_x, pos_x + swatch_size),
                )
                swatches[row, col] = (
                    np.sum(np.sum(resized_img[roi], axis=0), axis=0) / swatch_size**2
                )

        # For debugging puposes provide the possibility to plot the extracted swatches
        # compared to the reference swatches.
        if self.verbosity:
            # Plot the extracted swatch colors
            plt.figure("Registered swatches")
            plt.imshow(swatches)

            # Plot the reference swatches in the order and form of the classic color checker.
            ref_swatches = self.colorchecker.reference_swatches_rgb[
                :, np.newaxis, :
            ].reshape((4, 6, 3), order="F")
            plt.figure("Reference swatches")
            plt.imshow(ref_swatches)

            plt.show()

        # Reshape to same format as reference swatches, i.e., (24,3) format, column by column.
        swatches = np.squeeze(swatches.reshape((24, 1, 3), order="F"))

        return swatches
