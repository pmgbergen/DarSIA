"""Module containing the color correction based on the Classic
Color Checker from calibrite / x-rite. The algorithms here
heavily use code from colour-science (see their github).
This includes some ML-based algorithms. Sometimes, the algorithms
are not able to detect the color checker despite a rather precise
user-input on the location of the color checker. Or the swatches
are not extracted correcty. In such cases, daria.ColorCorrection
should be used which should always work. It is therefore recommended
to double-check the results of the any of the color correction
routines by setting verbosity to True once.
"""

from typing import Optional, Union, cast

import colour
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from colour_checker_detection import detect_colour_checkers_segmentation


class ClassicColorChecker:
    """Definition of the classic color checker under default
    illumination conditions.
    """

    def __init__(self):

        # Fetch reference colourchecker data (in chromaticity coordinates),
        # published by the manufacturer (X-Rite), latest data
        self.colorchecker = colour.CCS_COLOURCHECKERS[
            "ColorChecker24 - After November 2014"
        ]

        # Fetch standard colour names and reference colors in CIE xyY format
        self.color_names = self.colorchecker.data.keys()
        self.colors_xyY = list(self.colorchecker.data.values())

        # Swatches of the Classic Colour Checker in RGB format, using the CIE standards
        self.reference_swatches = colour.XYZ_to_RGB(
            colour.xyY_to_XYZ(self.colors_xyY),
            self.colorchecker.illuminant,
            colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"],
            colour.RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB,
        )


class EOTF:
    """
    Electro-optical transfer function (EOTF),the standard transfer
    function for sRGB, cf. https://en.wikipedia.org/wiki/SRGB
    """

    def __init__(self):
        """Define look up table (LUT), representing the EOTF."""

        # Use the pre-defined EOTF implemented by colour and apply to any possible
        # uint8 value.
        self.table_eotf = colour.cctf_decoding(np.arange(0, 256) / 255.0)

        # Define the inverse of EOTF, which acts on value in [0,1]. Here, the table
        # translates to uint8 values, i.e., for before applying values of images
        # with values in [0,1] have to be transformed accordingly to allow the application.
        self.table_eotf_inverse = (
            colour.cctf_encoding(np.arange(0, 256) / 255.0) * 255.0
        ).astype("uint8")

    def adjust(self, image: np.ndarray) -> np.ndarray:
        """Apply EOTF using the look up table.

        Arguments:
            image (np.ndarray): image in linear RGB (uint8) format.

        Returns:
            np.ndarray: image with tranformed color space, with values in [0,1].
        """
        return cv2.LUT(image, self.table_eotf)

    def inverse_approx(self, image: np.ndarray) -> np.ndarray:
        """Apply approximate of the inverse of EOTF using the look up table.
        In order to apply the exact inverse, the expensive function colout.cctf_encoding
        would have to be applied to all pixels (with values in [0,1]). By restricting the
        allowed input values to uint8 values, allows for faster LUT.

        Arguments:
            image (np.ndarray): image in gamma-corrected RGB format with values in [0,1].

        Returns:
            np.ndarray: image in linear RGB (uint8) format.
        """

        # Need to transform values to uint8 first before applying the LUT.
        return cv2.LUT(skimage.img_as_ubyte(image), self.table_eotf_inverse)


class ExperimentalColorCorrection:
    def __init__(
        self,
        roi: Optional[Union[tuple, np.ndarray]] = None,
        verbosity: bool = False,
        whitebalancing: bool = True,
    ):
        """
        Constructor of converter, setting up a priori all data needed for fast conversion.

        Attributes:
            eotf: LUTs for standard electro-optical transfer function
            roi (tuple of slices, np.ndarray, or None): ROI containing a colour checker,
                provided either as intervals, corner points, or nothing.
            verbosity (bool): displays corrected color checker on top of the reference one if
                True, default is False
            whitebalancing (bool): apply white balancing based on the third bottom left swatch
                if True, default is True
        """

        # Define look up tables approximating the standard electro-optical
        # transfer function for sRGB.
        self.eotf = EOTF()

        # Reference of the class color checker
        self.colorchecker = ClassicColorChecker()

        # Define ROI
        self.roi = roi

        # Store flags
        self.verbosity = verbosity
        self.whitebalancing = whitebalancing

    def __call__(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply workflow from colour-science to match the colors of the color checker with the
        corresponding color values, cf.
        https://github.com/colour-science/colour-checker-detection/blob/master/colour_checker_detection/examples/examples_detection.ipynb

        Arguments:
            image (np.ndarray): image with uint8 value in (linear) RGB color space

        Returns:
            np.ndarray: corrected image with uint8 values in (linear) RGB color space,
                with colors matched based on the color checker within the roi
        """
        # Make sure that the image is in uint8 format
        if image.dtype != np.uint8:
            raise ValueError("Provide image in uint8 format.")

        # Apply transfer function and convert to nonlinear RGB color space
        decoded_image = self.eotf.adjust(image)

        # Extract part of the image containing a color checker.
        colorchecker_image: np.ndarray = self._restrict_to_roi(decoded_image)

        # Retrieve swatch colors in transfered RGB format
        swatches_tmp = detect_colour_checkers_segmentation(colorchecker_image)
        if len(swatches_tmp) == 0:
            raise ValueError(
                """Color checker not identified. Choose a better ROI.
                Or in the worst case, provide pixels with reverse matrix indexing,
                i.e. (col,row) format, for the corner landmarks of the colorchecker,
                starting in the left upper corner and continuing counter-clockwise."""
            )
        else:
            assert len(swatches_tmp) == 1
            swatches: np.ndarray = cast(np.ndarray, swatches_tmp[0])

        if self.verbosity:
            swatches_img = swatches[:, np.newaxis, :].reshape((4, 6, 3), order="C")
            plt.figure()
            plt.imshow(swatches_img)
            plt.show()

        # Apply color correction onto full image based on the swatch colors in comparison with
        # the standard colors
        corrected_decoded_image = colour.colour_correction(
            decoded_image,
            swatches,
            self.colorchecker.reference_swatches,
            method="Cheung 2004",
        )

        # Apply white balancing, such that the third bottom left swatch of the color checker
        # is exact
        if self.whitebalancing:
            corrected_colorchecker_image: np.ndarray = self._restrict_to_roi(
                corrected_decoded_image
            )
            swatches = cast(
                np.ndarray,
                detect_colour_checkers_segmentation(corrected_colorchecker_image)[0],
            )
            corrected_decoded_image *= (
                self.colorchecker.reference_swatches[-4] / swatches[-4]
            )

        # The correction may result in values outside the feasible range [0., 1.].
        # Thus, simply clip the values for consistency.
        corrected_decoded_image = np.clip(corrected_decoded_image, 0, 1)

        # For debugging purposes (a pre/post analysis), the swatches are displayed on top
        # of the reference color checker.
        if self.verbosity:

            # Standard D65 illuminant
            D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]

            # NOTE: Mainly for debugging purposes, the visualize of the
            # 'before' and 'after' of the color correction, is following.
            # The 'before' is commented out, but can be activated by uncommenting.

            # Convert swatches from RGB to xyY
            # swatches_xyY = colour.XYZ_to_xyY(
            #     colour.RGB_to_XYZ(
            #         swatches,
            #         D65,
            #         D65,
            #         colour.RGB_COLOURSPACES["sRGB"].matrix_RGB_to_XYZ,
            #     )
            # )

            # Color correct swatches and also convert to xyY
            corrected_swatches = colour.colour_correction(
                swatches, swatches, self.colorchecker.reference_swatches
            )
            if self.whitebalancing:
                corrected_swatches *= (
                    self.colorchecker.reference_swatches[-4] / corrected_swatches[-4]
                )
            corrected_swatches_xyY = colour.XYZ_to_xyY(
                colour.RGB_to_XYZ(
                    corrected_swatches,
                    D65,
                    D65,
                    colour.RGB_COLOURSPACES["sRGB"].matrix_RGB_to_XYZ,
                )
            )

            # Define color checkers using the swatches pre and post color correction
            # colour_checker_pre = colour.characterisation.ColourChecker(
            #     "pre", dict(zip(self.colorchecker.color_names, swatches_xyY)), D65
            # )

            colour_checker_post = colour.characterisation.ColourChecker(
                "post",
                dict(zip(self.colorchecker.color_names, corrected_swatches_xyY)),
                D65,
            )

            # Plot the constructed color checkers on top of the reference classic
            # color checker.
            # colour.plotting.plot_multi_colour_checkers(
            #    [self.colorchecker.colorchecker, colour_checker_pre])

            colour.plotting.plot_multi_colour_checkers(
                [self.colorchecker.colorchecker, colour_checker_post]
            )

        # Convert to linear RGB by applying the inverse of the EOTF and return the
        # corrected image
        return self.eotf.inverse_approx(corrected_decoded_image)

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
        else:
            raise ValueError("ROI not supported.")

        return return_img
