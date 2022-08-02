import colour
import numpy as np
from colour_checker_detection import detect_colour_checkers_segmentation
from daria.corrections.color.transferfunctions import EOTF

class ClassicColorChecker:
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
            colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'],
            colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB
        )

class ColorCorrection:

    def __init__(self):
        """
        Constructor of converter, setting up a priori all data needed for fast conversion.

        Attributes:
            eotf: LUTs for standard electro-optical transfer function
        """

        # Define look up tables approximating the standard electro-optical transfer function for sRGB.
        self.eotf = EOTF()

        # Reference of the class color checker
        self.ccc = ClassicColorChecker()

    def adjust(self, image: np.ndarray, roi_cc: tuple = None, verbosity: bool = False) -> np.ndarray:
        """
        Apply workflow from colour-science to match the colors of the color checker with the corresponding
        color values, cf. https://github.com/colour-science/colour-checker-detection/blob/master/colour_checker_detection/examples/examples_detection.ipynb

        Arg#uments:
           # image (np.ndarray): image with uint8 value in (linear) RGB color space
           # roi_cc (tupe of slices): region of interest containing a colour checker
           # verbosity (bool): displays corrected color checker on top of the reference one if True, default is False

        Returns:
            np.ndarray: image with uint8 values in (linear) RGB color space, with colors
                matched based on the color checker within the roi
        """
        # Apply transfer function and convert to nonlinear RGB color space
        decoded_image = self.eotf.adjust(image)

        # Extract part of the image containing a color checker.
        colorchecker_image = decoded_image[roi_cc[0], roi_cc[1], :] if roi_cc is not None else decoded_image

        # Retrieve swatch colors in transfered RGB format
        swatches = detect_colour_checkers_segmentation(colorchecker_image)
        if len(swatches) == 0:
            raise ValueError("Color checker not identified. Choose a better ROI.")
        else:
            swatches = swatches[0]

        # For debugging purposes (a pre/post analysis), the swatches are displayed on top of the reference color checker.
        if verbosity==True:

            # Standard D65 illuminant
            D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']

            # Convert swatches from RGB to xyY
            swatches_xyY = colour.XYZ_to_xyY(colour.RGB_to_XYZ(swatches, D65, D65, colour.RGB_COLOURSPACES['sRGB'].matrix_RGB_to_XYZ))

            # Color correct swatches and also convert to xyY
            corrected_swatches = colour.colour_correction(
                swatches, swatches, self.ccc.reference_swatches
            )
            corrected_swatches_xyY = colour.XYZ_to_xyY(colour.RGB_to_XYZ(corrected_swatches, D65, D65, colour.RGB_COLOURSPACES['sRGB'].matrix_RGB_to_XYZ))

            # Define color checkers using the swatches pre and post color correction
            colour_checker_pre = colour.characterisation.ColourChecker(
                "pre",
                dict(zip(self.ccc.color_names, swatches_xyY)),
                D65)

            colour_checker_post = colour.characterisation.ColourChecker(
                "post",
                dict(zip(self.ccc.color_names, corrected_swatches_xyY)),
                D65)

            # Plot the constructed color checkers on top of the reference classic color checker.
            #colour.plotting.plot_multi_colour_checkers(
            #    [self.ccc.colorchecker, colour_checker_pre])

            colour.plotting.plot_multi_colour_checkers(
                [self.ccc.colorchecker, colour_checker_post])

        # Apply color correction onto full image based on the swatch colors in comparison with the standard colors
        corrected_decoded_image = colour.colour_correction(
            decoded_image,
            swatches,
            self.ccc.reference_swatches,
        )

        # Convert to linear RGB by applying the inverse of the EOTF and return the corrected image
        return self.eotf.inverse_approx(corrected_decoded_image)
