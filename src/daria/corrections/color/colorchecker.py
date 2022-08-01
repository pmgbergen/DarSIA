import colour
import numpy as np
from colour_checker_detection import detect_colour_checkers_segmentation
from daria.corrections.color.transferfunctions import EOTF

class ClassicColorChecker:
    def __init__(self):
        # Fetch conventional illuminant, used to define the hardcoded sRGB values for the classic colourchecker
        self.illuminant = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
            "D65"
        ]

        # Fetch reference colourchecker data published by the manufacturer (X-Rite), latest data
        self.colorchecker = colour.CCS_COLOURCHECKERS[
            "ColorChecker24 - After November 2014"
        ]

        # Fetch standard colour names and reference colors in CIE xyY format
        self.color_names = self.colorchecker.data.keys()
        self.colors_xyY = list(self.colorchecker.data.values())

        # Reference colors in XYZ format / CIE 1973 colour space
        self.colors_XYZ = colour.xyY_to_XYZ(self.colors_xyY)

        # Reference colors in EOTF transformed sRGB (float) format
        self.colors_eotf_RGB = (
            colour.XYZ_to_sRGB(
                XYZ = self.colors_XYZ,
                illuminant = self.colorchecker.illuminant,
                apply_cctf_encoding = False,
            )
        )

        # Reference colors in linear sRGB (int) format
        self.colors_RGB = (
            colour.XYZ_to_sRGB(
                XYZ = self.colors_XYZ,
                illuminant = self.colorchecker.illuminant,
                apply_cctf_encoding = True,
            )
            * 255
        ).astype("uint8")

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

    def adjust(self, image: np.ndarray, roi_cc) -> np.ndarray:
        """
        Apply workflow from colour-science to match the colors of the color checker with the corresponding
        color values, cf.

        Arguments:
            image (np.ndarray): image with uint8 value in (linear) RGB color space
            roi_cc: region of interest containing a colour checker

        Returns:
            np.ndarray: image with uint8 values in (linear) RGB color space, with colors
                matched based on the color checker within the roi
        """
        # Apply transfer function and convert to nonlinear RGB color space
        img_eotf_RGB = self.eotf.adjust(image)

        # Extract part of the image containing a color checker.
        img_cc = img_eotf_RGB[roi_cc[0], roi_cc[1], :]

        # Retrieve swatch colors
        swatch_colors_eotf_RGB = detect_colour_checkers_segmentation(img_cc)[0]

        # TODO check whether empty and throw error if so

        # Apply color correction onto full image based on the swatch colors in comparison with the standard colors
        corrected_img_eotf_RGB = colour.colour_correction(
            img_eotf_RGB, swatch_colors_eotf_RGB, self.ccc.colors_eotf_RGB
        )

        # Convert to linear RGB by applying the inverse of the EOTF and return the corrected image
        return self.eotf.inverse_approx(corrected_img_eotf_RGB)
