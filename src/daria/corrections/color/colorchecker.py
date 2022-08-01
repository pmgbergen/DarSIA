import colour
import numpy as np

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

        # Reference colors in gamma-compressed RGB (float) format
        self.colors_gamma_RGB = colour.XYZ_to_RGB(
            XYZ=self.colors_XYZ,
            illuminant_XYZ=self.colorchecker.illuminant,
            illuminant_RGB=self.illuminant,
            matrix_XYZ_to_RGB=colour.RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB,
        )

        # Reference colors in linear RGB (int) format
        self.colors_RGB = (255 * colour.cctf_encoding(self.colors_gamma_RGB)).astype(
            np.uint8
        )
