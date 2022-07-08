import cv2
import numpy as np
import colour
import daria as da
from colour_checker_detection import detect_colour_checkers_segmentation

# TODO 3: measure timing of all relevant components. what is the bottle neck and how bad is it?
# TODO 4: Implement as object in src/daria/correction/ccc.py
# TODO 5: what if we do this on the lin RGB scale? and not convert to nonlinear RGB (float)...

# ------- Colourchecker
# TODO move class to external file


class ClassicColourChecker:
    def __init__(self):
        # Fetch conventional illuminant, used to define the hardcoded sRGB values for the classic colourchecker
        self.illuminant = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
            "D65"
        ]

        # Fetch reference colourchecker data published by the manufacturer (X-Rite), latest data
        self.colourchecker = colour.CCS_COLOURCHECKERS[
            "ColorChecker24 - After November 2014"
        ]

        # Fetch standard colour names
        self.colour_names = self.colourchecker.data.keys()

        # Reference colours in xyY format
        self.colours_xyY = list(self.colourchecker.data.values())

        # Convert reference colours to XYZ
        self.colours_XYZ = colour.xyY_to_XYZ(self.colours_xyY)

        # Convert reference colours to (nonlinear) sRGB
        self.colours_nlin_RGB = colour.XYZ_to_RGB(
            XYZ=self.colours_XYZ,
            illuminant_XYZ=self.colourchecker.illuminant,
            illuminant_RGB=self.illuminant,
            matrix_XYZ_to_RGB=colour.RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB,
        )


# Define Classic Colourchecker
cc = ClassicColourChecker()

# ------- Read in baseline picture

# ... start with hardcoding the meta data - have to read a file of same size first
img_tst = cv2.imread("../images/fluidflower/Baseline.jpg")
dx = 2.8 / (7901 - 64)
dy = 1.3 / (4420 - 573)
num_pixels = np.array(img_tst.shape[:2])
height, width = np.array([dy, dx]) * num_pixels
true_origo = [4435, 64]  # y,x: in pixels
origo = [-true_origo[1] * dx, (true_origo[0] - num_pixels[0]) * dy]  # x,y: in meters

# ... create the Image
img_baseline = da.Image(
    img="../images/fluidflower/Baseline.jpg",
    origo=origo,
    height=height,
    width=width,
)

# -------- Convert the full picture into various colourspaces
# TODO use cv2.LUT for this?

# Fetch image, in BGR
img_BGR = img_baseline.img

# Convert to RGB (float)
img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB) / 255.0

# Convert to nonlinear RGB (float)
img_nlin_RGB = colour.cctf_decoding(img_RGB)

# -------- Restrict image to a ROI closer to the ColourChecker

# Hardcode retriction of image to a smaller picture containing the colourchecker card
img_cc_BGR = img_BGR[100:600, 350:600, :]

# NOTE da.Image uses BGR format with uint8 values
# Hardcode conversion: Convert to RGB colorspace width values in [0,1], instead of [0,255]
img_cc_RGB = cv2.cvtColor(img_cc_BGR, cv2.COLOR_BGR2RGB) / 255.0

# Conversion to nonlinear RGB ~ gamma correction with gamma approx. 2.2
img_cc_nlin_RGB = colour.cctf_decoding(img_cc_RGB)

# -------- Retrieve colours on the colourchecker card

# Get colours in RGB (float32)
swatch_colours_nlin_RGB = detect_colour_checkers_segmentation(img_cc_nlin_RGB)[0]
# TODO test whether swatch_colours empty

# -------- Match colours and apply correction to original picture

# Apply correction
corrected_baseline_nlin_RGB = colour.cctf_encoding(
    colour.colour_correction(img_nlin_RGB, swatch_colours_nlin_RGB, cc.colours_nlin_RGB)
)

# -------- Convert the corrected image to standard format again

# Convert from nonlinear RGB (float32) to linear RGB (unit8)
corrected_baseline_RGB = 255 * corrected_baseline_nlin_RGB
corrected_baseline_RGB = corrected_baseline_RGB.astype(np.uint8)

# Convert from RGB (uint8) to BGR (uint8)
corrected_baseline_BGR = cv2.cvtColor(corrected_baseline_RGB, cv2.COLOR_RGB2BGR)

# -------- Show both the original and corrected image

cv2.namedWindow("original baseline", cv2.WINDOW_NORMAL)
cv2.imshow("original baseline", img_BGR)
cv2.waitKey(0)

cv2.namedWindow("corrected baseline", cv2.WINDOW_NORMAL)
cv2.imshow("corrected baseline", corrected_baseline_BGR)
cv2.waitKey(0)
