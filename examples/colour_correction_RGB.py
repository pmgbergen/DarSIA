import time

import colour
import cv2
import daria as da
import numpy as np
from colour_checker_detection import detect_colour_checkers_segmentation

# TODO 3: measure timing of all relevant components. what is the bottle neck and how bad is it?
# Result: The bottleneck is the back and forth transformation between linear and nonlinear RGB formats.

# TODO 5: what if we do this on the lin RGB scale? and not convert to nonlinear RGB (float)...
# Result: This does not work satisfactorily. The final signal seems deterioarted (outside the interval?)
# and there is a substantial difference to the 'correct' way of doing it.

# NOTE: For different colour spaces and gamma-correction, cf. https://en.wikipedia.org/wiki/CIE_1931_color_space
# Another nice introductory web page: https://ninedegreesbelow.com/photography/xyz-rgb.html

# TODO 6: Is it possible to identify the white tone of the lamps used to illuminate the FluidFlower? D50, D65?
# OpenCV should work with D65.

# ------- START of script - begin with defining the classic colour checker

# Define Classic Colourchecker
cc = da.ClassicColorChecker()

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

# -------- Convert the full picture into various colorspaces
# Fetch image, in BGR
img_BGR = img_baseline.img

# Convert to RGB (int)
img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

cv2.imwrite("colour-correction/baseline.jpg", img_BGR)
cv2.namedWindow("original baseline", cv2.WINDOW_NORMAL)
cv2.imshow("original baseline", img_BGR)
cv2.waitKey(0)

# -------- Restrict image to a ROI closer to the ColourChecker

roi_cc = (slice(100, 600), slice(350, 600))

# -------- Match colors and apply correction to original picture using the nonlinear RGB colour space

st = time.time()

# Apply colour correction based on color checker
colorcorrection = da.corrections.color.colorchecker.ColorCorrection()
corrected_baseline_RGB = colorcorrection.adjust(img_RGB, roi_cc)

end = time.time()
print("match colors", end - st)

# Convert to BGR - for plotting
corrected_baseline_BGR = cv2.cvtColor(corrected_baseline_RGB, cv2.COLOR_RGB2BGR)
cv2.imwrite("colour-correction/baseline_corrected.jpg", corrected_baseline_BGR)
cv2.namedWindow("corrected baseline", cv2.WINDOW_NORMAL)
cv2.imshow("corrected baseline", corrected_baseline_BGR)
cv2.waitKey(0)

# -------- Simple colour correction based on linear RGB
# TODO review and remove as most likely not relevant anymore - need to find a cheaper way to transform

# -------- Define conversions between linear and nonlinear RGB

def linRGB_to_nlinRGB(img_lin_RGB: np.ndarray) -> np.ndarray:
    return colour.cctf_decoding(img_lin_RGB / 255.0)

def nlinRGB_to_linRGB(img_gamma_RGB: np.ndarray) -> np.ndarray:
    # Convert to linear RGB (float)
    img_lin_RGB = colour.cctf_encoding(img_gamma_RGB)

    # Convert to RGB (uint)
    img_RGB = (255 * img_lin_RGB).astype(np.uint8)

    return img_RGB

# -------- Define color corrections (simple and correct one)


def colour_correction_linRGB(img_lin_RGB: np.ndarray, roi_cc) -> np.ndarray:

    # Apply ROI to img, containing the ColourChecker
    img_cc_lin_RGB = img_lin_RGB[roi_cc[0], roi_cc[1], :]

    # Convert to nonlinear RGB space
    img_cc_nlin_RGB = linRGB_to_nlinRGB(img_cc_lin_RGB)

    # Retrieve swatch colors
    swatch_colors_gamma_RGB = detect_colour_checkers_segmentation(img_cc_nlin_RGB)[0]

    # Convert to linear RGB space
    swatch_colors_RGB = nlinRGB_to_linRGB(swatch_colors_gamma_RGB)

    # Apply correction onto full image
    corrected_img_lin_RGB = (
        colour.colour_correction(img_lin_RGB, swatch_colors_RGB, cc.colors_RGB)
    ).astype(np.uint8)

    return corrected_img_lin_RGB

st = time.time()

# Apply correction, and make sure that the result again has uint8 values
tst_corrected_baseline_RGB = colour_correction_linRGB(img_RGB, roi_cc)

end = time.time()
print("match colors simple", end - st)

# Convert to BGR - for plotting
tst_corrected_baseline_BGR = cv2.cvtColor(tst_corrected_baseline_RGB, cv2.COLOR_RGB2BGR)
cv2.imwrite(
    "colour-correction/baseline_fast_correction.jpg", tst_corrected_baseline_BGR
)
cv2.namedWindow("tst corrected baseline", cv2.WINDOW_NORMAL)
cv2.imshow("tst corrected baseline", tst_corrected_baseline_BGR)
cv2.waitKey(0)
