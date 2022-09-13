"""
Quick 5-step procedure to a decent config file for the correction of images of
the large FluidFlower.

The preprocessing of FluidFlower images requires a config file. It includes
several tuning parameters, which need to be provided. In order to reduce manual
tuning, this run script aims at a brief procedure describing how to choose them
in 5 steps.

This routine is not entirely free from tuning. However, tedious fine tuning is
not necessary anymore.
"""

# TODO transform this to a jupyter notebook

from pathlib import Path

import cv2
import matplotlib.pyplot as plt

import daria as da
from daria.corrections.shape.curvature import simple_curvature_correction

# !----- 1. Step: Read curved image and initialize the config file

# NOTE: User input needed.

# Choose a image of your choice.
image_name = "./baseline_with_laser_grid.jpg"
full_path = Path(image_name)

# Read image
img = cv2.imread(str(full_path))

# Convert to RGB space
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot for convenience
plt.imshow(img)
plt.show()

# All relevant config parameters will be stored in a dictionary collecting several configs.
# Initialize the config dict.
config: dict() = {}

# !----- 2. Step: Bulge

# NOTE: User input needed.

# Choose horizontal and vertical bulge such that all laser grid lines are bulged inwards.
# In some cases it might be necessary to define offsets for the image center;
# the default is to use the numerical center.
config["init"] = {
    "horizontal_bulge": 5e-9,
    # "vertical_bulge": 0,
    # "horizontal_center_offset": 0,
    # "vertical_center_offset": 0,
}

# Apply bulge 'correction'
img = simple_curvature_correction(img, **config["init"])

# !----- 3. Step: Bulge

# Read coordinates of 4 points, defining a rectangular of known dimensions.
# Here, we choose a bounding box with corners on the laser grid.
plt.imshow(img)
plt.show()

# NOTE: User input needed.

# Define config file for applying a homography, which after all transforms
# and crops the image to a box with known aspect ratio. This step already
# corrects for bulging in the normal direction due to the curvature of the
# FluidFlower.
config["crop"] = {
    "pts_src": [
        [11, 8],
        [16, 1755],
        [3165, 1748],
        [3165, 5],
    ],
    # Specify the true dimensions of the reference points - known as they are
    # points on the laser grid
    "width": 2.8,
    "height": 1.5,
    "in meters": True,
}

# Crop by applying a homography
img = da.homography_correction(img, **config["crop"])

# !----- 3. Step: Straighten the laser grid lines by correcting for bulge

# Plot...
plt.imshow(img)
plt.show()

# ... and determine the parameters as described in the daria-notes
# For this, require the dimensions of the image
Ny, Nx = img.shape[:2]

# NOTE: User input needed.

# Read the x-coordinates of the two largest impressions in x-direction
x1 = 1e-6
x2 = Nx - 1e-6
y1 = 53
y2 = 1646

# Determine the associated increments
dx1 = x1
dx2 = Nx - x2
dy1 = y1
dy2 = Ny - y2

# Determine the center of the image as described in the daria-notes
image_center = [int(Nx * dx1 / (dx1 + dx2)), int(Ny * dy1 / (dy1 + dy2))]

# Determine the offset of the numerical center of the image
horizontal_bulge_center_offset = image_center[0] - int(Nx / 2)
vertical_bulge_center_offset = image_center[1] - int(Ny / 2)

# Determine the bulge tuning coefficients as explained in the daria notes
# Assume here that the maximum impressions are applied at the image center
# (more accurate values can be read from the image)
horizontal_bulge = dx1 / (
    (x1 - image_center[0]) * image_center[1] * (Ny - image_center[1])
)
vertical_bulge = dy1 / (
    (y1 - image_center[1]) * image_center[0] * (Nx - image_center[0])
)

# Choose horizontal and vertical bulge such that all laser grid lines are bulged inwards
config["bulge"] = {
    "horizontal_bulge": horizontal_bulge,
    "horizontal_center_offset": horizontal_bulge_center_offset,
    "vertical_bulge": vertical_bulge,
    "vertical_center_offset": vertical_bulge_center_offset,
}

# Apply final curvature correction
img = simple_curvature_correction(img, **config["bulge"])

# !----- 4. Step: Correct for stretch

# Compare with a 'perfect' grid layed on top
# Determine coordinates of some point which is off
da_img = da.Image(img, width=2.8, height=1.5).add_grid(dx=0.1, dy=0.1)
plt.imshow(da_img.img)
plt.show()

# NOTE: User input needed.

# Use [x,y] in pixels, and specify the current location, and the ought to be location
pt_src = [585, 676]
pt_dst = [567, 676]

# Define 'center' as the point which can be trusted the most - can be omitted if a
# second reference point is considered # FIXME
stretch_center = [1476, 1020]

# Update the offset to the center
horizontal_stretch_center_offset = stretch_center[0] - int(Nx / 2)
vertical_stretch_center_offset = stretch_center[1] - int(Ny / 2)

# Compute the tuning parameter as explained in the notes
# TODO add to the notes
horizontal_stretch = -(pt_dst[0] - pt_src[0]) / (
    (pt_src[0] - stretch_center[0]) * pt_src[0] * (Nx - pt_src[0])
)
vertical_stretch = -(pt_dst[1] - pt_src[1]) / (
    (pt_src[1] - stretch_center[1]) * pt_src[1] * (Ny - pt_src[1])
)

# Choose horizontal and vertical bulge such that all laser grid lines are bulged inwards
config["stretch"] = {
    "horizontal_stretch": horizontal_stretch,
    "horizontal_center_offset": horizontal_stretch_center_offset,
    "vertical_stretch": vertical_stretch,
    "vertical_center_offset": vertical_stretch_center_offset,
}

# Apply final curvature correction
img = simple_curvature_correction(img, **config["stretch"])

# !----- 6. Step: Validation - Compare with a 'perfect' grid layed on top
da_img = da.Image(img, width=2.8, height=1.5).add_grid(dx=0.1, dy=0.1)
plt.imshow(da_img.img)
plt.show()

# !----- 7. Step: Color correction

# Need to define a coarse ROI which contains the color checker - use [y,x] pixel ordering
config["color"] = {
    "roi": (slice(0, 600), slice(0, 600)),
}

# !----- Summary of the config - copy and move to another file.
print(config)

# This config file can now be used to run the predefined correction routine
# for any image of a large FluidFlower.
#
# Example:
# TODO - not implemented yet - have to decide whether to use daria images here or not.
# fluidflower = FluidFlower(config)
# corrected_img = fluidflower.preprocessing(img)
