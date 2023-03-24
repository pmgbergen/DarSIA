"""Example showcasing DarSIA for photographs.
Perform curvature correction as in the unit tests.

"""

import json
import os

import cv2

import darsia

# ! ---- Define image array in RGB format
path = f"{os.path.dirname(__file__)}/images/co2_2.jpg"
array = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

# ! ---- Define physical specs
metadata = {
    "dimensions": [1.5, 2.8],
    "origin": [0.0, 1.5],
}

# ! ---- Setup correction

# Fetch config file, holding info to several correction routines.
config_path = f"{os.path.dirname(__file__)}/images/config.json"
with open(config_path, "r") as openfile:
    config = json.load(openfile)

# Define curvature correction object, initiated with config file
curvature_correction = darsia.CurvatureCorrection(config=config["curvature"])

# ! ---- Define corrected image

corrected_image = darsia.OpticalImage(
    img=array, transformations=[curvature_correction], **metadata
)
corrected_image.show("corrected image", 5)

# ! ---- Demonstrate effect of curvature correction

original_image = darsia.OpticalImage(array, **metadata)
original_image_with_grid = original_image.add_grid(origin=[0.0, 0.0], dx=0.1, dy=0.1)
original_image_with_grid.show("original image with grid", 5)

corrected_image_with_grid = darsia.OpticalImage(
    img=original_image_with_grid.img, transformations=[curvature_correction], **metadata
)
corrected_image_with_grid.show("corrected image with deformed grid", 5)

# ! ---- Convert color space

hsv_image: darsia.OpticalImage = corrected_image.to_trichromatic(
    "hsv", return_image=True
)
hsv_image.show("hsv", 5)

corrected_image.to_trichromatic("bgr")
corrected_image.show("bgr", 5)

red_image: darsia.ScalarImage = corrected_image.to_monochromatic("red")
red_image.show("red", 5)

# ! ---- Test data types
print(
    f"The corrected image with deformed grid has the type {type(corrected_image_with_grid)}."
)
print(f"The hsv image has the type {type(hsv_image)}.")
print(f"The red image has the type {type(red_image)}.")
