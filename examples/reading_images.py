"""Example demonstrating the I/O capabilities of
DarSIA, and in particular the reading functionality
to read multiple images from file, here demonstrated
for optical images. Extensions to DICOM and vtu files
exists as well.

"""


import json
import os

import darsia

# ! ---- Optical images

folder = f"{os.path.dirname(__file__)}/images"
optical_paths = [
    folder + "/" + "co2_0.jpg",
    folder + "/" + "co2_1.jpg",
    folder + "/" + "co2_2.jpg",
]

# ! ---- General metadata

optical_metadata = {
    "dimensions": [1.5, 2.8],
}

# ! ---- Corrections

# Fetch config file, holding info to several correction routines.
config_path = f"{os.path.dirname(__file__)}/images/config.json"
with open(config_path, "r") as openfile:
    config = json.load(openfile)
curvature_correction = darsia.CurvatureCorrection(config=config["curvature"])

# ! ---- Read images into DarSIA

# Single image
single_optical_image = darsia.imread(
    optical_paths[0], transformations=[curvature_correction], **optical_metadata
)
single_optical_image.show(
    f"single image - {single_optical_image.time} - {single_optical_image.date}", 5
)

# Space-time image
spacetime_optical_image = darsia.imread_from_optical(
    optical_paths,
    transformations=[curvature_correction],
    **optical_metadata,
)
print(f"Shape of the space-time image: {spacetime_optical_image.img.shape}.")

# Extract single slices of space-time image and plot these
for time_index in range(spacetime_optical_image.time_num):
    spacetime_slice: darsia.OpticalImage = spacetime_optical_image.time_slice(
        time_index
    )
    spacetime_slice.show(
        f"slice {time_index} in space time image - {spacetime_slice.time} - {spacetime_slice.date}",
        5,
    )

# Change color for an entire image.
spacetime_optical_image.to_trichromatic("hsv")
spacetime_optical_image.show("space time, hsv", 5)

# Extract space time scalar image
spacetime_red_image = spacetime_optical_image.to_monochromatic("red")
spacetime_red_image.show("space time, red", 5)
