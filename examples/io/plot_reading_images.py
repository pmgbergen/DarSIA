"""
Reading a sequence of optical images
====================================

Read several photographs into a single space-time :class:`~darsia.OpticalImage`,
applying a curvature correction while reading, then slice and recolour it.
"""

import json
import os

import darsia

folder = os.path.join("..", "images")
optical_paths = [os.path.join(folder, f"co2_{i}.jpg") for i in range(3)]
metadata = {"dimensions": [1.5, 2.8]}

# %%
# The ``config.json`` shipped with the examples holds parameters for several
# correction routines; here we build a curvature correction from it.

with open(os.path.join(folder, "config.json"), "r") as f:
    config = json.load(f)
curvature_correction = darsia.CurvatureCorrection(config=config["curvature"])

# %%
# A single corrected image.

single = darsia.imread(
    optical_paths[0], transformations=[curvature_correction], **metadata
)
single.show(f"single image - {single.time}")

# %%
# All three frames as one space-time image.

spacetime = darsia.imread_from_optical(
    optical_paths, transformations=[curvature_correction], **metadata
)
print("space-time shape:", spacetime.img.shape)

for time_index in range(spacetime.time_num):
    frame = spacetime.time_slice(time_index)
    frame.show(f"frame {time_index}")

# %%
# Recolour the whole space-time image, and reduce it to a single channel.

spacetime.to_trichromatic("hsv")
spacetime.show("space-time, HSV")

red = spacetime.to_monochromatic("red")
red.show("space-time, red channel")
