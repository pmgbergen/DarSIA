"""
Example script for simple image analysis. By comparison of images
of the same well test, a tracer concentration can be determined.
"""

import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import skimage
import os

import daria


def determine_tracer(img: daria.Image, base: daria.Image) -> daria.Image:
    """Extract tracer based on a reference image.

    Args:
        img (np.ndarray):  probe image array
        base (np.ndarray): baseline image array

    Returns:
        np.ndarray: concentration map
    """
    # Transform images to grayscale
    img = img.toGray()
    base = base.toGray()

    # Take (unsigned) difference
    diff = skimage.util.compare_images(img.img, base.img, method="diff")

    # Apply smoothing filter
    diff = skimage.filters.rank.median(diff, skimage.morphology.disk(20))

    # The next step is to translate the color intensity to the actual concentration.
    # Such post-processing is very much scenario dependent and require further knowledge
    # of the physical process, or injection schedule etc. Here, an arbitrary thresholding
    # routine is applied as an example. More advanced postprocessing routines are also
    # possible,

    # Apply thresholding to cut off noise
    thresh_min = 5
    thresh_max = 40
    tracer_min_mask = diff <= thresh_min
    diff[tracer_min_mask] = 0
    diff[~tracer_min_mask] -= thresh_min
    tracer_max_mask = diff >= thresh_max - thresh_min
    diff[tracer_max_mask] = thresh_max - thresh_min

    # Transform to float data type for simpler conversion to tracer concentration
    diff = skimage.util.img_as_float(diff)

    # Rescale image to physically meaningful range [0,1]
    diff = skimage.exposure.rescale_intensity(diff)

    return daria.Image(diff, base.metadata)


# Define curvature correction object, initiated with config file
# (which can be created the workflow presented in the Jupyter notebook
# examples/notebooks/curvature_correction_walkthrough.ipynb).
curvature_correction = daria.CurvatureCorrection(
    config_source=f"{os.path.dirname(__file__)}/images/config.json", width=2.8, height=1.5
)

# Define color correction object
roi_cc = (slice(0, 600), slice(0, 700))
color_correction = daria.ColorCorrection(ROI = roi_cc)

# !----- Main routine for co2 analysis

# Read baseline and co2 image and correct color and curvature
baseline_co2 = daria.Image(f"{os.path.dirname(__file__)}/images/co2_0.jpg", curvature_correction=curvature_correction, color_correction= color_correction)
co2_image = daria.Image(f"{os.path.dirname(__file__)}/images/co2_2.jpg", curvature_correction=curvature_correction, color_correction= color_correction )

# Determine co2
co2 = determine_tracer(co2_image, baseline_co2)

# Plot change 3 to larger number (or remove it) if it is desired to keep the images on the screen
co2.plt_show(3)

# !----- Main routine for tracer analysis

# Read in baseline and tracer image and correct color and curvature
baseline_tracer = daria.Image(f"{os.path.dirname(__file__)}/images/tracer_0.jpg",  curvature_correction=curvature_correction, color_correction= color_correction)
tracer_image = daria.Image(f"{os.path.dirname(__file__)}/images/tracer_3.jpg",  curvature_correction=curvature_correction, color_correction= color_correction)

# Determine tracer
tracer = determine_tracer(tracer_image, baseline_tracer)

# Plot
tracer.plt_show(3)