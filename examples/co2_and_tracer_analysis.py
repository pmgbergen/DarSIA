"""
Example script for simple image analysis. By comparison of images
of the same well test, a tracer concentration can be determined.
"""

import json
import os

import cv2
import numpy as np

import darsia

# Define path to image folder
image_folder = f"{os.path.dirname(__file__)}/images/"

# Define curvature correction object, initiated with config file
# (which can be created the workflow presented in the Jupyter notebook
# examples/notebooks/curvature_correction_walkthrough.ipynb).
with open(image_folder + "config.json", "r") as openfile:
    config = json.load(openfile)
curvature_correction = darsia.CurvatureCorrection(config=config["curvature"])

# Define color correction object
config = {
    "roi": darsia.make_voxel(
        [
            [152, 202],
            [225, 206],
            [226, 101],
            [153, 98],
        ]
    )
}
color_correction = darsia.ColorCorrection(config=config)

# !----- Main routine for co2 analysis

# Read baseline and co2 image and correct color and curvature
base_array = cv2.cvtColor(cv2.imread(image_folder + "co2_0.jpg"), cv2.COLOR_BGR2RGB)
co2_array = cv2.cvtColor(cv2.imread(image_folder + "co2_2.jpg"), cv2.COLOR_BGR2RGB)
baseline_co2 = darsia.Image(
    base_array,
    transformations=[color_correction, curvature_correction],
    width=2.8,
    height=1.5,
)
co2_image = darsia.Image(
    co2_array,
    transformations=[color_correction, curvature_correction],
    width=2.8,
    height=1.5,
)

# Construct concentration analysis for detecting the co2 concentration
co2_analysis = darsia.ConcentrationAnalysis(
    base=baseline_co2,  # baseline image
    signal_reduction=darsia.MonochromaticReduction(color="red"),  # signal reduction
    restoration=darsia.TVD(),  # restoration
    model=darsia.CombinedModel(  # signal to data conversion
        [
            darsia.LinearModel(scaling=4.0),
            darsia.ClipModel(**{"min value": 0.0, "max value": 1.0}),
        ]
    ),
)

# Given a series of baseline images one can setup a cleaning mask,
# which is used as lower threshold in the concentration analysis.
# The reason why this is useful is that illumination often is not
# constant over time. Thus local noise effectes are created just
# because of fluctuating light. Therefore, local thresholding is
# applied to remove these local noise. Code for such use has the
# form:
# co2_analysis.find_cleaning_filter(baseline_images)
# NOTE: That such illumination effects can be observed at the
# color checker and across the entire middle layer.

# Given a series of images, one can calibrate a concentration analysis
# object assuming a constant growth of the effective (injected) volume
# over time. For this code as the following has to be used:
# co2_analysis.calibrate(injection_rate, images, [lower_initial_guess, upper_inital_guess])
# The initial guesses are initial guesses for the internal scaling factor
# for converting signals to concentrations. Here, we simply set a scaling
# factor from outside without calibration.

# Determine co2
co2 = co2_analysis(co2_image)

# Plot change 10 to larger number (or remove it) if it is desired to
# keep the images longer on the screen
co2.show("co2", 10)

# !----- Main routine for tracer analysis

# Read in baseline and tracer image and correct color and curvature
base_tracer_array = cv2.cvtColor(
    cv2.imread(image_folder + "tracer_0.jpg"), cv2.COLOR_BGR2RGB
)
tracer_array = cv2.cvtColor(
    cv2.imread(image_folder + "tracer_3.jpg"), cv2.COLOR_BGR2RGB
)
baseline_tracer = darsia.Image(
    base_tracer_array,
    transformations=[color_correction, curvature_correction],
    width=2.8,
    height=1.5,
)
tracer_image = darsia.Image(
    tracer_array,
    transformations=[color_correction, curvature_correction],
    width=2.8,
    height=1.5,
)

# Define restoration routine


# Construct concentration analysis for detecting the tracer concentration
tracer_analysis = darsia.ConcentrationAnalysis(
    base=baseline_tracer,
    signal_reduction=darsia.MonochromaticReduction(color="gray"),
    restoration=darsia.TVD(),
    model=darsia.CombinedModel(
        [darsia.LinearModel(), darsia.ClipModel(**{"min value": 0.0, "max value": 1.0})]
    ),
)

# Determine tracer
tracer = tracer_analysis(tracer_image)

# Plot
tracer.show("tracer", 10)
