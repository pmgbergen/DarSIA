"""
Module providing structures for common analyses. In practice, such may
have to be tailored to the specific scenario. Yet, they already provide
many of the most relevant functionalities. If not applicable, they also
provide the approach for how to set up tailored analysis classes.
"""

import json
import time
from pathlib import Path
from typing import Union, cast

import cv2
import numpy as np
import skimage

import darsia


class AnalysisBase:
    """
    Standard setup for an image analysis, in particular useful when analyzing
    a larger set of images in a time series.
    """

    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        update_setup: bool = False,
    ) -> None:
        """
        Constructor for GeneralAnalysis.

        Sets up fixed config file required for preprocessing.

        Args:
            baseline (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config (str or Path): path to config dict
            update_setup (bool): flag controlling whether cache in setup
                routines is emptied.
        """

        # ! ---- Config

        f = open(config, "r")
        self.config = json.load(f)
        """Config dict from file."""
        f.close()

        # ! ---- General specs

        # Deduct specs from the transformations
        if (
            "physical_asset" in self.config
            and "dimensions" in self.config["physical_asset"]
        ):
            self.width = self.config["physical_asset"]["dimensions"]["width"]
            """Physical width of image."""

            self.height = self.config["physical_asset"]["dimensions"]["height"]
            """Physical height of image."""

            self.origin = [0.0, self.height]
            """Physical origin of origin voxel."""

        else:
            raise ValueError("Dimensions and origin not specified.")

        # ! ---- Reference to baseline images

        if not isinstance(baseline, list):
            reference_base = cast(Union[str, Path], baseline)

        else:
            reference_base = cast(Union[str, Path], baseline[0])
        self.processed_baseline_images = None
        """List of corrected baseline images."""

        # ! ---- Define absolute correction objects

        # Define correction objects
        self.translation_correction = (
            darsia.TranslationCorrection(translation=self.config["translation"])
            if "translation" in self.config
            else None
        )
        """Translation correction using some absolute translation."""

        # Define color correction and provide baseline to colorchecker
        self.color_correction = None
        """Color correction."""

        if "color" in self.config:
            if "baseline" not in self.config["color"]:
                self.config["color"]["baseline"] = reference_base
            self.color_correction = darsia.ColorCorrection(config=self.config["color"])

        self.curvature_correction = (
            darsia.CurvatureCorrection(config=self.config["curvature"])
            if "curvature" in self.config
            else None
        )
        """Curvature correction."""

        # ! ---- Pre-define empty relative correction objects.

        self.drift_correction = None
        """Drift correction wrt. baseline image."""

        self.deformation_correction = None
        """Local deformation correction wrt. baseline image."""

        # ! ---- Corrected baseline

        # Define baseline image as corrected image
        self.base = self._read(reference_base)

        # ! ---- Relative correction objects

        self.drift_correction = darsia.DriftCorrection(
            base=self.base,
            config=self.config["drift"] if "drift" in self.config else None,
        )

        self.deformation_correction = darsia.DeformationCorrection(
            base=self.base,
            config=self.config["deformation"] if "deformation" in self.config else None,
        )

    # ! ----- I/O

    def _read(self, path: Union[str, Path]) -> darsia.GeneralImage:
        """
        Auxiliary reading methods for darsia Images.

        Args:
            path (str or Path): path to file.

        Returns:
            darsia.GeneralImage: image corrected for curvature and color.

        """
        # Use general interface to read image from file and apply correction
        return darsia.imread(
            path,
            transformations=[
                self.drift_correction,
                self.color_correction,
                self.translation_correction,
                self.curvature_correction,
                self.deformation_correction,
            ],
            width=self.width,
            height=self.height,
            origin=self.origin,
        )

    def load_and_process_image(self, path: Union[str, Path]) -> darsia.GeneralImage:
        """
        Load image for further analysis. Do all corrections and processing needed.

        Args:
            path (str or Path): path to image

        Returns:
            darsia.GeneralImage: processed image

        """

        # Read and process
        self.img = self._read(path)

        return self.img

    # ! ---- Analysis tools

    def single_image_analysis(self, img: Union[Path, darsia.Image], **kwargs) -> None:
        """
        Standard workflow to analyze CO2 phases.

        Args:
            image (Path or Image): path to single image.
            kwargs: optional keyword arguments

        """
        raise NotImplementedError("Has to be implemented for each special case.")

    def batch_analysis(self, images: list[Path], **kwargs) -> None:
        """
        Standard batch analysis.

        Args:
            images (list of Path): paths to batch of images.
            kwargs: optional keyword arguments used in single_image_analysis.

        """

        for img in images:

            tic = time.time()

            # Determine binary mask detecting any(!) CO2, and CO2(g)
            self.single_image_analysis(img, **kwargs)

            # Information to the user
            if self.verbosity:
                print(f"Elapsed time for {img.name}: {time.time()- tic}.")
