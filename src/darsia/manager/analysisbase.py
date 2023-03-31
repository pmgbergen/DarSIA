"""
Module providing structures for common analyses. In practice, such may
have to be tailored to the specific scenario. Yet, they already provide
many of the most relevant functionalities. If not applicable, they also
provide the approach for how to set up tailored analysis classes.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, cast

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
        Constructor.

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

        # ! ---- Time
        reference_date_str: str = self.config.get("reference_date", None)
        self.reference_date: Optional[datetime] = (
            None
            if reference_date_str is None
            else datetime.strptime(reference_date_str, "%Y-%m-%d %H:%M:%S")
        )

        # ! ---- Reference to baseline images

        if isinstance(baseline, list):
            reference_base = cast(Union[str, Path], baseline[0])
        else:
            reference_base = cast(Union[str, Path], baseline)
        self.processed_baseline_images = None
        """List of corrected baseline images."""

        # ! ---- Pre-define empty correction objects.

        self.drift_correction = None
        """Drift correction wrt. baseline image."""
        self.color_correction = None
        """Color correction based on reference colors."""
        self.translation_correction = None
        """Translation correction based on fixed absolute translation."""
        self.curvature_correction = None
        """Curvature correction."""
        self.deformation_correction = None
        """Local deformation correction wrt. baseline image."""

        # ! ---- Define uncorrected baseline image.

        self.uncorrected_base = self._read(reference_base)
        """Baseline image stored as physical image but without corrections."""

        # ! ---- Define absolute correction objects

        if "translation" in self.config:
            self.translation_correction = darsia.TranslationCorrection(
                translation=self.config["translation"]
            )

        if "color" in self.config:
            if "baseline" not in self.config["color"]:
                self.config["color"]["baseline"] = reference_base
            self.color_correction = darsia.ColorCorrection(config=self.config["color"])

        if "curvature" in self.config:
            self.curvature_correction = darsia.CurvatureCorrection(
                config=self.config["curvature"]
            )

        # ! ---- Relative correction objects

        # NOTE: The order of application of the transformations decides over which
        # reference baseline image shall be chosen. Here, both corrections are applied
        # before applying any curvature correction. Thus, the uncorrected baseline
        # image is chosen as reference.

        if "drift" in self.config:
            self.drift_correction = darsia.DriftCorrection(
                base=self.uncorrected_base,
                config=self.config["drift"],
            )

        if "deformation" in self.config:
            self.deformation_correction = darsia.DeformationCorrection(
                base=self.uncorrected_base,
                config=self.config["deformation"],
            )

        # ! ---- Corrected baseline

        # Define baseline image as corrected image
        self.base = self._read(reference_base)

    # ! ----- I/O

    def _read(self, path: Union[str, Path]) -> darsia.Image:
        """
        Auxiliary reading methods for darsia Images.

        Args:
            path (str or Path): path to file.

        Returns:
            darsia.Image: image corrected for curvature and color.

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
            reference_date=self.reference_date,
        )

    def load_and_process_image(self, path: Union[str, Path]) -> darsia.Image:
        """
        Load image for further analysis. Do all corrections and processing needed.

        Args:
            path (str or Path): path to image

        Returns:
            darsia.Image: processed image

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

    def batch_analysis(self, images: Union[Path, list[Path]], **kwargs) -> None:
        """
        Standard batch analysis.

        Args:
            images (list of Path): paths to batch of images.
            kwargs: optional keyword arguments used in single_image_analysis.

        """

        if not isinstance(images, list):
            images = [images]

        for img in images:

            tic = time.time()

            # Determine binary mask detecting any(!) CO2, and CO2(g)
            self.single_image_analysis(img, **kwargs)

            # Information to the user
            if self.verbosity:
                print(f"Elapsed time for {img.name}: {time.time()- tic}.")
