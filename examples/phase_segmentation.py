"""Test phase segmentation, taken from FluidFlower analysis conducted in the work
"Ferno et al., Room-scale CO2 injections in a physical reservoir model with faults
https://doi.org/10.48550/arXiv.2301.06397"

This test tests the FluidFlower presets, and in particular the workflow for CO2
phase segmentation analysis for a fixed set of parameters.

NOTE: This test is mainly meant for developers. Thus the images (large in size)
are not shared - they can be obtained from 10.5281/zenodo.7510589

"""

import shutil
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pytest

from darsia.presets.fluidflower.fluidflowerco2analysis import FluidFlowerCO2Analysis
from darsia.presets.fluidflower.fluidflowerrig import FluidFlowerRig

# ! ---- (Minimal) FluidFlower CO2 analysis


class LargeFluidFlower(FluidFlowerRig):
    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        update_setup: bool = False,
    ) -> None:
        """
        Constructor for large FluidFlower rig specific data.

        Args:
            base (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config (str or Path): path to config dict
            update_setup (bool): flag controlling whether cache in setup
                routines is emptied.
        """
        super().__init__(baseline, config, update_setup)

    # ! ---- Auxiliary setup routines

    def _segment_geometry(self, update_setup: bool = False) -> None:
        """
        See SegmentedFluidFlower.

        """
        super()._segment_geometry(update_setup)

        # Identify water layer
        self.water = self._labels_to_mask(self.config["segmentation"]["water"])

        # Identify ESF layer
        self.esf_sand = self._labels_to_mask(self.config["segmentation"]["esf"])

        # Identify C layer
        self.c_sand = self._labels_to_mask(self.config["segmentation"]["c"])


class LargeRigCO2Analysis(LargeFluidFlower, FluidFlowerCO2Analysis):
    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        results: Union[str, Path],
        update_setup: bool = False,
        verbosity: bool = True,
    ) -> None:
        """
        Sets up fixed config file required for preprocessing.

        Args:
            baseline (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config (str or Path): path to config dict
            results (str or Path): path to results directory
            update_setup (bool): flag controlling whether cache in setup
                routines is emptied.
            verbosity  (bool): flag controlling whether results of the
                post-analysis are printed to screen; default is False.
        """
        LargeFluidFlower.__init__(self, baseline, config, update_setup)
        FluidFlowerCO2Analysis.__init__(
            self, baseline, config, results, update_setup, verbosity
        )

    def _expert_knowledge_co2_gas(self, co2) -> np.ndarray:
        """
        Retrieve expert knowledge, i.e., areas with possibility for CO2(g).

        Args:
            co2 (darsia.Image): mask for CO2.

        Returns:
            np.ndarray: mask with no CO2(g)

        """
        return np.logical_and(co2.img, np.logical_not(self.esf_sand))


"""Test whether phase segmentation prodcues a reference result.

NOTE: Only runs if test images available - not on GH.

"""

# ! ---- Input data

# Read user-defined paths to images, number of baseline images, and config file
# TODO make sure images are available and the correct path is provided here.
folder = Path("../tests/integration/segmentation")
images_folder = folder / Path("images")
images_exist = len(list(sorted(images_folder.glob("*.TIF")))) > 0

if not images_exist:
    pytest.xfail("Images required for test not available.")

# Define separation of baseline image and main image
images = list(sorted(images_folder.glob("*.TIF")))[10]
baseline = list(sorted(images_folder.glob("*.TIF")))[:10]
config = folder / Path("config.json")
results = folder / Path("results")
results.mkdir(parents=True, exist_ok=True)

# Define FluidFlower based on a full set of basline images
analysis = LargeRigCO2Analysis(
    baseline=baseline,  # paths to baseline images
    config=config,  # path to config file
    results=results,  # path to results directory
)

# Perform standardized CO2 batch analysis
analysis.batch_analysis(images, plot_contours=True, write_contours=True)
