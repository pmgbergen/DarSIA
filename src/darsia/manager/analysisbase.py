"""
Module providing structures for common analyses. In practice, such may
have to be tailored to the specific scenario. Yet, they already provide
many of the most relevant functionalities. If not applicable, they also
provide the approach for how to set up tailored analysis classes.
"""

import json
from pathlib import Path
from typing import Union, cast

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
        # Read general config file
        f = open(config, "r")
        self.config = json.load(f)
        f.close()

        # Define set of baseline images and initiate object for caching
        # processed baseline images.
        if not isinstance(baseline, list):
            reference_base = cast(Union[str, Path], baseline)

        else:
            reference_base = cast(Union[str, Path], baseline[0])
        self.processed_baseline_images = None

        # Define correction objects
        self.drift_correction = darsia.DriftCorrection(
            base=reference_base, config=self.config["drift"]
        )
        self.color_correction = darsia.ColorCorrection(config=self.config["color"])
        self.curvature_correction = darsia.CurvatureCorrection(
            config=self.config["curvature"]
        )

        # Define baseline image as corrected darsia Image
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
        return darsia.Image(
            img=path,
            drift_correction=self.drift_correction,
            curvature_correction=self.curvature_correction,
            color_correction=self.color_correction,
        )

    def load_and_process_image(self, path: Union[str, Path]) -> None:
        """
        Load image for further analysis. Do all corrections and processing needed.

        Args:
            path (str or Path): path to image
        """

        # Read and process
        self.img = self._read(path)
