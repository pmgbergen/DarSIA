"""
Module providing structures for common analyses. In practice, such may
have to be tailored to the specific scenario. Yet, they already provide
many of the most relevant functionalities. If not applicable, they also
provide the approach for how to set up tailored analysis classes.
"""

import json
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
        self.translation_correction = (
            darsia.TranslationCorrection(translation=self.config["translation"])
            if "translation" in self.config
            else None
        )

        self.drift_correction = (
            darsia.DriftCorrection(base=reference_base, config=self.config["drift"])
            if "drift" in self.config
            else None
        )

        # Define color correction and provide baseline to colorchecker
        if "color" in self.config:
            if "baseline" not in self.config["color"]:
                self.config["color"]["baseline"] = reference_base
            self.color_correction = darsia.ColorCorrection(config=self.config["color"])
        else:
            self.color_correction = None

        # Define curvature correction
        self.curvature_correction = (
            darsia.CurvatureCorrection(config=self.config["curvature"])
            if "curvature" in self.config
            else None
        )

        # NOTE: Need to initialize deformation correction to call _read; OK as
        # base is used as baseline.
        self.deformation_correction = None

        # Define baseline image as corrected darsia Image
        self.base = self._read(reference_base)

        # Define deformation correction for the corrected image
        # FIXME: More meaningful would be a definition based on
        # reference_base - FIXME: Needs rewrite of translation analysis.
        self.deformation_correction = darsia.DeformationCorrection(
            base=self.base,
            config=self.config["deformation"] if "deformation" in self.config else None,
        )

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
            translation_correction=self.translation_correction,
            deformation_correction=self.deformation_correction,
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

    def store(
        self,
        img: darsia.Image,
        path: Path,
        cartesian_indexing: bool = False,
        store_jpg: bool = False,
        suffix_jpg: str = "",
        suffix_npy: str = "",
    ) -> bool:
        """Convert to correct format (use Cartesian indexing by default)
        and store to file (both as image and numpy array).

        Args:
            img (darsia.Image): image
            path (Path): path to file
            cartesian_indexing (bool): flag controlling whether data is stored with
                matrix indexing
            store_jpg (bool): flag controlling whether a jpg representation is stored
            suffix_jpg (str): suffix to be added to the jpg file
            suffix_npy (str): suffix to be added to the npy file

        Returns:
            bool: success of the routine (always True).

        """
        plain_path = path.with_suffix("")

        # Store the image
        if store_jpg:
            cv2.imwrite(
                str(plain_path) + suffix_jpg + ".jpg",
                skimage.util.img_as_ubyte(img.img),
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )

        # Store numpy array
        np.save(
            str(plain_path) + suffix_npy + ".npy",
            darsia.matrixToCartesianIndexing(img.img)
            if cartesian_indexing
            else img.img,
        )

        return True
