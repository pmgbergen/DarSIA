"""
Module containing objects to align images with a baseline image,
when restricted to a significant ROI. By this correction for drift
is taking care of.
"""

import copy
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image as PIL_Image

import daria


class DriftCorrection:
    """
    Class for drift correction of images wrt. a baseline image.
    """

    def __init__(
        self,
        base: Union[str, Path, np.ndarray, daria.Image],
        roi: Optional[Union[np.ndarray, tuple, list]] = None,
        config: Optional[Union[dict, str, Path]] = None,
    ) -> None:
        """
        Constructor for DriftCorrection.

        Args:
            base (str, Path, or array): path to baseline array, or array.
            roi (2-tuple of slices or array): region of interest defining
                the considered area for detecting features and aligning
                images. Either as tuple of ranges, or array of points.
            config (dict, str, path): config file for initialization of
                images. Can replace roi.
        """

        # Read baseline image
        if isinstance(base, str) or isinstance(base, Path):
            pil_base = PIL_Image.open(Path(base))
            self.base = np.array(pil_base)
        elif isinstance(base, np.ndarray):
            self.base = np.copy(base)
        elif isinstance(base, daria.Image):
            self.base = np.copy(base.img)
        else:
            raise ValueError("Data type for baseline image not supported.")

        # Cache config
        if config is not None:
            if isinstance(config, str):
                with open(str(Path(config)), "r") as openfile:
                    self.config = json.load(openfile)
            elif isinstance(config, Path):
                with open(str(config), "r") as openfile:
                    self.config = json.load(openfile)
            else:
                self.config = copy.deepcopy(config)
        else:
            self.config: dict = {}

        # Cache ROI
        if isinstance(roi, np.ndarray):
            self.roi = daria.bounding_box(roi)
            self.config["roi_drift_correction"] = self.roi.tolist()
        elif isinstance(roi, list):
            self.roi = daria.bounding_box(
                np.array(roi),
                padding=round(0.05 * self.base.shape[0]),
                max_size=[self.base.shape[0], self.base.shape[1]],
            )
            self.config["roi_drift_correction"] = daria.bounding_box_inverse(
                self.roi
            ).tolist()
        elif isinstance(roi, tuple):
            self.roi = roi
            self.config["roi_drift_correction"] = daria.bounding_box_inverse(
                roi
            ).tolist()
        elif "roi_drift_correction" in self.config:
            self.roi = daria.bounding_box(np.array(self.config["roi_drift_correction"]))
        elif "roi_color_correction" in self.config:
            self.roi = daria.bounding_box(
                np.array(self.config["roi_color_correction"]),
                padding=round(0.05 * self.base.shape[0]),
                max_size=[self.base.shape[0], self.base.shape[1]],
            )
        else:
            self.roi = None

        # Define a translation estimator
        self.translation_estimator = daria.TranslationEstimator()

    def __call__(self, img: np.ndarray, roi: Optional[tuple] = None) -> np.ndarray:
        """
        Main routine for aligning image with baseline image.

        Args:
            img (np.ndarray): input image, to be aligned.
            roi (2-tuple of slices, optional): ROI to be applied to img; if None
                the cached roi is used.

        Returns:
            np.ndarray: aligned image array.
        """
        # Define roi for source image. Let input argument be dominating over self.roi.
        roi_src = self.roi if roi is None else roi
        # Match image with baseline image
        return self.translation_estimator.match_roi(
            img_src=img, img_dst=self.base, roi_src=roi_src, roi_dst=self.roi
        )

    def write_config_to_file(self, path: Union[Path, str]) -> None:
        """
        Writes the config dictionary to a json-file.

        Arguments:
            path (Path): path to the json file
        """

        with open(Path(path), "w") as outfile:
            json.dump(self.config, outfile, indent=4)
