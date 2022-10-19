"""
Module containing objects to align images with a baseline image,
when restricted to a significant ROI. By this correction for drift
is taking care of.
"""

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
        roi: Optional[Union[np.ndarray, tuple]] = None,
    ) -> None:
        """
        Constructor for DriftCorrection.

        Args:
            base (str, Path, or array): path to baseline array, or array.
            roi (2-tuple of slices or array): region of interest defining
                the considered area for detecting features and aligning
                images. Either as tuple of ranges, or array of points.
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

        # Cache roi
        self.roi = roi if isinstance(roi, tuple) else daria.bounding_box(roi)

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
