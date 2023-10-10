"""
Module containing objects to align images with a baseline image,
when restricted to a significant ROI. By this correction for drift
is taking care of.

"""

from typing import Optional, Union

import numpy as np

import darsia


class DriftCorrection(darsia.BaseCorrection):
    """
    Class for drift correction of images wrt. a baseline image.

    """

    def __init__(
        self,
        base: Union[np.ndarray, darsia.Image],
        config: Optional[dict] = None,
        roi: Optional[Union[tuple, np.ndarray, list]] = None,
    ) -> None:
        """
        Constructor for DriftCorrection.

        Args:
            base (array or Image): baseline.
            config (dict): config file for initialization of images. Main
                attributes:
                - roi (2-tuple of slices or array): region of interest defining
                    the considered area for detecting features and aligning
                    images. Either as tuple of ranges, or array of points.
                    Can also be provided in config; roi in config is
                    prioritized.
                - padding (float): relative factor for padding.
                - active (bool): flag whether drift correction should be
                    applied or not, default is True.
            roi (list or array), alternative access to roi.

        """

        # Read baseline image
        if isinstance(base, darsia.Image):
            self.base = np.copy(base.img)
            """Base image array."""

            if base.space_dim != 2:
                raise NotImplementedError

        elif isinstance(base, np.ndarray):
            self.base = np.copy(base)

        else:
            raise ValueError("Data type for baseline image not supported.")

        if config is None:
            config = {}

            self.active = False
            """Flag controlling whether correction is active."""

        else:

            self.active: bool = config.get("active", True)

        if self.active:
            roi: Optional[Union[list, tuple]] = config.get("roi", None)
            relative_padding: float = config.get("padding", 0.0)
            self.roi: Optional[tuple] = None
            """Basis/ROI for feature detection."""

            if "roi" in config:
                self.roi = np.array(config["roi"])
            elif roi is not None:
                assert isinstance(roi, list) or isinstance(
                    roi, np.ndarray
                ), "wrong format"
                self.roi = darsia.bounding_box(
                    np.array(roi),
                    padding=round(relative_padding * np.min(self.base.shape[:2])),
                    max_size=self.base.shape[:2],
                )
            elif roi is not None:
                raise ValueError("Provide 'roi' in config or as argument.")

            self.translation_estimator = darsia.TranslationEstimator()
            """Detection of effective translation based on feature detection."""

    # ! ---- Main correction routines

    def correct_array(self, img: np.ndarray, roi: Optional[tuple] = None) -> np.ndarray:
        """
        Main routine for aligning image with baseline image.

        Args:
            img (np.ndarray): input image, to be aligned.
            roi (2-tuple of slices, optional): ROI to be applied to img; if None
                the cached roi is used.

        Returns:
            np.ndarray: aligned image array.
        """
        if self.active:
            # Define roi for source image. Let input argument be dominating over self.roi.
            roi_src = self.roi if roi is None else roi
            # Match image with baseline image
            return self.translation_estimator.match_roi(
                img_src=img, img_dst=self.base, roi_src=roi_src, roi_dst=self.roi
            )
        else:
            return img
