"""Module containing a local deformation correction.

This is essentially a correction object through wrappping
:mod:`darsia.DiffeomorphicImageRegistration`.

"""
from typing import Optional

import numpy as np

import darsia


class DeformationCorrection(darsia.BaseCorrection):
    """Patch-wise defined deformation correction defined through image registration.

    Attributes:
        base (darsia.Image): reference (baseline) image.

    """

    def __init__(self, base: darsia.Image, config: Optional[dict]) -> None:
        """Constructor.

        Args:
            base (darsia.Image): baseline image
            config (dict, optional): contains all tuning parameters.

        """
        # Convert config to dictionary.
        if config is None:
            config = {}
            self.active = False
        else:
            # Check whether correction active
            self.active = config.get("active", True)

        if self.active:
            self.base = base
            if config is None:
                config = {}
            self.image_registration = darsia.ImageRegistration(self.base, **config)

    def correct_array(self, img: np.ndarray) -> np.ndarray:
        """Main routine for aligning image with baseline image.

        Args:
            img (np.ndarray): input image, to be aligned.

        Returns:
            np.ndarray: aligned image array.

        """
        if self.active:
            metadata = self.base.metadata()
            transformed_img: darsia.Image = self.image_registration(
                darsia.Image(img, **metadata)
            )
            return transformed_img.img
        else:
            return img
