"""
Module containing class with inpainting workflow for binary signals.
"""

import numpy as np
import skimage


class BinaryInpaint:
    def __init__(self, **kwargs) -> None:
        """
        Constructor.

        Read parameters from keyword arguments.
        """
        self.min_size: int = kwargs.pop("min area size", 1)

        # Parameters to fill holes
        self.area_threshold: int = kwargs.pop("max hole size", 0)

        # Parameters for local convex cover
        self.cover_patch_size: int = kwargs.pop("local convex cover patch size", 1)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Binary inpainting routine.

        Args:
            img (np.ndarray): boolean input image

        Returns:
            np.ndarray: boolean inpainted image
        """

        # Remove small objects
        if self.min_size > 1:
            img = skimage.morphology.remove_small_objects(img, min_size=self.min_size)

        # Fill holes
        if self.area_threshold > 0:
            img = skimage.morphology.remove_small_holes(
                img, area_threshold=self.area_threshold
            )

        # Loop through patches and fill up
        if self.cover_patch_size > 1:
            covered_img = np.zeros(img.shape[:2], dtype=bool)
            size = self.cover_patch_size
            Ny, Nx = img.shape[:2]
            for row in range(int(Ny / size)):
                for col in range(int(Nx / size)):
                    roi = (
                        slice(row * size, (row + 1) * size),
                        slice(col * size, (col + 1) * size),
                    )
                    covered_img[roi] = skimage.morphology.convex_hull_image(img[roi])
            # Update the img value
            img = covered_img

        return img
