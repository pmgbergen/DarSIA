"""
Module containing class with inpainting workflow for binary signals.
"""

from typing import Optional

import numpy as np
import skimage


class BinaryRemoveSmallObjects:
    """
    Wrapper for removing small objects in binary images with skimage.

    """

    def __init__(self, min_size: Optional[int] = None, key: str = "", **kwargs) -> None:
        """
        Args:
            min_size (int): min size of objects which will not be removed.
            key (str): prefix
            kwargs (keyword arguments)

        """
        self.min_size = (
            kwargs.get(key + "remove small objects size", 1)
            if min_size is None
            else min_size
        )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Remove small objects.

        Args:
            img (np.ndarray): boolean input image

        Returns:
            np.ndarray: boolean inpainted image

        """
        if self.min_size > 1:
            img = skimage.morphology.remove_small_objects(img, min_size=self.min_size)
        return img


class BinaryFillHoles:
    """
    Wrapper for filing holes in binary images with skimage.

    """

    def __init__(
        self, area_threshold: Optional[int] = None, key: str = "", **kwargs
    ) -> None:
        """
        Args:
            area_threshold (int): max size of holes which will be filled.
            key (str): prefix
            kwargs (keyword arguments)

        """
        self.area_threshold = (
            kwargs.get(key + "fill holes size", 0)
            if area_threshold is None
            else area_threshold
        )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Fill holes.

        Args:
            img (np.ndarray): boolean input image

        Returns:
            np.ndarray: boolean inpainted image

        """
        if self.area_threshold > 0:
            img = skimage.morphology.remove_small_holes(
                img, area_threshold=self.area_threshold
            )
        return img


class BinaryLocalConvexCover:
    """
    Local convex cover using local convex hulls with skimage.

    """

    def __init__(
        self, cover_patch_size: Optional[int] = None, key: str = "", **kwargs
    ) -> None:
        """
        Args:
            cover_patch_size (int): size of local patches
            key (str): prefix
            kwargs (keyword arguments)

        """
        self.cover_patch_size = (
            kwargs.get(key + "local convex cover size", 0)
            if cover_patch_size is None
            else cover_patch_size
        )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Fill holes.

        Args:
            img (np.ndarray): boolean input image

        Returns:
            np.ndarray: boolean inpainted image

        """
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
