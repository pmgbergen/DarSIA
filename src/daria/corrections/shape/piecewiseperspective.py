"""
Module containing class which manages performing a translation based on
a provided pointwise translation map.
"""

from typing import Callable

import cv2
import numpy as np
import skimage

import daria


class PiecewisePerspectiveTransform:
    """
    Class performing a piecewise perspective transform, applied to a structured grid.
    The final transformation is continuous.
    """

    def __init__(self) -> None:
        """
        Constructor method.

        Stores exteranl inputs as patches.
        """
        # Initialize flag
        self.have_transform = False

    def find_and_warp(
        self, patches: daria.Patches, displacement: Callable, reverse: bool = False
    ) -> daria.Image:
        """
        Continuously transform entire image via patchwise perspective transform.

        Perspective transforms determined by the function evaluation in all
        corners of each patch results in continuous transformations.
        Hence, stiching together the transformed images results in
        an image without overlap and gaps.

        Args:
            patches (daria.Patches): patched image
            displacement (Callable): relative deformation map; assumed to be continuous
            reverse (bool): flag whether displacement is applied with negative weight

        Returns:
            daria.Image: transformed image
        """
        # NOTE: At the moment, no normalization is considered.

        # Initialize empty image of same type as the original image
        (h, w) = patches.base.shape[:2]
        transformed_img = np.zeros(patches.base.shape, dtype=float)

        # Loop over all patches in a grid fashion:
        for i in range(patches.num_patches_x):
            for j in range(patches.num_patches_y):

                # Determine the pixels of the corners in compying to reverse matrix indexing
                global_corners = patches.global_corners_reverse_matrix[i, j]
                local_corners = patches.local_corners_reverse_matrix[i, j]

                # Determine the coordinates after applying the trasnformation
                # NOTE: Strangely, the interpolator reshapes the arrays from
                # input to output.
                pts_src = local_corners
                pts_dst = (
                    global_corners
                    + (-1.0 if reverse else 1.0) * displacement(global_corners).T
                )

                # Find perspective transform mapping src to dst pixels in reverse matrix format
                P = cv2.getPerspectiveTransform(
                    pts_src.astype(np.float32), pts_dst.astype(np.float32)
                )

                # Map patch onto transformed region
                patch_img = skimage.util.img_as_float(patches.images[i][j].img)
                transformed_img += cv2.warpPerspective(
                    patch_img, P, (w, h), flags=cv2.INTER_LINEAR
                )

        # Fetch metadata from originating patches
        origo = patches.base.origo
        width = patches.base.width
        height = patches.base.height

        return daria.Image(img=transformed_img, origo=origo, width=width, height=height)
