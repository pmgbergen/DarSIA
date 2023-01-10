"""
Module containing class which manages performing a translation based on
a provided pointwise translation map.
"""

from typing import Callable

import cv2
import numpy as np

import darsia
from math import ceil


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
        self, patches: darsia.Patches, displacement: Callable, reverse: bool = False
    ) -> darsia.Image:
        """
        Continuously transform entire image via patchwise perspective transform.

        Perspective transforms determined by the function evaluation in all
        corners of each patch results in continuous transformations.
        Hence, stiching together the transformed images results in
        an image without overlap and gaps.

        Args:
            patches (darsia.Patches): patched image
            displacement (Callable): relative deformation map; assumed to be continuous
            reverse (bool): flag whether displacement is applied with negative weight

        Returns:
            darsia.Image: transformed image
        """
        # Initialize empty image of same type as the original image.
        # Need float for cv2.warpPerspective.
        (h, w) = patches.base.img.shape[:2]
        dtype = patches.base.img.dtype
        transformed_img = np.zeros(patches.base.img.shape, dtype=np.float32)
        transformed_img_new = np.zeros(patches.base.img.shape, dtype=np.float32)

        import time
        total = 0
        total_new = 0

        # Loop over all patches in a grid fashion:
        for i in range(patches.num_patches_x):
            for j in range(patches.num_patches_y):

                # Determine the pixels of the corners in compying to reverse matrix indexing
                global_corners = patches.global_corners_reverse_matrix[i, j]
                local_corners = patches.local_corners_reverse_matrix[i, j]

                # Determine the coordinates after applying the transformation
                # NOTE: Strangely, the interpolator reshapes the arrays from
                # input to output.
                pts_src = local_corners
                pts_dst = (
                    global_corners
                    + (-1.0 if reverse else 1.0) * displacement(global_corners).T
                )

                # TODO rm and rename the eff variables?

                # # Find perspective transform mapping src to dst pixels in reverse matrix format
                # P = cv2.getPerspectiveTransform(
                #     pts_src.astype(np.float32), pts_dst.astype(np.float32)
                # )

                # # Map patch onto transformed region
                # patch_img = patches.images[i][j].img.astype(np.float32)
                # tic = time.time()
                # print(patch_img.shape, w, h)
                # transformed_img += cv2.warpPerspective(
                #     patch_img, P, (w, h), flags=cv2.INTER_LINEAR
                # )
                # total += time.time() - tic

                # Find effective origin, width and height of the transformed patch
                origin_eff = np.array([
                    max(0, np.min(pts_dst[:,0])),
                    max(0, np.min(pts_dst[:,1]))
                ]).astype(np.int32)
                pts_dst_eff = pts_dst - origin_eff
                w_eff = ceil(min(w, np.max(pts_dst[:,0])) - max(0, origin_eff[0]))
                h_eff = ceil(min(h, np.max(pts_dst[:,1])) - max(0, origin_eff[1]))

                # Continue with the next patch, if the effective size becomes at most a single pixel.
                if min(h_eff, w_eff) <= 1:
                    continue

                # NOTE: Flip of x and y to row and col.
                roi_eff = (
                    slice(max(0, int(origin_eff[1])), min(h, int(origin_eff[1] + h_eff))),
                    slice(max(0, int(origin_eff[0])), min(w, int(origin_eff[0] + w_eff))))
                roi_patch_eff = (
                    slice(0, int(min(origin_eff[1] + h_eff, h) - origin_eff[1])),
                    slice(0, min(origin_eff[0] + w_eff, w) - origin_eff[0])
                )

                # Find perspective transform mapping src to dst pixels in reverse matrix format
                P_eff = cv2.getPerspectiveTransform(
                    pts_src.astype(np.float32), pts_dst_eff.astype(np.float32)
                )

                # Map patch onto transformed region
                patch_img = patches.images[i][j].img.astype(np.float32)
                tic = time.time()
                tmp = cv2.warpPerspective(
                    patch_img, P_eff, (w_eff, h_eff), flags=cv2.INTER_LINEAR
                )
                transformed_img_new[roi_eff] += tmp[roi_patch_eff]
                total_new += time.time() - tic

        # Convert to the same data type as the input image
        transformed_img = transformed_img_new.astype(dtype)
        #print(f"total time: {total}")
        print(f"total time new: {total_new}")

        # Use same metadata as for the base of the patches
        metadata = patches.base.metadata

        return darsia.Image(img=transformed_img, metadata=metadata)
