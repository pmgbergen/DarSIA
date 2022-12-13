"""
Module containing tools for studying compaction.
"""

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

import darsia


class CompactionAnalysis:
    """
    Class to analyze compaction between different images.

    After all, CompactionAnalysis is a wrapper using TranslationAnalysis.
    """

    def __init__(self, base: darsia.Image, **kwargs) -> None:
        """Constructor for CompactionAnalysis.

        Args:
            base (darsia.Image): baseline image
            optional keyword arguments:
                N_patches (list of two int): number of patches in x and y direction
                rel_overlap (float): relative overlap in each direction, related to the
                patch size
                max_features (int) maximal number of features in thefeature detection
                tol (float): tolerance
        """
        # Create translation estimator
        max_features = kwargs.pop("max_features", 200)
        tol = kwargs.pop("tol", 0.05)
        self.translation_estimator = darsia.TranslationEstimator(max_features, tol)

        # Create translation analysis tool, and use the baseline image as reference point
        self.N_patches = kwargs.pop("N_patches", [1, 1])
        self.rel_overlap = kwargs.pop("rel_overlap", 0.0)
        self.translation_analysis = darsia.TranslationAnalysis(
            base,
            N_patches=self.N_patches,
            rel_overlap=self.rel_overlap,
            translationEstimator=self.translation_estimator,
        )

    def update_base(self, base: darsia.Image) -> None:
        """
        Update of baseline image.

        Args:
            img (np.ndarray): image array
        """
        self.translation_analysis.update_base(base)

    def __call__(
        self,
        img: darsia.Image,
        reverse: bool = True,
        plot_patch_translation: bool = False,
        return_patch_translation: bool = False,
    ):
        """
        Determine the compaction patter and apply compaction to the image
        aiming at matching the baseline image.

        This in the end only a wrapper for the translation analysis.

        Args:
            img (darsia.Image): test image
            reverse (bool): flag whether the translation is understood as from the
                test image to the baseline image, or reversed. The default is the
                latter.
            plot_patch_translation (bool): flag controlling whether the displacement is also
                visualized as vector field.
            return_patch_translation (bool): flag controlling whether the displacement
                in the patch centers is returned in the sense of dst to src image,
                complying to the plot; default is False.
        """
        transformed_img = self.translation_analysis(img)

        if return_patch_translation:
            patch_translation = self.translation_analysis.return_patch_translation(
                reverse
            )

        if plot_patch_translation:
            self.translation_analysis.plot_translation()

        if return_patch_translation:
            return transformed_img, patch_translation
        else:
            return transformed_img

    def evaluate(
        self,
        coords: Union[np.ndarray, darsia.Patches],
        reverse: bool = True,
        units: str = "metric",
    ) -> np.ndarray:
        """
        Evaluate compaction in arbitrary points.

        Args:
            coords (np.ndarray, or darsia.Patches): coordinate array with shape num_pts x 2,
                or alternatively num_rows_pts x num_cols_pts x 2, identifying points in a
                mesh/patched image, or equivalently patch.
            reverse (bool): flag whether the translation is understood as from the
                test image to the baseline image, or reversed. The default is the
                former latter.
            units (str): input and output units; "metric" default; otherwise assumed
                to be "pixel".

        Returns:
            np.ndarray: compaction vectors for all coordinates.

        """
        if isinstance(coords, darsia.Patches):
            coords = coords.global_centers_cartesian_matrix

        assert units in ["metric", "pixel"]
        assert coords.shape[-1] == 2

        # Reshape coords using a num_pts x 2 format.
        coords_shape = coords.shape
        coords = coords.reshape(-1, 2)

        # Convert coordinates to pixels with matrix indexing, if provided in metric units
        if units == "metric":
            base = self.translation_analysis.base
            pixel_coords = base.coordinatesystem.coordinateToPixel(coords, reverse=True)
        else:
            pixel_coords = coords

        # Interpolate at provided values - expect reverse matrix indexing
        translation = self.translation_analysis.translation(pixel_coords)

        # Flip, if required
        if reverse:
            translation *= -1.0

        # Collect results, use ordering of components consistent with matrix
        # indexing (i.e. flip of components needed)

        displacement = np.transpose(np.vstack((translation[1], translation[0])))

        # Convert to metric units if required; for pixels, use matrix indexing.
        if units == "metric":
            displacement = base.coordinatesystem.pixelToCoordinateVector(displacement)

        # Reshape to format used at input
        return displacement.reshape(coords_shape)

    def apply(self, img: darsia.Image, reverse: bool = True) -> darsia.Image:
        """
        Apply computed transformation onto arbitrary image.

        Args:
            img (np.ndarray or darsia.Image): image
            reverse (bool): flag whether the translation is understood as from the
                test image to the baseline image, or reversed. The default is the
                latter.

        Returns:
            np.ndarray, optional: transformed image, if input is array; no output otherwise
        """
        # Load the image into translation_analysis
        self.translation_analysis.load_image(img)

        # Apply the transformation, stored in translation_analysis
        return self.translation_analysis.translate_image(reverse)

    def plot(self) -> None:
        """
        Plots total compaction.
        """
        # Warpper for translation_analysis.
        self.translation_analysis.plot_translation()
