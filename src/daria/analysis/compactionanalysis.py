"""
Module containing tools for studying compaction.
"""

import numpy as np

import daria


class CompactionAnalysis:
    """
    Class to analyze compaction between different images.

    After all, CompactionAnalysis is a wrapper using TranslationAnalysis.
    """

    def __init__(self, base: daria.Image, **kwargs) -> None:
        """Constructor for CompactionAnalysis.

        Args:
            base (daria.Image): baseline image
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
        self.translation_estimator = daria.TranslationEstimator(max_features, tol)

        # Create translation analysis tool, and use the baseline image as reference point
        self.N_patches = kwargs.pop("N_patches", [1, 1])
        self.rel_overlap = kwargs.pop("rel_overlap", 0.0)
        self.translation_analysis = daria.TranslationAnalysis(
            base,
            N_patches=self.N_patches,
            rel_overlap=self.rel_overlap,
            translationEstimator=self.translation_estimator,
        )

    def update_base(self, base: daria.Image) -> None:
        """
        Update of baseline image.

        Args:
            img (np.ndarray): image array
        """
        self.translation_analysis.update_base(base)

    def __call__(
        self,
        img: daria.Image,
        plot: bool = False,
        reverse: bool = False,
        return_patch_translation: bool = False,
    ):
        """
        Determine the compaction patter and apply compaction to the image
        aiming at matching the baseline image.

        This in the end only a wrapper for the translation analysis.

        Args:
            img (daria.Image): test image
            plot (bool): flag controlling whether the deformation is also
                visualized as vector field.
            reverse (bool): flag whether the translation is understood as from the
                test image to the baseline image, or reversed. The default is the
                former one.
            return_patch_translation (bool): flag controlling whether the deformation
                in the patch centers is returned in the sense of dst to src image,
                complying to the plot; default is False.
        """
        transformed_img = self.translation_analysis(img)
        if return_patch_translation:
            patch_translation = self.translation_analysis.return_patch_translation()
            return transformed_img, patch_translation
        if plot:
            self.translation_analysis.plot_translation()
        return transformed_img

    def evaluate(self, coords: np.ndarray, units: str = "metric") -> np.ndarray:
        """
        Evaluate compaction in arbitrary points.

        Args:
            coords (np.ndarray): coordinate array with shape num_pts x 2, or alternatively
                num_x_pts x num_y_pts x 2, identifying points in a mesh/patched image.
            units (str): input and output units; "metric" default; otherwise assumed
                to be "pixel".

        Returns:
            np.ndarray: compaction vectors for all coordinates.

        """
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
        translation_x = self.translation_analysis.interpolator_translation_x(
            pixel_coords
        )
        translation_y = self.translation_analysis.interpolator_translation_y(
            pixel_coords
        )

        # Collect results, use ordering of components consistent with matrix indexing
        deformation = np.transpose(np.vstack((translation_y, translation_x)))

        # Convert to metric units if required; for pixels, use matrix indexing.
        if units == "metric":
            deformation = base.coordinatesystem.pixelToCoordinateVector(deformation)

        # Reshape to format used at input
        return deformation.reshape(coords_shape)

    def apply(self, img: daria.Image) -> daria.Image:
        """
        Apply computed transformation onto arbitrary image.

        Args:
            img (np.ndarray or daria.Image): image

        Returns:
            np.ndarray, optional: transformed image, if input is array; no output otherwise
        """
        # Load the image into translation_analysis
        self.translation_analysis.load_image(img)

        # Apply the transformation, stored in translation_analysis
        return self.translation_analysis.translate_image()
