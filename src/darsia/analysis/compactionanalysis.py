"""
Module containing tools for studying compaction.
"""

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

import darsia


class ReversedCompactionAnalysis:
    """
    Class to analyze compaction between different images.

    After all, CompactionAnalysis is a wrapper using TranslationAnalysis.
    """

    def __init__(self, dst: darsia.Image, **kwargs) -> None:
        """Constructor for CompactionAnalysis.

        Args:
            dst (darsia.Image): reference image which is supposed to be fixed in the compaction
            optional keyword arguments:
                N_patches (list of two int): number of patches in x and y direction
                rel_overlap (float): relative overlap in each direction, related to the
                patch size
                max_features (int) maximal number of features in thefeature detection
                tol (float): tolerance
                mask (np.ndarray, optional): roi in which features are considered.
        """
        # Create translation estimator
        max_features = kwargs.pop("max_features", 200)
        tol = kwargs.pop("tol", 0.05)
        self.translation_estimator = darsia.TranslationEstimator(max_features, tol)

        # Create translation analysis tool, and use the baseline image as reference point
        self.N_patches = kwargs.pop("N_patches", [1, 1])
        self.rel_overlap = kwargs.pop("rel_overlap", 0.0)
        mask: Optional[darsia.Image] = kwargs.get("mask", None)
        self.translation_analysis = darsia.TranslationAnalysis(
            dst,
            N_patches=self.N_patches,
            rel_overlap=self.rel_overlap,
            translationEstimator=self.translation_estimator,
            mask=mask,
        )

    def update_dst(self, dst: darsia.Image) -> None:
        """
        Update of dst image.

        Args:
            dst (np.ndarray): image array
        """
        self.translation_analysis.update_base(dst)

    def deduct(self, compaction_analysis) -> None:  #: CompactionAnalysis
        """
        Effectviely copy from external CompactionAnalysis.

        Args:
            compaction_analysis (darsia.ReversedCompactionAnalysis): Compaction
                analysis holding a translation analysis.

        """
        # The displacement is stored in the translation analysis as callable.
        # Thus, the current translation analysis has to be updated.
        self.translation_analysis.deduct_translation_analysis(
            compaction_analysis.translation_analysis
        )

    def add(self, compaction_analysis) -> None:  #: CompactionAnalysis
        """
        Update the store translation by adding the translation
        of an external compaction analysis.

        Args:
            compaction_analysis (darsia.ReversedCompactionAnalysis): Compaction
                analysis holding a translation analysis.
        """
        # The displacement is stored in the translation analysis as callable.
        # Thus, the current translation analysis has to be updated.
        self.translation_analysis.add_translation_analysis(
            compaction_analysis.translation_analysis
        )

    def __call__(
        self,
        img: darsia.Image,
        plot_patch_translation: bool = False,
        return_patch_translation: bool = False,
        mask: Optional[darsia.Image] = None,
    ):
        """
        Determine the compaction pattern and apply compaction to the image
        aiming at matching the reference (dst) image.

        This in the end only a wrapper for the translation analysis.

        Args:
            img (darsia.Image): test image
            reverse (bool): flag whether the translation is understood as from the
                test image to the dst image, or reversed. The default is the
                former.
            plot_patch_translation (bool): flag controlling whether the displacement is also
                visualized as vector field.
            return_patch_translation (bool): flag controlling whether the displacement
                in the patch centers is returned in the sense of img to dst,
                complying to the plot; default is False.
        """
        transformed_img = self.translation_analysis(img, mask=mask)

        if return_patch_translation:
            patch_translation = self.translation_analysis.return_patch_translation(
                False
            )

        if plot_patch_translation:
            self.translation_analysis.plot_translation(False)

        if return_patch_translation:
            return transformed_img, patch_translation
        else:
            return transformed_img

    def evaluate(
        self,
        coords: Union[np.ndarray, darsia.Patches],
        reverse: bool = False,
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

    def apply(self, img: darsia.Image, reverse: bool = False) -> darsia.Image:
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

    def plot(self, scaling: float = 1.0, mask: Optional[darsia.Image] = None) -> None:
        """
        Plots total compaction.
        """
        # Warpper for translation_analysis.
        self.translation_analysis.plot_translation(
            reverse=False, scaling=scaling, mask=mask
        )

    def displacement(self) -> np.ndarray:
        """
        Return displacement in metric units on all pixels.
        """
        # Define coordinates for each pixel
        Ny, Nx = self.translation_analysis.base.img.shape[:2]
        # Nz = 1
        x = np.arange(Nx)
        y = np.arange(Ny)
        X_pixel, Y_pixel = np.meshgrid(x, y)

        # Transform coordinates into the right format (vector)
        pixel_vector = np.transpose(np.vstack((np.ravel(X_pixel), np.ravel(Y_pixel))))
        # Reshape coords using a num_pts x 2 format.
        pixel_coords = pixel_vector.reshape(-1, 2)

        # Interpolate at provided values - expect reverse matrix indexing
        import time

        tic = time.time()
        translation = self.translation_analysis.translation(pixel_coords)
        print(f"translation evaluation: {time.time() - tic}")

        #        # Flip, if required
        #        if reverse:
        #            translation *= -1.0

        # results, use ardering of components consistent with matrix indexing
        displacement = np.transpose(np.vstack((translation[1], translation[0])))

        # Convert to metric units
        tic = time.time()
        displacement = (
            self.translation_analysis.base.coordinatesystem.pixelToCoordinateVector(
                displacement
            )
        )
        print(f"coordinate system: {time.time() - tic}")

        # Cache
        self.displacement = displacement

        return displacement
