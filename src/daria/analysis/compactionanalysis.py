"""
Module containing tools for studying compaction.
"""

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

    def __call__(self, img: daria.Image, plot: bool = False, reverse: bool = False):
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
        """
        transformed_img = self.translation_analysis(img)
        if plot:
            self.translation_analysis.plot_translation(reverse)
        return transformed_img
