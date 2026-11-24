"""Image porosity by k-means clustering and kernel interpolation (legacy method)."""

import darsia
from darsia.presets.analysis.porosity import patched_porosity_analysis
from darsia.presets.workflows.config.image_porosity import KMeansPorosityConfig


def detect(
    baseline: darsia.Image, labels: darsia.Image, options: KMeansPorosityConfig
) -> darsia.Image:
    """Image porosity of the baseline by the k-means method.

    Parameters
    ----------
    baseline : darsia.Image
        Baseline image.
    labels : darsia.Image
        Labels image.
    options : KMeansPorosityConfig
        Options of the method.

    Returns
    -------
    darsia.Image
        Image porosity in [0, 1].

    """
    return patched_porosity_analysis(
        baseline=baseline,
        patches=options.patches,
        labels=labels,
        num_clusters=options.num_clusters,
        tol_color_distance=options.tol_color_distance,
        tol_color_gradient=options.tol_color_gradient,
    )
