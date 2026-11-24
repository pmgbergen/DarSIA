"""No detection: the image porosity is 1 everywhere (before cutoff and boundary band)."""

import numpy as np

import darsia


def detect(baseline: darsia.Image, labels: darsia.Image, options: None) -> darsia.Image:
    """Ones on the grid of the baseline.

    Parameters
    ----------
    baseline : darsia.Image
        Baseline image (only its grid is used).
    labels : darsia.Image
        Labels image (not used).
    options : None
        The method has no options.

    Returns
    -------
    darsia.Image
        Ones (float32).

    """
    return darsia.ones_like(baseline, mode="voxels", dtype=np.float32)
