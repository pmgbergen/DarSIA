"""Module containing common reading functionality for corrections."""

from pathlib import Path

import numpy as np

from darsia import (
    ColorCorrection,
    CurvatureCorrection,
    DriftCorrection,
    IlluminationCorrection,
    TypeCorrection,
    Resize,
)

AnyCorrection = (
    TypeCorrection
    | DriftCorrection
    | CurvatureCorrection
    | IlluminationCorrection
    | ColorCorrection
    | Resize
)


def read_correction(path: Path) -> AnyCorrection:
    """General function to read corrections from a file.

    Args:
        path (Path): path to npz file

    Returns:
        BaseCorrection: correction object

    """

    # Read class name from npz file
    class_name = np.load(path, allow_pickle=True)["class_name"].item()

    # Load correction from file
    correction = eval(class_name)()
    correction.load(path)
    return correction
