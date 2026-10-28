"""Module containing common reading functionality for corrections."""

from darsia.fluidflower import preset_wasserstein_inputs as _preset_io

from pathlib import Path

import numpy as np

from darsia import (
    ColorCorrection,
    CurvatureCorrection,
    DriftCorrection,
    IlluminationCorrection,
    PatchwiseIlluminationCorrection,
    RelativeColorCorrection,
    ResizeCorrection,
    TypeCorrection,
)

AnyCorrection = (
    TypeCorrection
    | DriftCorrection
    | CurvatureCorrection
    | IlluminationCorrection
    | RelativeColorCorrection
    | PatchwiseIlluminationCorrection
    | ColorCorrection
    | ResizeCorrection
)


def read_correction(path: Path) -> AnyCorrection:
    """General function to read corrections from a file.

    Parameters
    ----------
    path : Path
        Path to npz file.

    Returns
    -------
    BaseCorrection
        Correction object.
    """

    # Read class name from npz file
    class_name = np.load(_preset_io.source(path), allow_pickle=True)["class_name"].item()

    # Older rigs saved the array resizer as "Resize". Its persisted settings
    # are shared with ResizeCorrection, which participates in the native rig's
    # BaseCorrection workflow and preserves the saved interpolation/geometry.
    correction_class = ResizeCorrection if class_name == "Resize" else eval(class_name)
    correction = correction_class()
    correction.load(path)
    return correction
