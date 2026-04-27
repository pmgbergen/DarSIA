"""Shared calibration/analysis basis utilities."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np

if TYPE_CHECKING:
    from darsia.presets.workflows.rig import Rig


class CalibrationBasis(StrEnum):
    """Label space used by calibration and analysis workflows."""

    LABELS = "labels"
    FACIES = "facies"
    GLOBAL = "global"


def parse_calibration_basis(
    value: str | CalibrationBasis | None,
    default: CalibrationBasis = CalibrationBasis.FACIES,
) -> CalibrationBasis:
    """Parse user/config input into a :class:`CalibrationBasis`."""

    if value is None:
        return default
    if isinstance(value, CalibrationBasis):
        return value
    if isinstance(value, str):
        token = value.lower().strip()
        if token == "single":
            token = CalibrationBasis.GLOBAL.value
        return CalibrationBasis(token)
    raise TypeError(f"Unsupported calibration basis value type: {type(value)}")


def calibration_basis_folder(basis: str | CalibrationBasis) -> str:
    """Return standard folder suffix for basis-aware calibration artifacts."""

    parsed = parse_calibration_basis(basis)
    return f"from_{parsed.value}"


def label_ids_from_image(labels_img) -> list[int]:
    """Extract sorted non-negative label ids from an image-like labels container."""

    return sorted([int(label) for label in np.unique(labels_img.img) if label >= 0])

