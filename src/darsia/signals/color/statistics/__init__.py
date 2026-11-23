"""Colour statistics: characteristic colours of pixel groups and scores against them.

`ColorStatistics` subclasses characterise a colour distribution (`ArmStatistics`: dark
and bright arm along the main colour axis plus side distance; `NormStatistics`:
distance from the median colour). `PatchedColorStatistics` fits any of them per patch
and label and interpolates the parameters smoothly over the image.
"""

from .arms import ArmStatistics
from .base import ColorStatistics
from .norm import NormStatistics
from .patched import PatchedColorStatistics

__all__ = [
    "ArmStatistics",
    "ColorStatistics",
    "NormStatistics",
    "PatchedColorStatistics",
]
