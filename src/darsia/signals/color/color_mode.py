"""Centralized definition of color modes."""

from enum import StrEnum


class ColorMode(StrEnum):
    """Whether a colour signal is treated as absolute or relative to a baseline."""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"
