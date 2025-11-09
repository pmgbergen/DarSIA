"""Centralized definition of color modes."""

from enum import StrEnum


class ColorMode(StrEnum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
