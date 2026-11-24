"""Colour statistics from config (`ColorStatisticsConfig`).

Builds the colour statistics of `darsia.signals.color.statistics` from a config and
applies them to an image with labels, e.g. as the image porosity method
`"color_statistics"`.
"""

from .color_statistics import build_color_statistics, color_statistics_map

__all__ = ["build_color_statistics", "color_statistics_map"]
