"""Build and apply patch-wise colour statistics from config."""

import logging

import darsia
from darsia.presets.workflows.config.color_statistics import ColorStatisticsConfig
from darsia.signals.color.statistics import (
    ArmStatistics,
    ColorStatistics,
    NormStatistics,
    PatchedColorStatistics,
)

logger = logging.getLogger(__name__)


def build_color_statistics(config: ColorStatisticsConfig) -> PatchedColorStatistics:
    """Patch-wise colour statistics of the configured kind.

    Parameters
    ----------
    config : ColorStatisticsConfig
        Configuration.

    Returns
    -------
    PatchedColorStatistics
        Statistics, not fitted yet.

    """
    statistics: ColorStatistics
    if config.statistic == "arms":
        statistics = ArmStatistics(
            dark=config.dark,
            bright=config.bright,
            side=config.side,
            noise_percentile=config.noise_percentile,
            key_percentile=config.key_percentile,
        )
    else:
        statistics = NormStatistics(
            noise_percentile=config.noise_percentile,
            key_percentile=config.key_percentile,
        )
    return PatchedColorStatistics(
        statistics,
        patches=config.patches,
        min_pixels=config.min_pixels,
        smoothing=config.smoothing,
    )


def color_statistics_map(
    image: darsia.Image, labels: darsia.Image, config: ColorStatisticsConfig
) -> darsia.Image:
    """Score every pixel against the local colour statistics of its label.

    Parameters
    ----------
    image : darsia.Image
        Colour image.
    labels : darsia.Image
        Labels on the grid of the image.
    config : ColorStatisticsConfig
        Configuration.

    Returns
    -------
    darsia.Image
        Values in [0, 1] (float32): 1 typical colour, 0 key colour or beyond.

    """
    statistics = build_color_statistics(config)
    with darsia.timed(f"colour statistics fit ({config.statistic})"):
        statistics.fit(image, labels)
    for label, info in statistics.summary().items():
        logger.info(f"Colour statistics, label {label}: {info}")
    with darsia.timed(f"colour statistics evaluation ({config.statistic})"):
        return statistics.evaluate(image, labels)
