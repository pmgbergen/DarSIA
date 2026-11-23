"""Configuration of patch-wise colour statistics (`darsia.signals.color.statistics`).

Used as `[image_porosity.color_statistics]`; usable as a sub-table of other sections.
"""

from dataclasses import dataclass, field
from typing import Literal, get_args

from .utils import load_choice

ColorStatisticName = Literal["arms", "norm"]
"""Statistics: dark/bright arms along the main colour axis plus side distance, or the
distance from the median colour."""

PERCENTILE_RANGE = (0.0, 100.0)
"""Allowed range of the noise and key percentile (percent, not a fraction)."""
MIN_PERCENTILE_GAP = 1.0
"""Minimum gap between noise and key percentile."""


@dataclass
class ColorStatisticsConfig:
    """Colour statistics per patch and label, interpolated smoothly over the image.

    Config section (TOML)::

        statistic = "arms"        # or "norm"
        patches = [20, 20]
        min_pixels = 2000
        smoothing = 0.0           # in patches
        noise_percentile = 90.0   # 0-100; local distances below: value 1
        key_percentile = 99.5     # 0-100, >= noise + 1; above: value 0
        dark = true               # arms only
        bright = true
        side = true

    """

    statistic: ColorStatisticName = field(
        default="arms",
        metadata={
            "name": "Statistic",
            "help": (
                "'arms': distance from the typical colour along the main colour axis, "
                "darker (dark arm) or brighter (bright arm), and off the axis (side). "
                "'norm': distance from the typical colour in any direction."
            ),
            "options": list(get_args(ColorStatisticName)),
        },
    )
    """Statistic, one of `ColorStatisticName`."""
    patches: tuple[int, int] = field(
        default=(20, 20),
        metadata={
            "name": "Patches",
            "help": (
                "Number of patches (rows, cols). The statistics are fitted per patch "
                "and label and interpolated smoothly, following illumination trends."
            ),
        },
    )
    """Number of patches ``(rows, cols)``."""
    min_pixels: int = field(
        default=2000,
        metadata={
            "name": "Minimum pixels",
            "help": (
                "Pixels of a label a patch needs for its own fit; patches with fewer "
                "take the values of their neighbours."
            ),
        },
    )
    """Pixels of a label a patch needs for its own fit."""
    smoothing: float = field(
        default=0.0,
        metadata={
            "name": "Smoothing (patches)",
            "help": (
                "Gaussian smoothing of the patch values, in patches, against noisy "
                "patches. 0 (default): the fields pass through the patch values. "
                "Smoothing flattens trends towards the image borders."
            ),
        },
    )
    """Gaussian standard deviation on the patch grid, in patches."""
    noise_percentile: float = field(
        default=90.0,
        metadata={
            "name": "Noise percentile (%)",
            "help": (
                "Percentile (0-100, not a fraction) of the local distances from the "
                "typical colour of a region in a patch. Colours closer than this count "
                "as typical (value 1); e.g. 90: the closest 90 % of the pixels. With "
                "'arms' it applies to the dark arm, the bright arm and the side "
                "separately. At least 1 below the key percentile."
            ),
            "range": PERCENTILE_RANGE,
            "less_than": {"field": "key_percentile", "gap": MIN_PERCENTILE_GAP},
        },
    )
    """Percentile of the local distances below which the value is 1."""
    key_percentile: float = field(
        default=99.5,
        metadata={
            "name": "Key percentile (%)",
            "help": (
                "Percentile (0-100, not a fraction) of the same distances. Colours "
                "farther than this get value 0, linear in between; e.g. 99.5: the "
                "farthest 0.5 % of the pixels. At least 1 above the noise percentile."
            ),
            "range": PERCENTILE_RANGE,
        },
    )
    """Percentile of the local distances from which on the value is 0."""
    dark: bool = field(
        default=True,
        metadata={
            "name": "Dark arm",
            "help": "Count colours darker than the typical colour (dark grains).",
            "depends_on": {"field": "statistic", "value": "arms"},
        },
    )
    """Whether the dark arm counts (arms only)."""
    bright: bool = field(
        default=True,
        metadata={
            "name": "Bright arm",
            "help": (
                "Count colours brighter than the typical colour (bright grains, "
                "reflections)."
            ),
            "depends_on": {"field": "statistic", "value": "arms"},
        },
    )
    """Whether the bright arm counts (arms only)."""
    side: bool = field(
        default=True,
        metadata={
            "name": "Side",
            "help": (
                "Count colours of another hue than the typical colour (off the main "
                "colour axis, e.g. orange or greenish grains)."
            ),
            "depends_on": {"field": "statistic", "value": "arms"},
        },
    )
    """Whether the side distance counts (arms only)."""

    def load(
        self, sec: dict, context: str = "[color_statistics]"
    ) -> "ColorStatisticsConfig":
        """Populate from a TOML table.

        Raises
        ------
        ValueError
            For an unknown statistic, invalid patches, percentiles or smoothing, or
            arms with all terms switched off.

        """
        self.statistic = load_choice(
            sec, "statistic", self.statistic, ColorStatisticName, context
        )
        patches = sec.get("patches", list(self.patches))
        if len(patches) != 2 or min(int(p) for p in patches) < 1:
            raise ValueError(
                f"{context} patches must be 2 positive integers, got {patches!r}"
            )
        self.patches = (int(patches[0]), int(patches[1]))
        self.min_pixels = int(sec.get("min_pixels", self.min_pixels))
        self.smoothing = float(sec.get("smoothing", self.smoothing))
        if self.smoothing < 0:
            raise ValueError(f"{context} smoothing must be >= 0, got {self.smoothing}")
        self.noise_percentile = float(
            sec.get("noise_percentile", self.noise_percentile)
        )
        self.key_percentile = float(sec.get("key_percentile", self.key_percentile))
        low, high = PERCENTILE_RANGE
        if not (
            low <= self.noise_percentile
            and self.noise_percentile + MIN_PERCENTILE_GAP <= self.key_percentile
            and self.key_percentile <= high
        ):
            raise ValueError(
                f"{context} needs {low:g} <= noise_percentile, noise_percentile + "
                f"{MIN_PERCENTILE_GAP:g} <= key_percentile <= {high:g} (percent), got "
                f"{self.noise_percentile} and {self.key_percentile}"
            )
        self.dark = bool(sec.get("dark", self.dark))
        self.bright = bool(sec.get("bright", self.bright))
        self.side = bool(sec.get("side", self.side))
        if self.statistic == "arms" and not (self.dark or self.bright or self.side):
            raise ValueError(f"{context} needs at least one of dark, bright, side.")
        return self
