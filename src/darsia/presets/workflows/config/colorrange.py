"""Configuration for named color ranges used in analysis modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .utils import _convert_none, _get_section_from_toml


@dataclass
class NamedColorRangeConfig:
    """Configuration for one named color range."""

    color_space: str = "RGB"
    ranges: list[tuple[float | None, float | None]] = field(default_factory=list)

    SUPPORTED_COLOR_SPACES = {"RGB", "BGR", "HSV", "HLS", "LAB"}

    def load(self, sec: dict, name: str) -> "NamedColorRangeConfig":
        color_space = sec.get("color_space", self.color_space)
        if not isinstance(color_space, str):
            raise ValueError(f"colorrange.{name}.color_space must be a string.")
        self.color_space = color_space.upper().strip()
        if self.color_space not in self.SUPPORTED_COLOR_SPACES:
            raise ValueError(
                f"Unsupported colorrange.{name}.color_space '{self.color_space}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_COLOR_SPACES))}."
            )

        raw_range = sec.get("range")
        if not isinstance(raw_range, list) or len(raw_range) != 3:
            raise ValueError(
                f"colorrange.{name}.range must be a list of 3 [min, max] bounds."
            )

        parsed: list[tuple[float | None, float | None]] = []
        for i, bound in enumerate(raw_range):
            if not isinstance(bound, list) or len(bound) != 2:
                raise ValueError(
                    f"colorrange.{name}.range[{i}] must be [min, max] with two entries."
                )
            lower_raw = _convert_none(bound[0])
            upper_raw = _convert_none(bound[1])
            lower = None if lower_raw is None else float(lower_raw)
            upper = None if upper_raw is None else float(upper_raw)
            # Allow wrap-around intervals for hue channels only; this check is applied
            # in runtime where channel context is available.
            parsed.append((lower, upper))

        self.ranges = parsed
        return self


@dataclass
class ColorRangeConfig:
    """Configuration container for all named color ranges."""

    ranges: dict[str, NamedColorRangeConfig] = field(default_factory=dict)

    def load(self, path: Path | list[Path]) -> "ColorRangeConfig":
        sec = _get_section_from_toml(path, "colorrange")
        if not isinstance(sec, dict):
            raise ValueError("[colorrange] must be a table.")

        self.ranges = {}
        for key in sec.keys():
            self.ranges[key] = NamedColorRangeConfig().load(sec[key], key)
        return self
