"""Configuration registry for named color channels used in analysis modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .utils import _get_section_from_toml


@dataclass
class NamedColorChannelConfig:
    """Configuration for one named color channel."""

    color_space: str = "RGB"
    channel: str = "r"

    SUPPORTED_CHANNELS = {
        "RGB": {"r", "g", "b"},
        "BGR": {"r", "g", "b"},
        "HSV": {"h", "s", "v"},
        "HLS": {"h", "l", "s"},
        "LAB": {"l", "a", "b"},
    }

    def load(self, sec: dict, name: str) -> "NamedColorChannelConfig":
        color_space = sec.get("color_space", self.color_space)
        if not isinstance(color_space, str):
            raise ValueError(f"colorchannel.{name}.color_space must be a string.")
        self.color_space = color_space.upper().strip()

        if self.color_space not in self.SUPPORTED_CHANNELS:
            raise ValueError(
                f"Unsupported colorchannel.{name}.color_space '{self.color_space}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_CHANNELS))}."
            )

        channel = sec.get("channel", self.channel)
        if not isinstance(channel, str):
            raise ValueError(f"colorchannel.{name}.channel must be a string.")
        self.channel = channel.lower().strip()
        allowed_channels = self.SUPPORTED_CHANNELS[self.color_space]
        if self.channel not in allowed_channels:
            raise ValueError(
                f"Unsupported colorchannel.{name}.channel '{self.channel}' for "
                f"color space '{self.color_space}'. Supported: "
                f"{', '.join(sorted(allowed_channels))}."
            )
        return self


@dataclass
class ColorChannelRegistry:
    """Configuration container for all named color channels."""

    channels: dict[str, NamedColorChannelConfig] = field(default_factory=dict)

    def load(self, path: Path | list[Path]) -> "ColorChannelRegistry":
        sec = _get_section_from_toml(path, "colorchannel")
        if not isinstance(sec, dict):
            raise ValueError("[colorchannel] must be a table.")

        self.channels = {}
        for key in sec.keys():
            self.channels[key] = NamedColorChannelConfig().load(sec[key], key)
        return self

    def resolve(self, keys: str | list[str]) -> dict[str, NamedColorChannelConfig]:
        if isinstance(keys, str):
            keys = [keys]
        result = {}
        for key in keys:
            if key not in self.channels:
                available = sorted(self.channels.keys())
                raise KeyError(
                    f"ColorChannelRegistry: key '{key}' not found. "
                    f"Available keys: {available}"
                )
            result[key] = self.channels[key]
        return result

    def keys(self) -> list[str]:
        return sorted(self.channels.keys())
