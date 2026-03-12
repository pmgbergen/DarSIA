"""Interface to configure restoration methods."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .utils import _get_key, _get_section_from_toml


@dataclass
class VolumeAveragingConfig:
    rev_size: int = 3

    def load(self, sec: dict) -> "VolumeAveragingConfig":
        self.rev_size = _get_key(sec, "rev_size", self.rev_size)
        return self


@dataclass
class RestorationConfig:
    method: Literal["volume_average"] | None = "volume_average"
    options: VolumeAveragingConfig | None = None

    def load(self, path: Path) -> "RestorationConfig":
        sec = _get_section_from_toml(path, "restoration")
        self.method = _get_key(sec, "method", required=True, type_=str)

        options_sec = sec.get("options", {})
        if self.method == "none":
            self.options = None
        elif self.method == "volume_average":
            self.options = VolumeAveragingConfig().load(options_sec)
        else:
            raise NotImplementedError(
                f"Restoration method {self.method} not supported."
            )
        return self
