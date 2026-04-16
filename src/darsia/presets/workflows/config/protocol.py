"""Protocol configuration for the setup."""

import logging
from dataclasses import dataclass
from pathlib import Path

from .utils import _get_section_from_toml

logger = logging.getLogger(__name__)


@dataclass
class ProtocolConfig:
    """Protocol configuration for the setup."""

    imaging: (
        Path | tuple[Path, str] | dict[Path, Path | tuple[Path, str]] | None
    ) = None
    """Path to imaging protocol, (file, sheet), or per-folder mapping."""
    injection: Path | tuple[Path, str] | None = None
    """Path to the injection protocol file or (file, sheet)."""
    blacklist: Path | tuple[Path, str] | None = None
    """Path to the blacklist protocol file or (file, sheet)."""
    pressure_temperature: Path | tuple[Path, str] | None = None
    """Path to the pressure-temperature protocol file or (file, sheet)."""
    imaging_mode: str = "exif"
    """Datetime extraction mode for imaging protocol setup: 'exif' or 'ctime'."""

    def _parse_protocol_value(
        self, value: str | list[str] | tuple[str, str]
    ) -> Path | tuple[Path, str]:
        if isinstance(value, list) or isinstance(value, tuple):
            return (Path(value[0]), value[1])
        if isinstance(value, str):
            return Path(value)
        raise ValueError("Protocol value must be a string or a list of [path, sheet].")

    def load(self, path: Path) -> "ProtocolConfig":
        sec = _get_section_from_toml(path, "protocols")
        try:
            imaging_protocol = sec["imaging"]
            if isinstance(imaging_protocol, dict):
                self.imaging = {
                    Path(folder): self._parse_protocol_value(protocol)
                    for folder, protocol in imaging_protocol.items()
                }
            else:
                self.imaging = self._parse_protocol_value(imaging_protocol)

        except KeyError:
            self.imaging = None

        try:
            injection_protocol = sec["injection"]
            self.injection = self._parse_protocol_value(injection_protocol)
        except KeyError:
            self.injection = None

        try:
            blacklist_protocol = sec["blacklist"]
            self.blacklist = self._parse_protocol_value(blacklist_protocol)
        except KeyError:
            self.blacklist = None

        try:
            pressure_temperature_protocol = sec["pressure_temperature"]
            self.pressure_temperature = self._parse_protocol_value(
                pressure_temperature_protocol
            )
        except KeyError:
            self.pressure_temperature = None

        self.imaging_mode = str(
            sec.get("imaging_mode", sec.get("mode", "exif"))
        ).lower()
        if self.imaging_mode not in {"exif", "ctime"}:
            raise ValueError(
                "Imaging mode must be either 'exif' or 'ctime' via "
                "[protocols].imaging_mode."
            )

        return self

    def error(self):
        raise ValueError(f"Use [protocols] in the config file to load protocols.")
