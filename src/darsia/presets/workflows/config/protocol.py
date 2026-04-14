"""Protocol configuration for the setup."""

import logging
from dataclasses import dataclass
from pathlib import Path

from .utils import _get_section_from_toml

logger = logging.getLogger(__name__)


@dataclass
class ProtocolConfig:
    """Protocol configuration for the setup."""

    imaging: Path | tuple[Path, str] | None = None
    """Path to the imaging protocol file or (file, sheet)."""
    injection: Path | tuple[Path, str] | None = None
    """Path to the injection protocol file or (file, sheet)."""
    blacklist: Path | tuple[Path, str] | None = None
    """Path to the blacklist protocol file or (file, sheet)."""
    pressure_temperature: Path | tuple[Path, str] | None = None
    """Path to the pressure-temperature protocol file or (file, sheet)."""
    imaging_mode: str = "exif"
    """Datetime extraction mode for imaging protocol setup: 'exif' or 'ctime'."""

    def load(self, path: Path) -> "ProtocolConfig":
        sec = _get_section_from_toml(path, "protocols")
        try:
            imaging_protocol = sec["imaging"]
            if isinstance(imaging_protocol, list):
                self.imaging = (Path(imaging_protocol[0]), imaging_protocol[1])
            elif isinstance(imaging_protocol, str):
                self.imaging = Path(imaging_protocol)
            else:
                raise ValueError(
                    "Imaging protocol must be a string or a list of [path, sheet]."
                )

        except KeyError:
            self.imaging = None

        try:
            injection_protocol = sec["injection"]
            if isinstance(injection_protocol, list):
                self.injection = (Path(injection_protocol[0]), injection_protocol[1])
            elif isinstance(injection_protocol, str):
                self.injection = Path(injection_protocol)
            else:
                raise ValueError(
                    "Injection protocol must be a string or a list of [path, sheet]."
                )
        except KeyError:
            self.injection = None

        try:
            blacklist_protocol = sec["blacklist"]
            if isinstance(blacklist_protocol, list):
                self.blacklist = (Path(blacklist_protocol[0]), blacklist_protocol[1])
            elif isinstance(blacklist_protocol, str):
                self.blacklist = Path(blacklist_protocol)
            else:
                raise ValueError(
                    "Blacklist protocol must be a string or a list of [path, sheet]."
                )
        except KeyError:
            self.blacklist = None

        try:
            pressure_temperature_protocol = sec["pressure_temperature"]
            if isinstance(pressure_temperature_protocol, list):
                self.pressure_temperature = (
                    Path(pressure_temperature_protocol[0]),
                    pressure_temperature_protocol[1],
                )
            elif isinstance(pressure_temperature_protocol, str):
                self.pressure_temperature = Path(pressure_temperature_protocol)
            else:
                raise ValueError(
                    """Pressure-temperature protocol must be a string or a list of """
                    """[path, sheet]."""
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
