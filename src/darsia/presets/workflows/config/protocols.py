"""Protocol configuration for the setup."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .utils import _get_section_from_toml

logger = logging.getLogger(__name__)


@dataclass
class ProtocolsConfig:
    """Protocol configuration for the setup."""

    imaging: dict[Path, Path | tuple[Path, str]] | None = field(
        default=None,
        metadata={
            "name": "Imaging protocol",
            "help": (
                "Table mapping each data folder to its imaging-protocol file, "
                "or [file, sheet]. Always a table — one row per folder, even "
                "with a single folder. Note: folder paths must exactly match "
                "those in [data].folders; use the 'Browse' button."
            ),
            "widget": "path_map",
            "key_is_directory": True,
            "group": "Imaging",
        },
    )
    """Per-folder mapping from data folder to imaging protocol file, or (file, sheet)."""
    blacklist: Path | tuple[Path, str] | None = field(
        default=None,
        metadata={
            "name": "Blacklist",
            "help": "Path to a file listing images to exclude, or [file, sheet].",
            "widget": "file",
            "group": "Imaging",
        },
    )
    """Path to the blacklist protocol file or (file, sheet)."""
    imaging_mode: str = field(
        default="exif",
        metadata={
            "name": "Imaging mode",
            "help": "Datetime extraction mode for imaging protocol setup.",
            "options": ["exif", "ctime"],
            "group": "Imaging",
        },
    )
    """Datetime extraction mode for imaging protocol setup: 'exif' or 'ctime'."""
    injection: Path | tuple[Path, str] | None = field(
        default=None,
        metadata={
            "name": "Injection protocol",
            "help": "Path to the injection-protocol file, or [file, sheet].",
            "widget": "file",
            "group": "Experiment",
        },
    )
    """Path to the injection protocol file or (file, sheet)."""
    pressure_temperature: Path | tuple[Path, str] | None = field(
        default=None,
        metadata={
            "name": "Pressure/Temperature protocol",
            "help": "Path to the pressure-temperature protocol file, or [file, sheet].",
            "widget": "file",
            "group": "Experiment",
        },
    )
    """Path to the pressure-temperature protocol file or (file, sheet)."""

    def _parse_protocol_value(
        self, value: str | Path | list[str] | tuple[str, str]
    ) -> Path | tuple[Path, str]:
        if isinstance(value, (list, tuple)):
            return (Path(value[0]), value[1])
        if isinstance(value, (str, Path)):
            return Path(value)
        raise ValueError(
            "Protocol value must be a string, Path, or a list of [path, sheet]."
        )

    def load(self, path: Path) -> "ProtocolsConfig":
        sec = _get_section_from_toml(path, "protocols")
        try:
            imaging_protocol = sec["imaging"]
            if not isinstance(imaging_protocol, dict):
                raise ValueError(
                    "[protocols].imaging must be a per-folder table:\n"
                    '[protocols.imaging]\n"<folder>" = "<path>" or ["<path>", "<sheet>"]\n'
                    "A bare scalar value is no longer supported."
                )
            self.imaging = {
                Path(folder): self._parse_protocol_value(protocol)
                for folder, protocol in imaging_protocol.items()
            }

        except KeyError:
            self.imaging = None

        try:
            injection_protocol = sec["injection"]
            if isinstance(injection_protocol, str) and not injection_protocol.strip():
                self.injection = None
            else:
                self.injection = self._parse_protocol_value(injection_protocol)
        except KeyError:
            self.injection = None

        try:
            blacklist_protocol = sec["blacklist"]
            if isinstance(blacklist_protocol, str) and not blacklist_protocol.strip():
                self.blacklist = None
            else:
                self.blacklist = self._parse_protocol_value(blacklist_protocol)
        except KeyError:
            self.blacklist = None

        try:
            pressure_temperature_protocol = sec["pressure_temperature"]
            if (
                isinstance(pressure_temperature_protocol, str)
                and not pressure_temperature_protocol.strip()
            ):
                self.pressure_temperature = None
            else:
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
        raise ValueError("Use [protocols] in the config file to load protocols.")
