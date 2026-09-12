"""Protocol configuration for the setup."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .utils import _get_section_from_toml

logger = logging.getLogger(__name__)

_SUPPORTED_IMAGING_MODES = {"exif", "ctime", "interval", "detailed"}
_SUPPORTED_START_REFERENCES = {"first_image", "baseline", "fixed", "image"}
_SUPPORTED_PRESSURE_TEMPERATURE_MODES = {"constant", "detailed"}
_SUPPORTED_INJECTION_MODES = {"constant", "detailed"}


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
                "with a single folder. The folder column mirrors [data].folders "
                "and is not editable here; add or remove folders in the Data tab."
            ),
            "widget": "path_map",
            "key_source": "data.folders",
            "group": "Imaging",
        },
    )
    """Per-folder mapping from data folder to imaging protocol file, or (file, sheet)."""
    blacklist: Path | tuple[Path, str] | dict[Path, Path | tuple[Path, str]] | None = (
        field(
            default=None,
            metadata={
                "name": "Blacklist",
                "help": (
                    "Table mapping each data folder to its blacklist file (listing "
                    "images to exclude), or [file, sheet]. A folder with no entry "
                    "has no blacklist. The folder column mirrors [data].folders and "
                    "is not editable here; add or remove folders in the Data tab."
                ),
                "widget": "path_map",
                "key_source": "data.folders",
                "group": "Imaging",
            },
        )
    )
    """Per-folder mapping from data folder to blacklist file, or (file, sheet).
    A single bare value is also accepted for single-folder configs."""
    imaging_mode: str = field(
        default="exif",
        metadata={
            "name": "Imaging mode",
            "help": (
                "Datetime extraction mode for imaging protocol setup. 'exif'/"
                "'ctime' read each image's real timestamp (auto from metadata); "
                "'interval' computes timestamps from a prescribed cadence instead "
                "(see 'Imaging interval'); 'detailed' leaves an existing imaging "
                "protocol file alone (author it by hand instead)."
            ),
            "options": ["exif", "ctime", "interval", "detailed"],
            "group": "Imaging",
        },
    )
    """Datetime extraction mode for imaging protocol setup: 'exif', 'ctime', or
    'interval' (prescribed time increments, no per-file reads)."""
    imaging_interval_seconds: dict[Path, float] | None = field(
        default=None,
        metadata={
            "name": "Imaging interval",
            "help": (
                "Table mapping each data folder to a fixed imaging interval in "
                "seconds, used when Imaging mode is 'interval'. One uniform "
                "interval per folder. The folder column mirrors [data].folders "
                "and is not editable here; add or remove folders in the Data tab."
            ),
            "widget": "number_map",
            "key_source": "data.folders",
            "group": "Imaging",
            "depends_on": {"field": "imaging_mode", "value": "interval"},
        },
    )
    """Per-folder fixed imaging interval in seconds (interval mode only)."""
    start_reference: str = field(
        default="first_image",
        metadata={
            "name": "Start reference",
            "help": (
                "What counts as time zero when anchoring interval-mode imaging "
                "and the injection/pressure-temperature templates: the first "
                "image in each folder, the configured baseline image, a fixed "
                "datetime, or a specific reference image."
            ),
            "options": ["first_image", "baseline", "fixed", "image"],
            "group": "Imaging",
        },
    )
    """What counts as time zero: 'first_image', 'baseline', 'fixed', or 'image'."""
    start_reference_fixed: str | None = field(
        default=None,
        metadata={
            "name": "Fixed start time",
            "help": "ISO-8601 datetime to use as time zero (start_reference='fixed').",
            "placeholder": "e.g., 2023-10-31 11:08:28",
            "depends_on": {"field": "start_reference", "value": "fixed"},
            "group": "Imaging",
        },
    )
    """ISO-8601 datetime string used as time zero when start_reference='fixed'."""
    start_reference_image: Path | None = field(
        default=None,
        metadata={
            "name": "Reference image",
            "help": "Image file to use as time zero (start_reference='image').",
            "widget": "file",
            "depends_on": {"field": "start_reference", "value": "image"},
            "group": "Imaging",
        },
    )
    """Reference image used as time zero when start_reference='image'."""
    injection: Path | tuple[Path, str] | None = field(
        default=None,
        metadata={
            "name": "Injection protocol",
            "help": "Path to the injection-protocol file, or [file, sheet].",
            "widget": "file",
            "group": "Experiment",
            "table_viewer": "csv",
        },
    )
    """Path to the injection protocol file or (file, sheet)."""
    injection_mode: str = field(
        default="constant",
        metadata={
            "name": "Injection mode",
            "help": (
                "'constant' writes a single-row protocol from the rate and "
                "coordinates below, spanning the whole run; 'detailed' leaves an "
                "existing injection-protocol file alone, or writes an empty "
                "(header-only) template if none exists yet."
            ),
            "options": ["constant", "detailed"],
            "group": "Experiment",
        },
    )
    """Injection template mode: 'constant' or 'detailed'."""
    injection_rate: float = field(
        default=0.0,
        metadata={
            "name": "Injection rate",
            "help": "Constant injection rate written to the injection template.",
            "depends_on": {"field": "injection_mode", "value": "constant"},
            "group": "Experiment",
        },
    )
    """Constant injection rate (injection_mode='constant' only)."""
    injection_coordinates: tuple[float, float] = field(
        default=(0.0, 0.0),
        metadata={
            "name": "Injection coordinates",
            "help": "Constant injection coordinates written to the injection template.",
            "depends_on": {"field": "injection_mode", "value": "constant"},
            "group": "Experiment",
        },
    )
    """Constant injection coordinates (injection_mode='constant' only)."""
    pressure_temperature: Path | tuple[Path, str] | None = field(
        default=None,
        metadata={
            "name": "Pressure/Temperature protocol",
            "help": "Path to the pressure-temperature protocol file, or [file, sheet].",
            "widget": "file",
            "group": "Experiment",
            "table_viewer": "csv",
        },
    )
    """Path to the pressure-temperature protocol file or (file, sheet)."""
    pressure_temperature_mode: str = field(
        default="constant",
        metadata={
            "name": "Pressure/Temperature mode",
            "help": (
                "'constant' writes a single-row template from the values below; "
                "'detailed' leaves the pressure-temperature protocol file alone "
                "(author/point to a full custom CSV instead)."
            ),
            "options": ["constant", "detailed"],
            "group": "Experiment",
        },
    )
    """Pressure/temperature template mode: 'constant' or 'detailed'."""
    pressure_bar: float = field(
        default=1.013,
        metadata={
            "name": "Pressure (bar)",
            "help": "Constant pressure written to the pressure-temperature template.",
            "depends_on": {"field": "pressure_temperature_mode", "value": "constant"},
            "group": "Experiment",
        },
    )
    """Constant pressure in bar (pressure_temperature_mode='constant' only)."""
    temperature_celsius: float = field(
        default=20.0,
        metadata={
            "name": "Temperature (C)",
            "help": "Constant temperature written to the pressure-temperature template.",
            "depends_on": {"field": "pressure_temperature_mode", "value": "constant"},
            "group": "Experiment",
        },
    )
    """Constant temperature in Celsius (pressure_temperature_mode='constant' only)."""

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
            if isinstance(blacklist_protocol, dict):
                self.blacklist = {
                    Path(folder): self._parse_protocol_value(protocol)
                    for folder, protocol in blacklist_protocol.items()
                }
            elif isinstance(blacklist_protocol, str) and not blacklist_protocol.strip():
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
        if self.imaging_mode not in _SUPPORTED_IMAGING_MODES:
            raise ValueError(
                "Imaging mode must be one of "
                f"{sorted(_SUPPORTED_IMAGING_MODES)} via [protocols].imaging_mode."
            )

        imaging_interval_seconds = sec.get("imaging_interval_seconds")
        if isinstance(imaging_interval_seconds, dict) and imaging_interval_seconds:
            self.imaging_interval_seconds = {
                Path(folder): float(value)
                for folder, value in imaging_interval_seconds.items()
            }
        else:
            self.imaging_interval_seconds = None

        self.start_reference = str(sec.get("start_reference", "first_image")).lower()
        if self.start_reference not in _SUPPORTED_START_REFERENCES:
            raise ValueError(
                "Start reference must be one of "
                f"{sorted(_SUPPORTED_START_REFERENCES)} via [protocols].start_reference."
            )

        start_reference_fixed = sec.get("start_reference_fixed")
        self.start_reference_fixed = (
            str(start_reference_fixed).strip()
            if start_reference_fixed and str(start_reference_fixed).strip()
            else None
        )

        start_reference_image = sec.get("start_reference_image")
        self.start_reference_image = (
            Path(start_reference_image)
            if start_reference_image and str(start_reference_image).strip()
            else None
        )

        self.injection_mode = str(sec.get("injection_mode", "constant")).lower()
        if self.injection_mode not in _SUPPORTED_INJECTION_MODES:
            raise ValueError(
                "Injection mode must be one of "
                f"{sorted(_SUPPORTED_INJECTION_MODES)} via [protocols].injection_mode."
            )

        self.injection_rate = float(sec.get("injection_rate", 0.0))

        injection_coordinates = sec.get("injection_coordinates", (0.0, 0.0))
        if len(injection_coordinates) != 2:
            raise ValueError(
                "[protocols].injection_coordinates must have exactly 2 entries "
                f"(x, y), got {injection_coordinates!r}."
            )
        self.injection_coordinates = tuple(
            float(value) for value in injection_coordinates
        )

        self.pressure_temperature_mode = str(
            sec.get("pressure_temperature_mode", "constant")
        ).lower()
        if self.pressure_temperature_mode not in _SUPPORTED_PRESSURE_TEMPERATURE_MODES:
            raise ValueError(
                "Pressure/temperature mode must be one of "
                f"{sorted(_SUPPORTED_PRESSURE_TEMPERATURE_MODES)} via "
                "[protocols].pressure_temperature_mode."
            )

        self.pressure_bar = float(sec.get("pressure_bar", 1.013))
        self.temperature_celsius = float(sec.get("temperature_celsius", 20.0))

        return self

    def error(self):
        raise ValueError("Use [protocols] in the config file to load protocols.")
