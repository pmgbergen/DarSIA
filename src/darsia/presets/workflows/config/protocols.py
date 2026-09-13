"""Protocol configuration for the setup.

DarSIA reads three protocols — imaging, injection and pressure/temperature — and
each one owns its own fields, defaults and parsing here. :class:`ProtocolsConfig`
gathers all three.

They are *mixed in* rather than nested, which keeps the TOML flat under a single
``[protocols]`` table. Nesting would collide with ``[protocols.imaging]``, which is
already the per-folder imaging-protocol table, and would break every existing run
config. Base classes are therefore listed in reverse order, since a dataclass
collects inherited fields in reverse-MRO order and the field order drives the GUI
layout.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .utils import _get_section_from_toml, _normalize_mode

logger = logging.getLogger(__name__)

_SUPPORTED_IMAGING_MODES = {"exif", "ctime", "interval", "Load from CSV"}
_SUPPORTED_START_REFERENCES = {"first_image", "baseline", "fixed", "image"}
_SUPPORTED_PRESSURE_TEMPERATURE_MODES = {"constant", "Load from CSV"}
_SUPPORTED_INJECTION_MODES = {"constant", "Load from CSV"}
_SUPPORTED_INJECTION_RATE_UNITS = {
    "kg/s",
    "g/s",
    "g/min",
    "g/hr",
    "mL/s",
    "mL/min",
    "mL/hr",
}
# Volumetric units need a fluid density to convert to the kg/s DarSIA uses internally.
_VOLUMETRIC_INJECTION_RATE_UNITS = ["mL/s", "mL/min", "mL/hr"]


def _parse_protocol_value(
    value: str | Path | list[str] | tuple[str, str],
) -> Path | tuple[Path, str]:
    """Normalize a protocol entry to a path, or a (path, sheet) pair."""
    if isinstance(value, (list, tuple)):
        return (Path(value[0]), value[1])
    if isinstance(value, (str, Path)):
        return Path(value)
    raise ValueError(
        "Protocol value must be a string, Path, or a list of [path, sheet]."
    )


@dataclass
class ImagingProtocolConfig:
    """Which image was taken when: the imaging protocol and how to build it.

    Also owns the start reference, since that is a statement about the imaging
    timeline — the injection and pressure/temperature templates are anchored to it.
    """

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
            "group": "Image registry",
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
                "group": "Image registry",
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
                "(see 'Imaging interval'); 'Load from CSV' leaves an existing "
                "imaging protocol file alone (author it by hand instead)."
            ),
            "options": ["exif", "ctime", "interval", "Load from CSV"],
            "group": "Image registry",
        },
    )
    """Datetime extraction mode for imaging protocol setup: 'exif', 'ctime',
    'interval' (prescribed time increments, no per-file reads), or 'Load from CSV'."""
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
            "group": "Image registry",
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
            "group": "Image registry",
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
            "group": "Image registry",
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
            "group": "Image registry",
        },
    )
    """Reference image used as time zero when start_reference='image'."""

    def load_imaging(self, sec: dict) -> None:
        """Read the imaging keys from a flat [protocols] section."""
        try:
            imaging_protocol = sec["imaging"]
            if not isinstance(imaging_protocol, dict):
                raise ValueError(
                    "[protocols].imaging must be a per-folder table:\n"
                    '[protocols.imaging]\n"<folder>" = "<path>" or ["<path>", "<sheet>"]\n'
                    "A bare scalar value is no longer supported."
                )
            self.imaging = {
                Path(folder): _parse_protocol_value(protocol)
                for folder, protocol in imaging_protocol.items()
            }
        except KeyError:
            self.imaging = None

        try:
            blacklist_protocol = sec["blacklist"]
            if isinstance(blacklist_protocol, dict):
                self.blacklist = {
                    Path(folder): _parse_protocol_value(protocol)
                    for folder, protocol in blacklist_protocol.items()
                }
            elif isinstance(blacklist_protocol, str) and not blacklist_protocol.strip():
                self.blacklist = None
            else:
                self.blacklist = _parse_protocol_value(blacklist_protocol)
        except KeyError:
            self.blacklist = None

        self.imaging_mode = _normalize_mode(
            sec.get("imaging_mode", sec.get("mode", "exif")),
            _SUPPORTED_IMAGING_MODES,
            key="[protocols].imaging_mode",
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


@dataclass
class InjectionProtocolConfig:
    """When injection ran, where, and at what rate."""

    injection: Path | tuple[Path, str] | None = field(
        default=None,
        metadata={
            "name": "Injection protocol",
            "help": "Path to the injection-protocol file, or [file, sheet].",
            "widget": "file",
            "group": "Operating conditions",
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
                "coordinates below, spanning the whole run; 'Load from CSV' leaves "
                "an existing injection-protocol file alone, or writes an empty "
                "(header-only) template if none exists yet."
            ),
            "options": ["constant", "Load from CSV"],
            "group": "Operating conditions",
        },
    )
    """Injection template mode: 'constant' or 'Load from CSV'."""
    injection_rate_unit: str = field(
        default="kg/s",
        metadata={
            "name": "Injection rate unit",
            "help": (
                "Unit the rate below is entered in; picks the matching csv "
                "column for the constant template. mL/s, mL/min and mL/hr are "
                "volumetric and need the fluid density below to convert to "
                "kg/s. sccm is only accepted in a hand-authored 'Load from "
                "CSV' protocol file, where DarSIA converts any supported unit "
                "to kg/s on read."
            ),
            "options": sorted(_SUPPORTED_INJECTION_RATE_UNITS),
            "depends_on": {"field": "injection_mode", "value": "constant"},
            "group": "Operating conditions",
        },
    )
    """Unit injection_rate is entered in."""
    injection_rate: float = field(
        default=0.0,
        metadata={
            "name": "Injection rate",
            "help": (
                "Constant mass injection rate written to the injection "
                "template, in the unit selected above."
            ),
            "depends_on": {"field": "injection_mode", "value": "constant"},
            "group": "Operating conditions",
        },
    )
    """Constant injection rate, in injection_rate_unit (injection_mode='constant' only)."""
    injection_density: float = field(
        default=0.0,
        metadata={
            "name": "Injection fluid density (kg/m3)",
            "help": (
                "Density of the injected fluid, used to convert a volumetric "
                "injection rate (mL/s, mL/min or mL/hr) to the kg/s DarSIA "
                "uses internally. Ignored for mass-based units (kg/s, g/s, "
                "g/min, g/hr)."
            ),
            "depends_on": {
                "field": "injection_rate_unit",
                "value": _VOLUMETRIC_INJECTION_RATE_UNITS,
            },
            "group": "Operating conditions",
        },
    )
    """Injected fluid density in kg/m3 (only used for volumetric rate units)."""
    injection_coordinates: tuple[float, float] = field(
        default=(0.0, 0.0),
        metadata={
            "name": "Injection coordinates (m)",
            "help": (
                "Constant injection location (x, y) written to the injection "
                "template, in meters — Cartesian coordinates in the same physical "
                "frame as [rig].width/height, not pixels."
            ),
            "depends_on": {"field": "injection_mode", "value": "constant"},
            "group": "Operating conditions",
        },
    )
    """Constant injection (x, y) in meters (injection_mode='constant' only)."""

    def load_injection(self, sec: dict) -> None:
        """Read the injection keys from a flat [protocols] section."""
        try:
            injection_protocol = sec["injection"]
            if isinstance(injection_protocol, str) and not injection_protocol.strip():
                self.injection = None
            else:
                self.injection = _parse_protocol_value(injection_protocol)
        except KeyError:
            self.injection = None

        self.injection_mode = _normalize_mode(
            sec.get("injection_mode", "constant"),
            _SUPPORTED_INJECTION_MODES,
            key="[protocols].injection_mode",
        )

        self.injection_rate_unit = _normalize_mode(
            sec.get("injection_rate_unit", "kg/s"),
            _SUPPORTED_INJECTION_RATE_UNITS,
            key="[protocols].injection_rate_unit",
        )
        self.injection_rate = float(sec.get("injection_rate", 0.0))
        self.injection_density = float(sec.get("injection_density", 0.0))

        injection_coordinates = sec.get("injection_coordinates", (0.0, 0.0))
        if len(injection_coordinates) != 2:
            raise ValueError(
                "[protocols].injection_coordinates must have exactly 2 entries "
                f"(x, y), got {injection_coordinates!r}."
            )
        self.injection_coordinates = tuple(
            float(value) for value in injection_coordinates
        )


@dataclass
class PressureTemperatureProtocolConfig:
    """The pressure and temperature conditions over the run."""

    pressure_temperature: Path | tuple[Path, str] | None = field(
        default=None,
        metadata={
            "name": "Pressure/Temperature protocol",
            "help": "Path to the pressure-temperature protocol file, or [file, sheet].",
            "widget": "file",
            "group": "Experimental conditions",
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
                "'Load from CSV' leaves the pressure-temperature protocol file "
                "alone (author/point to a full custom CSV instead)."
            ),
            "options": ["constant", "Load from CSV"],
            "group": "Experimental conditions",
        },
    )
    """Pressure/temperature template mode: 'constant' or 'Load from CSV'."""
    pressure_bar: float = field(
        default=1.013,
        metadata={
            "name": "Pressure (bar)",
            "help": "Constant pressure written to the pressure-temperature template.",
            "depends_on": {"field": "pressure_temperature_mode", "value": "constant"},
            "group": "Experimental conditions",
        },
    )
    """Constant pressure in bar (pressure_temperature_mode='constant' only)."""
    temperature_celsius: float = field(
        default=20.0,
        metadata={
            "name": "Temperature (C)",
            "help": "Constant temperature written to the pressure-temperature template.",
            "depends_on": {"field": "pressure_temperature_mode", "value": "constant"},
            "group": "Experimental conditions",
        },
    )
    """Constant temperature in Celsius (pressure_temperature_mode='constant' only)."""

    def load_pressure_temperature(self, sec: dict) -> None:
        """Read the pressure/temperature keys from a flat [protocols] section."""
        try:
            pressure_temperature_protocol = sec["pressure_temperature"]
            if (
                isinstance(pressure_temperature_protocol, str)
                and not pressure_temperature_protocol.strip()
            ):
                self.pressure_temperature = None
            else:
                self.pressure_temperature = _parse_protocol_value(
                    pressure_temperature_protocol
                )
        except KeyError:
            self.pressure_temperature = None

        self.pressure_temperature_mode = _normalize_mode(
            sec.get("pressure_temperature_mode", "constant"),
            _SUPPORTED_PRESSURE_TEMPERATURE_MODES,
            key="[protocols].pressure_temperature_mode",
        )

        self.pressure_bar = float(sec.get("pressure_bar", 1.013))
        self.temperature_celsius = float(sec.get("temperature_celsius", 20.0))


@dataclass
class ProtocolsConfig(
    # Reverse order on purpose: dataclasses collect inherited fields in
    # reverse-MRO order, and that order is what the GUI renders top to bottom.
    PressureTemperatureProtocolConfig,
    InjectionProtocolConfig,
    ImagingProtocolConfig,
):
    """All three protocols for a run, read from one flat [protocols] section."""

    def load(self, path: Path) -> "ProtocolsConfig":
        sec = _get_section_from_toml(path, "protocols")
        self.load_imaging(sec)
        self.load_injection(sec)
        self.load_pressure_temperature(sec)
        return self

    def error(self):
        raise ValueError("Use [protocols] in the config file to load protocols.")
