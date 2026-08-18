"""Configuration for helper workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .format_registry import FormatRegistry
from .roi_registry import RoiRegistry
from .utils import _convert_none, _get_key, _get_section, _get_section_from_toml


@dataclass
class HelperRoiConfig:
    """Configuration for ROI helper."""

    mode: str = "none"
    data_selection: str | list[str] | None = field(
        default=None,
        metadata={
            "name": "Data selection",
            "help": "Registry key name(s) whose data is unioned for ROI helper.",
            "widget": "registry_key_list",
        },
    )
    """Name(s) of data registry entries to use for ROI helper."""

    SUPPORTED_MODES = {
        "none",
        "concentration_aq",
        "saturation_g",
        "mass",
        "mass_total",
        "mass_g",
        "mass_aq",
        "rescaled_mass",
        "rescaled_saturation_g",
        "rescaled_concentration_aq",
    }

    def load(
        self,
        sec: dict,
        *,
        data: Path | None,
    ) -> "HelperRoiConfig":
        """Load ROI-helper configuration into this instance and return ``self``."""
        sub_sec = _get_section(sec, "roi")
        raw_mode = _get_key(
            sub_sec, "mode", default=self.mode, required=False, type_=str
        )
        self.mode = str(raw_mode).strip()
        if self.mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported helper.roi.mode '{self.mode}'. Supported modes: "
                f"{', '.join(sorted(self.SUPPORTED_MODES))}."
            )

        # Load data selection
        self.data_selection = _get_key(
            sub_sec, "data_selection", required=False, default=None
        )
        if self.data_selection is None:
            self.data_selection = _get_key(
                sub_sec, "data", required=False, default=None
            )
        return self


@dataclass
class HelperRoiViewerConfig:
    """Configuration for ROI viewer helper."""

    data_selection: str | list[str] | None = field(
        default=None,
        metadata={
            "name": "Data selection",
            "help": "Registry key name(s) whose data is unioned for ROI viewer helper.",
            "widget": "registry_key_list",
        },
    )
    """Name(s) of data registry entries to use for ROI viewer helper."""

    def load(
        self,
        sec: dict,
        *,
        data: Path | None,
    ) -> "HelperRoiViewerConfig":
        # Load data selection
        self.data_selection = _get_key(
            sec, "data_selection", required=False, default=None
        )
        if self.data_selection is None:
            self.data_selection = _get_key(sec, "data", required=False, default=None)
        return self


@dataclass
class HelperResultsConfig:
    """Configuration for result reader helper."""

    data_selection: str | list[str] | None = field(
        default=None,
        metadata={
            "name": "Data selection",
            "help": "Registry key name(s) whose data is unioned for results helper.",
            "widget": "registry_key_list",
        },
    )
    """Name(s) of data registry entries to use for results helper."""
    mode: str = "rescaled_mass"
    format: str = "npz"
    cmap: str | None = None
    roi: list[str] | None = None

    def load(
        self,
        sec: dict,
        *,
        data: Path | None,
        format_registry: FormatRegistry | None,
        roi_registry: RoiRegistry | None,
    ) -> "HelperResultsConfig":
        # Load data selection
        self.data_selection = _get_key(
            sec, "data_selection", required=False, default=None
        )
        if self.data_selection is None:
            self.data_selection = _get_key(sec, "data", required=False, default=None)

        self.mode = str(_get_key(sec, "mode", required=True, type_=str)).strip()
        if self.mode == "":
            raise ValueError("helper.results.mode must be a non-empty string.")

        self.format = str(
            _get_key(sec, "format", default=self.format, required=False, type_=str)
        ).strip()
        if self.format == "":
            raise ValueError("helper.results.format must be a non-empty string.")
        if format_registry is not None:
            try:
                specs = format_registry.resolve(self.format)
                if len(specs) != 1:
                    raise ValueError(
                        "helper.results.format must resolve to exactly one format entry."
                    )
                if specs[0].type not in {"csv", "npz"}:
                    raise ValueError(
                        "helper.results.format must resolve to csv or npz format."
                    )
            except KeyError:
                if self.format.lower() not in {"csv", "npz"}:
                    raise ValueError(
                        "helper.results.format must reference a format registry key "
                        "or use one of: csv, npz."
                    )
        elif self.format.lower() not in {"csv", "npz"}:
            raise ValueError("helper.results.format must be one of: csv, npz.")

        cmap = _convert_none(_get_key(sec, "cmap", default=None, required=False))
        self.cmap = None if cmap is None else str(cmap).strip()
        if self.cmap == "":
            self.cmap = None

        roi_value = _convert_none(_get_key(sec, "roi", default=None, required=False))
        if roi_value is None:
            self.roi = None
        else:
            if isinstance(roi_value, str):
                roi_keys = [roi_value]
            elif isinstance(roi_value, list):
                roi_keys = [str(key) for key in roi_value]
            else:
                raise ValueError(
                    "helper.results.roi must be None, a string, or a list of strings."
                )
            if roi_registry is None:
                raise ValueError(
                    "helper.results.roi references ROI keys, but no ROI registry "
                    "is available. Define top-level [roi.*] entries."
                )
            resolved = roi_registry.resolve_rois(roi_keys)
            missing = [key for key in roi_keys if key not in resolved]
            if missing:
                raise ValueError(
                    f"helper.results.roi contains non-plain ROI entries or unknown keys: "
                    f"{missing}"
                )
            self.roi = roi_keys

        return self


@dataclass
class HelperColorConfig:
    """Configuration for color helper."""

    data_selection: str | list[str] | None = field(
        default=None,
        metadata={
            "name": "Data selection",
            "help": "Registry key name(s) whose data is unioned for color helper.",
            "widget": "registry_key_list",
        },
    )
    """Name(s) of data registry entries to use for color helper."""

    def load(
        self,
        sec: dict,
        *,
        data: Path | None,
    ) -> "HelperColorConfig":
        # Load data selection
        self.data_selection = _get_key(
            sec, "data_selection", required=False, default=None
        )
        if self.data_selection is None:
            self.data_selection = _get_key(sec, "data", required=False, default=None)
        return self


@dataclass
class HelperConfig:
    """Configuration for helper workflows."""

    data_selection: str | list[str] | None = field(
        default=None,
        metadata={
            "name": "Data selection",
            "help": "Registry key name(s) whose data is unioned for helper.",
            "widget": "registry_key_list",
        },
    )
    """Name(s) of data registry entries to use for helper."""
    roi: HelperRoiConfig | None = None
    roi_viewer: HelperRoiViewerConfig | None = None
    results: HelperResultsConfig | None = None
    color: HelperColorConfig | None = None

    def load(
        self,
        path: Path | list[Path],
        *,
        data: Path | None,
        format_registry: FormatRegistry | None = None,
        roi_registry: RoiRegistry | None = None,
    ) -> "HelperConfig":
        sec = _get_section_from_toml(path, "helper")

        # Load data selection
        self.data_selection = _get_key(
            sec, "data_selection", required=False, default=None
        )
        if self.data_selection is None:
            self.data_selection = _get_key(sec, "data", required=False, default=None)
        try:
            self.roi = HelperRoiConfig().load(
                sec,
                data=data,
            )
        except KeyError:
            self.roi = None

        self.roi_viewer = None
        if "roi_viewer" in sec:
            self.roi_viewer = HelperRoiViewerConfig().load(
                _get_section(sec, "roi_viewer"),
                data=data,
            )

        self.results = None
        if "results" in sec:
            self.results = HelperResultsConfig().load(
                _get_section(sec, "results"),
                data=data,
                format_registry=format_registry,
                roi_registry=roi_registry,
            )

        self.color = None
        if "color" in sec:
            self.color = HelperColorConfig().load(
                _get_section(sec, "color"),
                data=data,
            )
        return self
