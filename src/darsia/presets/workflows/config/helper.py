"""Configuration for helper workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .data_registry import DataRegistry
from .data_selection import resolve_time_data_selector
from .time_data import TimeData
from .utils import _get_key, _get_section, _get_section_from_toml


@dataclass
class HelperRoiConfig:
    """Configuration for ROI helper."""

    mode: str = "none"
    data: TimeData | None = None

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
        data_registry: DataRegistry | None,
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
        self.data = resolve_time_data_selector(
            sub_sec,
            "data",
            section="helper.roi",
            data=data,
            data_registry=data_registry,
            required=True,
        )
        return self


@dataclass
class HelperRoiViewerConfig:
    """Configuration for ROI viewer helper."""

    data: TimeData | None = None

    def load(
        self,
        sec: dict,
        *,
        data: Path | None,
        data_registry: DataRegistry | None,
    ) -> "HelperRoiViewerConfig":
        self.data = resolve_time_data_selector(
            sec,
            "data",
            section="helper.roi_viewer",
            data=data,
            data_registry=data_registry,
            required=True,
        )
        return self


@dataclass
class HelperConfig:
    """Configuration for helper workflows."""

    roi: HelperRoiConfig | None = None
    roi_viewer: HelperRoiViewerConfig | None = None

    def load(
        self,
        path: Path | list[Path],
        *,
        data: Path | None,
        data_registry: DataRegistry | None,
    ) -> "HelperConfig":
        sec = _get_section_from_toml(path, "helper")
        try:
            self.roi = HelperRoiConfig().load(
                sec,
                data=data,
                data_registry=data_registry,
            )
        except KeyError:
            self.roi = None

        self.roi_viewer = None
        if "roi_viewer" in sec:
            self.roi_viewer = HelperRoiViewerConfig().load(
                _get_section(sec, "roi_viewer"),
                data=data,
                data_registry=data_registry,
            )
        elif "data" in sec and set(sec.keys()).issubset({"data", "roi"}):
            # Shorthand support:
            # [helper]
            # data = ["registry_key"]
            self.roi_viewer = HelperRoiViewerConfig().load(
                sec,
                data=data,
                data_registry=data_registry,
            )
        return self
