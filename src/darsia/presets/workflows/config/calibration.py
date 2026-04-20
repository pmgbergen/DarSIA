"""Calibration entrypoint configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from warnings import warn

from .data_registry import DataRegistry
from .data_selection import resolve_time_data_selector
from .time_data import TimeData
from .utils import _get_key, _get_section, _get_section_from_toml


@dataclass
class CalibrationColorConfig:
    """Config for selecting a color embedding for color calibration."""

    color: str = ""

    def load(self, sec: dict) -> "CalibrationColorConfig":
        self.color = _get_key(sec, "color", required=True, type_=str).strip()
        if not self.color:
            raise ValueError("calibration.color.color must be non-empty.")
        return self


@dataclass
class CalibrationMassConfig:
    """Config for mass calibration using a selected color embedding."""

    color: str = ""
    mode: str = "manual"
    fluid: str | None = "co2"
    data: TimeData | None = None
    threshold: float = 0.2
    rois: list[str] = field(default_factory=list)

    def load(
        self,
        sec: dict,
        *,
        data: Path | None,
        data_registry: DataRegistry | None = None,
    ) -> "CalibrationMassConfig":
        self.color = _get_key(sec, "color", required=True, type_=str).strip()
        self.mode = _get_key(sec, "mode", default="manual", required=False, type_=str)
        self.fluid = _get_key(sec, "fluid", default="co2", required=False, type_=str)
        self.threshold = _get_key(sec, "threshold", default=0.2, required=False, type_=float)
        self.rois = _get_key(sec, "rois", default=[], required=False, type_=list)
        try:
            self.data = resolve_time_data_selector(
                sec,
                "data",
                section="calibration.mass",
                data=data,
                data_registry=data_registry,
                required=True,
            )
        except KeyError:
            warn("No data found for calibration.mass. Use [calibration.mass].data.")
            self.data = None
        return self


@dataclass
class CalibrationConfig:
    """Root calibration config container."""

    color: CalibrationColorConfig | None = None
    mass: CalibrationMassConfig | None = None

    def load(
        self,
        path: Path | list[Path],
        *,
        data: Path | None,
        data_registry: DataRegistry | None = None,
    ) -> "CalibrationConfig":
        sec = _get_section_from_toml(path, "calibration")
        if not isinstance(sec, dict):
            raise ValueError("[calibration] must be a table.")

        try:
            self.color = CalibrationColorConfig().load(_get_section(sec, "color"))
        except KeyError:
            self.color = None

        try:
            self.mass = CalibrationMassConfig().load(
                _get_section(sec, "mass"),
                data=data,
                data_registry=data_registry,
            )
        except KeyError:
            self.mass = None

        return self

