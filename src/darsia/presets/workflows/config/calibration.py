"""Calibration entrypoint configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

from .data_registry import DataRegistry
from .data_selection import resolve_time_data_selector
from .time_data import TimeData
from .utils import _get_key, _get_section, _get_section_from_toml

if TYPE_CHECKING:
    from darsia.signals.color import ColorEmbedding

    from .color_embedding_registry import ColorEmbeddingRegistry


@dataclass
class CalibrationColorConfig:
    """Config for selecting a color embedding for color calibration."""

    color: "ColorEmbedding | None" = None

    def load(
        self,
        sec: dict,
        *,
        color_embedding_registry: "ColorEmbeddingRegistry | None" = None,
    ) -> "CalibrationColorConfig":
        color_key = _get_key(sec, "color", required=True, type_=str).strip()
        if not color_key:
            raise ValueError("calibration.color.color must be non-empty.")
        if color_embedding_registry is None:
            raise ValueError(
                "calibration.color.color references [color.*.*], but no "
                "ColorEmbeddingRegistry is available."
            )
        try:
            self.color = color_embedding_registry.resolve(color_key)
        except KeyError as exc:
            raise ValueError(
                "Unknown calibration.color.color embedding "
                f"'{color_key}'. Define it under [color.*.*]."
            ) from exc
        return self


@dataclass
class CalibrationMassConfig:
    """Config for mass calibration using a selected color embedding."""

    color: "ColorEmbedding | None" = None
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
        color_embedding_registry: "ColorEmbeddingRegistry | None" = None,
    ) -> "CalibrationMassConfig":
        color_key = _get_key(sec, "color", required=True, type_=str).strip()
        if color_embedding_registry is None:
            raise ValueError(
                "calibration.mass.color references [color.*.*], but no "
                "ColorEmbeddingRegistry is available."
            )
        try:
            self.color = color_embedding_registry.resolve(color_key)
        except KeyError as exc:
            raise ValueError(
                "Unknown calibration.mass.color embedding "
                f"'{color_key}'. Define it under [color.*.*]."
            ) from exc

        self.mode = _get_key(sec, "mode", default="manual", required=False, type_=str)
        self.mode = self.mode.lower().strip()
        if self.mode not in {"manual", "auto"}:
            raise ValueError("calibration.mass.mode must be either 'manual' or 'auto'.")
        self.fluid = _get_key(sec, "fluid", default="co2", required=False, type_=str)
        self.threshold = _get_key(
            sec, "threshold", default=0.2, required=False, type_=float
        )
        # This threshold is currently only meaningful for color-path embeddings.
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
        color_embedding_registry: "ColorEmbeddingRegistry | None" = None,
    ) -> "CalibrationConfig":
        sec = _get_section_from_toml(path, "calibration")
        if not isinstance(sec, dict):
            raise ValueError("[calibration] must be a table.")

        try:
            self.color = CalibrationColorConfig().load(
                _get_section(sec, "color"),
                color_embedding_registry=color_embedding_registry,
            )
        except KeyError:
            self.color = None

        try:
            self.mass = CalibrationMassConfig().load(
                _get_section(sec, "mass"),
                data=data,
                data_registry=data_registry,
                color_embedding_registry=color_embedding_registry,
            )
        except KeyError:
            self.mass = None

        return self
