"""Calibration entrypoint configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .roi_registry import _load_roi_key_list
from .utils import _get_key, _get_section, _get_section_from_toml

if TYPE_CHECKING:
    from darsia.signals.color import ColorEmbedding

    from .color_embedding_registry import ColorEmbeddingRegistry
    from .roi_registry import RoiRegistry


@dataclass
class CalibrationColorConfig:
    """Config for selecting a color embedding for color calibration."""

    color: "ColorEmbedding | None" = field(
        default=None,
        metadata={
            "name": "Color embedding",
            "widget": "color_key_list",
            "max_rows": 1,
        },
    )

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

    color: "ColorEmbedding | None" = field(
        default=None,
        metadata={
            "name": "Color embedding",
            "widget": "color_key_list",
            "max_rows": 1,
        },
    )
    mode: str = field(
        default="manual",
        metadata={
            "name": "Calibration mode",
            "help": "How the mass calibration threshold is determined: 'manual' "
            "(use the configured threshold) or 'auto' (reserved for future automatic "
            "calibration; not yet wired to any consumer).",
            "options": ["manual", "auto"],
        },
    )
    fluid: str | None = field(
        default="co2",
        metadata={
            "name": "Fluid",
            "help": "Fluid identifier for mass calibration (e.g. 'co2').",
        },
    )
    data_selection: str | list[str] | None = field(
        default=None,
        metadata={
            "name": "Data selection",
            "help": "Registry key name(s) whose data is unioned for mass calibration.",
            "widget": "registry_key_list",
        },
    )
    """Name(s) of data registry entries to use for mass calibration."""
    threshold: float = field(
        default=0.2,
        metadata={
            "name": "Threshold",
            "help": "Mass calibration threshold. Currently only meaningful for "
            "color-path embeddings.",
        },
    )
    rois: list[str] = field(
        default_factory=list,
        metadata={
            "name": "ROIs",
            "help": "ROI names for mass calibration.",
            "widget": "roi_key_list",
        },
    )

    def load(
        self,
        sec: dict,
        *,
        data: Path | None,
        color_embedding_registry: "ColorEmbeddingRegistry | None" = None,
        roi_registry: "RoiRegistry | None" = None,
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
        self.rois = _load_roi_key_list(
            sec,
            "rois",
            context="calibration.mass.rois",
            roi_registry=roi_registry,
            restricted=False,
        )

        # Load data selection
        self.data_selection = _get_key(
            sec, "data_selection", required=False, default=None
        )
        if self.data_selection is None:
            self.data_selection = _get_key(sec, "data", required=False, default=None)
        return self


@dataclass
class CalibrationConfig:
    """Root calibration config container."""

    color: CalibrationColorConfig | None = field(
        default=None,
        metadata={
            "name": "Color calibration",
            "help": "Color-embedding calibration entrypoint.",
        },
    )
    mass: CalibrationMassConfig | None = field(
        default=None,
        metadata={
            "name": "Mass calibration",
            "help": "Mass calibration entrypoint, built on a color embedding.",
        },
    )

    def load(
        self,
        path: Path | list[Path],
        *,
        data: Path | None,
        color_embedding_registry: "ColorEmbeddingRegistry | None" = None,
        roi_registry: "RoiRegistry | None" = None,
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
                color_embedding_registry=color_embedding_registry,
                roi_registry=roi_registry,
            )
        except KeyError:
            self.mass = None

        return self
