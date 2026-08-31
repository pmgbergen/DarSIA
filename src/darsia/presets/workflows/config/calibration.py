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
class ColorPathCalibrationConfig:
    """Config for color-path calibration fitting parameters."""

    num_segments: int = field(
        default=1,
        metadata={
            "name": "Num segments",
            "help": "Number of segments in the color path.",
        },
    )
    resolution: int = field(
        default=51,
        metadata={
            "name": "Resolution",
            "help": "Resolution of the color spectrum.",
        },
    )
    calibration_mode: str = field(
        default="auto",
        metadata={
            "name": "Calibration mode",
            "help": "Calibration mode: 'manual' or 'auto'.",
            "options": ["manual", "auto"],
        },
    )
    threshold_baseline: float = field(
        default=0.0,
        metadata={
            "name": "Threshold baseline",
            "help": "Threshold for baseline spectrum.",
        },
    )
    threshold_calibration: float = field(
        default=0.0,
        metadata={
            "name": "Threshold calibration",
            "help": "Threshold for calibration spectrum.",
        },
    )
    ignore_baseline_spectrum: str = field(
        default="expanded",
        metadata={
            "name": "Ignore baseline spectrum",
            "help": "How to handle baseline spectrum: 'none', 'baseline', or "
            "'expanded'.",
            "options": ["none", "baseline", "expanded"],
        },
    )
    histogram_weighting: str = field(
        default="threshold",
        metadata={
            "name": "Histogram weighting",
            "help": "Weighting method: 'threshold', 'wls', 'wls_sqrt', or 'wls_log'.",
            "options": ["threshold", "wls", "wls_sqrt", "wls_log"],
        },
    )
    ignore_labels: list[int] = field(
        default_factory=list,
        metadata={
            "name": "Ignore labels",
            "help": "Label IDs to skip during calibration (will receive zero paths).",
        },
    )


@dataclass
class CalibrationColorConfig:
    """Config for selecting a color embedding for color calibration."""

    embedding: ColorEmbedding | None = field(
        default=None,
        metadata={
            "name": "Color embedding",
            "widget": "color_key_list",
            "max_rows": 1,
        },
    )
    data_selection: str | list[str] | None = field(
        default=None,
        metadata={
            "name": "Data selection",
            "help": "Registry key name(s) for calibration images.",
            "widget": "registry_key_list",
        },
    )
    baseline: str | list[str] | None = field(
        default=None,
        metadata={
            "name": "Baseline",
            "help": "Registry key name(s) for baseline images.",
            "widget": "registry_key_list",
        },
    )
    rois: list[str] = field(
        default_factory=list,
        metadata={
            "name": "ROIs",
            "help": "ROI names for color-path calibration.",
            "widget": "roi_key_list",
        },
    )
    color_path: ColorPathCalibrationConfig | None = field(
        default_factory=ColorPathCalibrationConfig,
        metadata={
            "name": "Color Path Calibration",
            "depends_on": {"field": "embedding", "type": "color_path"},
        },
    )

    def load(
        self,
        sec: dict,
        *,
        color_embedding_registry: "ColorEmbeddingRegistry | None" = None,
        roi_registry: "RoiRegistry | None" = None,
    ) -> "CalibrationColorConfig":
        embedding_key = _get_key(sec, "embedding", required=True, type_=str).strip()
        if not embedding_key:
            raise ValueError("calibration.color.embedding must be non-empty.")
        if color_embedding_registry is None:
            raise ValueError(
                "calibration.color.embedding references [color.*.*], but no "
                "ColorEmbeddingRegistry is available."
            )
        try:
            self.embedding = color_embedding_registry.resolve(embedding_key)
        except KeyError as exc:
            raise ValueError(
                "Unknown calibration.color.embedding "
                f"'{embedding_key}'. Define it under [color.*.*]."
            ) from exc

        self.data_selection = _get_key(
            sec, "data_selection", required=False, default=None
        )
        self.baseline = _get_key(sec, "baseline", required=False, default=None)
        self.rois = _load_roi_key_list(
            sec,
            "rois",
            context="calibration.color.rois",
            roi_registry=roi_registry,
            restricted=False,
        )

        # Load color_path calibration config if present
        try:
            color_path_sec = _get_section(sec, "color_path")
            self.color_path = ColorPathCalibrationConfig()
            self.color_path.num_segments = _get_key(
                color_path_sec, "num_segments", required=False, default=1, type_=int
            )
            self.color_path.resolution = _get_key(
                color_path_sec, "resolution", required=False, default=51, type_=int
            )
            self.color_path.calibration_mode = _get_key(
                color_path_sec,
                "calibration_mode",
                required=False,
                default="auto",
                type_=str,
            )
            if self.color_path.calibration_mode not in {"manual", "auto"}:
                raise ValueError(
                    "calibration.color.color_path.calibration_mode must be "
                    f"'manual' or 'auto', got '{self.color_path.calibration_mode}'."
                )
            self.color_path.threshold_baseline = _get_key(
                color_path_sec,
                "threshold_baseline",
                required=False,
                default=0.0,
                type_=float,
            )
            self.color_path.threshold_calibration = _get_key(
                color_path_sec,
                "threshold_calibration",
                required=False,
                default=0.0,
                type_=float,
            )
            self.color_path.ignore_baseline_spectrum = _get_key(
                color_path_sec,
                "ignore_baseline_spectrum",
                required=False,
                default="expanded",
                type_=str,
            )
            if self.color_path.ignore_baseline_spectrum not in {
                "none",
                "baseline",
                "expanded",
            }:
                raise ValueError(
                    "calibration.color.color_path.ignore_baseline_spectrum must be "
                    "'none', 'baseline', or 'expanded', got "
                    f"'{self.color_path.ignore_baseline_spectrum}'."
                )
            self.color_path.histogram_weighting = _get_key(
                color_path_sec,
                "histogram_weighting",
                required=False,
                default="threshold",
                type_=str,
            )
            if self.color_path.histogram_weighting not in {
                "threshold",
                "wls",
                "wls_sqrt",
                "wls_log",
            }:
                raise ValueError(
                    "calibration.color.color_path.histogram_weighting must be one "
                    "of 'threshold', 'wls', 'wls_sqrt', 'wls_log', got "
                    f"'{self.color_path.histogram_weighting}'."
                )
            self.color_path.ignore_labels = list(
                _get_key(
                    color_path_sec,
                    "ignore_labels",
                    required=False,
                    default=[],
                    type_=list,
                )
            )
        except KeyError:
            self.color_path = ColorPathCalibrationConfig()

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
                roi_registry=roi_registry,
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
