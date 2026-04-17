"""Configuration for analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

from .data_registry import DataRegistry
from .data_selection import resolve_time_data_selector
from .fingers import FingersConfig
from .roi import RoiAndLabelConfig, RoiConfig
from .segmentation import SegmentationConfig
from .time_data import TimeData
from .utils import _get_key, _get_section, _get_section_from_toml

if TYPE_CHECKING:
    from .roi_registry import RoiRegistry

logger = logging.getLogger(__name__)


def _to_rgb(color: list[int] | tuple[int, int, int], name: str) -> tuple[int, int, int]:
    if len(color) != 3:
        raise ValueError(f"{name} must have exactly 3 entries [R, G, B].")
    vals = tuple(int(v) for v in color)
    if any(v < 0 or v > 255 for v in vals):
        raise ValueError(f"{name} entries must be in [0, 255].")
    return vals


@dataclass
class AnalysisThresholdingLegendConfig:
    show: bool = True
    font_scale: float = 0.7
    thickness: int = 2
    line_spacing: int = 8
    position: tuple[int, int] = (20, 20)
    text_color: tuple[int, int, int] = (255, 255, 255)
    box_enabled: bool = True
    box_color: tuple[int, int, int] = (0, 0, 0)
    box_alpha: float = 0.4
    box_padding: int = 10

    def load(self, sec: dict) -> "AnalysisThresholdingLegendConfig":
        self.show = bool(_get_key(sec, "show", required=False, default=self.show))
        self.font_scale = float(
            _get_key(sec, "font_scale", required=False, default=self.font_scale)
        )
        self.thickness = int(
            _get_key(sec, "thickness", required=False, default=self.thickness)
        )
        self.line_spacing = int(
            _get_key(sec, "line_spacing", required=False, default=self.line_spacing)
        )
        position = _get_key(sec, "position", required=False, default=self.position)
        if len(position) != 2:
            raise ValueError("analysis.thresholding.legend.position must be [x, y].")
        self.position = (int(position[0]), int(position[1]))
        self.text_color = _to_rgb(
            _get_key(sec, "text_color", required=False, default=self.text_color),
            "analysis.thresholding.legend.text_color",
        )
        self.box_enabled = bool(
            _get_key(sec, "box_enabled", required=False, default=self.box_enabled)
        )
        self.box_color = _to_rgb(
            _get_key(sec, "box_color", required=False, default=self.box_color),
            "analysis.thresholding.legend.box_color",
        )
        self.box_alpha = float(
            _get_key(sec, "box_alpha", required=False, default=self.box_alpha)
        )
        if not (0 <= self.box_alpha <= 1):
            raise ValueError(
                "analysis.thresholding.legend.box_alpha must be in [0, 1]."
            )
        self.box_padding = int(
            _get_key(sec, "box_padding", required=False, default=self.box_padding)
        )
        return self


@dataclass
class AnalysisThresholdingConfig:
    @dataclass
    class LayerConfig:
        mode: str = "concentration_aq"
        threshold_min: float | None = None
        threshold_max: float | None = None
        label: str = ""
        fill: tuple[int, int, int] = (255, 255, 255)
        stroke: tuple[int, int, int] = (0, 0, 0)
        fill_alpha: float = 0.35
        stroke_width: int = 2

        def load(
            self, sec: dict, *, key: str, supported_modes: set[str]
        ) -> "AnalysisThresholdingConfig.LayerConfig":
            self.mode = _get_key(sec, "mode", required=True, type_=str).strip()
            if self.mode not in supported_modes:
                raise ValueError(
                    f"Unsupported analysis.thresholding.layers.{key}.mode '{self.mode}'. "
                    f"Supported modes: {', '.join(sorted(supported_modes))}."
                )
            self.threshold_min = _get_key(sec, "threshold_min", required=False)
            self.threshold_max = _get_key(sec, "threshold_max", required=False)
            if self.threshold_min is not None:
                self.threshold_min = float(self.threshold_min)
            if self.threshold_max is not None:
                self.threshold_max = float(self.threshold_max)
            if (
                self.threshold_min is not None
                and self.threshold_max is not None
                and self.threshold_min > self.threshold_max
            ):
                raise ValueError(
                    f"analysis.thresholding.layers.{key} has threshold_min > threshold_max."
                )
            if self.threshold_min is None and self.threshold_max is None:
                raise ValueError(
                    f"analysis.thresholding.layers.{key} must have at least one of "
                    "threshold_min or threshold_max."
                )

            self.label = _get_key(sec, "label", required=False, default=key, type_=str)
            self.fill = _to_rgb(
                _get_key(sec, "fill", required=False, default=self.fill),
                f"analysis.thresholding.layers.{key}.fill",
            )
            self.stroke = _to_rgb(
                _get_key(sec, "stroke", required=False, default=self.stroke),
                f"analysis.thresholding.layers.{key}.stroke",
            )
            self.fill_alpha = float(
                _get_key(sec, "fill_alpha", required=False, default=self.fill_alpha)
            )
            if not (0.0 <= self.fill_alpha <= 1.0):
                raise ValueError(
                    f"analysis.thresholding.layers.{key}.fill_alpha must be in [0, 1]."
                )
            self.stroke_width = int(
                _get_key(sec, "stroke_width", required=False, default=self.stroke_width)
            )
            if self.stroke_width < 0:
                raise ValueError(
                    f"analysis.thresholding.layers.{key}.stroke_width must be >= 0."
                )

            return self

    formats: list[str] = field(default_factory=lambda: ["jpg", "npz"])
    layers: dict[str, LayerConfig] = field(default_factory=dict)
    modes: list[str] = field(
        default_factory=lambda: ["concentration_aq", "saturation_g"]
    )
    thresholds: dict[str, float] = field(default_factory=dict)
    legend: AnalysisThresholdingLegendConfig = field(
        default_factory=AnalysisThresholdingLegendConfig
    )
    folder: Path = field(default_factory=Path)
    """Path to the results folder for thresholding analysis."""

    SUPPORTED_MODES = {
        "concentration_aq",
        "saturation_g",
        "mass_total",
        "mass_g",
        "mass_aq",
    }

    def load(self, sec: dict, results: Path | None) -> "AnalysisThresholdingConfig":
        sub_sec = _get_section(sec, "thresholding")

        raw_formats = _get_key(sub_sec, "formats", required=False, default=self.formats)
        if not isinstance(raw_formats, list):
            raise ValueError("analysis.thresholding.formats must be a list.")
        if not all(isinstance(fmt, str) for fmt in raw_formats):
            raise ValueError("analysis.thresholding.formats entries must be strings.")
        self.formats = [fmt.strip().lower() for fmt in raw_formats if fmt.strip()]
        if len(self.formats) == 0:
            raise ValueError("analysis.thresholding.formats must not be empty.")
        supported_formats = {"jpg", "npz"}
        invalid_formats = sorted(set(self.formats) - supported_formats)
        if len(invalid_formats) > 0:
            raise ValueError(
                "Unsupported [analysis.thresholding].formats entries: "
                f"{', '.join(invalid_formats)}. Supported formats: "
                f"{', '.join(sorted(supported_formats))}."
            )

        raw_layers = _get_key(sub_sec, "layers", required=False, default={})
        if not isinstance(raw_layers, dict):
            raise ValueError("analysis.thresholding.layers must be a table/dict.")
        self.layers = {}
        if len(raw_layers) > 0:
            for key in raw_layers.keys():
                layer_sec = _get_section(raw_layers, key)
                self.layers[key] = self.LayerConfig().load(
                    layer_sec, key=key, supported_modes=self.SUPPORTED_MODES
                )
        legend = _get_key(sub_sec, "legend", required=False, default={})
        if not isinstance(legend, dict):
            raise ValueError("analysis.thresholding.legend must be a table/dict.")
        self.legend.load(legend)

        folder = _get_key(sub_sec, "folder", required=False, type_=Path)
        if not folder:
            assert results is not None
            self.folder = results / "thresholding"
        else:
            self.folder = folder

        return self

    def error(self):
        raise ValueError(
            "Use [analysis.thresholding] in the config file to load thresholding."
        )


@dataclass
class AnalysisSegmentationConfig:
    config: SegmentationConfig | dict[str, SegmentationConfig] = field(
        default_factory=lambda: SegmentationConfig()
    )
    folder: Path = field(default_factory=Path)
    """Path to the results folder for segmentation."""

    def load(self, sec: dict, results: Path | None) -> "AnalysisSegmentationConfig":
        # Allow for two scenarios: single segmentation or multiple segmentations
        sub_sec = _get_section(sec, "segmentation")

        try:
            self.config = SegmentationConfig().load(sub_sec)
        except KeyError:
            self.config = {}
            for key in sub_sec.keys():
                self.config[key] = SegmentationConfig().load(_get_section(sub_sec, key))
            try:
                self.config = {}
                for key in sub_sec.keys():
                    self.config[key] = SegmentationConfig().load(
                        _get_section(sub_sec, key)
                    )
            except KeyError as e:
                raise KeyError(
                    "Segmentation config must be either a single or multiple segmentations."
                ) from e

        folder = _get_key(sub_sec, "folder", required=False, type_=Path)
        if not folder:
            assert results is not None
            self.folder = results / "segmentation"
        return self

    def error(self):
        raise ValueError(
            f"Use [analysis.segmentation] in the config file to load segmentation."
        )


@dataclass
class AnalysisMassConfig:
    roi: dict[str, RoiConfig] = field(default_factory=dict)
    """ROI configurations for mass analysis."""
    roi_and_label: dict[str, RoiAndLabelConfig] = field(default_factory=dict)
    """ROI and label configurations for mass analysis."""
    folder: Path = field(default_factory=Path)
    """Path to the results folder for mass analysis."""

    def load(
        self, sec: dict, results: Path | None, roi_registry: RoiRegistry | None = None
    ) -> "AnalysisMassConfig":
        sub_sec = _get_section(sec, "mass")

        # Load ROIs – support registry-key references (list) and inline dicts.
        roi_raw = sub_sec.get("roi")
        if isinstance(roi_raw, list) and roi_registry is not None:
            self.roi = roi_registry.resolve_rois(roi_raw)
        elif isinstance(roi_raw, dict):
            self.roi = {}
            for key in roi_raw.keys():
                self.roi[key] = RoiConfig().load(_get_section(roi_raw, key))
        else:
            try:
                roi_sec = _get_section(sub_sec, "roi")
                self.roi = {}
                for key in roi_sec.keys():
                    self.roi[key] = RoiConfig().load(_get_section(roi_sec, key))
            except KeyError:
                self.roi = {}

        # Load ROIs with labels – support registry-key references and inline dicts.
        roi_label_raw = sub_sec.get("roi_and_label")
        if isinstance(roi_label_raw, list) and roi_registry is not None:
            self.roi_and_label = roi_registry.resolve_roi_and_labels(roi_label_raw)
        elif isinstance(roi_label_raw, dict):
            self.roi_and_label = {}
            for key in roi_label_raw.keys():
                self.roi_and_label[key] = RoiAndLabelConfig().load(
                    _get_section(roi_label_raw, key)
                )
        else:
            try:
                roi_label_sec = _get_section(sub_sec, "roi_and_label")
                self.roi_and_label = {}
                for key in roi_label_sec.keys():
                    self.roi_and_label[key] = RoiAndLabelConfig().load(
                        _get_section(roi_label_sec, key)
                    )
            except KeyError:
                self.roi_and_label = {}

        folder = _get_key(sub_sec, "folder", required=False, type_=Path)
        if not folder:
            assert results is not None
            self.folder = results / "mass"
        return self


@dataclass
class AnalysisVolumeConfig:
    roi: dict[str, RoiConfig] = field(default_factory=dict)
    """ROI configurations for volume analysis."""
    roi_and_label: dict[str, RoiAndLabelConfig] = field(default_factory=dict)
    """ROI and label configurations for volume analysis."""
    folder: Path = field(default_factory=Path)
    """Path to the results folder for volume analysis."""

    def load(
        self,
        sec: dict,
        results: Path | None,
        roi_registry: RoiRegistry | None = None,
    ) -> "AnalysisVolumeConfig":
        sub_sec = _get_section(sec, "volume")

        # Load ROIs – support registry-key references (list) and inline dicts.
        roi_raw = sub_sec.get("roi")
        if isinstance(roi_raw, list) and roi_registry is not None:
            self.roi = roi_registry.resolve_rois(roi_raw)
        elif isinstance(roi_raw, dict):
            self.roi = {}
            for key in roi_raw.keys():
                self.roi[key] = RoiConfig().load(_get_section(roi_raw, key))
        else:
            try:
                roi_sec = _get_section(sub_sec, "roi")
                self.roi = {}
                for key in roi_sec.keys():
                    self.roi[key] = RoiConfig().load(_get_section(roi_sec, key))
            except KeyError:
                self.roi = {}

        # Load ROIs with labels – support registry-key references and inline dicts.
        roi_label_raw = sub_sec.get("roi_and_label")
        if isinstance(roi_label_raw, list) and roi_registry is not None:
            self.roi_and_label = roi_registry.resolve_roi_and_labels(roi_label_raw)
        elif isinstance(roi_label_raw, dict):
            self.roi_and_label = {}
            for key in roi_label_raw.keys():
                self.roi_and_label[key] = RoiAndLabelConfig().load(
                    _get_section(roi_label_raw, key)
                )
        else:
            try:
                roi_label_sec = _get_section(sub_sec, "roi_and_label")
                self.roi_and_label = {}
                for key in roi_label_sec.keys():
                    self.roi_and_label[key] = RoiAndLabelConfig().load(
                        _get_section(roi_label_sec, key)
                    )
            except KeyError:
                self.roi_and_label = {}

        folder = _get_key(sub_sec, "folder", required=False, type_=Path)
        if not folder:
            assert results is not None
            self.folder = results / "volume"
        return self


@dataclass
class AnalysisFingersConfig:
    config: FingersConfig | dict[str, FingersConfig] = field(
        default_factory=lambda: FingersConfig()
    )
    folder: Path = field(default_factory=Path)
    """Path to the results folder for segmentation."""
    img_folder: Path = field(default_factory=Path)
    """Path to the image results folder."""

    def load(
        self,
        sec: dict,
        results: Path | None,
        roi_registry: RoiRegistry | None = None,
    ) -> "AnalysisFingersConfig":
        # Allow for two scenarios: single fingers or multiple fingers
        sub_sec = _get_section(sec, "fingers")

        try:
            self.config = FingersConfig().load(sub_sec, roi_registry=roi_registry)
        except KeyError:
            self.config = {}
            for key in sub_sec.keys():
                self.config[key] = FingersConfig().load(
                    _get_section(sub_sec, key), roi_registry=roi_registry
                )
            try:
                self.config = {}
                for key in sub_sec.keys():
                    self.config[key] = FingersConfig().load(
                        _get_section(sub_sec, key), roi_registry=roi_registry
                    )
            except KeyError as e:
                raise KeyError(
                    "Fingers config must be either a single or multiple fingers."
                ) from e

        folder = _get_key(sub_sec, "folder", required=False, type_=Path)
        if not folder:
            assert results is not None
            self.folder = results / "fingers"
        return self

    def error(self):
        raise ValueError(f"Use [analysis.fingers] in the config file to load fingers.")


@dataclass
class AnalysisCroppingConfig:
    formats: list[str] = field(default_factory=lambda: ["jpg"])
    """Output formats for cropping images."""

    def load(self, sec: dict) -> "AnalysisCroppingConfig":
        sub_sec = _get_section(sec, "cropping")

        raw_formats = _get_key(sub_sec, "formats", default=["jpg"], required=False)
        if not isinstance(raw_formats, list):
            raise ValueError("analysis.cropping.formats must be a list.")
        if not all(isinstance(fmt, str) for fmt in raw_formats):
            raise ValueError("analysis.cropping.formats entries must be strings.")
        self.formats = [fmt.strip().lower() for fmt in raw_formats]
        SUPPORTED_CROPPING_FORMATS = {"jpg", "npz"}
        invalid_formats = sorted(set(self.formats) - SUPPORTED_CROPPING_FORMATS)
        if len(invalid_formats) > 0:
            raise ValueError(
                "Unsupported [analysis.cropping].formats entries: "
                f"{', '.join(invalid_formats)}. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_CROPPING_FORMATS))}."
            )
        return self


@dataclass
class AnalysisConfig:
    data: TimeData | None = None
    """Analysis data configuration."""
    cropping: AnalysisCroppingConfig | None = None
    """Analysis cropping configuration."""
    segmentation: AnalysisSegmentationConfig | None = None
    """Analysis segmentation configuration."""
    mass: AnalysisMassConfig | None = None
    """Analysis mass configuration."""
    volume: AnalysisVolumeConfig | None = None
    """Analysis volume configuration."""
    fingers: AnalysisFingersConfig | None = None
    """Analysis fingers configuration."""
    thresholding: AnalysisThresholdingConfig | None = None
    """Analysis thresholding configuration."""

    def load(
        self,
        path: Path,
        data: Path | None,
        results: Path | None,
        data_registry: DataRegistry | None = None,
        roi_registry: RoiRegistry | None = None,
    ) -> "AnalysisConfig":
        sec = _get_section_from_toml(path, "analysis")

        # Config to load analysis data – centralized selector resolution.
        try:
            self.data = resolve_time_data_selector(
                sec,
                "data",
                section="analysis",
                data=data,
                data_registry=data_registry,
                required=True,
            )
        except KeyError:
            warn("No analysis data found. Use [analysis.data].")
            self.data = None

        # Config to load analysis cropping
        try:
            self.cropping = AnalysisCroppingConfig().load(sec)
        except KeyError:
            warn("No analysis cropping found. Use [analysis.cropping].")
            self.cropping = AnalysisCroppingConfig()  # Default to empty cropping config

        # Config to load analysis segmentation
        try:
            self.segmentation = AnalysisSegmentationConfig().load(sec, results)
        except KeyError:
            warn("No analysis segmentation found. Use [analysis.segmentation].")
            self.segmentation = None

        # Config to load analysis mass
        try:
            self.mass = AnalysisMassConfig().load(
                sec, results, roi_registry=roi_registry
            )
        except KeyError:
            warn("No analysis mass found. Use [analysis.mass].")
            self.mass = None

        # Config to load analysis volume
        try:
            self.volume = AnalysisVolumeConfig().load(
                sec, results, roi_registry=roi_registry
            )
        except KeyError:
            warn("No analysis volume found. Use [analysis.volume].")
            self.volume = None

        # Config to load analysis fingers
        try:
            self.fingers = AnalysisFingersConfig().load(
                sec, results, roi_registry=roi_registry
            )
        except KeyError:
            warn("No analysis fingers found. Use [analysis.fingers].")
            self.fingers = None

        # Config to load analysis thresholding
        try:
            self.thresholding = AnalysisThresholdingConfig().load(sec, results)
        except KeyError:
            warn("No analysis thresholding found. Use [analysis.thresholding].")
            self.thresholding = None

        return self
