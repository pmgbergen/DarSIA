"""Configuration for analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

from .data_registry import DataRegistry
from .fingers import FingersConfig
from .roi import RoiAndLabelConfig, RoiConfig
from .segmentation import SegmentationConfig
from .time_data import TimeData
from .utils import _convert_to_hours, _get_key, _get_section, _get_section_from_toml

if TYPE_CHECKING:
    from .roi_registry import RoiRegistry

logger = logging.getLogger(__name__)
SUPPORTED_CROPPING_FORMATS = {"jpg", "npz"}


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
    data: TimeData | str | list[str] | None = None
    """Cropping data selection configuration."""
    image_paths: list[Path] = field(default_factory=list)
    """Legacy direct image paths for cropping."""
    image_times: list[float] = field(default_factory=list)
    """Legacy direct image times for cropping."""
    formats: list[str] = field(default_factory=list)
    """Output formats for cropping images."""

    def load(
        self, sec: dict, data: Path | None, data_registry: DataRegistry | None = None
    ) -> "AnalysisCroppingConfig":
        sub_sec = _get_section(sec, "cropping")

        data_val = sub_sec.get("data")
        if isinstance(data_val, (str, list)) and data_registry is not None:
            self.data = data_val
        elif isinstance(data_val, dict):
            self.data = TimeData().load(data_val, data)
        else:
            try:
                self.data = TimeData().load(sub_sec, data)
            except ValueError:
                self.data = None

        raw_paths = _get_key(sub_sec, "image_paths", default=[], required=False)
        if not isinstance(raw_paths, list):
            raise ValueError("analysis.cropping.image_paths must be a list.")
        if len(raw_paths) > 0:
            self.image_paths = [
                Path(p) if Path(p).is_absolute() or data is None else data / p
                for p in raw_paths
            ]

        raw_times = _get_key(sub_sec, "image_times", default=[], required=False)
        if not isinstance(raw_times, list):
            raise ValueError("analysis.cropping.image_times must be a list.")
        if len(raw_times) > 0:
            self.image_times = [_convert_to_hours(t) for t in raw_times]

        raw_formats = _get_key(sub_sec, "formats", default=[], required=False)
        if not isinstance(raw_formats, list):
            raise ValueError("analysis.cropping.formats must be a list.")
        if not all(isinstance(fmt, str) for fmt in raw_formats):
            raise ValueError("analysis.cropping.formats entries must be strings.")
        self.formats = [fmt.strip().lower() for fmt in raw_formats]
        invalid_formats = sorted(set(self.formats) - SUPPORTED_CROPPING_FORMATS)
        if len(invalid_formats) > 0:
            raise ValueError(
                f"Unsupported analysis.cropping formats: {', '.join(invalid_formats)}. "
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

    def load(
        self,
        path: Path,
        data: Path | None,
        results: Path | None,
        data_registry: DataRegistry | None = None,
        roi_registry: RoiRegistry | None = None,
    ) -> "AnalysisConfig":
        sec = _get_section_from_toml(path, "analysis")

        # Config to load analysis data – support registry reference or inline
        data_val = sec.get("data")
        if isinstance(data_val, (str, list)) and data_registry is not None:
            self.data = data_registry.resolve(data_val)
        else:
            try:
                self.data = TimeData().load(sec["data"], data)
            except KeyError:
                warn("No analysis data found. Use [analysis.data].")
                self.data = None

        # Config to load analysis cropping
        try:
            self.cropping = AnalysisCroppingConfig().load(
                sec, data, data_registry=data_registry
            )
        except KeyError:
            warn("No analysis cropping found. Use [analysis.cropping].")
            self.cropping = None

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

        return self
