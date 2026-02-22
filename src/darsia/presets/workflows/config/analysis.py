"""Configuration for analysis."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from warnings import warn

logger = logging.getLogger(__name__)

from .utils import (
    _get_section,
    _get_key,
    _get_section_from_toml,
)
from .time_data import TimeData
from .segmentation import SegmentationConfig
from .roi import RoiConfig, RoiAndLabelConfig


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
                    "Segmentation config must be either a single segmentation or multiple segmentations."
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

    def load(self, sec: dict, results: Path | None) -> "AnalysisMassConfig":
        sub_sec = _get_section(sec, "mass")

        # Load ROIs
        try:
            roi_sec = _get_section(sub_sec, "roi")
            self.roi = {}
            for key in roi_sec.keys():
                self.roi[key] = RoiConfig().load(_get_section(roi_sec, key))
        except KeyError:
            self.roi = {}

        # Load ROIs with labels
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

    def load(self, sec: dict, results: Path | None) -> "AnalysisVolumeConfig":
        sub_sec = _get_section(sec, "volume")

        # Load ROIs
        try:
            roi_sec = _get_section(sub_sec, "roi")
            self.roi = {}
            for key in roi_sec.keys():
                self.roi[key] = RoiConfig().load(_get_section(roi_sec, key))
        except KeyError:
            self.roi = {}

        # Load ROIs with labels
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
class AnalysisConfig:
    data: TimeData | None = None
    """Analysis data configuration."""
    segmentation: AnalysisSegmentationConfig | None = None
    """Analysis segmentation configuration."""
    mass: AnalysisMassConfig | None = None
    """Analysis mass configuration."""
    volume: AnalysisVolumeConfig | None = None
    """Analysis volume configuration."""

    def load(
        self, path: Path, data: Path | None, results: Path | None
    ) -> "AnalysisConfig":
        sec = _get_section_from_toml(path, "analysis")

        # Config to load analysis data
        try:
            self.data = TimeData().load(sec["data"], data)
        except KeyError:
            warn("No analysis data found. Use [analysis.data].")
            self.data = None

        # Config to load analysis segmentation
        try:
            self.segmentation = AnalysisSegmentationConfig().load(sec, results)
        except KeyError:
            warn("No analysis segmentation found. Use [analysis.segmentation].")
            self.segmentation = None

        # Config to load analysis mass
        try:
            self.mass = AnalysisMassConfig().load(sec, results)
        except KeyError:
            warn("No analysis mass found. Use [analysis.mass].")
            self.mass = None

        # Config to load analysis volume
        try:
            self.volume = AnalysisVolumeConfig().load(sec, results)
        except KeyError:
            warn("No analysis volume found. Use [analysis.volume].")
            self.volume = None

        return self
