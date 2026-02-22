"""Standardized configuration for FluidFlower analysis with parsing from TOML."""

import json
import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from warnings import warn

logger = logging.getLogger(__name__)

from .utils import (
    _get_section,
    _get_key,
    _get_section_from_toml,
)
from .rig import RigConfig
from .time_data import TimeData
from .data import DataConfig
from .labeling import LabelingConfig
from .facies import FaciesConfig
from .depth import DepthConfig
from .protocol import ProtocolConfig
from .color_paths import ColorPathsConfig
from .color_to_mass import ColorToMassConfig
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


@dataclass
class FluidFlowerConfig:
    """Meta data for FluidFlower CO2 analysis."""

    def __init__(
        self,
        path: Path | list[Path],
        require_data: bool,
        require_results: bool,
    ):
        # Make sure that path is compatible
        if isinstance(path, list):
            path = [Path(p) for p in path]
        else:
            path = Path(path)

        # ! ---- DATA ---- ! #
        try:
            self.data: DataConfig | None = DataConfig()
            self.data.load(
                path,
                require_data=require_data,
                require_results=require_results,
            )
        except KeyError:
            self.data = None
            warn(f"Section data not found in {path}, use [data].")

        # ! ---- RIG ---- ! #
        try:
            self.rig: RigConfig | None = RigConfig()
            self.rig.load(
                path=path,
                results=self.data.results if self.data else None,
            )
        except KeyError:
            self.rig = None
            warn(f"Section rig not found in {path}, use [rig].")

        # ! ---- LABELING ---- ! #
        try:
            self.labeling: LabelingConfig | None = LabelingConfig()
            self.labeling.load(
                path=path,
                results=self.data.results if self.data else None,
            )
        except KeyError:
            self.labeling = None
            warn(f"Section labeling not found in {path}, use [labeling].")

        # ! ---- FACIES ---- ! #
        try:
            self.facies: FaciesConfig | None = FaciesConfig()
            self.facies.load(
                path=path,
                results=self.data.results if self.data else None,
            )
        except KeyError:
            self.facies = None
            warn(f"Section facies not found in {path}, use [facies].")

        # ! ---- DEPTH ---- ! #
        try:
            self.depth: DepthConfig | None = DepthConfig()
            self.depth.load(
                path=path,
                results=self.data.results if self.data else None,
            )
        except KeyError:
            self.depth = None
            warn(f"Section depth not found in {path}, use [depth].")

        # ! ---- PROTOCOLS ---- ! #
        try:
            self.protocol: ProtocolConfig | None = ProtocolConfig()
            self.protocol.load(path)
        except KeyError:
            self.protocol = None
            warn(f"Section protocols not found in {path}, use [protocols].")

        # ! ---- COLOR PATHS ---- ! #
        try:
            self.color_paths: ColorPathsConfig | None = ColorPathsConfig()
            self.color_paths.load(
                path=path,
                data=self.data.folder if self.data else None,
                results=self.data.results if self.data else None,
            )
        except ValueError:
            self.color_paths = None
            warn(f"Section color_paths not found in {path}.")

        # ! ---- COLOR TO MASS ---- ! #
        try:
            self.color_to_mass: ColorToMassConfig | None = ColorToMassConfig()
            self.color_to_mass.load(
                path=path,
                data=self.data.folder if self.data else None,
                results=self.data.results if self.data else None,
            )
        except ValueError:
            self.color_to_mass = None
            warn(f"Section color_to_mass not found in {path}.")

        # ! ---- ANALYSIS DATA ---- ! #
        try:
            self.analysis = AnalysisConfig()
            self.analysis.load(
                path,
                data=self.data.folder if self.data else None,
                results=self.data.results if self.data else None,
            )
        except KeyError:
            self.analysis = None
            warn(f"Section analysis not found in {path}, use [analysis].")

        ## Reference colorchecker
        # try:
        #    self.ref_colorchecker = (
        #        common_folder / meta_data["common"]["ref_colorchecker"]
        #    )
        # except KeyError:
        #    self.ref_colorchecker = None

        ## ! ---- CALIBRATION DATA ---- ! #
        # self.calibration = {
        #    "format": None,
        #    "scaling_image": None,
        #    "mass_images": None,
        # }
        # self.calibration["format"] = meta_data["calibration"].get("format", "JPG")

    def _check(self, key: str):
        if key == "data" and not self.data:
            DataConfig().error()
        elif key == "labeling" and not self.labeling:
            LabelingConfig().error()
        elif key == "depth" and not self.depth:
            DepthConfig().error()
        elif key == "rig" and not self.rig:
            RigConfig().error()
        elif key == "protocol" and not self.protocol:
            ProtocolConfig().error()
        elif key == "color_paths" and not self.color_paths:
            ColorPathsConfig().error()
        elif key == "analysis.data" and (not self.analysis or not self.analysis.data):
            TimeData().error()
        elif key == "analysis.segmentation" and (
            not self.analysis or not self.analysis.segmentation
        ):
            SegmentationConfig().error()
        elif key == "analysis.mass" and (not self.analysis or not self.analysis.mass):
            raise ValueError(
                "No mass analysis loaded. Use [analysis.mass] in the config file to load mass analysis."
            )

    def check(self, *args: str) -> None:
        """Check that required components are loaded.

        Args:
            keys (list[str]): List of keys to check. Possible keys are:
                "specs", "data", "labeling", "depth", "protocol", "color_paths",
                "analysis".

        Raises:
            ValueError: If a required component is not loaded.

        """
        for key in args:
            assert key in [
                "analysis",
                "analysis.data",
                "analysis.segmentation",
                "color_paths",
                "color_to_mass",
                "data",
                "depth",
                "facies",
                "labeling",
                "protocol",
                "rig",
            ], f"Key {key} not recognized for checking."
            self._check(key)

    # Loading
    def load_meta(self, meta: Path) -> dict:
        """Load meta data from file. Supports JSON and TOML formats."""
        if meta.suffix == ".json":
            with open(meta, "r") as f:
                meta_data = json.load(f)
        elif meta.suffix == ".toml":
            meta_data = tomllib.loads(meta.read_text())
        else:
            raise ValueError(f"Unsupported meta file format: {meta.suffix}")
        return meta_data
