"""Standardized configuration for FluidFlower analysis with parsing from TOML."""

import json
import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path
from warnings import warn

from .analysis import AnalysisConfig
from .color_paths import ColorPathsConfig
from .color_to_mass import ColorToMassConfig
from .colorchannel_registry import ColorChannelRegistry
from .colorrange import ColorRangeConfig
from .corrections import CorrectionsConfig
from .data import DataConfig
from .depth import DepthConfig
from .download import DownloadConfig
from .facies import FaciesConfig
from .format_registry import FormatRegistry
from .helper import HelperConfig
from .image_porosity import ImagePorosityConfig
from .labeling import LabelingConfig
from .protocol import ProtocolConfig
from .restoration import RestorationConfig
from .rig import RigConfig
from .roi_registry import RoiRegistry
from .segmentation import SegmentationConfig
from .time_data import TimeData
from .video import VideoConfig
from .workflow_utils import WorkflowUtilsConfig

logger = logging.getLogger(__name__)


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

        # ! ---- CORRECTIONS ---- ! #

        try:
            self.corrections: CorrectionsConfig | None = CorrectionsConfig()
            self.corrections.load(path=path)
        except KeyError:
            self.corrections = None
            warn(f"Section corrections not found in {path}, use [corrections].")

        # ! ---- RESTORATION ---- ! #
        try:
            self.restoration: RestorationConfig | None = RestorationConfig()
            self.restoration.load(path=path)
        except KeyError:
            self.restoration = None
            warn(f"Section restoration not found in {path}, use [restoration].")

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

        # ! ---- IMAGE POROSITY ---- ! #
        try:
            self.image_porosity: ImagePorosityConfig | None = ImagePorosityConfig()
            self.image_porosity.load(path=path)
        except KeyError:
            self.image_porosity = None

        # ! ---- PROTOCOLS ---- ! #
        try:
            self.protocol: ProtocolConfig | None = ProtocolConfig()
            self.protocol.load(path)
        except KeyError:
            self.protocol = None
            warn(f"Section protocols not found in {path}, use [protocols].")

        # ! ---- ROI REGISTRY ---- ! #
        # Must be loaded before ColorPathsConfig so that inline [color_paths.roi.*]
        # entries can be injected into the shared registry during color_paths.load().
        try:
            self.roi_registry: RoiRegistry | None = RoiRegistry()
            self.roi_registry.load(path)
        except KeyError:
            self.roi_registry = None

        # ! ---- COLOR PATHS ---- ! #
        try:
            self.color_paths: ColorPathsConfig | None = ColorPathsConfig()
            self.color_paths.load(
                path=path,
                data=self.data.folder if self.data else None,
                results=self.data.results if self.data else None,
                data_registry=self.data.registry if self.data else None,
                roi_registry=self.roi_registry,
            )
        except (ValueError, KeyError):
            # KeyError occurs when [color_paths] section is missing entirely.
            # ValueError covers malformed/incomplete section content.
            self.color_paths = None
            warn(f"Section color_paths not found in {path}.")

        # ! ---- COLOR TO MASS ---- ! #
        try:
            self.color_to_mass: ColorToMassConfig | None = ColorToMassConfig()
            self.color_to_mass.load(
                path=path,
                data=self.data.folder if self.data else None,
                results=self.data.results if self.data else None,
                data_registry=self.data.registry if self.data else None,
                roi_registry=self.roi_registry,
            )
        except (ValueError, KeyError):
            # KeyError occurs when [color_to_mass] section is missing entirely.
            # ValueError covers malformed/incomplete section content.
            self.color_to_mass = None
            warn(f"Section color_to_mass not found in {path}.")

        # ! ---- COLOR RANGE DEFINITIONS ---- ! #
        try:
            self.colorrange: ColorRangeConfig | None = ColorRangeConfig()
            self.colorrange.load(path)
        except KeyError:
            self.colorrange = None

        # ! ---- FORMAT REGISTRY ---- ! #
        try:
            self.format_registry: FormatRegistry | None = FormatRegistry()
            self.format_registry.load(path)
        except KeyError:
            self.format_registry = None
        # ! ---- COLOR CHANNEL REGISTRY ---- ! #
        try:
            self.colorchannel: ColorChannelRegistry | None = ColorChannelRegistry()
            self.colorchannel.load(path)
        except KeyError:
            self.colorchannel = None

        # ! ---- ANALYSIS DATA ---- ! #
        try:
            self.analysis = AnalysisConfig()
            self.analysis.load(
                path,
                data=self.data.folder if self.data else None,
                results=self.data.results if self.data else None,
                data_registry=self.data.registry if self.data else None,
                roi_registry=self.roi_registry,
                format_registry=self.format_registry,
                colorchannel_registry=self.colorchannel,
            )
        except KeyError:
            self.analysis = None
            warn(f"Section analysis not found in {path}, use [analysis].")

        # ! ---- HELPER ---- ! #
        try:
            self.helper = HelperConfig()
            self.helper.load(
                path,
                data=self.data.folder if self.data else None,
                data_registry=self.data.registry if self.data else None,
            )
        except KeyError:
            self.helper = None

        # ! ---- DOWNLOAD CONFIG ---- ! #
        # TODO make utils config
        try:
            self.download = DownloadConfig()
            self.download.load(
                path,
                data=self.data.folder if self.data else None,
                results=self.data.results if self.data else None,
            )
        except KeyError:
            self.download = None
            warn(f"Section download not found in {path}, use [download].")

        # ! ---- UTILS CONFIG ---- ! #
        try:
            self.workflow_utils = WorkflowUtilsConfig()
            self.workflow_utils.load(path)
        except KeyError:
            self.workflow_utils = None

        # ! ---- VIDEO CONFIG ---- ! #
        try:
            self.video = VideoConfig()
            self.video.load(
                path,
                results=self.data.results if self.data else None,
            )
        except KeyError:
            self.video = None

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
                "No mass analysis loaded. Use [analysis.mass] in the config file."
            )
        elif key == "video" and not self.video:
            VideoConfig().error()

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
                "video",
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
