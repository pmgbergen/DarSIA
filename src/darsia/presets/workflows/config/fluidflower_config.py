"""Standardized configuration for FluidFlower analysis with parsing from TOML."""

import logging
from pathlib import Path
from warnings import warn

from .analysis import AnalysisConfig
from .calibration import CalibrationConfig
from .color_embedding_registry import ColorEmbeddingRegistry
from .corrections import CorrectionsConfig
from .data import DataConfig
from .depth import DepthConfig
from .download import DownloadConfig
from .facies import FaciesConfig
from .format_registry import FormatRegistry
from .helper import HelperConfig
from .image_porosity import ImagePorosityConfig
from .labeling import LabelingConfig
from .protocols import ProtocolsConfig
from .restoration import RestorationConfig
from .rig import RigConfig
from .roi_registry import RoiRegistry
from .segmentation import SegmentationConfig
from .time_data import TimeData
from .video import VideoConfig
from .workflow_utils import WorkflowUtilsConfig

logger = logging.getLogger(__name__)


def _load_section(
    config,
    attr_name: str,
    config_obj,
    load_fn,
    warn_on_missing: bool = True,
    exception_types: tuple = (KeyError,),
) -> None:
    try:
        setattr(config, attr_name, config_obj)
        load_fn()
    except exception_types:
        setattr(config, attr_name, None)
        if warn_on_missing:
            warn(f"Section {attr_name} not found, use [{attr_name}].")


def _load_image_porosity(config, path: Path) -> None:
    try:
        config.image_porosity = ImagePorosityConfig()
        config.image_porosity.load(path=path)
        if not config.image_porosity.active:
            config.image_porosity = None
    except KeyError:
        config.image_porosity = None
        warn("Section image_porosity not found, use [image_porosity].")


class FluidFlowerConfig:
    """Meta data for FluidFlower CO2 analysis."""

    def __init__(
        self,
        path: Path | list[Path],
        require_data: bool,
        require_results: bool,
    ):
        if isinstance(path, list):
            path = [Path(p) for p in path]
        else:
            path = Path(path)

        self.data: DataConfig | None = None
        self.rig: RigConfig | None = None
        self.corrections: CorrectionsConfig | None = None
        self.restoration: RestorationConfig | None = None
        self.labeling: LabelingConfig | None = None
        self.facies: FaciesConfig | None = None
        self.depth: DepthConfig | None = None
        self.image_porosity: ImagePorosityConfig | None = None
        self.protocols: ProtocolsConfig | None = ProtocolsConfig()
        self.roi_registry: RoiRegistry | None = None
        self.color: ColorEmbeddingRegistry | None = None
        self.calibration: CalibrationConfig | None = None
        self.format_registry: FormatRegistry | None = None
        self.analysis: AnalysisConfig | None = None
        self.helper: HelperConfig | None = None
        self.download: DownloadConfig | None = None
        self.workflow_utils: WorkflowUtilsConfig | None = None
        self.video: VideoConfig | None = None

        _load_section(
            self,
            "data",
            DataConfig(),
            lambda: self.data.load(
                path,
                require_data=require_data,
                require_results=require_results,
            ),
            warn_on_missing=True,
        )

        _load_section(
            self,
            "rig",
            RigConfig(),
            lambda: self.rig.load(
                path=path,
                results=self.data.results if self.data else None,
            ),
            warn_on_missing=True,
        )

        _load_section(
            self,
            "corrections",
            CorrectionsConfig(),
            lambda: self.corrections.load(path=path),
            warn_on_missing=True,
        )

        _load_section(
            self,
            "restoration",
            RestorationConfig(),
            lambda: self.restoration.load(path=path),
            warn_on_missing=True,
        )

        _load_section(
            self,
            "labeling",
            LabelingConfig(),
            lambda: self.labeling.load(
                path=path,
                results=self.data.results if self.data else None,
            ),
            warn_on_missing=True,
        )

        _load_section(
            self,
            "facies",
            FaciesConfig(),
            lambda: self.facies.load(
                path=path,
                results=self.data.results if self.data else None,
            ),
            warn_on_missing=True,
        )

        _load_section(
            self,
            "depth",
            DepthConfig(),
            lambda: self.depth.load(
                path=path,
                results=self.data.results if self.data else None,
            ),
            warn_on_missing=True,
        )

        _load_image_porosity(self, path)

        _load_section(
            self,
            "protocols",
            ProtocolsConfig(),
            lambda: self.protocols.load(path),
            warn_on_missing=True,
        )

        _load_section(
            self,
            "roi_registry",
            RoiRegistry(),
            lambda: self.roi_registry.load(path),
            warn_on_missing=False,
        )

        _load_section(
            self,
            "color",
            ColorEmbeddingRegistry(),
            lambda: self.color.load(
                path=path,
                data=self.data.folder if self.data else None,
                results=self.data.results if self.data else None,
                data_registry=self.data.registry if self.data else None,
                roi_registry=self.roi_registry,
            ),
            warn_on_missing=True,
            exception_types=(ValueError, KeyError, NotImplementedError),
        )

        _load_section(
            self,
            "calibration",
            CalibrationConfig(),
            lambda: self.calibration.load(
                path=path,
                data=self.data.folder if self.data else None,
                data_registry=self.data.registry if self.data else None,
                color_embedding_registry=self.color,
            ),
            warn_on_missing=False,
            exception_types=(ValueError, KeyError),
        )

        _load_section(
            self,
            "format_registry",
            FormatRegistry(),
            lambda: self.format_registry.load(path),
            warn_on_missing=False,
        )

        _load_section(
            self,
            "analysis",
            AnalysisConfig(),
            lambda: self.analysis.load(
                path,
                data=self.data.folder if self.data else None,
                results=self.data.results if self.data else None,
                data_registry=self.data.registry if self.data else None,
                roi_registry=self.roi_registry,
                format_registry=self.format_registry,
                color_embedding_registry=self.color,
            ),
            warn_on_missing=True,
        )

        _load_section(
            self,
            "helper",
            HelperConfig(),
            lambda: self.helper.load(
                path,
                data=self.data.folder if self.data else None,
                data_registry=self.data.registry if self.data else None,
                format_registry=self.format_registry,
                roi_registry=self.roi_registry,
            ),
            warn_on_missing=False,
        )

        _load_section(
            self,
            "download",
            DownloadConfig(),
            lambda: self.download.load(
                path,
                data=self.data.folder if self.data else None,
                results=self.data.results if self.data else None,
                data_registry=self.data.registry if self.data else None,
            ),
            warn_on_missing=True,
        )

        _load_section(
            self,
            "workflow_utils",
            WorkflowUtilsConfig(),
            lambda: self.workflow_utils.load(path),
            warn_on_missing=False,
        )

        _load_section(
            self,
            "video",
            VideoConfig(),
            lambda: self.video.load(
                path,
                results=self.data.results if self.data else None,
            ),
            warn_on_missing=False,
        )

    def _check(self, key: str):
        if key == "data" and not self.data:
            DataConfig().error()
        elif key == "labeling" and not self.labeling:
            LabelingConfig().error()
        elif key == "depth" and not self.depth:
            DepthConfig().error()
        elif key == "rig" and not self.rig:
            RigConfig().error()
        elif key == "protocols" and not self.protocols:
            ProtocolsConfig().error()
        elif key == "color":
            if not self.color:
                raise ValueError(
                    "No color embedding registry loaded. Use [color.path.*], "
                    "[color.range.*], or [color.channel.*]."
                )
        elif key == "calibration" and (not self.calibration):
            raise ValueError(
                "No color calibration entrypoint loaded. Use [calibration]."
            )
        elif key == "calibration.color" and (
            not self.calibration or not self.calibration.color
        ):
            raise ValueError(
                "No color calibration entrypoint loaded. Use [calibration.color]."
            )
        elif key == "calibration.mass" and (
            not self.calibration or not self.calibration.mass
        ):
            raise ValueError(
                "No mass calibration entrypoint loaded. Use [calibration.mass]."
            )
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
            keys (list[str]): List of keys to check.

        Raises:
            ValueError: If a required component is not loaded.

        """
        for key in args:
            assert key in [
                "analysis",
                "analysis.data",
                "analysis.mass",
                "analysis.segmentation",
                "calibration",
                "calibration.color",
                "calibration.mass",
                "color",
                "corrections",
                "data",
                "depth",
                "download",
                "facies",
                "format_registry",
                "helper",
                "image_porosity",
                "labeling",
                "protocols",
                "restoration",
                "roi_registry",
                "rig",
                "video",
                "workflow_utils",
            ], f"Key {key} not recognized for checking."
            self._check(key)
