"""Color paths configuration for the calibration workflow."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .data_registry import DataRegistry
from .time_data import TimeData
from .utils import _get_key, _get_section_from_toml

logger = logging.getLogger(__name__)


@dataclass
class ColorPathsConfig:
    """Configuration for color paths calibration."""

    num_segments: int = 1
    """Number of segments for the color path."""
    ignore_labels: list[int] = field(default_factory=list)
    """List of labels to ignore in color analysis."""
    resolution: int = 51
    """Resolution for the color spectrum."""
    threshold_baseline: float = 0.0
    """Threshold for baseline images."""
    threshold_calibration: float = 0.0
    """Threshold for calibration images."""
    baseline_image_paths: list[Path] = field(default_factory=list[Path])
    """List of image paths used for baseline."""
    data: TimeData | None = None
    """Calibration data configuration."""
    calibration_file: Path = field(default_factory=Path)
    """Path to the calibration file."""
    baseline_color_spectrum_folder: Path = field(default_factory=Path)
    """Path to the baseline color spectrum file."""
    color_range_file: Path = field(default_factory=Path)
    """Path to the color range file."""
    reference_label: int = 0
    """Label to use as reference for visualization."""

    def load(
        self,
        path: Path,
        data: Path | None,
        results: Path | None = None,
        data_registry: DataRegistry | None = None,
    ) -> "ColorPathsConfig":
        """Load color paths config from a toml file from [section].

        The ``data`` and ``baseline`` keys inside ``[color_paths]`` can be specified
        in two ways:

        **Registry reference** (new, recommended)::

            [color_paths]
            data     = ["calibration1", "calibration2"]
            baseline = "baseline_images"

        Here the values are key name(s) into the global ``[data]`` registry (see
        :class:`~darsia.presets.workflows.config.data_registry.DataRegistry`).

        **Inline sub-section** (legacy / still supported)::

            [color_paths.data.interval.calibration1]
            start = "01:00:00"
            end   = "23:00:00"
            num   = 5
            tol   = "00:10:00"

            [color_paths.baseline.path.calibration]
            paths = ["baseline/DSC00155.JPG", "DSC00160.JPG"]
        """
        # Get section
        sec = _get_section_from_toml(path, "color_paths")

        # Get parameters
        self.num_segments = _get_key(
            sec, "num_segments", default=1, required=False, type_=int
        )
        self.ignore_labels = _get_key(sec, "ignore_labels", required=False, type_=list)
        self.resolution = _get_key(
            sec, "resolution", default=51, required=False, type_=int
        )
        self.threshold_baseline = _get_key(
            sec, "threshold_baseline", default=0.0, required=False, type_=float
        )
        self.threshold_calibration = _get_key(
            sec, "threshold_calibration", default=0.0, required=False, type_=float
        )
        self.reference_label = _get_key(
            sec, "reference_label", default=0, required=False, type_=int
        )

        # Data management – support registry reference or inline sub-section
        baseline_val = sec.get("baseline")
        if isinstance(baseline_val, (str, list)) and data_registry is not None:
            self.baseline_image_paths = data_registry.resolve(baseline_val).image_paths
        else:
            self.baseline_image_paths = TimeData().load(sec["baseline"], data).image_paths

        data_val = sec.get("data")
        if isinstance(data_val, (str, list)) and data_registry is not None:
            self.data = data_registry.resolve(data_val)
        else:
            self.data = TimeData().load(sec["data"], data)

        self.calibration_file = _get_key(
            sec, "calibration_file", required=False, type_=Path
        )
        if not self.calibration_file:
            assert results is not None
            self.calibration_file = results / "calibration" / "color_paths"
        self.baseline_color_spectrum_folder = _get_key(
            sec, "baseline_color_spectrum_folder", required=False, type_=Path
        )
        if not self.baseline_color_spectrum_folder:
            assert results is not None
            self.baseline_color_spectrum_folder = (
                results / "calibration" / "baseline_color_spectrum"
            )
        self.color_range_file = _get_key(
            sec, "color_range_file", required=False, type_=Path
        )
        if not self.color_range_file:
            assert results is not None
            self.color_range_file = results / "calibration" / "color_range"
        return self

    def error(self):
        raise ValueError(
            """Use [color_paths] in the config file to load color paths.

            Example (registry reference):
            ------------------------------

            [color_paths]
            ignore_labels = [0, 1]
            resolution = 51
            threshold_baseline = 0.0
            threshold_calibration = 0.0
            reference_label = 0
            data     = ["calibration1", "calibration2"]
            baseline = "baseline_images"

            Example (inline sub-section):
            ------------------------------

            [color_paths]
            ignore_labels = [0, 1]
            resolution = 51

            [color_paths.baseline.path.calibration]
            paths = [
                "relative/path/to/calibration/image1",
                "relative/path/to/calibration/image2"
            ]

            [color_paths.data.interval.calibration]
            start = "00:00:00"
            end = "10:00:00"
            step = "01:00:00"
            tol = "00:05:00"

            """
        )
