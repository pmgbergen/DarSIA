"""Color paths configuration for the calibration workflow."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..basis import CalibrationBasis, calibration_basis_folder, parse_calibration_basis
from .data_registry import DataRegistry
from .time_data import TimeData
from .utils import _get_key, _get_section_from_toml

if TYPE_CHECKING:
    from .roi_registry import RoiRegistry

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
    rois: list[str] = field(default_factory=list)
    """Registry key names of ROIs used to restrict the calibration area.

    When non-empty, only pixels that fall inside the union of the listed ROIs
    *and* inside the porosity mask are used during colour-path calibration.
    The keys must be present in the global ROI registry (``[roi.*]`` TOML
    section) **or** be defined as inline sub-sections under
    ``[color_paths.roi.*]``.
    """
    ignore_baseline_spectrum: str = "expanded"
    """Controls which baseline colour spectrum (if any) is used as the ``ignore``
    argument when computing the tracer spectrum and finding the colour path.

    Allowed values:

    - ``"none"``     – do not compute a baseline spectrum at all; pass
                       ``ignore=None`` downstream.
    - ``"baseline"`` – compute the baseline spectrum but **do not** expand it;
                       pass the unexpanded spectrum as ``ignore=baseline_color_spectrum``.
    - ``"expanded"`` – compute the baseline spectrum **and** expand it via
                       linear regression; pass the expanded spectrum as
                       ``ignore=`` *(default – preserves existing behaviour)*.
    """
    basis: CalibrationBasis = CalibrationBasis.LABELS
    """Label-space basis used for calibration (`facies` or `labels`)."""
    histogram_weighting: str = "threshold"
    """Controls how histogram counts are used when fitting the colour path.

    Allowed values:

    - ``"threshold"``  – binary threshold on the normalised histogram; counts are
                         discarded after thresholding *(default – preserves existing
                         behaviour)*.
    - ``"wls"``        – weighted least-squares path fit where per-bin weights are
                         proportional to the normalised histogram probability.
    - ``"wls_sqrt"``   – same as ``"wls"`` but weights are the square-root of the
                         normalised probability, reducing dominance of high-count bins.
    - ``"wls_log"``    – same as ``"wls"`` but weights are ``log(1 + count)``,
                         providing gentle compression of large counts.
    """
    mode: str = "auto"
    """Color-path calibration mode.

    Allowed values:

    - ``"auto"``   – fully automated path fitting *(default)*.
    - ``"manual"`` – start from automated key colors and allow interactive
                     user-controlled postprocessing of key relative colors.
    """
    calibration_scope: str = "full"
    """Calibration scope.

    Allowed values:

    - ``"full"``         – full recalibration (default, preserves existing behaviour).
    - ``"single_label"`` – update only selected labels using stored calibration
                           artifacts as basis.
    """
    target_labels: list[int] = field(default_factory=list)
    """Label ids to update when ``calibration_scope="single_label"``."""
    tracer_color_spectrum_folder: Path = field(default_factory=Path)
    """Path to stored tracer colour spectra used for color-path search."""
    strict_stored_artifacts: bool = False
    """Whether stored artifacts are required in ``single_label`` mode.

    If ``True``, missing stored artifacts raise an error.
    If ``False``, missing tracer spectrum falls back to recomputation.
    """

    def load(
        self,
        path: Path,
        data: Path | None,
        results: Path | None = None,
        data_registry: DataRegistry | None = None,
        roi_registry: "RoiRegistry | None" = None,
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

        **ROI support** (optional):

        ROIs can be referenced by key name from the global ``[roi.*]`` registry::

            [color_paths]
            rois = ["upper_layer", "dense_box"]

        Or defined inline as sub-sections, which are automatically added to the
        shared ROI registry::

            [color_paths.roi.box1]
            name     = "box1"
            corner_1 = [0.1, 0.2]
            corner_2 = [0.5, 0.6]

        When ``rois`` is non-empty, calibration uses only pixels in the union of
        those ROIs intersected with the porosity mask.
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
        self.basis = parse_calibration_basis(
            _get_key(
                sec, "basis", default=CalibrationBasis.LABELS.value, required=False
            )
        )

        # Data management – support registry reference or inline sub-section
        baseline_val = sec.get("baseline")
        if isinstance(baseline_val, (str, list)) and data_registry is not None:
            self.baseline_image_paths = data_registry.resolve(baseline_val).image_paths
        else:
            self.baseline_image_paths = (
                TimeData().load(sec["baseline"], data).image_paths
            )

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
            self.calibration_file = (
                results
                / "calibration"
                / "color_paths"
                / calibration_basis_folder(self.basis)
            )
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
        self.tracer_color_spectrum_folder = _get_key(
            sec, "tracer_color_spectrum_folder", required=False, type_=Path
        )
        if not self.tracer_color_spectrum_folder:
            assert results is not None
            self.tracer_color_spectrum_folder = (
                results / "calibration" / "tracer_color_spectrum"
            )

        # ROI support – registry reference list OR inline sub-sections
        self.rois = _get_key(sec, "rois", default=[], required=False, type_=list)

        # Baseline spectrum mode
        _allowed_ignore = {"none", "baseline", "expanded"}
        raw_ignore = _get_key(
            sec,
            "ignore_baseline_spectrum",
            default="expanded",
            required=False,
            type_=str,
        )
        if raw_ignore not in _allowed_ignore:
            raise ValueError(
                f"Invalid value '{raw_ignore}' for 'ignore_baseline_spectrum'. "
                f"Allowed values are: {sorted(_allowed_ignore)}."
            )
        self.ignore_baseline_spectrum = raw_ignore

        # Histogram weighting mode for colour-path fitting
        _allowed_weighting = {"threshold", "wls", "wls_sqrt", "wls_log"}
        raw_weighting = _get_key(
            sec,
            "histogram_weighting",
            default="threshold",
            required=False,
            type_=str,
        )
        if raw_weighting not in _allowed_weighting:
            raise ValueError(
                f"Invalid value '{raw_weighting}' for 'histogram_weighting'. "
                f"Allowed values are: {sorted(_allowed_weighting)}."
            )
        self.histogram_weighting = raw_weighting

        # Color-path mode
        _allowed_mode = {"auto", "manual"}
        raw_mode = _get_key(
            sec,
            "mode",
            default="auto",
            required=False,
            type_=str,
        )
        if raw_mode not in _allowed_mode:
            raise ValueError(
                f"Invalid value '{raw_mode}' for 'mode'. "
                f"Allowed values are: {sorted(_allowed_mode)}."
            )
        self.mode = raw_mode

        # Calibration scope
        _allowed_scope = {"full", "single_label"}
        raw_scope = _get_key(
            sec,
            "calibration_scope",
            default="full",
            required=False,
            type_=str,
        )
        if raw_scope not in _allowed_scope:
            raise ValueError(
                f"Invalid value '{raw_scope}' for 'calibration_scope'. "
                f"Allowed values are: {sorted(_allowed_scope)}."
            )
        self.calibration_scope = raw_scope

        # Label selection for single-label updates
        raw_targets = sec.get("target_labels", [])
        if isinstance(raw_targets, int):
            self.target_labels = [int(raw_targets)]
        elif isinstance(raw_targets, list):
            self.target_labels = [int(label) for label in raw_targets]
        else:
            raise ValueError(
                "Invalid value for 'target_labels'. Allowed types are int or list[int]."
            )

        self.strict_stored_artifacts = bool(
            _get_key(
                sec,
                "strict_stored_artifacts",
                default=False,
                required=False,
                type_=bool,
            )
        )

        # Handle inline [color_paths.roi.*] sub-sections: parse and inject into registry.
        if "roi" in sec and isinstance(sec["roi"], dict) and roi_registry is not None:
            from .roi import RoiAndLabelConfig, RoiConfig

            for key, entry in sec["roi"].items():
                roi_obj: RoiConfig | RoiAndLabelConfig
                if "label" in entry:
                    roi_obj = RoiAndLabelConfig().load(entry)
                else:
                    roi_obj = RoiConfig().load(entry)
                roi_registry.register(key, roi_obj)
                if key not in self.rois:
                    self.rois.append(key)

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
            ignore_baseline_spectrum = "expanded"  # "none", "baseline", or "expanded"
            histogram_weighting = "threshold"  # "threshold", "wls", "wls_sqrt", or "wls_log"
            mode = "auto"  # "auto" or "manual"
            calibration_scope = "full"  # "full" or "single_label"
            target_labels = [3]  # int or list[int], used for single_label scope
            strict_stored_artifacts = false

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

            Optional ROI restriction (registry reference):
            -----------------------------------------------

            [color_paths]
            rois = ["upper_layer"]

            [roi.upper_layer]
            name     = "upper_layer"
            corner_1 = [0.0, 0.0]
            corner_2 = [1.0, 0.5]

            Optional ROI restriction (inline sub-section):
            ------------------------------------------------

            [color_paths.roi.box1]
            name     = "box1"
            corner_1 = [0.1, 0.2]
            corner_2 = [0.5, 0.6]

            """
        )
