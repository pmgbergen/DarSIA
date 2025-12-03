"""Standardized configuration for FluidFlower analysis with parsing from TOML."""

import json
import logging
import tomllib
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from warnings import warn
from typing import Any

logger = logging.getLogger(__name__)


# Utility functions for TOML parsing
def _get_section(data: dict, section: str) -> dict:
    """Utility to get a section from a toml-loaded dictionary."""
    try:
        return data[section]
    except KeyError:
        raise KeyError(f"Section {section} not found.")


def _get_section_from_toml(path: Path | list[Path], section: str) -> dict:
    if isinstance(path, Path):
        data = tomllib.loads(path.read_text())
    elif isinstance(path, list):
        data = {}
        for p in path:
            part = tomllib.loads(p.read_text())
            data.update(part)
    sec = _get_section(data, section)
    return sec


def _get_key(section: dict, key: str, default=None, required=True, type_=None) -> Any:
    """Utility to get a key from a section with type conversion and default value."""
    if required and key not in section:
        raise KeyError(f"Missing key '{key}' in section {section}.")

    if key in section:
        value = section[key]
        return type_(value) if type_ else value
    else:
        return default


def _convert_none(v):
    return None if ((isinstance(v, str) and v.lower() == "none") or v is None) else v


@dataclass
class FluidFlowerRigConfig:
    """Specifications for the FluidFlower rig."""

    width: float = 0  # in meters
    """Width of the FluidFlower rig in meters."""
    height: float = 0  # in meters
    """Height of the FluidFlower rig in meters."""
    dim: int = 2  # spatial dimension (2 or 3)
    """Spatial dimension (2 or 3)."""
    resolution: tuple[int, int] = (500, 1000)
    """Default resolution for images (height, width)."""
    path: Path = field(default_factory=Path)
    """Path to the rig cache file."""

    def load(self, path: Path, results: Path | None = None) -> "FluidFlowerRigConfig":
        sec = _get_section_from_toml(path, "rig")
        self.width = _get_key(sec, "width", required=True, type_=float)
        self.height = _get_key(sec, "height", required=True, type_=float)
        self.dim = _get_key(sec, "dim", required=True, type_=int)
        self.resolution = _get_key(
            sec, "resolution", default=(500, 1000), required=False, type_=tuple
        )
        self.path = _get_key(sec, "path", required=False, type_=Path)
        if not self.path:
            assert results is not None
            self.path = results / "setup" / "rig"

        return self

    def error(self):
        raise ValueError("Use [specs] in the config file to load specs.")


@dataclass
class FluidFlowerDataConfig:
    """Data for the FluidFlower setup.

    Example for TOML section:
        [data]
        folder = "path/to/images"
        format = "JPG"
        baseline = "path/to/baseline.jpg"
        pad = 0
        results = "path/to/results"

    """

    folder: Path = field(default_factory=Path)
    """Path to the folder containing the image data."""
    format: str = "JPG"
    """Format of the image data (e.g., 'JPG', 'PNG')."""
    data: list[Path] = field(default_factory=list)
    """List of paths to the image data."""
    baseline: Path = field(default_factory=Path)
    """Path to the baseline image."""
    pad: int = 0
    """Pad for image names."""
    results: Path = field(default_factory=Path)
    """Path to the results folder."""

    def load(
        self,
        path: Path | list[Path],
        require_data: bool = True,
        require_results: bool = False,
    ) -> "FluidFlowerDataConfig":
        sec = _get_section_from_toml(path, "data")

        # Get folder
        self.folder = _get_key(sec, "folder", required=True, type_=Path)
        if require_data and not self.folder.is_dir():
            raise FileNotFoundError(f"Folder {self.folder} not found.")

        # Get baseline
        self.baseline = self.folder / _get_key(
            sec, "baseline", required=True, type_=Path
        )
        if require_data and not self.baseline.is_file():
            raise FileNotFoundError(f"Baseline image {self.baseline} not found.")

        # Get format
        numeric_part = "".join(filter(str.isdigit, self.baseline.stem))
        self.pad = len(numeric_part) if numeric_part else 0

        # Get data
        self.data = list(sorted(self.folder.glob(f"*{self.baseline.suffix}")))
        if require_data and len(self.data) == 0:
            raise FileNotFoundError(
                f"No image files with suffix {self.baseline.suffix} found in {self.folder}."
            )

        # Get results
        self.results = _get_key(sec, "results", required=True, type_=Path)
        if require_results and not self.results.is_dir():
            raise FileNotFoundError(f"Results folder {self.results} not found.")
        try:
            self.results.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise PermissionError(
                f"Cannot create results directory at {self.results}."
            ) from e

        return self

    def error(self):
        raise ValueError("Use [data] in the config file to load data.")


@dataclass
class FluidFlowerLabelingConfig:
    colored_image: Path = field(default_factory=Path)
    """Path to the segmented file."""
    unite_labels: list[list[int]] = field(default_factory=list)
    """List of lists of labels to be united."""
    water_label: int | None = None
    """Label corresponding to water (if any)."""
    colorchecker_label: int | None = None
    """Label corresponding to colorchecker (if any)."""
    labels: Path = field(default_factory=Path)
    """Path to the labels file."""

    def load(
        self, path: Path, results: Path | None = None
    ) -> "FluidFlowerLabelingConfig":
        """Load labeling config from a toml file from [section]."""
        sec = _get_section_from_toml(path, "labeling")
        self.colored_image = _get_key(sec, "colored_image", required=True, type_=Path)
        self.unite_labels = _get_key(sec, "unite_labels", required=False, type_=list)
        self.water_label = _get_key(sec, "water_label", required=False, type_=int)
        self.colorchecker_label = _get_key(
            sec, "colorchecker_label", required=False, type_=int
        )
        self.labels = _get_key(sec, "labels", required=False, type_=Path)
        if not self.labels:
            assert results is not None
            self.labels = results / "setup" / "labels.npz"
        return self

    def error(self):
        raise ValueError("Use [labeling] in the config file to load labeling.")


@dataclass
class FluidFlowerFaciesConfig:
    id: list[int] = field(default_factory=list)
    """List of facies IDs."""
    props: Path = field(default_factory=Path)
    """Path to the facies properties file."""
    path: Path = field(default_factory=Path)
    """Path to the facies file."""
    groups: dict[int, str] = field(default_factory=dict)
    """Mapping from facies ID to facies."""

    def load(
        self, path: Path, results: Path | None = None
    ) -> "FluidFlowerFaciesConfig":
        """Load facies config from a toml file from [section]."""
        sec = _get_section_from_toml(path, "facies")
        self.id = _get_key(sec, "id", required=True, type_=list)
        self.props = _get_key(sec, "props", required=True, type_=Path)
        self.path = _get_key(sec, "path", required=False, type_=Path)
        self.id_label_map = {i: sec[str(i)]["labels"] for i in self.id}
        if not self.path:
            assert results is not None
            self.path = results / "setup" / "facies.npz"

        return self


@dataclass
class FluidFlowerDepthConfig:
    measurements: Path = field(default_factory=Path)
    """Path to the csv file containing the depth measurements."""
    depth_map: Path = field(default_factory=Path)
    """Path to the depth map file."""

    def load(self, path: Path, results: Path | None = None) -> "FluidFlowerDepthConfig":
        """Load depth config from a toml file from [section]."""
        sec = _get_section_from_toml(path, "depth")
        self.measurements = _get_key(sec, "measurements", required=True, type_=Path)
        self.depth_map = _get_key(sec, "depth_map", required=False, type_=Path)
        if not self.depth_map:
            assert results is not None
            self.depth_map = results / "setup" / "depth_map.npz"
        return self

    def error(self):
        raise ValueError("Use [depth] in the config file to load depth.")


@dataclass
class FluidFlowerProtocolConfig:
    imaging: Path | tuple[Path, str] | None = None
    """Path to the imaging protocol file or (file, sheet)."""
    injection: Path | tuple[Path, str] | None = None
    """Path to the injection protocol file or (file, sheet)."""
    blacklist: Path | tuple[Path, str] | None = None
    """Path to the blacklist protocol file or (file, sheet)."""
    pressure_temperature: Path | tuple[Path, str] | None = None
    """Path to the pressure-temperature protocol file or (file, sheet)."""

    def load(self, path: Path) -> "FluidFlowerProtocolConfig":
        sec = _get_section_from_toml(path, "protocols")
        try:
            imaging_protocol = sec["imaging"]
            if isinstance(imaging_protocol, list):
                self.imaging = (Path(imaging_protocol[0]), imaging_protocol[1])
            elif isinstance(imaging_protocol, str):
                self.imaging = Path(imaging_protocol)
            else:
                raise ValueError(
                    "Imaging protocol must be a string or a list of [path, sheet]."
                )

        except KeyError:
            self.imaging = None

        try:
            injection_protocol = sec["injection"]
            if isinstance(injection_protocol, list):
                self.injection = (Path(injection_protocol[0]), injection_protocol[1])
            elif isinstance(injection_protocol, str):
                self.injection = Path(injection_protocol)
            else:
                raise ValueError(
                    "Injection protocol must be a string or a list of [path, sheet]."
                )
        except KeyError:
            self.injection = None

        try:
            blacklist_protocol = sec["blacklist"]
            if isinstance(blacklist_protocol, list):
                self.blacklist = (Path(blacklist_protocol[0]), blacklist_protocol[1])
            elif isinstance(blacklist_protocol, str):
                self.blacklist = Path(blacklist_protocol)
            else:
                raise ValueError(
                    "Blacklist protocol must be a string or a list of [path, sheet]."
                )
        except KeyError:
            self.blacklist = None

        try:
            pressure_temperature_protocol = sec["pressure_temperature"]
            if isinstance(pressure_temperature_protocol, list):
                self.pressure_temperature = (
                    Path(pressure_temperature_protocol[0]),
                    pressure_temperature_protocol[1],
                )
            elif isinstance(pressure_temperature_protocol, str):
                self.pressure_temperature = Path(pressure_temperature_protocol)
            else:
                raise ValueError(
                    "Pressure-temperature protocol must be a string or a list of [path, sheet]."
                )
        except KeyError:
            self.pressure_temperature = None

        return self

    def error(self):
        raise ValueError(f"Use [protocols] in the config file to load protocols.")


@dataclass
class ColorPathsConfig:
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
    baseline_image_times: list[float] = field(default_factory=list[float])
    """List of image times used for baseline."""
    calibration_image_paths: list[Path] = field(default_factory=list[Path])
    """List of image paths used for calibration."""
    calibration_image_times: list[float] = field(default_factory=list[float])
    """List of image times used for calibration."""
    calibration_file: Path = field(default_factory=Path)
    """Path to the calibration file."""
    baseline_color_spectrum_folder: Path = field(default_factory=Path)
    """Path to the baseline color spectrum file."""
    color_range_file: Path = field(default_factory=Path)
    """Path to the color range file."""
    reference_label: int = 0
    """Label to use as reference for visualization."""

    def load(
        self, path: Path, data: Path | None, results: Path | None = None
    ) -> "ColorPathsConfig":
        """Load color paths config from a toml file from [section]."""
        sec = _get_section_from_toml(path, "color_paths")
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
        self.baseline_image_paths = sorted(
            [
                Path(p) if data is None else data / p
                for p in _get_key(
                    sec,
                    "baseline_image_paths",
                    default=[],
                    required=False,
                    type_=list[Path],
                )
            ]
        )
        self.baseline_image_times = sorted(
            [
                float(t)
                for t in _get_key(
                    sec,
                    "baseline_image_times",
                    default=[],
                    required=False,
                    type_=list[float],
                )
            ]
        )
        self.calibration_image_paths = sorted(
            [
                Path(p) if data is None else data / p
                for p in _get_key(
                    sec,
                    "calibration_image_paths",
                    default=[],
                    required=False,
                    type_=list[Path],
                )
            ]
        )
        self.calibration_image_times = sorted(
            [
                float(t)
                for t in _get_key(
                    sec,
                    "calibration_image_times",
                    default=[],
                    required=False,
                    type_=list[float],
                )
            ]
        )
        if (
            len(self.calibration_image_paths) > 0
            and len(self.calibration_image_times) > 0
        ):
            raise ValueError(
                "Provide either calibration_image_times or calibration_image_paths, not both."
            )
        self.reference_label = _get_key(
            sec, "reference_label", default=0, required=False, type_=int
        )
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
            
            Example:
            [color_paths]
            ignore_labels = [0, 1]
            resolution = 51
            threshold_baseline = 0.0
            threshold_calibration = 0.0
            baseline_images = [
               "relative/path/to/baseline/image1",
               "relative/path/to/baseline/image2"
            ]
            calibration_images = [
               "relative/path/to/calibration/image1",
               "relative/path/to/calibration/image2"
            ]
            reference_label = 0
            calibration_file = "path/to/calibration/file"
            
            """
        )


@dataclass
class ColorToMassConfig:
    calibration_image_paths: list[Path] = field(default_factory=list[Path])
    """List of image paths used for calibration."""
    calibration_image_times: list[float] = field(default_factory=list[float])
    """List of image times used for calibration."""
    calibration_folder: Path = field(default_factory=Path)
    """Path to the calibration folder."""

    def load(
        self, path: Path, data: Path | None, results: Path | None = None
    ) -> "ColorToMassConfig":
        sec = _get_section_from_toml(path, "color_to_mass")
        self.calibration_image_paths = sorted(
            [
                Path(p) if data is None else data / p
                for p in _get_key(
                    sec,
                    "calibration_image_paths",
                    default=[],
                    required=False,
                    type_=list[Path],
                )
            ]
        )
        self.calibration_image_times = sorted(
            [
                float(t)
                for t in _get_key(
                    sec,
                    "calibration_image_times",
                    default=[],
                    required=False,
                    type_=list[float],
                )
            ]
        )
        if (
            len(self.calibration_image_paths) > 0
            and len(self.calibration_image_times) > 0
        ):
            raise ValueError(
                "Provide either calibration_image_times or calibration_image_paths, not both."
            )
        self.calibration_folder = _get_key(
            sec, "calibration_folder", required=False, type_=Path
        )
        if not self.calibration_folder:
            assert results is not None
            self.calibration_folder = results / "calibration" / "color_to_mass"
        return self


@dataclass
class SegmentationConfig:
    labels: list[str] = field(default_factory=list)
    """List of labels for segmentation."""
    thresholds: dict[str, list[float, float]] = field(default_factory=dict)
    """List of (min, max) tuples for thresholding."""
    colors: dict[str, list[int, int, int]] = field(default_factory=dict)
    """List of RGB colors for contours."""
    alphas: dict[str, list[float]] = field(default_factory=dict)
    """List of alpha values for contours."""
    linewidth: int = 2
    """Line width for contour visualization."""

    def load(
        self, sec: dict, section: str, data: Path | None = None
    ) -> "SegmentationConfig":
        sub_sec = _get_section(sec, section)

        # Annotations and line style
        self.labels = _get_key(sub_sec, "labels", required=True, type_=list)
        num_labels = len(self.labels)
        self.linewidth = _get_key(
            sub_sec, "line_width", default=2, required=False, type_=int
        )

        # Cache the threshold values
        _thresholds = _get_key(sub_sec, "thresholds", required=True, type_=list)
        assert len(_thresholds) == num_labels
        thresholds = []
        for _t in _thresholds:
            thresholds.append([_convert_none(t) for t in _t])

        # Define RGB colors for each contour line
        _single_colors = _get_key(sub_sec, "colors", required=True, type_=list)
        assert len(_single_colors) == num_labels
        colors = []
        for _c, _t in zip(_single_colors, thresholds):
            colors.append([_c] * len(_t))

        # Convert from RGB to RGBA
        _alphas = _get_key(sub_sec, "alphas", required=False, type_=list)
        if not _alphas:
            _alphas = [[1.0] * len(t) for t in thresholds]

        # Create mappings
        self.thresholds = dict(zip(self.labels, thresholds))
        self.colors = dict(zip(self.labels, colors))
        self.alphas = dict(zip(self.labels, _alphas))
        return self

    def error(self):
        raise ValueError(
            f"Use [analysis.segmentation] in the config file to load segmentation."
        )


@dataclass
class ImageTimeInterval:
    start: float
    """Start time of the interval."""
    end: float
    """End time of the interval."""
    step: float
    """Step size between images."""
    num: int
    """Number of images in the interval."""

    def __post_init__(self):
        self.step = (self.end - self.start) / self.num if self.num > 0 else 0
        self.num = int((self.end - self.start) / self.step) + 1 if self.step > 0 else 0

    def generate_times(self) -> list[float]:
        return np.unique(np.linspace(self.start, self.end, self.num)).tolist()


@dataclass
class AnalysisData:
    image_times: list[float] = field(default_factory=list)
    """List of image times in hours since experiment start."""
    image_paths: list[Path] = field(default_factory=list)
    """List of image paths corresponding to the image times."""

    def load(self, path: Path, data: Path | None = None) -> "AnalysisData":
        sec = _get_section_from_toml(path, "analysis")
        sub_sec = _get_section(sec, "data")
        self.image_times = sorted(
            [
                float(t)
                for t in _get_key(
                    sub_sec, "image_times", default=[], required=False, type_=list
                )
            ]
        )
        self.image_paths = sorted(
            [
                Path(p) if data is None else data / p
                for p in _get_key(
                    sub_sec, "image_paths", default=[], required=False, type_=list[Path]
                )
            ]
        )
        if len(self.image_times) > 0 and len(self.image_paths) > 0:
            raise ValueError("Provide either image_times or image_paths, not both.")

        # Add times provided as intervals
        try:
            intervals_sec = _get_section(sub_sec, "image_time_interval")

            # Loop through interval sections
            interval_times = []
            for interval_key in intervals_sec.keys():
                interval_data = intervals_sec[interval_key]

                # Create ImageTimeInterval object
                start = _get_key(interval_data, "start", required=True, type_=float)
                end = _get_key(interval_data, "end", required=True, type_=float)
                step = _get_key(interval_data, "step", required=False, type_=float)
                num = _get_key(interval_data, "num", required=False, type_=int)

                # Create interval and generate times
                interval = ImageTimeInterval(start=start, end=end, step=step, num=num)

                # Generate and collect times from this interval
                interval_times.extend(interval.generate_times())

            # Append interval times to existing image_times and sort
            self.image_times.extend(interval_times)
            # Remove duplicates and sort
            self.image_times = sorted(list(set(self.image_times)))

        except KeyError:
            # No image_time_interval section found, which is okay
            pass

        return self

    def error(self):
        raise ValueError(f"Use [analysis] in the config file to load analysis data.")


@dataclass
class FluidFlowerConfig:
    """Meta data for FluidFlower CO2 analysis."""

    def __init__(
        self,
        path: Path | list[Path],
        require_data: bool = False,
        require_results: bool = False,
    ):
        # ! ---- DATA ---- ! #

        # Make sure that path is compatible
        if isinstance(path, list):
            path = [Path(p) for p in path]
        else:
            path = Path(path)

        try:
            self.data: FluidFlowerDataConfig | None = FluidFlowerDataConfig()
            self.data.load(
                path, require_data=require_data, require_results=require_results
            )
        except KeyError:
            self.data = None
            warn(f"Section data not found in {path}, use [data].")

        # ! ---- RIG ---- ! #
        try:
            self.rig: FluidFlowerRigConfig | None = FluidFlowerRigConfig()
            self.rig.load(
                path=path,
                results=self.data.results if self.data else None,
            )
        except KeyError:
            self.rig = None
            warn(f"Section rig not found in {path}, use [rig].")

        # ! ---- LABELING ---- ! #
        try:
            self.labeling: FluidFlowerLabelingConfig | None = (
                FluidFlowerLabelingConfig()
            )
            self.labeling.load(
                path=path,
                results=self.data.results if self.data else None,
            )
        except KeyError:
            self.labeling = None
            warn(f"Section labeling not found in {path}, use [labeling].")

        # ! ---- FACIES ---- ! #
        try:
            self.facies: FluidFlowerFaciesConfig | None = FluidFlowerFaciesConfig()
            self.facies.load(
                path=path,
                results=self.data.results if self.data else None,
            )
        except KeyError:
            self.facies = None
            warn(f"Section facies not found in {path}, use [facies].")

        # ! ---- DEPTH ---- ! #
        try:
            self.depth: FluidFlowerDepthConfig | None = FluidFlowerDepthConfig()
            self.depth.load(
                path=path,
                results=self.data.results if self.data else None,
            )
        except KeyError:
            self.depth = None
            warn(f"Section depth not found in {path}, use [depth].")

        # ! ---- PROTOCOLS ---- ! #
        try:
            self.protocol: FluidFlowerProtocolConfig | None = (
                FluidFlowerProtocolConfig()
            )
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
        except KeyError:
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
        except KeyError:
            self.color_to_mass = None
            warn(f"Section color_to_mass not found in {path}.")

        # ! ---- ANALYSIS DATA ---- ! #

        try:
            self.analysis: AnalysisData | None = AnalysisData()
            self.analysis.load(path, data=self.data.folder if self.data else None)
        except KeyError:
            self.analysis = None
            warn(f"Section analysis not found in {path}.")

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
            FluidFlowerDataConfig().error()
        elif key == "labeling" and not self.labeling:
            FluidFlowerLabelingConfig().error()
        elif key == "depth" and not self.depth:
            FluidFlowerDepthConfig().error()
        elif key == "rig" and not self.rig:
            FluidFlowerRigConfig().error()
        elif key == "protocol" and not self.protocol:
            FluidFlowerProtocolConfig().error()
        elif key == "color_paths" and not self.color_paths:
            ColorPathsConfig().error()
        elif key == "analysis.data" and (not self.analysis or not self.analysis.data):
            AnalysisData().error()
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


@dataclass
class MultiFluidFlowerDataConfig:
    """Data configuration for multiple FluidFlower runs comparison."""

    results: Path = field(default_factory=Path)
    """Path to the results folder for comparison data."""

    def load(self, path: Path | list[Path]) -> "MultiFluidFlowerDataConfig":
        """Load data configuration from TOML file."""
        data_section = _get_section_from_toml(path, "data")
        self.results = _get_key(data_section, "results", required=True, type_=Path)

        # Create results directory if it doesn't exist
        try:
            self.results.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise PermissionError(
                f"Cannot create results directory at {self.results}."
            ) from e

        return self


@dataclass
class EventConfig:
    """Configuration for a single event."""

    event_id: str = ""
    """ID of the event."""
    type: str = ""
    """Type of the event (e.g., 'mass', 'volume')."""
    key: str = ""
    """Key to monitor for this event."""
    relative_threshold: float = 0.01
    """Relative threshold for this event detection, defaults to 1%."""

    def load(self, event_id: str, event_data: dict) -> "EventConfig":
        """Load event configuration from event data dictionary."""
        self.event_id = event_id
        self.type = _get_key(event_data, "type", required=True, type_=str)
        self.key = _get_key(event_data, "key", required=True, type_=str)
        self.relative_threshold = _get_key(
            event_data,
            "relative_threshold",
            default=0.01,
            required=False,
            type_=float,
        )
        return self


@dataclass
class EventsConfig:
    """Configuration for events detection and analysis."""

    events: dict[str, EventConfig] = field(default_factory=dict)
    """Dictionary of individual event configurations keyed by event ID."""
    path: Path = field(default_factory=Path)
    """Path for storage."""

    def load(self, path: Path | list[Path], results: Path | None) -> "EventsConfig":
        """Load events configuration from TOML file."""
        # Load the entire TOML data to access events section
        events_section = _get_section_from_toml(path, "events")

        # Load each individual event
        for event_id, event_data in events_section.items():
            event_config = EventConfig()
            event_config.load(event_id, event_data)
            self.events[event_id] = event_config

        # Set path for storage
        if "path" in events_section:
            self.path = Path(events_section["path"])
        elif results is not None:
            self.path = results / "events" / "events.csv"
        else:
            raise ValueError(
                f"Events path not specified and results path is None in {path}."
            )

        return self


class MultiFluidFlowerConfig:
    """Meta data for multiple FluidFlower CO2 analysis."""

    def __init__(
        self, path: Path, require_data: bool = False, require_results: bool = False
    ):
        """Initialize from a comparison config file like runs_comparison.toml.

        Args:
            path (Path): Path to the comparison config file.
            require_data (bool): Whether to require data section in each run config.
            require_results (bool): Whether to require results section in each run config.

        """
        self.path = path
        self.sub_config: dict[str, FluidFlowerConfig] = {}
        self.events: EventsConfig | None = None
        self.data: MultiFluidFlowerDataConfig | None = None

        # Load the comparison config
        comparison_data = tomllib.loads(path.read_text())

        # Load data configuration if present
        try:
            self.data = MultiFluidFlowerDataConfig()
            self.data.load(path)
        except KeyError:
            raise ValueError(f"Section [data] not found in {path}.")

        # Load individual run config
        if "run" in comparison_data:
            # Allow for common config, to be added to any other run
            if "common" in comparison_data["run"]:
                run_config = comparison_data["run"]["common"]
                common_config_path = path.parent / run_config["config"]
            else:
                common_config_path = None

            # Setup config for single runs - combine with common config if provided
            for run_id, run_config in comparison_data["run"].items():
                if run_id == "common":
                    continue
                config_path = path.parent / run_config["config"]
                self.sub_config[run_id] = FluidFlowerConfig(
                    [config_path, common_config_path]
                    if common_config_path
                    else config_path,
                    require_data=require_data,
                    require_results=require_results,
                )
                logger.info(f"FluidFlowerConfig finished setup for run {run_id}.")

        self.runs = list(self.sub_config.keys())
        """List of run IDs available in the comparison config."""  # Load events configuration if present

        # Events
        try:
            self.events = EventsConfig()
            self.events.load(path, results=self.data.results if self.data else None)
        except KeyError:
            self.events = None
            logger.info(f"Section [events] not found in {path}.")

    def check(self, *sections: str) -> None:
        """Check that all specified sections exist in all run sub_config."""
        for run_id, config in self.sub_config.items():
            try:
                config.check(*sections)
            except ValueError as e:
                raise ValueError(f"Run {run_id}: {e}")
