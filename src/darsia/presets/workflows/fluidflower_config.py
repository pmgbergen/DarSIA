"""Standardized configuration for FluidFlower analysis with parsing from TOML."""

import json
from pathlib import Path
import logging
import tomllib
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _get_section(data: dict, section: str) -> dict:
    """Utility to get a section from a toml-loaded dictionary."""
    try:
        return data[section]
    except KeyError:
        raise ValueError(f"Section {section} not found.")


def _get_key(section: dict, key: str, default=None, required=True, type_=None):
    """Utility to get a key from a section with type conversion and default value."""
    if key in section and section[key] is not None:
        value = section[key]
        return type_(value) if type_ else value
    elif required:
        raise ValueError(f"Missing key '{key}' in section {section}.")
    else:
        return default


@dataclass
class FluidFlowerSpecs:
    """Specifications for the FluidFlower setup."""

    width: float = 0  # in meters
    """Width of the FluidFlower setup in meters."""
    height: float = 0  # in meters
    """Height of the FluidFlower setup in meters."""
    dim: int = 2  # spatial dimension (2 or 3)
    """Spatial dimension (2 or 3)."""
    porosity: float = 0  # porosity of the medium
    """Porosity of the medium."""
    resolution: tuple[int, int] = (500, 1000)
    """Default resolution for images (height, width)."""

    def load(self, path: Path, section: str) -> "FluidFlowerSpecs":
        data = tomllib.loads(path.read_text())
        sec = _get_section(data, section)
        self.width = _get_key(sec, "width", required=True, type_=float)
        self.height = _get_key(sec, "height", required=True, type_=float)
        self.dim = _get_key(sec, "dim", required=True, type_=int)
        self.porosity = _get_key(sec, "porosity", required=True, type_=float)
        self.resolution = _get_key(
            sec, "resolution", default=(500, 1000), required=False, type_=tuple
        )
        return self


@dataclass
class FluidFlowerData:
    """Data for the FluidFlower setup."""

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
    log: Path = field(default_factory=Path)
    """Path to the log folder."""

    def load(self, path: Path, section: str) -> "FluidFlowerData":
        data = tomllib.loads(path.read_text())
        sec = _get_section(data, section)
        self.format = sec.get("format", "JPG")
        self.folder = _get_key(sec, "folder", required=True, type_=Path)
        self.data = list(sorted(self.folder.glob(f"*.{self.format}")))
        self.baseline = self.folder / _get_key(
            sec, "baseline", required=True, type_=Path
        )
        self.pad = _get_key(sec, "pad", required=True, type_=int)
        self.results = _get_key(sec, "results", required=True, type_=Path)
        self.log = _get_key(sec, "log", required=True, type_=Path)
        return self


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

    def load(self, path: Path, section: str) -> "FluidFlowerLabelingConfig":
        """Load labeling config from a toml file from [section]."""
        data = tomllib.loads(path.read_text())
        sec = _get_section(data, section)
        self.colored_image = _get_key(sec, "colored_image", required=True, type_=Path)
        self.unite_labels = _get_key(sec, "unite_labels", required=False, type_=list)
        self.water_label = _get_key(sec, "water_label", required=False, type_=int)
        self.colorchecker_label = _get_key(
            sec, "colorchecker_label", required=False, type_=int
        )
        self.labels = _get_key(sec, "labels", required=True, type_=Path)
        return self


@dataclass
class FluidFlowerDepthConfig:
    measurements: Path = field(default_factory=Path)
    """Path to the csv file containing the depth measurements."""
    depth_map: Path = field(default_factory=Path)
    """Path to the depth map file."""

    def load(self, path: Path, section: str) -> "FluidFlowerDepthConfig":
        """Load depth config from a toml file from [section]."""
        data = tomllib.loads(path.read_text())
        sec = _get_section(data, section)
        self.measurements = _get_key(sec, "measurements", required=True, type_=Path)
        self.depth_map = _get_key(sec, "depth_map", required=True, type_=Path)
        return self


@dataclass
class FluidFlowerProtocolConfig:
    imaging: str | tuple[Path, str] | None = None
    """Path to the imaging protocol file or (file, sheet)."""
    injection: str | tuple[Path, str] | None = None
    """Path to the injection protocol file or (file, sheet)."""
    blacklist: str | tuple[Path, str] | None = None
    """Path to the blacklist protocol file or (file, sheet)."""
    pressure_temperature: str | tuple[Path, str] | None = None
    """Path to the pressure-temperature protocol file or (file, sheet)."""

    def load(self, path: Path, section: str) -> "FluidFlowerProtocolConfig":
        data = tomllib.loads(path.read_text())
        sec = _get_section(data, section)
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


@dataclass
class ColorPathsConfig:
    ignore_labels: list[int] = field(default_factory=list)
    """List of labels to ignore in color analysis."""
    resolution: tuple[int, int, int] = (51, 51, 51)
    """Resolution for the color spectrum."""
    threshold_baseline: float = 0.0
    """Threshold for baseline images."""
    threshold_calibration: float = 0.0
    """Threshold for calibration images."""
    baseline_images: list[Path] = field(default_factory=list[Path])
    """List of image paths used for baseline."""
    calibration_images: list[Path] = field(default_factory=list[Path])
    """List of image paths used for calibration."""
    calibration_file: Path = field(default_factory=Path)
    """Path to the calibration file."""
    reference_label: int = 0
    """Label to use as reference for visualization."""

    def load(self, path: Path, section: str) -> "ColorPathsConfig":
        """Load color paths config from a toml file from [section]."""
        data = tomllib.loads(path.read_text())
        sec = _get_section(data, section)
        self.ignore_labels = _get_key(sec, "ignore_labels", required=False, type_=list)
        self.resolution = _get_key(
            sec, "resolution", default=(51,) * 3, required=False, type_=tuple
        )
        self.threshold_baseline = _get_key(
            sec, "threshold_baseline", default=0.0, required=False, type_=float
        )
        self.threshold_calibration = _get_key(
            sec, "threshold_calibration", default=0.0, required=False, type_=float
        )
        self.baseline_images = sorted(
            [
                Path(p)
                for p in _get_key(
                    sec, "baseline_images", required=False, type_=list[Path]
                )
            ]
        )
        self.calibration_images = sorted(
            [
                Path(p)
                for p in _get_key(
                    sec, "calibration_images", required=False, type_=list[Path]
                )
            ]
        )
        self.calibration_file = _get_key(
            sec, "calibration_file", required=True, type_=Path
        )
        self.reference_label = _get_key(
            sec, "reference_label", default=0, required=False, type_=int
        )
        return self


@dataclass
class ColorSignalConfig:
    num_clusters: int = 5
    """Number of clusters to identify background colors."""
    calibration_images: list[Path] = field(default_factory=list[Path])
    """List of image paths used for calibration."""
    calibration_file: Path = field(default_factory=Path)
    """Path to the calibration file."""

    def load(self, path: Path, section: str) -> "ColorSignalConfig":
        data = tomllib.loads(path.read_text())
        sec = _get_section(data, section)
        self.num_clusters = _get_key(
            sec, "num_clusters", default=5, required=False, type_=int
        )
        self.calibration_images = sorted(
            [
                Path(p)
                for p in _get_key(
                    sec, "calibration_images", required=False, type_=list[Path]
                )
            ]
        )
        self.calibration_file = _get_key(
            sec, "calibration_file", required=True, type_=Path
        )
        return self


@dataclass
class MassAnalysisConfig:
    calibration_image_times: list[float] = field(default_factory=list[float])
    """List of image times used for mass calibration."""
    calibration_file: Path = field(default_factory=Path)
    """Path to the mass calibration file."""

    def load(self, path: Path, section: str) -> "MassAnalysisConfig":
        data = tomllib.loads(path.read_text())
        sec = _get_section(data, section)
        self.calibration_image_times = sorted(
            [
                float(t)
                for t in _get_key(
                    sec, "calibration_image_times", required=False, type_=list[float]
                )
            ]
        )
        self.calibration_file = _get_key(
            sec, "calibration_file", required=True, type_=Path
        )
        return self


@dataclass
class FluidFlowerConfig:
    """Meta data for FluidFlower CO2 analysis."""

    def __init__(self, path: Path):
        # ! ---- SPECS ---- ! #
        try:
            self.specs = FluidFlowerSpecs()
            self.specs.load(path, "specs")
        except KeyError:
            raise ValueError(f"Section specs not found in {path}.")

        # ! ---- DATA ---- ! #
        try:
            self.data = FluidFlowerData()
            self.data.load(path, "data")
        except KeyError:
            raise ValueError(f"Section data not found in {path}.")

        # ! ---- LABELING ---- ! #
        try:
            self.labeling = FluidFlowerLabelingConfig()
            self.labeling.load(path, "labeling")
        except KeyError:
            raise ValueError(f"Section labeling not found in {path}.")

        # ! ---- DEPTH ---- ! #
        try:
            self.depth = FluidFlowerDepthConfig()
            self.depth.load(path, "depth")
        except KeyError:
            raise ValueError(f"Section depth not found in {path}.")

        # ! ---- PROTOCOLS ---- ! #
        try:
            self.protocol = FluidFlowerProtocolConfig()
            self.protocol.load(path, "protocols")
        except KeyError:
            raise ValueError(f"Section protocols not found in {path}.")

        # ! ---- COLOR ANALYSIS ---- ! #
        try:
            self.color_paths = ColorPathsConfig()
            self.color_paths.load(path, "color_paths")
        except KeyError:
            self.color_paths = None
            raise UserWarning(f"Section color_paths not found in {path}.")

        try:
            self.color_signal = ColorSignalConfig()
            self.color_signal.load(path, "color_signal")
        except KeyError:
            self.color_signal = None
            raise UserWarning(f"Section color_signal not found in {path}.")

        # ! ---- MASS ANALYSIS ---- ! #

        try:
            self.mass = MassAnalysisConfig()
            self.mass.load(path, "mass")
        except KeyError:
            self.mass = None
            raise UserWarning(f"Section mass not found in {path}.")

        ## ! ---- COMMON DATA ---- ! #
        # common_folder = Path(meta_data["common"]["folder"])

        ## Labels
        # try:
        #    self.labels = common_folder / meta_data["common"]["labels"]
        # except KeyError:
        #    self.labels = None

        ## Reference colorchecker
        # try:
        #    self.ref_colorchecker = (
        #        common_folder / meta_data["common"]["ref_colorchecker"]
        #    )
        # except KeyError:
        #    self.ref_colorchecker = None

        ## ! ---- COLOR ANALYSIS DATA ---- ! #
        # self.color_analysis = {
        #    "format": None,
        #    "baseline_images": None,
        #    "calibration_images": None,
        # }
        # self.color_analysis["format"] = meta_data["color_analysis"].get("format", "JPG")
        # try:
        #    self.color_analysis["baseline_images"] = (
        #        common_folder / meta_data["color_analysis"]["baseline_images"]
        #    )
        # except KeyError:
        #    self.color_analysis["baseline_images"] = None

        # try:
        #    self.color_analysis["calibration_images"] = (
        #        common_folder / meta_data["color_analysis"]["calibration_images"]
        #    )
        # except KeyError:
        #    self.color_analysis["calibration_images"] = None

        ## ! ---- CALIBRATION DATA ---- ! #
        # self.calibration = {
        #    "format": None,
        #    "scaling_image": None,
        #    "mass_images": None,
        # }
        # self.calibration["format"] = meta_data["calibration"].get("format", "JPG")

        ## CO2 calibration data
        # try:
        #    self.co2_calibration = data_folder / meta_data["data"]["co2_calibration"]
        # except KeyError:
        #    self.co2_calibration = None

        ## CO2(g) calibration data
        # try:
        #    self.co2_g_calibration = (
        #        data_folder / meta_data["data"]["co2_g_calibration"]
        #    )
        # except KeyError:
        #    self.co2_g_calibration = None

        ## Scaling calibration data
        # try:
        #    self.calibration["scaling_image"] = meta_data["calibration"][
        #        "scaling_image"
        #    ]
        # except KeyError:
        #    self.calibration["scaling_image"] = None

        ## Mass calibration data
        # try:
        #    self.calibration["mass_images"] = sorted(
        #        Path(meta_data["calibration"]["mass_images"]).glob(
        #            f"*.{self.calibration['format']}"
        #        )
        #    )
        # except KeyError:
        #    self.calibration["mass_images"] = None

        ## FluidFlower
        # try:
        #    self.fluidflower_folder = (
        #        self.results_folder / meta_data["results"]["fluidflower"]
        #    )
        # except KeyError:
        #    self.fluidflower_folder = None

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

    # More results data

    @property
    def log_folder(self):
        """Path to the log folder."""
        return Path.cwd() / "log"

    # Calibration data

    # @property
    # def co2_analysis_data(self):
    #    """Path to the CO2 analysis data."""
    #    return self.fluidflower_folder / "co2_analysis.csv"

    # @property
    # def co2_g_analysis_data(self):
    #    """Path to the CO2 gas analysis data."""
    #    return self.fluidflower_folder / "co2_g_analysis.csv"

    # @property
    # def pw_transformation_g_data(self):
    #    """Path to the pressure-weighted transformation data for gas."""
    #    return self.fluidflower_folder / "pw_transformation_g.csv"

    # @property
    # def pw_transformation_aq_data(self):
    #    """Path to the pressure-weighted transformation data for aqueous phase."""
    #    return self.fluidflower_folder / "pw_transformation_aq.csv"

    # def update(self, key: str, path: Path) -> None:
    #    """Update the folder path for a given key in the meta data.

    #    Args:
    #        key (str): Key to update. Currently only "fluidflower" is supported.
    #        folder (Path): New folder path.

    #    """
    #    # Update the folder path
    #    if key == "fluidflower":
    #        self.fluidflower_folder = path
    #    elif key == "labels":
    #        self.labels = path
    #    else:
    #        raise ValueError(f"Key {key} not recognized.")

    # def save(self, meta: Path) -> None:
    #    """Save the updated meta data to a file.

    #    Args:
    #        meta (Path): Path to the meta data file.

    #    """
    #    # Load existing meta data
    #    meta_data = self.load_meta(meta)

    #    # Make paths relative (if needed)
    #
    #    if "results" not in meta_data:
    #        meta_data["results"] = {}
    #    meta_data["results"]["fluidflower"] = str(
    #        self.fluidflower_folder.relative_to(self.results_folder)
    #    )

    #    # Save the updated meta data
    #    if meta.suffix == ".json":
    #        with open(meta, "w") as f:
    #            json.dump(meta_data, f, indent=4)
    #    elif meta.suffix == ".toml":
    #        import tomli_w

    #        with open(meta, "wb") as f:
    #            f.write(tomli_w.dumps(meta_data).encode("utf-8"))
    #        logger.info(f"Saved updated meta data to {meta}.")
    #    else:
    #        raise ValueError(f"Unsupported meta file format: {meta.suffix}")
