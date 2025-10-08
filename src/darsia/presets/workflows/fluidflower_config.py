"""Standardized configuration for FluidFlower analysis with parsing from TOML."""

import json
from pathlib import Path
import logging
import tomllib
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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

    def load(self, path: Path, section: str) -> dict:
        """Load specifications from a toml file from [section]."""
        data = tomllib.loads(path.read_text())
        try:
            specs_section = data[section]
        except KeyError:
            raise ValueError(f"Section {section} not found in {path}.")
        try:
            self.width = specs_section["width"]
            self.height = specs_section["height"]
            self.dim = specs_section["dim"]
            self.porosity = specs_section["porosity"]
        except KeyError as e:
            raise ValueError(f"Missing key {e} in section {section} of {path}.")


@dataclass
class FluidFlowerData:
    """Data for the FluidFlower setup."""

    folder: Path = field(default_factory=Path)
    """Path to the folder containing the image data."""
    data: list[Path] = field(default_factory=list)
    """List of paths to the image data."""
    baseline: Path = field(default_factory=Path)
    """Path to the baseline image."""
    pad: int = 0
    """Pad for image names."""

    def load(self, path: Path, section: str) -> dict:
        """Load data from a toml file from [section]."""
        data = tomllib.loads(path.read_text())
        try:
            data_section = data[section]
        except KeyError:
            raise ValueError(f"Section {section} not found in {path}.")
        try:
            self.folder = Path(data_section["folder"])
            format = data_section.get("format", "JPG")
            self.data = list(sorted(self.folder.glob(f"*.{format}")))
            if len(self.data) == 0:
                raise ValueError(f"No images found in the data folder {self.folder}.")
            self.baseline = self.folder / data_section["baseline"]
            self.pad = int(data_section["pad"])
        except KeyError as e:
            raise ValueError(f"Missing key {e} in section {section} of {path}.")


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

    def load(self, path: Path, section: str) -> dict:
        """Load labeling config from a toml file from [section]."""
        data = tomllib.loads(path.read_text())
        try:
            label_section = data[section]
        except KeyError:
            raise ValueError(f"Section {section} not found in {path}.")
        try:
            self.colored_image = Path(label_section["colored_image"])
        except KeyError as e:
            raise ValueError(f"Missing key {e} in section {section} of {path}.")
        try:
            self.unite_labels = label_section.get("unite_labels", [])
        except KeyError as e:
            raise UserWarning(f"Missing key {e} in section {section} of {path}.")
        try:
            self.water_label = label_section.get("water_label", None)
        except KeyError as e:
            raise UserWarning(f"Missing key {e} in section {section} of {path}.")
        try:
            self.colorchecker_label = label_section.get("colorchecker_label", None)
        except KeyError as e:
            raise UserWarning(f"Missing key {e} in section {section} of {path}.")
        try:
            self.labels = Path(label_section["labels"])
        except KeyError as e:
            raise UserWarning(f"Missing key {e} in section {section} of {path}.")


@dataclass
class FluidFlowerDepthConfig:
    measurements: Path = field(default_factory=Path)
    """Path to the csv file containing the depth measurements."""
    depth_map: Path = field(default_factory=Path)
    """Path to the depth map file."""

    def load(self, path: Path, section: str) -> dict:
        """Load depth config from a toml file from [section]."""
        data = tomllib.loads(path.read_text())
        try:
            depth_section = data[section]
        except KeyError:
            raise ValueError(f"Section {section} not found in {path}.")
        try:
            self.measurements = Path(depth_section["measurements"])
        except KeyError as e:
            raise ValueError(f"Missing key {e} in section {section} of {path}.")
        try:
            self.depth_map = Path(depth_section["depth_map"])
        except KeyError as e:
            raise UserWarning(f"Missing key {e} in section {section} of {path}.")


@dataclass
class FluidFlowerConfig:
    """Meta data for FluidFlower CO2 analysis."""

    def __init__(self, path: Path):
        # ! ---- LOAD META DATA ---- ! #
        meta_data = self.load_meta(path)

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

        # ! ---- COMMON DATA ---- ! #

        common_folder = Path(meta_data["common"]["folder"])
        self.common_folder = common_folder

        # TODO: Rm?
        ## Common baseline image
        # try:
        #    self.common_baseline = common_folder / meta_data["common"]["baseline"]
        # except KeyError:
        #    self.common_baseline = None

        # Labels
        try:
            self.labels = common_folder / meta_data["common"]["labels"]
        except KeyError:
            self.labels = None

        # Depth measurements
        try:
            self.depth_measurements = common_folder / "depth" / "depth_measurements.csv"
        except KeyError:
            self.depth_measurements = None

        # Depth map
        try:
            self.depth_map = common_folder / "depth" / "depth_map.npz"
        except KeyError:
            self.depth_map = None

        # Reference colorchecker
        try:
            self.ref_colorchecker = (
                common_folder / meta_data["common"]["ref_colorchecker"]
            )
        except KeyError:
            self.ref_colorchecker = None

        # ! ---- COLOR ANALYSIS DATA ---- ! #
        self.color_analysis = {
            "format": None,
            "baseline_images": None,
            "calibration_images": None,
        }
        self.color_analysis["format"] = meta_data["color_analysis"].get("format", "JPG")
        try:
            self.color_analysis["baseline_images"] = (
                common_folder / meta_data["color_analysis"]["baseline_images"]
            )
        except KeyError:
            self.color_analysis["baseline_images"] = None

        try:
            self.color_analysis["calibration_images"] = (
                common_folder / meta_data["color_analysis"]["calibration_images"]
            )
        except KeyError:
            self.color_analysis["calibration_images"] = None

        # ! ---- CALIBRATION DATA ---- ! #
        self.calibration = {
            "format": None,
            "scaling_image": None,
            "mass_images": None,
        }
        self.calibration["format"] = meta_data["calibration"].get("format", "JPG")

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

        # Scaling calibration data
        try:
            self.calibration["scaling_image"] = meta_data["calibration"][
                "scaling_image"
            ]
        except KeyError:
            self.calibration["scaling_image"] = None

        # Mass calibration data
        try:
            self.calibration["mass_images"] = sorted(
                Path(meta_data["calibration"]["mass_images"]).glob(
                    f"*.{self.calibration['format']}"
                )
            )
        except KeyError:
            self.calibration["mass_images"] = None

        # ! ---- PROTOCOLS ---- ! #
        try:
            imaging_protocol = meta_data["protocols"]["imaging"]
            if isinstance(imaging_protocol, list):
                self.imaging_protocol = (
                    Path(imaging_protocol[0]),
                    imaging_protocol[1],
                )
            elif isinstance(imaging_protocol, str):
                self.imaging_protocol = Path(imaging_protocol)
            else:
                raise ValueError(
                    "Imaging protocol must be a string or a list of [path, sheet]."
                )

        except KeyError:
            self.imaging_protocol = None

        try:
            injection_protocol = meta_data["protocols"]["injection"]
            if isinstance(injection_protocol, list):
                self.injection_protocol = (
                    Path(injection_protocol[0]),
                    injection_protocol[1],
                )
            elif isinstance(injection_protocol, str):
                self.injection_protocol = Path(injection_protocol)
            else:
                raise ValueError(
                    "Injection protocol must be a string or a list of [path, sheet]."
                )
        except KeyError:
            self.injection_protocol = None

        try:
            blacklist_protocol = meta_data["protocols"]["blacklist"]
            if isinstance(blacklist_protocol, list):
                self.blacklist_protocol = (
                    Path(blacklist_protocol[0]),
                    blacklist_protocol[1],
                )
            elif isinstance(blacklist_protocol, str):
                self.blacklist_protocol = Path(blacklist_protocol)
            else:
                raise ValueError(
                    "Blacklist protocol must be a string or a list of [path, sheet]."
                )
        except KeyError:
            self.blacklist_protocol = None

        try:
            pressure_temperature_protocol = meta_data["protocols"][
                "pressure_temperature"
            ]
            if isinstance(pressure_temperature_protocol, list):
                self.pressure_temperature_protocol = (
                    Path(pressure_temperature_protocol[0]),
                    pressure_temperature_protocol[1],
                )
            elif isinstance(pressure_temperature_protocol, str):
                self.pressure_temperature_protocol = Path(pressure_temperature_protocol)
            else:
                raise ValueError(
                    "Pressure temperature protocol must be a string or a list of [path, sheet]."
                )
        except KeyError:
            self.pressure_temperature_protocol = None

        # ! ---- RESULTS DATA ---- ! #

        # Results
        self.results_folder = Path(meta_data["results"]["folder"])

        # FluidFlower
        try:
            self.fluidflower_folder = (
                self.results_folder / meta_data["results"]["fluidflower"]
            )
        except KeyError:
            self.fluidflower_folder = None

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

    @property
    def co2_analysis_data(self):
        """Path to the CO2 analysis data."""
        return self.fluidflower_folder / "co2_analysis.csv"

    @property
    def co2_g_analysis_data(self):
        """Path to the CO2 gas analysis data."""
        return self.fluidflower_folder / "co2_g_analysis.csv"

    @property
    def pw_transformation_g_data(self):
        """Path to the pressure-weighted transformation data for gas."""
        return self.fluidflower_folder / "pw_transformation_g.csv"

    @property
    def pw_transformation_aq_data(self):
        """Path to the pressure-weighted transformation data for aqueous phase."""
        return self.fluidflower_folder / "pw_transformation_aq.csv"

    def update(self, key: str, path: Path) -> None:
        """Update the folder path for a given key in the meta data.

        Args:
            key (str): Key to update. Currently only "fluidflower" is supported.
            folder (Path): New folder path.

        """
        # Update the folder path
        if key == "fluidflower":
            self.fluidflower_folder = path
        elif key == "labels":
            self.labels = path
        else:
            raise ValueError(f"Key {key} not recognized.")

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
