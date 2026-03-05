import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FluidFlowerCO2Meta:
    """Meta data for FluidFlower CO2 analysis."""

    def __init__(self, meta: Path):
        # ! ---- LOAD META DATA ---- ! #
        meta_data = self.load_meta(meta)

        # ! ---- RUN DATA ---- ! #

        # Data
        data_folder = Path(meta_data["data"]["folder"])
        format = meta_data["data"].get("format", "JPG")
        self.data = list(sorted(data_folder.glob(f"*.{format}")))
        assert len(self.data) > 0, "No images found in the data folder."

        # Baseline image
        try:
            self.baseline = data_folder / meta_data["data"]["baseline"]
        except KeyError:
            self.baseline = None

        # Pad for image names
        try:
            self.pad = int(meta_data["data"]["pad"])
        except KeyError:
            raise ValueError("Pad for image names must be specified in the meta data.")

        # ! ---- INPUT DATA ---- ! #
        try:
            self.input_folder = Path(meta_data["input"]["folder"])
        except KeyError:
            self.input_folder = None

        # Common segmented baseline image
        try:
            self.segmentation = self.input_folder / meta_data["input"]["segmentation"]
        except KeyError:
            self.segmentation = None

        # ! ---- COMMON DATA ---- ! #

        common_folder = Path(meta_data["common"]["folder"])
        self.common_folder = common_folder

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
                    """Pressure temperature protocol must be a string or a list of """
                    """[path, sheet]."""
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
            import tomllib

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
