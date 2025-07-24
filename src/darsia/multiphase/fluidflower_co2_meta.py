from pathlib import Path
import json


class FluidFlowerCO2Meta:
    """Meta data for FluidFlower CO2 analysis."""

    def __init__(self, meta: Path):
        # Open meta data - json file
        with open(meta, "r") as f:
            meta_data = json.load(f)

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

        # ! ---- COMMON DATA ---- ! #

        common_folder = Path(meta_data["common"]["folder"])

        # Labels
        try:
            self.labels = common_folder / meta_data["common"]["labels"]
        except KeyError:
            self.labels = None

        # Depth map
        try:
            self.depth_map = common_folder / meta_data["common"]["depth_map"]
        except KeyError:
            self.depth_map = None

        # Reference colorchecker
        try:
            self.ref_colorchecker = (
                common_folder / meta_data["common"]["ref_colorchecker"]
            )
        except KeyError:
            self.ref_colorchecker = None

        # ! ---- CALIBRATION DATA ---- ! #

        # Scaling calibration data
        try:
            self.scaling_calibration = (
                data_folder / meta_data["data"]["scaling_calibration"]
            )
        except KeyError:
            self.scaling_calibration = None

        # Mass calibration data
        try:
            mass_calibration_data_folder = (
                data_folder / meta_data["data"]["mass_calibration"]
            )
        except KeyError:
            mass_calibration_data_folder = data_folder
        self.mass_calibration_data = list(
            sorted(mass_calibration_data_folder.glob(f"*.{format}"))
        )

        # ! ---- RESULTS DATA ---- ! #

        # Results
        self.results_folder = Path(meta_data["results"]["folder"])

        # FluidFlower
        self.fluidflower_folder = (
            self.results_folder / meta_data["results"]["fluidflower"]
        )

    # More results data

    @property
    def log_folder(self):
        """Path to the log folder."""
        return self.fluidflower_folder / "log"

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
