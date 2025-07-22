from pathlib import Path
import json


class FluidFlowerCO2Meta:
    """Meta data for FluidFlower CO2 analysis."""

    def __init__(self, meta: Path):
        # Open meta data - json file
        with open(meta, "r") as f:
            meta_data = json.load(f)

        # Data
        data_folder = Path(meta_data["data"]["folder"])
        format = meta_data["data"].get("format", "JPG")
        self.data = list(sorted(data_folder.glob(f"*.{format}")))
        assert len(self.data) > 0, "No images found in the data folder."

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

        # Results
        self.results_folder = Path(meta_data["results"]["folder"])

        # FluidFlower
        self.fluidflower_folder = (
            self.results_folder / meta_data["results"]["fluidflower"]
        )

    # Results data

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
