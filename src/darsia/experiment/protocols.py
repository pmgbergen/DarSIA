"""Module for organizing data with equi-distant time intervals."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd
import darsia


@dataclass
class ImagingInterval:
    """Right-open imaging interval."""

    start_date: datetime
    """Start date of the imaging interval."""
    start_id: int
    """Start id of the imaging interval."""
    time_interval: timedelta
    """Time interval between images in the interval."""

    def contains(self, image_id: int) -> bool:
        """Check if the image id is contained in the interval.

        Args:
            image_id (int): Image id to check.

        Returns:
            bool: True if the image id is contained in the interval, False otherwise.

        """
        return self.start_id <= image_id

    def get_datetime(self, image_id: int) -> datetime:
        """Get the datetime of the image based on its id.

        Args:
            image_id (int): Image id to get the datetime for.

        Returns:
            datetime: The datetime of the image.

        """
        return self.start_date + (image_id - self.start_id) * self.time_interval


class ImagingProtocolOld:
    """Collection of ImagingInterval objects.

    Provides methods to get the datetime of an image based on its file name.

    """

    def __init__(
        self, intervals: Optional[list[ImagingInterval]] = None, pad: int = 5
    ) -> None:
        # NOTE: Assume intervals are provided in chronologically increasing order
        self.intervals = intervals
        """List of imaging intervals."""
        self.pad = pad
        """Number of digits in the image id in the file name."""

    def get_datetime(self, file_name: Path) -> Optional[datetime]:
        """Get the datetime of the image based on the file name.

        Args:
            file_name (Path): Path to the image file. The file name should end
                with a number. If contained in a considered imaging interval,
                the id can be correlated to a datetime.

        Returns:
            Optional[datetime]: The datetime of the image. None, if the file name
                is not contained in any of the imaging intervals.

        """
        # Fetch id from input file
        current_id = int(file_name.stem[-self.pad :])

        # Find correct interval based on id
        interval: Optional[ImagingInterval] = None
        for _interval in self.intervals:
            if _interval.contains(current_id):
                interval = _interval
            else:
                # Stop searching. Recall assumption of ordered intervals.
                break

        return interval.get_datetime(current_id) if interval else None

    # ! ---- I/O ----

    def save(self, file_name: Path) -> None:
        """Dump the protocol to a json file allowing to recreate via a load method."""
        data = {
            "pad": self.pad,
            "intervals": [
                {
                    "start_date": i.start_date.isoformat(),
                    "start_id": i.start_id,
                    "time_interval": i.time_interval.total_seconds(),
                }
                for i in self.intervals
            ],
        }
        # Dump to json file
        with open(file_name, "w") as f:
            json.dump(data, f)

    def load(self, file_name: Path) -> None:
        """Load a protocol from a json file."""
        with open(file_name, "r") as f:
            data = json.load(f)

        self.pad = data["pad"]
        self.intervals = [
            ImagingInterval(
                start_date=datetime.fromisoformat(_interval["start_date"]),
                start_id=_interval["start_id"],
                time_interval=timedelta(seconds=_interval["time_interval"]),
            )
            for _interval in data["intervals"]
        ]


class ImagingProtocol:
    def __init__(
        self,
        path: Path | tuple[Path, str],
        pad: int,
        blacklist: Optional[Path | tuple[Path, str]] = None,
    ) -> None:
        self.df = self._load_protocol(path)
        """DataFrame containing the protocol."""
        self.pad = pad
        """Number of digits in the image id in the file name."""

        if blacklist is not None:
            self.blacklist_df = self._load_blacklist(blacklist)
        else:
            self.blacklist_df = pd.DataFrame(columns=["image_id"])

    def image_id(self, path: Path) -> int:
        """Extract image id from file name."""
        return int(path.stem[-self.pad :])

    def is_blacklisted(self, file_name: Path) -> bool:
        """Check if the image is blacklisted based on the file name.

        Args:
            file_name (Path): Path to the image file. The file name should end
                with a number. If contained in a considered imaging interval,
                the id can be correlated to a datetime.
        Returns:
            bool: True if the image is blacklisted, False otherwise.

        """
        if self.blacklist_df.empty:
            return False

        # Fetch id from input file
        current_id = self.image_id(file_name)

        # Find correct interval based on id
        row = self.blacklist_df[self.blacklist_df["image_id"] == current_id]

        return not row.empty

    def get_datetime(self, file_name: Path) -> Optional[datetime]:
        """Get the datetime of the image based on the file name.

        Args:
            file_name (Path): Path to the image file. The file name should end
                with a number. If contained in a considered imaging interval,
                the id can be correlated to a datetime.

        Returns:
            Optional[datetime]: The datetime of the image. None, if the file name
                is not contained in any of the imaging intervals.

        """
        # Fetch id from input file
        current_id = self.image_id(file_name)

        # Find correct interval based on id
        row = self.df[self.df["image_id"] == current_id].iloc[0]

        if row.empty:
            raise ValueError(f"Image id {current_id} not found in protocol.")

        return row["datetime"]

    def _load_protocol(self, path: Path | tuple[Path, str]) -> pd.DataFrame:
        if isinstance(path, list) or isinstance(path, tuple):
            protocol_path = path[0]
            sheet = path[1]
        else:
            protocol_path = path
            sheet = None

        if protocol_path.suffix == ".csv":
            assert sheet is None, "Sheet name should not be provided for CSV files."
            df = pd.read_csv(protocol_path)
        elif protocol_path.suffix in [".xls", ".xlsx"]:
            df = pd.read_excel(protocol_path, sheet_name=sheet)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")

        # Make all columns lowercase
        df.columns = [col.lower() for col in df.columns]

        # Associate image id and time
        # assert "path" in df.columns, "Column 'Path' not found in the protocol file."
        assert "image_id" in df.columns, (
            "Column 'Image_id' not found in the protocol file."
        )
        assert "datetime" in df.columns, (
            "Column 'Datetime' not found in the protocol file."
        )
        if "path" in df.columns:
            paths = df["path"]
        else:
            paths = [None] * len(df)
        images = df["image_id"]
        datetimes = df["datetime"]

        # Convert datetimes to pandas datetime objects
        datetimes = pd.to_datetime(datetimes)

        # Make a new dataframe object and return
        return pd.DataFrame({"path": paths, "image_id": images, "datetime": datetimes})

    def _load_blacklist(self, path: Path | tuple[Path, str]) -> pd.DataFrame:
        if isinstance(path, list) or isinstance(path, tuple):
            protocol_path = path[0]
            sheet = path[1]
        else:
            protocol_path = path
            sheet = None

        if protocol_path.suffix == ".csv":
            assert sheet is None, "Sheet name should not be provided for CSV files."
            df = pd.read_csv(protocol_path)
        elif protocol_path.suffix in [".xls", ".xlsx"]:
            df = pd.read_excel(protocol_path, sheet_name=sheet)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")

        # Expect only one column without any header
        assert df.shape[1] == 1, (
            "Blacklist protocol file should contain only one column."
        )
        df.columns = ["image_id"]
        images = df["image_id"]
        return pd.DataFrame({"image_id": images})

    def find_images_for_datetimes(
        self,
        paths: Path | list[Path],
        datetimes: list[datetime],
    ) -> list[Path]:
        """Find images in the folder that are closest to the specified times.

        Args:
            paths (Path | list[Path]): Path to the folder containing images.
            times (list[float]): List of times (in seconds) to find corresponding images for.

        Returns:
            list[darsia.Image]: List of images corresponding to the specified times.

        """
        # Restrict df from imagign_interval to available image ids
        if isinstance(paths, list):
            all_paths = paths
        else:
            all_paths = list(paths.glob("*"))

        # Remove blacklisted paths
        available_paths = [p for p in all_paths if not self.is_blacklisted(p)]

        # Convert to ID.
        available_image_ids = {self.image_id(p): p for p in available_paths}
        df = self.df[self.df["image_id"].isin(available_image_ids.keys())]

        # Collect the closest images
        closest_available_image_paths = []
        for dt in datetimes:
            closest_available_time = min(
                df["datetime"], key=lambda t: abs((t - dt).total_seconds())
            )
            closest_available_image_path = available_image_ids[
                df[df["datetime"] == closest_available_time]["image_id"].values[0]
            ]
            closest_available_image_paths.append(closest_available_image_path)

        # Return unique paths only but keep order
        return list(dict.fromkeys(closest_available_image_paths))

    def find_ideal_images_for_datetimes(
        self,
        datetimes: list[datetime],
    ) -> list:
        """Find images in the folder that are closest to the specified times.

        Args:
            times (list[float]): List of times (in seconds) to find corresponding images for.

        Returns:
            list: List of images (ids) corresponding to the specified times.

        """
        # Collect the closest images
        image_ids = []

        for dt in datetimes:
            closest_available_time = min(
                self.df["datetime"], key=lambda t: abs((t - dt).total_seconds())
            )
            image_id = self.df[self.df["datetime"] == closest_available_time][
                "image_id"
            ]
            image_ids.append(image_id.values[0])

        # Return unique paths only but keep order
        return list(dict.fromkeys(image_ids))


class InjectionProtocol:
    def __init__(self, path: Path | tuple[Path, str]) -> None:
        self.df = self._load_protocol(path)
        """DataFrame containing the protocol."""

        self.num_injections = len(self.df)
        """Number of injections in the protocol."""

    def injected_mass(
        self, date: datetime, roi: Optional[darsia.Image] = None
    ) -> float:
        """Get the cumulative injected mass until the given date.

        Args:
            date (datetime): Date to get the cumulative injected mass for.
            roi (Optional[darsia.Image]): Region of interest for the mass calculation.

        Returns:
            float: Cumulative injected mass [kg] until the given date.

        """
        # Loop over all injection intervals and sum up the contributions
        mass = 0.0
        for _, row in self.df.iterrows():
            location_x = row["location_x"]
            location_y = row["location_y"]
            start = row["start"]
            end = row["end"]
            rate_mass = row["rate_kg_s"]  # kg/s

            if roi is not None:
                # Check if the injection location is within the ROI
                if not roi.coordinate_system.contains_point(
                    darsia.Coordinate([location_x, location_y])
                ):
                    continue

            # Determine how much time of the interval has passed (between 0 and the full interval)
            if date <= start:
                # No time has passed in this interval
                time_passed = 0.0
            elif start < date < end:
                # Partial time has passed in this interval in
                time_passed = (date - start).total_seconds()
            else:  # date >= end
                # Full time has passed in this interval
                time_passed = (end - start).total_seconds()

            mass += time_passed * rate_mass

        return mass

    # def injected_volume(
    #    self, date: datetime, roi: Optional[darsia.Image] = None
    # ) -> float:
    #    """Determine the cumulative injected volume until the given date.
    #
    #    NOTE: This function assumes that the injection is given in standard conditions.

    #    """

    def _load_protocol(self, path: Path | tuple[Path, str]) -> pd.DataFrame:
        if isinstance(path, list) or isinstance(path, tuple):
            protocol_path = path[0]
            sheet = path[1]
        else:
            protocol_path = path
            sheet = None

        if protocol_path.suffix == ".csv":
            assert sheet is None, "Sheet name should not be provided for CSV files."
            df = pd.read_csv(protocol_path)
        elif protocol_path.suffix in [".xls", ".xlsx"]:
            df = pd.read_excel(protocol_path, sheet_name=sheet)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")

        # Make columns all lowercase
        df.columns = [col.lower() for col in df.columns]

        # Expect columns 'id', 'start', 'end', 'rate_mass'
        assert "id" in df.columns, "Column 'ID' not found in the protocol file."
        assert "location_x" in df.columns, (
            "Column 'Location_X' not found in the protocol file."
        )
        assert "location_y" in df.columns, (
            "Column 'Location_Y' not found in the protocol file."
        )
        assert "start" in df.columns, "Column 'Start' not found in the protocol."
        assert "end" in df.columns, "Column 'End' not found in the protocol file."
        if "rate_sccm" in df.columns:
            assert "density kg/m3" in df.columns, (
                "Column 'Density kg/m3' not found in the protocol file."
            )
            rate = df["rate_sccm"].astype(float)
            density = df["density kg/m3"].astype(float)
            mass_rate_kg_s = rate * density * 1e-6 / 60.0  # kg/s
        elif "rate_ml/min" in df.columns:
            assert "density kg/m3" in df.columns, (
                "Column 'Density kg/m3' not found in the protocol file."
            )
            rate = df["rate_ml/min"].astype(float)
            density = df["density kg/m3"].astype(float)
            mass_rate_kg_s = rate * density * 1e-6 / 60.0  # kg/s
        elif "rate_g/min" in df.columns:
            rate = df["rate_g/min"].astype(float)
            mass_rate_kg_s = rate * 1e-3 / 60.0  # kg/s
        elif "rate_g/s" in df.columns:
            mass_rate_kg_s = df["rate_g/s"].astype(float) * 1e-3  # kg/s
        else:
            assert "rate_kg/s" in df.columns, (
                "Column 'Rate kg/s' not found in the protocol file."
            )
            mass_rate_kg_s = df["rate_kg/s"].astype(float)

        # Generate a new dataframe with correct types
        ids = df["id"].astype(int)
        location_x = df["location_x"].astype(float)
        location_y = df["location_y"].astype(float)
        starts = pd.to_datetime(df["start"])
        ends = pd.to_datetime(df["end"])

        return pd.DataFrame(
            {
                "id": ids,
                "location_x": location_x,
                "location_y": location_y,
                "start": starts,
                "end": ends,
                "rate_kg_s": mass_rate_kg_s,
            }
        )


# Introduce pressure temperature dataclass
@dataclass
class ThermodynamicState:
    pressure: float
    temperature: float


class PressureTemperatureProtocol:
    def __init__(self, path: Path | tuple[Path, str]) -> None:
        self.df = self._load_protocol(path)
        """DataFrame containing the protocol."""

    def get_state(self, date: datetime) -> ThermodynamicState:
        """Get the pressure and temperature at the given date.

        Args:
            date (datetime): Date to get the pressure and temperature for.

        Returns:
            ThermodynamicState: Pressure [bar] and temperature [Celsius] at the given date.

        """
        # Find the two rows surrounding the date
        before = (
            self.df[self.df["datetime"] <= date].iloc[-1]
            if not self.df[self.df["datetime"] <= date].empty
            else None
        )
        after = (
            self.df[self.df["datetime"] >= date].iloc[0]
            if not self.df[self.df["datetime"] >= date].empty
            else None
        )

        if before is None and after is None:
            raise ValueError("Date is outside the range of the protocol.")

        if before is None:
            return ThermodynamicState(
                pressure=after["pressure_bar"], temperature=after["temperature_celsius"]
            )

        if after is None:
            return ThermodynamicState(
                pressure=before["pressure_bar"],
                temperature=before["temperature_celsius"],
            )

        if before["datetime"] == after["datetime"]:
            return ThermodynamicState(
                pressure=before["pressure_bar"],
                temperature=before["temperature_celsius"],
            )
        else:
            # Linear interpolation
            total_seconds = (after["datetime"] - before["datetime"]).total_seconds()
            weight = (date - before["datetime"]).total_seconds() / total_seconds
            pressure = before["pressure_bar"] + weight * (
                after["pressure_bar"] - before["pressure_bar"]
            )
            temperature = before["temperature_celsius"] + weight * (
                after["temperature_celsius"] - before["temperature_celsius"]
            )
            return ThermodynamicState(pressure=pressure, temperature=temperature)

    def _load_protocol(self, path: Path | tuple[Path, str]) -> pd.DataFrame:
        if isinstance(path, list) or isinstance(path, tuple):
            protocol_path = path[0]
            sheet = path[1]
        else:
            protocol_path = path
            sheet = None

        if protocol_path.suffix == ".csv":
            assert sheet is None, "Sheet name should not be provided for CSV files."
            df = pd.read_csv(protocol_path)
        elif protocol_path.suffix in [".xls", ".xlsx"]:
            df = pd.read_excel(protocol_path, sheet_name=sheet)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")

        # Make columns all lowercase
        df.columns = [col.lower() for col in df.columns]

        # Expect columns 'datetime', 'pressure_bar, 'temperature_celsius'
        assert "datetime" in df.columns, (
            "Column 'Datetime' not found in the protocol file."
        )
        assert "pressure_bar" in df.columns, (
            "Column 'Pressure_Bar' not found in the protocol file."
        )
        assert "temperature_celsius" in df.columns, (
            "Column 'Temperature_Celsius' not found in the protocol file."
        )

        # Generate a new dataframe with correct types
        datetimes = pd.to_datetime(df["datetime"])
        pressure_bars = df["pressure_bar"].astype(float)
        temperature_celsius = df["temperature_celsius"].astype(float)
        return pd.DataFrame(
            {
                "datetime": datetimes,
                "pressure_bar": pressure_bars,
                "temperature_celsius": temperature_celsius,
            }
        )
