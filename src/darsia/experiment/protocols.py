"""Module for organizing data with equi-distant time intervals."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


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


class ImagingProtocol:
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
