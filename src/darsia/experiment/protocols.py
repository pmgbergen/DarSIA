"""Module for extracting the physical time of current image."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ImagingInterval:
    """Imaging interval definition."""

    start_date: datetime
    """Start date of the imaging interval."""
    start_id: int
    """Start id of the imaging interval."""
    time_interval: timedelta
    """Time interval between images in the interval."""


class ImagingProtocol:
    """Collection of ImagingInterval objects."""

    def __init__(self, intervals: list[ImagingInterval] = None, pad: int = 5) -> None:
        # NOTE: Assume intervals are provided in chronologically increasing order
        self.intervals = intervals
        self.pad = pad

    def get_datetime(self, file_name: Path) -> datetime:
        # Fetch id from input file
        current_id = int(file_name.stem[-self.pad :])

        # Find correct interval based on id
        interval = None
        for i in self.intervals:
            if current_id >= i.start_id:
                interval = i
            else:
                # Recall assumption of ordered intervals
                break

        if interval is None:
            return None
        else:
            return (
                interval.start_date
                + (current_id - interval.start_id) * interval.time_interval
            )

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
                start_date=datetime.fromisoformat(i["start_date"]),
                start_id=i["start_id"],
                time_interval=timedelta(seconds=i["time_interval"]),
            )
            for i in data["intervals"]
        ]
