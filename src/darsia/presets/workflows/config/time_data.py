"""Configuration for data handling in FluidFlower workflows."""

# Add imports
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .utils import _convert_to_hours, _get_key, _get_section

logger = logging.getLogger(__name__)


@dataclass
class TimeInterval:
    start: float
    """Start time of the interval, relative to experiment start, in hours."""
    end: float
    """End time of the interval, relative to experiment start, in hours."""
    step: float
    """Step size between images, in hours."""
    num: int
    """Number of images in the interval."""
    tol: float | None = None
    """Tolerance for time matching, in hours."""

    def __init__(
        self,
        start: float | str,
        end: float | str,
        step: float | str | None = None,
        num: int = 0,
        tol: float | None = None,
    ):
        self.start = _convert_to_hours(start)
        self.end = _convert_to_hours(end)
        self.step = _convert_to_hours(step or 0.0)
        self.num = num
        self.tol = _convert_to_hours(tol or 0.0)

    def __post_init__(self):
        self.step = (self.end - self.start) / self.num if self.num > 0 else 0
        self.num = (
            int((self.end - self.start) / self.step) + 1
            if not np.isclose(self.step, 0)
            else 0
        )

    def generate_times(self) -> list[float]:
        return np.unique(np.linspace(self.start, self.end, self.num)).tolist()

    def generate_times_with_uncertainty(self) -> list[tuple[float, float]]:
        times = self.generate_times()
        return [(t, self.tol) for t in times]


@dataclass
class ImageTimeData:
    """Data specified as explicit image times."""

    times: list[float] = field(default_factory=list)
    """List of image times in hours since experiment start."""
    times_with_tolerance: list[tuple[float, float]] = field(default_factory=list)
    """List of tuples (time, tolerance) for each image time."""

    def load(self, sec: dict) -> "ImageTimeData":
        """Load explicit image times from config section."""
        try:
            times_sec = _get_section(sec, "time")
            for times_key in times_sec.keys():
                times_data = times_sec[times_key]
                self.times.extend(
                    [
                        _convert_to_hours(t)
                        for t in _get_key(
                            times_data,
                            "times",
                            default=[],
                            required=False,
                            type_=list[float | str],
                        )
                    ]
                )

                tolerance = _convert_to_hours(
                    _get_key(times_data, "tol", required=False) or 0.0
                )
                self.times_with_tolerance.extend([(t, tolerance) for t in self.times])

                # Sort by time
                self.times.sort()
                self.times_with_tolerance.sort(key=lambda x: x[0])

        except KeyError:
            pass

        return self

    def get_times_with_uncertainty(self) -> list[tuple[float, float]]:
        """Return times with associated uncertainty."""
        return self.times_with_tolerance.copy()


@dataclass
class ImageTimeIntervalData:
    """Data specified as time intervals."""

    intervals: dict[str, TimeInterval] = field(default_factory=dict)
    """Dictionary of time intervals keyed by interval name."""

    def load(self, sec: dict) -> "ImageTimeIntervalData":
        """Load time intervals from config section."""
        try:
            intervals_sec = _get_section(sec, "interval")
            for interval_key in intervals_sec.keys():
                interval_data = intervals_sec[interval_key]

                start = _get_key(interval_data, "start", required=True)
                end = _get_key(interval_data, "end", required=True)
                step = _get_key(interval_data, "step", required=False)
                num = _get_key(interval_data, "num", required=False, type_=int)
                tol = _get_key(interval_data, "tol", required=False)

                self.intervals[interval_key] = TimeInterval(
                    start=start, end=end, step=step, num=num, tol=tol
                )
        except KeyError:
            pass

        return self

    def get_times_with_uncertainty(self) -> list[tuple[float, float]]:
        """Return all times from all intervals with associated uncertainty."""
        all_times = []
        for interval in self.intervals.values():
            all_times.extend(interval.generate_times_with_uncertainty())
        return all_times


@dataclass
class PathData:
    """Data specified as direct file paths."""

    paths: list[Path] = field(default_factory=list)
    """List of image file paths."""

    def load(
        self, sec: dict, data_folder: Path | list[Path] | None = None
    ) -> "PathData":
        """Load image paths from config section."""
        try:
            paths_sec = _get_section(sec, "path")
            for paths_key in paths_sec.keys():
                paths_data = paths_sec[paths_key]
                paths = paths_data.get("paths", [])

                if not isinstance(paths, list) and isinstance(paths, str):
                    paths = [paths]

                # Treat paths containing '*' as glob patterns
                for p in paths:
                    if "*" in p:
                        if isinstance(data_folder, list):
                            all_paths = []
                            for folder in data_folder:
                                all_paths.extend(sorted(folder.glob(p)))
                            self.paths.extend(all_paths)
                        else:
                            all_paths = sorted((data_folder or Path(".")).glob(p))
                            self.paths.extend(all_paths)
                    else:
                        candidate = Path(p)
                        if candidate.is_absolute() or data_folder is None:
                            self.paths.append(candidate)
                        elif isinstance(data_folder, list):
                            existing_candidates = [
                                folder / candidate
                                for folder in data_folder
                                if (folder / candidate).exists()
                            ]
                            if len(existing_candidates) > 0:
                                self.paths.extend(existing_candidates)
                            else:
                                self.paths.append(data_folder[0] / candidate)
                        else:
                            self.paths.append(data_folder / candidate)

                # Remove duplicates and sort
                self.paths = sorted(set(self.paths))
        except KeyError:
            pass

        return self

    def validate(self) -> bool:
        """Validate that all paths exist."""
        missing = [p for p in self.paths if not p.exists()]
        if missing:
            logger.warning(f"Missing image paths: {missing}")
            return False
        return True


@dataclass
class TimeData:
    """Manager class combining all data specification modes.

    Supports three modes:
    1. Explicit image_times with tolerance
    2. image_time_interval (start/end/step or start/end/num)
    3. image_paths (direct file paths)

    Data from all modes can be combined together.
    """

    image_time_data: ImageTimeData = field(default_factory=ImageTimeData)
    """Explicit image times."""
    image_interval_data: ImageTimeIntervalData = field(
        default_factory=ImageTimeIntervalData
    )
    """Time intervals with time step."""
    image_path_data: PathData = field(default_factory=PathData)
    """Direct file paths."""

    # Combined results
    image_paths: list[Path] = field(default_factory=list)
    """Combined list of image paths."""
    image_times: list[float] = field(default_factory=list)
    """Combined list of image times."""
    mode: str = ""
    """Primary data mode used: 'times', 'intervals', 'paths', or 'mixed'."""

    def load(self, sec: dict, data_folder: Path | None = None) -> "TimeData":
        """Load all available data modes from config section.

        Args:
            sec: Configuration section dictionary
            data_folder: Base folder for resolving relative paths

        Returns:
            self with all available data loaded
        """
        # Load all three modes
        self.image_time_data.load(sec)
        self.image_interval_data.load(sec)
        self.image_path_data.load(sec, data_folder)

        # Combine all data
        self._combine_data()

        return self

    def _combine_data(self) -> None:
        """Combine all loaded data modes."""

        # Count how many modes have data
        has_times = len(self.image_time_data.times) > 0
        has_intervals = len(self.image_interval_data.intervals) > 0
        has_paths = len(self.image_path_data.paths) > 0

        # Determine mode
        mode_count = sum([has_times, has_intervals, has_paths])
        if mode_count == 0:
            raise ValueError(
                "No data specified. Use one of: 'time', " "'interval', or 'path'"
            )
        elif mode_count > 1:
            self.mode = "mixed"
        elif has_times:
            self.mode = "times"
        elif has_intervals:
            self.mode = "intervals"
        else:
            self.mode = "paths"

        # Combine paths (if any)
        if has_paths:
            self.image_path_data.validate()
            self.image_paths = self.image_path_data.paths.copy()

        # Combine times (remove duplicates and sort)
        combined_times = []
        if has_times:
            combined_times.extend(self.image_time_data.get_times_with_uncertainty())
        if has_intervals:
            combined_times.extend(self.image_interval_data.get_times_with_uncertainty())

        if combined_times:
            # Remove duplicates and sort
            combined_times = sorted(list(set(combined_times)), key=lambda x: x[0])
            self.image_times = [t for t, _ in combined_times]

            logger.info(
                f"Combined {len(self.image_times)} times from "
                f"{'times' if has_times else ''} "
                f"{'intervals' if has_intervals else ''} "
                f"(mode: {self.mode})"
            )

    def get_times_with_uncertainty(self) -> list[tuple[float, float]]:
        """Get all times with associated uncertainty."""
        all_times = []
        if self.image_time_data.times:
            all_times.extend(self.image_time_data.get_times_with_uncertainty())
        if self.image_interval_data.intervals:
            all_times.extend(self.image_interval_data.get_times_with_uncertainty())

        # Remove duplicates and sort
        all_times = sorted(list(set(all_times)), key=lambda x: x[0])
        return all_times

    def error(self):
        raise ValueError(
            f"Use key `data` within the considered subsection in the config file. "
            f"Supported modes: time, interval, or path. "
            f"Multiple modes can be combined."
        )


def load_image_times(
    sec: dict, include_uncertainty: bool = False
) -> list[tuple[float, float]]:
    """Load image times from a toml section together with tolerance."""

    warnings.warn(
        "load_image_times is deprecated. Use TimeData().load() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Start with explicit times
    image_times = sorted(
        [
            _convert_to_hours(t)
            for t in _get_key(
                sec,
                "image_times",
                default=[],
                required=False,
                type_=list[float | str],
            )
        ]
    )
    image_time_tolerance = _convert_to_hours(
        _get_key(sec, "image_time_tolerance", required=False, type_=str | float) or 0.0
    )
    image_times_with_uncertainty = [(t, image_time_tolerance) for t in image_times]

    # Add times provided as intervals
    try:
        intervals_sec = _get_section(sec, "image_time_interval")

        # Loop through interval sections
        interval_times_with_uncertainty = []
        for interval_key in intervals_sec.keys():
            interval_data = intervals_sec[interval_key]

            # Create TimeInterval object
            # Allow start/end/step to be either float or string (HH:MM:SS format)
            start = _get_key(interval_data, "start", required=True)
            end = _get_key(interval_data, "end", required=True)
            step = _get_key(interval_data, "step", required=False)
            num = _get_key(interval_data, "num", required=False, type_=int)
            tol = _get_key(interval_data, "tol", required=False)

            # Create interval and generate times
            interval = TimeInterval(
                start=start, end=end, step=step, num=num, tol=tol
            )
            interval_times_with_uncertainty.extend(
                interval.generate_times_with_uncertainty()
            )

        # Append interval times to existing image_times and sort
        image_times_with_uncertainty.extend(interval_times_with_uncertainty)

    except KeyError:
        # No image_time_interval section found, which is okay
        pass

    # TODO Add events.
    ...

    # Remove duplicates and sort based on first entry of the tuple
    image_times_with_uncertainty = sorted(
        list(set(image_times_with_uncertainty)), key=lambda x: x[0]
    )

    # Return based on user preference
    if not include_uncertainty:
        return [t for t, _ in image_times_with_uncertainty]
    return image_times_with_uncertainty
