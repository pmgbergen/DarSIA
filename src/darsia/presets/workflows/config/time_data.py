"""Configuration for data handling in FluidFlower workflows."""

# Add imports
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .utils import _convert_to_hours, _get_key, _get_section, _normalize_time_string

logger = logging.getLogger(__name__)


@dataclass
class TimeInterval:
    start: str = field(
        default="00:00:00",
        metadata={
            "name": "Start time",
            # "help": "Start time of interval (HH:MM:SS format or hours).",
            "placeholder": "Start time HH:MM:SS",
            "group": "Interval",
            "widget": "time",
        },
    )
    """Start time of the interval, relative to experiment start, in HH:MM:SS format."""
    end: str = field(
        default="00:00:00",
        metadata={
            "name": "End time",
            # "help": "End time of interval (HH:MM:SS format or hours).",
            "placeholder": "End time HH:MM:SS",
            "group": "Interval",
            "widget": "time",
        },
    )
    """End time of the interval, relative to experiment start, in HH:MM:SS format."""
    step: str | None = field(
        default=None,
        metadata={
            "name": "Step size",
            # "help": "Step size between images (hours).",
            "placeholder": "Step size HH:MM:SS",
            "group": "Interval",
            "widget": "time",
        },
    )
    """Step size between images, in HH:MM:SS format."""
    num: int | None = field(
        default=None,
        metadata={
            "name": "Number of images",
            # "help": "Number of images in the interval.",
            "placeholder": "number, e.g., 10",
            "group": "Interval",
        },
    )
    """Number of images in the interval."""
    tol: str | None = field(
        default=None,
        metadata={
            "name": "Tolerance",
            # "help": "Tolerance for time matching (hours).",
            "placeholder": "tolerance HH:MM:SS",
            "group": "Interval",
            "widget": "time",
        },
    )
    """Tolerance for time matching, in HH:MM:SS format."""

    def __init__(
        self,
        start: float | str,
        end: float | str,
        step: float | str | None = None,
        num: int | None = None,
        tol: float | str | None = None,
    ):
        self.start = _normalize_time_string(start)
        self.end = _normalize_time_string(end)
        self.step = _normalize_time_string(step) if step is not None else None
        self.num = num
        self.tol = _normalize_time_string(tol) if tol is not None else None

    @property
    def start_hours(self) -> float:
        """Start time in hours."""
        return _convert_to_hours(self.start)

    @property
    def end_hours(self) -> float:
        """End time in hours."""
        return _convert_to_hours(self.end)

    @property
    def step_hours(self) -> float | None:
        """Step size in hours, if defined."""
        return _convert_to_hours(self.step) if self.step is not None else None

    @property
    def tol_hours(self) -> float:
        """Tolerance in hours."""
        return _convert_to_hours(self.tol) if self.tol is not None else 0.0

    @property
    def resolved_num(self) -> int:
        """Effective number of images, resolved from num or derived from step.

        Returns num if explicitly set; otherwise derives from step and start/end times.
        Raises ValueError if neither num nor step is set.
        """
        if self.num is not None:
            return self.num
        if self.step_hours is not None and self.step_hours > 0:
            return int((self.end_hours - self.start_hours) / self.step_hours) + 1
        raise ValueError(
            f"Cannot resolve number of images: num is None and step is "
            f"{self.step!r} (step_hours={self.step_hours})"
        )

    def generate_times(self) -> list[float]:
        return np.unique(
            np.linspace(self.start_hours, self.end_hours, self.resolved_num)
        ).tolist()

    def generate_times_with_uncertainty(self) -> list[tuple[float, float]]:
        times = self.generate_times()
        return [(t, self.tol_hours) for t in times]


@dataclass
class TimeWindow:
    start: str = field(
        default="00:00:00",
        metadata={
            "name": "Start time",
            "help": "Start time of window (HH:MM:SS format or hours).",
            "group": "Window",
            "widget": "time",
        },
    )
    """Start time of the window, relative to experiment start, in HH:MM:SS format."""
    end: str = field(
        default="00:00:00",
        metadata={
            "name": "End time",
            "help": "End time of window (HH:MM:SS format or hours).",
            "group": "Window",
            "widget": "time",
        },
    )
    """End time of the window, relative to experiment start, in HH:MM:SS format."""
    step: str | None = field(
        default=None,
        metadata={
            "name": "Step size",
            "help": "Step size (preserved but not used for window time generation).",
            "group": "Window",
            "widget": "time",
        },
    )
    """Step size, preserved for round-tripping but not used for window time generation."""

    def __init__(
        self,
        start: float | str,
        end: float | str,
        step: float | str | None = None,
    ):
        self.start = _normalize_time_string(start)
        self.end = _normalize_time_string(end)
        self.step = _normalize_time_string(step) if step is not None else None

    @property
    def start_hours(self) -> float:
        """Start time in hours."""
        return _convert_to_hours(self.start)

    @property
    def end_hours(self) -> float:
        """End time in hours."""
        return _convert_to_hours(self.end)

    @property
    def step_hours(self) -> float | None:
        """Step size in hours, if defined."""
        return _convert_to_hours(self.step) if self.step is not None else None


@dataclass
class ImageTimeData:
    """Data specified as explicit image times."""

    times: list[float] = field(
        default_factory=list,
        metadata={
            "name": "Image times",
            "help": "List of image times (HH:MM:SS format or hours).",
            "group": "Times",
            "hidden": True,
        },
    )
    """List of image times in hours since experiment start."""
    times_with_tolerance: list[tuple[float, float]] = field(
        default_factory=list,
        metadata={
            "hidden": True,
        },
    )
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
    windows: dict[str, TimeWindow] = field(default_factory=dict)
    """Dictionary of time windows keyed by window name."""

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

                # Route to TimeWindow only if both num and step are absent
                if num is None and step is None:
                    self.windows[interval_key] = TimeWindow(
                        start=start,
                        end=end,
                    )
                else:
                    # Create TimeInterval; resolved_num will derive from step if needed
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

    paths: list[Path] = field(
        default_factory=list,
        metadata={
            "name": "Image paths",
            "help": (
                "List of image file paths (supports glob patterns with *). "
                "Format: file1.jpg, file2.jpg"
            ),
            "group": "Paths",
            "placeholder": "file1.jpg, file2.jpg, DSC*.JPG, *",
        },
    )
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
    image_windows: list[TimeWindow] = field(default_factory=list)
    mode: str = ""
    """Primary data mode used: 'times', 'intervals', 'paths', or 'mixed'."""

    def _combine_data(self) -> None:
        """Combine all loaded data modes."""

        # Count how many modes have data
        has_times = len(self.image_time_data.times) > 0
        has_intervals = len(self.image_interval_data.intervals) > 0
        has_windows = len(self.image_interval_data.windows) > 0
        has_paths = len(self.image_path_data.paths) > 0

        # Determine mode
        mode_count = sum([has_times, has_intervals, has_windows, has_paths])
        if mode_count == 0:
            raise ValueError(
                "No data specified. Use one of: 'time', 'interval', or 'path'"
            )
        elif mode_count > 1:
            self.mode = "mixed"
        elif has_times:
            self.mode = "times"
        elif has_intervals:
            self.mode = "intervals"
        elif has_windows:
            self.mode = "windows"
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

        # Collect windows.
        if has_windows:
            self.image_windows = [w for w in self.image_interval_data.windows.values()]

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
            "Use key `data` within the considered subsection in the config file. "
            "Supported modes: time, interval, or path. "
            "Multiple modes can be combined."
        )
