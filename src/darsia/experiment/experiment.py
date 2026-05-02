"""Abstract experiment class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_left
from datetime import datetime
from pathlib import Path
from typing import Optional

import darsia
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.config.time_data import TimeWindow


# TODO rm.
class Experiment(ABC):
    @property
    @abstractmethod
    def atmospheric_pressure(self): ...

    @property
    @abstractmethod
    def temperature(self): ...

    @property
    @abstractmethod
    def injection_start(self): ...

    @property
    @abstractmethod
    def injection_end(self): ...

    def hours_since_start(self, date: datetime) -> float:
        """Hours since start of the experiment."""
        return (date - self.injection_start).total_seconds() / 3600


class ProtocolledExperiment:
    def __init__(
        self,
        data: list[Path],
        imaging_protocol: Path | tuple[Path, str] | dict[Path, Path | tuple[Path, str]],
        injection_protocol: Optional[Path | tuple[Path, str]] = None,
        pressure_temperature_protocol: Optional[Path | tuple[Path, str]] = None,
        blacklist_protocol: Optional[Path | tuple[Path, str]] = None,
        pad: int = 5,
    ):
        self.data = data
        """Pool of data paths."""
        if isinstance(imaging_protocol, dict):
            self.imaging_protocol = None
            self.imaging_protocols = {
                Path(folder): darsia.ImagingProtocol(protocol, pad, blacklist_protocol)
                for folder, protocol in imaging_protocol.items()
            }
            self._resolved_protocol_folders = sorted(
                [
                    (folder.resolve(), protocol)
                    for folder, protocol in self.imaging_protocols.items()
                ],
                key=lambda item: len(item[0].parts),
                reverse=True,
            )
        else:
            self.imaging_protocol = darsia.ImagingProtocol(
                imaging_protocol, pad, blacklist_protocol
            )
            self.imaging_protocols = None
            self._resolved_protocol_folders = []
        """Imaging protocol(s)."""
        self.injection_protocol = darsia.InjectionProtocol(injection_protocol)
        """Injection protocol."""
        self.pressure_temperature_protocol = darsia.PressureTemperatureProtocol(
            pressure_temperature_protocol
        )
        """Pressure and temperature protocol."""

        # Reference date
        self.experiment_start = self.injection_protocol.df["start"].min()
        """Start of the experiment."""
        self._path_protocol_cache: dict[Path, darsia.ImagingProtocol] = {}
        self._parent_protocol_cache: dict[Path, darsia.ImagingProtocol] = {}
        self._available_timeline_cache: dict[
            tuple[tuple[Path, ...], tuple[int, ...]],
            tuple[list[float], list[Path], list[int]],
        ] = {}
        if self.imaging_protocol is not None:
            self._timeline_cache_token = (id(self.imaging_protocol),)
        else:
            assert self.imaging_protocols is not None
            self._timeline_cache_token = tuple(
                id(protocol) for _, protocol in self.imaging_protocols.items()
            )

    @classmethod
    def init_from_config(cls, config: FluidFlowerConfig):
        assert config.data is not None
        assert config.protocol is not None
        if len(config.data.folders) > 1 and not isinstance(
            config.protocol.imaging, dict
        ):
            raise ValueError(
                "Multiple [data].folders require [protocols].imaging to be a per-folder table."
            )
        return cls(
            data=config.data.data,
            imaging_protocol=config.protocol.imaging,
            injection_protocol=config.protocol.injection,
            pressure_temperature_protocol=config.protocol.pressure_temperature,
            blacklist_protocol=config.protocol.blacklist,
            pad=config.data.pad,
        )

    def time_since_start(self, date: datetime) -> float:
        """Hours since start of the experiment.

        Args:
            date (datetime): Date to compute the time since start for.

        Returns:
            float: Time since start in hours.

        """
        return (date - self.experiment_start).total_seconds() / 3600

    def find_images_for_paths(self, paths: list[Path]) -> list[Path]:
        """Find image paths for given paths.

        Args:
            paths (list[Path]): Paths to search for.

        Returns:
            list[Path]: Image paths found for the given paths.

        """
        available_paths: list[Path] = []
        for path in paths:
            if not self.is_blacklisted(path):
                available_paths.append(path)
        return available_paths

    def find_images_for_time_windows(
        self,
        windows: list[TimeWindow],
        data: list[Path] | None = None,
    ) -> list[Path]:
        """Find image paths for given time windows since start of the experiment.

        Args:
            windows (list[TimeWindow]): Time windows to search for.
            data (list[Path], optional): Pool of data paths to search in. If None,
                uses the experiment's data pool.

        """
        source_paths = data or self.data

        available_seconds, available_paths, _ = self._available_timeline(source_paths)

        if not available_paths:
            raise ValueError("No available images found in the specified paths.")

        # Find selected paths within the time interval.
        selected_paths: list[Path] = []
        for window in windows:
            start_in_seconds = window.start * 3600
            end_in_seconds = window.end * 3600
            selected_paths.extend(
                p
                for (sec, p) in zip(available_seconds, available_paths)
                if start_in_seconds <= sec <= end_in_seconds
            )

        # Remove duplicates and sort by time.
        selected_paths = list(dict.fromkeys(selected_paths))
        selected_paths.sort(key=lambda p: self.get_datetime(p))

        return selected_paths

    def find_images_for_times(
        self,
        times: float | list[float],
        tol: float | None = None,
        data: list[Path] | None = None,
    ) -> list[Path]:
        """Find image paths for given times since start of the experiment.

        Args:
            times (list[float]): Times since start in hours.
            tol (float | None, optional): Maximum allowed absolute time distance in
                seconds between requested and matched image times. Inclusive when
                provided.
            data (list[Path], optional): Pool of data paths to search in. If None,
                uses the experiment's data pool.

        """
        times_is_list = isinstance(times, list)
        if not times_is_list:
            times = [times]
        datetimes = [self.experiment_start + darsia.timedelta(hours=t) for t in times]
        source_paths = data or self.data

        available_seconds, available_paths, available_indices = (
            self._available_timeline(source_paths)
        )

        if not available_paths:
            raise ValueError("No available images found in the specified paths.")

        requested_seconds = [
            (dt - self.experiment_start).total_seconds() for dt in datetimes
        ]
        selected_paths: list[Path] = []
        for requested in requested_seconds:
            min_index = self._closest_index(
                available_seconds, available_indices, requested
            )
            min_distance = abs(available_seconds[min_index] - requested)
            if tol is None or min_distance <= tol:
                selected_paths.append(available_paths[min_index])
        paths = list(dict.fromkeys(selected_paths))
        if times_is_list:
            return paths
        else:
            return paths[0] if paths else None

    def get_datetime(self, path: Path) -> datetime:
        """Get datetime for a path using the matching imaging protocol."""
        return self._protocol_for_path(path).get_datetime(path)

    def is_blacklisted(self, path: Path) -> bool:
        """Check blacklist state for a path using the matching imaging protocol."""
        return self._protocol_for_path(path).is_blacklisted(path)

    def _protocol_for_path(self, path: Path) -> darsia.ImagingProtocol:
        if self.imaging_protocol is not None:
            return self.imaging_protocol

        # NOTE: The correct way would be to use:
        # resolved_path = path.resolve()
        # However, this can be very expensive for large datasets.
        # We take the risk, and use:
        resolved_path = path
        resolved_parent = resolved_path.parent
        parent_cached = self._parent_protocol_cache.get(resolved_parent)
        if parent_cached is not None:
            return parent_cached

        cached = self._path_protocol_cache.get(resolved_path)
        if cached is not None:
            self._parent_protocol_cache[resolved_parent] = cached
            return cached

        for resolved_folder, protocol in self._resolved_protocol_folders:
            try:
                resolved_path.relative_to(resolved_folder)
                self._path_protocol_cache[resolved_path] = protocol
                self._parent_protocol_cache[resolved_parent] = protocol
                return protocol
            except ValueError:
                continue
        raise ValueError(f"No imaging protocol configured for image path: {path}")

    # NOTE: The followoing method vectorizes the above logic. Kept for future reference.

    # def _protocol_for_paths(self, paths: list[Path]) -> list[darsia.ImagingProtocol]:
    #     if self.imaging_protocol is not None:
    #         # The single-protocol case.
    #         protocols = [self.imaging_protocol] * len(paths)
    #     else:
    #         # The multi-protocol case.
    #         parents = [path.parent for path in paths]
    #         protocol_map = {
    #             folder: protocol for folder, protocol in self._resolved_protocol_folders
    #         }
    #         protocols = [protocol_map.get(parent_id, None) for parent_id in parents]
    #     return protocols

    def iter_available(self, paths: list[Path]) -> list[tuple[int, Path, datetime]]:
        available: list[tuple[int, Path, datetime]] = []
        for idx, path in enumerate(paths):
            try:
                protocol = self._protocol_for_path(path)
                if protocol.is_blacklisted(path):
                    continue
                date = protocol.get_datetime(path)
            except ValueError:
                continue
            if date is None:
                continue
            available.append((idx, path, date))
        return available

    def _available_timeline(
        self, paths: list[Path]
    ) -> tuple[list[float], list[Path], list[int]]:
        cache_key = (tuple(paths), self._timeline_cache_token)
        cached = self._available_timeline_cache.get(cache_key)
        if cached is not None:
            return cached

        timeline_rows = sorted(
            [
                (
                    (date - self.experiment_start).total_seconds(),
                    path,
                    idx,
                )
                for idx, path, date in self.iter_available(paths)
            ],
            key=lambda row: (row[0], row[2]),
        )
        seconds = [row[0] for row in timeline_rows]
        available_paths = [row[1] for row in timeline_rows]
        available_indices = [row[2] for row in timeline_rows]
        result = (seconds, available_paths, available_indices)
        self._available_timeline_cache[cache_key] = result
        return result

    @staticmethod
    def _closest_index(
        sorted_values: list[float], source_indices: list[int], target: float
    ) -> int:
        pos = bisect_left(sorted_values, target)
        if pos <= 0:
            return 0
        if pos >= len(sorted_values):
            return len(sorted_values) - 1
        left = pos - 1
        right = pos
        left_distance = abs(sorted_values[left] - target)
        right_distance = abs(sorted_values[right] - target)
        if left_distance < right_distance:
            return left
        if right_distance < left_distance:
            return right
        return left if source_indices[left] <= source_indices[right] else right
