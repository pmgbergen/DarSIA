"""Abstract experiment class."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

import darsia
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig


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
        imaging_protocol: (
            Path | tuple[Path, str] | dict[Path, Path | tuple[Path, str]]
        ),
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
        else:
            self.imaging_protocol = darsia.ImagingProtocol(
                imaging_protocol, pad, blacklist_protocol
            )
            self.imaging_protocols = None
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

    @classmethod
    def init_from_config(cls, config: FluidFlowerConfig):
        assert config.data is not None
        assert config.protocol is not None
        if (
            len(config.data.folders) > 1
            and not isinstance(config.protocol.imaging, dict)
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
            protocol = self._protocol_for_path(path)
            if not protocol.is_blacklisted(path):
                available_paths.append(path)
        return available_paths

    def find_images_for_times(
        self,
        times: float | list[float],
        tol: float | None = None,
        data: list[Path] | None = None,
    ) -> list[Path]:
        """Find image paths for given times since start of the experiment.

        Args:
            times (list[float]): Times since start in hours.
            data (list[Path], optional): Pool of data paths to search in. If None,
                uses the experiment's data pool.

        """
        times_is_list = isinstance(times, list)
        if not times_is_list:
            times = [times]
        datetimes = [self.experiment_start + darsia.timedelta(hours=t) for t in times]
        source_paths = data or self.data

        available_paths: list[Path] = []
        available_datetimes: list[datetime] = []
        for path in source_paths:
            protocol = self._protocol_for_path(path)
            if protocol.is_blacklisted(path):
                continue
            try:
                date = protocol.get_datetime(path)
            except ValueError:
                continue
            if date is None:
                continue
            available_paths.append(path)
            available_datetimes.append(date)

        if len(available_paths) == 0:
            raise ValueError("No available images found in the specified paths.")

        selected_paths: list[Path] = []
        for dt in datetimes:
            min_index = min(
                range(len(available_datetimes)),
                key=lambda i: abs((available_datetimes[i] - dt).total_seconds()),
            )
            min_distance = abs((available_datetimes[min_index] - dt).total_seconds())
            if tol is None or min_distance < tol:
                selected_paths.append(available_paths[min_index])
        paths = list(dict.fromkeys(selected_paths))
        if times_is_list:
            return paths
        else:
            return paths[0] if paths else None

    def _protocol_for_path(self, path: Path) -> darsia.ImagingProtocol:
        if self.imaging_protocol is not None:
            return self.imaging_protocol
        assert self.imaging_protocols is not None
        resolved_path = path.resolve()
        best_match: Path | None = None
        for folder in self.imaging_protocols.keys():
            resolved_folder = folder.resolve()
            try:
                resolved_path.relative_to(resolved_folder)
                if best_match is None or len(resolved_folder.parts) > len(
                    best_match.resolve().parts
                ):
                    best_match = folder
            except ValueError:
                continue
        if best_match is None:
            raise ValueError(f"No imaging protocol configured for image path: {path}")
        return self.imaging_protocols[best_match]
