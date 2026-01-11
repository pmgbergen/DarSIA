"""Abstract experiment class."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional
import darsia
from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig


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
        imaging_protocol: Path | tuple[Path, str],
        injection_protocol: Optional[Path | tuple[Path, str]] = None,
        pressure_temperature_protocol: Optional[Path | tuple[Path, str]] = None,
        blacklist_protocol: Optional[Path | tuple[Path, str]] = None,
        pad: int = 5,
    ):
        self.data = data
        """Pool of data paths."""
        self.imaging_protocol = darsia.ImagingProtocol(
            imaging_protocol, pad, blacklist_protocol
        )
        """Imaging protocol."""
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

    def find_images_for_times(
        self,
        times: list[float],
        data: list[Path] | None = None,
    ) -> list[Path]:
        """Find image paths for given times since start of the experiment.

        Args:
            times (list[float]): Times since start in hours.
            data (list[Path], optional): Pool of data paths to search in. If None,
                uses the experiment's data pool.

        """
        datetimes = [self.experiment_start + darsia.timedelta(hours=t) for t in times]
        paths = self.imaging_protocol.find_images_for_datetimes(
            paths=data or self.data,
            datetimes=datetimes,
        )
        return paths
