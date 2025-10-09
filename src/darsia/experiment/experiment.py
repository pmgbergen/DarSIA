"""Abstract experiment class."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional
import darsia


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
        imaging_protocol: Path | tuple[Path, str],
        injection_protocol: Optional[Path | tuple[Path, str]] = None,
        pressure_temperature_protocol: Optional[Path | tuple[Path, str]] = None,
        blacklist_protocol: Optional[Path | tuple[Path, str]] = None,
        pad: int = 5,
    ):
        self.imaging_protocol = darsia.ImagingProtocol(
            imaging_protocol, pad, blacklist_protocol
        )
        self.injection_protocol = darsia.InjectionProtocol(injection_protocol)
        self.pressure_temperature_protocol = darsia.PressureTemperatureProtocol(
            pressure_temperature_protocol
        )

        # Reference date
        self.experiment_start = self.injection_protocol.df["start"].min()

    def time_since_start(self, date: datetime) -> float:
        """Hours since start of the experiment.

        Args:
            date (datetime): Date to compute the time since start for.

        Returns:
            float: Time since start in hours.

        """
        return (date - self.experiment_start).total_seconds() / 3600
