"""Abstract experiment class."""

from abc import ABC, abstractmethod
from datetime import datetime


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
