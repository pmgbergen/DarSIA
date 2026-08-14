"""Interface to configure contour smoother methods."""

from abc import ABC
from dataclasses import dataclass

from .utils import _get_key


@dataclass
class ContourSmootherConfig(ABC):
    pass


@dataclass
class SavitzkyGolaySmootherConfig(ContourSmootherConfig):
    """Configuration for Savitzky-Golay contour smoother."""

    window_length: int = 21
    polyorder: int = 3

    def load(self, sec: dict) -> "SavitzkyGolaySmootherConfig":
        self.window_length = int(
            _get_key(sec, "window_length", default=self.window_length, required=False)
        )
        self.polyorder = int(
            _get_key(sec, "polyorder", default=self.polyorder, required=False)
        )
        return self
