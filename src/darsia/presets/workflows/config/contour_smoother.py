"""Interface to configure contour smoother methods."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import darsia

from .utils import _get_key

if TYPE_CHECKING:
    pass

SUPPORTED_CONTOUR_SMOOTHER_TYPES = {
    "none",
    "poly_dp",
    "moving_average",
    "gaussian",
    "savitzky_golay",
}


@dataclass
class ContourSmootherConfig(ABC):
    pass


@dataclass
class PolyDPSmootherConfig(ContourSmootherConfig):
    """Configuration for PolyDP contour smoother."""

    epsilon: float = 0.01
    closed: bool = True
    use_ratio: bool = True

    def load(self, sec: dict) -> "PolyDPSmootherConfig":
        self.epsilon = float(
            _get_key(sec, "epsilon", default=self.epsilon, required=False)
        )
        self.closed = _get_key(
            sec, "closed", default=self.closed, required=False, type_=bool
        )
        self.use_ratio = _get_key(
            sec, "use_ratio", default=self.use_ratio, required=False, type_=bool
        )
        return self


@dataclass
class MovingAverageSmootherConfig(ContourSmootherConfig):
    """Configuration for moving average contour smoother."""

    window: int = 9
    closed: bool | None = None

    def load(self, sec: dict) -> "MovingAverageSmootherConfig":
        self.window = int(_get_key(sec, "window", default=self.window, required=False))
        self.closed = _get_key(
            sec, "closed", default=self.closed, required=False, type_=bool
        )
        return self


@dataclass
class GaussianSmootherConfig(ContourSmootherConfig):
    """Configuration for Gaussian contour smoother."""

    sigma: float = 2.0
    window: int | None = None
    closed: bool | None = None

    def load(self, sec: dict) -> "GaussianSmootherConfig":
        self.sigma = float(_get_key(sec, "sigma", default=self.sigma, required=False))
        self.window = _get_key(
            sec, "window", default=self.window, required=False, type_=int
        )
        self.closed = _get_key(
            sec, "closed", default=self.closed, required=False, type_=bool
        )
        return self


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


@dataclass
class ContourSmootherSelection:
    """Unified config for selecting and configuring a contour smoother."""

    type: str = field(
        default="none",
        metadata={
            "name": "Contour smoother",
            "help": "Contour smoothing algorithm.",
            "options": sorted(SUPPORTED_CONTOUR_SMOOTHER_TYPES),
        },
    )
    """Type of contour smoother."""
    poly_dp_options: PolyDPSmootherConfig | None = field(
        default=None,
        metadata={
            "name": "Options",
            "toml_key": "contour_smoother_options",
            "depends_on": {"field": "type", "value": "poly_dp"},
        },
    )
    """Options for PolyDP smoother."""
    moving_average_options: MovingAverageSmootherConfig | None = field(
        default=None,
        metadata={
            "name": "Options",
            "toml_key": "contour_smoother_options",
            "depends_on": {"field": "type", "value": "moving_average"},
        },
    )
    """Options for moving average smoother."""
    gaussian_options: GaussianSmootherConfig | None = field(
        default=None,
        metadata={
            "name": "Options",
            "toml_key": "contour_smoother_options",
            "depends_on": {"field": "type", "value": "gaussian"},
        },
    )
    """Options for Gaussian smoother."""
    savitzky_golay_options: SavitzkyGolaySmootherConfig | None = field(
        default=None,
        metadata={
            "name": "Options",
            "toml_key": "contour_smoother_options",
            "depends_on": {"field": "type", "value": "savitzky_golay"},
        },
    )
    """Options for Savitzky-Golay smoother."""

    def load(self, sec: dict) -> "ContourSmootherSelection":
        """Load contour smoother config from TOML section."""
        self.type = _get_key(
            sec, "contour_smoother", required=False, default="none", type_=str
        ).lower()
        if self.type not in SUPPORTED_CONTOUR_SMOOTHER_TYPES:
            raise ValueError(
                f"Unsupported contour_smoother: '{self.type}'. "
                f"Supported values: {', '.join(sorted(SUPPORTED_CONTOUR_SMOOTHER_TYPES))}."
            )

        options_sec = sec.get("contour_smoother_options", {})
        if self.type == "poly_dp":
            self.poly_dp_options = PolyDPSmootherConfig().load(options_sec)
        elif self.type == "moving_average":
            self.moving_average_options = MovingAverageSmootherConfig().load(
                options_sec
            )
        elif self.type == "gaussian":
            self.gaussian_options = GaussianSmootherConfig().load(options_sec)
        elif self.type == "savitzky_golay":
            self.savitzky_golay_options = SavitzkyGolaySmootherConfig().load(
                options_sec
            )

        return self

    def build(self) -> darsia.ContourSmoother | None:
        """Construct the configured ContourSmoother instance."""
        if self.type == "none":
            return None
        elif self.type == "poly_dp":
            o = self.poly_dp_options
            return darsia.PolyDPSmoother(
                epsilon=o.epsilon, closed=o.closed, use_ratio=o.use_ratio
            )
        elif self.type == "moving_average":
            o = self.moving_average_options
            return darsia.MovingAverageSmoother(window=o.window, closed=o.closed)
        elif self.type == "gaussian":
            o = self.gaussian_options
            return darsia.GaussianSmoother(
                sigma=o.sigma, window=o.window, closed=o.closed
            )
        elif self.type == "savitzky_golay":
            o = self.savitzky_golay_options
            return darsia.SavitzkyGolaySmoother(
                window_length=o.window_length, polyorder=o.polyorder
            )
        return None
