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

    epsilon: float = field(
        default=0.01,
        metadata={
            "name": "Epsilon",
            "help": "Approximation accuracy (pixels or ratio of arc length).",
        },
    )
    closed: bool = field(
        default=True,
        metadata={"name": "Closed", "help": "Whether contour is closed."},
    )
    use_ratio: bool = field(
        default=True,
        metadata={
            "name": "Use ratio",
            "help": "If true, epsilon is ratio of arc length; else absolute pixels.",
        },
    )

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

    window: int = field(
        default=9,
        metadata={
            "name": "Window size",
            "help": "Averaging window size in number of points.",
        },
    )
    closed: bool | None = field(
        default=None,
        metadata={
            "name": "Closed",
            "help": "Closed contour (true/false/unset for auto-detect).",
        },
    )

    def load(self, sec: dict) -> "MovingAverageSmootherConfig":
        self.window = int(_get_key(sec, "window", default=self.window, required=False))
        self.closed = _get_key(
            sec, "closed", default=self.closed, required=False, type_=bool
        )
        return self


@dataclass
class GaussianSmootherConfig(ContourSmootherConfig):
    """Configuration for Gaussian contour smoother."""

    sigma: float = field(
        default=2.0,
        metadata={
            "name": "Sigma",
            "help": "Gaussian standard deviation.",
        },
    )
    window: int | None = field(
        default=None,
        metadata={
            "name": "Window size",
            "help": "Gaussian kernel size (computed from sigma if unset).",
        },
    )
    closed: bool | None = field(
        default=None,
        metadata={
            "name": "Closed",
            "help": "Closed contour (true/false/unset for auto-detect).",
        },
    )

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

    window_length: int = field(
        default=21,
        metadata={
            "name": "Window length",
            "help": "Filter window length (must be odd).",
        },
    )
    polyorder: int = field(
        default=3,
        metadata={
            "name": "Polynomial order",
            "help": "Order of the polynomial fit (must be < window_length).",
        },
    )

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

    active: bool = field(
        default=True,
        metadata={
            "name": "Activate contour smoothing",
            "help": "Enable contour smoothing for extracted contours.",
            "hidden": True,
        },
    )
    """Whether contour smoothing is enabled."""
    type: str = field(
        default="savitzky_golay",
        metadata={
            "name": "Method",
            "help": "Smoothing algorithm to apply.",
            "options": sorted(SUPPORTED_CONTOUR_SMOOTHER_TYPES),
        },
    )
    """Type of contour smoother."""
    poly_dp_options: PolyDPSmootherConfig | None = field(
        default=None,
        metadata={
            "name": "Polynomial DP",
            "toml_key": "contour_smoother_options",
            "depends_on": {"field": "type", "value": "poly_dp"},
        },
    )
    """Options for PolyDP smoother."""
    moving_average_options: MovingAverageSmootherConfig | None = field(
        default=None,
        metadata={
            "name": "Moving Average",
            "toml_key": "contour_smoother_options",
            "depends_on": {"field": "type", "value": "moving_average"},
        },
    )
    """Options for moving average smoother."""
    gaussian_options: GaussianSmootherConfig | None = field(
        default=None,
        metadata={
            "name": "Gaussian",
            "toml_key": "contour_smoother_options",
            "depends_on": {"field": "type", "value": "gaussian"},
        },
    )
    """Options for Gaussian smoother."""
    savitzky_golay_options: SavitzkyGolaySmootherConfig | None = field(
        default=None,
        metadata={
            "name": "Savitzky-Golay",
            "toml_key": "contour_smoother_options",
            "depends_on": {"field": "type", "value": "savitzky_golay"},
        },
    )
    """Options for Savitzky-Golay smoother."""

    def load(self, sec: dict) -> "ContourSmootherSelection":
        """Load contour smoother config from TOML section."""
        # Check if smoothing is enabled
        self.active = bool(sec.get("active", self.active))

        if not self.active:
            return self

        # Load smoother type
        self.type = _get_key(
            sec, "contour_smoother", required=False, default=self.type, type_=str
        ).lower()
        if self.type not in SUPPORTED_CONTOUR_SMOOTHER_TYPES:
            raise ValueError(
                f"Unsupported contour_smoother: '{self.type}'. "
                f"Supported values: {', '.join(sorted(SUPPORTED_CONTOUR_SMOOTHER_TYPES))}."
            )

        # Load type-specific options
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
        if not self.active:
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
