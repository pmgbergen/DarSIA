"""Configuration for Wasserstein comparison."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .roi import MultiRoiConfig, RoiConfig
from .time_data import TimeData
from .utils import _get_key, _get_section_from_toml

logger = logging.getLogger(__name__)


@dataclass
class WassersteinDistancesConfig:
    """Configuration for Wasserstein comparison."""

    results: Path = field(default_factory=Path)
    """Path to the results folder for Wasserstein comparison."""
    runs: list[str] = field(default_factory=list)
    """List of run IDs to compare."""
    resize_factor: float | None = None
    """Spatial resize factor for Wasserstein computation."""
    relative_tol: float | None = None
    """Relative tolerance for Wasserstein computation to consider."""
    roi: dict[str, RoiConfig] | None = None
    """ROIs to consider for Wasserstein computation."""
    times: list[tuple[float, float]] = field(default_factory=list)
    """List of times (with uncertainty) at which to compute Wasserstein distances."""

    def load(
        self,
        path: Path | list[Path],
        results: Path | None,
        roi: MultiRoiConfig | None = None,
    ) -> "WassersteinDistancesConfig":
        """Load Wasserstein comparison configuration from TOML file."""

        data_section = _get_section_from_toml(path, "wasserstein")

        # Define results directory
        self.results = _get_key(data_section, "results", required=False, type_=Path)
        if not self.results:
            assert results is not None
            self.results = results / "wasserstein"

        # Create results directory if it doesn't exist
        try:
            self.results.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise PermissionError(
                f"Cannot create results directory at {self.results}."
            ) from e

        # Define runs to compare
        self.runs = _get_key(data_section, "runs", required=True, type_=list)

        # Spatial resize factor
        self.resize_factor = _get_key(
            data_section, "resize", required=False, type_=float
        )

        # Load times.
        try:
            time_data = TimeData()
            time_data.load(data_section["data"], data_folder=None)
            self.times = time_data.get_times_with_uncertainty()
        except KeyError:
            logger.info("No times specified in Wasserstein config.")
            self.times = []

        # Load ROIs
        roi_names = _get_key(data_section, "roi", required=True, type_=list)
        self.roi = {name: roi.roi[name] for name in roi_names}

        return self
