import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

from .fluidflower_config import FluidFlowerConfig
from .roi import MultiRoiConfig
from .utils import _get_key, _get_section_from_toml
from .wasserstein import WassersteinDistancesConfig


@dataclass
class MultiFluidFlowerRunsConfig:
    """Database configuration for multiple FluidFlower runs comparison."""

    config: dict[str, FluidFlowerConfig] = field(default_factory=dict)
    """Dictionary of FluidFlowerConfig objects for each run."""

    def load(
        self,
        path: Path,
        require_data: bool,
        require_results: bool,
    ) -> "MultiFluidFlowerRunsConfig":
        """Load database configuration from TOML file."""

        run_section = _get_section_from_toml(path, "run")
        # self.results = _get_key(data_section, "results", required=True, type_=Path)

        # Allow for common configs, to be added to any other run
        if "common" in run_section:
            common_config = run_section["common"]
            if "config" in common_config:
                if isinstance(common_config["config"], str):
                    common_config_paths = [path.parent / common_config["config"]]
                else:
                    assert isinstance(common_config["config"], list)
                    common_config_paths = [
                        path.parent / p for p in common_config["config"]
                    ]
        else:
            common_config_paths = []

        # Setup config for single runs - combine with common config if provided
        for run_id, run_config in run_section.items():
            # Skip the common config entry
            if run_id in ["common"]:
                continue

            # Make run_id numeric if possible
            if run_id.isdigit():
                run_id = int(run_id)

            # Attach run specific config(s)
            config_paths = [path.parent / run_config["config"]]
            config_paths.extend(common_config_paths)

            # Create FluidFlowerConfig for this run
            self.config[run_id] = FluidFlowerConfig(
                config_paths,
                require_data=require_data,
                require_results=require_results,
            )
            logger.info(f"FluidFlowerConfig finished setup for run {run_id}.")

        return self


@dataclass
class MultiFluidFlowerDataConfig:
    """Data configuration for multiple FluidFlower runs comparison."""

    results: Path = field(default_factory=Path)
    """Path to the results folder for comparison data."""

    def load(self, path: Path | list[Path]) -> "MultiFluidFlowerDataConfig":
        """Load data configuration from TOML file."""
        data_section = _get_section_from_toml(path, "data")
        self.results = _get_key(data_section, "results", required=True, type_=Path)

        # Create results directory if it doesn't exist
        try:
            self.results.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise PermissionError(
                f"Cannot create results directory at {self.results}."
            ) from e

        return self


@dataclass
class EventConfig:
    """Configuration for a single event."""

    event_id: str = ""
    """ID of the event."""
    mode: str = ""
    """Type of the event (e.g., 'mass', 'volume')."""
    roi_name: str = ""
    """Region of interest for this event."""
    relative_threshold: float = 0.01
    """Relative threshold for this event detection, defaults to 1%."""

    def load(self, event_id: str, event_data: dict) -> "EventConfig":
        """Load event configuration from event data dictionary."""
        self.event_id = event_id
        self.mode = _get_key(event_data, "mode", required=True, type_=str)
        self.roi_name = _get_key(event_data, "roi_name", required=True, type_=str)
        self.relative_threshold = _get_key(
            event_data,
            "relative_threshold",
            default=0.01,
            required=False,
            type_=float,
        )
        return self


@dataclass
class EventsConfig:
    """Configuration for events detection and analysis."""

    events: dict[str, EventConfig] = field(default_factory=dict)
    """Dictionary of individual event configurations keyed by event ID."""
    path: Path = field(default_factory=Path)
    """Path for storage."""

    def load(self, path: Path | list[Path], results: Path | None) -> "EventsConfig":
        """Load events configuration from TOML file."""
        # Load the entire TOML data to access events section
        events_section = _get_section_from_toml(path, "events")

        # Load each individual event
        for event_id, event_data in events_section.items():
            event_config = EventConfig()
            event_config.load(event_id, event_data)
            self.events[event_id] = event_config

        # Set path for storage
        if "path" in events_section:
            self.path = Path(events_section["path"])
        elif results is not None:
            self.path = results / "events" / "events.csv"
        else:
            raise ValueError(
                f"Events path not specified and results path is None in {path}."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)

        return self


class MultiFluidFlowerConfig:
    """Meta data for multiple FluidFlower CO2 analysis."""

    def __init__(
        self,
        path: Path,
        require_data: bool = True,
        require_results: bool = False,
    ):
        """Initialize from a comparison config file like runs_comparison.toml.

        Args:
            path (Path): Path to the comparison config file.
            require_data (bool): Whether to require data section in each run config.
            require_results (bool): Whether to require results section in each run config.

        """
        # Make sure that path is compatible
        if isinstance(path, list):
            raise ValueError(
                "Path must be a single Path object for MultiFluidFlowerConfig."
            )
        path = Path(path)

        self.data: MultiFluidFlowerDataConfig | None = None
        self.events: EventsConfig | None = None
        self.roi: MultiRoiConfig | None = None
        self.wasserstein: WassersteinDistancesConfig | None = None

        # Load the comparison config
        try:
            self.runs = MultiFluidFlowerRunsConfig()
            self.runs.load(
                path,
                require_data=require_data,
                require_results=require_results,
            )
        except KeyError:
            self.runs = None
            logger.info(f"Section [database] not found in {path}.")

        # Load data configuration if present
        try:
            self.data = MultiFluidFlowerDataConfig()
            self.data.load(path)
        except KeyError:
            raise ValueError(f"Section [data] not found in {path}.")

        # Events
        try:
            self.events = EventsConfig()
            self.events.load(path, results=self.data.results if self.data else None)
        except KeyError:
            self.events = None
            logger.info(f"Section [events] not found in {path}.")

        # Load ROIs
        try:
            self.roi = MultiRoiConfig()
            self.roi.load(path)
        except KeyError:
            self.roi = None
            logger.info(f"Section [roi] not found in {path}.")

        # Wasserstein distances
        try:
            self.wasserstein = WassersteinDistancesConfig()
            self.wasserstein.load(
                path, results=self.data.results if self.data else None, roi=self.roi
            )
        except Exception as e:
            self.wasserstein = None
            logger.info(f"Section [wasserstein] not found in {path}: {e}")

    def check(self, *sections: str) -> None:
        """Check that all specified sections exist in all run sub_config."""
        for run_id, config in self.runs.config.items():
            try:
                config.check(*sections)
            except ValueError as e:
                raise ValueError(f"Run {run_id}: {e}")
