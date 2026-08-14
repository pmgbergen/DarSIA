"""Configuration for data download within analysis workflows."""

import logging
from dataclasses import dataclass
from pathlib import Path
from warnings import warn

from .data_registry import DataRegistry
from .time_data import TimeData
from .utils import _get_section_from_toml

logger = logging.getLogger(__name__)


@dataclass
class DownloadConfig:
    source: Path | None = None
    """Source folder - if `None`, retrieved from arguments."""
    data: TimeData | None = None
    """Download data selection configuration."""
    skip_existing: bool = True
    """Flag for skipping existing data."""
    folder: Path | None = None
    """Path to the folder where downloaded data will be stored.
    If not provided, defaults to [data.results/raw_data]."""

    def load(
        self,
        path: Path,
        data: Path | None,
        results: Path | None,
        data_registry: DataRegistry | None,
    ) -> "DownloadConfig":
        sec = _get_section_from_toml(path, "download")

        # Config to source folder
        raw_source = sec.get("source", data)
        if raw_source is None:
            raise ValueError(
                """No source folder specified. Provide a valid source folder """
                """in [download.source] when no [data] section (and thus no """
                """data path) is available."""
            )
        self.source = Path(raw_source)
        if not self.source.exists():
            raise ValueError(
                f"""Source folder {self.source} does not exist. """
                """Provide a valid source folder in [download.source] or """
                """ensure that [data] is correctly specified."""
            )

        # Config to load download data
        try:
            self.data = (
                data_registry.resolve(sec.get("data")) if data_registry else None
            )
        except KeyError:
            warn("No download data found. Use [download.data].")
            self.data = None

        # Config to skip existing files
        self.skip_existing = sec.get("skip_existing", True)

        # Config to load download folder
        try:
            self.folder = Path(sec["folder"])
        except KeyError:
            warn(
                """No download folder found. Use [download.folder]. """
                """Defaulting to [data.results/raw_data]."""
            )
            if results is not None:
                self.folder = results / "raw_data"
            else:
                raise ValueError(
                    "No download folder configured and no results path provided. "
                    "Specify [download.folder] in the configuration or provide a "
                    "valid results path to determine a default download folder."
                )

        return self
