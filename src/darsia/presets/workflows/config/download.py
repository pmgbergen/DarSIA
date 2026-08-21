"""Configuration for data download within analysis workflows."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from warnings import warn

from .utils import _get_key, _get_section_from_toml

logger = logging.getLogger(__name__)


@dataclass
class DownloadConfig:
    source: Path | None = field(
        default=None,
        metadata={
            "name": "Source folder",
            "help": "Directory containing data to download. If not set, uses [data] folder.",
            "widget": "folder",
        },
    )
    """Source folder - if `None`, retrieved from arguments."""
    data_selection: str | list[str] | None = field(
        default=None,
        metadata={
            "name": "Data selection",
            "help": "Registry key name(s) whose data is unioned for download.",
            "widget": "registry_key_list",
        },
    )
    """Name(s) of data registry entries to use for download."""
    skip_existing: bool = field(
        default=True,
        metadata={
            "name": "Skip existing files",
            "help": "Skip downloading files that already exist.",
        },
    )
    """Flag for skipping existing data."""
    folder: Path | None = field(
        default=None,
        metadata={
            "name": "Download folder",
            "help": (
                "Destination folder for downloaded data. Defaults to [data.results/raw_data] "
                "if not set."
            ),
            "widget": "folder",
        },
    )
    """Path to the folder where downloaded data will be stored.
    If not provided, defaults to [data.results/raw_data]."""

    def load(
        self,
        path: Path,
        data: Path | None,
        results: Path | None,
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
        self.data_selection = _get_key(
            sec, "data_selection", required=False, default=None
        )
        if self.data_selection is None:
            self.data_selection = _get_key(sec, "data", required=False, default=None)

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
