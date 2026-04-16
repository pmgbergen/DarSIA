"""Data configuration for the setup."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from .data_registry import DataRegistry
from .time_data import TimeData
from .utils import _get_key, _get_section_from_toml

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data for the setup.

    Example for TOML section:
        [data]
        folder = "path/to/images"
        format = "JPG"
        baseline = "path/to/baseline.jpg"
        pad = 0
        results = "path/to/results"

    """

    folder: Path = field(default_factory=Path)
    """Path to the folder containing the image data."""
    format: str = "JPG"
    """Format of the image data (e.g., 'JPG', 'PNG')."""
    data: list[Path] = field(default_factory=list)
    """List of paths to the image data."""
    baseline: Path = field(default_factory=Path)
    """Path to the baseline image."""
    pad: int = 0
    """Pad for image names."""
    results: Path = field(default_factory=Path)
    """Path to the results folder."""
    cache: Path | None = None
    """Path to the cache folder, or None if caching is disabled."""
    raw_cache: Path | None = None
    """Path to the raw cache folder, or None if caching is disabled."""
    use_cache: bool = False
    """Whether to use the cache folder for reading/writing cached images."""
    time_data: TimeData | None = None
    """Calibration data configuration."""
    registry: DataRegistry | None = None
    """Optional global data registry loaded from [data.interval.*], [data.time.*],
    and [data.path.*] sub-sections."""

    def load(
        self,
        path: Path | list[Path],
        require_data: bool,
        require_results: bool,
    ) -> "DataConfig":
        sec = _get_section_from_toml(path, "data")

        # Get folder
        self.folder = _get_key(sec, "folder", required=True, type_=Path)
        if require_data and not self.folder.is_dir():
            raise FileNotFoundError(f"Folder {self.folder} not found.")

        # Get baseline
        self.baseline = self.folder / _get_key(
            sec, "baseline", required=True, type_=Path
        )
        if require_data and not self.baseline.is_file():
            raise FileNotFoundError(f"Baseline image {self.baseline} not found.")

        # Get format
        numeric_part = "".join(filter(str.isdigit, self.baseline.stem))
        self.pad = len(numeric_part) if numeric_part else 0

        # Get data
        if require_data:
            all_data: list[Path] = []
            for folder in self.folders:
                all_data.extend(
                    sorted(
                        folder / file
                        for file in os.listdir(folder)
                        if file.endswith(self.baseline.suffix)
                    )
                )
            self.data = sorted(set(all_data))
            if len(self.data) == 0:
                raise FileNotFoundError(
                    f"""No image files with suffix {self.baseline.suffix} found in """
                    f"""{self.folder}."""
                )
        else:
            self.data = None

        # Get results
        self.results = _get_key(sec, "results", required=True, type_=Path)
        if require_results:
            if not self.results.is_dir():
                raise FileNotFoundError(f"Results folder {self.results} not found.")
        else:
            try:
                self.results.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise PermissionError(
                    f"Cannot create results directory at {self.results}."
                ) from e

        # Whether to use the cache folder for reading/writing cached images
        self.use_cache = _get_key(
            sec, "use_cache", default=False, required=False, type_=bool
        )

        # Define cache folder and only create it when caching is enabled
        if self.use_cache:
            self.cache = self.results / "cache"
            self.raw_cache = self.results / "raw_cache"
            try:
                self.cache.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise PermissionError(
                    f"Cannot create cache directory at {self.cache}."
                ) from e
            try:
                self.raw_cache.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise PermissionError(
                    f"Cannot create raw cache directory at {self.raw_cache}."
                ) from e
        else:
            self.cache = None

        # Attempt to load global DataRegistry from [data.interval.*], [data.time.*],
        # and [data.path.*] sub-sections. This is optional; if none are present the
        # registry is set to None.
        has_registry_sections = any(key in sec for key in ("interval", "time", "path"))
        if has_registry_sections:
            try:
                self.registry = DataRegistry().load(sec, self.folder)
            except Exception as e:
                logger.warning(f"Failed to load DataRegistry: {e}")
                self.registry = None
        else:
            self.registry = None

        return self

    def error(self):
        raise ValueError("Use [data] in the config file to load data.")
