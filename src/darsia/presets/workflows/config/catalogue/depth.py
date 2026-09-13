"""Catalogue of depth-measurement presets loaded from TOML array-of-tables."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from ..utils import _normalize_mode
from .base import ArrayOfTablesCatalogue, catalogue_path

logger = logging.getLogger(__name__)

_SUPPORTED_MEASUREMENTS_MODES = {"constant", "Load from CSV"}


@dataclass
class DepthPreset:
    """One named depth-measurement preset: a constant depth or a CSV reference.

    Mirrors the [depth] section schema (see config/depth.py's DepthConfig), so a
    preset's :meth:`to_dict` output can be written straight into a run's [depth]
    table.

    Attributes
    ----------
    description
        Human-readable summary shown in the setup wizard.
    measurements
        Where the measurements csv lives (or should be generated), matching
        [depth].measurements -- required there regardless of mode, since a
        'constant' run still needs somewhere to write its generated grid.
        Ignored when `bundled_measurements_file` is set.
    measurements_mode
        'constant' (uniform depth via constant_depth, auto-generatable from the
        wizard/CLI) or 'Load from CSV' (an existing, hand-authored file).
    constant_depth
        Uniform depth in meters, used when measurements_mode == 'constant'.
    bundled_measurements_file
        Filename of a real, hand-authored measurements csv shipped alongside
        this catalogue (resolved to an absolute path at apply time), or None.
        Lets a 'Load from CSV' preset point straight at usable data instead of
        a placeholder the user still has to fill in themselves.
    resolution
        Interpolation resolution for [depth], or None (falls back to the
        [depth] section's own default).
    """

    description: str = ""
    measurements: Path = field(
        default_factory=lambda: Path("./data/depth_measurements.csv")
    )
    measurements_mode: str = "constant"
    constant_depth: float = 0.0
    bundled_measurements_file: str | None = None
    resolution: tuple[int, int] | None = None

    def load(self, entry: dict) -> "DepthPreset":
        """Load one depth preset from its [[depth_preset]] entry dict."""
        self.description = str(entry.get("description", ""))
        self.measurements = Path(
            entry.get("measurements", "./data/depth_measurements.csv")
        )
        self.measurements_mode = _normalize_mode(
            entry.get("measurements_mode", "constant"),
            _SUPPORTED_MEASUREMENTS_MODES,
            key="depth_preset.measurements_mode",
        )
        self.constant_depth = float(entry.get("constant_depth", 0.0))
        self.bundled_measurements_file = entry.get("bundled_measurements_file")
        resolution = entry.get("resolution")
        self.resolution = (
            tuple(int(value) for value in resolution) if resolution else None
        )
        return self

    def to_dict(self) -> dict:
        """Serialize to a TOML-ready [depth] section payload."""
        measurements = (
            catalogue_path(self.bundled_measurements_file)
            if self.bundled_measurements_file
            else self.measurements
        )
        section: dict = {
            "measurements": measurements.as_posix(),
            "measurements_mode": self.measurements_mode,
        }
        if self.measurements_mode == "constant":
            section["constant_depth"] = self.constant_depth
        if self.resolution is not None:
            section["resolution"] = list(self.resolution)
        return section


@dataclass
class DepthCatalogue(ArrayOfTablesCatalogue):
    """A catalogue of named depth-measurement presets."""

    array_key: ClassVar[str] = "depth_preset"

    presets: dict[str, DepthPreset] = field(default_factory=dict)

    def _build_entry(self, entry: dict) -> DepthPreset:
        return DepthPreset().load(entry)

    def get(self, name: str) -> DepthPreset:
        """Retrieve a depth preset by name."""
        return super().get(name)
