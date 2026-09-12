"""Catalogue of experiment-series presets loaded from TOML array-of-tables.

A series preset is the answer to "which physical rig is this run on?". It does not
restate rig geometry or crop corners — those already live in the rig and curvature
catalogues, so a series just *references* them by name. Only settings that have no
catalogue of their own (type correction, depth measurements) are given inline.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from .base import ArrayOfTablesCatalogue, catalogue_path
from .corrections import CurvatureCatalogue
from .rig import RigCatalogue

logger = logging.getLogger(__name__)

PIECE_LABELS = {
    "rig": "Rig geometry",
    "type": "Type correction",
    "curvature": "Curvature correction (crop)",
    "depth": "Depth measurements",
}
"""Display labels for the pieces a series preset can provide, in apply order."""


@dataclass
class SeriesPreset:
    """One named experiment series: references to rig/curvature, plus extras.

    Attributes
    ----------
    description
        Human-readable summary shown in the setup wizard.
    rig
        Name of an entry in the rig catalogue, or None.
    curvature
        Name of an entry in the curvature catalogue, or None.
    target_type
        Image dtype for [corrections.type] (no catalogue of its own), or None.
    depth_measurements
        Path to a depth-measurements CSV (no catalogue of its own), or None.
    depth_resolution
        Interpolation resolution for [depth], or None.
    """

    description: str = ""
    rig: str | None = None
    curvature: str | None = None
    target_type: str | None = None
    depth_measurements: Path | None = None
    depth_resolution: tuple[int, int] | None = None

    def load(self, entry: dict) -> "SeriesPreset":
        """Load one series preset from its [[series_preset]] entry dict."""
        self.description = str(entry.get("description", ""))
        self.rig = entry.get("rig")
        self.curvature = entry.get("curvature")
        self.target_type = entry.get("target_type")

        depth_section = entry.get("depth") or {}
        measurements = depth_section.get("measurements")
        self.depth_measurements = Path(measurements) if measurements else None
        resolution = depth_section.get("resolution")
        self.depth_resolution = (
            tuple(int(value) for value in resolution) if resolution else None
        )
        return self

    def pieces(self) -> dict[str, tuple[tuple[str, ...], dict]]:
        """Return the pieces this preset provides, ready to write into a config.

        Referenced rig/curvature presets are fetched from their own catalogues
        here, so a series never holds a second copy of those values.

        Returns
        -------
            Mapping of piece name (see :data:`PIECE_LABELS`) to
            ``(config_path, section_dict)``, where ``config_path`` is the nested
            key path in the TOML (e.g. ``("corrections", "curvature")``) and
            ``section_dict`` is the TOML-ready payload. Pieces this preset does
            not define — or whose reference cannot be resolved — are omitted.
        """
        available: dict[str, tuple[tuple[str, ...], dict]] = {}

        if self.rig:
            rig_config = self._resolve(RigCatalogue(), "rig.toml", self.rig, "rig")
            if rig_config is not None:
                available["rig"] = (("rig",), rig_config.to_dict())

        if self.target_type:
            available["type"] = (
                ("corrections", "type"),
                {"target_type": self.target_type},
            )

        if self.curvature:
            curvature_config = self._resolve(
                CurvatureCatalogue(), "corrections.toml", self.curvature, "curvature"
            )
            if curvature_config is not None:
                section = curvature_config.to_dict()
                stages = [
                    stage
                    for stage in ("init", "crop", "bulge", "stretch")
                    if stage in section
                ]
                if stages:
                    section["active"] = stages
                available["curvature"] = (("corrections", "curvature"), section)

        if self.depth_measurements is not None:
            depth_section: dict = {"measurements": self.depth_measurements.as_posix()}
            if self.depth_resolution is not None:
                depth_section["resolution"] = list(self.depth_resolution)
            available["depth"] = (("depth",), depth_section)

        return available

    @staticmethod
    def _resolve(catalogue, filename: str, name: str, kind: str):
        """Fetch a referenced preset, warning (not raising) on a dangling name."""
        try:
            return catalogue.load(catalogue_path(filename)).get(name)
        except KeyError:
            logger.warning(
                "Series references unknown %s preset '%s'; skipping that piece.",
                kind,
                name,
            )
            return None


@dataclass
class SeriesCatalogue(ArrayOfTablesCatalogue):
    """A catalogue of named experiment-series presets."""

    array_key: ClassVar[str] = "series_preset"

    presets: dict[str, SeriesPreset] = field(default_factory=dict)

    def _build_entry(self, entry: dict) -> SeriesPreset:
        return SeriesPreset().load(entry)

    def get(self, name: str) -> SeriesPreset:
        """Retrieve a series preset by name."""
        return super().get(name)
