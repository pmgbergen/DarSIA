"""Catalogue of experiment-series presets loaded from TOML array-of-tables.

A series preset bundles together the rig geometry, type/resize/curvature
corrections, and depth-measurements path that are all tied to one physical
FluidFlower rig setup, so the setup wizard can offer them as one named choice
(e.g. "B049", "BC0xx family") instead of requiring each piece to be configured
separately.
"""

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..corrections import (
    CurvatureCorrectionConfig,
    ResizeCorrectionConfig,
    TypeCorrectionConfig,
)
from ..rig import RigConfig

logger = logging.getLogger(__name__)

PIECE_LABELS = {
    "rig": "Rig geometry",
    "type": "Type correction",
    "resize": "Resize correction",
    "curvature": "Curvature correction (crop)",
    "depth": "Depth measurements",
}
"""Display labels for the pieces a series preset can provide, in apply order."""


@dataclass
class SeriesPreset:
    """A named bundle of rig/corrections/depth settings for one experiment series.

    Entries are loaded from a top-level [[series_preset]] TOML array-of-tables,
    each with a required 'name' field, an optional 'description', and optional
    nested [series_preset.rig/type/resize/curvature/depth] sub-sections using the
    same field names as the corresponding real config sections (so presets can be
    copy-pasted directly from known-good configs). Any sub-section may be omitted;
    a preset need not provide every piece.
    """

    description: str = ""
    rig: RigConfig | None = None
    type: TypeCorrectionConfig | None = None
    resize: ResizeCorrectionConfig | None = None
    curvature: CurvatureCorrectionConfig | None = None
    depth_measurements: Path | None = None
    depth_target_resolution: tuple[int, int] | None = None

    def load(self, entry: dict) -> "SeriesPreset":
        """Load one series preset from its [[series_preset]] entry dict.

        Parameters
        ----------
        entry : dict
            One [[series_preset]] table entry (already parsed from TOML).

        Returns
        -------
            Self.
        """
        self.description = str(entry.get("description", ""))

        rig_sec = entry.get("rig")
        if rig_sec:
            spec = RigConfig()
            spec.width = float(rig_sec.get("width", 0))
            spec.height = float(rig_sec.get("height", 0))
            spec.dim = int(rig_sec.get("dim", 2))
            self.rig = spec
        else:
            self.rig = None

        type_sec = entry.get("type")
        self.type = TypeCorrectionConfig().load(type_sec) if type_sec else None

        resize_sec = entry.get("resize")
        self.resize = ResizeCorrectionConfig().load(resize_sec) if resize_sec else None

        curvature_sec = entry.get("curvature")
        self.curvature = (
            CurvatureCorrectionConfig().load(curvature_sec) if curvature_sec else None
        )

        depth_sec = entry.get("depth")
        if depth_sec:
            measurements = depth_sec.get("measurements")
            self.depth_measurements = Path(measurements) if measurements else None
            resolution = depth_sec.get("resolution")
            self.depth_target_resolution = (
                tuple(int(v) for v in resolution) if resolution else None
            )
        else:
            self.depth_measurements = None
            self.depth_target_resolution = None

        return self

    def pieces(self) -> dict[str, tuple[tuple[str, ...], dict]]:
        """Return the pieces this preset provides, ready to write into a config.

        Returns
        -------
            Mapping of piece name (see :data:`PIECE_LABELS`) to
            ``(config_path, section_dict)``, where ``config_path`` is the nested
            key path in the TOML (e.g. ``("corrections", "curvature")``) and
            ``section_dict`` is the TOML-ready payload for that path. Pieces the
            preset does not define are omitted.
        """
        available: dict[str, tuple[tuple[str, ...], dict]] = {}

        if self.rig is not None:
            available["rig"] = (("rig",), self.rig.to_dict())

        if self.type is not None:
            available["type"] = (
                ("corrections", "type"),
                {"target_type": np.dtype(self.type.target_type).name},
            )

        if self.resize is not None:
            resize_section: dict = {"mode": self.resize.mode}
            if self.resize.scale is not None:
                resize_section["scale"] = self.resize.scale
            if self.resize.target_shape is not None:
                resize_section["target_shape"] = list(self.resize.target_shape)
            available["resize"] = (("corrections", "resize"), resize_section)

        if self.curvature is not None:
            curvature_section = self.curvature.to_dict()
            stages = [
                stage
                for stage in ("init", "crop", "bulge", "stretch")
                if stage in curvature_section
            ]
            if stages:
                curvature_section["active"] = stages
            available["curvature"] = (("corrections", "curvature"), curvature_section)

        if self.depth_measurements is not None:
            depth_section: dict = {"measurements": self.depth_measurements.as_posix()}
            if self.depth_target_resolution is not None:
                depth_section["resolution"] = list(self.depth_target_resolution)
            available["depth"] = (("depth",), depth_section)

        return available


@dataclass
class SeriesCatalogue:
    """A catalogue of named experiment-series presets."""

    presets: dict[str, SeriesPreset] = field(
        default_factory=dict,
        metadata={
            "name": "Series presets",
            "help": "Named experiment-series presets",
        },
    )
    """Dict of named series presets."""

    def load(self, path: Path | list[Path]) -> "SeriesCatalogue":
        """Load all series preset entries from the top-level [[series_preset]]
        array-of-tables in TOML.

        Hand-parses TOML (like CurvatureCatalogue/RigCatalogue) since
        array-of-tables is not supported by the generic _get_section_from_toml
        helper.

        Parameters
        ----------
        path : Path | list[Path]
            Path or list of Paths to TOML catalogue file(s).

        Returns
        -------
            Self.

        Raises
        ------
        ValueError
            If [series_preset] section is not an array-of-tables, or if any
            preset entry has a missing or duplicate name.
        """
        if isinstance(path, list):
            paths = [Path(p) for p in path]
        else:
            paths = [Path(path)]
        self.presets = {}
        seen_names: set[str] = set()

        for p in paths:
            if not p.exists():
                continue
            with open(p, "rb") as f:
                data = tomllib.load(f)

            if "series_preset" not in data:
                continue

            preset_data = data["series_preset"]
            if not isinstance(preset_data, list):
                raise ValueError(
                    "The [series_preset] section must be an array-of-tables format "
                    "(use [[series_preset]]), not nested tables."
                )

            for idx, entry in enumerate(preset_data):
                name = entry.get("name")
                if name is None:
                    raise ValueError(
                        f"[[series_preset]] entry {idx} must have a required 'name' field."
                    )
                name = str(name).strip()

                if name in seen_names:
                    raise ValueError(
                        f"Preset name '{name}' is duplicated. Preset names must be "
                        "globally unique."
                    )
                seen_names.add(name)

                self.presets[name] = SeriesPreset().load(entry)

        return self

    def names(self) -> list[str]:
        """Return all registered preset names, sorted alphabetically."""
        return sorted(self.presets.keys())

    def get(self, name: str) -> SeriesPreset:
        """Retrieve a preset by name.

        Parameters
        ----------
        name : str
            The preset name.

        Returns
        -------
            The SeriesPreset.

        Raises
        ------
        KeyError
            If the preset name is not found.
        """
        if name not in self.presets:
            raise KeyError(
                f"Preset '{name}' not found in catalogue. "
                f"Available presets: {self.names()}"
            )
        return self.presets[name]
