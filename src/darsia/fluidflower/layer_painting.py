"""Layer painting: polygons and brush strokes on a raw photograph, a proposed
layer map and its conversion to a geometry revision for native Setup.

The simple view of the Geometry and facies tab lets a user outline layers with
polygons and paint brush strokes on the raw baseline photograph of the
reference run. A pixel classifier (DarSIA's facies features and extremely
randomized trees) is trained on the marked pixels and labels the rest of the
photograph inside the rig frame:

* Inside a polygon, farther than the snap distance from every polygon edge,
  the polygon's layer is kept. Where polygons of different layers overlap, the
  smaller polygon wins, so a lens drawn inside a layer keeps its own layer.
* Near polygon edges the classifier places the boundary, choosing only between
  the layers whose polygons or strokes lie within the snap distance. Near areas
  that no polygon covers, every layer competes.
* Away from polygons every layer competes, as with strokes alone.

Class probabilities are smoothed, painted pixels keep their layer and regions
smaller than a set area join their largest neighbour. Layers are grouped into
facies that share a role, porosity and permeability. Saving converts the
proposal to a label map on the raw photograph's grid, stores it as a geometry
revision and exports it for native Setup.

Polygons and strokes are stored in raw-image pixels (row, col), strokes with a
brush radius in raw pixels, in ``<package>/annotations/layer_painting/<run>.json``.
The latest proposal is stored next to it.

Hand corrections are drawn on top of a proposal: brush strokes and lasso
outlines that give pixels a layer, region fills, and restore strokes that bring
the proposal back. They are stored in their own file next to the session and
never change the polygons, the strokes or the proposal; the saved map is the
proposal with the corrections applied in drawing order.

Versions of the map are saved on request: the session, the proposal and the
corrections in a new folder under ``versions/<run>`` that is never changed
afterwards. A version can be restored later; its SHA-256 identifies the map.
"""

from __future__ import annotations

from darsia.fluidflower import preset_wasserstein_inputs as _preset_io

import hashlib
import json
import math
import os
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from darsia.fluidflower.project import FluidFlowerProject, utc_now
from darsia.fluidflower.series_geometry import (
    NATIVE_MAX_REGIONS,
    RigFrame,
    SeriesRunInputs,
    _resolved_points,
    native_series_link,
    photo_size,
    saved_injection_points,
)
from darsia.fluidflower.storage_paths import machine_local_text, machine_stored_text

LAYER_PAINTING_FORMAT = "darsia-fluidflower-layer-painting-v1"
LAYER_PROPOSAL_FORMAT = "darsia-fluidflower-layer-proposal-v1"
PROPOSAL_METHOD = "painted_strokes_extra_trees_v1"
# Layer ID of eraser strokes, and class of pixels outside the rig frame.
ERASER = 0
OUTSIDE = 0
ANALYSIS_WIDTH = 1200
DISPLAY_LIMIT_PX = 2400
# Up to this many pixels on its longest side, the painting tab also keeps the
# photograph at full resolution and shows it when zoomed in (display only).
# 8400 covers the series cameras, the Nikon Z 7II RAW (8288 x 5520) included.
DETAIL_LIMIT_PX = 8400
LAYER_ROLES: dict[str, str] = {
    "porous_medium": "Porous medium",
    "free_water": "Free water",
    "excluded": "Excluded",
    "colorchecker": "ColorChecker",
}
# Porosity of the roles without a porous medium: free water fills the whole
# volume, excluded regions and the area outside the rig frame hold none. Native
# DarSIA integrates mass with depth x porosity, and a facies without porosity
# would make every integral over the rig NaN.
ROLE_POROSITY: dict[str, float] = {"free_water": 1.0, "excluded": 0.0, "colorchecker": 0.0}
OUTSIDE_POROSITY = 0.0
SMOOTHING_SIGMA: dict[str, float] = {"low": 1.5, "medium": 3.0, "high": 6.0}
PERMEABILITY_UNITS = ("darcy", "mD", "m2")
LAYER_COLORS = (
    "#3d8bd9",
    "#f2c14e",
    "#e05d5d",
    "#9aa0a6",
    "#4fb38a",
    "#a77bd6",
    "#e8893a",
    "#5cc6d0",
    "#d36aa8",
    "#8fa04a",
    "#c9a27e",
    "#6f86c9",
)
DEFAULT_LAYERS: tuple[tuple[str, str], ...] = (
    ("Free water", "free_water"),
    ("Fine sand", "porous_medium"),
    ("Coarse sand", "porous_medium"),
    ("Excluded", "excluded"),
)
OUTSIDE_FACIES = "Outside the rig frame"
# Constant depth written into the revision, which requires a depth map. Native
# Setup takes the depth from its own [depth] section.
PLACEHOLDER_DEPTH_M = 0.01
# Pixels whose largest smoothed class probability is below this are uncertain.
UNCERTAIN_BELOW = 0.6
# How far polygon edges may lie from the true boundary, by default.
DEFAULT_SNAP_MM = 10.0
MAX_SNAP_MM = 100.0
# Uncovered strips this close to the rig-frame edge take the nearest polygon's
# layer (polygons drawn to the edge seldom reach it exactly).
DEFAULT_EDGE_FILL_MM = 5.0
# The grid a proposal is computed on: "normal" (ANALYSIS_WIDTH columns) or
# "fine" (the photograph as displayed), which keeps thin layers.
PROPOSAL_DETAILS = ("normal", "fine")
# Without a rig size, the photograph is taken to be this wide.
NOMINAL_IMAGE_WIDTH_M = 1.0
# Polygon points this close (raw pixels) are one shared point.
SHARED_POINT_PX = 0.5

ProgressCallback = Callable[[str], None]


def _progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _number(value: Any) -> float | None:
    from .mass_inputs import _number as shared_number
    return shared_number(value)


# ---------------------------------------------------------------------------
# Session: layers, facies and strokes


@dataclass
class PaintedLayer:
    """A layer the user paints; ``facies`` names the facies it belongs to.

    ``snap_mm`` is this layer's own snap distance for its polygons; None uses
    the session's.
    """

    layer_id: int
    name: str
    color: str
    facies: str
    snap_mm: float | None = None
    always_on_top: bool = False  # Manual-map/display priority; classifier rules are unchanged.


@dataclass
class FaciesEntry:
    """Shared role and properties of the layers in one facies."""

    role: str = "porous_medium"
    porosity: float | None = None
    permeability: float | None = None

    def effective_porosity(self) -> float | None:
        """The porosity exported: entered for a porous medium, else the role's."""

        if self.role == "porous_medium":
            return self.porosity
        return ROLE_POROSITY.get(self.role)


@dataclass
class Stroke:
    """A brush stroke in raw-image pixels; layer 0 erases."""

    layer_id: int
    radius_px: float
    points: list[tuple[float, float]]

    def to_list(self) -> list[Any]:
        return [
            int(self.layer_id),
            round(float(self.radius_px), 2),
            [[round(float(row), 1), round(float(col), 1)] for row, col in self.points],
        ]

    @classmethod
    def from_list(cls, payload: Sequence[Any]) -> Stroke:
        layer_id, radius, points = payload
        return cls(
            int(layer_id),
            float(radius),
            [(float(row), float(col)) for row, col in points],
        )


@dataclass
class Polygon:
    """A closed polygon of one layer; vertices in raw-image pixels (row, col)."""

    layer_id: int
    points: list[tuple[float, float]]

    def area(self) -> float:
        """Enclosed area in raw pixels² (shoelace formula)."""

        if len(self.points) < 3:
            return 0.0
        rows = np.asarray([point[0] for point in self.points], dtype=np.float64)
        cols = np.asarray([point[1] for point in self.points], dtype=np.float64)
        return float(
            0.5 * abs(np.dot(cols, np.roll(rows, -1)) - np.dot(rows, np.roll(cols, -1)))
        )

    def to_list(self) -> list[Any]:
        return [
            int(self.layer_id),
            [[round(float(row), 1), round(float(col), 1)] for row, col in self.points],
        ]

    @classmethod
    def from_list(cls, payload: Sequence[Any]) -> Polygon:
        layer_id, points = payload
        return cls(int(layer_id), [(float(row), float(col)) for row, col in points])


_UNSET: Any = object()


@dataclass
class PaintingSession:
    """Layers, facies, polygons and strokes of one reference run.

    Every layer belongs to one facies. Facies keep their order and may be
    empty (an empty facies is not exported).
    """

    run_id: str
    image_path: str = ""
    image_size: tuple[int, int] | None = None
    # File size and modification time of the photograph the strokes belong to.
    image_stamp: str = ""
    layers: list[PaintedLayer] = field(default_factory=list)
    facies: dict[str, FaciesEntry] = field(default_factory=dict)
    polygons: list[Polygon] = field(default_factory=list)
    strokes: list[Stroke] = field(default_factory=list)
    permeability_unit: str = "darcy"
    smoothing: str = "medium"
    min_region_cm2: float = 2.0
    snap_mm: float = DEFAULT_SNAP_MM
    edge_fill_mm: float = DEFAULT_EDGE_FILL_MM
    detail: str = "normal"
    # Whether the classifier may move edges that polygons of different layers
    # share, within the snap distance like other edges. False keeps them
    # exactly where they are drawn.
    move_shared_edges: bool = True
    updated_utc: str = ""
    updated_by: str = ""
    # Set when the run's baseline photograph differs from the one painted on.
    photo_changed: bool = field(default=False, compare=False)

    @classmethod
    def default(
        cls,
        run_id: str,
        image_path: str | Path = "",
        image_size: tuple[int, int] | None = None,
    ) -> PaintingSession:
        session = cls(run_id=str(run_id), image_path=str(image_path or ""))
        session.image_size = tuple(image_size) if image_size else None
        for index, (name, role) in enumerate(DEFAULT_LAYERS):
            session.layers.append(
                PaintedLayer(index + 1, name, LAYER_COLORS[index], name)
            )
            session.facies[name] = FaciesEntry(role=role)
        return session

    # ------------------------------------------------------------ layers

    def layer(self, layer_id: int) -> PaintedLayer | None:
        return next((row for row in self.layers if row.layer_id == layer_id), None)

    def layer_snap_mm(self, layer_id: int) -> float:
        """The snap distance of one layer's polygons: its own, else the session's."""

        layer = self.layer(layer_id)
        if layer is None or layer.snap_mm is None:
            return float(self.snap_mm)
        return float(layer.snap_mm)

    def _unique_name(self, wanted: str, *, ignore: int | None = None) -> str:
        used = {row.name.casefold() for row in self.layers if row.layer_id != ignore}
        if wanted.casefold() not in used:
            return wanted
        number = 2
        while f"{wanted} {number}".casefold() in used:
            number += 1
        return f"{wanted} {number}"

    def add_layer(
        self, name: str | None = None, role: str = "porous_medium"
    ) -> PaintedLayer:
        """A new layer in a facies of its own."""

        layer_id = max((row.layer_id for row in self.layers), default=0) + 1
        if layer_id > 255:
            raise ValueError("At most 255 layers can be painted")
        used_colors = {row.color for row in self.layers}
        color = next(
            (value for value in LAYER_COLORS if value not in used_colors),
            LAYER_COLORS[(layer_id - 1) % len(LAYER_COLORS)],
        )
        text = self._unique_name(str(name or "").strip() or f"Layer {layer_id}")
        facies = self._unique_facies_name(text)
        layer = PaintedLayer(layer_id, text, color, facies)
        self.layers.append(layer)
        self.facies[facies] = FaciesEntry(
            role=role if role in LAYER_ROLES else "porous_medium"
        )
        return layer

    def remove_layer(self, layer_id: int) -> None:
        """Remove a layer with its polygons and strokes; a facies left without
        layers goes too."""

        layer = self.layer(layer_id)
        self.layers = [row for row in self.layers if row.layer_id != layer_id]
        self.strokes = [row for row in self.strokes if row.layer_id != layer_id]
        self.polygons = [row for row in self.polygons if row.layer_id != layer_id]
        if layer is not None and not self.layers_in_facies(layer.facies):
            self.facies.pop(layer.facies, None)

    def rename_layer(self, layer_id: int, name: str) -> None:
        layer = self.layer(layer_id)
        text = str(name or "").strip()
        if layer is None or not text or text == layer.name:
            return
        if text.casefold() in {
            row.name.casefold() for row in self.layers if row.layer_id != layer_id
        }:
            raise ValueError(f"Another layer is already called {text!r}")
        old = layer.name
        layer.name = text
        # A facies that only carries this layer's old name follows the rename.
        alone = self.layers_in_facies(layer.facies) == [layer]
        if layer.facies == old and alone and text not in self.facies:
            self._rename_facies_key(old, text)

    # ------------------------------------------------------------ facies

    def _unique_facies_name(self, wanted: str) -> str:
        used = {name.casefold() for name in self.facies}
        if wanted.casefold() not in used:
            return wanted
        number = 2
        while f"{wanted} {number}".casefold() in used:
            number += 1
        return f"{wanted} {number}"

    def _rename_facies_key(self, old: str, new: str) -> None:
        """Give a facies a new (unused) name in the same position."""

        self.facies = {
            (new if name == old else name): entry for name, entry in self.facies.items()
        }
        for row in self.layers:
            if row.facies == old:
                row.facies = new

    def _insert_facies(
        self, name: str, entry: FaciesEntry, *, after: str | None = None
    ) -> None:
        items = list(self.facies.items())
        position = len(items)
        if after is not None:
            names = [key for key, _value in items]
            if after in names:
                position = names.index(after) + 1
        items.insert(position, (name, entry))
        self.facies = dict(items)

    def facies_order(self) -> list[str]:
        """All facies in their order, empty ones included."""

        return list(self.facies)

    def add_facies(self, name: str | None = None, role: str = "porous_medium") -> str:
        """An empty facies; layers are added with :meth:`set_layer_in_facies`."""

        text = str(name or "").strip()
        if not text:
            number = len(self.facies) + 1
            text = f"Facies {number}"
            while text.casefold() in {key.casefold() for key in self.facies}:
                number += 1
                text = f"Facies {number}"
        if text in self.facies:
            raise ValueError(f"A facies is already called {text!r}")
        self.facies[text] = FaciesEntry(
            role=role if role in LAYER_ROLES else "porous_medium"
        )
        return text

    def remove_facies(self, name: str) -> None:
        """Remove a facies; each of its layers gets a facies of its own."""

        if name not in self.facies:
            return
        members = self.layers_in_facies(name)
        role = self.facies[name].role
        items: list[tuple[str, FaciesEntry]] = []
        for key, entry in self.facies.items():
            if key != name:
                items.append((key, entry))
                continue
            taken = {other.casefold() for other in self.facies if other != name}
            for layer in members:
                text = layer.name
                number = 2
                while text.casefold() in taken:
                    text = f"{layer.name} {number}"
                    number += 1
                taken.add(text.casefold())
                layer.facies = text
                items.append((text, FaciesEntry(role=role)))
        self.facies = dict(items)

    def set_layer_facies(self, layer_id: int, facies: str) -> None:
        """Move a layer into a facies (created when new).

        A new name for a layer alone in its facies renames that facies; a
        facies that loses its last layer this way is removed (merged).
        """

        layer = self.layer(layer_id)
        if layer is None:
            return
        text = str(facies or "").strip() or layer.name
        if text == layer.facies:
            return
        old = layer.facies
        alone = self.layers_in_facies(old) == [layer]
        if text not in self.facies:
            if alone:
                self._rename_facies_key(old, text)
                return
            role = self.facies.get(old, FaciesEntry()).role
            self._insert_facies(text, FaciesEntry(role=role), after=old)
        layer.facies = text
        if alone:
            self.facies.pop(old, None)

    def set_layer_in_facies(self, layer_id: int, facies: str, member: bool) -> None:
        """Tick or untick a layer in a facies' layer list.

        Ticking moves the layer from its facies (which is removed when this
        leaves it empty). Unticking gives the layer a facies of its own, named
        after it, and keeps the (possibly empty) facies.
        """

        layer = self.layer(layer_id)
        if layer is None or facies not in self.facies:
            return
        if member:
            if layer.facies != facies:
                self.set_layer_facies(layer_id, facies)
            return
        if layer.facies != facies:
            return
        text = self._unique_facies_name(layer.name)
        self._insert_facies(
            text, FaciesEntry(role=self.facies[facies].role), after=facies
        )
        layer.facies = text

    def rename_facies(self, old: str, new: str) -> None:
        """Rename a facies; a name already in use merges the two facies."""

        text = str(new or "").strip()
        if not text or text == old or old not in self.facies:
            return
        if text not in self.facies:
            self._rename_facies_key(old, text)
            return
        self.facies.pop(old)
        for row in self.layers:
            if row.facies == old:
                row.facies = text

    def set_facies(
        self,
        name: str,
        *,
        role: Any = _UNSET,
        porosity: Any = _UNSET,
        permeability: Any = _UNSET,
    ) -> None:
        entry = self.facies.setdefault(name, FaciesEntry())
        if role is not _UNSET:
            if role not in LAYER_ROLES:
                raise ValueError(f"Unknown role {role!r}")
            entry.role = role
        if porosity is not _UNSET:
            entry.porosity = _number(porosity)
        if permeability is not _UNSET:
            entry.permeability = _number(permeability)

    def ensure_layer_facies(self) -> None:
        """Give every layer an existing facies."""

        for row in self.layers:
            if not row.facies:
                row.facies = self._unique_facies_name(row.name)
            self.facies.setdefault(row.facies, FaciesEntry())

    def prune_facies(self) -> None:
        """Remove facies without layers."""

        used = {row.facies for row in self.layers}
        self.facies = {
            name: entry for name, entry in self.facies.items() if name in used
        }
        self.ensure_layer_facies()

    def facies_of(self, layer_id: int) -> FaciesEntry:
        layer = self.layer(layer_id)
        if layer is None:
            return FaciesEntry(role="excluded")
        return self.facies.setdefault(layer.facies, FaciesEntry())

    def layers_in_facies(self, name: str) -> list[PaintedLayer]:
        return [row for row in self.layers if row.facies == name]

    # ----------------------------------------------------------- strokes

    def add_stroke(self, stroke: Stroke) -> None:
        if stroke.layer_id != ERASER and self.layer(stroke.layer_id) is None:
            raise ValueError(f"Unknown layer {stroke.layer_id}")
        if not stroke.points or stroke.radius_px <= 0.0:
            raise ValueError("A stroke needs points and a positive radius")
        self.strokes.append(stroke)

    def stroke_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for row in self.strokes:
            counts[row.layer_id] = counts.get(row.layer_id, 0) + 1
        return counts

    def clear_layer_strokes(self, layer_id: int) -> int:
        before = len(self.strokes)
        self.strokes = [row for row in self.strokes if row.layer_id != layer_id]
        return before - len(self.strokes)

    # ---------------------------------------------------------- polygons

    def _check_polygon(self, polygon: Polygon) -> None:
        if self.layer(polygon.layer_id) is None:
            raise ValueError(f"Unknown layer {polygon.layer_id}")
        points = [(float(row), float(col)) for row, col in polygon.points]
        if len(points) < 3 or not all(
            math.isfinite(value) for point in points for value in point
        ):
            raise ValueError("A polygon needs at least three points")
        if Polygon(polygon.layer_id, points).area() < 1.0:
            raise ValueError("The polygon encloses no area")

    def add_polygon(self, polygon: Polygon) -> int:
        """Add a polygon; returns its index."""

        self._check_polygon(polygon)
        self.polygons.append(polygon)
        return len(self.polygons) - 1

    def replace_polygon(self, index: int, polygon: Polygon) -> None:
        self._check_polygon(polygon)
        self.polygons[index] = polygon

    def remove_polygon(self, index: int) -> Polygon:
        return self.polygons.pop(index)

    def move_polygon(self, index: int, layer_id: int) -> None:
        """Give a polygon to another layer."""

        if self.layer(layer_id) is None:
            raise ValueError(f"Unknown layer {layer_id}")
        self.polygons[index].layer_id = int(layer_id)

    def polygon_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for row in self.polygons:
            counts[row.layer_id] = counts.get(row.layer_id, 0) + 1
        return counts

    def clear_layer_marks(self, layer_id: int) -> int:
        """Remove the polygons and strokes of a layer; returns how many."""

        before = len(self.polygons)
        self.polygons = [row for row in self.polygons if row.layer_id != layer_id]
        return before - len(self.polygons) + self.clear_layer_strokes(layer_id)

    def marked_layers(self) -> set[int]:
        """Layers with at least one polygon or stroke."""

        return {row.layer_id for row in self.polygons} | {
            row.layer_id for row in self.strokes if row.layer_id != ERASER
        }

    # ------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "format": LAYER_PAINTING_FORMAT,
            "run_id": self.run_id,
            "image": {
                "path": machine_stored_text(self.image_path),
                "size": list(self.image_size) if self.image_size else None,
                "stamp": self.image_stamp,
            },
            "layers": [
                {
                    "layer_id": row.layer_id,
                    "name": row.name,
                    "color": row.color,
                    "facies": row.facies,
                    **({} if row.snap_mm is None else {"snap_mm": row.snap_mm}),
                    **({"always_on_top": True} if row.always_on_top else {}),
                }
                for row in self.layers
            ],
            "facies": {
                name: {
                    "role": entry.role,
                    "porosity": entry.porosity,
                    "permeability": entry.permeability,
                }
                for name, entry in self.facies.items()
            },
            "permeability_unit": self.permeability_unit,
            "proposal_settings": {
                "smoothing": self.smoothing,
                "min_region_cm2": self.min_region_cm2,
                "snap_mm": self.snap_mm,
                "edge_fill_mm": self.edge_fill_mm,
                "detail": self.detail,
                # Only when shared edges are kept: sessions with the default
                # stay as they were.
                **({} if self.move_shared_edges else {"move_shared_edges": False}),
            },
            "updated_utc": self.updated_utc,
            "updated_by": self.updated_by,
            # One polygon or stroke per line keeps the file readable:
            # [layer, [[row, col], ...]] and [layer, radius, [[row, col], ...]].
            "polygons": [row.to_list() for row in self.polygons],
            "strokes": [row.to_list() for row in self.strokes],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PaintingSession:
        if payload.get("format") != LAYER_PAINTING_FORMAT:
            raise ValueError("Unsupported layer painting file")
        image = dict(payload.get("image") or {})
        size = image.get("size")
        settings = dict(payload.get("proposal_settings") or {})
        snap = _number(settings.get("snap_mm"))
        edge = _number(settings.get("edge_fill_mm"))
        detail = str(settings.get("detail") or "normal")
        session = cls(
            run_id=str(payload.get("run_id") or ""),
            image_path=machine_local_text(str(image.get("path") or "")),
            image_size=(int(size[0]), int(size[1])) if size else None,
            image_stamp=str(image.get("stamp") or ""),
            permeability_unit=str(payload.get("permeability_unit") or "darcy"),
            smoothing=str(settings.get("smoothing") or "medium"),
            min_region_cm2=float(_number(settings.get("min_region_cm2")) or 0.0),
            snap_mm=(
                DEFAULT_SNAP_MM
                if snap is None
                else float(min(max(snap, 0.0), MAX_SNAP_MM))
            ),
            edge_fill_mm=(
                DEFAULT_EDGE_FILL_MM
                if edge is None
                else float(min(max(edge, 0.0), MAX_SNAP_MM))
            ),
            detail=detail if detail in PROPOSAL_DETAILS else "normal",
            move_shared_edges=bool(settings.get("move_shared_edges", True)),
            updated_utc=str(payload.get("updated_utc") or ""),
            updated_by=str(payload.get("updated_by") or ""),
        )
        if session.smoothing not in SMOOTHING_SIGMA:
            session.smoothing = "medium"
        for row in payload.get("layers") or []:
            layer_snap = _number(row.get("snap_mm"))
            session.layers.append(
                PaintedLayer(
                    int(row["layer_id"]),
                    str(row.get("name") or f"Layer {row['layer_id']}"),
                    str(row.get("color") or LAYER_COLORS[0]),
                    str(row.get("facies") or row.get("name") or ""),
                    None
                    if layer_snap is None
                    else float(min(max(layer_snap, 0.0), MAX_SNAP_MM)),
                    always_on_top=row.get("always_on_top") is True,
                )
            )
        for name, row in dict(payload.get("facies") or {}).items():
            role = str(row.get("role") or "porous_medium")
            session.facies[str(name)] = FaciesEntry(
                role=role if role in LAYER_ROLES else "porous_medium",
                porosity=_number(row.get("porosity")),
                permeability=_number(row.get("permeability")),
            )
        layer_ids = {row.layer_id for row in session.layers}
        known = layer_ids | {ERASER}
        session.strokes = [
            stroke
            for stroke in (
                Stroke.from_list(row) for row in payload.get("strokes") or []
            )
            if stroke.layer_id in known and stroke.points
        ]
        session.polygons = [
            polygon
            for polygon in (
                Polygon.from_list(row) for row in payload.get("polygons") or []
            )
            if polygon.layer_id in layer_ids and len(polygon.points) >= 3
        ]
        session.ensure_layer_facies()
        return session


def painting_folder(project: FluidFlowerProject) -> Path:
    return Path(project.root) / "annotations" / "layer_painting"


def _file_stem(run_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id)).strip("._")
    return stem or "run"


def session_path(project: FluidFlowerProject, run_id: str) -> Path:
    return painting_folder(project) / f"{_file_stem(run_id)}.json"


def proposal_path(project: FluidFlowerProject, run_id: str) -> Path:
    return painting_folder(project) / f"{_file_stem(run_id)}.proposal.npz"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_painting_session(
    project: FluidFlowerProject, reference: SeriesRunInputs
) -> PaintingSession:
    """The saved session of the reference run, or the default layers."""

    path = session_path(project, reference.run_id)
    stamp = image_stamp(reference.baseline_image)
    if _preset_io.exists(path, 'is_file'):
        session = PaintingSession.from_dict(
            json.loads(_preset_io.read_text(path,encoding="utf-8-sig"))
        )
        session.run_id = reference.run_id
        if reference.baseline_image is not None:
            current = Path(reference.baseline_image)
            session.photo_changed = bool(session.strokes) and (
                Path(session.image_path).name != current.name
                or bool(session.image_stamp and stamp and session.image_stamp != stamp)
            )
            session.image_path = str(current)
            session.image_stamp = stamp
        return session
    size = None
    if reference.baseline_image is not None:
        try:
            size = photo_size(reference.baseline_image,
                              camera_u=reference_camera_u(project, reference))
        except (OSError, ValueError):
            size = None
    session = PaintingSession.default(
        reference.run_id, reference.baseline_image or "", size
    )
    session.image_stamp = stamp
    return session


def image_stamp(path: Path | str | None) -> str:
    """Size and modification time of a file ("" when unavailable)."""

    if path is None or not str(path):
        return ""
    try:
        status = _preset_io.file_stat(Path(path))
    except OSError:
        return ""
    return f"{status.st_size}:{status.st_mtime_ns}"


def save_painting_session(
    project: FluidFlowerProject, session: PaintingSession, *, actor: str = ""
) -> Path:
    session.updated_utc = utc_now()
    session.updated_by = str(actor or session.updated_by or "")
    path = session_path(project, session.run_id)
    _atomic_write_text(path, _session_text(session))
    return path


def _session_text(session: PaintingSession) -> str:
    """The session file's text: settings indented, one polygon or stroke per line."""

    payload = session.to_dict()
    marks = {key: payload.pop(key) for key in ("polygons", "strokes")}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    text = text[: text.rfind("}")].rstrip()
    # Polygons and strokes one per line: readable diffs and a compact file.
    for key, rows in marks.items():
        lines = ",\n".join(
            "    " + json.dumps(row, separators=(",", ":")) for row in rows
        )
        text += f',\n  "{key}": [' + ("\n" + lines + "\n  " if rows else "") + "]"
    text += "\n}\n"
    return text


# ---------------------------------------------------------------------------
# Photograph


def contrast_limits(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    low: float = 0.5,
    high: float = 99.5,
) -> tuple[np.ndarray, np.ndarray]:
    """The two channel percentiles that ``enhance_contrast`` stretches between."""

    image = np.asarray(rgb, dtype=np.float32)
    if image.max(initial=0.0) > 1.5:
        image = image / 255.0
    sample = image[mask] if mask is not None and np.any(mask) else image.reshape(-1, 3)
    if sample.shape[0] > 400_000:
        step = int(math.ceil(sample.shape[0] / 400_000))
        sample = sample[::step]
    return np.percentile(sample, low, axis=0), np.percentile(sample, high, axis=0)


def enhance_contrast(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    low: float = 0.5,
    high: float = 99.5,
) -> np.ndarray:
    """Stretch each channel between two percentiles; returns uint8 RGB."""

    image = np.asarray(rgb, dtype=np.float32)
    if image.max(initial=0.0) > 1.5:
        image = image / 255.0
    lower, upper = contrast_limits(image, mask, low=low, high=high)
    span = np.maximum(upper - lower, 1.0e-3)
    stretched = np.clip((image - lower) / span, 0.0, 1.0)
    return np.rint(255.0 * stretched).astype(np.uint8)


def stretch_contrast_rgb8(
    rgb: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """``enhance_contrast``'s stretch between given limits, for uint8 RGB.

    Gives the same values as ``enhance_contrast`` with these limits, through
    one lookup table per channel instead of a floating-point copy, so that a
    full-resolution photograph can be stretched with little memory.
    """

    values = np.arange(256, dtype=np.float32)[:, None] / 255.0
    span = np.maximum(upper - lower, 1.0e-3)
    table = np.rint(255.0 * np.clip((values - lower) / span, 0.0, 1.0)).astype(np.uint8)
    image = np.asarray(rgb, dtype=np.uint8)
    stretched = np.empty_like(image)
    for channel in range(3):
        stretched[..., channel] = table[image[..., channel], channel]
    return stretched


@dataclass
class PaintingImage:
    """A raw photograph at display and analysis resolution."""

    path: Path
    image_size: tuple[int, int]
    display: np.ndarray
    analysis: np.ndarray
    _features: np.ndarray | None = None
    # The run's camera-U profile when a RAW is shown on the camera grid U.
    camera_u: Any = None

    @property
    def analysis_shape(self) -> tuple[int, int]:
        return int(self.analysis.shape[0]), int(self.analysis.shape[1])

    def analysis_scale(self) -> tuple[float, float]:
        """Analysis pixels per raw pixel along rows and columns."""

        height, width = self.analysis_shape
        return height / self.image_size[1], width / self.image_size[0]

    def features(self) -> np.ndarray:
        if self._features is None:
            from darsia.fluidflower.facies_model import extract_facies_features

            self._features = extract_facies_features(self.analysis)
        return self._features


def with_analysis_width(image: PaintingImage, width: int) -> PaintingImage:
    """The same photograph with its analysis grid ``width`` columns wide.

    The analysis grid is reduced from the display image, so it is at most as
    wide as that. The display image is shared; features are computed anew.
    """

    import cv2

    display = image.display
    target = max(8, min(int(width), int(display.shape[1])))
    if target == image.analysis.shape[1]:
        return image
    if target == display.shape[1]:
        analysis = display
    else:
        height = max(8, int(round(display.shape[0] * target / display.shape[1])))
        analysis = cv2.resize(display, (target, height), interpolation=cv2.INTER_AREA)
    return PaintingImage(
        image.path,
        tuple(image.image_size),
        display,
        np.ascontiguousarray(analysis, dtype=np.float32),
        camera_u=image.camera_u,
    )


def photo_camera_u(project: FluidFlowerProject, run_id: str, path: Path | str | None):
    """The camera-U profile one photograph of a run is shown on, or None.

    Only a camera RAW of a run with a configured profile (series.toml [native],
    N106) is shown on the camera grid U; a JPEG or an unprofiled RAW keeps its
    own grid. A present but malformed profile raises.
    """

    from darsia.fluidflower.image_formats import image_family

    if path is None or image_family(path) != "raw":
        return None
    return run_camera_u(project, run_id)


def run_camera_u(project: FluidFlowerProject, run_id: str):
    """The camera-U profile configured for a run of the project's series, or None.

    The profiles live in the series' series.toml [native] document. A
    geometry-and-facies package is not the series root: it names that root in
    its manifest (tools.native_series.series_root). Any other project is taken
    to be the series root itself. A present but malformed profile raises.
    """

    manifest = getattr(project, "manifest", None)
    linked = native_series_link(project).get("series_root") if isinstance(manifest, Mapping) else None
    root = linked or getattr(project, "root", None)
    if root is None:
        return None
    from darsia.fluidflower.series_native_context import camera_u_profile

    return camera_u_profile(root, run_id)


def reference_camera_u(project: FluidFlowerProject, reference: SeriesRunInputs | None):
    """The camera-U profile of a run's reference photograph, or None."""

    if reference is None:
        return None
    return photo_camera_u(project, reference.run_id, reference.baseline_image)


def load_painting_image(
    path: Path,
    *,
    camera_u: Any = None,
    analysis_width: int = ANALYSIS_WIDTH,
    display_limit: int = DISPLAY_LIMIT_PX,
) -> PaintingImage:
    """Read a photograph reduced for display and for the pixel classifier.

    With the run's camera-U profile a RAW is read on the camera grid U.
    """

    import cv2

    from darsia.fluidflower.image_formats import image_family
    from darsia.fluidflower.photograph_io import load_reduced

    selected = Path(path)
    width, height = photo_size(selected, camera_u=camera_u)
    factor = max(1, int(math.ceil(max(width, height) / float(display_limit))))
    display = load_reduced(selected, factor, camera_u=camera_u)
    target_width = min(int(analysis_width), display.shape[1])
    if target_width < display.shape[1]:
        target_height = max(8, int(round(height * target_width / width)))
        analysis = cv2.resize(
            display, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
    else:
        analysis = display.copy()
    return PaintingImage(
        selected,
        (int(width), int(height)),
        np.ascontiguousarray(display, dtype=np.float32),
        np.ascontiguousarray(analysis, dtype=np.float32),
        camera_u=camera_u if camera_u is not None and image_family(selected) == "raw" else None,
    )


# ---------------------------------------------------------------------------
# Rasterization


def _grid_points(
    points: Sequence[tuple[float, float]], scale: tuple[float, float], shift: int
) -> np.ndarray:
    factor = float(1 << shift)
    return np.asarray(
        [
            (
                int(round(((col + 0.5) * scale[1] - 0.5) * factor)),
                int(round(((row + 0.5) * scale[0] - 0.5) * factor)),
            )
            for row, col in points
        ],
        dtype=np.int32,
    )


def rasterize_strokes(
    strokes: Sequence[Stroke],
    shape: tuple[int, int],
    scale: tuple[float, float],
) -> np.ndarray:
    """Layer ID of each grid pixel after all strokes (0: not painted).

    ``scale`` gives grid pixels per raw pixel along rows and columns.
    """

    import cv2

    painted = np.zeros(shape, dtype=np.uint8)
    shift = 4
    factor = 0.5 * (scale[0] + scale[1])
    for stroke in strokes:
        radius = max(0.5, float(stroke.radius_px) * factor)
        points = _grid_points(stroke.points, scale, shift)
        value = int(stroke.layer_id)
        if len(points) == 1 or np.all(points == points[0]):
            cv2.circle(
                painted,
                (int(points[0][0]), int(points[0][1])),
                max(1, int(round(radius * (1 << shift)))),
                value,
                thickness=-1,
                lineType=cv2.LINE_8,
                shift=shift,
            )
            continue
        cv2.polylines(
            painted,
            [points.reshape(-1, 1, 2)],
            isClosed=False,
            color=value,
            thickness=max(1, int(round(2.0 * radius))),
            lineType=cv2.LINE_8,
            shift=shift,
        )
    return painted


def _fill_polygon(
    target: np.ndarray, polygon: Polygon, scale: tuple[float, float], value: int
) -> None:
    import cv2

    if len(polygon.points) < 3:
        return
    shift = 4
    points = _grid_points(polygon.points, scale, shift).reshape(-1, 1, 2)
    cv2.fillPoly(target, [points], int(value), lineType=cv2.LINE_8, shift=shift)


def rasterize_polygons(
    polygons: Sequence[Polygon],
    shape: tuple[int, int],
    scale: tuple[float, float],
) -> np.ndarray:
    """Layer ID of the smallest polygon covering each grid pixel (0: none).

    Polygons are filled from the largest to the smallest, so that where
    polygons of different layers overlap the smaller one wins (a lens drawn
    inside a layer keeps its own layer); equal areas keep the drawing order.
    """

    owner = np.zeros(shape, dtype=np.uint8)
    order = sorted(range(len(polygons)), key=lambda index: -polygons[index].area())
    for index in order:
        _fill_polygon(owner, polygons[index], scale, polygons[index].layer_id)
    return owner


def _point_groups(
    points_by_polygon: Mapping[int, Sequence[tuple[float, float]]], tolerance: float
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Every polygon point as (polygon, point), and the group of each.

    Points within ``tolerance`` raw pixels of each other, also through a chain
    of such points, are in one group.
    """

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    keys = [
        (int(index), vertex)
        for index, points in points_by_polygon.items()
        for vertex in range(len(points))
    ]
    if not keys:
        return [], np.zeros(0, dtype=np.int64)
    coordinates = np.asarray(
        [points_by_polygon[index][vertex] for index, vertex in keys], dtype=np.float64
    )
    pairs = cKDTree(coordinates).query_pairs(float(tolerance), output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(keys), len(keys))
    )
    _count, groups = connected_components(graph, directed=False)
    return keys, groups


def shared_points(
    points_by_polygon: Mapping[int, Sequence[tuple[float, float]]],
    tolerance: float = SHARED_POINT_PX,
) -> list[list[tuple[int, int]]]:
    """Points that several polygons share.

    ``points_by_polygon`` maps a polygon index to its points (raw row, col).
    Points within ``tolerance`` raw pixels of each other, also through a
    chain of such points, are one shared point when they belong to at least
    two polygons. Returns each shared point as its sorted (polygon, point)
    indices, ordered by the first of them.
    """

    keys, groups = _point_groups(points_by_polygon, tolerance)
    members: dict[int, list[tuple[int, int]]] = {}
    for key, group in zip(keys, groups):
        members.setdefault(int(group), []).append(key)
    return sorted(
        sorted(rows)
        for rows in members.values()
        if len({index for index, _vertex in rows}) >= 2
    )


def _edge_sharing(
    polygons: Sequence[Polygon], tolerance: float
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[Any]]:
    """The edges that polygons of different layers share, and all other
    edges, each as its two end points (raw row, col)."""

    keys, groups = _point_groups(
        {index: polygon.points for index, polygon in enumerate(polygons)}, tolerance
    )
    group_of = {key: int(group) for key, group in zip(keys, groups)}
    edges: list[tuple[frozenset[int], tuple[float, float], tuple[float, float]]] = []
    layers: dict[frozenset[int], set[int]] = {}
    for index, polygon in enumerate(polygons):
        count = len(polygon.points)
        for vertex in range(count):
            following = (vertex + 1) % count
            ends = frozenset(
                {group_of[(index, vertex)], group_of[(index, following)]}
            )
            edges.append((ends, polygon.points[vertex], polygon.points[following]))
            if len(ends) == 2:
                layers.setdefault(ends, set()).add(int(polygon.layer_id))
    shared: list[tuple[tuple[float, float], tuple[float, float]]] = []
    other: list[tuple[tuple[float, float], tuple[float, float]]] = []
    seen: set[frozenset[int]] = set()
    for ends, start, end in edges:
        if len(layers.get(ends, ())) < 2:
            other.append((start, end))
        elif ends not in seen:
            seen.add(ends)
            shared.append((start, end))
    return shared, other


def shared_edges(
    polygons: Sequence[Polygon], tolerance: float = SHARED_POINT_PX
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Edges that polygons of different layers share.

    An edge is shared when polygons of at least two layers each have an edge
    between the same two shared points (see :func:`shared_points`). Each
    shared edge is returned once, as its two end points (raw row, col) in the
    first polygon that has it.
    """

    return _edge_sharing(polygons, tolerance)[0]


def _segments_mask(
    segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    shape: tuple[int, int],
    scale: tuple[float, float],
    width_px: int,
) -> np.ndarray:
    """Grid pixels on lines ``width_px`` grid pixels wide along segments
    given by their end points (raw row, col)."""

    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    shift = 4
    for start, end in segments:
        first, second = _grid_points([start, end], scale, shift)
        cv2.line(
            mask,
            (int(first[0]), int(first[1])),
            (int(second[0]), int(second[1])),
            1,
            thickness=max(1, int(width_px)),
            lineType=cv2.LINE_8,
            shift=shift,
        )
    return mask.astype(bool)


def _distance_to(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (grid pixels) to the nearest pixel of ``mask``."""

    import cv2

    if not np.any(mask):
        return np.full(mask.shape, np.inf, dtype=np.float32)
    return cv2.distanceTransform(
        np.ascontiguousarray(~mask, dtype=np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )


@dataclass
class PolygonGuide:
    """What the polygons fix, and which layers they allow, on one grid."""

    # Layer of the smallest polygon at each pixel inside the frame (0: none).
    owner: np.ndarray
    # Polygon pixels farther than the snap distance from every polygon edge.
    fixed: np.ndarray
    # Pixels near polygon edges where only nearby layers compete.
    restricted: np.ndarray
    # Distance to each layer's polygons inside the frame (grid pixels).
    distance: dict[int, np.ndarray]
    # Layer of each pixel that can serve as a training sample (0: none).
    training: np.ndarray
    snap_px: float
    # Layers with their own snap distance (grid pixels).
    layer_snap_px: dict[int, float] = field(default_factory=dict)

    def layer_snap(self, layer_id: int) -> float:
        """A layer's snap distance in grid pixels: its own, else the common one."""

        return self.layer_snap_px.get(int(layer_id), self.snap_px)

    def near(self, layer_id: int) -> np.ndarray | None:
        """Pixels within the layer's snap distance of its polygons."""

        distance = self.distance.get(int(layer_id))
        return None if distance is None else distance <= self.layer_snap(layer_id)


def polygon_guide(
    polygons: Sequence[Polygon],
    shape: tuple[int, int],
    scale: tuple[float, float],
    inside: np.ndarray,
    snap_px: float,
    *,
    min_training_pixels: int = 50,
    layer_snap_px: Mapping[int, float] | None = None,
    keep_shared_edges: bool = False,
) -> PolygonGuide:
    """Fixed interiors and allowed layers of a set of polygons on a grid.

    A polygon pixel farther than ``snap_px`` from every polygon edge keeps the
    polygon's layer (the smallest polygon wins where layers overlap); with
    ``snap_px`` 0 every polygon pixel does. Edges are the outlines of each
    layer's union of polygons, so touching polygons of one layer form no edge.

    Near edges, where the area around is covered by polygons, only layers
    whose polygons (or strokes, see :func:`propose_layers`) lie within
    ``snap_px`` compete. Areas that no polygon covers may hold any layer, so
    pixels within twice the snap distance of such an area are not restricted.

    ``layer_snap_px`` gives single layers their own snap distance in place of
    ``snap_px``, both for fixing their polygon pixels and for how far from
    their polygons they compete. With 0 a layer keeps its polygons exactly and
    does not spread beyond them where the area is covered by polygons.

    With ``keep_shared_edges`` the edges that polygons of different layers
    share (:func:`shared_edges`) get no snap band: polygon pixels along them
    are fixed, so the boundary stays on the shared edge, unless they lie
    within the snap distance of another edge. Which pixels may serve as
    training samples does not change.
    """

    import cv2

    owner = rasterize_polygons(polygons, shape, scale)
    owner[~inside] = 0
    covered = owner > 0
    masks: dict[int, np.ndarray] = {}
    for polygon in polygons:
        mask = masks.get(polygon.layer_id)
        if mask is None:
            mask = masks[polygon.layer_id] = np.zeros(shape, dtype=np.uint8)
        _fill_polygon(mask, polygon, scale, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    edges = np.zeros(shape, dtype=bool)
    distance: dict[int, np.ndarray] = {}
    for layer_id, mask in masks.items():
        # The image border is not an edge (erosion keeps pixels there).
        edges |= (mask > 0) & (cv2.erode(mask, kernel) == 0)
        distance[layer_id] = _distance_to((mask > 0) & inside)
    edges &= inside
    distance_edge = _distance_to(edges)
    # The edges a snap band lies along.
    band_edge = distance_edge
    if keep_shared_edges:
        shared, other = _edge_sharing(polygons, SHARED_POINT_PX)
        if shared:
            # Edge pixels on a shared edge, nearer to it than to any other
            # polygon edge, do not count.
            to_shared = _distance_to(_segments_mask(shared, shape, scale, 1))
            to_other = _distance_to(_segments_mask(other, shape, scale, 1))
            along = (to_shared <= to_other) & (to_shared <= 1.5)
            band_edge = _distance_to(edges & ~along)

    snap = max(0.0, float(snap_px))
    own_snap = {int(key): max(0.0, float(value)) for key, value in (layer_snap_px or {}).items()}
    if own_snap:
        snap_map = np.full(shape, snap, dtype=np.float32)
        for layer_id, value in own_snap.items():
            snap_map[owner == layer_id] = value
        fixed = covered & ((band_edge > snap_map) | (snap_map <= 0.0))
    else:
        fixed = covered & (band_edge > snap) if snap > 0.0 else covered.copy()
    reach = max([snap, *own_snap.values()])
    if reach > 0.0 and distance:
        nearest = np.minimum.reduce(list(distance.values()))
        # Far from every polygon, any layer may occur.
        free = inside & (nearest > reach)
        restricted = inside & ~fixed & (_distance_to(free) > 2.0 * reach)
    else:
        restricted = np.zeros(shape, dtype=bool)

    training = np.zeros(shape, dtype=np.uint8)
    for layer_id in masks:
        own = owner == layer_id
        margin = max(own_snap.get(int(layer_id), snap), 2.0)
        for limit in (margin, min(margin, 2.0), -1.0):
            chosen = own & (distance_edge > limit)
            if np.count_nonzero(chosen) >= min_training_pixels:
                break
        training[chosen] = layer_id
    return PolygonGuide(
        owner=owner,
        fixed=fixed,
        restricted=restricted,
        distance=distance,
        training=training,
        snap_px=snap,
        layer_snap_px=own_snap,
    )


def frame_edge_fill(guide: PolygonGuide, inside: np.ndarray, edge_px: float) -> np.ndarray:
    """Layer of each uncovered pixel in a strip along the rig-frame edge (0: none).

    Polygons drawn up to the frame edge seldom reach it exactly. A pixel no
    polygon covers that lies within ``edge_px`` of the edge of ``inside`` (the
    photograph's border counts where the frame reaches beyond it) and within
    ``edge_px`` of a polygon takes the layer of the nearest polygon, so the
    classifier does not label the strip as some other layer.
    """

    fill = np.zeros(inside.shape, dtype=np.uint8)
    if edge_px <= 0.0 or not guide.distance:
        return fill
    padded = np.pad(inside, 1, constant_values=False)
    to_edge = _distance_to(~padded)[1:-1, 1:-1]
    layers = sorted(guide.distance)
    stack = np.stack([guide.distance[layer] for layer in layers])
    nearest = np.argmin(stack, axis=0)
    near = np.take_along_axis(stack, nearest[None], axis=0)[0]
    strip = inside & (guide.owner == 0) & (to_edge <= edge_px) & (near <= edge_px)
    fill[strip] = np.asarray(layers, dtype=np.uint8)[nearest[strip]]
    return fill


def snap_pixels(
    snap_mm: float,
    frame: RigFrame | None,
    scale: tuple[float, float],
    image_size: tuple[int, int],
) -> float:
    """The snap distance in grid pixels.

    Without a rig size the photograph is taken to be
    ``NOMINAL_IMAGE_WIDTH_M`` wide.
    """

    if snap_mm <= 0.0:
        return 0.0
    per_metre = frame_pixels_per_metre(frame)
    if per_metre is None:
        per_metre = float(image_size[0]) / NOMINAL_IMAGE_WIDTH_M
    return float(snap_mm) * 1.0e-3 * per_metre * math.sqrt(scale[0] * scale[1])


def frame_polygon(frame: RigFrame, scale: tuple[float, float]) -> np.ndarray | None:
    """Corners of a complete frame in grid (x, y) coordinates, clockwise."""

    if frame is None or not frame.complete():
        return None
    return np.asarray(
        [
            (
                (frame.corners[name][1] + 0.5) * scale[1] - 0.5,
                (frame.corners[name][0] + 0.5) * scale[0] - 0.5,
            )
            for name in ("top_left", "top_right", "bottom_right", "bottom_left")
        ],
        dtype=np.float64,
    )


def frame_mask(
    frame: RigFrame | None, shape: tuple[int, int], scale: tuple[float, float]
) -> np.ndarray:
    """Pixels inside the rig frame (all pixels without a complete frame)."""

    import cv2

    polygon = frame_polygon(frame, scale) if frame is not None else None
    if polygon is None:
        return np.ones(shape, dtype=bool)
    mask = np.zeros(shape, dtype=np.uint8)
    shift = 4
    points = np.rint(polygon * (1 << shift)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [points], 1, lineType=cv2.LINE_8, shift=shift)
    return mask.astype(bool)


def frame_pixels_per_metre(frame: RigFrame | None) -> float | None:
    """Raw pixels per metre of a frame with corners and size."""

    if frame is None or not frame.complete() or not frame.has_size():
        return None
    corners = {name: np.asarray(frame.corners[name], float) for name in frame.corners}
    width_px = 0.5 * (
        np.linalg.norm(corners["top_right"] - corners["top_left"])
        + np.linalg.norm(corners["bottom_right"] - corners["bottom_left"])
    )
    height_px = 0.5 * (
        np.linalg.norm(corners["bottom_left"] - corners["top_left"])
        + np.linalg.norm(corners["bottom_right"] - corners["top_right"])
    )
    values = [width_px / float(frame.width_m), height_px / float(frame.height_m)]
    return float(np.sqrt(values[0] * values[1])) if all(values) else None


def minimum_region_pixels(
    min_region_cm2: float,
    frame: RigFrame | None,
    scale: tuple[float, float],
    inside_pixels: int,
) -> int:
    """Smallest kept region on a grid; 0 keeps every region.

    Without a frame size the area is taken relative to a 1 m² window, that is
    ``min_region_cm2`` times 10⁻⁴ of the pixels inside the frame.
    """

    if min_region_cm2 <= 0.0:
        return 0
    per_metre = frame_pixels_per_metre(frame)
    if per_metre is not None:
        grid_per_metre = per_metre * math.sqrt(scale[0] * scale[1])
        return max(1, int(round(min_region_cm2 * 1.0e-4 * grid_per_metre**2)))
    return max(1, int(round(min_region_cm2 * 1.0e-4 * inside_pixels)))


def merge_small_regions(
    classes: np.ndarray,
    min_pixels: int,
    *,
    protected: np.ndarray | None = None,
    fixed: Sequence[int] = (OUTSIDE,),
    passes: int = 3,
) -> tuple[np.ndarray, int]:
    """Give connected regions below ``min_pixels`` the class of their largest
    neighbour; returns the classes (modified in place) and the merge count.

    Regions of a ``fixed`` class are kept and never receive merged pixels.
    Regions that contain a ``protected`` pixel of their own class are kept.
    """

    from scipy import ndimage as ndi

    if min_pixels <= 1:
        return classes, 0
    structure = ndi.generate_binary_structure(2, 1)
    fixed_values = np.asarray(sorted({int(value) for value in fixed}), dtype=np.int64)
    merged = 0
    height, width = classes.shape
    for _ in range(max(1, passes)):
        changed = 0
        counts = np.bincount(classes.ravel(), minlength=256)
        for value in np.flatnonzero(counts):
            if int(value) in fixed_values:
                continue
            components = np.zeros(classes.shape, dtype=np.int32)
            count = ndi.label(classes == value, structure=structure, output=components)
            if not count:
                continue
            sizes = np.bincount(components.ravel(), minlength=count + 1)
            small = np.flatnonzero(sizes < min_pixels)
            small = small[small != 0]
            if protected is not None and small.size:
                keep = np.unique(components[(protected == value) & (components > 0)])
                small = np.setdiff1d(small, keep)
            if not small.size:
                continue
            slices = ndi.find_objects(components)
            for component in small[np.argsort(sizes[small], kind="stable")]:
                bounds = slices[int(component) - 1]
                if bounds is None:
                    continue
                rows = slice(
                    max(0, bounds[0].start - 1), min(height, bounds[0].stop + 1)
                )
                cols = slice(
                    max(0, bounds[1].start - 1), min(width, bounds[1].stop + 1)
                )
                region = components[rows, cols] == component
                window = classes[rows, cols]
                region &= window == value
                if not np.any(region):
                    continue
                ring = ndi.binary_dilation(region, structure) & ~region
                neighbours = window[ring].astype(np.int64)
                neighbours = neighbours[
                    (neighbours != value) & ~np.isin(neighbours, fixed_values)
                ]
                if not neighbours.size:
                    continue
                window[region] = int(np.bincount(neighbours).argmax())
                changed += 1
        merged += changed
        if not changed:
            break
    return classes, merged


# ---------------------------------------------------------------------------
# Proposal


def _frame_key(frame: RigFrame | None) -> Any:
    if frame is None or not frame.complete():
        return None
    return {
        "corners": {
            name: [round(float(value), 1) for value in frame.corners[name]]
            for name in sorted(frame.corners)
        },
        "width_m": round(float(frame.width_m), 6) if frame.width_m else None,
        "height_m": round(float(frame.height_m), 6) if frame.height_m else None,
    }


def proposal_digest(session: PaintingSession, frame: RigFrame | None) -> str:
    """Digest of everything a proposal depends on."""

    payload = {
        "method": PROPOSAL_METHOD,
        "image": [
            Path(session.image_path).name,
            list(session.image_size) if session.image_size else None,
            session.image_stamp,
        ],
        "strokes": [row.to_list() for row in session.strokes],
        "smoothing": session.smoothing,
        "min_region_cm2": round(float(session.min_region_cm2), 4),
        "frame": _frame_key(frame),
    }
    if session.polygons:
        # Only with polygons, so that proposals from strokes alone stay current.
        payload["polygons"] = [row.to_list() for row in session.polygons]
        payload["snap_mm"] = round(float(session.snap_mm), 3)
        payload["edge_fill_mm"] = round(float(session.edge_fill_mm), 3)
        layer_snaps = {
            str(row.layer_id): round(float(row.snap_mm), 3)
            for row in session.layers
            if row.snap_mm is not None
        }
        if layer_snaps:
            payload["layer_snap_mm"] = layer_snaps
        if not session.move_shared_edges:
            # Only when shared edges are kept, so that other proposals stay current.
            payload["move_shared_edges"] = False
    if session.detail != "normal":
        payload["detail"] = session.detail
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


@dataclass
class LayerProposal:
    """Layer of each analysis pixel; 0 outside the rig frame."""

    classes: np.ndarray
    confidence: np.ndarray
    image_size: tuple[int, int]
    digest: str
    created_utc: str = ""
    elapsed_s: float = 0.0
    stroke_count: int = 0
    merged_regions: int = 0
    min_region_pixels: int = 0
    frame: dict[str, Any] | None = None
    layers: list[dict[str, Any]] = field(default_factory=list)
    polygon_count: int = 0
    # Snap distance on the proposal's grid (pixels).
    snap_px: float = 0.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema": 1,
            "format": LAYER_PROPOSAL_FORMAT,
            "method": PROPOSAL_METHOD,
            "image_size": list(self.image_size),
            "digest": self.digest,
            "created_utc": self.created_utc,
            "elapsed_s": self.elapsed_s,
            "stroke_count": self.stroke_count,
            "merged_regions": self.merged_regions,
            "min_region_pixels": self.min_region_pixels,
            "frame": self.frame,
            "layers": self.layers,
            "polygon_count": self.polygon_count,
            "snap_px": self.snap_px,
        }
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                classes=self.classes.astype(np.uint8),
                confidence=self.confidence.astype(np.uint8),
                meta=np.asarray(json.dumps(meta)),
            )
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path) -> LayerProposal:
        with np.load(_preset_io.source(path), allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            if meta.get("format") != LAYER_PROPOSAL_FORMAT:
                raise ValueError("Unsupported layer proposal file")
            return cls(
                classes=np.asarray(archive["classes"], dtype=np.uint8),
                confidence=np.asarray(archive["confidence"], dtype=np.uint8),
                image_size=(int(meta["image_size"][0]), int(meta["image_size"][1])),
                digest=str(meta.get("digest") or ""),
                created_utc=str(meta.get("created_utc") or ""),
                elapsed_s=float(meta.get("elapsed_s") or 0.0),
                stroke_count=int(meta.get("stroke_count") or 0),
                merged_regions=int(meta.get("merged_regions") or 0),
                min_region_pixels=int(meta.get("min_region_pixels") or 0),
                frame=meta.get("frame"),
                layers=list(meta.get("layers") or []),
                polygon_count=int(meta.get("polygon_count") or 0),
                snap_px=float(meta.get("snap_px") or 0.0),
            )

    def uncertain(self) -> np.ndarray:
        return (self.classes != OUTSIDE) & (
            self.confidence < int(round(255 * UNCERTAIN_BELOW))
        )


def _region_counts(classes: np.ndarray) -> dict[int, tuple[int, int]]:
    """Pixels and connected regions (4-connected) of each class."""

    from scipy import ndimage as ndi

    structure = ndi.generate_binary_structure(2, 1)
    result: dict[int, tuple[int, int]] = {}
    counts = np.bincount(classes.ravel(), minlength=256)
    for value in np.flatnonzero(counts):
        _labels, regions = ndi.label(classes == value, structure=structure)
        result[int(value)] = (int(counts[value]), int(regions))
    return result


def _training_index(
    layer_id: int,
    painted: np.ndarray,
    training: np.ndarray | None,
    budget: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Training pixels of one layer: painted pixels first, then polygon
    interiors (at most half the budget goes to strokes when both exist)."""

    strokes = np.flatnonzero(painted.ravel() == layer_id)
    polygons = (
        np.flatnonzero(training.ravel() == layer_id)
        if training is not None
        else np.empty(0, dtype=np.int64)
    )
    if not polygons.size:
        if strokes.size > budget:
            strokes = rng.choice(strokes, int(budget), replace=False)
        return strokes
    if strokes.size > budget // 2:
        strokes = rng.choice(strokes, int(budget // 2), replace=False)
    remaining = int(budget) - int(strokes.size)
    if polygons.size > remaining:
        polygons = rng.choice(polygons, remaining, replace=False)
    return np.concatenate([strokes, polygons])


def propose_layers(
    image: PaintingImage,
    session: PaintingSession,
    *,
    frame: RigFrame | None = None,
    n_estimators: int = 100,
    max_samples_per_layer: int = 6000,
    random_seed: int = 20_261_010,
    progress: ProgressCallback | None = None,
) -> LayerProposal:
    """Label every pixel inside the rig frame with one of the marked layers."""

    started = time.monotonic()
    if session.image_size and tuple(session.image_size) != tuple(image.image_size):
        raise ValueError(
            "The layers were marked on a photograph of another size "
            f"({session.image_size[0]} x {session.image_size[1]} px)."
        )
    shape = image.analysis_shape
    scale = image.analysis_scale()
    inside = frame_mask(frame, shape, scale)
    painted = rasterize_strokes(session.strokes, shape, scale)
    painted[~inside] = 0
    snap_px = snap_pixels(session.snap_mm, frame, scale, image.image_size)
    layer_snap_px = {
        row.layer_id: snap_pixels(row.snap_mm, frame, scale, image.image_size)
        for row in session.layers
        if row.snap_mm is not None
    }
    guide: PolygonGuide | None = None
    if session.polygons:
        _progress(progress, "Reading the polygons…")
        guide = polygon_guide(
            session.polygons,
            shape,
            scale,
            inside,
            snap_px,
            layer_snap_px=layer_snap_px,
            keep_shared_edges=not session.move_shared_edges,
        )
        # A stroke corrects the polygons under it.
        guide.training[painted > 0] = 0

    # Pixels whose layer is given: polygon interiors and the strips they leave
    # along the frame edge, then strokes on top.
    hard = np.zeros(shape, dtype=np.uint8)
    if guide is not None:
        hard[guide.fixed] = guide.owner[guide.fixed]
        # snap_pixels converts any length in mm to grid pixels.
        edge = frame_edge_fill(
            guide, inside, snap_pixels(session.edge_fill_mm, frame, scale, image.image_size)
        )
        hard[edge > 0] = edge[edge > 0]
    hard[painted > 0] = painted[painted > 0]
    todo = inside & (hard == 0)

    counts = np.bincount(painted.ravel(), minlength=256)
    if guide is not None:
        counts = counts + np.bincount(guide.training.ravel(), minlength=256)
    layer_ids = [int(value) for value in np.flatnonzero(counts) if value != 0]
    classes = hard.copy()
    confidence_values = np.ones(shape, dtype=np.float32)
    if np.any(todo):
        if len(layer_ids) < 2:
            raise ValueError(
                "Mark at least two layers (inside the rig frame) before proposing "
                "boundaries."
            )
        classes, confidence_values = _classify(
            image,
            session,
            painted=painted,
            guide=guide,
            hard=hard,
            todo=todo,
            inside=inside,
            layer_ids=layer_ids,
            snap_px=snap_px,
            n_estimators=n_estimators,
            max_samples_per_layer=max_samples_per_layer,
            random_seed=random_seed,
            progress=progress,
        )
    classes[~inside] = OUTSIDE
    classes[hard > 0] = hard[hard > 0]

    _progress(progress, "Merging small islands…")
    min_pixels = minimum_region_pixels(
        session.min_region_cm2, frame, scale, int(np.count_nonzero(inside))
    )
    classes, merged = merge_small_regions(classes, min_pixels, protected=hard)
    confidence = np.clip(np.rint(255.0 * confidence_values), 0, 255).astype(np.uint8)
    confidence[~inside] = 255
    confidence[hard > 0] = 255

    summary = _region_counts(classes)
    marked = session.marked_layers()
    polygon_counts = session.polygon_counts()
    layers = []
    for layer in session.layers:
        pixels, regions = summary.get(layer.layer_id, (0, 0))
        layers.append(
            {
                "layer_id": layer.layer_id,
                "name": layer.name,
                "pixels": pixels,
                "regions": regions,
                "painted": layer.layer_id in marked,
                "polygons": polygon_counts.get(layer.layer_id, 0),
            }
        )
    return LayerProposal(
        classes=classes,
        confidence=confidence,
        image_size=tuple(image.image_size),
        digest=proposal_digest(session, frame),
        created_utc=utc_now(),
        elapsed_s=round(time.monotonic() - started, 2),
        stroke_count=len(session.strokes),
        merged_regions=int(merged),
        min_region_pixels=int(min_pixels),
        frame=_frame_key(frame),
        layers=layers,
        polygon_count=len(session.polygons),
        snap_px=round(float(snap_px), 3),
    )


def _classify(
    image: PaintingImage,
    session: PaintingSession,
    *,
    painted: np.ndarray,
    guide: PolygonGuide | None,
    hard: np.ndarray,
    todo: np.ndarray,
    inside: np.ndarray,
    layer_ids: Sequence[int],
    snap_px: float,
    n_estimators: int,
    max_samples_per_layer: int,
    random_seed: int,
    progress: ProgressCallback | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Classes and confidence from the pixel classifier and the smoothing."""

    import cv2
    from sklearn.ensemble import ExtraTreesClassifier

    shape = painted.shape
    _progress(progress, "Computing image features…")
    features = image.features()
    rng = np.random.default_rng(int(random_seed))
    sample_index: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    training = guide.training if guide is not None else None
    for layer_id in layer_ids:
        index = _training_index(
            layer_id, painted, training, int(max_samples_per_layer), rng
        )
        sample_index.append(index)
        targets.append(np.full(index.size, layer_id, dtype=np.uint8))
    _progress(progress, "Training on the marked pixels…")
    model = ExtraTreesClassifier(
        n_estimators=max(8, int(n_estimators)),
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=int(random_seed),
    )
    model.fit(features[np.concatenate(sample_index)], np.concatenate(targets))
    classes_order = [int(value) for value in model.classes_]

    _progress(progress, "Labelling the photograph…")
    todo_index = np.flatnonzero(todo.ravel())
    probabilities = np.zeros((shape[0] * shape[1], len(classes_order)), np.float32)
    chunk = 200_000
    for start in range(0, todo_index.size, chunk):
        part = todo_index[start : start + chunk]
        probabilities[part] = model.predict_proba(features[part])
    flat_hard = hard.ravel()
    for position, layer_id in enumerate(classes_order):
        probabilities[flat_hard == layer_id, position] = 1.0
    probabilities = probabilities.reshape(shape[0], shape[1], len(classes_order))

    # Near polygon edges only layers marked nearby compete.
    disallowed: dict[int, np.ndarray] = {}
    if guide is not None and np.any(guide.restricted):
        for position, layer_id in enumerate(classes_order):
            near = guide.near(layer_id)
            allowed = near.copy() if near is not None else np.zeros(shape, bool)
            strokes = painted == layer_id
            if np.any(strokes):
                allowed |= _distance_to(strokes) <= guide.layer_snap(layer_id)
            blocked = guide.restricted & ~allowed
            if np.any(blocked):
                disallowed[position] = blocked
                probabilities[blocked, position] = 0.0

    sigma = SMOOTHING_SIGMA.get(session.smoothing, SMOOTHING_SIGMA["medium"])
    weight = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), sigma)
    np.maximum(weight, 1.0e-6, out=weight)
    for index in range(len(classes_order)):
        layer = np.ascontiguousarray(probabilities[..., index])
        probabilities[..., index] = cv2.GaussianBlur(layer, (0, 0), sigma) / weight
    for position, blocked in disallowed.items():
        probabilities[blocked, position] = 0.0
    total = probabilities.sum(axis=2)
    best = np.argmax(probabilities, axis=2)
    confidence = np.take_along_axis(probabilities, best[..., None], axis=2)[..., 0]
    confidence = confidence / np.maximum(total, 1.0e-6)
    classes = np.asarray(classes_order, dtype=np.uint8)[best]
    empty = inside & (total <= 1.0e-6)
    if guide is not None and np.any(empty):
        # No allowed layer has any support here: take the nearest polygon.
        polygon_layers = sorted(guide.distance)
        nearest = np.argmin(
            np.stack([guide.distance[layer] for layer in polygon_layers]), axis=0
        )
        fallback = np.asarray(polygon_layers, dtype=np.uint8)[nearest]
        classes[empty] = fallback[empty]
        confidence[empty] = 0.0
    return classes, confidence


def load_proposal(project: FluidFlowerProject, run_id: str) -> LayerProposal | None:
    path = proposal_path(project, run_id)
    if not _preset_io.exists(path, 'is_file'):
        return None
    try:
        return LayerProposal.load(path)
    except (OSError, ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Hand corrections on top of a proposal

LAYER_CORRECTIONS_FORMAT = "darsia-fluidflower-layer-corrections-v1"
CORRECTION_KINDS = ("brush", "restore", "lasso", "fill")
_CORRECTION_POINTS = {"brush": 1, "restore": 1, "lasso": 3, "fill": 1}


@dataclass
class Correction:
    """A hand correction drawn on top of a proposal; points in raw pixels (row, col).

    ``brush`` gives the pixels along ``points`` (radius ``radius_px``) the
    layer; ``restore`` brushes the proposal back there, which undoes earlier
    corrections; ``lasso`` gives the inside of the closed outline ``points``
    the layer; ``fill`` gives the connected region at ``points[0]`` the layer
    while that region still has ``replaced_layer``. ``proposal_digest`` names
    the proposal the correction was drawn on. Coordinates are kept to 0.1 px
    and the radius to 0.01 px, as saved.
    """

    kind: str
    layer_id: int
    points: list[tuple[float, float]]
    radius_px: float = 0.0
    replaced_layer: int | None = None
    proposal_digest: str = ""
    created_utc: str = ""

    def __post_init__(self) -> None:
        if self.kind not in CORRECTION_KINDS:
            raise ValueError(f"Unknown correction {self.kind!r}")
        self.points = [
            (round(float(row), 1), round(float(col), 1)) for row, col in self.points
        ]
        if len(self.points) < _CORRECTION_POINTS[self.kind]:
            raise ValueError(
                "A lasso needs an outline of at least three points"
                if self.kind == "lasso"
                else "A correction needs a point"
            )
        if self.kind == "restore":
            self.layer_id = ERASER
        elif not 0 < int(self.layer_id) < 256:
            raise ValueError("A correction needs a layer")
        self.layer_id = int(self.layer_id)
        self.radius_px = round(float(self.radius_px), 2)
        if self.kind in ("brush", "restore") and self.radius_px <= 0.0:
            raise ValueError("A brush correction needs a positive radius")
        if self.kind == "fill":
            if self.replaced_layer is None:
                raise ValueError("A fill names the layer it replaces")
            self.replaced_layer = int(self.replaced_layer)

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {"kind": self.kind, "layer_id": self.layer_id}
        if self.kind in ("brush", "restore"):
            row["radius_px"] = self.radius_px
        if self.kind == "fill":
            row["replaced_layer"] = self.replaced_layer
        row["points"] = [[row_px, col_px] for row_px, col_px in self.points]
        row["proposal_digest"] = self.proposal_digest
        row["created_utc"] = self.created_utc
        return row

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Correction:
        replaced = payload.get("replaced_layer")
        return cls(
            kind=str(payload.get("kind") or ""),
            layer_id=int(payload.get("layer_id") or 0),
            points=[
                (float(row), float(col)) for row, col in payload.get("points") or []
            ],
            radius_px=float(payload.get("radius_px") or 0.0),
            replaced_layer=None if replaced is None else int(replaced),
            proposal_digest=str(payload.get("proposal_digest") or ""),
            created_utc=str(payload.get("created_utc") or ""),
        )


def corrections_path(project: FluidFlowerProject, run_id: str) -> Path:
    """The run's hand-correction file, next to its session and proposal."""

    return painting_folder(project) / f"{_file_stem(run_id)}.corrections.json"


def load_corrections(project: FluidFlowerProject, run_id: str) -> list[Correction]:
    """The run's hand corrections in drawing order; none without a file."""

    path = corrections_path(project, run_id)
    if not _preset_io.exists(path, 'is_file'):
        return []
    payload = json.loads(_preset_io.read_text(path,encoding="utf-8-sig"))
    if payload.get("format") != LAYER_CORRECTIONS_FORMAT:
        raise ValueError("Unsupported layer correction file")
    return [Correction.from_dict(row) for row in payload.get("corrections") or []]


def save_corrections(
    project: FluidFlowerProject, run_id: str, corrections: Sequence[Correction]
) -> Path | None:
    """Write the run's hand corrections, one per line; returns the file.

    Without corrections and without an earlier file nothing is written (None),
    so a proposal that is never corrected gains no file.
    """

    path = corrections_path(project, run_id)
    if not corrections and not _preset_io.exists(path, 'exists'):
        return None
    _atomic_write_text(path, _corrections_text(run_id, corrections))
    return path


def _corrections_text(run_id: str, corrections: Sequence[Correction]) -> str:
    """The correction file's text: a short header and one correction per line."""

    header = {
        "schema": 1,
        "format": LAYER_CORRECTIONS_FORMAT,
        "run_id": str(run_id),
        "updated_utc": utc_now(),
    }
    text = json.dumps(header, indent=2, ensure_ascii=False)
    text = text[: text.rfind("}")].rstrip()
    lines = ",\n".join(
        "    " + json.dumps(row.to_dict(), separators=(",", ":"), ensure_ascii=False)
        for row in corrections
    )
    body = "\n" + lines + "\n  " if corrections else ""
    text += ',\n  "corrections": [' + body + "]"
    text += "\n}\n"
    return text


def corrections_digest(corrections: Sequence[Correction]) -> str:
    """SHA-256 of the corrections as saved ("" for none)."""

    if not corrections:
        return ""
    rows = [row.to_dict() for row in corrections]
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _grid_seed(
    point: tuple[float, float], shape: tuple[int, int], scale: tuple[float, float]
) -> tuple[int, int] | None:
    row = int(round((point[0] + 0.5) * scale[0] - 0.5))
    col = int(round((point[1] + 0.5) * scale[1] - 0.5))
    if 0 <= row < shape[0] and 0 <= col < shape[1]:
        return row, col
    return None


def layer_at(
    classes: np.ndarray, point: tuple[float, float], scale: tuple[float, float]
) -> int | None:
    """The class under a raw point (row, col) on a grid, or None off the grid."""

    seed = _grid_seed(point, classes.shape, scale)
    return None if seed is None else int(classes[seed])


def _correction_mask(
    correction: Correction, shape: tuple[int, int], scale: tuple[float, float]
) -> tuple[tuple[slice, slice], np.ndarray] | None:
    """Where a brush or lasso correction lies on a grid: its bounds and mask.

    Only the bounding box is drawn, so that a correction costs its own size,
    not the grid's. None when it misses the grid.
    """

    rows = [(row + 0.5) * scale[0] - 0.5 for row, _col in correction.points]
    cols = [(col + 0.5) * scale[1] - 0.5 for _row, col in correction.points]
    pad = 2.0
    if correction.kind != "lasso":
        pad += float(correction.radius_px) * 0.5 * (scale[0] + scale[1])
    top = max(0, int(math.floor(min(rows) - pad)))
    left = max(0, int(math.floor(min(cols) - pad)))
    bottom = min(shape[0], int(math.ceil(max(rows) + pad)) + 1)
    right = min(shape[1], int(math.ceil(max(cols) + pad)) + 1)
    if top >= bottom or left >= right:
        return None
    # Raw points moved so that the box starts at the grid's origin.
    offset_row, offset_col = top / scale[0], left / scale[1]
    points = [(row - offset_row, col - offset_col) for row, col in correction.points]
    box = (bottom - top, right - left)
    if correction.kind == "lasso":
        mask = np.zeros(box, dtype=np.uint8)
        _fill_polygon(mask, Polygon(1, points), scale, 1)
    else:
        mask = rasterize_strokes([Stroke(1, correction.radius_px, points)], box, scale)
    return (slice(top, bottom), slice(left, right)), mask > 0


def apply_corrections(
    classes: np.ndarray,
    corrections: Sequence[Correction],
    scale: tuple[float, float],
    *,
    inside: np.ndarray | None = None,
    layer_ids: Sequence[int] | set[int] | None = None,
    start: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """The classes with the corrections applied in drawing order.

    ``classes`` (left unchanged) is the proposal on a grid with ``scale`` grid
    pixels per raw pixel along rows and columns. Corrections act only inside
    ``inside``, the rig frame. Returns the corrected classes, the mask of the
    pixels a correction decided, and the indices of the corrections that do
    not apply: a fill whose region no longer has the layer it replaced, or a
    correction whose layer is not in ``layer_ids``. ``start`` continues from
    the corrected classes and decided mask of earlier corrections, as
    returned, so that corrections can be added one at a time; a restore
    stroke still brings back ``classes``.
    """

    from scipy import ndimage as ndi

    if start is None:
        result = classes.copy()
        decided = np.zeros(classes.shape, dtype=bool)
    else:
        result, decided = start[0].copy(), start[1].copy()
    skipped: list[int] = []
    structure = ndi.generate_binary_structure(2, 1)
    known = None if layer_ids is None else {int(value) for value in layer_ids}
    for index, correction in enumerate(corrections):
        if (
            correction.kind != "restore"
            and known is not None
            and correction.layer_id not in known
        ):
            skipped.append(index)
            continue
        if correction.kind == "fill":
            seed = _grid_seed(correction.points[0], classes.shape, scale)
            if (
                seed is None
                or (inside is not None and not inside[seed])
                or int(result[seed]) != int(correction.replaced_layer)
            ):
                skipped.append(index)
                continue
            components, _count = ndi.label(result == result[seed], structure=structure)
            box = (slice(None), slice(None))
            region = components == components[seed]
        else:
            found = _correction_mask(correction, classes.shape, scale)
            if found is None:
                continue
            box, region = found
            if inside is not None:
                region &= inside[box]
        if correction.kind == "restore":
            result[box][region] = classes[box][region]
            decided[box][region] = False
        else:
            result[box][region] = correction.layer_id
            decided[box][region] = True
    return result, decided, skipped


def proposal_grid_scale(proposal: LayerProposal) -> tuple[float, float]:
    """Proposal grid pixels per raw pixel along rows and columns."""

    height, width = proposal.classes.shape
    return height / proposal.image_size[1], width / proposal.image_size[0]


def corrected_classes(
    proposal: LayerProposal,
    corrections: Sequence[Correction],
    *,
    layer_ids: Sequence[int] | set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """The proposal's classes with the corrections, on the proposal's grid."""

    return apply_corrections(
        proposal.classes,
        corrections,
        proposal_grid_scale(proposal),
        inside=proposal.classes != OUTSIDE,
        layer_ids=layer_ids,
    )


def corrections_not_applied(
    proposal: LayerProposal | None,
    corrections: Sequence[Correction],
    *,
    layer_ids: Sequence[int] | set[int] | None = None,
) -> list[int]:
    """Indices of the corrections that do not apply to ``proposal``."""

    if proposal is None or not corrections:
        return []
    return corrected_classes(proposal, corrections, layer_ids=layer_ids)[2]


# ---------------------------------------------------------------------------
# Saved versions of the map

LAYER_VERSION_FORMAT = "darsia-fluidflower-layer-painting-version-v1"
VERSION_FILES = ("session.json", "proposal.npz", "corrections.json")
LAYER_APPROVAL_FORMAT = "darsia-fluidflower-layer-map-approval-v1"


def versions_folder(project: FluidFlowerProject, run_id: str) -> Path:
    """The folder of a run's saved versions, one subfolder per version."""

    return painting_folder(project) / "versions" / _file_stem(run_id)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _preset_io.open_read(Path(path), "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def map_digest(
    session: PaintingSession,
    proposal: LayerProposal,
    corrections: Sequence[Correction],
) -> str:
    """SHA-256 of what the painted map depends on.

    The session without its save time and author, the proposal's cells,
    confidence, size and input digest, and the corrections. The same map
    gives the same digest whenever it is saved.
    """

    payload = session.to_dict()
    payload.pop("updated_utc", None)
    payload.pop("updated_by", None)
    digest = hashlib.sha256()
    settings = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest.update(settings.encode("utf-8"))
    shape = [list(proposal.image_size), list(proposal.classes.shape), proposal.digest]
    digest.update(json.dumps(shape).encode("utf-8"))
    digest.update(np.ascontiguousarray(proposal.classes, dtype=np.uint8).tobytes())
    digest.update(np.ascontiguousarray(proposal.confidence, dtype=np.uint8).tobytes())
    digest.update(corrections_digest(corrections).encode("ascii"))
    return digest.hexdigest()


def save_version(
    project: FluidFlowerProject,
    session: PaintingSession,
    proposal: LayerProposal,
    corrections: Sequence[Correction],
    *,
    name: str,
) -> dict[str, Any]:
    """Save the map as a new version and return its record.

    The session, proposal and corrections go into a new folder that is never
    changed afterwards, so no earlier version is overwritten. The record holds
    the name, the kind (``proposal`` or ``edited map``), the time, ``sha256``
    (``map_digest`` of the content) and the SHA-256 of each file.
    """

    name = str(name).strip()
    if not name:
        raise ValueError("A version needs a name")
    digest = map_digest(session, proposal, corrections)
    root = versions_folder(project, session.run_id)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    folder = root / f"{stamp}-{digest[:12]}"
    count = 1
    while True:
        try:
            folder.mkdir()
            break
        except FileExistsError:
            count += 1
            folder = root / f"{stamp}-{digest[:12]}-{count}"
    (folder / "session.json").write_text(_session_text(session), encoding="utf-8")
    proposal.save(folder / "proposal.npz")
    (folder / "corrections.json").write_text(
        _corrections_text(session.run_id, corrections), encoding="utf-8"
    )
    record = {
        "schema": 1,
        "format": LAYER_VERSION_FORMAT,
        "run_id": session.run_id,
        "name": name,
        "kind": "edited map" if corrections else "proposal",
        "sha256": digest,
        "created_utc": utc_now(),
        "proposal_digest": proposal.digest,
        "correction_count": len(corrections),
        "files": {file: _file_sha256(folder / file) for file in VERSION_FILES},
    }
    _atomic_write_text(
        folder / "version.json", json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    )
    return dict(record, folder=str(folder))


def list_versions(project: FluidFlowerProject, run_id: str) -> list[dict[str, Any]]:
    """The run's saved versions, oldest first; folders without a readable
    record are left out."""

    root = versions_folder(project, run_id)
    if not _preset_io.exists(root, 'is_dir'):
        return []
    rows = []
    for folder in sorted(root.iterdir()):
        try:
            record = json.loads(_preset_io.read_text(folder / "version.json",encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if record.get("format") == LAYER_VERSION_FORMAT:
            rows.append(dict(record, folder=str(folder)))
    return sorted(
        rows, key=lambda row: (str(row.get("created_utc") or ""), row["folder"])
    )


def load_version(
    record: Mapping[str, Any],
) -> tuple[PaintingSession, LayerProposal, list[Correction]]:
    """The session, proposal and corrections of a saved version.

    Each file is checked against the SHA-256 in its record first.
    """

    from .storage_paths import resolve_storage_path
    folder = resolve_storage_path(str(record["folder"]))
    for file, expected in dict(record.get("files") or {}).items():
        if _file_sha256(folder / file) != expected:
            raise ValueError(
                f"The saved version's {file} has changed since it was saved"
            )
    session = PaintingSession.from_dict(
        json.loads(_preset_io.read_text(folder / "session.json",encoding="utf-8"))
    )
    proposal = LayerProposal.load(folder / "proposal.npz")
    payload = json.loads(_preset_io.read_text(folder / "corrections.json",encoding="utf-8"))
    rows = payload.get("corrections") or []
    corrections = [Correction.from_dict(row) for row in rows]
    return session, proposal, corrections


def restore_version(
    project: FluidFlowerProject, record: Mapping[str, Any], *, actor: str = ""
) -> tuple[PaintingSession, LayerProposal, list[Correction]]:
    """Make a saved version the run's current session, proposal and corrections.

    The version itself is not changed. The caller saves the current state as a
    version first when it should be kept.
    """

    session, proposal, corrections = load_version(record)
    save_painting_session(project, session, actor=actor)
    proposal.save(proposal_path(project, session.run_id))
    path = corrections_path(project, session.run_id)
    if corrections or _preset_io.exists(path, 'exists'):
        save_corrections(project, session.run_id, corrections)
    return session, proposal, corrections


def approvals_path(project: FluidFlowerProject, run_id: str) -> Path:
    """The run's map approvals, one record per line, beside its saved versions."""

    return versions_folder(project, run_id) / "approvals.jsonl"


def read_map_approvals(
    project: FluidFlowerProject, run_id: str
) -> list[dict[str, Any]]:
    """Every approval of the run's saved map versions, oldest first.

    A malformed line raises ValueError, so that a damaged record is never
    read as "not approved".
    """

    path = approvals_path(project, run_id)
    if not _preset_io.exists(path, 'is_file'):
        return []
    records = []
    for number, line in enumerate(_preset_io.read_text(path,encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or record.get("format") != LAYER_APPROVAL_FORMAT:
            raise ValueError(f"Line {number} of {path} is not a map approval record")
        records.append(record)
    return records


def map_approval(
    approvals: Sequence[Mapping[str, Any]], sha256: str, *, version: str | None = None
) -> dict[str, Any] | None:
    """The latest active approval, with optional literal saved-version scope.

    A withdrawal cancels only the exact approval named by its append-only
    event. Another same-content version or a subsequent approval is unchanged.
    Histories without withdrawal events retain their original latest result.
    """
    active = {}
    for position, row in enumerate(approvals):
        if version is not None and row.get("version") != version:
            continue
        key = (row.get("run_id"), row.get("version"))
        if row.get("event") == "approval_withdrawn":
            prior = active.get(key)
            if (prior is not None and row.get("version_sha256") == sha256
                    and row.get("withdrawn_approval") == prior[1]):
                del active[key]
        elif not row.get("event") and sha256 and row.get("sha256") == sha256:
            active[key] = (position, row)
    return dict(max(active.values(), key=lambda item: item[0])[1]) if active else None


def map_approval_withdrawal(approvals, sha256, *, version):
    """Latest retained withdrawal of this exact saved version for GUI display."""
    rows = [row for row in approvals if row.get("event") == "approval_withdrawn"
            and row.get("version_sha256") == sha256 and row.get("version") == version]
    return dict(rows[-1]) if rows else None


def approval_withdrawal(approvals, approval):
    """Whether a saved placement's particular original approval was withdrawn."""
    if not isinstance(approval, Mapping):
        return None
    rows = [row for row in approvals if row.get("event") == "approval_withdrawn"
            and row.get("withdrawn_approval") == approval]
    return dict(rows[-1]) if rows else None


def approve_version(
    record: Mapping[str, Any],
    *,
    reviewer: str | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Approve a saved version as its run's map: append a record and return it.

    The version's files are checked against their SHA-256 first, as for a
    restore. The record names the version (its folder, name and kind), the
    map's SHA-256 (``map_digest``), the SHA-256 of each file, ``reviewer``
    (the series' reviewer setting; None when there is none) and the UTC
    time. It goes to ``approvals.jsonl`` beside the run's versions, and
    earlier records stay as they are. Approving changes no map, session,
    proposal, version or geometry revision. A transient file error is
    retried (N111).
    """

    from darsia.fluidflower.pipeline.job_queue import retry_transient

    load_version(record)  # A changed file raises ValueError.
    from .storage_paths import resolve_storage_path
    folder = resolve_storage_path(str(record["folder"]))
    approval = {
        "format": LAYER_APPROVAL_FORMAT,
        "run_id": str(record.get("run_id") or ""),
        "version": folder.name,
        "name": str(record.get("name") or ""),
        "kind": str(record.get("kind") or ""),
        "sha256": str(record.get("sha256") or ""),
        "files": dict(record.get("files") or {}),
        "reviewer": (str(reviewer).strip() or None) if reviewer else None,
        "approved_utc": now or utc_now(),
    }
    line = json.dumps(approval, ensure_ascii=False, sort_keys=True) + "\n"
    path = folder.parent / "approvals.jsonl"

    def append() -> None:
        with path.open("ab") as stream:
            stream.write(line.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())

    retry_transient(append)
    return approval


# ---------------------------------------------------------------------------
# Revision and export


def upsample_classes(
    classes: np.ndarray,
    size: tuple[int, int],
    values: Sequence[int],
    *,
    sigma: float = 0.6,
    strip_rows: int = 512,
) -> np.ndarray:
    """Classes on a finer grid with smooth boundaries.

    Each class indicator is slightly blurred and bilinearly interpolated; every
    pixel takes the class with the largest value. ``size`` is (width, height).
    """

    import cv2

    width, height = int(size[0]), int(size[1])
    grid_height, grid_width = classes.shape
    indicators = [
        cv2.GaussianBlur((classes == value).astype(np.float32), (0, 0), sigma)
        for value in values
    ]
    map_x = (np.arange(width, dtype=np.float32) + 0.5) * (grid_width / width) - 0.5
    output = np.zeros((height, width), dtype=np.uint8)
    for start in range(0, height, strip_rows):
        stop = min(height, start + strip_rows)
        rows = (np.arange(start, stop, dtype=np.float32) + 0.5) * (
            grid_height / height
        ) - 0.5
        map_xs = np.ascontiguousarray(np.broadcast_to(map_x, (stop - start, width)))
        map_ys = np.ascontiguousarray(
            np.broadcast_to(rows[:, None], (stop - start, width))
        )
        best = np.full((stop - start, width), -1.0, dtype=np.float32)
        strip = np.zeros((stop - start, width), dtype=np.uint8)
        for value, indicator in zip(values, indicators):
            sampled = cv2.remap(
                indicator,
                map_xs,
                map_ys,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            better = sampled > best
            best[better] = sampled[better]
            strip[better] = int(value)
        output[start:stop] = strip
    return output


def facies_ids(session: PaintingSession, present: Sequence[int]) -> dict[str, int]:
    """Facies numbers 1, 2, ... in the facies order, for facies with a layer in
    ``present`` (empty facies get no number)."""

    used = {layer.facies for layer in session.layers if layer.layer_id in present}
    order: dict[str, int] = {}
    for name in session.facies:
        if name in used:
            order[name] = len(order) + 1
    for layer in session.layers:
        if layer.layer_id in present and layer.facies not in order:
            order[layer.facies] = len(order) + 1
    return order


def check_ready_for_revision(
    session: PaintingSession,
    proposal: LayerProposal | None,
    frame: RigFrame | None,
    reference: SeriesRunInputs | None,
    corrections: Sequence[Correction] = (),
) -> list[str]:
    """Reasons why the proposal cannot be saved yet (empty when it can).

    With hand corrections the layers present and the region count are those
    of the corrected map.
    """

    problems: list[str] = []
    if reference is None or reference.baseline_image is None:
        problems.append("The reference run has no baseline photograph.")
    if proposal is None:
        problems.append("Propose the boundaries before saving a version.")
        return problems
    if proposal.digest != proposal_digest(session, frame):
        problems.append(
            "The polygons, strokes, settings or rig frame changed after the last "
            "proposal; propose the boundaries again."
        )
    size = _rig_size(frame, reference)
    if size is None:
        problems.append(
            "Enter the rig's inner width and height in Rig Geometry; the revision needs "
            "the rig size."
        )
    present = {int(row["layer_id"]) for row in proposal.layers if row.get("pixels")}
    classes = None
    if corrections:
        classes = corrected_classes(
            proposal, corrections, layer_ids={row.layer_id for row in session.layers}
        )[0]
        present = {
            int(value)
            for value in np.flatnonzero(np.bincount(classes.ravel(), minlength=256))
            if value != OUTSIDE
        }
    missing: list[str] = []
    for name in facies_ids(session, sorted(present)):
        entry = session.facies.get(name, FaciesEntry())
        if entry.role != "porous_medium":
            continue
        porosity_ok = entry.porosity is not None and 0.0 < entry.porosity <= 1.0
        permeability_ok = entry.permeability is not None and entry.permeability > 0
        if not (porosity_ok and permeability_ok):
            missing.append(name)
    if missing:
        problems.append(
            "Enter porosity (0-1) and a positive permeability for "
            + ", ".join(missing)
            + " in Facies."
        )
    if classes is None:
        regions = proposal_region_count(proposal)
    else:
        regions = class_region_count(classes)
    if regions > NATIVE_MAX_REGIONS:
        problems.append(
            f"The proposal has {regions} regions; native Setup supports at most "
            f"{NATIVE_MAX_REGIONS}. Raise 'Merge regions under' or paint over "
            "small islands, and propose again."
        )
    return problems


def proposal_region_count(proposal: LayerProposal) -> int:
    """Connected regions of a proposal, the area outside the frame included."""

    count = sum(int(row.get("regions") or 0) for row in proposal.layers)
    if np.any(proposal.classes == OUTSIDE):
        count += 1
    return count


def class_region_count(classes: np.ndarray) -> int:
    """Connected regions of a class map, counted as for a proposal."""

    counts = _region_counts(classes)
    count = sum(
        regions for value, (_pixels, regions) in counts.items() if value != OUTSIDE
    )
    if OUTSIDE in counts:
        count += 1
    return count


def _rig_size(
    frame: RigFrame | None, reference: SeriesRunInputs | None
) -> tuple[float, float] | None:
    if frame is not None and frame.has_size():
        return float(frame.width_m), float(frame.height_m)
    if reference is not None and reference.width_m and reference.height_m:
        return float(reference.width_m), float(reference.height_m)
    return None


def _kept_shared_edges(
    session: PaintingSession,
    proposal: LayerProposal,
    frame: RigFrame | None,
    shape: tuple[int, int],
) -> np.ndarray | None:
    """Pixels of the raw grid (``shape``) near the shared edges a proposal
    kept exactly, or None.

    These are the proposal grid cells within two cells of a shared edge that
    the proposal fixed to their polygon's layer (``move_shared_edges`` off).
    """

    if session.move_shared_edges or not session.polygons:
        return None
    shared = shared_edges(session.polygons)
    if not shared:
        return None
    grid = proposal.classes.shape
    scale = proposal_grid_scale(proposal)
    size = tuple(proposal.image_size)
    guide = polygon_guide(
        session.polygons,
        grid,
        scale,
        frame_mask(frame, grid, scale),
        snap_pixels(session.snap_mm, frame, scale, size),
        layer_snap_px={
            row.layer_id: snap_pixels(row.snap_mm, frame, scale, size)
            for row in session.layers
            if row.snap_mm is not None
        },
        keep_shared_edges=True,
    )
    near = _segments_mask(shared, grid, scale, 5) & guide.fixed
    if not np.any(near):
        return None
    height, width = shape
    rows = np.minimum(
        ((np.arange(height) + 0.5) * (grid[0] / height)).astype(np.intp), grid[0] - 1
    )
    cols = np.minimum(
        ((np.arange(width) + 0.5) * (grid[1] / width)).astype(np.intp), grid[1] - 1
    )
    return near[rows[:, None], cols[None, :]]


def painted_label_map(
    session: PaintingSession,
    proposal: LayerProposal,
    *,
    frame: RigFrame | None,
    progress: ProgressCallback | None = None,
    corrections: Sequence[Correction] = (),
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Label map on the raw photograph's grid and one region row per label.

    Polygons used exactly (snap 0) and shared edges kept exactly
    (``move_shared_edges`` off) are drawn at full resolution. Hand
    corrections are applied at full resolution after the proposal and its
    exact polygons; the pixels they decide are never merged away.
    """

    from scipy import ndimage as ndi

    width, height = proposal.image_size
    present = [
        int(value)
        for value in np.flatnonzero(
            np.bincount(proposal.classes.ravel(), minlength=256)
        )
        if value != OUTSIDE
    ]
    _progress(progress, "Scaling the layers to the photograph's full resolution…")
    classes = upsample_classes(proposal.classes, (width, height), present)
    scale = (1.0, 1.0)
    inside = frame_mask(frame, (height, width), scale)
    classes[~inside] = OUTSIDE
    exact = sorted(
        {row.layer_id for row in session.polygons if session.layer_snap_mm(row.layer_id) <= 0.0}
    )
    kept = _kept_shared_edges(session, proposal, frame, (height, width))
    if exact or kept is not None:
        # Polygons used exactly (snap 0, common or the layer's own) keep their
        # outline at full resolution, and so do shared edges kept exactly;
        # strokes inside them still correct them.
        _progress(progress, "Drawing the polygons at full resolution…")
        owner = rasterize_polygons(session.polygons, (height, width), scale)
        owner[~inside] = 0
        covered = np.isin(owner, exact)
        if kept is not None:
            covered |= kept & (owner > 0)
            del kept
        classes[covered] = owner[covered]
        del owner
        painted = rasterize_strokes(session.strokes, (height, width), scale)
        restore = covered & (painted > 0)
        classes[restore] = painted[restore]
        del painted, restore, covered
        present = [
            int(value)
            for value in np.flatnonzero(np.bincount(classes.ravel(), minlength=256))
            if value != OUTSIDE
        ]
    protected = None
    if corrections:
        _progress(progress, "Applying the hand corrections…")
        classes, decided, _skipped = apply_corrections(
            classes,
            corrections,
            scale,
            inside=inside,
            layer_ids={row.layer_id for row in session.layers},
        )
        protected = np.where(decided, classes, 0).astype(np.uint8)
        del decided
        present = [
            int(value)
            for value in np.flatnonzero(np.bincount(classes.ravel(), minlength=256))
            if value != OUTSIDE
        ]
    grid_ratio = (width / proposal.classes.shape[1]) * (
        height / proposal.classes.shape[0]
    )
    # Islands were merged on the analysis grid, where painted regions are
    # protected; here only slivers of the upsampling (two grid cells) go.
    min_pixels = int(math.ceil(2.0 * grid_ratio))
    _progress(progress, "Cleaning up small regions…")
    classes, _merged = merge_small_regions(classes, min_pixels, protected=protected)
    del inside, protected

    _progress(progress, "Numbering the regions…")
    structure = ndi.generate_binary_structure(2, 1)
    labels = np.zeros((height, width), dtype=np.uint16)
    order = facies_ids(session, present)
    regions: list[dict[str, Any]] = []
    next_label = 1
    for layer in session.layers:
        if layer.layer_id not in present:
            continue
        components = np.zeros((height, width), dtype=np.int32)
        count = ndi.label(
            classes == layer.layer_id, structure=structure, output=components
        )
        if not count:
            continue
        if next_label + count > np.iinfo(np.uint16).max:
            raise ValueError("Too many regions; merge small islands first.")
        sizes = np.bincount(components.ravel(), minlength=count + 1)
        entry = session.facies.get(layer.facies, FaciesEntry())
        porous = entry.role == "porous_medium"
        table = np.zeros(count + 1, dtype=np.uint16)
        for rank, component in enumerate(
            sorted(range(1, count + 1), key=lambda value: -int(sizes[value])), start=1
        ):
            table[component] = next_label
            regions.append(
                {
                    "label_id": next_label,
                    "facies_id": order[layer.facies],
                    "name": layer.facies,
                    "layer": layer.name,
                    "layer_id": layer.layer_id,
                    "region_index": rank,
                    "pixels": int(sizes[component]),
                    "analysis_role": entry.role,
                    "porosity": entry.effective_porosity(),
                    "permeability": entry.permeability if porous else None,
                    "enabled": False if entry.role == "colorchecker" else entry.role != "excluded",
                }
            )
            next_label += 1
        labels += table[components]
        del components
    outside = classes == OUTSIDE
    if np.any(outside):
        components = np.zeros((height, width), dtype=np.int32)
        count = ndi.label(outside, structure=structure, output=components)
        if next_label + count > np.iinfo(np.uint16).max:
            raise ValueError("Too many regions; merge small islands first.")
        sizes = np.bincount(components.ravel(), minlength=count + 1)
        facies_id = len(order) + 1
        table = np.zeros(count + 1, dtype=np.uint16)
        for rank, component in enumerate(range(1, count + 1), start=1):
            table[component] = next_label
            regions.append(
                {
                    "label_id": next_label,
                    "facies_id": facies_id,
                    "name": OUTSIDE_FACIES,
                    "layer": OUTSIDE_FACIES,
                    "layer_id": OUTSIDE,
                    "region_index": rank,
                    "pixels": int(sizes[component]),
                    "analysis_role": "frame",
                    "porosity": OUTSIDE_POROSITY,
                    "permeability": None,
                    "enabled": False,
                }
            )
            next_label += 1
        labels += table[components]
        del components
    return labels, regions


def revision_injectors(
    project: FluidFlowerProject,
    frame: RigFrame | None,
    size: tuple[float, float],
) -> tuple[list[dict[str, Any]], list[str]]:
    """The series' injection points as revision injectors.

    Points without rig coordinates or outside the rig are left out; their IDs
    are returned as the second value.
    """

    injectors: list[dict[str, Any]] = []
    skipped: list[str] = []
    width_m, height_m = size
    for point in _resolved_points(saved_injection_points(project), frame):
        identifier = str(point.get("location_id") or "").strip()
        x_m, y_m = _number(point.get("x_m")), _number(point.get("y_m"))
        if (
            not identifier
            or x_m is None
            or y_m is None
            or not (0.0 <= x_m <= width_m and 0.0 <= y_m <= height_m)
        ):
            skipped.append(identifier or "unnamed point")
            continue
        injectors.append(
            {
                "injector_id": identifier,
                "protocol_location_id": identifier,
                "name": str(point.get("name") or "").strip() or identifier,
                "x_m": x_m,
                "y_m": y_m,
                "uncertainty_m": None,
                "coordinate_source": (
                    "geometry workspace: " + str(point.get("source") or "typed")
                ),
            }
        )
    return injectors, skipped


def save_painted_revision(
    project: FluidFlowerProject,
    session: PaintingSession,
    proposal: LayerProposal,
    reference: SeriesRunInputs,
    *,
    frame: RigFrame | None,
    actor: str = "",
    progress: ProgressCallback | None = None,
    corrections: Sequence[Correction] = (),
) -> dict[str, Any]:
    """Store the proposal, with any hand corrections, as a geometry revision."""

    from darsia.fluidflower.geometry_facies import (
        GeometryRevisionStore,
        _save_darsia_field,
        default_geometry_config,
        validate_geometry_config,
    )

    problems = check_ready_for_revision(
        session, proposal, frame, reference, corrections
    )
    if problems:
        raise ValueError(" ".join(problems))
    assert reference.baseline_image is not None
    width_px, height_px = photo_size(reference.baseline_image,
                                     camera_u=reference_camera_u(project, reference))
    if (width_px, height_px) != tuple(proposal.image_size):
        raise ValueError(
            f"The baseline photograph of {reference.run_id} has {width_px} x "
            f"{height_px} px; the layers were painted on a "
            f"{proposal.image_size[0]} x {proposal.image_size[1]} px photograph."
        )
    size = _rig_size(frame, reference)
    assert size is not None
    width_m, height_m = size

    labels, regions = painted_label_map(
        session, proposal, frame=frame, progress=progress, corrections=corrections
    )
    if len(regions) > NATIVE_MAX_REGIONS:
        raise ValueError(
            f"The layers form {len(regions)} regions; native Setup supports at most "
            f"{NATIVE_MAX_REGIONS}. Raise 'Merge regions under' or paint over small "
            "islands, and propose again."
        )
    injectors, skipped = revision_injectors(project, frame, (width_m, height_m))
    _progress(progress, "Writing the label map…")
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    destination = (
        Path(project.root)
        / "overrides"
        / "geometry_facies"
        / "painted"
        / f"{stamp}-{uuid.uuid4().hex[:8]}"
    )
    destination.mkdir(parents=True, exist_ok=False)
    labels_path = destination / "labels.npz"
    depth_path = destination / "depth.npz"
    _save_darsia_field(labels_path, labels, width_m=width_m, height_m=height_m)
    shape = labels.shape
    del labels
    _save_darsia_field(
        depth_path,
        np.full(shape, PLACEHOLDER_DEPTH_M, dtype=np.float32),
        width_m=width_m,
        height_m=height_m,
    )
    session_file = session_path(project, session.run_id)
    (destination / "regions.json").write_text(
        json.dumps(regions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    config = default_geometry_config(project)
    config.update(
        {
            "baseline_path": str(reference.baseline_image),
            "depth_path": str(depth_path),
            "labels_path": str(labels_path),
            "display_image_path": str(reference.baseline_image),
            "draft_source_image_path": str(reference.baseline_image),
            "width_m": float(width_m),
            "height_m": float(height_m),
            "grouping_mode": "shared_facies",
            "permeability_unit": session.permeability_unit,
            "regions": regions,
            "injectors": injectors,
            "depth_note": (
                "Constant placeholder; native Setup takes the depth from its own "
                "[depth] section."
            ),
            "facies_generation": {
                "method": PROPOSAL_METHOD,
                "status": "painted layers reviewed in the layer view",
                "reference_run": reference.run_id,
                "source_image_path": str(reference.baseline_image),
                "session_path": str(session_file),
                "proposal_digest": proposal.digest,
                "proposal_created_utc": proposal.created_utc,
                "stroke_count": len(session.strokes),
                "polygon_count": len(session.polygons),
                "snap_mm": session.snap_mm if session.polygons else None,
                **(
                    {}
                    if session.move_shared_edges
                    else {"move_shared_edges": False}
                ),
                "layers": [
                    {
                        "layer_id": layer.layer_id,
                        "name": layer.name,
                        "facies": layer.facies,
                    }
                    for layer in session.layers
                ],
                "smoothing": session.smoothing,
                "min_region_cm2": session.min_region_cm2,
                "rig_frame": _frame_key(frame),
                "injection_points_left_out": skipped,
            },
        }
    )
    if corrections:
        config["facies_generation"].update(
            {
                "corrections_path": str(corrections_path(project, session.run_id)),
                "corrections_sha256": corrections_digest(corrections),
                "correction_count": len(corrections),
            }
        )
    _progress(progress, "Validating…")
    validation = validate_geometry_config(config)
    errors = [
        str(row.get("message") or row.get("code") or "")
        for row in validation.get("issues") or []
        if row.get("severity") == "error"
    ]
    if errors:
        raise ValueError("The revision is not valid: " + "; ".join(errors))
    _progress(progress, "Saving the revision…")
    used = {int(row["layer_id"]) for row in regions}
    names = ", ".join(layer.name for layer in session.layers if layer.layer_id in used)
    return GeometryRevisionStore(project).save_revision(
        config,
        actor=actor,
        reason=f"Painted layers of run {reference.run_id}: {names}",
        validation=validation,
    )


__all__ = [
    "CORRECTION_KINDS",
    "LAYER_CORRECTIONS_FORMAT",
    "Correction",
    "SHARED_POINT_PX",
    "apply_corrections",
    "class_region_count",
    "corrected_classes",
    "corrections_digest",
    "corrections_not_applied",
    "corrections_path",
    "layer_at",
    "list_versions",
    "approve_version",
    "approvals_path",
    "map_approval",
    "read_map_approvals",
    "LAYER_APPROVAL_FORMAT",
    "load_corrections",
    "load_version",
    "map_digest",
    "restore_version",
    "save_version",
    "versions_folder",
    "LAYER_VERSION_FORMAT",
    "VERSION_FILES",
    "proposal_grid_scale",
    "save_corrections",
    "DEFAULT_EDGE_FILL_MM",
    "DEFAULT_LAYERS",
    "DEFAULT_SNAP_MM",
    "ERASER",
    "LAYER_COLORS",
    "LAYER_ROLES",
    "OUTSIDE",
    "OUTSIDE_FACIES",
    "OUTSIDE_POROSITY",
    "PERMEABILITY_UNITS",
    "PROPOSAL_DETAILS",
    "ROLE_POROSITY",
    "SMOOTHING_SIGMA",
    "FaciesEntry",
    "LayerProposal",
    "PaintedLayer",
    "PaintingImage",
    "PaintingSession",
    "Polygon",
    "PolygonGuide",
    "Stroke",
    "check_ready_for_revision",
    "contrast_limits",
    "enhance_contrast",
    "facies_ids",
    "frame_edge_fill",
    "frame_mask",
    "frame_pixels_per_metre",
    "load_painting_image",
    "load_painting_session",
    "photo_camera_u",
    "reference_camera_u",
    "run_camera_u",
    "load_proposal",
    "merge_small_regions",
    "minimum_region_pixels",
    "painted_label_map",
    "painting_folder",
    "polygon_guide",
    "proposal_digest",
    "proposal_path",
    "propose_layers",
    "rasterize_polygons",
    "rasterize_strokes",
    "save_painted_revision",
    "save_painting_session",
    "session_path",
    "shared_edges",
    "shared_points",
    "snap_pixels",
    "stretch_contrast_rgb8",
    "upsample_classes",
    "with_analysis_width",
]
