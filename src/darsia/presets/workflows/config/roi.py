"""Configuration for ROIs."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from darsia import CoordinateArray

from ._key_list_sources import FACIES_SOURCES, ROI_SOURCES
from .utils import _get_key

logger = logging.getLogger(__name__)


@dataclass
class RoiConfig:
    """Configuration for a single ROI."""

    name: str = field(
        default="",
        metadata={
            "name": "Name",
            "help": "Unique registry key for this ROI.",
        },
    )
    shape: Literal["box", "polygon", "label", "facies", "intersection", "union"] = (
        field(
            default="box",
            metadata={
                "name": "Shape",
                "help": (
                    "'box': axis-aligned box from corner_1/corner_2. 'polygon': an "
                    "ordered list of vertices, masked to its true footprint (not "
                    "just its bounding box). 'label': a raw segmentation label "
                    "index, with no material association. 'facies': a named "
                    "material class, resolved to one or more raw labels. "
                    "'intersection'/'union': combine other named ROIs (any shape) "
                    "via 'members'."
                ),
                "options": [
                    "box",
                    "polygon",
                    "label",
                    "facies",
                    "intersection",
                    "union",
                ],
            },
        )
    )
    corner_1: tuple[float, float] = field(
        default=(0.0, 0.0),
        metadata={
            "name": "Corner 1",
            "help": "First corner coordinate as x, y.",
            "placeholder": "x, y",
            "depends_on": {"field": "shape", "value": "box"},
        },
    )
    corner_2: tuple[float, float] = field(
        default=(0.0, 0.0),
        metadata={
            "name": "Corner 2",
            "help": "Second corner coordinate as x, y.",
            "placeholder": "x, y",
            "depends_on": {"field": "shape", "value": "box"},
        },
    )
    vertices: list[tuple[float, float]] = field(
        default_factory=list,
        metadata={
            "name": "Vertices",
            "help": "Ordered polygon vertices as x, y pairs (>= 3).",
            "depends_on": {"field": "shape", "value": "polygon"},
        },
    )
    label_index: int | None = field(
        default=None,
        metadata={
            "name": "Label index",
            "help": (
                "Raw label index in the run's labels image this ROI covers "
                "(shape='label' only). Typed freely - the GUI has no cheap "
                "way to validate this against the actual labels image; check "
                "the ROI/labels helper viewer for valid values."
            ),
            "depends_on": {"field": "shape", "value": "label"},
        },
    )
    facies_id: int | None = field(
        default=None,
        metadata={
            "name": "Facies",
            "help": (
                "Facies ID, resolved via [facies] to one or more raw labels "
                "(shape='facies' only). Offers [facies.<id>] entries as "
                "suggestions when defined, but many real configs define "
                "facies implicitly (e.g. via an external properties CSV) "
                "with no entries to list - type the ID directly in that case."
            ),
            "widget": "key_list",
            "key_list_sources": FACIES_SOURCES,
            "max_rows": 1,
            "editable": True,
            "depends_on": {"field": "shape", "value": "facies"},
        },
    )
    members: list[str] = field(
        default_factory=list,
        metadata={
            "name": "Member ROIs",
            "help": (
                "ROI registry keys to combine (>= 2), for "
                "shape='intersection'/'union' only."
            ),
            "widget": "key_list",
            "key_list_sources": ROI_SOURCES,
            "depends_on": {"field": "shape", "value": ["intersection", "union"]},
        },
    )

    @property
    def roi(self) -> CoordinateArray:
        """Computed ROI coordinates from the configured shape.

        Only meaningful for shape='box'/'polygon' - the other shapes have
        no plain geometric representation (no corners/vertices) and must
        be resolved via a shape-aware consumer instead (e.g.
        ExpertKnowledgeAdapter._mask_for_roi, which dispatches on `.shape`
        before ever reaching this property).
        """
        if self.shape == "polygon":
            return CoordinateArray(self.vertices)
        elif self.shape == "box":
            return CoordinateArray([self.corner_1, self.corner_2])
        else:
            raise ValueError(
                f"ROI '{self.name}' has shape={self.shape!r}, which has no "
                "plain geometric representation - use a shape-aware "
                "consumer (e.g. ExpertKnowledgeAdapter) instead of "
                "RoiConfig.roi."
            )

    def load(self, sec: dict) -> "RoiConfig":
        self.name = _get_key(sec, "name", required=True, type_=str)
        shape = _get_key(sec, "shape", required=False, type_=str, default=self.shape)
        valid_shapes = ("box", "polygon", "label", "facies", "intersection", "union")
        if shape not in valid_shapes:
            raise ValueError(
                f"[[roi]] '{self.name}': shape must be one of {valid_shapes}, "
                f"got {shape!r}."
            )
        self.shape = shape

        if shape == "box":
            self.corner_1 = _get_key(sec, "corner_1", required=True, type_=tuple)
            self.corner_2 = _get_key(sec, "corner_2", required=True, type_=tuple)
        elif shape == "polygon":
            vertices_raw = _get_key(sec, "vertices", required=True, type_=list)
            vertices = [tuple(v) for v in vertices_raw]
            if len(vertices) < 3:
                raise ValueError(
                    f"[[roi]] '{self.name}': shape='polygon' requires >= 3 "
                    f"vertices, got {len(vertices)}."
                )
            self.vertices = vertices
        elif shape == "label":
            self.label_index = _get_key(sec, "label_index", required=True, type_=int)
        elif shape == "facies":
            self.facies_id = _get_key(sec, "facies_id", required=True, type_=int)
        else:  # shape in ("intersection", "union")
            members_raw = _get_key(sec, "members", required=True, type_=list)
            members = [str(m) for m in members_raw]
            if len(members) < 2:
                raise ValueError(
                    f"[[roi]] '{self.name}': shape={shape!r} requires >= 2 "
                    f"members, got {len(members)}."
                )
            self.members = members

        return self


@dataclass
class MultiRoiConfig:
    """Configuration for multiple ROIs."""

    roi: dict[str, RoiConfig] = field(default_factory=dict)
    """Dictionary of ROI configurations."""

    def load(self, path: Path) -> "MultiRoiConfig":
        """Read current ``[[roi]]`` entries through the original ROI registry."""
        # Import here because the registry itself uses RoiConfig.
        from .roi_registry import RoiRegistry

        self.roi = RoiRegistry().load(path).rois
        return self
