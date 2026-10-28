"""Setup crop area: the reference quadrilateral or a separate area.

The reference quadrilateral (the exported rig frame) defines alignment. Its
four corners map onto the rig rectangle, which gives rig coordinates in
metres, and every run's baseline photograph is registered through it. By
default Setup also crops to it.

A series may instead choose a separate crop area: four points marked on the
reference run's raw baseline photograph. Setup then crops to the rig-aligned
rectangle between those points. Its size and position come from the
reference quadrilateral, so the area needs no physical measurement, and the
cropped image keeps the reference quadrilateral's coordinates through the
crop's ``origin``.

The choice and the marked areas are stored with the rig frames in the
series' geometry package, never in ``series.toml``. A native Setup export
records the area next to its rig frame only when the separate area is
chosen, so the default writes nothing new anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from darsia.fluidflower.project import FluidFlowerProject, utc_now
from darsia.fluidflower.series_geometry import (
    CROP_CORNERS,
    RigFrame,
    _update_link,
    native_series_link,
    order_corners,
)
from darsia.fluidflower.storage_paths import machine_local_text, machine_stored_text

CROP_REFERENCE_QUAD = "reference_quad"
CROP_SEPARATE_AREA = "separate_area"
CROP_CHOICES = (CROP_REFERENCE_QUAD, CROP_SEPARATE_AREA)
CHOICE_KEY = "crop_choice"
AREAS_KEY = "crop_areas"
EXPORT_KEY = "crop_area"
# Rectangle edges are rounded inward to 0.1 mm.
_DECIMALS = 4


def _corner_points(raw: Any) -> dict[str, tuple[float, float]]:
    """Four named, finite and distinct (row, col) points, named by position."""
    if not isinstance(raw, Mapping) or set(raw) != set(CROP_CORNERS):
        raise ValueError("A separate crop area needs its four named corners.")
    corners = {}
    for name in CROP_CORNERS:
        point = raw[name]
        try:
            if isinstance(point, (str, bytes)) or len(point) != 2:
                raise ValueError(name)
            values = tuple(float(value) for value in point)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid corner of the separate crop area: {name}") from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite corner of the separate crop area: {name}")
        corners[name] = values
    if len(set(corners.values())) != 4:
        raise ValueError("The corners of the separate crop area are not distinct.")
    return order_corners(corners)


@dataclass(frozen=True)
class CropArea:
    """Four points marked on one run's raw baseline photograph.

    ``corners`` are (row, col) pixels with the rig frame's corner names, in
    the same photograph and pixel grid as the reference quadrilateral. They
    are named again by their position, as for a marked rig frame.
    ``image_size`` is (width, height).
    """

    corners: Mapping[str, tuple[float, float]]
    run_id: str
    image: str = ""
    image_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not str(self.run_id or "").strip():
            raise ValueError("A separate crop area belongs to a run.")
        object.__setattr__(self, "corners", _corner_points(self.corners))
        object.__setattr__(self, "run_id", str(self.run_id))
        object.__setattr__(self, "image", str(self.image or ""))
        if self.image_size is not None:
            width, height = (int(value) for value in self.image_size)
            if min(width, height) <= 0:
                raise ValueError("The photograph of a separate crop area has no size.")
            object.__setattr__(self, "image_size", (width, height))

    def to_dict(self) -> dict[str, Any]:
        """The area as stored in the geometry package and in an export."""
        return {
            "corners": {name: [float(v) for v in self.corners[name]] for name in CROP_CORNERS},
            "run_id": self.run_id,
            "image": machine_stored_text(self.image),
            "image_size": list(self.image_size) if self.image_size else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CropArea":
        """Read a stored or exported area; refuse anything incomplete."""
        if not isinstance(payload, Mapping):
            raise ValueError("A separate crop area must be a table.")
        size = payload.get("image_size")
        return cls(
            corners=payload.get("corners"),
            run_id=str(payload.get("run_id") or ""),
            image=machine_local_text(str(payload.get("image") or "")),
            image_size=tuple(size) if size else None,
        )


def crop_area_from_frame(frame: RigFrame) -> CropArea:
    """A separate crop area that starts as a copy of the reference quadrilateral."""
    if not frame.complete():
        raise ValueError("Mark the four corners of the reference quadrilateral first.")
    return CropArea(
        corners={name: frame.corners[name] for name in CROP_CORNERS},
        run_id=frame.run_id,
        image=frame.image,
        image_size=frame.image_size,
    )


def separate_crop_reason(
    project: FluidFlowerProject, *, configuration: Mapping[str, Any] | None = None
) -> str:
    """Why a separate crop area cannot be chosen now, or "" when it can.

    A geometry drawn on the corrected rig grid is already cropped, so its
    Setup crop is not changed by "Apply rig corners to Setup".
    ``configuration`` is the geometry configuration to judge, by default that
    of the current saved revision.
    """
    if configuration is None:
        from darsia.fluidflower.geometry_facies import GeometryRevisionStore

        try:
            configuration = GeometryRevisionStore(project).current_config()
        except (OSError, ValueError) as error:
            return f"The saved geometry revision could not be read: {error}"
    if configuration.get("frame", "raw") == "rig-corrected":
        return (
            "This geometry is drawn on the corrected rig grid, which is already "
            "cropped. A separate crop area needs geometry drawn on a raw "
            "baseline photograph."
        )
    return ""


def saved_crop_choice(project: FluidFlowerProject) -> str:
    """The series' Setup crop choice; the reference quadrilateral unless changed."""
    value = native_series_link(project).get(CHOICE_KEY)
    if value is None:
        return CROP_REFERENCE_QUAD
    if value not in CROP_CHOICES:
        raise ValueError(f"Unknown Setup crop choice in the geometry package: {value!r}")
    return value


def save_crop_choice(
    project: FluidFlowerProject, choice: str, *, actor: str = ""
) -> bool:
    """Store the Setup crop choice; write nothing when it is unchanged.

    Returns whether the geometry package was written. Changing the choice
    needs a new export, Apply rig corners to Setup and Build afterwards.
    """
    if choice not in CROP_CHOICES:
        raise ValueError(f"Unknown Setup crop choice: {choice!r}")
    # Read the stored text directly, so that an unknown value can be replaced.
    stored = native_series_link(project).get(CHOICE_KEY)
    if stored == choice or (stored is None and choice == CROP_REFERENCE_QUAD):
        return False
    _update_link(
        project,
        {CHOICE_KEY: choice},
        actor=actor,
        reason=f"Setup crop choice set to {choice}",
    )
    return True


def saved_crop_areas(project: FluidFlowerProject) -> dict[str, CropArea]:
    """Every stored separate crop area, by run."""
    raw = native_series_link(project).get(AREAS_KEY) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("The separate crop areas of the geometry package are not a table.")
    return {str(run_id): CropArea.from_dict(payload) for run_id, payload in raw.items()}


def save_crop_area(
    project: FluidFlowerProject, area: CropArea, *, actor: str = ""
) -> None:
    """Store the separate crop area marked on one run's photograph."""
    areas = dict(native_series_link(project).get(AREAS_KEY) or {})
    payload = area.to_dict()
    payload["updated_utc"] = utc_now()
    areas[area.run_id] = payload
    _update_link(
        project,
        {AREAS_KEY: areas},
        actor=actor,
        reason=f"Separate crop area of run {area.run_id} updated",
    )


def remove_crop_area(
    project: FluidFlowerProject, run_id: str, *, actor: str = ""
) -> bool:
    """Remove one run's separate crop area; returns whether one was stored."""
    areas = dict(native_series_link(project).get(AREAS_KEY) or {})
    if areas.pop(str(run_id), None) is None:
        return False
    _update_link(
        project,
        {AREAS_KEY: areas},
        actor=actor,
        reason=f"Separate crop area of run {run_id} removed",
    )
    return True


def _inward(value: float, *, up: bool) -> float:
    """Round a rectangle edge to 0.1 mm towards the inside of the area."""
    scaled = value * 10**_DECIMALS
    nearest = round(scaled)
    if abs(scaled - nearest) < 1e-6:
        scaled = nearest
    return (math.ceil(scaled) if up else math.floor(scaled)) / 10**_DECIMALS


@dataclass(frozen=True)
class CropExtent:
    """The rig-aligned rectangle that Setup crops to.

    ``left_m`` and ``top_m`` are rig coordinates of its top-left corner (x
    from the reference quadrilateral's left edge, y above its bottom edge);
    ``width_m`` and ``height_m`` are its size, all in metres. ``corners``
    are its (row, col) pixels in the reference quadrilateral's photograph.
    """

    left_m: float
    top_m: float
    width_m: float
    height_m: float
    corners: Mapping[str, tuple[float, float]]

    @property
    def right_m(self) -> float:
        return round(self.left_m + self.width_m, _DECIMALS)

    @property
    def bottom_m(self) -> float:
        return round(self.top_m - self.height_m, _DECIMALS)

    @property
    def origin_m(self) -> tuple[float, float]:
        """Rig coordinates of the cropped image's top-left corner."""
        return self.left_m, self.top_m

    def native_crop(
        self, corners: Mapping[str, Sequence[float]] | None = None
    ) -> dict[str, Any]:
        """The rectangle as DarSIA's ``[corrections.curvature.crop]`` table.

        ``corners`` are the rectangle's corners in one run's photograph after
        registration, by default those in the reference photograph. Pixels
        are rounded as for the reference quadrilateral's crop, and ``origin``
        keeps the reference quadrilateral's rig coordinates.
        """
        points = self.corners if corners is None else corners
        table: dict[str, Any] = {
            name: [int(round(points[name][0])), int(round(points[name][1]))]
            for name in CROP_CORNERS
        }
        table.update(
            width=self.width_m,
            height=self.height_m,
            in_meters=True,
            origin=[self.left_m, self.top_m],
        )
        return table


def crop_extent(frame: RigFrame, area: CropArea) -> CropExtent:
    """The rig-aligned rectangle between the marked points of a separate area.

    x runs from the inner of the two left points to the inner of the two
    right points, and y from the inner of the two bottom points to the inner
    of the two top points, so nothing outside the marked area is cropped in.
    The edges are rounded inward to 0.1 mm. The reference quadrilateral
    (``frame``) gives every coordinate; the area needs no measurement.
    """
    if not frame.complete() or not frame.has_size():
        raise ValueError(
            "The reference quadrilateral needs its four corners and the rig's "
            "inner size before a separate crop area can be used."
        )
    if frame.run_id and area.run_id != frame.run_id:
        raise ValueError(
            f"The separate crop area was marked on run {area.run_id}; the "
            f"reference quadrilateral belongs to run {frame.run_id}."
        )
    if (
        frame.image_size
        and area.image_size
        and tuple(frame.image_size) != tuple(area.image_size)
    ):
        raise ValueError(
            "The separate crop area was marked on a photograph of another size "
            "than the reference quadrilateral."
        )
    rig = {name: frame.pixel_to_rig(*area.corners[name]) for name in CROP_CORNERS}
    if not all(math.isfinite(value) for point in rig.values() for value in point):
        raise ValueError("The separate crop area lies outside the reference photograph's plane.")
    left = _inward(max(rig["top_left"][0], rig["bottom_left"][0]), up=True)
    right = _inward(min(rig["top_right"][0], rig["bottom_right"][0]), up=False)
    bottom = _inward(max(rig["bottom_left"][1], rig["bottom_right"][1]), up=True)
    top = _inward(min(rig["top_left"][1], rig["top_right"][1]), up=False)
    if not (right > left and top > bottom):
        raise ValueError(
            "The separate crop area leaves no rectangle between its points. "
            "Mark its corners around the area to keep."
        )
    width = round(right - left, _DECIMALS)
    height = round(top - bottom, _DECIMALS)
    corners = {
        "top_left": frame.rig_to_pixel(left, top),
        "bottom_left": frame.rig_to_pixel(left, bottom),
        "bottom_right": frame.rig_to_pixel(right, bottom),
        "top_right": frame.rig_to_pixel(right, top),
    }
    return CropExtent(left, top, width, height, corners)


def export_crop_area(
    project: FluidFlowerProject, frame: RigFrame | None
) -> dict[str, Any] | None:
    """The separate crop area that an export records next to its rig frame.

    For the default choice (same as the reference quadrilateral) this is
    ``None`` and the export stays as it was. With a separate area, the
    reference run needs an area on its photograph, and its rectangle must
    exist; otherwise the export is refused with the reason.
    """
    if saved_crop_choice(project) != CROP_SEPARATE_AREA:
        return None
    if frame is None or not frame.run_id:
        raise ValueError(
            "A separate crop area needs the reference quadrilateral of the reference run."
        )
    area = saved_crop_areas(project).get(frame.run_id)
    if area is None:
        raise ValueError(
            f"Mark the separate crop area on the reference photograph of run "
            f"{frame.run_id}, or choose Same as reference quad."
        )
    crop_extent(frame, area)
    return area.to_dict()


__all__ = [
    "AREAS_KEY",
    "CHOICE_KEY",
    "CROP_CHOICES",
    "CROP_REFERENCE_QUAD",
    "CROP_SEPARATE_AREA",
    "EXPORT_KEY",
    "CropArea",
    "CropExtent",
    "crop_area_from_frame",
    "crop_extent",
    "export_crop_area",
    "remove_crop_area",
    "save_crop_area",
    "save_crop_choice",
    "saved_crop_areas",
    "saved_crop_choice",
    "separate_crop_reason",
]
