"""Manual maps: the polygons of a layer painting exactly as drawn.

A manual map is a saved version of its own kind, ``manual map``: the session's
polygons rasterised at the photograph's full resolution, without a classifier.
Where polygons of different layers overlap the smaller one wins, unless a
layer has explicit Always on top priority. The
proposal's frame-edge fill gives the strip along the rig-frame edge that no
polygon reaches the nearest polygon's layer. Every other pixel inside the rig
frame that no polygon covers stays uncovered and is counted; a map with such
pixels can be saved, but not approved.

Its version lies beside the run's other versions
(:func:`layer_painting.versions_folder`), with the same record format and its
own files: the session and ``manual_map.npz``. It is approved into the same
approvals file and placed and adopted like an edited map
(:func:`version_layers`). Saving or restoring one never changes the run's
proposal or corrections.
"""

from __future__ import annotations

from darsia.fluidflower import preset_wasserstein_inputs as _preset_io

import hashlib
import json
import os
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower.project import FluidFlowerProject, utc_now
from darsia.fluidflower.series_geometry import RigFrame

MANUAL_MAP_FORMAT = "darsia-fluidflower-manual-map-v1"
# The kind of a manual map's version and of its approval.
MANUAL_MAP_KIND = "manual map"
# The files of a manual map's version.
MANUAL_VERSION_FILES = ("session.json", "manual_map.npz")


@dataclass
class ManualMap:
    """The session's polygons drawn exactly on the photograph's full-resolution grid.

    ``classes`` is the layer of each pixel: where polygons of different layers
    overlap the smaller one wins (:func:`layer_painting.rasterize_polygons`),
    and ``edge_filled`` marks the pixels along the rig-frame edge that no
    polygon covers and that took the nearest polygon's layer by the
    frame-edge fill. ``classes`` is ``OUTSIDE`` outside the rig frame
    (``inside`` false) and also 0 at the pixels inside it that no polygon
    covers; :meth:`uncovered` gives those, and nothing fills them.
    """

    classes: np.ndarray
    inside: np.ndarray
    edge_filled: np.ndarray
    image_size: tuple[int, int]
    edge_fill_mm: float = 0.0

    def uncovered(self) -> np.ndarray:
        """Pixels inside the rig frame that no polygon covers."""

        return self.inside & (self.classes == lp.OUTSIDE)

    def uncovered_pixels(self) -> int:
        """How many pixels inside the rig frame no polygon covers."""

        return int(np.count_nonzero(self.uncovered()))

    def edge_filled_pixels(self) -> int:
        """How many pixels the frame-edge fill gave a layer."""

        return int(np.count_nonzero(self.edge_filled))

    def save(self, path: Path) -> None:
        """Write the map as a compressed ``.npz`` file, whole or not at all."""

        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema": 1,
            "format": MANUAL_MAP_FORMAT,
            "image_size": list(self.image_size),
            "edge_fill_mm": self.edge_fill_mm,
        }
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                classes=np.asarray(self.classes, dtype=np.uint8),
                inside=np.asarray(self.inside, dtype=bool),
                edge_filled=np.asarray(self.edge_filled, dtype=bool),
                meta=np.asarray(json.dumps(meta)),
            )
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path) -> ManualMap:
        """Read a map written by :meth:`save`; arrays of another size raise
        ValueError."""

        with np.load(_preset_io.source(path), allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            if meta.get("format") != MANUAL_MAP_FORMAT:
                raise ValueError("Unsupported manual map file")
            classes = np.asarray(archive["classes"], dtype=np.uint8)
            inside = np.asarray(archive["inside"], dtype=bool)
            edge_filled = np.asarray(archive["edge_filled"], dtype=bool)
        size = (int(meta["image_size"][0]), int(meta["image_size"][1]))
        if not classes.shape == inside.shape == edge_filled.shape == (size[1], size[0]):
            raise ValueError("The manual map's arrays do not have its photograph's size")
        return cls(
            classes, inside, edge_filled, size, float(meta.get("edge_fill_mm") or 0.0)
        )


def manual_map(
    session: lp.PaintingSession,
    image_size: tuple[int, int],
    *,
    frame: RigFrame | None,
    progress: lp.ProgressCallback | None = None,
) -> ManualMap:
    """The session's polygons as a map at the photograph's full resolution.

    Each pixel takes the layer of the smallest polygon that covers it; an
    explicitly marked Always on top layer wins over ordinary layers, keeping
    the same smaller-polygon rule among marked layers. The polygons are drawn
    exactly as given: no classifier and no snap distance are used, and strokes are
    left out (they guide the proposal). Pixels outside the rig frame are
    ``OUTSIDE``. Inside it, the proposal's frame-edge fill
    (``session.edge_fill_mm``) gives the pixels along the frame edge that no
    polygon reaches the layer of the nearest polygon; every other pixel that
    no polygon covers stays 0 and is counted by :meth:`ManualMap.uncovered`.
    ``image_size`` is the photograph's (width, height).
    """

    if session.image_size and tuple(session.image_size) != tuple(image_size):
        raise ValueError(
            "The layers were marked on a photograph of another size "
            f"({session.image_size[0]} x {session.image_size[1]} px)."
        )
    width, height = int(image_size[0]), int(image_size[1])
    shape = (height, width)
    scale = (1.0, 1.0)
    lp._progress(progress, "Drawing the polygons at full resolution…")
    inside = lp.frame_mask(frame, shape, scale)
    from .painting_layer_priority import rasterize_polygons

    classes = rasterize_polygons(session, shape, scale)
    classes[~inside] = lp.OUTSIDE
    edge_filled = np.zeros(shape, dtype=bool)
    edge_px = lp.snap_pixels(session.edge_fill_mm, frame, scale, (width, height))
    if edge_px > 0.0 and session.polygons:
        lp._progress(progress, "Filling the strip along the rig-frame edge…")
        layer_ids = sorted({polygon.layer_id for polygon in session.polygons})

        def distances() -> Iterable[tuple[int, np.ndarray]]:
            # One layer's polygons at a time, as layer_painting.polygon_guide
            # draws them for the proposal.
            for layer_id in layer_ids:
                mask = np.zeros(shape, dtype=np.uint8)
                for polygon in session.polygons:
                    if polygon.layer_id == layer_id:
                        lp._fill_polygon(mask, polygon, scale, 1)
                yield layer_id, lp._distance_to((mask > 0) & inside)

        fill = _frame_edge_strip(classes, inside, edge_px, distances())
        edge_filled = fill > 0
        classes[edge_filled] = fill[edge_filled]
    return ManualMap(
        classes, inside, edge_filled, (width, height), float(session.edge_fill_mm)
    )


def _frame_edge_strip(
    owner: np.ndarray,
    inside: np.ndarray,
    edge_px: float,
    distances: Iterable[tuple[int, np.ndarray]],
) -> np.ndarray:
    """The rule of :func:`layer_painting.frame_edge_fill`, one layer's
    distances at a time (0: not filled).

    A pixel inside the frame that no polygon covers (``owner`` 0), within
    ``edge_px`` of the frame edge (the photograph's border counts) and within
    ``edge_px`` of a polygon takes the nearest polygon's layer; between
    equally near layers the first one wins. ``distances`` gives each layer
    with polygons, in ascending order, with the distance to its polygons
    inside the frame. Only the strip's pixels are kept, so that a
    full-resolution map holds one distance array at a time rather than all.
    """

    fill = np.zeros(inside.shape, dtype=np.uint8)
    padded = np.pad(inside, 1, constant_values=False)
    to_edge = lp._distance_to(~padded)[1:-1, 1:-1]
    candidate = inside & (owner == 0) & (to_edge <= edge_px)
    del padded, to_edge
    near: np.ndarray | None = None
    nearest: np.ndarray | None = None
    for layer, distance in distances:
        values = distance[candidate]
        if near is None or nearest is None:
            near = values
            nearest = np.full(values.shape, layer, dtype=np.uint8)
            continue
        closer = values < near
        near[closer] = values[closer]
        nearest[closer] = layer
    if near is None or nearest is None:
        return fill
    strip = near <= edge_px
    rows, cols = np.nonzero(candidate)
    fill[rows[strip], cols[strip]] = nearest[strip]
    return fill


def manual_map_digest(session: lp.PaintingSession, manual: ManualMap) -> str:
    """SHA-256 of what a manual map depends on.

    The session as its file reads back, without its save time and author,
    and the map's size and arrays. The same drawing on the same photograph
    and rig frame gives the same digest whenever it is saved, and a saved
    version's files give it again.
    """

    stored = lp.PaintingSession.from_dict(json.loads(json.dumps(session.to_dict())))
    payload = stored.to_dict()
    payload.pop("updated_utc", None)
    payload.pop("updated_by", None)
    digest = hashlib.sha256()
    settings = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest.update(settings.encode("utf-8"))
    shape = [MANUAL_MAP_FORMAT, list(manual.image_size), list(manual.classes.shape)]
    digest.update(json.dumps(shape).encode("utf-8"))
    digest.update(np.ascontiguousarray(manual.classes, dtype=np.uint8).tobytes())
    digest.update(np.packbits(np.asarray(manual.inside, dtype=bool)).tobytes())
    digest.update(np.packbits(np.asarray(manual.edge_filled, dtype=bool)).tobytes())
    return digest.hexdigest()


def manual_map_differences(manual: ManualMap, layers: np.ndarray) -> np.ndarray:
    """Pixels where a manual map and another map of the same photograph disagree.

    ``layers`` is the other map's layer of each pixel, as from
    :func:`painted_layers`. A pixel counts where the manual map gives it a
    layer and the other map another one, or none. Pixels outside the rig
    frame and pixels no polygon covers do not count; see
    :meth:`ManualMap.uncovered` for those.
    """

    if tuple(layers.shape) != tuple(manual.classes.shape):
        raise ValueError("The two maps are not on the same grid")
    return (manual.classes != lp.OUTSIDE) & (layers != manual.classes)


def pixel_area_cm2(frame: RigFrame | None, image_size: tuple[int, int]) -> float:
    """Area of one pixel of the raw photograph in cm².

    From the rig frame's corners and inner size; without a rig size the
    photograph is taken to be ``NOMINAL_IMAGE_WIDTH_M`` wide, as in
    :func:`layer_painting.snap_pixels`.
    """

    per_metre = lp.frame_pixels_per_metre(frame)
    if per_metre is None:
        per_metre = float(image_size[0]) / lp.NOMINAL_IMAGE_WIDTH_M
    return 1.0e4 / per_metre**2


def painted_layers(
    session: lp.PaintingSession,
    proposal: lp.LayerProposal,
    *,
    frame: RigFrame | None,
    progress: lp.ProgressCallback | None = None,
    corrections: Sequence[lp.Correction] = (),
) -> np.ndarray:
    """The layer of each pixel of a painted map on the raw photograph's grid.

    The label map of :func:`layer_painting.painted_label_map` with each
    region replaced by its layer (uint8); pixels outside the rig frame are
    ``OUTSIDE``.
    """

    labels, regions = lp.painted_label_map(
        session, proposal, frame=frame, progress=progress, corrections=corrections
    )
    table = np.zeros(int(labels.max()) + 1, dtype=np.uint8)
    for row in regions:
        table[int(row["label_id"])] = int(row["layer_id"])
    return table[labels]


# ---------------------------------------------------------------------------
# A manual map as a saved version


def is_manual_version(record: Mapping[str, Any]) -> bool:
    """Whether a saved version is a manual map."""

    return record.get("kind") == MANUAL_MAP_KIND


def _new_version_folder(project: FluidFlowerProject, run_id: str, digest: str) -> Path:
    """A new, empty folder for a version, named as
    :func:`layer_painting.save_version` names its folders: the UTC time and
    the map's SHA-256; an existing folder is never reused."""

    root = lp.versions_folder(project, run_id)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    folder = root / f"{stamp}-{digest[:12]}"
    count = 1
    while True:
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            count += 1
            folder = root / f"{stamp}-{digest[:12]}-{count}"


def save_manual_version(
    project: FluidFlowerProject,
    session: lp.PaintingSession,
    manual: ManualMap,
    *,
    name: str,
) -> dict[str, Any]:
    """Save a manual map as a new version and return its record.

    The session and the map (``manual_map.npz``) go into a new folder beside
    the run's other versions that is never changed afterwards; no proposal is
    needed. The record has the format of
    :func:`layer_painting.save_version`, with the kind ``manual map``,
    ``sha256`` from :func:`manual_map_digest`, how many pixels inside the rig
    frame no polygon covers and how many the frame-edge fill gave a layer.
    """

    name = str(name).strip()
    if not name:
        raise ValueError("A version needs a name")
    if session.image_size and tuple(session.image_size) != tuple(manual.image_size):
        raise ValueError("The manual map was drawn on a photograph of another size")
    digest = manual_map_digest(session, manual)
    folder = _new_version_folder(project, session.run_id, digest)
    (folder / "session.json").write_text(lp._session_text(session), encoding="utf-8")
    manual.save(folder / "manual_map.npz")
    record = {
        "schema": 1,
        "format": lp.LAYER_VERSION_FORMAT,
        "run_id": session.run_id,
        "name": name,
        "kind": MANUAL_MAP_KIND,
        "sha256": digest,
        "created_utc": utc_now(),
        "uncovered_pixels": manual.uncovered_pixels(),
        "edge_filled_pixels": manual.edge_filled_pixels(),
        "files": {file: lp._file_sha256(folder / file) for file in MANUAL_VERSION_FILES},
    }
    lp._atomic_write_text(
        folder / "version.json", json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    )
    return dict(record, folder=str(folder))


def load_manual_version(
    record: Mapping[str, Any],
) -> tuple[lp.PaintingSession, ManualMap]:
    """The session and map of a saved manual map.

    The record must list the version's files, and each is checked against
    its SHA-256 first; a changed file raises ValueError.
    """

    if not is_manual_version(record):
        raise ValueError("The saved version is not a manual map")
    files = dict(record.get("files") or {})
    if set(files) != set(MANUAL_VERSION_FILES):
        raise ValueError("The saved manual map's record does not list its files")
    from .storage_paths import resolve_storage_path
    folder = resolve_storage_path(str(record["folder"]))
    for file, expected in files.items():
        if lp._file_sha256(folder / file) != expected:
            raise ValueError(
                f"The saved version's {file} has changed since it was saved"
            )
    session = lp.PaintingSession.from_dict(
        json.loads(_preset_io.read_text(folder / "session.json",encoding="utf-8"))
    )
    return session, ManualMap.load(folder / "manual_map.npz")


def approve_manual_version(
    record: Mapping[str, Any],
    *,
    reviewer: str | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Approve a saved manual map as its run's map: append a record and return it.

    The record is that of :func:`layer_painting.approve_version`, in the same
    approvals file, so that placement and adoption read it alike. Before it
    is written the version's files are checked against their SHA-256, every
    pixel inside the rig frame must have a layer (from a polygon or the
    frame-edge fill), and the version's SHA-256 must be that of its content;
    otherwise ValueError says why. Approving changes no map, session,
    proposal, version or geometry revision. A transient file error is
    retried.
    """

    from darsia.fluidflower.pipeline.job_queue import retry_transient

    session, manual = load_manual_version(record)
    uncovered = manual.uncovered_pixels()
    if uncovered:
        raise ValueError(
            f"{uncovered:,} pixels inside the rig frame are covered by no polygon "
            "and lie beyond the frame-edge fill; cover them with polygons and save "
            "a new version before approving the manual map"
        )
    if manual_map_digest(session, manual) != record.get("sha256"):
        raise ValueError("The saved manual map's SHA-256 does not match its content")
    from .storage_paths import resolve_storage_path
    folder = resolve_storage_path(str(record["folder"]))
    approval = {
        "format": lp.LAYER_APPROVAL_FORMAT,
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


def restore_manual_version(
    project: FluidFlowerProject, record: Mapping[str, Any], *, actor: str = ""
) -> lp.PaintingSession:
    """Make a saved manual map's session the run's current session.

    Its polygons, layers and facies come back; the run's proposal and
    corrections stay as they are, and the version is not changed. The files
    are checked first.
    """

    session, _manual = load_manual_version(record)
    lp.save_painting_session(project, session, actor=actor)
    return session


def version_layers(
    record: Mapping[str, Any],
    *,
    frame: RigFrame | None,
    progress: lp.ProgressCallback | None = None,
) -> tuple[lp.PaintingSession, np.ndarray]:
    """The session of a saved version and the layer of each pixel of its map.

    A manual map is used exactly as saved and must cover every pixel inside
    the rig frame. A proposal or an edited map is read by
    :func:`layer_painting.load_version` and drawn by :func:`painted_layers`
    in the rig frame ``frame``. The files are checked first; pixels outside
    the rig frame are ``OUTSIDE``.
    """

    if is_manual_version(record):
        session, manual = load_manual_version(record)
        if manual.uncovered_pixels():
            raise ValueError(
                "The manual map has pixels inside the rig frame that no polygon covers"
            )
        return session, manual.classes.copy()
    session, proposal, corrections = lp.load_version(record)
    return session, painted_layers(
        session, proposal, frame=frame, progress=progress, corrections=corrections
    )


__all__ = [
    "MANUAL_MAP_FORMAT",
    "MANUAL_MAP_KIND",
    "MANUAL_VERSION_FILES",
    "ManualMap",
    "approve_manual_version",
    "is_manual_version",
    "load_manual_version",
    "manual_map",
    "manual_map_differences",
    "manual_map_digest",
    "painted_layers",
    "pixel_area_cm2",
    "restore_manual_version",
    "save_manual_version",
    "version_layers",
]
