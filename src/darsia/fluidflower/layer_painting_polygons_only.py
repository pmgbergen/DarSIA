"""Exact native-grid polygon proposals, separate from the classifier format.

Zero pixels inside the saved frame are undecided. Only explicit corrections
may assign them; classification, smoothing, snapping and edge filling are not
part of this method. The ordinary layer-painting branches delegate unchanged.
"""

from __future__ import annotations

from darsia.fluidflower import preset_wasserstein_inputs as _preset_io

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import uuid
import zipfile

import numpy as np

from . import layer_painting as lp
from .series_geometry import CROP_CORNERS, RigFrame


METHOD = "polygons_only_v1"
PROPOSAL_FORMAT = "darsia-fluidflower-polygons-only-proposal-v1"
_MAX_META = 16 * 1024 * 1024
_MAX_APPROVAL_HISTORY = 16 * 1024 * 1024
_MAX_PIXELS = 250_000_000


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf8")


def _size(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(v) is not int or v <= 0 for v in value)
            or value[0] * value[1] > _MAX_PIXELS):
        raise ValueError("Invalid native image size")
    return tuple(value)


def _saved_frame(payload, size):
    if payload is None:
        return None
    keys = {"corners", "width_m", "height_m", "run_id", "source", "image", "image_size"}
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValueError("Invalid complete frame identity")
    corners = payload["corners"]
    if not isinstance(corners, dict) or set(corners) != set(CROP_CORNERS):
        raise ValueError("Polygons-only proposals require a complete rig frame")
    for point in corners.values():
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in point)):
            raise ValueError("Invalid frame corner")
    for key in ("width_m", "height_m"):
        v = payload[key]
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                              or not math.isfinite(v) or v <= 0):
            raise ValueError("Invalid physical frame size")
    if payload["image_size"] is not None and _size(payload["image_size"]) != size:
        raise ValueError("Frame and proposal image sizes differ")
    if any(not isinstance(payload[key], str) for key in ("run_id", "source", "image")):
        raise ValueError("Invalid frame metadata")
    # Do not reorder, round, fit or infer the saved geometry.
    return RigFrame(corners={k: tuple(v) for k, v in corners.items()},
                    width_m=payload["width_m"], height_m=payload["height_m"],
                    run_id=payload["run_id"], source=payload["source"],
                    image=payload["image"], image_size=tuple(payload["image_size"])
                    if payload["image_size"] is not None else None)


def _source(session, frame, size):
    size = _size(size)
    if session.image_size is not None and tuple(session.image_size) != size:
        raise ValueError("Session and photograph image sizes differ")
    ids = [row.layer_id for row in session.layers]
    if (not ids or any(type(v) is not int or not 1 <= v <= 255 for v in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Layer IDs must be distinct integers from 1 to 255")
    polygons = []
    for polygon in session.polygons:
        if type(polygon.layer_id) is not int or polygon.layer_id not in ids:
            raise ValueError("A polygon refers to an unknown layer")
        points = []
        for point in polygon.points:
            if len(point) != 2 or any(not math.isfinite(float(v)) for v in point):
                raise ValueError("Polygon vertices must be finite row/column pairs")
            points.append([float(point[0]), float(point[1])])
        if len(points) < 3 or lp.Polygon(polygon.layer_id, points).area() <= 0:
            raise ValueError("A polygon needs at least three vertices and positive area")
        polygons.append([polygon.layer_id, points])
    full_frame = None if frame is None else frame.to_dict()
    _saved_frame(full_frame, size)
    result = dict(method=METHOD, image_size=list(size), run_id=session.run_id,
                  image_path=str(session.image_path), image_stamp=session.image_stamp,
                  layer_ids=ids, polygons=polygons, frame=full_frame)
    _json_bytes(result)
    return result


def _parse_source(source):
    keys = {"method", "image_size", "run_id", "image_path", "image_stamp",
            "layer_ids", "polygons", "frame"}
    if not isinstance(source, dict) or set(source) != keys or source["method"] != METHOD:
        raise ValueError("Invalid polygons-only source identity")
    size = _size(source["image_size"])
    if any(not isinstance(source[key], str) for key in ("run_id", "image_path", "image_stamp")):
        raise ValueError("Invalid source image identity")
    ids = source["layer_ids"]
    if (not isinstance(ids, list) or not ids
            or any(type(v) is not int or not 1 <= v <= 255 for v in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Invalid source layer IDs")
    polygons = []
    if not isinstance(source["polygons"], list):
        raise ValueError("Invalid source polygons")
    for row in source["polygons"]:
        if not isinstance(row, list) or len(row) != 2 or type(row[0]) is not int or row[0] not in ids:
            raise ValueError("Invalid source polygon layer")
        points = row[1]
        if (not isinstance(points, list) or len(points) < 3
                or any(not isinstance(p, list) or len(p) != 2
                       or any(isinstance(v, bool) or not isinstance(v, (int, float))
                              or not math.isfinite(v) for v in p) for p in points)):
            raise ValueError("Invalid exact source polygon vertices")
        polygon = lp.Polygon(row[0], [tuple(p) for p in points])
        if polygon.area() <= 0:
            raise ValueError("A source polygon has no area")
        polygons.append(polygon)
    frame = _saved_frame(source["frame"], size)
    _json_bytes(source)
    return size, frame, polygons


def _arrays(source):
    size, frame, polygons = _parse_source(source)
    shape = (size[1], size[0])
    inside = lp.frame_mask(frame, shape, (1., 1.))
    classes = lp.rasterize_polygons(polygons, shape, (1., 1.))
    classes[~inside] = lp.OUTSIDE
    confidence = np.where(classes != lp.OUTSIDE, 255, 0).astype(np.uint8)
    return classes, confidence, inside


def _digest(source, classes, confidence, inside):
    digest = hashlib.sha256(_json_bytes(source))
    for value in (classes, confidence, inside):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


@dataclass
class PolygonsOnlyProposal(lp.LayerProposal):
    inside: np.ndarray = field(default_factory=lambda: np.empty((0, 0), bool))
    source: dict[str, Any] = field(default_factory=dict)

    def save(self, path):
        save_proposal(self, path)

    def uncovered_pixels(self):
        return int(np.count_nonzero(self.inside & (self.classes == lp.OUTSIDE)))


def is_polygons_only(proposal):
    return isinstance(proposal, PolygonsOnlyProposal)


def _validate(proposal, *, raster=False):
    if not is_polygons_only(proposal):
        raise ValueError("Not a polygons-only proposal")
    size, _frame, polygons = _parse_source(proposal.source)
    if tuple(proposal.image_size) != size or proposal.frame != proposal.source["frame"]:
        raise ValueError("Proposal and source frame/size identities differ")
    shape = (size[1], size[0])
    for value, dtype in ((proposal.classes, np.dtype("uint8")),
                         (proposal.confidence, np.dtype("uint8")),
                         (proposal.inside, np.dtype("bool"))):
        if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape:
            raise ValueError("Invalid native proposal array dtype or shape")
    if (np.any(proposal.classes[~proposal.inside] != 0)
            or not set(np.unique(proposal.classes)).issubset({0, *proposal.source["layer_ids"]})):
        raise ValueError("Proposal contains outside or unknown layer assignments")
    if (proposal.stroke_count != 0 or proposal.merged_regions != 0
            or proposal.min_region_pixels != 0 or proposal.snap_px != 0
            or proposal.polygon_count != len(polygons)):
        raise ValueError("Polygons-only proposal contains classifier processing metadata")
    if proposal.digest != _digest(proposal.source, proposal.classes, proposal.confidence, proposal.inside):
        raise ValueError("Polygons-only proposal digest mismatch")
    if raster:
        expected = _arrays(proposal.source)
        if any(not np.array_equal(a, b) for a, b in zip(
                (proposal.classes, proposal.confidence, proposal.inside), expected)):
            raise ValueError("Proposal arrays do not match the exact polygons and frame")


def make_proposal(image_or_image_size, session, frame=None, progress=None):
    """Rasterize only polygons at photograph resolution; retain all inside gaps."""
    start = time.monotonic()
    size = getattr(image_or_image_size, "image_size", image_or_image_size)
    source = _source(session, frame, tuple(size))
    lp._progress(progress, "Drawing the polygons at full resolution…")
    classes, confidence, inside = _arrays(source)
    counts = lp._region_counts(classes)
    layers = [dict(layer_id=row.layer_id, name=row.name,
                   pixels=counts.get(row.layer_id, (0, 0))[0],
                   regions=counts.get(row.layer_id, (0, 0))[1]) for row in session.layers]
    return PolygonsOnlyProposal(
        classes=classes, confidence=confidence, inside=inside, source=source,
        image_size=tuple(size), digest=_digest(source, classes, confidence, inside),
        created_utc=lp.utc_now(), elapsed_s=time.monotonic() - start,
        frame=deepcopy(source["frame"]), layers=layers, polygon_count=len(session.polygons))


def proposal_current(proposal, session, frame):
    if proposal is None:
        return False
    if not is_polygons_only(proposal):
        return proposal.digest == lp.proposal_digest(session, frame)
    try:
        return proposal.source == _source(session, frame, tuple(proposal.image_size))
    except (TypeError, ValueError, OverflowError):
        return False


def checked_current_session(proposal, session, frame):
    """Restore verified exact vertices only when every current input agrees.

    Current files written by this method preserve full precision. A stale or
    legacy-rounded session is never taken as permission to adopt another draft.
    """
    if not is_polygons_only(proposal):
        return session
    if not proposal_current(proposal, session, frame):
        raise ValueError("The current draft differs from the polygons-only proposal; propose again")
    restored = deepcopy(session)
    restored.polygons = [lp.Polygon(layer, [tuple(p) for p in points])
                         for layer, points in proposal.source["polygons"]]
    return restored


def save_current_session(project, session, *, actor=""):
    """Save this method's current draft with exact vertices, using the old schema."""
    session.updated_utc = lp.utc_now()
    session.updated_by = str(actor or session.updated_by or "")
    payload = session.to_dict()
    payload["polygons"] = [[int(p.layer_id), [[float(r), float(c)] for r, c in p.points]]
                           for p in session.polygons]
    payload["strokes"] = [[int(s.layer_id), float(s.radius_px),
                           [[float(r), float(c)] for r, c in s.points]] for s in session.strokes]
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path = lp.session_path(project, session.run_id)
    lp._atomic_write_text(path, text)
    return path


def save_proposal(proposal, path):
    if not is_polygons_only(proposal):
        return proposal.save(path)
    _validate(proposal, raster=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(schema=1, format=PROPOSAL_FORMAT, method=METHOD,
                image_size=list(proposal.image_size), digest=proposal.digest,
                created_utc=proposal.created_utc, elapsed_s=proposal.elapsed_s,
                frame=proposal.frame, layers=proposal.layers,
                polygon_count=proposal.polygon_count, source=proposal.source,
                stroke_count=0, merged_regions=0, min_region_pixels=0, snap_px=0.)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, classes=proposal.classes, confidence=proposal.confidence,
                            inside=proposal.inside, meta=np.asarray(_json_bytes(meta).decode("utf8")))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _meta(path):
    # Inspect metadata before allocating image arrays; duplicates are ambiguous.
    with zipfile.ZipFile(_preset_io.source(path)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "meta.npy" not in names:
            raise ValueError("Invalid proposal archive members")
        if archive.getinfo("meta.npy").file_size > _MAX_META:
            raise ValueError("Proposal metadata exceeds the bounded read")
        with archive.open("meta.npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError("Unsupported proposal metadata header version")
            if shape != () or dtype.kind != "U" or dtype.itemsize > _MAX_META:
                raise ValueError("Invalid proposal metadata array header")
        with archive.open("meta.npy") as stream:
            value = np.load(_preset_io.source(stream), allow_pickle=False)
        if value.shape != () or value.dtype.kind != "U":
            raise ValueError("Invalid proposal metadata array")
        meta = json.loads(str(value))
        if not isinstance(meta, dict):
            raise ValueError("Invalid proposal metadata")
        if meta.get("format") == PROPOSAL_FORMAT:
            if set(names) != {"meta.npy", "classes.npy", "confidence.npy", "inside.npy"}:
                raise ValueError("Invalid polygons-only archive closure")
            size = _size(meta.get("image_size"))
            for name in ("classes.npy", "confidence.npy", "inside.npy"):
                if archive.getinfo(name).file_size > size[0] * size[1] + 65536:
                    raise ValueError("Proposal array exceeds its declared native shape")
                with archive.open(name) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError("Unsupported proposal array header version")
                expected = np.dtype("bool") if name == "inside.npy" else np.dtype("uint8")
                if shape != (size[1], size[0]) or dtype != expected:
                    raise ValueError("Proposal array header dtype or native shape differs")
        return meta


def load_proposal(path):
    path = Path(path)
    meta = _meta(path)
    if meta.get("format") == lp.LAYER_PROPOSAL_FORMAT:
        return lp.LayerProposal.load(path)
    if (meta.get("format") != PROPOSAL_FORMAT or type(meta.get("schema")) is not int
            or meta.get("schema") != 1
            or meta.get("method") != METHOD):
        raise ValueError("Unsupported polygons-only proposal file")
    with np.load(_preset_io.source(path), allow_pickle=False) as archive:
        proposal = PolygonsOnlyProposal(
            classes=archive["classes"], confidence=archive["confidence"], inside=archive["inside"],
            image_size=_size(meta["image_size"]), digest=meta["digest"],
            source=meta["source"], frame=meta["frame"], layers=meta["layers"],
            created_utc=meta["created_utc"], elapsed_s=meta["elapsed_s"],
            polygon_count=meta["polygon_count"], stroke_count=meta["stroke_count"],
            merged_regions=meta["merged_regions"], min_region_pixels=meta["min_region_pixels"],
            snap_px=meta["snap_px"])
    _validate(proposal, raster=True)
    return proposal


def corrected_classes(proposal, corrections, *, layer_ids=None):
    if not is_polygons_only(proposal):
        return lp.corrected_classes(proposal, corrections, layer_ids=layer_ids)
    from scipy import ndimage as ndi

    known = set(proposal.source["layer_ids"]) if layer_ids is None else set(layer_ids)
    if not known.issubset(set(proposal.source["layer_ids"])):
        raise ValueError("Corrections refer to layers outside the proposal identity")
    result = proposal.classes.copy()
    decided = np.zeros(result.shape, bool)
    skipped = []
    for index, correction in enumerate(corrections):
        if (correction.kind not in lp.CORRECTION_KINDS or not correction.points
                or (correction.kind != "restore" and correction.layer_id not in known)):
            skipped.append(index)
            continue
        if correction.kind == "fill":
            seed = lp._grid_seed(correction.points[0], result.shape, (1., 1.))
            if (seed is None or not proposal.inside[seed]
                    or int(result[seed]) != correction.replaced_layer):
                skipped.append(index)
                continue
            components, _ = ndi.label((result == result[seed]) & proposal.inside,
                                      structure=ndi.generate_binary_structure(2, 1))
            region = components == components[seed]
            result[region] = correction.layer_id
            decided[region] = True
        else:
            result, decided, rejected = lp.apply_corrections(
                proposal.classes, [correction], (1., 1.), inside=proposal.inside,
                layer_ids=known, start=(result, decided))
            if rejected:
                skipped.append(index)
    return result, decided, skipped


def gap_count(proposal, corrections=(), layer_ids=None):
    if not is_polygons_only(proposal):
        return 0
    classes, _decided, _skipped = corrected_classes(proposal, corrections, layer_ids=layer_ids)
    return int(np.count_nonzero(proposal.inside & (classes == lp.OUTSIDE)))


def painted_layers(session, proposal, *, frame, corrections=(), progress=None,
                   require_complete=True):
    if not is_polygons_only(proposal):
        from . import layer_painting_manual as manual
        return manual.painted_layers(session, proposal, frame=frame,
                                     corrections=corrections, progress=progress)
    _validate(proposal, raster=True)
    if not proposal_current(proposal, session, frame):
        raise ValueError("The exact polygons, source image or full rig frame changed; propose again")
    lp._progress(progress, "Applying the hand corrections at full resolution…")
    classes, _decided, skipped = corrected_classes(
        proposal, corrections, layer_ids={row.layer_id for row in session.layers})
    if skipped:
        raise ValueError("Some hand corrections no longer apply; review the corrections")
    gaps = int(np.count_nonzero(proposal.inside & (classes == lp.OUTSIDE)))
    if gaps and require_complete:
        raise ValueError(f"The polygons-only proposal has {gaps} unassigned pixels inside the rig frame")
    return classes


def _number_regions(session, classes):
    """The original LP component numbering and physical region properties."""
    from scipy import ndimage as ndi

    structure = ndi.generate_binary_structure(2, 1)
    present = [int(v) for v in np.unique(classes) if v != lp.OUTSIDE]
    order = lp.facies_ids(session, present)
    labels = np.zeros(classes.shape, np.uint16)
    regions = []
    next_label = 1
    for layer in session.layers:
        if layer.layer_id not in present:
            continue
        components, count = ndi.label(classes == layer.layer_id, structure=structure)
        if next_label + count > np.iinfo(np.uint16).max:
            raise ValueError("Too many regions; correct small islands first")
        sizes = np.bincount(components.ravel(), minlength=count + 1)
        entry = session.facies.get(layer.facies, lp.FaciesEntry())
        porous = entry.role == "porous_medium"
        table = np.zeros(count + 1, np.uint16)
        for rank, component in enumerate(sorted(range(1, count + 1),
                                                key=lambda v: -int(sizes[v])), 1):
            table[component] = next_label
            regions.append(dict(label_id=next_label, facies_id=order[layer.facies],
                name=layer.facies, layer=layer.name, layer_id=layer.layer_id,
                region_index=rank, pixels=int(sizes[component]), analysis_role=entry.role,
                porosity=entry.effective_porosity(), permeability=entry.permeability if porous else None,
                enabled=False if entry.role == "colorchecker" else entry.role != "excluded"))
            next_label += 1
        labels += table[components]
    if np.any(classes == lp.OUTSIDE):
        components, count = ndi.label(classes == lp.OUTSIDE, structure=structure)
        if next_label + count > np.iinfo(np.uint16).max:
            raise ValueError("Too many regions; correct small islands first")
        sizes = np.bincount(components.ravel(), minlength=count + 1)
        table = np.zeros(count + 1, np.uint16)
        for rank, component in enumerate(range(1, count + 1), 1):
            table[component] = next_label
            regions.append(dict(label_id=next_label, facies_id=len(order) + 1,
                name=lp.OUTSIDE_FACIES, layer=lp.OUTSIDE_FACIES, layer_id=lp.OUTSIDE,
                region_index=rank, pixels=int(sizes[component]), analysis_role="frame",
                porosity=lp.OUTSIDE_POROSITY, permeability=None, enabled=False))
            next_label += 1
        labels += table[components]
    return labels, regions


def painted_label_map(session, proposal, *, frame, corrections=(), progress=None):
    if not is_polygons_only(proposal):
        return lp.painted_label_map(session, proposal, frame=frame,
                                    corrections=corrections, progress=progress)
    classes = painted_layers(session, proposal, frame=frame,
                             corrections=corrections, progress=progress)
    lp._progress(progress, "Numbering the regions…")
    return _number_regions(session, classes)


def check_ready_for_revision(session, proposal, frame, reference, corrections=()):
    if not is_polygons_only(proposal):
        return lp.check_ready_for_revision(session, proposal, frame, reference, corrections)
    problems = []
    if reference is None or reference.baseline_image is None:
        problems.append("The reference run has no baseline photograph.")
    if lp._rig_size(frame, reference) is None:
        problems.append("Enter the rig's inner width and height; the revision needs the rig size.")
    try:
        classes = painted_layers(session, proposal, frame=frame, corrections=corrections)
    except ValueError as error:
        problems.append(str(error))
        return problems
    present = [int(v) for v in np.unique(classes) if v != lp.OUTSIDE]
    missing = []
    for name in lp.facies_ids(session, present):
        entry = session.facies.get(name, lp.FaciesEntry())
        if entry.role == "porous_medium" and not (
                entry.porosity is not None and math.isfinite(entry.porosity)
                and 0 < entry.porosity <= 1 and entry.permeability is not None
                and math.isfinite(entry.permeability) and entry.permeability > 0):
            missing.append(name)
    if missing:
        problems.append("Enter porosity (0-1) and a positive permeability for " + ", ".join(missing) + ".")
    if len(_number_regions(session, classes)[1]) > lp.NATIVE_MAX_REGIONS:
        problems.append(f"The proposal exceeds native Setup's {lp.NATIVE_MAX_REGIONS} region limit; correct small islands.")
    return problems


def load_version(record):
    if record.get("proposal_method") != METHOD:
        return lp.load_version(record)
    folder = Path(str(record["folder"]))
    actual = json.loads(_preset_io.read_text(folder / "version.json",encoding="utf8"))
    if actual.get("format") != lp.LAYER_VERSION_FORMAT or actual.get("schema") != 1:
        raise ValueError("Unsupported saved version format")
    for key in ("run_id", "name", "kind", "sha256", "files", "proposal_method",
                "proposal_digest", "correction_count"):
        if actual.get(key) != record.get(key):
            raise ValueError("The selected saved version identity changed")
    if actual.get("proposal_method") != METHOD or set(actual.get("files") or {}) != set(lp.VERSION_FILES):
        raise ValueError("Invalid polygons-only saved version closure")
    for leaf in lp.VERSION_FILES:
        if lp._file_sha256(folder / leaf) != actual["files"][leaf]:
            raise ValueError(f"The saved version's {leaf} has changed")
    raw_session = json.loads(_preset_io.read_text(folder / "session.json",encoding="utf8"))
    session = lp.PaintingSession.from_dict(raw_session)
    proposal = load_proposal(folder / "proposal.npz")
    if not is_polygons_only(proposal):
        raise ValueError("The saved version method and proposal format differ")
    payload = json.loads(_preset_io.read_text(folder / "corrections.json",encoding="utf8"))
    if (payload.get("format") != lp.LAYER_CORRECTIONS_FORMAT
            or payload.get("run_id") != actual["run_id"] or session.run_id != actual["run_id"]):
        raise ValueError("Invalid saved correction or run identity")
    corrections = [lp.Correction.from_dict(row) for row in payload.get("corrections") or []]
    # The ordinary session serialization rounds vertices. Its original bytes
    # remain the map SHA input; the new proposal separately preserves exact ones.
    class SavedSession:
        def to_dict(self):
            return deepcopy(raw_session)
    if (lp.map_digest(SavedSession(), proposal, corrections) != actual["sha256"]
            or proposal.digest != actual["proposal_digest"]
            or len(corrections) != actual["correction_count"]):
        raise ValueError("The saved map content digest differs")
    source = proposal.source
    if [polygon.to_list() for polygon in session.polygons] != [
            lp.Polygon(layer, [tuple(p) for p in points]).to_list()
            for layer, points in source["polygons"]]:
        raise ValueError("The saved session and exact source polygons differ")
    session.polygons = [lp.Polygon(layer, [tuple(p) for p in points])
                        for layer, points in source["polygons"]]
    saved_frame = _saved_frame(source["frame"], proposal.image_size)
    if not proposal_current(proposal, session, saved_frame):
        raise ValueError("The saved session and proposal source identities differ")
    return session, proposal, corrections


def version_layers(record, *, frame=None, progress=None):
    if record.get("proposal_method") != METHOD:
        from . import layer_painting_manual as manual
        return manual.version_layers(record, frame=frame, progress=progress)
    session, proposal, corrections = load_version(record)
    if frame is None:
        frame = _saved_frame(proposal.source["frame"], proposal.image_size)
    return session, painted_layers(session, proposal, frame=frame,
                                   corrections=corrections, progress=progress)


def _checked_approval_history(stream):
    """Read bounded, valid history, including compatible library events."""
    raw = stream.read(_MAX_APPROVAL_HISTORY + 1)
    if len(raw) > _MAX_APPROVAL_HISTORY:
        raise ValueError("The approvals history exceeds the metadata limit")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate approvals history key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite approvals history value")

    try:
        for line in raw.decode("utf8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line, object_pairs_hook=unique, parse_constant=invalid)
            if not isinstance(row, dict) or row.get("format") != lp.LAYER_APPROVAL_FORMAT:
                raise ValueError("Invalid approvals history record")
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("The approvals history is unreadable or invalid") from error
    return raw


def approve_version(record, *, reviewer=None, now=None):
    if record.get("proposal_method") != METHOD:
        return lp.approve_version(record, reviewer=reviewer, now=now)
    folder = Path(str(record["folder"]))
    history = folder.parent / "approvals.jsonl"
    try:
        with _preset_io.open_read(history, "rb") as stream:
            _checked_approval_history(stream)
    except FileNotFoundError:
        pass
    # Exact map completeness and applicability are approval gates, not merely
    # display warnings. No files in the saved version are modified.
    version_layers(record)
    approval = dict(format=lp.LAYER_APPROVAL_FORMAT, run_id=record["run_id"],
        version=folder.name, name=record["name"], kind=record["kind"],
        sha256=record["sha256"], files=dict(record["files"]),
        reviewer=(str(reviewer).strip() or None) if reviewer else None,
        approved_utc=now or lp.utc_now())
    from .pipeline.job_queue import retry_transient
    line = (json.dumps(approval, ensure_ascii=False, sort_keys=True) + "\n").encode("utf8")
    def append():
        # Revalidate the actual append handle on every attempt, including
        # retries after a transient write failure or changed history.
        with history.open("a+b") as stream:
            stream.seek(0)
            raw = _checked_approval_history(stream)
            separator = b"\n" if raw and not raw.endswith(b"\n") else b""
            if len(raw) + len(separator) + len(line) > _MAX_APPROVAL_HISTORY:
                raise ValueError("The approvals history exceeds the metadata limit")
            stream.seek(0, os.SEEK_END)
            stream.write(separator + line)
            stream.flush()
            os.fsync(stream.fileno())
    retry_transient(append)
    return approval


def save_painted_revision(project, session, proposal, reference, *, frame,
                          actor="", progress=None, corrections=()):
    if not is_polygons_only(proposal):
        return lp.save_painted_revision(project, session, proposal, reference,
            frame=frame, actor=actor, progress=progress, corrections=corrections)
    from .geometry_facies import (GeometryRevisionStore, _save_darsia_field,
                                  default_geometry_config, validate_geometry_config)

    problems = check_ready_for_revision(session, proposal, frame, reference, corrections)
    if problems:
        raise ValueError(" ".join(problems))
    size_px = lp.photo_size(reference.baseline_image,
                           camera_u=lp.reference_camera_u(project, reference))
    if tuple(size_px) != tuple(proposal.image_size):
        raise ValueError("The baseline photograph and proposal native image sizes differ")
    width_m, height_m = lp._rig_size(frame, reference)
    labels, regions = painted_label_map(session, proposal, frame=frame,
                                        progress=progress, corrections=corrections)
    if len(regions) > lp.NATIVE_MAX_REGIONS:
        raise ValueError(f"Native Setup supports at most {lp.NATIVE_MAX_REGIONS} regions")
    injectors, skipped = lp.revision_injectors(project, frame, (width_m, height_m))
    destination = (Path(project.root) / "overrides" / "geometry_facies" / "painted"
                   / f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}")
    destination.mkdir(parents=True, exist_ok=False)
    labels_path, depth_path = destination / "labels.npz", destination / "depth.npz"
    _save_darsia_field(labels_path, labels, width_m=width_m, height_m=height_m)
    _save_darsia_field(depth_path, np.full(labels.shape, lp.PLACEHOLDER_DEPTH_M, np.float32),
                        width_m=width_m, height_m=height_m)
    (destination / "regions.json").write_text(json.dumps(regions, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    config = default_geometry_config(project)
    generation = dict(method=METHOD, status="painted layers reviewed in the layer view",
        reference_run=reference.run_id, source_image_path=str(reference.baseline_image),
        session_path=str(lp.session_path(project, session.run_id)),
        proposal_digest=proposal.digest, proposal_created_utc=proposal.created_utc,
        polygon_count=len(session.polygons), stroke_count=0, automatic_edge_fill=False,
        layers=[dict(layer_id=row.layer_id, name=row.name, facies=row.facies) for row in session.layers],
        rig_frame=deepcopy(proposal.source["frame"]), injection_points_left_out=skipped)
    if corrections:
        generation.update(corrections_path=str(lp.corrections_path(project, session.run_id)),
            corrections_sha256=lp.corrections_digest(corrections), correction_count=len(corrections))
    config.update(baseline_path=str(reference.baseline_image), depth_path=str(depth_path),
        labels_path=str(labels_path), display_image_path=str(reference.baseline_image),
        draft_source_image_path=str(reference.baseline_image), width_m=float(width_m),
        height_m=float(height_m), grouping_mode="shared_facies",
        permeability_unit=session.permeability_unit, regions=regions, injectors=injectors,
        depth_note="Constant placeholder; native Setup takes the depth from its own [depth] section.",
        facies_generation=generation)
    lp._progress(progress, "Validating…")
    validation = validate_geometry_config(config)
    errors = [str(row.get("message") or row.get("code") or "")
              for row in validation.get("issues") or [] if row.get("severity") == "error"]
    if errors:
        raise ValueError("The revision is not valid: " + "; ".join(errors))
    lp._progress(progress, "Saving the revision…")
    used = {int(row["layer_id"]) for row in regions}
    names = ", ".join(row.name for row in session.layers if row.layer_id in used)
    return GeometryRevisionStore(project).save_revision(config, actor=actor,
        reason=f"Painted layers of run {reference.run_id}: {names}", validation=validation)


__all__ = ["METHOD", "PROPOSAL_FORMAT", "PolygonsOnlyProposal", "is_polygons_only",
           "make_proposal", "proposal_current", "checked_current_session", "save_current_session",
           "save_proposal", "load_proposal",
           "corrected_classes", "gap_count", "painted_layers", "painted_label_map",
           "check_ready_for_revision", "load_version", "version_layers", "approve_version",
           "save_painted_revision"]
