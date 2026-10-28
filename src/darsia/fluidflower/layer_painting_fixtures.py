"""Explicit fixture footprints and drawn layers beneath them.

The analysis map keeps a ColorChecker excluded. Its underlay is a separate,
version-bound simulator input: only explicit polygons count as drawn, never a
classifier, strokes or a frame-edge fill. Nothing here changes a session,
approves a version, adopts a revision or runs a scientific worker.
"""
from __future__ import annotations

from darsia.fluidflower import preset_wasserstein_inputs as _preset_io

from copy import copy, deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import layer_painting as lp
from .series_geometry import RigFrame

FORMAT = "darsia-painted-fixture-underlay-v1"
COLORCHECKER = "colorchecker"


def _json(value) -> bytes:
    """Encode stable finite metadata for fixture identity."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _array_sha(array) -> str:
    """Hash exact contiguous array bytes, dtype and shape."""
    value = np.ascontiguousarray(array)
    return hashlib.sha256(_json([value.dtype.str, list(value.shape)]) + value.tobytes()).hexdigest()


@dataclass(frozen=True)
class FixtureUnderlay:
    analysis_layers: np.ndarray
    fixture_mask: np.ndarray
    drawn_layers: np.ndarray
    provenance: dict[str, Any]

    def digest(self) -> str:
        """Return the complete immutable fixture-underlay identity."""
        return hashlib.sha256(_json(dict(
            provenance=self.provenance, analysis=_array_sha(self.analysis_layers),
            fixture=_array_sha(self.fixture_mask), drawn=_array_sha(self.drawn_layers),
        ))).hexdigest()


def _drawn_polygons(session, polygons, shape):
    # The current display/manual priority is explicit and never alters vertices.
    """Rasterize explicit non-fixture polygons with the current drawing priority."""
    from .painting_layer_priority import rasterize_polygons
    subset = copy(session)
    subset.polygons = list(polygons)
    return rasterize_polygons(subset, shape, (1.0, 1.0))


def fixture_underlay(session: lp.PaintingSession, layers: np.ndarray, *,
                     frame: RigFrame | None, version: Mapping | None = None) -> FixtureUnderlay | None:
    """Preview inputs from the actual polygons; a preview is not adoptable.

    A ColorChecker polygon remains a fixture even when a smaller sand polygon
    is drawn over it. Every other role's polygons are rasterised separately,
    with the existing smaller-polygon-wins rule. Old excluded facies are never
    fixtures and remain excluded when drawn beneath one.
    """
    roles = {row.layer_id: session.facies.get(row.facies, lp.FaciesEntry()).role
             for row in session.layers}
    fixture_ids = [key for key, role in roles.items() if role == COLORCHECKER]
    if not fixture_ids:
        return None
    original = np.asarray(layers)
    if original.ndim != 2 or original.dtype.kind not in "iu" or not all(original.shape):
        raise ValueError("A fixture underlay needs a nonempty integer layer map")
    if not session.image_size or original.shape != tuple(reversed(session.image_size)):
        raise ValueError("Drawn fixture polygons need the full reference photograph's pixel grid")
    if np.any(original < 0) or np.any(original > 255):
        raise ValueError("Painting layer ids must fit their existing uint8 map")
    inside = lp.frame_mask(frame, original.shape, (1.0, 1.0))
    explicit = _drawn_polygons(session,
        [p for p in session.polygons if p.layer_id in fixture_ids], original.shape)
    fixture = inside & (np.isin(original, fixture_ids) | (explicit > 0))
    if not fixture.any():
        return None
    analysis = np.asarray(original, dtype=np.uint8).copy()
    analysis[(explicit > 0) & inside] = explicit[(explicit > 0) & inside]
    analysis[~inside] = lp.OUTSIDE
    underlying = _drawn_polygons(session,
        [p for p in session.polygons if p.layer_id in roles and roles[p.layer_id] != COLORCHECKER],
        original.shape)
    drawn = np.where(fixture, underlying, 0).astype(np.uint8)
    provenance = dict(format=FORMAT, grid="reference camera pixels",
        roles={str(key): value for key, value in sorted(roles.items())},
        frame=frame.to_dict() if frame is not None else None,
        session_sha256=hashlib.sha256(_json(session.to_dict())).hexdigest(),
        approved_layers_sha256=_array_sha(np.asarray(original, dtype=np.uint8)),
        version=deepcopy(dict(version)) if version is not None else None,
        drawn_source="all non-ColorChecker polygons; explicit Top priority then smaller polygon wins; no classifier or strokes")
    return FixtureUnderlay(analysis, fixture, drawn, provenance)


def version_fixture_underlay(record: Mapping, *, frame: RigFrame | None) -> FixtureUnderlay | None:
    """Read and verify the frozen version before making adoption inputs."""
    from . import layer_painting_manual as manual

    files = dict(record.get("files") or {})
    if record.get("proposal_method") == "polygons_only_v1":
        # This format deliberately cannot pass the worker's legacy loader.
        # Its own reader verifies method, files, map digest, full frame and
        # corrections, and refuses uncovered interior pixels.
        from . import layer_painting_polygons_only as polygons_only
        session, layers = polygons_only.version_layers(record, frame=frame)
        digest = record["sha256"]
    elif manual.is_manual_version(record):
        if set(files) != set(manual.MANUAL_VERSION_FILES):
            raise ValueError("The manual version must bind all its original files")
        session, saved = manual.load_manual_version(record)
        if saved.uncovered_pixels():
            raise ValueError("The manual map has uncovered pixels inside the rig frame")
        digest = manual.manual_map_digest(session, saved)
        layers = saved.classes.copy()
        if not np.array_equal(saved.inside, lp.frame_mask(frame, layers.shape, (1.0, 1.0))):
            raise ValueError("The current frame differs from the saved manual map's inside mask")
    else:
        if set(files) != {"session.json", "proposal.npz", "corrections.json"}:
            raise ValueError("The map version must bind its session, proposal and corrections")
        session, proposal, corrections = lp.load_version(record)
        if proposal.frame is not None and proposal.frame != (frame.to_dict() if frame is not None else None):
            raise ValueError("The current frame differs from the frozen proposal frame")
        if proposal.digest != lp.proposal_digest(session, frame):
            raise ValueError("The frozen proposal does not bind this session and frame")
        digest = lp.map_digest(session, proposal, corrections)
        layers = manual.painted_layers(session, proposal, frame=frame, corrections=corrections)
    if record.get("sha256") != digest:
        raise ValueError("The frozen version does not have its recorded map SHA-256")
    return fixture_underlay(session, layers, frame=frame, version=record)


def checked_for_adoption(underlay: FixtureUnderlay, layers, version: Mapping, *,
                         frame: RigFrame | None) -> FixtureUnderlay:
    """Recompute from the bound frozen files and frame; accept no free mask."""
    if (frame is None or not frame.complete() or not frame.has_size()
            or not np.isfinite([frame.width_m, frame.height_m]).all()
            or min(frame.width_m, frame.height_m) <= 0):
        raise ValueError("Fixture adoption needs an independently supplied full rig frame")
    if (version.get("approval") or {}).get("sha256") != version.get("sha256"):
        raise ValueError("The fixture map version needs approval of its own SHA-256")
    record = underlay.provenance.get("version")
    if not isinstance(record, dict) or record.get("sha256") != version.get("sha256"):
        raise ValueError("A fixture underlay needs the exact approved map version")
    frame_data = underlay.provenance.get("frame")
    if frame_data != (frame.to_dict() if frame is not None else None):
        raise ValueError("The fixture underlay differs from the independently supplied frame")
    expected = version_fixture_underlay(record, frame=frame)
    if expected is None or expected.digest() != underlay.digest():
        raise ValueError("Fixture masks do not replay their frozen session and frame")
    if not np.array_equal(np.asarray(layers), expected.analysis_layers):
        raise ValueError("The analysis map is not the version's excluded fixture map")
    return expected


def continue_fixture_underlay(layers, roles: Mapping[int, str], fixture, drawn, y_m) -> dict:
    """Continue only undrawn fixture cells on an explicitly supplied grid."""
    from .published_dataset.simulator_geometry import continue_layers

    labels = np.asarray(layers)
    hidden = np.asarray(fixture, dtype=bool)
    painted = np.asarray(drawn)
    heights = np.asarray(y_m, dtype=np.float64)
    if labels.ndim != 2 or hidden.shape != labels.shape or painted.shape != labels.shape:
        raise ValueError("Fixture underlay arrays must share one two-dimensional grid")
    if heights.shape != labels.shape or not np.isfinite(heights).all():
        raise ValueError("Explicit finite cell heights on the same grid are required")
    porous = [int(key) for key, role in roles.items() if role in {"porous_medium", "seal"}]
    explicit = hidden & (painted != 0)
    allowed = [int(key) for key, role in roles.items() if role in lp.LAYER_ROLES and role != COLORCHECKER]
    if np.any((painted != 0) & ~hidden) or np.any(explicit & ~np.isin(painted, allowed)):
        raise ValueError("Drawn layers must have an explicit non-ColorChecker role inside the fixture")
    seed = labels.copy()
    seed[explicit] = painted[explicit]
    # An excluded or free-water drawing is protected, but never donates sand.
    known = (np.isin(labels, porous) & ~hidden) | (explicit & np.isin(painted, porous))
    result, filled = continue_layers(seed, known, hidden & ~explicit, heights)
    unfilled = hidden & ~explicit & ~filled
    return dict(layers=result, drawn_behind=explicit, fixture_filled=filled,
                unfilled=unfilled, counts=dict(drawn=int(explicit.sum()),
                    continued=int(filled.sum()), unfilled=int(unfilled.sum())))


def project_underlay(underlay: FixtureUnderlay, reference, target, analysis_layers, *,
                     frame_boundary_samples=None) -> FixtureUnderlay:
    """Carry drawn pixels without edge filling them; never invent drawings."""
    import cv2
    from .map_adoption import native_transport

    native, shape = native_transport(target)
    matrix = native @ target.metric_to_grid @ np.linalg.inv(reference.metric_to_grid)
    if not np.isfinite(matrix).all():
        raise ValueError("The fixture transport must be finite")
    drawn = cv2.warpPerspective(underlay.drawn_layers, matrix, (shape[1], shape[0]),
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if frame_boundary_samples is not None:
        target_rc, source_rc = (frame_boundary_samples[key] for key in ('target_rc', 'source_rc'))
        drawn[tuple(target_rc.T)] = underlay.drawn_layers[tuple(source_rc.T)]
    analysis = np.asarray(analysis_layers, dtype=np.uint8)
    if analysis.shape != shape:
        raise ValueError("The projected underlay is not on the native label grid")
    roles = underlay.provenance["roles"]
    fixture = np.isin(analysis, [int(key) for key, value in roles.items() if value == COLORCHECKER])
    drawn[~fixture] = 0
    provenance = deepcopy(underlay.provenance)
    provenance.update(grid="rig-corrected native pixels", source_underlay_sha256=underlay.digest(),
                      composition_reference_u_to_native=matrix.tolist(), run_id=target.run_id)
    if frame_boundary_samples is not None:
        from .photo_edge_continuation import FRAME_RASTER
        provenance['frame_raster'] = FRAME_RASTER
    return FixtureUnderlay(analysis.copy(), fixture, drawn, provenance)


def save_underlay(path: Path, underlay: FixtureUnderlay) -> dict:
    """Write a new bound artifact; existing files are never overwritten."""
    path = Path(path)
    metadata = dict(format=FORMAT, provenance=underlay.provenance, digest=underlay.digest())
    with path.open("xb") as stream:
        np.savez_compressed(stream, analysis_layers=underlay.analysis_layers,
                            fixture_mask=underlay.fixture_mask, drawn_layers=underlay.drawn_layers,
                            metadata=np.asarray(_json(metadata).decode("utf-8")))
    raw = _preset_io.read_bytes(path)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def read_underlay(bound: Mapping) -> FixtureUnderlay:
    """Read and validate a bound immutable fixture-underlay artifact."""
    path = Path(str(bound["path"]))
    raw = _preset_io.read_bytes(path)
    if len(raw) != bound.get("size_bytes") or hashlib.sha256(raw).hexdigest() != bound.get("sha256"):
        raise ValueError("The bound fixture underlay changed")
    with np.load(_preset_io.source(path), allow_pickle=False) as saved:
        if set(saved.files) != {"analysis_layers", "fixture_mask", "drawn_layers", "metadata"}:
            raise ValueError("The fixture underlay archive has unknown or missing fields")
        metadata = json.loads(str(saved["metadata"]))
        analysis, fixture, drawn = (np.asarray(saved[key]) for key in
                                    ("analysis_layers", "fixture_mask", "drawn_layers"))
    if (metadata.get("format") != FORMAT or analysis.dtype != np.uint8 or drawn.dtype != np.uint8
            or fixture.dtype != np.bool_ or analysis.ndim != 2
            or analysis.shape != fixture.shape or analysis.shape != drawn.shape):
        raise ValueError("The fixture underlay has an invalid format or grid")
    result = FixtureUnderlay(analysis, fixture, drawn, metadata["provenance"])
    if result.digest() != metadata.get("digest"):
        raise ValueError("The fixture underlay content digest differs")
    return result


def _bound_json(bound: Mapping) -> dict:
    """Read a JSON descriptor only after its exact file binding is verified."""
    path = Path(str(bound["path"]))
    raw = _preset_io.read_bytes(path)
    if (hashlib.sha256(raw).hexdigest() != bound.get("sha256")
            or ("size_bytes" in bound and len(raw) != bound["size_bytes"])):
        raise ValueError("A fixture underlay's bound metadata changed")
    return json.loads(raw)


def adopted_underlay(config: Mapping, run: str, labels: np.ndarray) -> dict | None:
    """Read only the underlay named by the original painted preparation.

    The exact existing registration schema stays unchanged. Its measurement
    descriptor binds the index which names both underlays and the models; the
    native masks must replay the frozen version and original transport.
    """
    records = config.get("painted_fixture_underlays") or {}
    row = next((r for r in (config.get("run_geometry") or {}).get("runs") or []
                if str(r.get("run_id", "")).casefold() == run.casefold()), None)
    registration = (row or {}).get("registration") or {}
    if run not in records:
        if records:
            raise ValueError("No bound fixture underlay exists for this run")
        if registration.get("kind") in {"painted-map-placement-v1", "painted-map-placement-v2"}:
            index = _bound_json(registration["measurement"])
            if index.get("source_fixture_underlay") is not None:
                raise ValueError("The adopted revision lost its prepared fixture underlay binding")
        return None
    bound = records[run]
    if set(bound) != {"artifact", "source", "preparation", "layer_to_facies"}:
        raise ValueError("The adopted fixture underlay has unknown or missing fields")
    if (registration.get("kind") not in {"painted-map-placement-v1", "painted-map-placement-v2"}
            or registration.get("measurement") != bound["preparation"]):
        raise ValueError("The fixture underlay has no matching original painted registration")
    index = _bound_json(bound["preparation"])
    folder = Path(bound["preparation"]["path"]).parent
    own = next((r for r in index.get("runs") or [] if r.get("run_id") == run), {})

    def absolute(value):
        value = dict(value or {})
        value["path"] = str((folder / value["path"]).resolve())
        return value

    if (absolute(own.get("fixture_underlay")) != bound["artifact"]
            or absolute(index.get("source_fixture_underlay")) != bound["source"]):
        raise ValueError("The adopted fixture underlay is not the one its preparation names")
    source = read_underlay(bound["source"])
    frame_data = index.get("fixture_frame")
    frame = RigFrame.from_dict(frame_data) if frame_data is not None else None
    checked_for_adoption(source, source.analysis_layers, index["map_version"], frame=frame)
    source_map = registration.get("source_map") or {}
    raw = _preset_io.read_bytes(Path(str(source_map["path"])))
    if (len(raw) != source_map.get("size_bytes")
            or hashlib.sha256(raw).hexdigest() != source_map.get("sha256")
            or (index.get("source_map") or {}).get("sha256") != source_map.get("sha256")):
        raise ValueError("The fixture underlay's original source map changed")
    with np.load(_preset_io.source(Path(str(source_map["path"]))), allow_pickle=False) as saved:
        if not np.array_equal(saved["array"], source.analysis_layers):
            raise ValueError("The source map does not have its frozen fixture footprint")
    from types import SimpleNamespace
    from .map_adoption import native_labels
    from .rig_point_alignment import effective_model_matrix
    models = [_bound_json(registration[name]) for name in ("reference_model", "target_model")]
    reference, target = [SimpleNamespace(metric_to_grid=effective_model_matrix(model, index.get("geometry_index")),
                                        model=model, run_id=key)
                         for model, key in zip(models, (index["reference_run"], run))]
    replayed, _filled, _support, _matrix = native_labels(source.analysis_layers, reference, target)
    samples = None
    if registration.get("kind") == "painted-map-placement-v2":
        from .photo_edge_continuation import load_contract, apply_policy
        from .geometry_raster_sources import _verify
        sidecar = registration["continuation"]
        _verify(sidecar)
        contract = load_contract(sidecar["path"], run)
        if "frame_raster" in contract:
            classified = apply_policy(source.analysis_layers, contract)
            replayed, samples = classified["labels"], classified["frame_boundary_samples"]
    expected = project_underlay(source, reference, target, replayed,
                               frame_boundary_samples=samples)
    actual = read_underlay(bound["artifact"])
    if expected.digest() != actual.digest() or not np.array_equal(actual.analysis_layers, labels):
        raise ValueError("The native fixture underlay differs from its original placement replay")
    mapping = {int(key): int(value) for key, value in bound["layer_to_facies"].items()}
    regions = {int(r["label_id"]): r for r in config.get("regions") or []}
    drawn_ids = set(int(v) for v in np.unique(actual.drawn_layers) if v)
    needed = drawn_ids | set(int(v) for v in np.unique(actual.analysis_layers) if v)
    frozen_roles = actual.provenance["roles"]
    if any(key not in mapping or key not in regions or mapping[key] != int(regions[key]["facies_id"])
           or regions[key].get("analysis_role") != frozen_roles.get(str(key))
           or frozen_roles.get(str(key)) not in lp.LAYER_ROLES
           or (key in drawn_ids and frozen_roles.get(str(key)) == COLORCHECKER) for key in needed):
        raise ValueError("Every fixture and drawn layer needs its unchanged explicit role and facies assignment")
    for key in needed:
        if any(r.get("analysis_role") != frozen_roles[str(key)] for r in regions.values()
               if int(r["facies_id"]) == mapping[key]):
            raise ValueError("Different drawn roles cannot share one facies id")
    drawn = np.zeros(actual.drawn_layers.shape, np.int32)
    for key in drawn_ids:
        # Mirror _canonical_facies: per-layer revisions keep the label id as
        # their simulator category, while shared-facies revisions use the
        # explicitly assigned facies id. The frozen drawing stays layer ids.
        value = key if config.get("grouping_mode") == "per_layer" else mapping[key]
        drawn[actual.drawn_layers == key] = value
    return dict(fixture_mask=actual.fixture_mask, drawn_facies=drawn,
                drawn_mask=actual.drawn_layers != 0,
                provenance=dict(sha256=bound["artifact"]["sha256"],
                    source_sha256=bound["source"]["sha256"],
                    version_sha256=index["map_version"]["sha256"],
                    grid="rig-corrected native pixels"))


def fixture_boundary_agreement(fixture, polygons_m, *, width_m: float, height_m: float) -> dict:
    """Symmetric maximum boundary distance in mm, report only, on a native grid."""
    from scipy import ndimage
    from .published_dataset.simulator_geometry import fixture_mask

    if not polygons_m:
        return dict(status="unavailable", maximum_boundary_distance_mm=None,
                    reason="No measured fixture polygons were supplied", report_only=True)
    mask = np.asarray(fixture, bool)
    if (mask.ndim != 2 or not all(mask.shape) or not np.isfinite([width_m, height_m]).all()
            or width_m <= 0 or height_m <= 0):
        raise ValueError("Fixture comparison needs a nonempty native grid and positive metric dimensions")
    measured = fixture_mask(mask.shape, width_m, height_m, polygons_m)
    boundaries = [value & ~ndimage.binary_erosion(value) for value in (mask, measured)]
    if not all(value.any() for value in boundaries):
        return dict(status="unavailable", maximum_boundary_distance_mm=None,
                    reason="One fixture has no comparable boundary", report_only=True)
    sampling = (float(height_m) / mask.shape[0], float(width_m) / mask.shape[1])
    distance = max(float(ndimage.distance_transform_edt(~boundaries[0], sampling=sampling)[boundaries[1]].max()),
                   float(ndimage.distance_transform_edt(~boundaries[1], sampling=sampling)[boundaries[0]].max()))
    return dict(status="compared", maximum_boundary_distance_mm=1000.0 * distance, report_only=True,
                definition="symmetric maximum distance between native boundary-cell centres",
                pixel_size_mm=[1000.0 * sampling[0], 1000.0 * sampling[1]])
