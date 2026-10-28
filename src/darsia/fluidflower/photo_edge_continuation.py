"""Versioned authorization of existing photo-edge replication for new placements.

The categorical carrier, original filled provenance, and original support are
unchanged. ``support`` means a nonzero label originating inside the source
photograph. Authorized continuation has separate effective coverage; it grants
no optical/CO2 support and never changes a saved map or adopts a revision.
"""
from __future__ import annotations

from darsia.fluidflower import preset_wasserstein_inputs as _preset_io

from copy import deepcopy
from collections import OrderedDict
import os
from threading import RLock
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib

import numpy as np

FORMAT = "darsia-photo-edge-continuation-v1"
SIDECAR_FORMAT = "darsia-photo-edge-continuation-sidecar-v2"
POLICY = "nearest-photo-edge-to-target-frame-v1"
FRAME_RASTER = "physical-frame-centres-boundary-nearest-v1"
REGISTRATION = "painted-map-placement-v2"
MAX_BYTES = 16 * 1024 * 1024
_NATIVE_PROOFS = OrderedDict()
_NATIVE_PROOFS_LOCK = RLock()
_NATIVE_PROOFS_PID = os.getpid()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _snapshot(path):
    path = Path(path).resolve()
    before = _preset_io.file_stat(path)
    if before.st_size > MAX_BYTES:
        raise ValueError("Continuation metadata exceeds the existing 16 MiB bound.")
    raw = _preset_io.read_bytes(path)
    after = _preset_io.file_stat(path)
    stamp = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    if stamp(before) != stamp(after) or len(raw) != before.st_size:
        raise ValueError("Continuation metadata changed while being captured.")
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw), text=raw.decode("utf-8-sig"))


def _document(snapshot, *, toml=False):
    # The UTF-8 text is frozen in the closed placement contract. A BOM is
    # represented explicitly so its source byte SHA can be replayed exactly.
    raw = snapshot["text"].encode("utf-8")
    if snapshot.get("bom"):
        raw = b"\xef\xbb\xbf" + raw
    if (len(raw) != snapshot["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != snapshot["sha256"]):
        raise ValueError("The frozen frame authority differs from its source byte descriptor.")
    return tomllib.loads(snapshot["text"]) if toml else json.loads(snapshot["text"])


def _freeze(path):
    result = _snapshot(path)
    # _snapshot decodes a possible BOM; retain that byte distinction.
    if _preset_io.read_bytes(Path(path)).startswith(b"\xef\xbb\xbf"):
        result["bom"] = True
    _document(result, toml=str(path).lower().endswith(".toml"))
    return result


def _shape(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(v) is not int or v < 1 for v in value):
        raise ValueError("A continuation needs an exact positive native grid.")
    return tuple(value)


def _descriptor(value):
    if (not isinstance(value, dict) or not str(value.get("path") or "")
            or re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256") or "")) is None
            or type(value.get("size_bytes")) is not int or value["size_bytes"] < 1):
        raise ValueError("A continuation needs the exact source photograph/model descriptor.")
    return {key: value[key] for key in ("path", "sha256", "size_bytes")}


def _model(model):
    frozen = _freeze(model.source)
    document = _document(frozen)
    if document != model.model:
        raise ValueError("The continuation model changed after the existing geometry reader loaded it.")
    result = dict(descriptor={k: frozen[k] for k in ("path", "sha256", "size_bytes")},
        document=document, run_id=model.run_id, shape_hw=list(_shape(model.shape_hw)),
        photograph=_descriptor(dict(model.reference_image)),
        metric_to_grid=np.asarray(model.metric_to_grid, np.float64).tolist())
    point_review = model.entry.get("_reference_point_review")
    if point_review is not None:
        result["point_review"] = deepcopy(point_review)
    return result


def _checked_model(value, *, verify_files):
    from . import map_placement as placement
    required = {"descriptor", "document", "run_id", "shape_hw", "photograph", "metric_to_grid"}
    if not isinstance(value, dict) or set(value) not in (required, required | {"point_review"}):
        raise ValueError("The continuation model closure has unexpected fields.")
    document = value["document"]
    bound = _descriptor(value["descriptor"])
    if verify_files:
        actual = _freeze(bound["path"])
        if any(actual[k] != bound[k] for k in bound) or _document(actual) != document:
            raise ValueError("The continuation's measured model changed.")
    if document.get("format", placement.GEOMETRY_FORMAT) != placement.GEOMETRY_FORMAT:
        raise ValueError("The continuation model is not an existing measured model.")
    declared = [document[key] for key in ("run_id", "run") if document.get(key) is not None]
    if any(not isinstance(run, str) or run.strip().casefold() != value["run_id"] for run in declared):
        raise ValueError("The continuation model declares another run.")
    if "H_metric_mm_to_camera_u" in document:
        matrix = placement._matrix(document["H_metric_mm_to_camera_u"], "continuation model")
    elif "H_camera_u_to_metric_mm" in document:
        matrix = np.linalg.inv(placement._matrix(document["H_camera_u_to_metric_mm"], "continuation model"))
    else:
        raise ValueError("The continuation model has no measured homography.")
    if "point_review" in value:
        from .rig_point_alignment import effective_matrix
        matrix = effective_matrix(value["point_review"], value["run_id"], document.get("reference_image") or {})
    shape = _shape(document.get("shape_hw") or (document.get("reference_image") or {}).get("shape_hw"))
    if (shape != _shape(value["shape_hw"]) or not np.array_equal(matrix, np.asarray(value["metric_to_grid"]))
            or _descriptor(dict(document.get("reference_image") or {})) != value["photograph"]):
        raise ValueError("The continuation model's transform, photograph or native grid differs.")
    return matrix, shape


def _own_frame(authority, run):
    """Replay the existing saved-frame/configuration choice from frozen bytes."""
    from . import series_geometry as geometry
    from .storage_paths import same_storage_path
    project = _document(authority["project"])
    link = (project.get("tools") or {}).get("native_series") or {}
    root = Path(str(link.get("series_root") or ""))
    if not str(link.get("series_root") or ""):
        raise ValueError("The continuation frame has no native-series authority.")
    saved = link.get("rig_frames") or {}
    matches = [payload for key, payload in saved.items() if str(key).strip().casefold() == run]
    if matches:
        if len(matches) != 1 or authority["kind"] != "saved rig frame":
            raise ValueError("The continuation's saved-frame authority is ambiguous.")
        return geometry.RigFrame.from_dict(matches[0])
    if authority["kind"] != "series configuration":
        raise ValueError("The continuation's own saved frame is missing.")
    manifest = _document(authority["series"], toml=True)
    ids = [str(v).strip().casefold() for v in (manifest.get("runs") or {}).get("ids") or []]
    if len(set(ids)) != len(ids) or run not in ids:
        raise ValueError("The continuation run is not uniquely bound by its series.")
    merged = geometry._merge(_document(authority["common"], toml=True), _document(authority["run"], toml=True))
    paths, runs = manifest.get("paths") or {}, manifest.get("runs") or {}
    original_id = ((manifest.get("runs") or {}).get("ids") or [])[ids.index(run)]
    expected = geometry._storage_path((runs.get("config_files") or {}).get(str(original_id))
        or f"{paths.get('run_configs') or 'config/runs'}/{original_id}.toml", root)
    if not same_storage_path(expected, authority["run"]["path"]):
        raise ValueError("The own frame comes from another run configuration.")
    expected_common = geometry._storage_path(paths.get("common_config", "config/common.toml"), root)
    if not same_storage_path(expected_common, authority["common"]["path"]):
        raise ValueError("The own frame comes from another common configuration.")
    data = merged.get("data") or {}
    baseline = geometry.resolve_storage_path(str(data.get("baseline") or ""))
    if not baseline.is_absolute():
        folders = [data.get("folder"), *(data.get("folders") or [])]
        folders = [geometry._storage_path(v, root) for v in folders if v]
        if not folders:
            raise ValueError("The own-frame configuration has no baseline photograph folder.")
        baseline = folders[0] / baseline
    crop = ((merged.get("corrections") or {}).get("curvature") or {}).get("crop")
    rig = merged.get("rig") or {}
    row = geometry.SeriesRunInputs(run, Path(authority["run"]["path"]), baseline_image=baseline,
        width_m=geometry._float(rig.get("width")), height_m=geometry._float(rig.get("height")), crop=crop)
    return geometry.rig_frame_from_inputs(row)


def build_contracts(project, geometry_index, reference_run, runs, known_ids, *, target_grid="camera_u",
                    run_progress=None, source_frame=None):
    """Metadata-worker operation; freeze actual own-frame authority, never photos."""
    from . import series_geometry as geometry
    from .painting_run_preview import read_preview_geometry
    models, binding = read_preview_geometry(geometry_index)
    source_run = str(reference_run).strip().casefold()
    if source_run not in models or target_grid not in {"camera_u", "native"}:
        raise ValueError("The continuation needs its exact reference model and destination grid.")
    root = Path(str(geometry.native_series_link(project).get("series_root") or ""))
    project_file = _freeze(getattr(project, "manifest_path", Path(project.root) / "series.json"))
    frozen_link = (_document(project_file).get("tools") or {}).get("native_series") or {}
    if frozen_link != geometry.native_series_link(project):
        raise ValueError("The project frame authority changed after GUI context capture.")
    rows = geometry.series_run_inputs(root, check_files=False)
    by_run = {row.run_id.strip().casefold(): row for row in rows}
    if len(by_run) != len(rows):
        raise ValueError("The own-frame series repeats a run identifier.")
    saved = frozen_link.get("rig_frames") or {}
    if source_frame is None:
        authority = dict(project=deepcopy(project_file))
        if any(str(key).strip().casefold() == source_run for key in saved):
            authority['kind'] = 'saved rig frame'
        else:
            authority.update(kind='series configuration', series=_freeze(root / 'series.toml'))
            manifest = _document(authority['series'], toml=True)
            common = geometry._storage_path((manifest.get('paths') or {}).get('common_config', 'config/common.toml'), root)
            authority.update(common=_freeze(common), run=_freeze(by_run[source_run].config_path))
        source_frame = _own_frame(authority, source_run)
    if source_frame is None:
        raise ValueError('New frame-centre transport needs its exact source frame')
    source_frame = deepcopy(source_frame)
    if source_frame.image_size is None:
        source_frame.image_size = tuple(models[source_run].shape_hw)[::-1]
    keys = [str(v).strip().casefold() for v in runs]
    if len(keys) != len(set(keys)) or not keys:
        raise ValueError("New continuation placements must name each run once.")
    result = {}
    phase = "frame_models_camera" if target_grid == "camera_u" else "frame_models_native"
    for number, run in enumerate(keys, 1):
        if run_progress is not None:
            run_progress(dict(phase=phase, completed=number - 1, total=len(keys), run_id=run))
        if run not in models or run not in by_run:
            raise ValueError(f"{run}: no exact own-frame/model entry is available.")
        authority = dict(project=deepcopy(project_file))
        saved_matches = [v for k, v in saved.items() if str(k).strip().casefold() == run]
        if saved_matches:
            authority["kind"] = "saved rig frame"
        else:
            if by_run[run].issues:
                raise ValueError(f"{run}: " + "; ".join(by_run[run].issues))
            authority["kind"] = "series configuration"
            authority["series"] = _freeze(root / "series.toml")
            manifest = _document(authority["series"], toml=True)
            common = geometry._storage_path((manifest.get("paths") or {}).get("common_config", "config/common.toml"), root)
            authority["common"], authority["run"] = _freeze(common), _freeze(by_run[run].config_path)
        frame = _own_frame(authority, run)
        if frame is None:
            raise ValueError(f"{run}: no own rig frame is saved or configured.")
        contract = dict(format=FORMAT, policy=POLICY, target_grid=target_grid,
            frame_raster=FRAME_RASTER, source_frame=source_frame.to_dict(),
            source_run=source_run, target_run=run, source_model=_model(models[source_run]),
            geometry_index=deepcopy(binding["geometry_index"]),
            target_model=_model(models[run]), own_frame=frame.to_dict(), frame_sha256=_json_sha(frame.to_dict()),
            frame_authority=authority, known_ids=sorted(set(int(v) for v in known_ids)),
            source_fill="no-inherited-photo-fill", adopted=False)
        validate_contract(contract, verify_files=True)
        result[run] = contract
        if run_progress is not None:
            run_progress(dict(phase=phase, completed=number, total=len(keys), run_id=run))
    # Source/config drift after any earlier capture fails before a writer starts.
    for contract in result.values():
        for snapshot in contract["frame_authority"].values():
            if isinstance(snapshot, dict) and "text" in snapshot:
                current = _freeze(snapshot["path"])
                if current != snapshot:
                    raise ValueError("Own-frame source metadata changed during continuation capture.")
        validate_contract(contract, verify_files=True)
    return result


def validate_contract(contract, *, verify_files=False):
    from . import layer_painting as lp
    from .painting_run_preview import _same_path
    from .painting_preview_frame_domain import _check_frame_polygon, _project
    from . import map_placement as placement
    from .storage_paths import same_storage_path
    keys = {"format", "policy", "target_grid", "source_run", "target_run", "source_model", "target_model", "geometry_index",
        "own_frame", "frame_sha256", "frame_authority", "known_ids", "source_fill", "adopted"}
    new_raster = isinstance(contract, dict) and 'frame_raster' in contract
    if new_raster:
        keys |= {'frame_raster', 'source_frame'}
    if (not isinstance(contract, dict) or set(contract) != keys or contract["format"] != FORMAT
            or contract["policy"] != POLICY or contract["adopted"] is not False
            or contract["source_fill"] != "no-inherited-photo-fill"
            or contract["target_grid"] not in {"camera_u", "native"}):
        raise ValueError("An exact new-placement photo-continuation contract is required.")
    if new_raster:
        from .painted_adoption_table import _frame
        source_frame = _frame(contract['source_frame'])
        if (contract['frame_raster'] != FRAME_RASTER
                or source_frame.run_id.casefold() != contract['source_run']
                or tuple(source_frame.image_size)[::-1] != tuple(contract['source_model']['shape_hw'])):
            raise ValueError('The physical-frame raster rule needs its exact source frame/grid')
    for key in ("source_run", "target_run"):
        if not isinstance(contract[key], str) or re.fullmatch(r"[a-z0-9_-]+", contract[key]) is None:
            raise ValueError("Continuation run identifiers must be explicit and normalized.")
    known = contract["known_ids"]
    if (not isinstance(known, list) or not known or any(type(v) is not int or not 0 < v < 256 for v in known)
            or known != sorted(set(known))):
        raise ValueError("A continuation needs the map's exact nonzero layer IDs.")
    index = _descriptor(contract["geometry_index"])
    has_point_review = any("point_review" in contract[side + "_model"] for side in ("source", "target"))
    if has_point_review:
        from .rig_point_alignment import validate_contract_binding
        validate_contract_binding(index, contract["source_model"], contract["target_model"])
    if verify_files:
        frozen_index = _freeze(index["path"])
        if any(frozen_index[key] != index[key] for key in index):
            raise ValueError("The continuation's geometry index changed.")
        document = _document(frozen_index)
        if document.get("reference_point_review") is not None and not has_point_review:
            raise ValueError("The reviewed geometry index requires its exact point-review closure.")
        runs = document.get("runs")
        if isinstance(runs, dict):
            entries = [dict(value, run_id=key) for key, value in runs.items()]
        elif isinstance(runs, list):
            entries = runs
        else:
            raise ValueError("The continuation index has no exact run entries.")
        ids = [str(row.get("run_id") or row.get("run") or "").strip().casefold() for row in entries]
        if len(set(ids)) != len(ids) or any(not run for run in ids):
            raise ValueError("The continuation index repeats a run.")
        for side in ("source", "target"):
            run = contract[side + "_run"]
            if run not in ids:
                raise ValueError("The continuation model is not in its captured geometry index.")
            path = placement._model_path(Path(index["path"]), run, entries[ids.index(run)].get("reference_model"))
            if not same_storage_path(path, contract[side + "_model"]["descriptor"]["path"]):
                raise ValueError("The continuation model belongs to another index entry.")
    source_h, source_shape = _checked_model(contract["source_model"], verify_files=verify_files)
    target_h, target_shape = _checked_model(contract["target_model"], verify_files=verify_files)
    if (contract["source_model"]["run_id"] != contract["source_run"]
            or contract["target_model"]["run_id"] != contract["target_run"]):
        raise ValueError("Continuation model/run identity differs.")
    frame = _own_frame(contract["frame_authority"], contract["target_run"])
    if frame is None or frame.to_dict() != contract["own_frame"] or _json_sha(frame.to_dict()) != contract["frame_sha256"]:
        raise ValueError("The target own-frame differs from its frozen authority.")
    photograph = contract["target_model"]["photograph"]
    photo_path = photograph["path"]
    if frame.image and not _same_path(frame.image, photo_path) and "point_review" in contract["target_model"]:
        from types import SimpleNamespace
        from .rig_point_alignment import model_photo_path, ENTRY_KEY
        photo_path = model_photo_path(SimpleNamespace(reference_image=photograph,
            entry={ENTRY_KEY: contract["target_model"]["point_review"]}, run_id=contract["target_run"]))
    if (frame.run_id.strip().casefold() != contract["target_run"]
            or not frame.image or not _same_path(frame.image, photo_path)
            or (frame.image_size is not None and tuple(frame.image_size)[::-1] != target_shape)):
        raise ValueError("The target own-frame belongs to another photograph/grid/run.")
    corners = lp.frame_polygon(frame, (1., 1.))
    _check_frame_polygon(corners)
    if contract["target_grid"] == "native":
        grid = contract["target_model"]["document"].get("baseline_native_grid") or {}
        to_native = placement._matrix(grid.get("H_baseline_camera_u_to_rig_corrected_native"), "target native grid")
        target_shape = _shape(grid.get("shape_hw"))
        target_h = to_native @ target_h
        corners = _project(corners, to_native)
        _check_frame_polygon(corners)
    return source_h, target_h, source_shape, target_shape, frame, corners


def _inside_convex(points, corners):
    """Closed physical domain at exact pixel centres; no raster rounding band."""
    points, corners = np.asarray(points, np.float64), np.asarray(corners, np.float64)
    edges = np.roll(corners, -1, axis=0) - corners
    orientation = np.sign(edges[0, 0] * edges[1, 1] - edges[0, 1] * edges[1, 0])
    inside = np.ones(len(points), bool)
    for corner, edge in zip(corners, edges):
        cross = edge[0] * (points[:, 1] - corner[1]) - edge[1] * (points[:, 0] - corner[0])
        inside &= orientation * cross >= 0.
    return inside


def frame_centre_mask(corners, shape):
    """Rasterize a convex physical frame by membership of integer pixel centres."""
    from .painting_preview_frame_domain import _check_frame_polygon
    _check_frame_polygon(corners)
    height, width = _shape(shape)
    result = np.zeros((height, width), bool)
    for start in range(0, height, 128):
        rows, cols = np.mgrid[start:min(start+128, height), :width]
        points = np.column_stack((cols.ravel(), rows.ravel()))
        result[start:start+rows.shape[0]] = _inside_convex(points, corners).reshape(rows.shape)
    return result


def _boundary_samples(source, contract, source_h, target_h, inside, labels):
    """Restrict nearest sampling to the frozen source domain at its physical edge.

    Only a centre physically inside BOTH frames whose ordinary nearest source
    cell is OUTSIDE the saved source frame may use a boundary sample. We select
    the nearest frame-boundary centre, including its zero/unknown value. Thus
    neither an interior hole nor an unknown boundary donor can be filled.
    """
    import cv2
    from scipy.spatial import cKDTree
    from . import layer_painting as lp
    from .painted_adoption_table import _frame
    from .painting_preview_frame_domain import _project
    target_rc = np.argwhere(inside & (labels == 0))
    empty = dict(target_rc=np.empty((0, 2), np.intp), source_rc=np.empty((0, 2), np.intp))
    if not len(target_rc):
        return empty
    frame = _frame(contract['source_frame'])
    source_xy = _project(target_rc[:, ::-1], source_h @ np.linalg.inv(target_h))
    nearest = np.rint(source_xy).astype(np.intp)
    valid = ((nearest[:, 0] >= 0) & (nearest[:, 0] < source.shape[1])
             & (nearest[:, 1] >= 0) & (nearest[:, 1] < source.shape[0])
             & _inside_convex(source_xy, lp.frame_polygon(frame, (1., 1.))))
    source_inside = lp.frame_mask(frame, source.shape, (1., 1.))
    selected = np.flatnonzero(valid)
    selected = selected[~source_inside[nearest[selected, 1], nearest[selected, 0]]]
    if not len(selected):
        return empty
    boundary = source_inside & ~cv2.erode(source_inside.astype(np.uint8), np.ones((3, 3), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    boundary_rc = np.argwhere(boundary)
    if not len(boundary_rc):
        return empty
    boundary_xy = boundary_rc[:, ::-1].astype(np.float64)
    tree = cKDTree(boundary_xy)
    points = source_xy[selected]
    distance, nearest_boundary = tree.query(points)
    # Preserve deterministic row/column tie-breaking without a physical
    # distance tolerance: the next floating number includes exact-distance ties.
    chosen = []
    for point, radius, closest in zip(points, distance, nearest_boundary):
        candidates = sorted(set(tree.query_ball_point(point, np.nextafter(radius, np.inf))) | {int(closest)})
        square = np.sum((boundary_xy[candidates] - point) ** 2, axis=1)
        chosen.append(candidates[int(np.argmin(square))])
    return dict(target_rc=target_rc[selected], source_rc=boundary_rc[chosen])


def apply_policy(source_layers, contract, *, labels=None, filled=None, support=None):
    """Replay unchanged transport and classify only photo-origin continuation."""
    from . import layer_painting as lp
    from .geometry_raster_sources import carry_label_reference
    source_h, target_h, source_shape, shape, frame, corners = validate_contract(contract, verify_files=True)
    source = np.asarray(source_layers)
    if source.shape != source_shape or source.dtype != np.uint8:
        raise ValueError("The continuation source is not the exact native uint8 map.")
    carried = carry_label_reference(source, reference_matrix=source_h.tolist(), target_matrix=target_h.tolist(),
        target_shape_hw=list(shape), reference_filled=np.zeros(source_shape, bool), source_fill_status="KNOWN",
        reference_support=source != lp.OUTSIDE)
    expected = dict(labels=carried["labels"], filled=carried["geometry_filled"], support=carried["support"])
    for name, value in (("labels", labels), ("filled", filled), ("support", support)):
        if value is not None and (np.asarray(value).shape != shape or not np.array_equal(value, expected[name])):
            raise ValueError("The new placement differs from its original categorical replay: " + name)
    frame = deepcopy(frame)
    frame.corners = {name: (float(point[1]), float(point[0])) for name, point in zip(
        ("top_left", "top_right", "bottom_right", "bottom_left"), corners)}
    samples = None
    if contract.get('frame_raster') == FRAME_RASTER:
        inside = frame_centre_mask(corners, shape)
        samples = _boundary_samples(source, contract, source_h, target_h, inside, expected['labels'])
        if len(samples['target_rc']):
            target_rc, source_rc = samples['target_rc'], samples['source_rc']
            expected['labels'][tuple(target_rc.T)] = source[tuple(source_rc.T)]
            expected['support'][tuple(target_rc.T)] = source[tuple(source_rc.T)] != 0
    else:
        inside = lp.frame_mask(frame, shape, (1., 1.))
    known = np.isin(expected["labels"], contract["known_ids"])
    continued = inside & expected["filled"] & known
    effective = inside & known & (expected["support"] | continued)
    gaps = inside & ~effective
    result = dict(**expected, inside=inside, continued=continued, effective_coverage=effective, gaps=gaps,
        counts=dict(continued_pixels=int(continued.sum()), inside_gap_pixels=int(gaps.sum()),
            inside_pixels=int(inside.sum()), original_support_pixels=int(expected["support"].sum()),
            effective_coverage_pixels=int(effective.sum())))
    if samples is not None:
        result['frame_boundary_samples'] = samples
        result['counts']['frame_boundary_sampled_pixels'] = len(samples['target_rc'])
    return result


def verify_source(binding, source_layers, contracts, *, preparation=None, folder=None):
    """Use the unchanged approved-version verifier, not a new approval parser."""
    from .painted_adoption_table import verify_version
    from .painting_run_preview import _same_path
    session, expected, frame = verify_version(binding)
    if preparation is not None:
        from .painted_adoption_table import _local, _same, checked_fixture_analysis
        if not _same(preparation.get("painting_facies"), binding):
            raise ValueError("The continuation and current painting preparation bind different saved versions.")
        if preparation.get("source_fixture_underlay") is not None:
            from .layer_painting_fixtures import read_underlay
            underlay = read_underlay(_local(folder, preparation["source_fixture_underlay"]))
            checked, derivation = checked_fixture_analysis(binding, underlay)
            if (not _same(preparation.get("painting_analysis_derivation"), derivation)
                    or not _same(preparation.get("fixture_frame"), binding["frame"])):
                raise ValueError("The continuation lost its exact existing fixture analysis derivation.")
            expected = checked.analysis_layers
        elif preparation.get("painting_analysis_derivation") is not None:
            raise ValueError("A continuation fixture derivation needs its bound source underlay.")
    source_layers = np.asarray(source_layers)
    if source_layers.dtype != expected.dtype or source_layers.shape != expected.shape or source_layers.tobytes() != expected.tobytes():
        raise ValueError("The new placement source is not the exact approved saved map.")
    photo = _descriptor(binding["frame_basis"]["photograph"])
    known = sorted({int(layer.layer_id) for layer in session.layers})
    for run, contract in contracts.items():
        validate_contract(contract, verify_files=True)
        if ('frame_raster' in contract and contract['source_frame'] != frame.to_dict()):
            raise ValueError('Frame-centre transport lost the approved painting source frame')
        bound_photo = contract["source_model"]["photograph"]
        if (run != contract["target_run"] or contract["source_run"] != session.run_id.casefold()
                or contract["known_ids"] != known
                or tuple(contract["source_model"]["shape_hw"]) != expected.shape
                or not _same_path(bound_photo["path"], photo["path"])
                or any(bound_photo[key] != photo[key] for key in ("sha256", "size_bytes"))):
            raise ValueError("The continuation contract is not bound to this approved map/photo/layer table.")
    return session, expected, frame


def verify_placement_inputs(contracts, geometries, geometry_index, reference, runs, grid):
    """A positive writer must consume the exact models captured in its policy."""
    from .geometry_raster_sources import descriptor
    index = descriptor(geometry_index)
    if not isinstance(contracts, dict) or set(contracts) != (set(runs) & set(geometries)) or len(runs) != len(set(runs)):
        raise ValueError("Every new placement needs one exact continuation contract per run.")
    for run, contract in contracts.items():
        validate_contract(contract, verify_files=True)
        if (contract["source_run"] != reference or contract["target_run"] != run
                or contract["target_grid"] != grid or contract["geometry_index"] != index
                or run not in geometries or reference not in geometries
                or contract["source_model"] != _model(geometries[reference])
                or contract["target_model"] != _model(geometries[run])):
            raise ValueError("The new placement does not consume its captured index/model/grid inputs.")


def verify_product(path, source_layers, contract, *, counts=None):
    """Replay every V2 mask independently; old filled/support are still present."""
    replay = apply_policy(source_layers, contract)
    native = contract["target_grid"] == "native"
    label_key, fill_key = ("array", "filled") if native else ("layers", "uncovered")
    required = {label_key, fill_key, "support", "inside", "continued", "effective_coverage", "gaps"}
    if native:
        required.add("metadata")
    with np.load(_preset_io.source(path), allow_pickle=True) as data:
        if set(data.files) != required:
            raise ValueError("The V2 placement product has an incomplete or unexpected mask closure.")
        actual = np.asarray(data[label_key])
        dtype = np.dtype("int32") if native else np.dtype("uint8")
        if actual.dtype != dtype or actual.shape != replay["labels"].shape or not np.array_equal(actual, replay["labels"]):
            raise ValueError("V2 labels differ from the original carrier or declared dtype/grid.")
        for leaf, key in ((fill_key, "filled"), ("support", "support"), ("inside", "inside"),
                          ("continued", "continued"), ("effective_coverage", "effective_coverage"), ("gaps", "gaps")):
            actual = np.asarray(data[leaf])
            if actual.dtype != np.dtype(bool) or actual.shape != replay[key].shape or not np.array_equal(actual, replay[key]):
                raise ValueError("V2 continuation mask differs from its exact replay: " + leaf)
    if counts is not None and any(type(counts.get(k)) is not int or counts[k] != v for k, v in replay["counts"].items()):
        raise ValueError("The V2 placement continuation counts differ from their masks.")
    return replay


def _revalidation_metadata(checks):
    """Retain original file/ownership validators, sharing repeated metadata only."""
    pooled, names = {}, {}
    def share(value):
        key = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return pooled.setdefault(key, value)
    for checked in checks:
        # revalidate() consumes only these ownership/output facts plus _Pins.
        checked.document = {key: checked.document[key] for key in ("output_ownership", "outputs")}
        checked.parent_labels = checked.compatibility = None
        checked.guards.documents.clear()
        for attr in ("rows", "aliases", "copyable"):
            values = getattr(checked.guards, attr)
            setattr(checked.guards, attr, {names.setdefault(name, name): share(value)
                                         for name, value in values.items()})
    return checks


def _metadata_size(value, seen=None):
    """Bound actual retained metadata memory without counting shared values twice."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    cost = sys.getsizeof(value)
    if isinstance(value, dict):
        return cost + sum(_metadata_size(key, seen) + _metadata_size(item, seen) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return cost + sum(_metadata_size(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        return cost + _metadata_size(vars(value), seen)
    return cost


def verify_native_proof(preparation, folder):
    """Reuse an unchanged all-run proof through the existing dependency cache.

    One canonical Check calls this from every run. Repeating RAW-basis decoding
    there adds no evidence; every consumed bound file is stat-checked on reuse.
    Fresh-publication contexts retain the existing forced-read behavior.
    """
    from .painted_adoption_table import _FILES, _checked, _bound, _local
    from .local_map_adoption import PROOF_FORMAT, verify_local_preparation
    if (preparation.get("painting_geometry") or {}).get("format") == PROOF_FORMAT:
        return verify_local_preparation(preparation, folder)
    global _NATIVE_PROOFS_PID
    key = _json_sha(dict(preparation=preparation, folder=str(Path(folder).resolve())))
    with _NATIVE_PROOFS_LOCK:
        if _NATIVE_PROOFS_PID != os.getpid():
            _NATIVE_PROOFS.clear()
            _NATIVE_PROOFS_PID = os.getpid()
        cached = None if _FILES.requires_fresh() else _NATIVE_PROOFS.get(key)
        if cached is not None and _FILES.unchanged(cached["dependencies"]):
            captured = read_contracts(Path(folder) / "continuation.json")
            for contract in captured["runs"].values():
                validate_contract(contract, verify_files=True)
            # Original readers always rehash RAW/archives/lineage and recheck
            # owned-directory identities/membership, even if stamps collide.
            for checked in cached["checks"]:
                checked.revalidate()
            _revalidation_metadata(cached["checks"])
            _NATIVE_PROOFS.move_to_end(key)
            return deepcopy(cached["guarded"])
        _NATIVE_PROOFS.pop(key, None)
    with _FILES.capture() as dependencies:
        _checked(_bound(Path(folder) / "continuation.json"))
        checks = {}
        guarded = _verify_native_proof(preparation, folder, basis_checks=checks)
        # The original verifier returns the full admitted file closure, including
        # retained RAW-basis source copies. Record all dependencies in the same
        # existing cache used by the automatic facies-table derivation.
        for bound in guarded:
            _checked(bound)
        for contract in read_contracts(Path(folder) / "continuation.json")["runs"].values():
            _checked(contract["geometry_index"])
        for row in preparation["runs"]:
            _local(folder, row["labels"])
            _checked(row["reference_model"])
            _checked(row["target_model"])
    validators = _revalidation_metadata(list(checks.values()))
    entry = dict(guarded=deepcopy(guarded), dependencies=deepcopy(dependencies),
                 checks=validators, cost=0)
    # Measure the objects actually retained after copying; include a conservative
    # allowance for replacing the temporary cost integer with its measured value.
    entry["cost"] = _metadata_size(entry) + sys.getsizeof(MAX_BYTES)
    if checks and entry["cost"] <= MAX_BYTES and _FILES.unchanged(dependencies):
        with _NATIVE_PROOFS_LOCK:
            _NATIVE_PROOFS[key] = entry
            while len(_NATIVE_PROOFS) > 4 or sum(row["cost"] for row in _NATIVE_PROOFS.values()) > MAX_BYTES:
                _NATIVE_PROOFS.popitem(last=False)
    return guarded


def _verify_native_proof(preparation, folder, *, basis_checks):
    """Replay the unchanged admitted source/renderer proof for positive V2.

    Expected models come from the actual prepared run records and continuation
    contracts, independently of the proof's own declarations. Camera review
    does not call this native/adoption-only gate.
    """
    from .painted_adoption_geometry import verify_proof
    reference = preparation.get("reference_run")
    sidecar = read_contracts(Path(folder) / "continuation.json")
    models, seen = {}, set()
    for row in preparation.get("runs") or []:
        run = row.get("run_id")
        if (not isinstance(run, str) or run in seen or row.get("status") != "prepared"):
            raise ValueError("Every V2 native run needs one complete prepared record.")
        seen.add(run)
        contract = sidecar["runs"].get(run)
        validate_contract(contract, verify_files=True)
        if (contract["source_run"] != reference or contract["target_grid"] != "native"
                or contract["source_model"]["descriptor"] != row.get("reference_model")
                or contract["target_model"]["descriptor"] != row.get("target_model")):
            raise ValueError("The V2 native proof names other source/target models.")
        for key, bound in ((reference, row["reference_model"]), (run, row["target_model"])):
            if key in models and models[key] != bound:
                raise ValueError("The V2 native proof has conflicting model identities.")
            models[key] = bound
    if not seen or reference not in models:
        raise ValueError("The V2 native proof has no complete reference/run closure.")
    if set(sidecar["runs"]) != seen:
        raise ValueError("The native continuation and preparation have different run sets.")
    binding = preparation.get("approved_source")
    guarded = verify_proof(binding, preparation.get("painting_geometry"), models, Path(folder), basis_checks=basis_checks)
    from .geometry_facies import _darsia_metadata
    from .geometry_raster_sources import _same_grid_metadata
    from .painted_adoption_table import _same
    proof = preparation.get("painting_geometry")
    basis = {run: checked.parent_labels for run, checked in basis_checks.items()} or None
    if basis is None:
        revision = json.loads(_preset_io.read_bytes(Path(binding["frame_basis"]["workspace_revision"]["path"])))
        parent = revision["configuration"]["canonical_import_parent"]["imported_configuration"]["labels_path"]
        legacy_metadata = _darsia_metadata(Path(parent))
    for row in preparation["runs"]:
        run = row["run_id"]
        expected = basis[run] if basis is not None else None
        if expected is not None and not _same(row.get("current_labels"), expected["source"]):
            raise ValueError("The V2 native product changed its exact RAW-basis parent label identity.")
        metadata = expected["metadata"] if expected is not None else legacy_metadata
        product = Path(folder) / run / "labels.npz"
        if not _same_grid_metadata(_darsia_metadata(product), metadata):
            raise ValueError("The V2 native product changed its imported typed grid metadata.")
    # Repeated RAW-basis releases share many physical sources. Consumers only
    # recheck their identities, so retain every unique bound file once.
    unique = {}
    for bound in guarded:
        identity = tuple(bound[key] for key in ("path", "sha256", "size_bytes"))
        unique.setdefault(identity, bound)
    return list(unique.values())


def _pooled_document(contracts):
    """Store repeated closed model/snapshot DATA once, preserving every value."""
    pool = {}

    def reference(value):
        digest = _json_sha(value)
        pool[digest] = deepcopy(value)
        return {"pool_sha256": digest}

    runs = {}
    for run, contract in contracts.items():
        value = deepcopy(contract)
        for key in ("source_model", "target_model"):
            value[key] = reference(value[key])
        authority = value["frame_authority"]
        for key in ("project", "series", "common", "run"):
            if key in authority:
                authority[key] = reference(authority[key])
        runs[run] = value
    return dict(format=SIDECAR_FORMAT, policy=POLICY, runs=runs,
                pool=pool, adopted=False)


def save_contracts(path, contracts):
    """Write one bounded fresh sidecar with shared immutable metadata values."""
    verify_capture(contracts)
    for run, contract in contracts.items():
        validate_contract(contract, verify_files=True)
        if run != contract["target_run"]:
            raise ValueError("The continuation sidecar names another run.")
    value = _pooled_document(contracts)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > MAX_BYTES:
        raise ValueError("The new-placement continuation metadata exceeds its read limit.")
    with Path(path).open("xb") as stream:
        stream.write(raw)


def verify_capture(contracts):
    """Before/after a new writer, actual mutable authority must still match.

    Later historical replay uses these exact closed bytes; adopting a revision
    may legitimately update unrelated live project metadata.
    """
    for contract in contracts.values():
        validate_contract(contract, verify_files=True)
        for snapshot in contract["frame_authority"].values():
            if isinstance(snapshot, dict) and "text" in snapshot and _freeze(snapshot["path"]) != snapshot:
                raise ValueError("The own-frame source metadata changed before this new placement closed.")


def read_contracts(path):
    """Read old/full or pooled DATA as the same expanded in-memory document.

    Pool references only encode metadata already admitted by the writer. They
    grant no source authority: callers retain the ordinary contract/file checks.
    Every pooled value must have its canonical digest and be consumed exactly
    by an allowed model/snapshot reference; arbitrary keys are not executable.
    """
    with _preset_io.open_read(Path(path), "rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("The continuation metadata size is invalid.")

    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Repeated continuation metadata key.")
            result[key] = value
        return result

    def invalid(_value):
        raise ValueError("Nonfinite continuation metadata.")

    value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique, parse_constant=invalid)
    base = {"format", "policy", "runs", "adopted"}
    if (not isinstance(value, dict) or value.get("policy") != POLICY
            or value.get("adopted") is not False or not isinstance(value.get("runs"), dict)
            or not value["runs"]):
        raise ValueError("The new placement continuation sidecar is invalid.")
    if value.get("format") == FORMAT and set(value) == base:
        return value  # Historical full contracts retain their existing reader.
    if value.get("format") != SIDECAR_FORMAT or set(value) != base | {"pool"}:
        raise ValueError("The pooled continuation sidecar is invalid.")
    pool = value["pool"]
    if (not isinstance(pool, dict) or not pool
            or any(not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None
                   or not isinstance(item, dict) or _json_sha(item) != key
                   for key, item in pool.items())):
        raise ValueError("The continuation metadata pool changed.")
    used = set()

    def resolve(reference):
        if (not isinstance(reference, dict) or set(reference) != {"pool_sha256"}
                or not isinstance(reference["pool_sha256"], str)
                or reference["pool_sha256"] not in pool):
            raise ValueError("The continuation metadata pool reference is invalid.")
        used.add(reference["pool_sha256"])
        return deepcopy(pool[reference["pool_sha256"]])

    expanded = {}
    contract_keys = {"format", "policy", "target_grid", "source_run", "target_run", "source_model", "target_model",
        "geometry_index", "own_frame", "frame_sha256", "frame_authority", "known_ids", "source_fill", "adopted"}
    for run, packed in value["runs"].items():
        if (not isinstance(run, str) or re.fullmatch(r"[a-z0-9_-]+", run) is None
                or not isinstance(packed, dict) or set(packed) not in (contract_keys, contract_keys | {'frame_raster', 'source_frame'})
                or packed.get("target_run") != run):
            raise ValueError("The pooled continuation sidecar names another run.")
        contract = deepcopy(packed)
        for key in ("source_model", "target_model"):
            contract[key] = resolve(contract.get(key))
        authority = contract.get("frame_authority")
        if not isinstance(authority, dict):
            raise ValueError("The pooled continuation frame authority is invalid.")
        keys = ({"kind", "project"} if authority.get("kind") == "saved rig frame"
                else {"kind", "project", "series", "common", "run"}
                if authority.get("kind") == "series configuration" else set())
        if not keys or set(authority) != keys:
            raise ValueError("The pooled continuation frame authority is invalid.")
        for key in keys - {"kind"}:
            authority[key] = resolve(authority[key])
        expanded[run] = contract
    if used != set(pool):
        raise ValueError("The continuation metadata pool contains an unconsumed value.")
    return dict(format=FORMAT, policy=POLICY, runs=expanded, adopted=False)


def load_contract(path, run):
    """Read one expanded contract and retain its original exact file checks."""
    if not isinstance(run, str) or re.fullmatch(r"[a-z0-9_-]+", run) is None:
        raise ValueError("The continuation run is invalid.")
    value = read_contracts(path)
    if run not in value["runs"]:
        raise ValueError("The new placement continuation sidecar is invalid.")
    contract = value["runs"][run]
    validate_contract(contract, verify_files=True)
    if contract["target_run"] != run:
        raise ValueError("The continuation sidecar key belongs to another run.")
    return contract
