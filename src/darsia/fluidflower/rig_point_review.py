"""Explicit photograph-bound review of four physical reference points per run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from darsia.fluidflower.rig_reference_quadrilateral import reference_frame_geometry
from darsia.fluidflower.series_geometry import RigFrame, native_series_link, stored_native_series_link


FORMAT = "darsia-rig-point-review-v1"
POINT_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")
_KEY = "reference_point_review"


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id(value: Any) -> str:
    run = str(value or "").strip().casefold()
    if not run:
        raise ValueError("Every point review needs a run ID")
    return run


def _located_path(value):
    from .storage_paths import resolve_storage_path
    return os.path.normcase(str(resolve_storage_path(value).resolve())) if value else value


def _same_photo_binding(first, second):
    """Keep authored identities exact; compare their explicit physical locators."""
    _check_photo(first)
    _check_photo(second)
    def located(value):
        return {key: _located_path(item) if key == 'path' else item
                for key, item in value.items() if key != 'identity'}
    return located(first) == located(second)


def _same_frame(first, second):
    def located(value):
        return dict(value, image=_located_path(value.get('image')))
    return located(first) == located(second)


def photo_binding(path: Path | str, image_size: Sequence[int] | None = None) -> dict[str, Any]:
    """Bind the actual displayed file by resolved path, size and exact mtime.

    This performs only a stat, so it can also be called before a background RAW
    read. The UI must compare it again after loading before recording a view.
    Missing files remain explicit rows that cannot be approved.
    """
    text = str(path or "").strip()
    binding = dict(path=os.path.normcase(str(Path(text).expanduser().resolve())) if text else "",
                   available=False, size_bytes=None, mtime_ns=None)
    if image_size is not None:
        size = list(image_size)
        if len(size) != 2 or any(int(value) != value or int(value) <= 0 for value in size):
            raise ValueError("Image size must contain positive width and height")
        binding["image_size"] = [int(value) for value in size]
    try:
        from .storage_paths import resolve_storage_path
        file = resolve_storage_path(binding["path"])
        stat = file.stat()
        if not text or not file.is_file():
            raise OSError("Photograph is not an available file")
        binding.update(available=True, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    except OSError as error:
        binding["error"] = str(error)
    binding["identity"] = _digest(binding)
    return binding


def _check_photo(binding: Any, *, available: bool = False) -> None:
    if not isinstance(binding, Mapping) or binding.get("identity") != _digest({
            key: value for key, value in binding.items() if key != "identity"}):
        raise ValueError("Photograph binding changed; reload its photograph")
    if available and not binding.get("available"):
        raise ValueError("The bound photograph is unavailable")


def _photo_descriptor(value):
    path, digest, size = value.get('path'), value.get('sha256'), value.get('size_bytes')
    if (not path or not isinstance(digest, str) or len(digest) != 64
            or any(char not in '0123456789abcdef' for char in digest)
            or type(size) is not int or size <= 0):
        raise ValueError('The original model photograph needs its exact path, SHA-256 and size')
    return dict(path=os.path.normcase(str(Path(path).resolve())), sha256=digest, size_bytes=size)


def verify_photo_relocation(expected: Mapping[str, Any], path: Path | str) -> dict[str, Any]:
    """Prove a renamed model photograph has the same bytes, without changing it."""
    original = _photo_descriptor(expected)
    before = photo_binding(path)
    _check_photo(before, available=True)
    if before['size_bytes'] != original['size_bytes']:
        raise ValueError('The relocated photograph has another byte count')
    digest = hashlib.sha256()
    with Path(_located_path(before['path'])).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if photo_binding(path) != before:
        raise ValueError('The relocated photograph changed while its bytes were checked')
    if digest.hexdigest() != original['sha256']:
        raise ValueError('The relocated photograph does not match the original model SHA-256')
    return dict(original=original,
                current=dict(path=before['path'], sha256=digest.hexdigest(), size_bytes=before['size_bytes']),
                mtime_ns=before['mtime_ns'])


def check_photo_relocation(relocation, photo, expected=None):
    """Validate the frozen same-byte path witness against the current photo binding."""
    original = _photo_descriptor(relocation['original'])
    current = _photo_descriptor(relocation['current'])
    if (original['sha256'] != current['sha256'] or original['size_bytes'] != current['size_bytes']
            or _located_path(current['path']) != _located_path(photo['path']) or current['size_bytes'] != photo['size_bytes']
            or relocation['mtime_ns'] != photo['mtime_ns']
            or (expected is not None and (original['sha256'] != _photo_descriptor(expected)['sha256']
                or original['size_bytes'] != _photo_descriptor(expected)['size_bytes']
                or _located_path(original['path']) != _located_path(_photo_descriptor(expected)['path'])))):
        raise ValueError('The relocated photograph differs from its original model binding')


def reference_signature(frame: RigFrame) -> str:
    """Exact fingerprint of the saved frame, including its physical metadata."""
    return _digest(frame.to_dict())


def _points(points: Any, *, complete: bool) -> dict[str, list[float]]:
    if not isinstance(points, Mapping):
        raise ValueError("Place four reference points on this photograph")
    if set(points) - set(POINT_NAMES) or (complete and set(points) != set(POINT_NAMES)):
        raise ValueError("Reference points must name TL, TR, BR and BL")
    result = {}
    for name in POINT_NAMES:
        if name not in points:
            continue
        point = np.asarray(points[name], dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("Reference points must contain finite row and column values")
        result[name] = point.tolist()
    return result


def _snapshot(frame: RigFrame) -> dict[str, Any]:
    saved = frame.to_dict()
    geometry = saved.get("reference_geometry") or {}
    observed = geometry.get("photo_corners_rc") or saved["corners"]
    observed = _points(observed, complete=True)
    shape = geometry.get("shape", "rectangle")
    measurements = deepcopy(geometry.get("measurements_m") or {})
    if shape == "rectangle" and not measurements:
        measurements = dict(width=frame.width_m, height=frame.height_m)
    measured = reference_frame_geometry(observed, shape=shape, measurements_m=measurements)
    for name in ("width_m", "height_m"):
        if saved.get(name) is None or not np.isclose(float(saved[name]), measured[name], rtol=1e-8, atol=1e-10):
            raise ValueError("Saved frame dimensions disagree with its measured physical shape")
    return dict(frame=saved, photo_corners_rc=observed, shape=shape,
                measurements_m=measurements,
                point_descriptions=deepcopy(geometry.get("point_descriptions") or {}))


def _validated_geometry(reference: Mapping[str, Any], points: Any) -> dict[str, Any]:
    return reference_frame_geometry(_points(points, complete=True), shape=reference["shape"],
                                    measurements_m=reference["measurements_m"])


def _matrix(model: Any) -> np.ndarray:
    matrix = np.asarray(model.metric_to_grid, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("The measured geometry model is not an invertible finite homography")
    return matrix


def _proposal_identity(reference_identity: str, row: Mapping[str, Any]) -> str:
    return _digest(dict(reference_identity=reference_identity, run_id=row["run_id"], photo=row["photo"],
                        proposed_rc=row["proposed_rc"], before_rc=row["before_rc"],
                        proposal_binding=row["proposal_binding"]))


def _current_identity(row: Mapping[str, Any]) -> str:
    return _digest(dict(current_rc=row["current_rc"], after_rc=row["after_rc"], changes=row["changes"]))


def prepare_review(frame: RigFrame, runs: Sequence[Mapping[str, Any]], models: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Prepare unapproved proposals, keeping the original and edited points apart."""
    reference = _snapshot(frame)
    signature = reference_signature(frame)
    document = dict(format=FORMAT, reference=reference, reference_identity=signature,
                    snapshot_identity=_digest(reference), created_utc=_now(), runs={})
    indexed = {_run_id(name): model for name, model in (models or {}).items()}
    reference_run = _run_id(frame.run_id)
    source = indexed.get(reference_run)
    for run in runs:
        run_id = _run_id(run.get("run_id"))
        if run_id in document["runs"]:
            raise ValueError("Point review run IDs must be distinct")
        photo = photo_binding(run.get("path", ""), run.get("image_size"))
        proposed = None
        binding = dict(method="manual_required", reason="Measured geometry models are unavailable")
        if run_id == reference_run:
            proposed = deepcopy(reference["photo_corners_rc"])
            binding = dict(method="reference", reason="Reference points require photograph review")
        elif source is not None and indexed.get(run_id) is not None:
            try:
                source_matrix, target_matrix = _matrix(source), _matrix(indexed[run_id])
                transform = target_matrix @ np.linalg.inv(source_matrix)
                points = np.array([reference["photo_corners_rc"][name][::-1] for name in POINT_NAMES])
                moved = np.column_stack((points, np.ones(4))) @ transform.T
                denominator = moved[:, 2]
                if not (np.all(denominator > 1e-12) or np.all(denominator < -1e-12)):
                    raise ValueError("The proposal crosses a perspective horizon")
                mapped = moved[:, :2] / denominator[:, None]
                proposed = {name: mapped[index, ::-1].tolist() for index, name in enumerate(POINT_NAMES)}
                _validated_geometry(reference, proposed)
                binding = dict(method="geometry_model", source_matrix=source_matrix.tolist(),
                               target_matrix=target_matrix.tolist(), reason="Unverified model proposal")
            except (ValueError, TypeError, AttributeError, np.linalg.LinAlgError) as error:
                proposed = None
                binding = dict(method="manual_required", reason=str(error))
        row = dict(run_id=run_id, photo=photo, proposed_rc=deepcopy(proposed), before_rc=deepcopy(proposed),
                   current_rc=deepcopy(proposed), after_rc=deepcopy(proposed), changes=[],
                   proposal_binding=binding, status="pending", approval=None)
        if run.get('photograph_relocation') is not None:
            check_photo_relocation(run['photograph_relocation'], photo)
            binding['photograph_relocation'] = deepcopy(run['photograph_relocation'])
        row["proposal_identity"] = _proposal_identity(signature, row)
        row["current_identity"] = _current_identity(row)
        document["runs"][run_id] = row
    return document


def recover_missing_proposals(document, prepared):
    """Add newly available proposals only to untouched, unapproved empty rows."""
    _check_document(document)
    _check_document(prepared)
    if prepared['reference_identity'] != document['reference_identity']:
        raise ValueError('Recovered proposals must use the same original reference snapshot')
    result = deepcopy(document)
    for run, fresh in prepared['runs'].items():
        old = result['runs'].get(run)
        if (old is not None and old['status'] == 'pending' and old['approval'] is None
                and old['proposed_rc'] is None and old['current_rc'] is None and not old['changes']
                and fresh['proposed_rc'] is not None):
            # An absent image-size field may become known, but the actual file must be unchanged.
            keys = ('path', 'available', 'size_bytes', 'mtime_ns')
            if all(old['photo'].get(key) == fresh['photo'].get(key) for key in keys):
                result['runs'][run] = deepcopy(fresh)
    return result


def _check_document(document: Any) -> None:
    if not isinstance(document, Mapping) or document.get("format") != FORMAT:
        raise ValueError("Prepare the reference point review first")
    reference = document["reference"]
    if document.get("snapshot_identity") != _digest(reference):
        raise ValueError("The reference snapshot changed; prepare the point review again")
    frame = RigFrame.from_dict(reference["frame"])
    if document.get("reference_identity") != reference_signature(frame) or _snapshot(frame) != reference:
        raise ValueError("The saved reference or physical measurements changed")
    if not isinstance(document["runs"], Mapping):
        raise ValueError("The point review has no run list")
    for run_id, row in document["runs"].items():
        if _run_id(run_id) != run_id or row["run_id"] != run_id:
            raise ValueError("The point review run identity changed")
        _check_photo(row["photo"])
        relocation = row['proposal_binding'].get('photograph_relocation')
        if relocation is not None:
            check_photo_relocation(relocation, row['photo'])
        if row.get("proposal_identity") != _proposal_identity(document["reference_identity"], row):
            raise ValueError(f"{run_id}: the original point proposal changed; prepare review again")
        if row.get("current_identity") != _current_identity(row) or row["current_rc"] != row["after_rc"]:
            raise ValueError(f"{run_id}: reference point corrections changed after review")
        if row["status"] not in ("pending", "corrected", "approved"):
            raise ValueError(f"{run_id}: unknown point review status")
        if row["current_rc"] is not None:
            points = _points(row["current_rc"], complete=False)
            if len(points) == 4:
                _validated_geometry(reference, points)


def update_run(document: Mapping[str, Any], run_id: str, current_rc: Mapping[str, Any]) -> dict[str, Any]:
    """Return a corrected review with four valid points and previous approval removed."""
    _check_document(document)
    points = _points(current_rc, complete=True)
    _validated_geometry(document["reference"], points)
    return _replace_points(document, _run_id(run_id), points)


def _replace_points(document: Mapping[str, Any], run_id: str, points: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(document)
    row = result["runs"][run_id]
    if row["current_rc"] == points:
        return result
    before = deepcopy(row["current_rc"])
    approved_frame = reviewed_frame(document,run_id).to_dict() if row['status']=='approved' else None
    row.update(current_rc=deepcopy(points), after_rc=deepcopy(points), status="corrected", approval=None)
    change = dict(before_rc=before, after_rc=deepcopy(points), changed_utc=_now())
    if approved_frame is not None:
        change['approved_frame_before'] = approved_frame
    row["changes"].append(change)
    row["current_identity"] = _current_identity(row)
    return result


def set_point(document: Mapping[str, Any], run_id: str, name: str, row: float, col: float) -> dict[str, Any]:
    """Set one point; a manual-only run can be assembled one click at a time."""
    _check_document(document)
    if name not in POINT_NAMES:
        raise ValueError("Choose top-left, top-right, bottom-right or bottom-left")
    run_id = _run_id(run_id)
    points = deepcopy(document["runs"][run_id]["current_rc"] or {})
    points[name] = [row, col]
    points = _points(points, complete=False)
    if len(points) == 4:
        _validated_geometry(document["reference"], points)
    return _replace_points(document, run_id, points)


def _state_identity(document: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    return _digest(dict(reference_identity=document["reference_identity"], run_id=row["run_id"],
                        proposal_identity=row["proposal_identity"], current_identity=row["current_identity"],
                        photo=row["photo"]))


def approve_run(document: Mapping[str, Any], run_id: str, *, viewed_identity: str,
                actor: str) -> dict[str, Any]:
    """Approve only the four points actually reviewed on the exact bound photo.

    ``actor`` is who approves; the application takes it from the series
    configuration (series_decisions.series_actor).
    """
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("Name who approves the reference points")
    _check_document(document)
    result = deepcopy(document)
    row = result["runs"][_run_id(run_id)]
    _validated_geometry(result["reference"], row["current_rc"])
    _check_photo(row["photo"], available=True)
    if viewed_identity != row["photo"]["identity"]:
        raise ValueError("Load and view this run's bound photograph before approving its points")
    actual = photo_binding(row["photo"]["path"], row["photo"].get("image_size"))
    if not _same_photo_binding(actual, row["photo"]):
        raise ValueError("The photograph changed after loading; reload it before approval")
    approval = dict(state_identity=_state_identity(result, row), viewed_identity=viewed_identity,
                    actor=actor, approved_utc=_now())
    approval["identity"] = _digest(approval)
    row.update(status="approved", approval=approval)
    return result


def _check_approval(document: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    approval = row.get("approval")
    if row["status"] != "approved" or not isinstance(approval, Mapping):
        raise ValueError(f"{row['run_id']}: review and approve the four reference points")
    if (approval.get("identity") != _digest({key: value for key, value in approval.items() if key != "identity"})
            or approval.get("state_identity") != _state_identity(document, row)
            or approval.get("viewed_identity") != row["photo"]["identity"]):
        raise ValueError(f"{row['run_id']}: point approval is stale")
    _validated_geometry(document["reference"], row["current_rc"])


def review_blocker(document: Any, frame: RigFrame, run_ids: Sequence[str],
                   current_photo_bindings: Mapping[str, Mapping[str, Any]]) -> str | None:
    """A normal status reason, or None only when every required row is current."""
    try:
        _check_document(document)
        if not reference_matches(document, frame):
            raise ValueError("Reference points, descriptions or physical dimensions changed; prepare review again")
        bindings = {_run_id(run): binding for run, binding in current_photo_bindings.items()}
        for name in run_ids:
            run_id = _run_id(name)
            if run_id not in document["runs"]:
                raise ValueError(f"{run_id}: prepare and review its reference points")
            row = document["runs"][run_id]
            current = bindings.get(run_id)
            if current is None:
                raise ValueError(f"{run_id}: the current photograph binding is missing")
            _check_photo(current, available=True)
            if not _same_photo_binding(current, row["photo"]):
                raise ValueError(f"{run_id}: photograph changed; prepare and review its reference points again")
            _check_approval(document, row)
        return None
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, np.linalg.LinAlgError) as error:
        return str(error) or "The reference point review is incomplete or stale"


def reference_matches(document: Any, frame: RigFrame) -> bool:
    """Accept the original reference or its exact approved saved replacement."""
    try:
        _check_document(document)
        if document["reference_identity"] == reference_signature(frame):
            return True
        if _same_frame(document['reference']['frame'], frame.to_dict()):
            return True
        source_run = document["reference"]["frame"]["run_id"]
        if any(isinstance(change.get('approved_frame_before'), Mapping)
                and _same_frame(change['approved_frame_before'], frame.to_dict())
                for change in document['runs'][source_run.casefold()]['changes']):
            return True
        return _same_frame(frame.to_dict(), reviewed_frame(document, source_run).to_dict())
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, np.linalg.LinAlgError):
        return False


def reviewed_frame(document: Mapping[str, Any], run_id: str) -> RigFrame:
    """Build the exact approved per-run frame; never replace the original proposal."""
    _check_document(document)
    row = document["runs"][_run_id(run_id)]
    _check_approval(document, row)
    actual = photo_binding(row["photo"]["path"], row["photo"].get("image_size"))
    if not _same_photo_binding(actual, row["photo"]):
        raise ValueError("The approved photograph changed; review its reference points again")
    geometry = _validated_geometry(document["reference"], row["current_rc"])
    payload = deepcopy(document["reference"]["frame"])
    metadata = dict(payload.get("reference_geometry") or {})
    metadata.update(geometry["reference_geometry"])
    metadata["point_review_identity"] = row["approval"]["identity"]
    payload.update(geometry, reference_geometry=metadata, run_id=row["run_id"], image=row["photo"]["path"],
                   image_size=row["photo"].get("image_size"), disabled_corners=[], source="reviewed reference points")
    return RigFrame.from_dict(payload)


def metric_to_grid(document: Mapping[str, Any], run_id: str) -> np.ndarray:
    """Approved physical millimetres to camera-U (column, row) homography.

    Every run uses the reference shape's same metric bounding-box coordinates,
    including the preserved nonrectangular reference. This only derives the
    effective reviewed transform; it does not rewrite original model evidence.
    """
    frame = reviewed_frame(document, run_id)
    millimetres_to_unit = np.array([
        [1.0 / (1000.0 * frame.width_m), 0.0, 0.0],
        [0.0, -1.0 / (1000.0 * frame.height_m), 1.0],
        [0.0, 0.0, 1.0],
    ])
    return np.linalg.inv(frame._forward()) @ millimetres_to_unit


def load_review(project: Any) -> dict[str, Any] | None:
    value = native_series_link(project).get(_KEY)
    return deepcopy(dict(value)) if isinstance(value, Mapping) else None


def save_review(project: Any, document: Mapping[str, Any]) -> None:
    """Persist only through the project metadata writer, preserving other fields."""
    _check_document(document)
    for row in document["runs"].values():
        if row["status"] == "approved":
            _check_approval(document, row)
    from .series_decisions import project_series_root, series_actor
    link = deepcopy(stored_native_series_link(project))
    link[_KEY] = deepcopy(document)
    project.update_metadata("tools", {"native_series": link},
                            updated_by=series_actor(project_series_root(project)),
                            reason="Save reference point review")


__all__ = ["POINT_NAMES", "photo_binding", "reference_signature", "reference_matches", "prepare_review", "set_point", "update_run",
           "approve_run", "load_review", "save_review", "review_blocker", "reviewed_frame", "metric_to_grid"]
