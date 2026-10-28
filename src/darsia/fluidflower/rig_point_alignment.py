"""Immutable reviewed transforms beside, never inside, measured model evidence."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from . import preset_wasserstein_inputs as _io
from . import rig_point_review as review
from .storage_paths import resolve_storage_path, same_storage_descriptor, same_storage_path


FORMAT = "darsia-reviewed-reference-point-alignment-v1"
KEY = "reference_point_review"
ENTRY_KEY = "_reference_point_review"
MAX_BYTES = 16 * 1024 * 1024


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _descriptor(value):
    if (not isinstance(value, Mapping) or not str(value.get("path") or "")
            or re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256") or "")) is None
            or type(value.get("size_bytes")) is not int or not 0 < value["size_bytes"] <= MAX_BYTES):
        raise ValueError("Reviewed alignment needs an exact metadata path, SHA256 and size")
    return {key: value[key] for key in ("path", "sha256", "size_bytes")}


def _read(path, bound=None):
    path = Path(path)
    if _io.current_scope() is None:
        path = resolve_storage_path(path)
    path = path.resolve()
    before = _io.file_stat(path)
    if not 0 < before.st_size <= MAX_BYTES:
        raise ValueError("Reviewed alignment metadata exceeds the existing JSON size bound")
    raw = _io.read_bytes(path)
    after = _io.file_stat(path)
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    descriptor = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    if stamp(before) != stamp(after) or len(raw) != before.st_size:
        raise ValueError("Reviewed alignment metadata changed while reading")
    if bound is not None and not same_storage_descriptor(descriptor, _descriptor(bound)):
        raise ValueError("The bound reviewed alignment metadata changed")
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Reviewed alignment metadata must be a JSON object")
    return value, descriptor


def _entries(document):
    runs = document.get("runs")
    if isinstance(runs, Mapping):
        rows = [(str(key).strip().casefold(), row) for key, row in runs.items()]
    elif isinstance(runs, list):
        rows = [(str(row.get("run_id") or row.get("run") or "").strip().casefold(), row) for row in runs]
    else:
        raise ValueError("The measured geometry index has no run entries")
    ids = [key for key, _ in rows]
    if not all(ids) or len(set(ids)) != len(ids) or any(not isinstance(row, dict) for _, row in rows):
        raise ValueError("The measured geometry index must name each run once")
    return rows


def _absolute_index(document, path):
    from .map_placement import GEOMETRY_FORMAT, _model_path
    if document.get("format") != GEOMETRY_FORMAT or KEY in document:
        raise ValueError("Reviewed alignment must bind an original measured geometry index")
    result = deepcopy(document)
    for run, row in _entries(result):
        original = row.get("reference_model")
        model_path = _model_path(Path(path), run, original).resolve()
        _, actual = _read(model_path)
        descriptor = deepcopy(dict(original)) if isinstance(original, Mapping) else {}
        descriptor.update(actual)
        row["reference_model"] = descriptor
    return result


def _load_package(descriptor, *, current_runs=None):
    package, bound = _read(_descriptor(descriptor)["path"], descriptor)
    if (package.get("format") != FORMAT or set(package) != {"format", "base_index", "review", "reviewed_runs", "reference_run"}
            or not isinstance(package["reviewed_runs"], list) or not package["reviewed_runs"]
            or package["reference_run"] not in package["reviewed_runs"]
            or len(set(package["reviewed_runs"])) != len(package["reviewed_runs"])):
        raise ValueError("The reviewed point alignment closure is malformed")
    base, _ = _read(_descriptor(package["base_index"])["path"], package["base_index"])
    if KEY in base:
        raise ValueError("A reviewed point alignment cannot replace its original base-index authority")
    frame = review.RigFrame.from_dict(package["review"]["reference"]["frame"])
    if frame.run_id.strip().casefold() != package["reference_run"]:
        raise ValueError("The reviewed point alignment names another source run")
    current = set(package['reviewed_runs'] if current_runs is None else [package['reference_run'],*current_runs])
    if not current <= set(package['reviewed_runs']):
        raise ValueError('The point alignment contains an unreviewed run')
    bindings = {}
    for run in package["reviewed_runs"]:
        photo = package["review"]["runs"][run]["photo"]
        bindings[run] = review.photo_binding(photo["path"], photo.get("image_size")) if run in current else photo
    blocker = review.review_blocker(package["review"], frame, package["reviewed_runs"], bindings)
    if blocker:
        raise ValueError(blocker)
    if not set(package["reviewed_runs"]) <= {run for run, _ in _entries(base)}:
        raise ValueError("Reviewed runs are absent from the original measured geometry index")
    return package, base, bound


def read_alignment(document, index_path, *, current_runs=None):
    """Validate one derived index against its exact immutable review and base."""
    descriptor = document.get(KEY)
    if descriptor is None:
        return None
    package, base, bound = _load_package(descriptor,current_runs=current_runs)
    expected = _absolute_index(base, package["base_index"]["path"])
    expected[KEY] = bound
    if not _same_derived_index(document, expected):
        raise ValueError("The reviewed geometry index differs from its exact original model/cohort inputs")
    return dict(descriptor=_descriptor(descriptor), package=package)


def _same_derived_index(first, second):
    """Only model/review descriptor locators may differ after an explicit move."""
    left, right = deepcopy(first), deepcopy(second)
    if not same_storage_descriptor(left.get(KEY), right.get(KEY)):
        return False
    left.pop(KEY); right.pop(KEY)
    left_rows, right_rows = _entries(left), _entries(right)
    if [run for run, _ in left_rows] != [run for run, _ in right_rows]:
        return False
    for (_, a), (_, b) in zip(left_rows, right_rows):
        if not same_storage_descriptor(a.get('reference_model'), b.get('reference_model')):
            return False
        a['reference_model'] = b['reference_model'] = None
    return left == right


def effective_matrix(descriptor, run_id, expected_photo=None):
    """Read the approved transform while retaining original model/photo evidence."""
    run_id = str(run_id).strip().casefold()
    package, _, _ = _load_package(descriptor,current_runs=[run_id])
    if run_id not in package["reviewed_runs"]:
        raise ValueError(f"{run_id}: its reference points are not in the approved transfer cohort")
    row = package["review"]["runs"][run_id]
    photo = row["photo"]
    if expected_photo is not None:
        same_path = same_storage_path
        if not same_path(photo["path"], expected_photo.get("path", "")):
            relocation = row['proposal_binding'].get('photograph_relocation')
            if relocation is None:
                raise ValueError("Reviewed points belong to another original model photograph")
            review.check_photo_relocation(relocation, photo, expected_photo)
        elif photo["size_bytes"] != expected_photo.get("size_bytes"):
            raise ValueError("Reviewed points belong to another original model photograph")
    return review.metric_to_grid(package["review"], run_id)


def model_photo_path(geometry, photograph=None):
    """Locate the same approved exposure without changing its model identity."""
    original = getattr(geometry, "reference_image", photograph)
    photograph = original if photograph is None else photograph
    path = Path(photograph["path"])
    descriptor = (getattr(geometry, "entry", None) or {}).get(ENTRY_KEY)
    same_path = same_storage_path
    if descriptor is None or not same_path(path, original["path"]):
        return resolve_storage_path(path)
    if any(photograph.get(key) != original.get(key) for key in ("sha256", "size_bytes")):
        raise ValueError("The display photograph differs from its original model binding")
    run = str(geometry.run_id).strip().casefold()
    package, _, _ = _load_package(descriptor, current_runs=[run])
    row = package["review"]["runs"][run]
    if same_path(row["photo"]["path"], path):
        return resolve_storage_path(path)
    relocation = row["proposal_binding"].get("photograph_relocation")
    if relocation is None:
        raise ValueError("Reviewed points belong to another original model photograph")
    review.check_photo_relocation(relocation, row["photo"], photograph)
    verified = review.verify_photo_relocation(photograph, row["photo"]["path"])
    if verified != relocation:
        raise ValueError("The reviewed photograph relocation changed")
    return resolve_storage_path(verified["current"]["path"])


def effective_model_matrix(model_document, geometry_index_descriptor=None):
    """Replay the original measured matrix or its exact bound reviewed override."""
    from .map_placement import _matrix, _model_path
    if "H_metric_mm_to_camera_u" in model_document:
        matrix = _matrix(model_document["H_metric_mm_to_camera_u"], "measured metric-to-camera-U")
    elif "H_camera_u_to_metric_mm" in model_document:
        matrix = np.linalg.inv(_matrix(model_document["H_camera_u_to_metric_mm"], "measured camera-U-to-metric"))
    else:
        raise ValueError("The original measured model has no metric homography")
    if geometry_index_descriptor is None:
        return matrix
    if (not isinstance(geometry_index_descriptor, Mapping)
            or not geometry_index_descriptor.get('path')
            or re.fullmatch(r'[0-9a-f]{64}',str(geometry_index_descriptor.get('sha256',''))) is None):
        raise ValueError('The geometry index needs its original path and SHA256')
    index, descriptor = _read(geometry_index_descriptor['path'])
    if (descriptor['sha256'] != geometry_index_descriptor['sha256']
            or ('size_bytes' in geometry_index_descriptor
                and descriptor['size_bytes'] != geometry_index_descriptor['size_bytes'])):
        raise ValueError('The bound geometry index changed')
    run = str(model_document.get("run_id") or model_document.get("run") or "").strip().casefold()
    alignment = read_alignment(index, descriptor["path"],current_runs=[run])
    if alignment is None:
        return matrix
    entries = dict(_entries(index))
    if not run or run not in entries:
        raise ValueError("The reviewed index does not bind this measured model run")
    path = _model_path(Path(descriptor["path"]), run, entries[run].get("reference_model"))
    original, _ = _read(path)
    if original != model_document:
        raise ValueError("The supplied measured model differs from its reviewed index entry")
    return effective_matrix(alignment["descriptor"], run, model_document.get("reference_image") or {})


def validate_contract_binding(index_descriptor, source_model, target_model):
    """Reject a closure that selects a sidecar other than its derived index's."""
    document, _ = _read(_descriptor(index_descriptor)["path"], index_descriptor)
    alignment = read_alignment(document,index_descriptor["path"],
        current_runs=[source_model['run_id'],target_model['run_id']])
    expected = None if alignment is None else alignment["descriptor"]
    for model in (source_model, target_model):
        if model.get("point_review") != expected:
            raise ValueError("The continuation point review is not bound to its captured geometry index")
        if expected is not None and model["run_id"] not in alignment["package"]["reviewed_runs"]:
            raise ValueError("The continuation run has no approved point alignment")


def _write_immutable(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError("Existing immutable point-alignment inputs differ")


def capture_reviewed_selection(project, selection, document, frame, run_ids):
    """Freeze the approved transfer cohort without modifying its original models."""
    from . import layer_painting as lp
    from .painting_frame_models import read_adoption_cohort, verify_frame_model_selection
    verify_frame_model_selection(selection)
    source_index, descriptor = _read(selection.index_path)
    if source_index.get(KEY) is not None:
        prior = read_alignment(source_index, selection.index_path)
        base_descriptor = prior["package"]["base_index"]
        base, _ = _read(base_descriptor["path"], base_descriptor)
    else:
        base, base_descriptor = source_index, descriptor
    reference_run = document["reference"]["frame"]["run_id"].strip().casefold()
    cohort = read_adoption_cohort(base_descriptor["path"], list(run_ids), reference_run)["runs"]
    if document.get("required_runs") is not None and document["required_runs"] != cohort:
        raise ValueError("The reviewed point cohort differs from the selected measured geometry index")
    current = {}
    for run in cohort:
        row = document.get("runs", {}).get(run)
        if row is None:
            raise ValueError(f"{run}: prepare and approve its reference points")
        current[run] = review.photo_binding(row["photo"]["path"], row["photo"].get("image_size"))
    blocker = review.review_blocker(document, frame, cohort, current)
    if blocker:
        raise ValueError(blocker)
    if not set(cohort) <= {run for run, _ in _entries(base)}:
        raise ValueError("The selected transfer cohort has missing measured geometry models")
    package = dict(format=FORMAT, base_index=base_descriptor, review=deepcopy(document),
                   reviewed_runs=cohort, reference_run=reference_run)
    raw = _bytes(package)
    digest = hashlib.sha256(raw).hexdigest()
    folder = lp.painting_folder(project) / "reference_point_reviews" / digest
    if not (folder / 'index.json').is_file():
        cached = _saved_capture(folder.parent, package)
        if cached is not None:
            index_path, bound_index = cached
            verify_frame_model_selection(selection)
            return replace(selection, index_path=str(index_path), source='reviewed reference points',
                sha256=bound_index['sha256'], size_bytes=bound_index['size_bytes'])
    review_path = folder / "review.json"
    _write_immutable(review_path, raw)
    bound_review = dict(path=str(review_path.resolve()), sha256=digest, size_bytes=len(raw))
    derived = _absolute_index(base, base_descriptor["path"])
    derived[KEY] = bound_review
    index_raw = _bytes(derived)
    index_path = folder / "index.json"
    _write_immutable(index_path, index_raw)
    read_alignment(derived, index_path)
    verify_frame_model_selection(selection)
    return replace(selection, index_path=str(index_path.resolve()), source="reviewed reference points",
                   sha256=hashlib.sha256(index_raw).hexdigest(), size_bytes=len(index_raw))


def _saved_capture(root, package):
    """Reuse an exact existing review capsule when only its storage moved."""
    if not root.is_dir():
        return None
    candidates = list(root.iterdir())
    if len(candidates) > 512:
        return None
    total = 0
    for folder in sorted(candidates, reverse=True):
        review_path, index_path = folder/'review.json', folder/'index.json'
        if not folder.is_dir() or not review_path.is_file() or not index_path.is_file():
            continue
        size = review_path.stat().st_size
        total += size
        if total > 64 * 1024 * 1024:
            return None
        if not 0 < size <= MAX_BYTES:
            continue
        saved, review_bound = _read(review_path)
        if not same_storage_descriptor(saved.get('base_index'), package['base_index']):
            continue
        if {key: value for key, value in saved.items() if key != 'base_index'} != {
                key: value for key, value in package.items() if key != 'base_index'}:
            continue
        index, bound_index = _read(index_path)
        if not same_storage_descriptor(index.get(KEY), review_bound):
            raise ValueError('The cached reviewed index differs from its saved review')
        read_alignment(index, index_path)
        return index_path.resolve(), bound_index
    return None
