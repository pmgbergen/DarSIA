"""Original producer inventories to immutable shared metadata publications.

This is an operational adapter. It neither enumerates input directories nor
changes a completed result, its scientific products, or its producer identity.
"""
from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath

from ..storage_profile import COLOUR_COORDINATE_ARCHIVE
from .job_queue import descriptor, name, read_json
from .products_publication import (
    MAX_METADATA_BYTES, _bound, _copy, _fresh_root, _target, _verified_bytes,
    _json, product, publish_run_metadata, published_metadata_descriptor,
    publish_indexed_products,
)


def _pure(value):
    text = str(value)
    windows = PureWindowsPath(text)
    return windows if windows.drive or "\\" in text else PurePosixPath(text)


def _relative(path, root):
    """Lexical producer identity only; never probe a foreign local drive."""
    path, root = _pure(path), _pure(root)
    if type(path) is not type(root) or not path.is_absolute() or ".." in path.parts:
        raise ValueError("Invalid original producer location")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("Producer inventory escapes its actual attempt") from error
    if not relative.parts:
        raise ValueError("A product must be a file below its producer attempt")
    return relative.as_posix()


def _kind(filename):
    if filename == "commit.json":
        return "commit_receipt"
    if filename == "result.json" or filename == "native_statistics.json.gz":
        return "frame_scalar"
    if filename == "manifest.json":
        return "manifest"
    if filename.endswith(".json"):
        return "operational_receipt"
    if filename.endswith(".csv"):
        return "scalar_table"
    if filename.endswith((".mp4", ".mkv", ".mov")):
        return "film"
    if filename in {"phase_labels.png", "phase_classes.npz", "phase_area.npz",
                    "phase_operator_masks.npz", "products.npz", "native_geometry.npz",
                    "phase_probabilities.npz", COLOUR_COORDINATE_ARCHIVE}:
        return "map"
    return "intermediate"


_METADATA = {"commit_receipt", "frame_scalar", "manifest",
             "operational_receipt", "scalar_table"}


def extract_products(result, claim, machine_id, *, producer_origin_for_attempt):
    """Use only closed producers' explicit inventories, retaining prefix origins.

    The origin callback is supplied by the original validated resume proof. It
    must never infer a machine from the spelling of a local path or hostname.
    It returns machine_id, proof_descriptor and an explicitly accessible root.
    The current attempt instead follows this actual claim's admitted output.
    """
    if result.get("pipeline") == "preset-wasserstein":
        from ..preset_wasserstein_results import extract_preset_wasserstein_products
        return extract_preset_wasserstein_products(result, claim, machine_id)
    if result.get("pipeline") == "series-setup":
        from .series_setup_results import extract_setup_products
        return extract_setup_products(result, claim, machine_id)
    if result.get("pipeline") == "native-phase-statistics":
        from .phase1_protocol import extract_phase1_products
        return extract_phase1_products(result, claim, machine_id)
    if result.get("pipeline") == "n1-calibration-refit":
        from .common_calibration import extract_refit_products
        return extract_refit_products(result, claim, machine_id)
    if result.get("worker_references_closed") is not True or result.get("complete") is not True:
        raise ValueError("Original scientific producers have not closed their references")
    if result.get("run") != claim.task["run"] or result.get("attempt_id") != claim.token:
        raise ValueError("Original completion belongs to another claim")
    records, producers = result["records"], result["frame_producers"]
    ordered = result["ordered_case_ids"]
    if ([row["frame_id"] for row in records] != ordered
            or [row["case_id"] for row in producers] != ordered
            or len(set(ordered)) != len(ordered)):
        raise ValueError("Original frame inventories differ from their exact case order")
    if result.get("case_count") != len(ordered):
        raise ValueError("Original completion has an inconsistent case count")
    rows, metadata, accessible, by_id = [], [], {}, {}
    origins = {}

    def origin(attempt):
        key = str(_pure(attempt))
        if _pure(attempt) == _pure(claim.output):
            return dict(machine_id=machine_id, accessible_root=Path(claim.output))
        if key not in origins:
            if not callable(producer_origin_for_attempt):
                raise ValueError("A genuine retired prefix origin proof is required")
            proof = producer_origin_for_attempt(attempt)
            if (not isinstance(proof, dict)
                    or not isinstance(proof.get("machine_id"), str)
                    or not isinstance(proof.get("proof_descriptor"), dict)
                    or not proof.get("accessible_root")):
                raise ValueError("Original prefix producer has no admitted origin")
            name(proof["machine_id"])
            bound = _bound(proof["proof_descriptor"])
            relative = _relative(bound["path"], attempt)
            actual = Path(proof["accessible_root"]).joinpath(*relative.split("/"))
            retirement = _json(_verified_bytes(actual, bound, limit=MAX_METADATA_BYTES))
            if (retirement.get("complete") is not True
                    or retirement.get("all_absent") is not True
                    or retirement.get("worker_os_references_closed") is not True
                    or retirement.get("run") != claim.task["run"]
                    or retirement.get("attempt_id") != _pure(attempt).name
                    or _pure(retirement.get("attempt_directory", "")) != _pure(attempt)
                    or name(retirement.get("machine_id")) != name(proof["machine_id"])):
                raise ValueError("Original retired prefix proof differs from its run, attempt or machine")
            origins[key] = dict(proof)
        return origins[key]

    def add(asset, attempt, *, logical_id=None, kind=None):
        bound = _bound(asset)
        relative = _relative(bound["path"], attempt)
        proof = origin(attempt)
        token = _pure(attempt).name
        logical_id = logical_id or ("attempts/" + token + "/" + relative)
        declared = asset.get("kind")
        if declared is not None and kind is not None and declared != kind:
            raise ValueError("Explicit product kind conflicts with its known metadata role")
        # Keep an existing producer-declared export role. Never invent an
        # export from a filename or relabel ordinary NPZ maps as exports.
        if kind is None:
            kind = declared if declared is not None else _kind(_pure(bound["path"]).name)
        entry = product(logical_id, machine_id=proof["machine_id"],
                        local_path=bound["path"], sha256=bound["sha256"],
                        bytes=bound["bytes"], kind=kind)
        if logical_id in by_id:
            if by_id[logical_id] != entry:
                raise ValueError("Conflicting final product identity: " + logical_id)
            return accessible[logical_id]
        by_id[logical_id] = entry
        rows.append(entry)
        actual = Path(proof["accessible_root"]).joinpath(*relative.split("/"))
        accessible[logical_id] = actual
        if kind in _METADATA:
            metadata.append(logical_id)
        return actual

    add(result["manifest"], str(claim.output), kind="manifest")
    add(result["table"], str(claim.output), logical_id="mass_table.csv", kind="scalar_table")
    for record, producer in zip(records, producers):
        attempt = producer["attempt_directory"]
        case = record["frame_id"]
        name(case)  # Validate, while preserving the original selected case identity.
        for leaf, asset in record["files"].items():
            if len(PurePosixPath(leaf).parts) != 1 or PureWindowsPath(leaf).name != leaf:
                raise ValueError("Frame inventory contains an invalid filename")
            original = dict(asset, path=str(_pure(record["nas_directory"]) / leaf))
            add(original, attempt, logical_id="frames/" + case + "/" + leaf)
        for field in ("manifest", "commit", "frame_registration", "publication", "result"):
            add(producer[field], attempt)
        result_bound = _bound(producer["result"])
        result_relative = _relative(result_bound["path"], attempt)
        frame_path = Path(origin(attempt)["accessible_root"]).joinpath(*result_relative.split("/"))
        frame = _json(_verified_bytes(frame_path, result_bound, limit=MAX_METADATA_BYTES))
        from ..storage_profile import frame_storage, store_assets, dropped_after_commit
        lean = frame_storage(frame)
        if lean is not None:
            # A lean frame regenerates its omitted static fields from this
            # once-per-run store; it is a final product of its producer.
            index, archive = store_assets(lean)
            add(index, attempt, kind="operational_receipt")
            add(archive, attempt, kind="map")
        dropped = dropped_after_commit(frame)
        binding = (frame.get("prediction") or {}).get("pipeline")
        if binding is None:
            continue  # Original standalone N1 output has no native/phase stage.
        if binding.get("complete") is not True or binding.get("case_id") != record["frame_id"]:
            raise ValueError("Original native/phase stage belongs to another frame")
        if "storage" in binding:
            from ..storage_outputs import validate_publication_binding
            validate_publication_binding(binding)
        # The existing producer's exact inventory includes phase maps and the
        # temporal predecessor assets. No preparation directory is scanned.
        # Colour intermediates a lean 'drop' frame removed are not products.
        for leaf, asset in binding["files"].items():
            if leaf in dropped:
                continue
            add(asset, attempt)
    # Keep the exact original origin proof as a small immutable shared copy;
    # a later consumer must not depend on access to a predecessor's local disk.
    for key, proof in list(origins.items()):
        add(proof["proof_descriptor"], key, kind="operational_receipt")
    return rows, metadata, accessible


def _lease_reference(asset, lease):
    bound = _bound(asset)
    return dict(relative_path=_relative(bound["path"], lease),
                sha256=bound["sha256"], bytes=bound["bytes"])


def _publish_kinds(job):
    values = job.get("publish", ["film", "lossless_export"])
    if (not isinstance(values, (list, tuple))
            or any(not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value)
                   for value in values)
            or len(values) != len(set(values))):
        raise ValueError("The immutable job publish list must contain unique declared product kinds")
    return tuple(values)


def _original_frame_commit(claim, event):
    if (not isinstance(event, dict) or event.get("state") != "frame_completed"
            or event.get("complete") is not True or event.get("run") != claim.task["run"]
            or event.get("case_id") not in claim.task["ordered_case_ids"]):
        raise ValueError("An actual completed frame from this exact claimed task is required")
    record = event.get("publication")
    if not isinstance(record, dict) or record.get("frame_id") != event["case_id"]:
        raise ValueError("Original frame publication identity differs")
    original = _pure(record["nas_directory"]) / "commit.json"
    # Only the current actual owned attempt emits new progress. Old prefix
    # products are validated through their original retirement/resume proof.
    _relative(original, claim.output)
    bound = _bound(dict(record["files"]["commit.json"], path=str(original)))
    raw = _verified_bytes(Path(original), bound, limit=MAX_METADATA_BYTES)
    commit = _json(raw)
    files = {key: value for key, value in record["files"].items() if key != "commit.json"}
    if (commit.get("complete") is not True or commit.get("frame_id") != event["case_id"]
            or commit.get("manifest_sha256") != record.get("manifest_sha256")
            or commit.get("files") != files):
        raise ValueError("Original frame commit differs from its exact publication record")
    return bound


def make_publication_factory(job_root, job, machine_id, *, producer_origin_for_attempt):
    """Return the queue's explicit byte-copy callback; it grants no lease rights."""
    root = Path(job_root)
    job_path = root / "job.json"
    if read_json(job_path) != job:
        raise ValueError("Immutable job changed before publication binding")
    job_asset = descriptor(job_path)
    name(machine_id)
    kinds = _publish_kinds(job)

    def publish(claim, candidate_descriptor, success):
        if type(success) is not bool:
            raise ValueError("Publication success must be an explicit boolean")
        candidate = _bound(candidate_descriptor)
        expected = _pure(claim.output) / ("candidate_completed.json" if success else "failed.json")
        if _pure(candidate["path"]) != expected:
            raise ValueError("Publication candidate is outside this actual local attempt or has another identity")
        result = _json(_verified_bytes(candidate["path"], candidate, limit=MAX_METADATA_BYTES))
        if result.get("run") != claim.task["run"] or result.get("attempt_id") != claim.token:
            raise ValueError("Publication candidate belongs to another claim")
        bundle = Path(claim.lease) / "artifacts"
        if not success:
            if result.get("complete") is not False or not isinstance(result.get("traceback"), str):
                raise ValueError("An original failure with its verbatim traceback is required")
            # A preserved partial success bundle must never prevent the
            # original worker from publishing its subsequent literal failure.
            target = _target(_fresh_root(Path(claim.lease) / "failure_artifacts"), "failed.json")
            _copy(candidate["path"], {"sha256": candidate["sha256"], "bytes": candidate["bytes"]}, target)
            return dict(candidate=_lease_reference(dict(candidate, path=str(target)), claim.lease))
        if claim.task.get("pipeline") == "preset-wasserstein":
            from ..preset_wasserstein_protocol import validate_preset_wasserstein_claim
            from ..preset_wasserstein_results import validate_preset_wasserstein_result
            validate_preset_wasserstein_claim(claim.task, job)
            validate_preset_wasserstein_result(result, task=claim.task, job=job)
        elif claim.task.get("pipeline") == "native-phase-statistics":
            from .phase1_protocol import validate_phase1_claim
            validate_phase1_claim(claim.task, job)
        elif claim.task.get("pipeline") == "series-setup":
            from .series_setup_protocol import validate_setup_claim
            from .series_setup_results import validate_setup_result
            validate_setup_claim(claim.task, job)
            validate_setup_result(result, task=claim.task, job=job)
        else:
            if (result.get("task") != claim.task or result.get("commit") != claim.task["commit"]
                    or result.get("calibration_sha256") != claim.task["calibration_sha256"]):
                raise ValueError("Original Main completion task, code or calibration identity differs")
        products, metadata, accessible = extract_products(
            result, claim, machine_id, producer_origin_for_attempt=producer_origin_for_attempt)
        index = publish_run_metadata(
            bundle, job=job_asset, run=claim.task["run"], attempt_id=claim.token,
            machine_id=machine_id, completed=candidate, products=products,
            metadata_product_ids=metadata, file_writers_closed=True,
            all_final_products_listed=True, source_paths=accessible)
        copied = published_metadata_descriptor(index)
        prepared = dict(candidate=_lease_reference(copied, claim.lease),
                        products_index=_lease_reference(index, claim.lease))
        selected = {row["product_id"] for row in products if row["kind"] in kinds}
        if "completion" in kinds:
            selected.add("completed.json")  # The helper indexes this original candidate itself.
        if selected:
            payload = publish_indexed_products(
                index, bundle / "default_products", current_machine_id=machine_id,
                publish=kinds, source_paths={key: value for key, value in accessible.items() if key in selected},
                file_writers_closed=True)
            prepared["payload_publication"] = _lease_reference(payload, claim.lease)
        return prepared

    def publish_progress(claim, event):
        if event.get("state") == "phase1_frame_completed":
            from .phase1_protocol import phase1_frame_commit
            original = _bound(phase1_frame_commit(claim, event))
            leaf = "phase1_commit.json"
        else:
            original = _original_frame_commit(claim, event)
            leaf = "commit.json"
        root = _fresh_root(Path(claim.lease) / "progress_artifacts" / name(event["case_id"]))
        target = _target(root, leaf)
        _copy(original["path"], {"sha256": original["sha256"], "bytes": original["bytes"]}, target)
        return _lease_reference(dict(original, path=str(target)), claim.lease)

    publish.publish_progress = publish_progress
    return publish
