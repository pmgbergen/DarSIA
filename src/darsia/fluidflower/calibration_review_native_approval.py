"""Qualify an explicit native run review for the original append-only approval.

The series' original locked choice, current Jobs terminal validator and shared
publication readers remain authoritative. This adapter admits metadata only:
it neither interprets chemistry nor loads arrays, adopts a fit or starts work.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

from .calibration_review_native import NativeSavedReview
from .calibration_review_native_catalog import (
    _NATIVE_PIPELINES, _bound_owned, _bytes_identity, _registered_series_id, _series_key,
)
from .jobs_metadata import JobSnapshotReader, _key as _metadata_path_key
from .pipeline import products_publication as publication
from .pipeline.job_queue import name
from .published_jobs_snapshot import _SnapshotMetadata
from .series_calibration_approval import read_approval
from .storage_saved_products import _json


EVIDENCE_FORMAT = "darsia-native-calibration-run-review-v1"
_SERIES_MANIFEST_LIMIT = 4 * 1024 * 1024


def _metadata_descriptor(path, *, limit=None):
    """Pin bounded metadata without the queue descriptor's unbounded allocation."""
    limit = publication.MAX_METADATA_BYTES if limit is None else limit
    path = publication._file(path)
    before = path.stat()
    if before.st_size > limit:
        raise ValueError("Metadata exceeds its bounded read: " + str(path))
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    after = path.stat()
    if len(raw) > limit or len(raw) != before.st_size or publication._stamp(before) != publication._stamp(after):
        raise ValueError("Metadata changed or exceeded its bounded read: " + str(path))
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


class _ApprovalSnapshotReader(JobSnapshotReader):
    """Retain original snapshot admission and rehash its observed metadata."""

    def __init__(self):
        super().__init__()
        self.observed = {}

    def read(self, path, expected=None):
        bound = publication._bound(expected if expected is not None else _metadata_descriptor(path))
        bound = dict(path=bound["path"], sha256=bound["sha256"], size_bytes=bound["bytes"])
        if _metadata_path_key(path) != _metadata_path_key(bound["path"]):
            raise ValueError("Metadata path differs from its descriptor")
        bound["path"] = str(Path(path).resolve())
        key = _metadata_path_key(bound["path"])
        previous = self.observed.get(key)
        if previous is not None and _bytes_identity(previous) != _bytes_identity(bound):
            raise ValueError("Native approval source metadata changed during review")
        # Retain the original JobSnapshotReader's state/owner qualification,
        # using the original publication reader for strictly bounded JSON I/O.
        raw = publication._verified_bytes(path, publication._bound(bound), limit=publication.MAX_METADATA_BYTES)
        value = publication._json(raw)
        self.observed[key] = previous or bound
        return value

    def finish(self):
        # Rehash exact bytes through the original bounded publication reader;
        # never reuse a stat-only cached JSON result.
        for bound in self.observed.values():
            publication._verified_bytes(bound["path"], publication._bound(bound), limit=publication.MAX_METADATA_BYTES)


def _action_current(action_guard):
    if action_guard is not None and action_guard() is not True:
        raise ValueError("The displayed native review changed; choose and review it again")


def _phase_source_task(registrations, actual_series, run, phase1, reader):
    """Select the exact original complete phase-one task from explicit registrations."""
    from .pipeline.phase1_protocol import PIPELINE
    candidates = {}
    for registration in registrations:
        if not isinstance(registration, dict) or not isinstance(registration.get("job"), dict):
            raise ValueError("Phase-one source registration lacks its immutable job descriptor")
        root = Path(registration["root"])
        candidate_reader = _ApprovalSnapshotReader()
        job = candidate_reader.read(root / "job.json", registration["job"])
        if (job.get("format") != "darsia-alias-job-v1" or job.get("pipeline") != PIPELINE
                or _series_key(job.get("series_id")) != _series_key(actual_series)):
            continue
        if run not in job.get("expected_runs", []):
            continue
        snapshot = candidate_reader(deepcopy(registration))
        if not any(row.get("run") == run and row.get("state") == "Complete"
                   for row in snapshot.get("runs", [])):
            continue
        if snapshot.get("job_facts") != job or snapshot.get("errors"):
            raise ValueError("The original phase-one source has no currently qualified complete result")
        task = candidate_reader.read(root / "results" / name(run) / "task.json")
        if _bytes_identity(task.get("configuration", {})) != _bytes_identity(phase1["configuration"]):
            continue
        done = candidate_reader.read(root / "done" / (name(run) + ".json"))
        index = _bound_owned(root, done["products_index"])
        completed_bound = _bound_owned(root, done["completed"])
        completed = candidate_reader.read(completed_bound["path"], completed_bound)
        if _bytes_identity(completed["phase1_completion"]) != _bytes_identity(phase1["completion"]):
            continue
        index_root, document, products = publication._read_index(index)
        if (document["run"] != run or document["attempt_id"] != done["attempt_id"]
                or _bytes_identity(document["job"]) != _bytes_identity(registration["job"])
                or _bytes_identity(products["completed.json"]) != _bytes_identity(completed_bound)):
            raise ValueError("Phase-one source index differs from its current original publication")
        source_metadata = _SnapshotMetadata(index, index_root, products)
        for key, product in products.items():
            if product.get("metadata_copy") is not None:
                source_metadata.read(key)
        source_metadata.finish()
        candidate_reader.finish()
        identity = (_bytes_identity(registration["job"]), _bytes_identity(index),
                    _bytes_identity(completed_bound), done["attempt_id"])
        candidates.setdefault(identity, (task, dict(job=deepcopy(registration["job"]),
            index=index, completion=completed_bound, attempt_id=done["attempt_id"]),
            source_metadata, candidate_reader.observed))
    if len(candidates) != 1:
        raise ValueError("Native approval needs one exact registered complete phase-one source; missing or ambiguous source")
    task, source_job, source_metadata, observed = next(iter(candidates.values()))
    for bound in observed.values():
        reader.read(bound["path"], bound)
    return task, source_job, source_metadata


def _source_reference(task, original, use, *, role):
    """Require an exact original consumer role and immutable byte identity."""
    from .pipeline.job_inputs import validate_catalog
    catalog = validate_catalog(task["input_catalog"])
    if catalog["run"] != task["run"] or catalog["configuration"] != task["configuration"]:
        raise ValueError("Original source input catalogue differs from its task")
    rows = [row for row in catalog["inputs"] if use in row["uses"]]
    if len(rows) != 1:
        raise ValueError("No unique original source input consumer: " + use)
    reference = rows[0]["reference"]
    if use == "run.configuration" and reference != task["configuration"]:
        raise ValueError("Original run.configuration consumer differs from its task reference")
    same_bytes = _bytes_identity(reference) == _bytes_identity(original)
    if role == "model":
        # Original InputCatalog model declarations may omit a byte count; its
        # admitted reference always supplies the exact physical size to verify.
        same_bytes = reference["sha256"] == original["sha256"] and all(
            type(original[key]) is int and original[key] == reference["bytes"]
            for key in ("size_bytes", "bytes") if key in original)
    if reference["role"] != role or not same_bytes:
        raise ValueError("Original source " + role + " differs from its input catalogue: " + use)
    return reference


def _mapped_metadata(task, original, use, stores, series_id, reader, *, role="configuration"):
    """Resolve one named original consumer; never probe the producer's path."""
    from .pipeline.store_runtime import _available_input
    reference = _source_reference(task, original, use, role=role)
    physical = _available_input(reference, stores, series_id, stores.roots(series_id))
    bound = dict(path=str(physical), sha256=reference["sha256"], size_bytes=reference["bytes"])
    return bound, reader.read(physical, bound)


def _mapped_model_phase(task, phase, stores, series_id, reader):
    """Map exactly the model consumers declared by the original InputCatalog."""
    from .raw_production_review import _assets

    mapped = deepcopy(phase)
    bundles = mapped["model_bundles"]
    original_assets = deepcopy(list(_assets(bundles)))
    consumers = []
    for role in ("native", "edge"):
        item = bundles[role]
        for key, digest in item["requirements"]["expected_sha256"].items():
            descriptor = item["documents"][key]
            if descriptor["sha256"] != digest:
                raise ValueError("Original model requirement differs from its declared descriptor")
            consumers.append(("phase.model." + role + "." + key, descriptor))
    placement = bundles["placement"]
    for key in ("pixel_model", "regional_model"):
        consumers.append(("phase.placement." + key, placement[key]))
    if placement.get("projected_models") is not None:
        for key in ("pixel", "regional"):
            consumers.append(("phase.placement.projected." + key, placement["projected_models"][key]))
    # _origin recursively visits every declared descriptor. Refuse an unnamed
    # declaration before allowing that reader to see any producer literal.
    declared = {id(descriptor) for descriptor in _assets(bundles)}
    if declared != {id(descriptor) for _use, descriptor in consumers}:
        raise ValueError("A declared model descriptor has no original named input consumer")
    model_consumers = [dict(use=use, reference=deepcopy(_source_reference(task, descriptor, use, role="model")))
                       for use, descriptor in consumers]
    for use, descriptor in consumers:
        physical, _document = _mapped_metadata(task, descriptor, use, stores, series_id,
                                              reader, role="model")
        descriptor["path"] = physical["path"]
        descriptor["size_bytes"] = physical["size_bytes"]
    return mapped, original_assets, model_consumers


def _phase_source_binding(selection, manifest, completion, native_review, reader, *,
                          series_root, actual_series, source_registrations, stores_path):
    """Use the existing single-origin proof, never equate different model hashes.

    Saved N1 metadata binds the original phase-one configuration. That exact
    metadata and its phase configuration must resolve through their original
    source task's input catalogue and this receiver's explicit machine stores;
    there is no nearby-file search or foreign producer-path probe. Mixed
    model origins retain the original origin reader's explicit refusal.
    """
    from . import storage_n1_saved_products as native
    from . import storage_saved_products as saved
    from .pipeline.staged_data import validate_phase1_run
    from .raw_production_review import _origin
    from .storage_indexed_products import indexed_asset_resolver
    from .store_map import read_store_map

    phase1 = manifest["storage"]["phase1"]
    source_task, source_job, source_metadata = _phase_source_task(source_registrations, actual_series,
        native_review.expected_run, phase1, reader)
    stores = read_store_map(stores_path, forbidden_roots=(Path(series_root), Path(__file__).resolve().parents[3]))
    if stores.machine_id.casefold() != native_review.current_machine_id.casefold():
        raise ValueError("Source store mapping belongs to another physical machine")
    store_bound = {key: stores.source[key] for key in ("path", "sha256", "size_bytes")}
    physical_config, config = _mapped_metadata(source_task, phase1["configuration"],
        "run.configuration", stores, actual_series, reader)
    config = validate_phase1_run(config)
    if (config["run"] != native_review.expected_run
            or config["precision"] != manifest["settings"]["precision"]
            or config["ordered_case_ids"] != list(native_review.case_ids)):
        raise ValueError("Original phase-one source configuration differs from the reviewed native run")
    phase_bound = config["phase"]["configuration"]
    physical_phase, phase = _mapped_metadata(source_task, phase_bound, "phase.configuration",
        stores, actual_series, reader)
    mapped_phase, model_assets, model_consumers = _mapped_model_phase(source_task, phase, stores, actual_series, reader)
    model, retained_identity, physical_model_assets = _origin(selection["phase_model"]["root"], mapped_phase)
    if model != Path(selection["phase_model"]["root"]).resolve():
        raise ValueError("Original phase model origin differs from the current locked series choice")
    resolve = indexed_asset_resolver(native_review.index,
        current_machine_id=native_review.current_machine_id, collections=native_review.collections,
        source_paths=native_review.source_paths)
    specs = {spec["case_id"]: spec for spec in config["frames"]}
    for record in completion["records"]:
        receipt = saved._record_asset(record, "storage_completed.json")
        document = saved._json(saved._read_bytes(receipt, resolve, []))
        case = record["frame_id"]
        committed = native._phase1(document, manifest, manifest["frames"][case], resolve=resolve, loaded=[])
        if (committed["source"] != specs[case]["source"]
                or committed["frame_index"] != specs[case]["frame_index"]
                or committed["elapsed_h"] != specs[case]["elapsed_h"]):
            raise ValueError("Original phase-one acquisition differs from the displayed native source")
    source_metadata.finish()
    # Preserve both original identity vocabularies as separate facts. Only
    # _origin establishes correspondence; their different digests are never
    # compared as if they were interchangeable.
    return dict(configuration=deepcopy(phase1["configuration"]),
                phase_configuration=deepcopy(phase_bound), model_root=str(model),
                retained_model_identity=list(retained_identity), model_assets=model_assets,
                model_consumers=model_consumers,
                source_job=source_job, stores=store_bound,
                physical_configuration=physical_config, physical_phase_configuration=physical_phase,
                mapped_phase=mapped_phase, physical_model_assets=physical_model_assets)


def prepare_native_run_approval(series_root, series_id, run, native_review, registration, *,
                                source_registrations=(), stores_path=None):
    """Capture exact current review evidence, without creating an approval.

    Registration must be the original explicit Jobs registration retained with
    this displayed attachment. A display label or matching path is insufficient.
    The returned binding must be retained until the explicit approval action;
    recreating it at that action would discard the stale-review guard.
    """
    if type(native_review) is not NativeSavedReview:
        raise ValueError("Choose an admitted original native saved result")
    if native_review.pipeline != "n1":
        raise ValueError("Native phase results have no original N1 phase-one source approval contract")
    if not isinstance(registration, dict) or not isinstance(registration.get("job"), dict):
        raise ValueError("Native approval requires the displayed result's original job registration")
    root = Path(series_root).resolve(strict=True)
    series_manifest = _metadata_descriptor(root / "series.toml", limit=_SERIES_MANIFEST_LIMIT)
    actual_series = _registered_series_id(root, series_id)
    locked = read_approval(root, verify=True)
    if locked is None or locked.get("locked") is not True:
        raise ValueError("Use and lock the reviewed mass calibration for this series before approving a native run")
    selection = locked["binding"]
    calibration = dict(reference=selection["mass_calibration"],
                       sha256=selection["calibration"]["sha256"])
    if (not isinstance(run, str) or not run
            or run.casefold() != native_review.expected_run.casefold()):
        raise ValueError("The displayed native result belongs to another run")
    if calibration["sha256"] != native_review.expected_calibration_sha256:
        raise ValueError("The displayed native result differs from the series' locked calibration")
    run = native_review.expected_run
    job_root = Path(registration["root"])
    reader = _ApprovalSnapshotReader()
    job = reader.read(job_root / "job.json", registration["job"])
    if (job.get("format") != "darsia-alias-job-v1"
            or job.get("pipeline", "n1") not in _NATIVE_PIPELINES
            or job.get("pipeline", "n1") != native_review.pipeline
            or _series_key(job.get("series_id")) != _series_key(actual_series)
            or job.get("calibration_sha256") != calibration["sha256"]
            or run not in job.get("expected_runs", [])):
        raise ValueError("Original native job differs from the selected series, run or locked calibration")
    snapshot = reader(deepcopy(registration))
    rows = snapshot.get("runs")
    if (snapshot.get("job_facts") != job or snapshot.get("errors")
            or not isinstance(rows, list)
            or [row.get("run") for row in rows] != job.get("expected_runs")
            or not any(row.get("run") == run and row.get("state") == "Complete" for row in rows)):
        raise ValueError("Original native job has no currently qualified complete result for this run")
    safe_run = name(run)
    task = reader.read(job_root / "results" / safe_run / "task.json")
    done_path = job_root / "done" / (safe_run + ".json")
    done = reader.read(done_path)
    if (task.get("pipeline", "n1") != native_review.pipeline
            or task.get("run") != run or task.get("calibration_sha256") != calibration["sha256"]
            or done.get("format") != "darsia-alias-done-v1" or done.get("run") != run
            or done.get("completed", {}).get("relative_path") != "results/" + safe_run + "/completed.json"):
        raise ValueError("Current native task or done record differs from the displayed run")
    index = _bound_owned(job_root, done["products_index"])
    if index != native_review.index:
        raise ValueError("The current native publication differs from the result that was displayed")
    completed_bound = _bound_owned(job_root, done["completed"])
    completion = reader.read(completed_bound["path"], completed_bound)
    index_root, index_document, products = publication._read_index(index)
    if (index_document["run"] != run or index_document["attempt_id"] != done.get("attempt_id")
            or _bytes_identity(index_document["job"]) != _bytes_identity(registration["job"])
            or _bytes_identity(products["completed.json"]) != _bytes_identity(completed_bound)
            or completion.get("complete") is not True or completion.get("run") != run
            or completion.get("calibration", {}).get("sha256") != calibration["sha256"]):
        raise ValueError("Current native publication no longer matches its original job/completion/calibration")
    metadata = _SnapshotMetadata(index, index_root, products)
    for key, product in products.items():
        if product.get("metadata_copy") is not None:
            metadata.read(key)  # Includes gzip metadata as bytes; never decode it as JSON.
    manifest = _json(metadata.read("manifest.json")[1])
    fresh = NativeSavedReview(index, current_machine_id=native_review.current_machine_id,
        expected_run=run, expected_calibration_sha256=calibration["sha256"],
        expected_pipeline=native_review.pipeline,
        collections=native_review.collections, source_paths=native_review.source_paths,
        geometry_producer_sha256=native_review.geometry_producer_sha256)
    if (fresh.case_ids != native_review.case_ids
            or any(fresh.frame_metadata(i) != native_review.frame_metadata(i)
                   for i in range(len(fresh.case_ids)))):
        raise ValueError("The native source/frame metadata differs from the displayed review")
    source_binding = _phase_source_binding(selection, manifest, completion, native_review, reader,
        series_root=root, actual_series=actual_series, source_registrations=source_registrations,
        stores_path=stores_path)
    metadata.finish()
    reader.finish()
    for asset in source_binding["physical_model_assets"]:
        publication._verify_file(asset["path"], publication._bound(asset), limit=asset["size_bytes"])
    from .store_map import LIMIT as store_limit
    if _metadata_descriptor(source_binding["stores"]["path"], limit=store_limit) != source_binding["stores"]:
        raise ValueError("The source machine store mapping changed during review")
    if (_metadata_descriptor(root / "series.toml", limit=_SERIES_MANIFEST_LIMIT) != series_manifest
            or read_approval(root, verify=True) != locked):
        raise ValueError("The series choice changed while the native review was checked")
    # Receiver aliases guard this action, but do not change the original result
    # that an approval certifies on another correctly mapped receiver.
    source_access = {key: source_binding.pop(key) for key in
                     ("stores", "physical_configuration", "physical_phase_configuration",
                      "model_root", "mapped_phase", "physical_model_assets")}
    evidence = dict(format=EVIDENCE_FORMAT, series_id=actual_series, run=run,
        job=deepcopy(registration["job"]), index=index, completion=completed_bound,
        attempt_id=done["attempt_id"], manifest=deepcopy(completion["manifest"]),
        phase1={key: deepcopy(manifest["storage"]["phase1"][key])
                for key in ("configuration", "completion", "code_commit")},
        source_model=source_binding,
        ordered_case_ids=list(fresh.case_ids), series_approval_binding=deepcopy(selection))
    return dict(calibration=calibration, result=index["path"], run=run,
                native_evidence=evidence, series_manifest=series_manifest, source_access=source_access,
                observed_metadata=sorted(reader.observed.values(), key=lambda bound: bound["path"]))


def approve_native_run(series_root, series_id, run, native_review, registration, *,
                       expected_binding, reviewer=None, now=None, retry=None, action_guard=None,
                       source_registrations=(), stores_path=None):
    """Recheck the captured review at each append attempt, then use the original writer."""
    from .calibration_approvals import approve_run

    expected = deepcopy(expected_binding)
    if not isinstance(expected, dict) or expected.get("native_evidence", {}).get("format") != EVIDENCE_FORMAT:
        raise ValueError("Prepare and retain this native review before approving it")

    def guard():
        _action_current(action_guard)
        actual = prepare_native_run_approval(series_root, series_id, run, native_review, registration,
            source_registrations=source_registrations, stores_path=stores_path)
        if actual != expected:
            raise ValueError("The native review or series choice changed since it was shown")
        _action_current(action_guard)

    guard()
    return approve_run(series_root, expected["run"], expected["calibration"],
        result=expected["result"], reviewer=reviewer, now=now, retry=retry,
        native_evidence=expected["native_evidence"], before_append=guard)
