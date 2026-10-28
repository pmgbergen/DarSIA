"""Positive phase-statistics identities and original queue publication adapters.

This private candidate adds no worker program, job creator, fitted calibration,
or resume implementation. The existing queue alone owns claims and leases.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

PIPELINE = "native-phase-statistics"
MANIFEST_FORMAT = "darsia-phase-statistics-run-manifest-v1"
RESULT_FORMAT = "darsia-phase-statistics-worker-result-v1"
COMPLETION_FORMAT = "darsia-phase-statistics-completion-v1"
MAP_FIELD = "phase1_configurations"
MAP_SHA_FIELD = "phase1_configurations_sha256"


def is_phase1(value):
    return isinstance(value, dict) and value.get("pipeline") == PIPELINE


def _sha(value, length, label):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{%d}" % length, value) is None:
        raise ValueError("Explicit phase-one " + label + " identity required")


def _same(left, right):
    from .staged_data import _same_json
    return _same_json(left, right)


def configuration_map_sha(configurations):
    """Digest the explicit root-alias configuration map, never call it a K SHA."""
    from .job_queue import name
    from ..store_map import validate_input_reference
    if not isinstance(configurations, dict) or not configurations:
        raise ValueError("Explicit nonempty phase-one configuration map required")
    if len({name(run) for run in configurations}) != len(configurations):
        raise ValueError("Distinct phase-one run names required")
    for value in configurations.values():
        validate_input_reference(value)
    raw = json.dumps(configurations, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_phase1_job(job):
    """Require one actual stage, commit, precision and immutable per-run map."""
    if (not is_phase1(job) or job.get("format") != "darsia-alias-job-v1"
            or "calibration_sha256" in job):
        raise ValueError("Phase-one job requires its positive alias stage and no fitted calibration SHA")
    _sha(job.get("commit"), 40, "code commit")
    if job.get("precision") not in {"float32", "float64"}:
        raise ValueError("One explicit phase-one precision required")
    actual = configuration_map_sha(job.get(MAP_FIELD))
    if job.get(MAP_SHA_FIELD) != actual:
        raise ValueError("Phase-one configuration-map SHA differs")
    if (not isinstance(job.get("expected_runs"), list)
            or set(job["expected_runs"]) != set(job[MAP_FIELD])
            or len(job["expected_runs"]) != len(job[MAP_FIELD])):
        raise ValueError("Phase-one expected runs differ from the explicit configuration map")
    return job


def validate_phase1_task(task):
    if (not is_phase1(task) or task.get("format") != "darsia-alias-task-v1"
            or "calibration_sha256" in task):
        raise ValueError("Phase-one task requires its positive alias stage and no fitted calibration SHA")
    _sha(task.get("commit"), 40, "code commit")
    _sha(task.get(MAP_SHA_FIELD), 64, "configuration-map")
    if type(task.get("threads")) is not int or task["threads"] < 1:
        raise ValueError("Positive phase-one internal thread count required")
    return task


def bind_phase1_identity(job, tasks):
    """Pure pre-serialization binding for explicitly supplied alias DATA tasks.

    This does not construct a catalogue, infer DATA, create job files, or claim
    execution readiness. The existing producer still needs its positive stage
    configuration/catalogue route and its separate numeric acceptance gates.
    """
    from .job_queue import validate_task
    if not isinstance(tasks, (list, tuple)) or not tasks:
        raise ValueError("Explicit phase-one alias task list required")
    result = deepcopy(job)
    if "calibration_sha256" in result:
        raise ValueError("Do not relabel an existing calibrated mass job as phase one")
    copied = deepcopy(list(tasks))
    mapping = {}
    for task in copied:
        if task.get("pipeline") != PIPELINE or "calibration_sha256" in task:
            raise ValueError("Each task must already declare genuine phase-one DATA")
        if task["run"] in mapping:
            raise ValueError("Repeated phase-one run")
        mapping[task["run"]] = deepcopy(task["configuration"])
    result.update(pipeline=PIPELINE, **{MAP_FIELD: mapping, MAP_SHA_FIELD: configuration_map_sha(mapping)})
    validate_phase1_job(result)
    for task in copied:
        task[MAP_SHA_FIELD] = result[MAP_SHA_FIELD]
        validate_task(task)
        validate_phase1_claim(task, result)
    return result, copied


def validate_phase1_claim(task, job):
    validate_phase1_job(job)
    validate_phase1_task(task)
    if (task.get("commit") != job["commit"] or task.get("precision") != job["precision"]
            or task[MAP_SHA_FIELD] != job[MAP_SHA_FIELD]
            or task.get("run") not in job[MAP_FIELD]
            or not _same(task.get("configuration"), job[MAP_FIELD][task["run"]])):
        raise ValueError("Phase-one claimed code, precision or configuration-map binding differs")


def phase1_resume_source(shared, claim):
    """Preserve original lineage/retirement checks, reject any scientific prefix."""
    from .checkpoint import resume_source_for_claim
    from .store_runtime import RunUnavailable
    source = resume_source_for_claim(shared, claim)
    if source is not None:
        raise RunUnavailable("Phase-one saved-prefix resume is not admitted; preserve the prefix without replay")
    return None


def worker_result(result, *, task, job, commit, owner, token):
    validate_phase1_claim(task, job)
    if (result.get("format") != RESULT_FORMAT or not is_phase1(result)
            or result.get("complete") is not True or result.get("whole_run") is not True
            or result.get("worker_references_closed") is not True
            or result.get("run") != task["run"] or result.get("precision") != task["precision"]
            or result.get("fitted_mass_calibration_read") is not False
            or type(result.get("mass_evaluation_calls")) is not int or result["mass_evaluation_calls"] != 0
            or "calibration_sha256" in result or commit != task["commit"]
            or token != task["attempt_id"]):
        raise ValueError("A genuine closed phase-one worker result is required")
    return dict(result, actual_worker=deepcopy(owner), commit=commit, attempt_id=token,
                task=deepcopy(task), **{MAP_SHA_FIELD: task[MAP_SHA_FIELD]})


def _read_owned(asset, output):
    from .publication_runtime import _relative
    from .products_publication import _bound, _verified_bytes, _json, MAX_METADATA_BYTES
    bound = _bound(asset)
    _relative(bound["path"], output)
    return _json(_verified_bytes(Path(bound["path"]), bound, limit=MAX_METADATA_BYTES))


def _manifest(claim, asset):
    from .publication_runtime import _pure
    task = claim.task
    validate_phase1_task(task)
    if _pure(asset["path"]) != _pure(claim.output) / "manifest.json":
        raise ValueError("Phase-one manifest is outside its actual owned attempt")
    value = _read_owned(asset, claim.output)
    expected = dict(pipeline=PIPELINE, configuration=deepcopy(task["configuration"]),
        run=task["run"], commit=task["commit"], precision=task["precision"],
        backend=value.get("settings", {}).get("backend"), threads=task["threads"],
        attempt_id=claim.token, **{MAP_SHA_FIELD: task[MAP_SHA_FIELD]})
    config = value.get("configuration", {})
    code = value.get("native_code_admission", {}).get("entries")
    if (value.get("format") != MANIFEST_FORMAT or value.get("run") != task["run"]
            or value.get("job_id") != task["job_id"]
            or not _same(value.get("stage_identity"), expected)
            or value.get("code") != dict(commit=task["commit"], identity_algorithm="sha256-git-commit-ascii")
            or value.get("commit_identity_digest") != hashlib.sha256(task["commit"].encode("ascii")).hexdigest()
            or value.get("settings", {}).get("precision") != task["precision"]
            or not _same(value.get("settings", {}).get("threads"), task["threads"])
            or expected["backend"] not in {"cpu", "cuda"}
            or task.get("backend", "any") not in {"any", expected["backend"]}
            or config.get("sha256") != task["configuration"]["sha256"]
            or config.get("size_bytes", config.get("bytes")) != task["configuration"]["bytes"]
            or not isinstance(code, list) or not code
            or any(not isinstance(row, dict) or row.get("commit") != task["commit"] for row in code)
            or not isinstance(value.get("frames"), dict)
            or set(value["frames"]) != set(task["ordered_case_ids"])):
        raise ValueError("Phase-one manifest differs from the actual claimed stage and original source admission")
    return value


def phase1_frame_commit(claim, event):
    """Read only the exact newly committed case metadata in this owned attempt."""
    from .job_queue import descriptor
    from .publication_runtime import _pure
    from .staged_data import _frame_record
    case = event.get("case_id") if isinstance(event, dict) else None
    if (not isinstance(event, dict) or event.get("state") != "phase1_frame_completed"
            or event.get("stage") != PIPELINE or event.get("complete") is not True
            or event.get("run") != claim.task["run"] or case not in claim.task["ordered_case_ids"]):
        raise ValueError("Actual committed phase-one progress from this claim required")
    bound = event["phase1_commit"]
    if _pure(bound["path"]) != _pure(claim.output) / "preparation" / case / "phase1_commit.json":
        raise ValueError("Phase-one progress points outside its original case")
    manifest = _manifest(claim, descriptor(Path(claim.output) / "manifest.json"))
    record = _frame_record(_read_owned(bound, claim.output), manifest["frames"][case],
                           manifest["configuration"], claim.task["precision"], bound)
    if not _same(event.get("frame_index"), record["frame_index"]):
        raise ValueError("Phase-one progress frame index differs")
    return bound


def shared_phase1_progress(queue, claim, event):
    """Same original ownership fence and byte-copy proof; no foreign-local read."""
    from .job_queue import _own_asset
    from .products_publication import _bound
    original = _bound(phase1_frame_commit(claim, event))
    publisher = getattr(claim.publication_factory, "publish_progress", None)
    if not callable(publisher):
        raise RuntimeError("Phase-one progress requires the original lease byte-copy publisher")
    queue.assert_owner(claim)
    reference = publisher(claim, deepcopy(event))
    queue.assert_owner(claim)
    copied = _own_asset(claim.lease, reference)
    if reference["sha256"] != original["sha256"] or reference["bytes"] != original["bytes"]:
        raise ValueError("Shared phase-one progress is not the exact committed bytes")
    # _own_asset validates the shared copy itself; no producer path is followed
    # by the reclaimer, which retains its original server-file-time lease clock.
    if not copied.is_file():
        raise ValueError("Actual shared phase-one commit copy is missing")
    return deepcopy(reference)


def extract_phase1_products(result, claim, machine_id):
    """Index genuine phase-one files; never synthesize Main mass result fields."""
    from .job_queue import name
    from .publication_runtime import _relative, _pure, _kind, _METADATA
    from .products_publication import _bound, product
    from .staged_data import _frame_record, _keys
    if (result.get("format") != RESULT_FORMAT or not is_phase1(result)
            or result.get("complete") is not True or result.get("whole_run") is not True
            or result.get("worker_references_closed") is not True
            or result.get("run") != claim.task["run"] or result.get("attempt_id") != claim.token
            or not _same(result.get("task"), claim.task) or result.get("commit") != claim.task["commit"]
            or result.get(MAP_SHA_FIELD) != claim.task[MAP_SHA_FIELD]
            or "calibration_sha256" in result):
        raise ValueError("A closed genuine phase-one completion for this exact claim is required")
    manifest = _manifest(claim, result["manifest"])
    core_bound = result["phase1_completion"]
    if _pure(core_bound["path"]) != _pure(claim.output) / "phase1_completed.json":
        raise ValueError("Phase-one core completion is outside its actual attempt")
    core = _read_owned(core_bound, claim.output)
    _keys(core, {"format", "complete", "run", "configuration", "precision", "code_commit",
                 "ordered_case_ids", "frames", "history_complete"}, {"opencv_ipp"}, where="phase-one core completion")
    if "opencv_ipp" in core:
        from darsia._opencv_runtime import validate_observation
        validate_observation(core["opencv_ipp"])
    cases = claim.task["ordered_case_ids"]
    if (core["format"] != COMPLETION_FORMAT or core["complete"] is not True
            or core["history_complete"] is not True or core["run"] != claim.task["run"]
            or not _same(core["configuration"], manifest["configuration"])
            or core["precision"] != claim.task["precision"] or core["code_commit"] != claim.task["commit"]
            or not _same(core["ordered_case_ids"], cases) or not _same(result.get("ordered_case_ids"), cases)
            or type(result.get("case_count")) is not int or result["case_count"] != len(cases)
            or not isinstance(core["frames"], list) or len(core["frames"]) != len(cases)
            or not _same(core["frames"], result.get("phase1_frames"))
            or result.get("backend") != manifest["settings"]["backend"]
            or result.get("precision") != core["precision"]
            or not _same(result.get("configuration"), manifest["configuration"])
            or result.get("fitted_mass_calibration_read") is not False
            or type(result.get("mass_evaluation_calls")) is not int or result["mass_evaluation_calls"] != 0):
        raise ValueError("Phase-one core completion differs from its actual manifest and case order")
    rows, metadata, accessible, ids = [], [], {}, set()

    def add(asset, logical, kind):
        bound = _bound(asset)
        _relative(bound["path"], claim.output)
        if logical.casefold() in ids:
            raise ValueError("Repeated phase-one product identity")
        ids.add(logical.casefold())
        rows.append(product(logical, machine_id=machine_id, local_path=bound["path"],
                            sha256=bound["sha256"], bytes=bound["bytes"], kind=kind))
        accessible[logical] = Path(bound["path"])
        if kind in _METADATA:
            metadata.append(logical)

    add(result["manifest"], "manifest.json", "manifest")
    add(core_bound, "phase1_completed.json", "operational_receipt")
    for case, bound in zip(cases, core["frames"]):
        name(case)
        if _pure(bound["path"]) != _pure(claim.output) / "preparation" / case / "phase1_commit.json":
            raise ValueError("Phase-one core commit is outside its original case")
        record = _frame_record(_read_owned(bound, claim.output), manifest["frames"][case],
                               manifest["configuration"], core["precision"], bound)
        add(bound, "phase1/" + case + "/phase1_commit.json", "commit_receipt")
        for leaf, asset in record["files"].items():
            kind = _kind(leaf)
            if leaf == "storage_gas_probability_before_rules.png":
                kind = "phase_before_png"
            elif leaf == "registered_original.png":
                kind = "registered_original_photo"
            add(asset, "phase1/" + case + "/" + leaf, kind)
    return rows, metadata, accessible
