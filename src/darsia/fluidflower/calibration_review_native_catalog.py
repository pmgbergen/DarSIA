"""Passive native review discovery through explicitly registered Jobs metadata.

The original snapshot reader qualifies completion; the original shared-asset
resolver binds the exact done publication. This module never constructs a
queue, searches result trees, resolves input aliases, collects products, reads
frame arrays or infers the machine on which it is running.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path, PureWindowsPath
import tomllib

from .calibration_review_native import NativeSavedReview, completion_pipeline
from .jobs_metadata import JobSnapshotReader
from .pipeline import products_publication as publication
from .pipeline.job_queue import _own_asset, name


# Original task dispatch literals from pipeline.series.pipeline_kind. Only an
# absent TASK pipeline selects compatible N1; prepare_alias_job omits this
# field from the job envelope for both supported protocols.
_NATIVE_PIPELINES = {"n1", "native-phase-n1"}


def _bound_owned(root, reference):
    path = _own_asset(root, reference)
    # Preserve the original done descriptor; do not adopt a new digest merely
    # because an unrelated current file can be opened or hashed.
    return {"path": str(path), "sha256": reference["sha256"], "size_bytes": reference["bytes"]}


def _bytes_identity(reference):
    return reference["sha256"], reference.get("size_bytes", reference.get("bytes"))


def _series_key(value):
    # Same bounded spelling equivalence used by the review approval boundary:
    # e.g. the UI's ffac and the scientific job's FF_AC, never fuzzy matching.
    return value.casefold().replace("_", "") if isinstance(value, str) else None


def _registered_series_id(series_root, selected_series_id):
    # Match NativeJobsWorkspace.set_series: the selected folder's schema-1
    # manifest supplies the literal registry key, which is not alias-normalized.
    if not series_root:
        raise ValueError("An explicit selected series root is required")
    with (Path(series_root) / "series.toml").open("rb") as stream:
        raw = stream.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("Series manifest is too large for Jobs UI metadata")
    document = tomllib.loads(raw.decode("utf-8-sig"))
    actual = document.get("series_id")
    if (type(document.get("schema_version")) is not int
            or document["schema_version"] != 1 or not isinstance(actual, str) or not actual):
        raise ValueError("A registered schema-1 series with its own series_id is required")
    if _series_key(actual) != _series_key(selected_series_id):
        raise ValueError("Selected series manifest differs from the selected Studio series")
    return actual


def _label(registration, completion):
    freeze = completion.get("calibration") or {}
    recorded = next((value.strip() for value in (freeze.get("display_name"), freeze.get("name"),
                     completion.get("calibration_name")) if isinstance(value, str) and value.strip()), None)
    path = freeze.get("path")
    label = recorded or ("Saved calibration: " + PureWindowsPath(path).name
                         if isinstance(path, str) and path else "Saved calibration")
    context = next((value.strip() for value in (registration.get("label"), registration.get("name"))
                    if isinstance(value, str) and value.strip()), None)
    return label + " · Execution: " + context if context else label


def discover_native_reviews(registry, series_root, series_id, *, current_machine_id,
                            cancelled=lambda: False, extra_collections=None):
    """Return (attachments, errors) from explicit immutable job registrations.

    ``current_machine_id`` is a required caller-owned physical-store identity.
    The local registry's series association is only a UI label; the immutable
    job must name the selected series itself. Registry lookup uses the literal
    ID in the selected folder's series.toml; attachments retain the UI alias.
    An existing payload publication is resolved through its original done
    binding. Photo geometry remains per-frame original storage evidence.
    Unsupported and unfinished protocols are skipped. Broken eligible metadata
    is reported, while other explicitly registered jobs can still be reviewed.
    """
    attachments, errors = [], []
    if cancelled():
        return attachments, errors
    try:
        name(current_machine_id)  # Validate, but retain the caller's exact ID.
        if not _series_key(series_id):
            raise ValueError("An explicit selected series ID is required")
        registered_series_id = _registered_series_id(series_root, series_id)
        if cancelled():
            return attachments, errors
        registrations = registry.jobs(series_root, registered_series_id)
    except Exception as error:
        return attachments, ["Registered native reviews unavailable: " + str(error)]
    reader = JobSnapshotReader()
    for registration in registrations:
        if cancelled():
            break
        job_label = "Registered job"
        try:
            if not isinstance(registration, Mapping):
                raise ValueError("Registered job must be an explicit metadata record")
            job_label = str(registration.get("name") or registration.get("root") or job_label)
            root = Path(registration["root"])
            expected_job = registration.get("job")
            if not isinstance(expected_job, dict):
                raise ValueError("Registered job lacks its immutable metadata descriptor")
            job = reader.read(root / "job.json", expected_job)
            if job.get("format") != "darsia-alias-job-v1":
                continue
            if "pipeline" in job and job["pipeline"] not in _NATIVE_PIPELINES:
                continue
            if _series_key(job.get("series_id")) != _series_key(registered_series_id):
                raise ValueError("Immutable job series_id differs from the selected series; UI association is insufficient")
            if cancelled():
                break
            snapshot = reader(registration)
            if (snapshot.get("job_facts") != job
                    or reader.read(root / "job.json", expected_job) != job):
                raise ValueError("Job snapshot differs from its registered immutable job")
            rows = snapshot.get("runs")
            if (not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows)
                    or [row.get("run") for row in rows] != job.get("expected_runs")):
                raise ValueError("Job snapshot does not retain the exact expected run order")
            errors.extend(job_label + ": " + str(error) for error in snapshot.get("errors", []))
        except Exception as error:
            errors.append(job_label + ": " + str(error))
            continue
        for row in rows:
            if cancelled():
                return attachments, errors
            if row.get("state") != "Complete":
                continue
            run = row["run"]
            try:
                safe_run = name(run)
                task = reader.read(root / "results" / safe_run / "task.json")
                task_pipeline = task.get("pipeline", "n1")
                if task_pipeline not in _NATIVE_PIPELINES:
                    continue
                # Complete comes from the original task/terminal validator.
                # The valid absence of job.pipeline supplies no protocol;
                # an explicitly recorded value must still agree with the task.
                if "pipeline" in job and job["pipeline"] != task_pipeline:
                    raise ValueError("Native task protocol differs from its authenticated job")
                expected_freeze = job["calibration_sha256"]
                if task.get("run") != run or task.get("calibration_sha256") != expected_freeze:
                    raise ValueError("Completed native task differs from the selected job/run/calibration")
                done = reader.read(root / "done" / (safe_run + ".json"))
                if (done.get("format") != "darsia-alias-done-v1" or done.get("run") != run
                        or done.get("completed", {}).get("relative_path")
                        != "results/" + safe_run + "/completed.json"):
                    raise ValueError("Original done record differs from its canonical completed run")
                index = _bound_owned(root, done["products_index"])
                completion_bound = _bound_owned(root, done["completed"])
                completion = reader.read(completion_bound["path"], completion_bound)
                _, index_document, products = publication._read_index(index)
                if (index_document["run"] != run
                        or index_document["attempt_id"] != done.get("attempt_id")
                        or _bytes_identity(index_document["job"]) != _bytes_identity(expected_job)
                        or _bytes_identity(products["completed.json"]) != _bytes_identity(completion_bound)
                        or completion.get("complete") is not True or completion.get("run") != run
                        or completion.get("calibration", {}).get("sha256") != expected_freeze):
                    raise ValueError("Done publication differs from its original registered job/run/freeze")
                if completion_pipeline(completion) != task_pipeline:
                    raise ValueError("Native completion protocol differs from its authenticated job/task")
                if cancelled():
                    return attachments, errors
                from .calibration_review_native_bindings import collect_bindings
                collections = collect_bindings(index, root, done,
                    current_machine_id=current_machine_id,
                    collections=(extra_collections or {}).get(index["sha256"], ()))
                native = NativeSavedReview(index, current_machine_id=current_machine_id,
                    expected_run=run, expected_calibration_sha256=expected_freeze,
                    collections=collections, expected_pipeline=task_pipeline)
                attachments.append({"reader": native, "dataset": series_id,
                                    "calibration_name": _label(registration, completion),
                                    "registration": deepcopy(registration),
                                    "source_registrations": deepcopy([item for item in registrations
                                        if isinstance(item, dict)])})
            except Exception as error:
                errors.append(job_label + " · " + str(run) + ": " + str(error))
    return attachments, errors
