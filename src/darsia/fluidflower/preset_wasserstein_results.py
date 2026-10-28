"""Original comparison products plus strict operational publication evidence.

Scientific JSON keeps its original schema and NaN/Infinity spelling. Only the
separate receipt and execution envelope are strict finite shared metadata.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

from .preset_wasserstein_protocol import (
    PIPELINE, RESULT_FORMAT, descriptor, validate_time,
    validate_preset_wasserstein_claim, validate_preset_wasserstein_job,
    validate_preset_wasserstein_task,
)

RECEIPT = "comparison_execution.json"
RECEIPT_FORMAT = "darsia-preset-wasserstein-execution-receipt-v1"
ROW_FIELDS = {"run_1", "run_2", "time", "time_1", "time_2", "roi_name",
              "roi_exact_mass", "roi_detected_mass_1", "roi_detected_mass_2",
              "distance", "normalized_distance", "computation_time", "timestamp", "status"}
_RECEIPT_FIELDS = ("pipeline", "complete", "run", "job_id", "backend", "precision",
    "comparison_time", "preset_inputs_sha256", "expected_count", "comparison_count",
    "comparisons", "files", "output_count", "vtk", "worker_references_closed",
    "nonfinite_distances", "nonfinite_normalized_distances", "result_statuses", "validation")


def _filename(row):
    from darsia.presets.workflows.comparison.comparison_wasserstein import WassersteinDistanceResult
    return str(WassersteinDistanceResult.get_filename(
        row["run_1"], row["run_2"], row["time_hours"], row["roi_name"]))


def _inventory(result):
    """Validate the explicit original inventory without visiting its locations."""
    from .pipeline.products_publication import _relative
    files, rows = result.get("files"), result.get("comparisons")
    if (not isinstance(files, dict) or not files or not isinstance(rows, list) or not rows
            or type(result.get("expected_count")) is not int or result["expected_count"] < 1
            or type(result.get("comparison_count")) is not int
            or result["comparison_count"] != result["expected_count"]
            or len(rows) != result["comparison_count"]
            or type(result.get("output_count")) is not int or result["output_count"] != len(files)):
        raise ValueError("Complete original comparison inventory/counts are required")
    paths = set()
    for key, bound in files.items():
        if _relative(key) != key or key.casefold() in paths:
            raise ValueError("Comparison products need distinct canonical relative names")
        paths.add(key.casefold())
        descriptor(bound)
    identities, json_files = set(), []
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"run_1", "run_2", "time_hours", "roi_name", "status", "file"}
                or any(type(row[key]) not in (str, int) or isinstance(row[key], str) and not row[key]
                       for key in ("run_1", "run_2", "roi_name"))
                or type(row["run_1"]) is not type(row["run_2"])
                or row["run_1"] >= row["run_2"]
                or type(row["time_hours"]) not in (int, float) or not math.isfinite(row["time_hours"])
                or row["time_hours"] != result["comparison_time"]["time_hours"]
                or row["status"] not in ("success", "missing", None)):
            raise ValueError("Original comparison row identity/status differs")
        identity = tuple(row[key] for key in ("run_1", "run_2", "time_hours", "roi_name"))
        filename = _filename(row)
        if (identity in identities or PurePosixPath(filename).name != filename
                or PureWindowsPath(filename).name != filename or filename in json_files
                or files.get(filename) != descriptor(row["file"])):
            raise ValueError("Original comparison filenames or byte identities differ")
        identities.add(identity)
        json_files.append(filename)
    vtk = result.get("vtk")
    vtk_files = sorted(key for key in files if key.startswith("vtk/"))
    if (not isinstance(vtk, dict) or set(vtk) != {"files", "actual_count", "state"}
            or vtk["files"] != vtk_files or type(vtk["actual_count"]) is not int
            or vtk["actual_count"] != len(vtk_files)
            or vtk["state"] != ("files_written" if vtk_files else "no_files_written")
            or set(files) != set(json_files) | set(vtk_files)):
        raise ValueError("Original JSON/VTK inventory differs from its declared products")
    return json_files


def validate_preset_wasserstein_result(result, task=None, job=None):
    """Validate finite execution metadata; do not interpret completion as science."""
    from .pipeline.job_queue import name
    if (not isinstance(result, dict) or result.get("format") != RESULT_FORMAT
            or result.get("pipeline") != PIPELINE or result.get("complete") is not True
            or result.get("worker_references_closed") is not True
            or result.get("backend") != "cpu" or result.get("precision") != "original"
            or type(result.get("frames")) is not int or result["frames"] != 0
            or result.get("validation", {}).get("state") != "not_scored"
            or re.fullmatch(r"[0-9a-f]{64}", str(result.get("preset_inputs_sha256", ""))) is None):
        raise ValueError("A closed original Comparison execution envelope is required")
    name(result.get("run"))
    name(result.get("job_id"))
    validate_time(result.get("comparison_time"))
    _inventory(result)
    descriptor(result.get("execution_receipt"))
    statuses = dict(Counter(row["status"] if row["status"] is not None else "not_reported"
                            for row in result["comparisons"]))
    if (result.get("result_statuses") != statuses
            or any(type(result.get(key)) is not int or result[key] < 0
                   for key in ("nonfinite_distances", "nonfinite_normalized_distances"))
            or result["nonfinite_distances"] > result["comparison_count"]):
        raise ValueError("Comparison execution receipt has invalid result summaries")
    # The worker's candidate is parsed by the existing strict metadata reader.
    json.dumps(result, allow_nan=False)
    if task is not None:
        validate_preset_wasserstein_task(task)
        if (any(result[key] != task[key] for key in
                ("run", "job_id", "backend", "precision", "comparison_time", "preset_inputs_sha256"))
                or result["expected_count"] != task["expected_comparisons"]
                or ("commit" in result and result["commit"] != task["commit"])):
            raise ValueError("Original comparison result differs from its owned task")
    if job is not None:
        validate_preset_wasserstein_job(job)
        time = result["comparison_time"]
        if (result["run"] not in job["expected_runs"]
                or time["order"] >= len(job["preset_comparison_times"])
                or job["preset_comparison_times"][time["order"]] != time
                or result["expected_count"] != job["comparisons_per_time"]
                or any(result[key] != job[key] for key in ("job_id", "backend", "precision", "preset_inputs_sha256"))
                or ("commit" in result and result["commit"] != job["commit"])):
            raise ValueError("Original comparison result differs from its immutable job")
        if task is not None:
            validate_preset_wasserstein_claim(task, job)
    return result


def _scientific_row(raw, declaration):
    """Decode original scientific scalars, including original NaNs, without casts."""
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate original comparison JSON key")
            value[key] = item
        return value
    row = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs)
    if (not isinstance(row, dict) or set(row) != ROW_FIELDS
            or any(row[key] != declaration[key] for key in ("run_1", "run_2", "roi_name", "status"))
            or type(row["time"]) not in (int, float) or row["time"] != declaration["time_hours"]
            or not isinstance(row["timestamp"], str) or not row["timestamp"]):
        raise ValueError("Original comparison JSON differs from its complete declared row")
    for key in ("time_1", "time_2", "roi_exact_mass", "roi_detected_mass_1",
                "roi_detected_mass_2", "distance", "computation_time"):
        if row[key] is None and key in {"time_1", "time_2"}:
            continue
        if type(row[key]) not in (int, float):
            raise ValueError("Original comparison has a nonnumeric scalar: " + key)
    normalized = row["normalized_distance"]
    if isinstance(normalized, dict):
        if set(normalized) != {"plume_based", "rig_based", "mass_based"}:
            raise ValueError("Original normalized distance keys differ")
        values = list(normalized.values())
    else:
        values = [normalized]
    if any(type(value) not in (int, float) for value in values):
        raise ValueError("Original normalized distances are not numeric")
    return int(not math.isfinite(row["distance"])), sum(not math.isfinite(value) for value in values)


def _summaries(result, read_scientific):
    counts = [_scientific_row(read_scientific(row["file"]), row) for row in result["comparisons"]]
    return dict(nonfinite_distances=sum(row[0] for row in counts),
        nonfinite_normalized_distances=sum(row[1] for row in counts),
        result_statuses=dict(Counter(row["status"] if row["status"] is not None else "not_reported"
                                     for row in result["comparisons"])))


def _receipt_document(result):
    return dict(format=RECEIPT_FORMAT, **{key: deepcopy(result[key]) for key in _RECEIPT_FIELDS})


def _check_receipt(result, document):
    if document != _receipt_document(result):
        raise ValueError("Strict comparison execution receipt differs from its completion")


def _owned_files(result, output_root):
    """Read/hash only explicit products in the actual local owned attempt."""
    from .pipeline.products_publication import _bound, _verify_file, _verified_bytes, MAX_METADATA_BYTES
    from .pipeline.publication_runtime import _relative
    from .pipeline.job_queue import descriptor as file_descriptor
    root = Path(output_root).resolve()
    paths = {}
    for key, source in result["files"].items():
        bound = _bound(source)
        if _relative(bound["path"], root) != key:
            raise ValueError("Comparison product is outside its owned attempt")
        path = root.joinpath(*key.split("/"))
        if path.resolve() != path:
            raise ValueError("Comparison product traverses a link")
        _verify_file(path, bound, limit=bound["bytes"])
        paths[bound["path"]] = path
    def read(source):
        return _verified_bytes(paths[source["path"]], _bound(source), limit=MAX_METADATA_BYTES)
    def finish():
        for source in result["files"].values():
            if file_descriptor(paths[source["path"]]) != source:
                raise ValueError("Original comparison product changed during validation")
    return read, finish


def write_preset_wasserstein_receipt(result, *, task, output_root):
    """Bind actual closed JSON/VTK bytes without changing any scientific product."""
    from .mass.n1_pipeline import _write_json
    value = deepcopy(result)
    value["preset_inputs_sha256"] = task["preset_inputs_sha256"]
    _inventory(value)
    read, finish = _owned_files(value, output_root)
    value.update(_summaries(value, read))
    # Validate the complete strict envelope before writing the one new receipt.
    path = Path(output_root).resolve() / RECEIPT
    value["execution_receipt"] = dict(path=str(path), sha256="0" * 64, size_bytes=0)
    validate_preset_wasserstein_result(value, task=task)
    finish()
    value["execution_receipt"] = _write_json(path, _receipt_document(value))
    return value


def _envelope(result, task, token, machine):
    from .pipeline.job_queue import name
    validate_preset_wasserstein_result(result, task=task)
    actual = result.get("actual_worker")
    if (result.get("task") != task or result.get("attempt_id") != token
            or task.get("attempt_id") != token or result.get("commit") != task["commit"]
            or not isinstance(actual, dict)
            or name(actual.get("machine")) != name(machine)):
        raise ValueError("Comparison completion differs from its actual runtime envelope")


def extract_preset_wasserstein_products(result, claim, machine_id):
    """Validate the real owned bytes before original immutable publication."""
    from .pipeline.job_queue import name
    from .pipeline.products_publication import product, _bound, _verified_bytes, _json, MAX_METADATA_BYTES
    name(machine_id)
    if name(claim.machine) != name(machine_id + "-" + claim.task["backend"]):
        raise ValueError("Comparison publication target differs from the actual claim service")
    _envelope(result, claim.task, claim.token, claim.machine)
    json_files = set(_inventory(result))
    read, finish = _owned_files(result, claim.output)
    if any(result[key] != value for key, value in _summaries(result, read).items()):
        raise ValueError("Comparison summaries differ from the actual original JSON")
    receipt = _bound(result["execution_receipt"])
    root = Path(claim.output).resolve()
    path = root / RECEIPT
    if Path(receipt["path"]).resolve() != path or path.is_symlink():
        raise ValueError("Comparison execution receipt is outside its owned attempt")
    _check_receipt(result, _json(_verified_bytes(path, receipt, limit=MAX_METADATA_BYTES)))
    rows, metadata, accessible = [], [], {}
    for key, source in {**result["files"], RECEIPT: result["execution_receipt"]}.items():
        bound = _bound(source)
        kind = "operational_receipt" if key == RECEIPT else "scalar_table" if key in json_files else "intermediate"
        rows.append(product(key, machine_id=machine_id, local_path=bound["path"],
                    sha256=bound["sha256"], bytes=bound["bytes"], kind=kind))
        accessible[key] = root.joinpath(*key.split("/"))
        if key == RECEIPT or key in json_files:
            metadata.append(key)
    finish()
    return rows, metadata, accessible


def preset_wasserstein_terminal(root, base_task, job, done, reader):
    """Verify owner/index/receipt lineage using only immutable shared copies."""
    from .published_jobs_snapshot import (
        _read_reference, _same_bytes as same, _task_identity, _lease_chain,
        _owner_machine, _original_metadata, _pure, _SnapshotMetadata,
    )
    from .pipeline.products_publication import _read_index, published_metadata_descriptor
    from .pipeline.job_queue import descriptor as file_descriptor, name, _own_asset
    root = Path(root)
    validate_preset_wasserstein_claim(base_task, job)
    if (done.get("format") != "darsia-alias-done-v1" or done.get("run") != base_task["run"]
            or done["completed"]["relative_path"] != "results/" + base_task["run"] + "/completed.json"):
        raise ValueError("Actual alias comparison done identity required")
    completed = _read_reference(root, done["completed"], reader)
    index_path = _own_asset(root, done["products_index"])
    index = file_descriptor(index_path)
    folder, document, products = _read_index(index)
    metadata = _SnapshotMetadata(index, folder, products)
    task = completed.get("task") or {}
    if (document["run"] != base_task["run"] or document["attempt_id"] != done["attempt_id"]
            or not same(document["job"], file_descriptor(root / "job.json"))
            or not same(published_metadata_descriptor(index), done["completed"])
            or not _task_identity(base_task, task)):
        raise ValueError("Original comparison index/task/source identity differs")
    validate_preset_wasserstein_result(completed, task=task, job=job)
    owner = reader(index_path.parent.parent / "owner.json")
    if (owner.get("task") != task or owner.get("token") != done["attempt_id"]
            or owner.get("actual_owner") != completed.get("actual_worker")):
        raise ValueError("Comparison publication owner differs from the actual claim")
    machine = _owner_machine(owner, base_task, done["attempt_id"])
    _envelope(completed, task, done["attempt_id"], machine + "-" + task["backend"])
    original = products["completed.json"]
    attempt = _pure(original["local_path"]).parent
    if (not same(original, done["completed"]) or name(original["machine_id"]) != name(machine)
            or _pure(original["local_path"]).name != "candidate_completed.json"
            or attempt.name != done["attempt_id"] or attempt.parent.name != ".attempts"
            or name(attempt.parent.parent.name) != name(task["run"])
            or name(attempt.parent.parent.parent.name) != name(task["job_id"])):
        raise ValueError("Comparison producer is outside its actual owned job/run/attempt")
    expected = {**completed["files"], RECEIPT: completed["execution_receipt"]}
    if set(products) != {"completed.json", *expected}:
        raise ValueError("Comparison publication omits or adds an original output")
    json_files = set(_inventory(completed))
    for key, bound in expected.items():
        product = products[key]
        kind = "operational_receipt" if key == RECEIPT else "scalar_table" if key in json_files else "intermediate"
        if (not same(bound, product) or _pure(product["local_path"]) != attempt / key
                or _pure(bound["path"]) != attempt / key or product["kind"] != kind
                or name(product["machine_id"]) != name(machine)):
            raise ValueError("Indexed comparison product differs from its original descriptor")
    receipt = _original_metadata(index, products, completed["execution_receipt"], attempt,
                                 machine, metadata=metadata)
    _check_receipt(completed, receipt)
    def scientific(source):
        return _original_metadata(index, products, source, attempt, machine,
                                  metadata=metadata, decode_json=False)
    if any(completed[key] != value for key, value in _summaries(completed, scientific).items()):
        raise ValueError("Published original comparison scalars differ from their strict receipt")
    lineage = _lease_chain(root, base_task["run"], done["attempt_id"], base_task, reader)
    metadata.finish()
    return completed, {}, lineage
