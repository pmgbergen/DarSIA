"""Typed queue identities for the unchanged original Comparison calculation.

One queue operation contains one requested time and every original selected
run/ROI. The first selected run remains the reference for every pair. Working
precision is deliberately ``original``: this route adds no array conversion.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import re

PIPELINE = "preset-wasserstein"
INPUTS_FORMAT = "darsia-preset-wasserstein-inputs-v1"
OPERATION_FORMAT = "darsia-preset-wasserstein-operation-v1"
RESULT_FORMAT = "darsia-preset-wasserstein-result-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def absolute_path(value):
    if (not isinstance(value, str) or not value or "\x00" in value
            or not (PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute())):
        raise ValueError("An explicit original absolute source location is required")
    return value


def descriptor(value):
    if (not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}
            or re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))) is None
            or type(value.get("size_bytes")) is not int or value["size_bytes"] < 0):
        raise ValueError("An exact original source descriptor is required")
    absolute_path(value["path"])
    return value


def operation_id(order):
    if type(order) is not int or order < 0:
        raise ValueError("A requested-time order must be a nonnegative integer")
    return f"comparison-time-{order:08d}"


def validate_time(value):
    if (not isinstance(value, dict) or set(value) != {"order", "time_hours", "tolerance_hours"}
            or type(value["order"]) is not int or value["order"] < 0
            or any(type(value[key]) not in (int, float) or not math.isfinite(value[key])
                   for key in ("time_hours", "tolerance_hours"))
            or value["tolerance_hours"] < 0):
        raise ValueError("The exact original requested time and tolerance are required")
    return value


def _inventory_name(value, *, leaf_only):
    if not isinstance(value, str) or not value or "\x00" in value:
        return False
    windows, posix = PureWindowsPath(value), PurePosixPath(value)
    if windows.anchor or posix.anchor or ".." in windows.parts or ".." in posix.parts:
        return False
    return not leaf_only or windows.name == value and posix.name == value


def validate_inputs_contract(value):
    from .store_map import ROLES
    fields = {"format", "source_configuration", "working_directory", "config_paths",
              "runs", "rois", "requested_times", "reference_run", "files",
              "inventories", "existence", "config_roots", "rig_folder",
              "selected_mass", "series_manifest", "path_aliases"}
    if not isinstance(value, dict) or set(value) != fields or value.get("format") != INPUTS_FORMAT:
        raise ValueError("An exact original preset input closure is required")
    descriptor(value["source_configuration"])
    absolute_path(value["working_directory"])
    absolute_path(value["rig_folder"])
    if value["series_manifest"] is not None:
        descriptor(value["series_manifest"])
    for key, minimum in (("runs", 2), ("rois", 1)):
        items = value[key]
        if (not isinstance(items, list) or len(items) < minimum
                or any(type(item) not in (str, int) or isinstance(item, str)
                       and (not item or "\x00" in item) for item in items)
                or len(set(items)) != len(items)):
            raise ValueError("Original distinct ordered " + key + " are required")
    if len({type(run) for run in value["runs"]}) != 1:
        raise ValueError("Original run IDs must support the original ordered comparison")
    if value["reference_run"] != value["runs"][0]:
        raise ValueError("The original first selected reference run must be retained")
    times = value["requested_times"]
    if not isinstance(times, list) or not times:
        raise ValueError("Choose explicit original comparison times")
    for index, item in enumerate(times):
        validate_time(item)
        if item["order"] != index:
            raise ValueError("Original requested-time ordering differs")
    paths, uses = set(), set()
    if not isinstance(value["files"], list) or not value["files"]:
        raise ValueError("Original consumed source files are required")
    for row in value["files"]:
        if not isinstance(row, dict) or set(row) != {"descriptor", "uses", "role"}:
            raise ValueError("Each source needs an exact descriptor, role and consumers")
        bound = descriptor(row["descriptor"])
        key = str(PureWindowsPath(bound["path"])).casefold()
        if key in paths or row["role"] not in ROLES:
            raise ValueError("Duplicate source or unsupported input role")
        paths.add(key)
        if (not isinstance(row["uses"], list) or not row["uses"]
                or any(not isinstance(use, str) or not use or use in uses for use in row["uses"])
                or len(set(row["uses"])) != len(row["uses"])):
            raise ValueError("Source consumers must be explicit and unambiguous")
        uses.update(row["uses"])
    if (not isinstance(value["config_paths"], list) or not value["config_paths"]
            or value["source_configuration"] not in value["config_paths"]):
        raise ValueError("The complete ordered configuration source closure is required")
    for bound in value["config_paths"]:
        descriptor(bound)
        if bound not in [row["descriptor"] for row in value["files"]]:
            raise ValueError("A configuration is outside the admitted source closure")
    selected = value["selected_mass"]
    if (not isinstance(selected, list)
            or len(selected) != len(times) * len(value["runs"])):
        raise ValueError("Every original requested-time/run mass selection must be explicit")
    expected_selection = [(time["order"], run) for time in times for run in value["runs"]]
    if [(row.get("time_order"), row.get("run")) for row in selected] != expected_selection:
        raise ValueError("Original mass selection coverage/order differs")
    for row in selected:
        if set(row) != {"time_order", "run", "descriptor"}:
            raise ValueError("Mass selection must bind its exact payload or explicit no-match")
        if row["descriptor"] is not None:
            descriptor(row["descriptor"])
            if not any(source["descriptor"] == row["descriptor"] and source["role"] == "image"
                       for source in value["files"]):
                raise ValueError("The original selected mass payload is not admitted")
    for key in ("inventories", "existence", "config_roots"):
        if not isinstance(value[key], list):
            raise ValueError("Frozen original path decisions must be explicit lists")
    for row in value["inventories"]:
        if (not isinstance(row, dict) or set(row) != {"path", "pattern", "names"}
                or row["pattern"] is not None and
                   (not isinstance(row["pattern"], str) or not row["pattern"])
                or not isinstance(row["names"], list)
                or len(set(row["names"])) != len(row["names"])
                or any(not _inventory_name(name, leaf_only=row["pattern"] is None)
                       for name in row["names"])):
            raise ValueError("A frozen directory inventory must contain safe original relative names")
        absolute_path(row["path"])
    for row in value["existence"]:
        if (not isinstance(row, dict) or set(row) != {"path", "kind", "exists"}
                or row["kind"] not in {"is_file", "is_dir", "exists"}
                or type(row["exists"]) is not bool):
            raise ValueError("An original optional-path decision is required")
        absolute_path(row["path"])
    for row in value["config_roots"]:
        if (not isinstance(row, dict) or set(row) != {"paths", "root"}
                or not isinstance(row["paths"], list) or not row["paths"]):
            raise ValueError("Original configuration-relative path roots are required")
        for path in row["paths"]:
            absolute_path(path)
        if row["root"] is not None:
            absolute_path(row["root"])
    aliases = value["path_aliases"]
    if not isinstance(aliases, list):
        raise ValueError("Frozen original storage aliases must be explicit")
    sources = []
    for row in aliases:
        if not isinstance(row, dict) or set(row) != {"source", "destination"}:
            raise ValueError("An exact original storage prefix mapping is required")
        for key in ("source", "destination"):
            prefix = row[key]
            # The original registry strips the slash even for a drive root.
            absolute_path(prefix + "/" if isinstance(prefix, str)
                          and re.fullmatch(r"[A-Za-z]:", prefix) else prefix)
        sources.append(row["source"])
    if len(set(sources)) != len(sources) or list(map(len, sources)) != sorted(map(len, sources), reverse=True):
        raise ValueError("Keep the distinct original longest-prefix alias order")
    # Use the original filename function, not a parallel naming convention.
    # Check every time up front so two jobs cannot silently overwrite one name.
    from darsia.presets.workflows.comparison.comparison_wasserstein import WassersteinDistanceResult
    filenames = set()
    for time in times:
        for row in expected_comparisons(value, time):
            path = WassersteinDistanceResult.get_filename(row["run_1"], row["run_2"],
                row["time_hours"], row["roi_name"])
            if (path.name != str(path) or PureWindowsPath(str(path)).name != str(path)
                    or str(path).casefold() in filenames):
                raise ValueError("Original comparison names collide or escape the output folder")
            filenames.add(str(path).casefold())
    return value


def expected_comparisons(inputs, time):
    """Preserve the original nested run traversal and lexical upper triangle."""
    return [dict(run_1=left, run_2=right, time_hours=time["time_hours"], roi_name=roi)
            for left in inputs["runs"] for right in inputs["runs"] if left < right
            for roi in inputs["rois"]]


def validate_operation(value):
    if not isinstance(value, dict) or set(value) != {"format", "run", "preset_inputs", "comparison_time"}:
        raise ValueError("One exact original preset requested-time operation is required")
    if value["format"] != OPERATION_FORMAT:
        raise ValueError("Unsupported preset comparison operation")
    inputs, time = validate_inputs_contract(value["preset_inputs"]), validate_time(value["comparison_time"])
    if (time["order"] >= len(inputs["requested_times"])
            or time != inputs["requested_times"][time["order"]]
            or value["run"] != operation_id(time["order"])):
        raise ValueError("Operation changed the frozen original requested time")
    return value


def _identity(value, kind):
    from .pipeline.job_queue import name
    if (not isinstance(value, dict) or value.get("format") != f"darsia-alias-{kind}-v1"
            or value.get("pipeline") != PIPELINE or value.get("backend") != "cpu"
            or value.get("precision") != "original" or value.get("format_version") != 1
            or value.get("storage_profile") != "full" or value.get("legacy_paths") is not False
            or value.get("execution_ready") is not True
            or value.get("runtime_support") != "darsia-role-io-runtime-v1"
            or re.fullmatch(r"[0-9a-f]{40}", str(value.get("commit", ""))) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(value.get("preset_inputs_sha256", ""))) is None
            or "calibration_sha256" in value):
        raise ValueError("A typed CPU original-precision preset comparison identity is required")
    if name(value.get("job_id")) != value["job_id"]:
        raise ValueError("Explicit canonical comparison job ID required")


def validate_preset_wasserstein_task(task):
    from .pipeline.job_inputs import validate_catalog
    from .store_map import validate_input_reference
    _identity(task, "task")
    time = validate_time(task.get("comparison_time"))
    if (task.get("run") != operation_id(time["order"])
            or task.get("ordered_case_ids") != [task["run"]] or task.get("threads") != 1
            or type(task.get("expected_comparisons")) is not int or task["expected_comparisons"] < 1):
        raise ValueError("One original requested-time operation and CPU thread required")
    for key in ("configuration", "series", "series_manifest"):
        validate_input_reference(task.get(key))
    catalog = validate_catalog(task["input_catalog"])
    if catalog["run"] != task["run"] or catalog["configuration"] != task["configuration"]:
        raise ValueError("Comparison task/catalogue identity differs")
    return task


def validate_preset_wasserstein_job(job):
    from .store_map import validate_input_reference
    _identity(job, "job")
    for key in ("series", "series_manifest"):
        validate_input_reference(job.get(key))
    times = job.get("preset_comparison_times")
    if (not isinstance(times, list) or not times
            or job.get("expected_runs") != [operation_id(index) for index in range(len(times))]):
        raise ValueError("Exact original requested-time job coverage required")
    for index, time in enumerate(times):
        validate_time(time)
        if time["order"] != index:
            raise ValueError("Comparison job changed the requested-time order")
    if type(job.get("comparisons_per_time")) is not int or job["comparisons_per_time"] < 1:
        raise ValueError("Actual original pair/ROI result count required")
    return job


def validate_preset_wasserstein_claim(task, job, *, config=None):
    validate_preset_wasserstein_task(task)
    validate_preset_wasserstein_job(job)
    for key in ("commit", "backend", "precision", "job_id", "preset_inputs_sha256", "series", "series_manifest"):
        if task[key] != job[key]:
            raise ValueError("Preset comparison task/job differs: " + key)
    time = task["comparison_time"]
    if (time["order"] >= len(job["preset_comparison_times"])
            or job["preset_comparison_times"][time["order"]] != time
            or task["expected_comparisons"] != job["comparisons_per_time"]):
        raise ValueError("Preset task is outside the exact job coverage")
    if config is not None:
        validate_operation(config)
        if (config["run"] != task["run"] or config["comparison_time"] != time
                or digest(config["preset_inputs"]) != task["preset_inputs_sha256"]
                or len(expected_comparisons(config["preset_inputs"], time)) != task["expected_comparisons"]):
            raise ValueError("Consumed comparison configuration differs from the owned task")
    return task
