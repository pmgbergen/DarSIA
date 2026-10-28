"""Metadata contracts for saved Wasserstein comparisons on the whole-run queue.

This is a separate CPU-float64 analysis contract. It does not change production
precision, image selection, calibration or scientific acceptance. A queue task
is one comparison, not a production image or an experimental run.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import re

PIPELINE = "wasserstein-comparison"
OPERATION_FORMAT = "darsia-wasserstein-job-operation-v1"
RESULT_FORMAT = "darsia-wasserstein-job-result-v1"
REVISION_FORMAT = "darsia-fluidflower-wasserstein-revision-v1"
CONTRACT_FORMAT = "darsia-fluidflower-wasserstein-task-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def descriptor(value):
    if (not isinstance(value, dict) or not isinstance(value.get("path"), str)
            or not (PurePosixPath(value["path"]).is_absolute() or PureWindowsPath(value["path"]).is_absolute())
            or re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))) is None
            or type(value.get("size_bytes")) is not int or value["size_bytes"] < 0):
        raise ValueError("A consumed Wasserstein input needs its absolute path, SHA-256 and size")
    return {key: value[key] for key in ("path", "sha256", "size_bytes")}


def validate_revision(plan):
    """Check the original frozen revision and its original validation contract."""
    if not isinstance(plan, dict) or plan.get("format") != REVISION_FORMAT:
        raise ValueError("A saved Wasserstein revision is required")
    for value in (plan, plan.get("task_contract", {})):
        if not isinstance(value, dict) or value.get("content_sha256") != digest(
                {key: item for key, item in value.items() if key != "content_sha256"}):
            raise ValueError("Saved Wasserstein content identity differs")
    config, contract, report = plan.get("configuration", {}), plan["task_contract"], plan.get("validation", {})
    if (contract.get("format") != CONTRACT_FORMAT or not plan.get("plan_id")
            or plan.get("execution", {}).get("ready") is not True
            or report.get("valid") is not True or report.get("execution_ready") is not True
            or report.get("summary", {}).get("error_count") != 0
            or report.get("config_sha256") != digest(config)
            or report.get("task_contract") != contract):
        raise ValueError("The saved Wasserstein plan is not execution-ready")
    # Rebuild with the original contract builder; never replace transport or
    # normalization semantics with settings supplied by a launch dialog.
    from .wasserstein import wasserstein_task_contract
    rebuilt = wasserstein_task_contract(config, contract.get("mass_result"),
                                       contract.get("analysis_domains"), contract.get("resolved_time_selection"))
    if rebuilt != contract or contract.get("execution_profile", {}).get("backend") != "cpu-float64":
        raise ValueError("The original Wasserstein numerical contract differs")
    if contract.get("outputs", {}).get("store_transport_diagnostics") is not False:
        raise ValueError("This queue route does not export transport diagnostics")
    descriptor(contract.get("mass_result"))
    return contract


def comparisons(plan):
    """Enumerate the frozen order without selecting any new images or times."""
    contract = validate_revision(plan)
    selection, compare = contract["resolved_time_selection"], contract["comparison"]
    indices, times = selection.get("selected_time_indices"), selection.get("selected_times_hours")
    if (not isinstance(indices, list) or not indices or any(type(i) is not int or i < 0 for i in indices)
            or indices != sorted(set(indices)) or not isinstance(times, list) or len(times) != len(indices)
            or any(type(t) not in (int, float) or not math.isfinite(t) for t in times)):
        raise ValueError("The frozen Wasserstein time selection is invalid")
    runs = contract.get("run_ids", [])
    pairs, domains, components = (compare.get(key) for key in ("pairs", "domains", "mass_components"))
    if (not isinstance(runs, list) or len(runs) < 2 or len(set(runs)) != len(runs)
            or any(not isinstance(values, list) or not values for values in (pairs, domains, components))):
        raise ValueError("Explicit Wasserstein runs, pairs, domains and components are required")
    if (len(set(domains)) != len(domains) or len(set(components)) != len(components)
            or set(components) - {"mass_total", "mass_aqueous", "mass_free_gas"}):
        raise ValueError("Wasserstein domains/components must be distinct and supported")
    seen = set()
    for pair in pairs:
        if (not isinstance(pair, dict) or set(pair) != {"run_1", "run_2"}
                or pair["run_1"] not in runs or pair["run_2"] not in runs or pair["run_1"] == pair["run_2"]):
            raise ValueError("A Wasserstein pair must name two distinct selected runs")
        key = tuple(sorted(pair.values()))
        if key in seen:
            raise ValueError("Duplicate unordered Wasserstein pair")
        seen.add(key)
    rows = []
    for index, time in zip(indices, times):
        for pair in pairs:
            for domain in domains:
                for component in components:
                    rows.append(dict(order=len(rows), time_index=index, time_hours=time,
                                     **pair, domain_id=domain, mass_component=component))
    if len(rows) != plan["validation"]["summary"].get("task_count"):
        raise ValueError("The saved validation count differs from its actual comparisons")
    return rows


def operation_id(order):
    if type(order) is not int or order < 0:
        raise ValueError("A comparison order must be a nonnegative integer")
    return f"w1-{order:08d}"


def validate_operation(config, plan):
    if (not isinstance(config, dict) or set(config) != {"format", "run", "plan", "comparison"}
            or config.get("format") != OPERATION_FORMAT):
        raise ValueError("A Wasserstein queue operation is required")
    descriptor(config["plan"])
    selected = config["comparison"]
    rows = comparisons(plan)
    order = selected.get("order") if isinstance(selected, dict) else None
    if (type(order) is not int or not 0 <= order < len(rows) or selected != rows[order]
            or config["run"] != operation_id(order)):
        raise ValueError("The queue operation differs from the frozen comparison")
    return plan["task_contract"]


def _identity(value, kind):
    if (not isinstance(value, dict) or value.get("pipeline") != PIPELINE
            or value.get("format") != f"darsia-alias-{kind}-v1"
            or re.fullmatch(r"[0-9a-f]{40}", str(value.get("commit", ""))) is None
            or value.get("backend") != "cpu" or value.get("precision") != "float64"
            or value.get("format_version") != 1 or value.get("storage_profile") != "full"
            or value.get("legacy_paths") is not False or value.get("execution_ready") is not True
            or value.get("runtime_support") != "darsia-role-io-runtime-v1"
            or "calibration_sha256" in value):
        raise ValueError("An explicit CPU-float64 Wasserstein alias identity is required")
    if re.fullmatch(r"[0-9a-f]{64}", str(value.get("wasserstein_contract_sha256", ""))) is None:
        raise ValueError("The original Wasserstein contract identity is required")


def validate_wasserstein_task(task):
    _identity(task, "task")
    from .pipeline.job_inputs import validate_catalog
    from .store_map import validate_input_reference
    if (task.get("run") != operation_id(task.get("comparison", {}).get("order"))
            or task.get("ordered_case_ids") != [task["run"]] or task.get("threads") != 1):
        raise ValueError("A Wasserstein task is exactly one comparison on one CPU thread")
    for key in ("configuration", "series", "series_manifest", "wasserstein_plan"):
        validate_input_reference(task.get(key))
    catalog = validate_catalog(task["input_catalog"])
    if (catalog.get("run") != task["run"] or catalog["configuration"] != task["configuration"]
            or not any(row["reference"] == task["wasserstein_plan"]
                       and "wasserstein.plan" in row["uses"] for row in catalog["inputs"])):
        raise ValueError("The Wasserstein task and consumed catalogue differ")
    return task


def validate_wasserstein_job(job):
    _identity(job, "job")
    from .store_map import validate_input_reference
    validate_input_reference(job.get("wasserstein_plan"))
    rows = job.get("wasserstein_comparisons")
    if (not isinstance(rows, list) or not rows
            or [row.get("order") for row in rows] != list(range(len(rows)))
            or job.get("expected_runs") != [operation_id(row["order"]) for row in rows]):
        raise ValueError("The Wasserstein job needs its exact ordered comparison coverage")
    return job


def validate_wasserstein_claim(task, job, *, config=None, plan=None):
    validate_wasserstein_task(task)
    validate_wasserstein_job(job)
    for key in ("commit", "precision", "backend", "wasserstein_contract_sha256", "wasserstein_plan",
                "series", "series_manifest", "job_id", "storage_profile"):
        if task.get(key) != job.get(key):
            raise ValueError("Wasserstein task/job identity differs: " + key)
    order = task["comparison"]["order"]
    if order >= len(job["wasserstein_comparisons"]) or task["comparison"] != job["wasserstein_comparisons"][order]:
        raise ValueError("Wasserstein task is outside this job's exact coverage")
    if config is not None:
        contract = validate_operation(config, plan)
        if (config["comparison"] != task["comparison"] or config["run"] != task["run"]
                or contract["content_sha256"] != task["wasserstein_contract_sha256"]):
            raise ValueError("Wasserstein consumed operation differs from its task")
    return task
