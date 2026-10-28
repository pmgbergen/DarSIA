"""One genuine saved Wasserstein comparison under the original worker I/O.

There is no secondary queue, process launcher, synthetic mass field or solver.
The original materializer owns field validation, physical grids, conservative
resizing, symmetric weights, transport solving and normalization. Only its I/O
locators are resolved through the already admitted whole-run binding.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import socket
import time

from .wasserstein_job_protocol import (
    PIPELINE, RESULT_FORMAT, descriptor, validate_operation, validate_wasserstein_task,
)


def _frame(reader, field, original_index, number):
    from .wasserstein_job_inputs import relative_descriptor
    source = relative_descriptor(original_index, field.frames[number])
    path = reader.path(source)
    frames = list(field.frames)
    frames[number] = dict(frames[number], path=str(path))
    # Retain the original archive reader and all shape/time/unit checks. Never
    # write a relocated index or let its relative locator fall through to disk.
    values, display = replace(field, frames=tuple(frames)).load_frame(number)
    reader.path(source)  # A changed admitted file must fail after consumption too.
    return values, display


def _solve_operation(reader, plan, selected):
    """Return only a scalar original solver row; no arrays escape this scope."""
    import numpy as np
    from . import wasserstein_materialization as original
    from .wasserstein_job_inputs import PRODUCTS
    contract = plan["task_contract"]
    mass = reader.json(descriptor(contract["mass_result"]))
    if (mass.get("complete") is not True
            or mass.get("format") != "darsia-fluidflower-mass-analysis-result-v1"
            or mass.get("mass_analysis_plan_id") != contract["mass_analysis_plan_id"]
            or mass.get("mass_task_contract_sha256") != contract["mass_task_contract_sha256"]):
        raise ValueError("Original mass results differ from the comparison contract")
    rows = {row["run_id"]: row for row in mass["run_results"]}
    references = original._reference_map(contract)
    relocated = {}
    for run in contract["run_ids"]:
        for product in PRODUCTS:
            source = descriptor(references[(run, product)])
            relocated[(run, product)] = dict(source, path=str(reader.path(source)))
    # The total-mass index supplies the original RunInputs.shape_rc even when
    # the selected comparison is aqueous or free gas. No extra array is read.
    runs = [original._load_run_inputs(rows[run], relocated, PRODUCTS[:-1]) for run in contract["run_ids"]]
    times = runs[0].times_hours
    if any(run.times_hours != times for run in runs[1:]):
        raise ValueError("Wasserstein inputs no longer share one exact target-time grid")
    index = selected["time_index"]
    if index >= len(times) or times[index] != selected["time_hours"]:
        raise ValueError("Actual mass field time differs from the frozen comparison")
    domain_rows = {row["domain_id"]: row for row in contract["analysis_domains"].get("domains", [])}
    domain = selected["domain_id"]
    cell = float(contract["spatial_discretization"]["target_cell_size_m"])
    grids = {run.run_id: original._domain_grid(domain, domain_rows, run, cell) for run in runs}
    signatures = {(grid.target_shape_rc, round(grid.width_m, 12), round(grid.height_m, 12)) for grid in grids.values()}
    if len(signatures) != 1:
        raise ValueError("Selected runs do not share the original physical comparison grid")
    weights = []
    for run in runs:
        values, _ = _frame(reader, run.transport_weight, references[(run.run_id, "transport_weight")]["path"], 0)
        weights.append(original._reduced_weight(values, grids[run.run_id], label=run.run_id + "/" + domain))
    if any(not np.allclose(weights[0], value, rtol=1e-9, atol=1e-12) for value in weights[1:]):
        raise ValueError("Transport weights differ between runs; no symmetric cost field exists")
    run_map = {run.run_id: run for run in runs}
    component = selected["mass_component"]
    arrays = []
    for run_id in (selected["run_1"], selected["run_2"]):
        values, _ = _frame(reader, run_map[run_id].fields[component], references[(run_id, component)]["path"], index)
        arrays.append(original._reduced_field(values, grids[run_id], label=run_id + "/" + component))
    return original._solve(original.SolveRequest(order=selected["order"],
        run_1=run_map[selected["run_1"]], run_2=run_map[selected["run_2"]], time_index=index,
        domain=grids[selected["run_1"]], component=component,
        mass_1=arrays[0], mass_2=arrays[1], weight=weights[0]))


def validate_solver_row(row, selected):
    """Check the complete original scalar result, without rerunning science."""
    import math
    from darsia._opencv_runtime import validate_observation
    from .wasserstein_materialization import RESULT_COLUMNS
    if not isinstance(row, dict) or set(row) != set(RESULT_COLUMNS) | {"order", "opencv_ipp"}:
        raise ValueError("Wasserstein solver result needs the complete original row schema")
    identity = {key: row.get(key) for key in selected}
    if identity != selected:
        raise ValueError("Solver result belongs to another comparison")
    if row.get("status") not in {"success", "both_zero_mass"} or row.get("solver_converged") is not True:
        raise RuntimeError("Wasserstein comparison has no converged distance: " + str(row.get("status")))
    def number(key, *, positive=False, nullable=False):
        value = row[key]
        if value is None and nullable:
            return value
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (positive and value == 0)):
            raise RuntimeError("Wasserstein result has an invalid finite scalar: " + key)
        return value

    for key in ("order", "time_index", "solver_iterations", "solver_warning_count", "grid_rows", "grid_columns"):
        if type(row[key]) is not int or row[key] < (1 if key.startswith("grid_") else 0):
            raise RuntimeError("Wasserstein result has an invalid integer: " + key)
    if type(row["time_hours"]) not in (int, float) or not math.isfinite(row["time_hours"]):
        raise RuntimeError("Wasserstein result has an invalid target time")
    for key in ("run_1", "run_2", "domain_id", "mass_component"):
        if not isinstance(row[key], str) or not row[key]:
            raise RuntimeError("Wasserstein result has an invalid identity: " + key)
    if (not isinstance(row["solver_warnings"], str)
            or bool(row["solver_warnings"]) != bool(row["solver_warning_count"])
            or row["transport_mass_balancing"] != "each_distribution_scaled_to_pair_mean"):
        raise RuntimeError("Wasserstein result has invalid solver diagnostics")
    for key in ("runtime_seconds", "source_mass_1_g", "source_mass_2_g", "pair_mean_detected_mass_g",
                "absolute_mass_difference_g", "relative_mass_difference", "transport_cost_mass_m",
                "injected_mass_reference_g", "reference_plume_radius_m"):
        number(key)
    for key in ("rig_mean_dimension_m", "rig_diagonal_m", "domain_width_m", "domain_height_m",
                "cell_width_m", "cell_height_m"):
        number(key, positive=True)
    normalized = ("mass_based_distance_m", "rig_based_normalized_distance",
                  "rig_diagonal_normalized_distance", "plume_based_normalized_distance")
    for key in normalized:
        number(key, nullable=True)
    observation = row["opencv_ipp"]
    if not isinstance(observation, dict) or not {"before", "after"} <= set(observation):
        raise ValueError("Wasserstein result needs completed original solver observations")
    validate_observation(observation)
    if (observation["before"]["stage"] != "wasserstein-solve:before"
            or observation["after"]["stage"] != "wasserstein-solve:after"):
        raise ValueError("Wasserstein observations belong to another scientific stage")
    mass_1, mass_2, cost = row["source_mass_1_g"], row["source_mass_2_g"], row["transport_cost_mass_m"]
    zero_masses = mass_1 <= 1e-12 and mass_2 <= 1e-12  # Original np.isclose(m, 0, atol=1e-12).
    if row["status"] == "both_zero_mass":
        if not zero_masses or cost != 0 or row["solver_iterations"] != 0 or row["solver_warning_count"] != 0:
            raise RuntimeError("Both-zero Wasserstein result contradicts its masses or distance")
    elif mass_1 <= 1e-12 or mass_2 <= 1e-12 or row["solver_iterations"] < 1:
        raise RuntimeError("Successful Wasserstein result lacks nonzero solved mass")

    def same(key, expected):
        actual = row[key]
        if ((actual is None) != (expected is None)
                or (expected is not None and not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0))):
            raise RuntimeError("Wasserstein result has inconsistent derived scalar: " + key)

    mean = .5 * (mass_1 + mass_2)
    difference = abs(mass_1 - mass_2)
    same("pair_mean_detected_mass_g", mean)
    same("absolute_mass_difference_g", difference)
    same("relative_mass_difference", difference / mean if mean > 0 else 0.0)
    same("cell_width_m", row["domain_width_m"] / row["grid_columns"])
    same("cell_height_m", row["domain_height_m"] / row["grid_rows"])
    # Reproduce only scalar identities of _normalized; input reduction/solving
    # remains wholly owned by the original scientific implementation.
    def normalize(value, denominator):
        return value / denominator if denominator > 0 else (0.0 if cost == 0 else None)
    mass_based = normalize(cost, mean)
    same("mass_based_distance_m", mass_based)
    same("rig_based_normalized_distance", normalize(cost, row["injected_mass_reference_g"] * row["rig_mean_dimension_m"]))
    same("rig_diagonal_normalized_distance", None if mass_based is None else normalize(mass_based, row["rig_diagonal_m"]))
    same("plume_based_normalized_distance", normalize(cost, row["injected_mass_reference_g"] * row["reference_plume_radius_m"]))
    return row


def _solve_capped(reader, plan, selected):
    """Use run_series' existing OpenCV/BLAS cap and restore it on every exit."""
    from darsia._opencv_runtime import scientific_thread
    ipp_start = scientific_thread("wasserstein-job:start")
    import cv2
    import scipy.linalg  # Load the same BLAS libraries before inspecting/capping them.
    from threadpoolctl import threadpool_info, threadpool_limits
    old_threads = cv2.getNumThreads()
    try:
        cv2.setNumThreads(1)
        with threadpool_limits(limits=1):
            runtime = dict(requested_threads=1, cv2_threads=cv2.getNumThreads(),
                           threadpools=threadpool_info(), opencv_ipp=ipp_start)
            return _solve_operation(reader, plan, selected), runtime
    finally:
        cv2.setNumThreads(old_threads)


def run_wasserstein(task, output, backend, precision, on_progress=None):
    """Execute one comparison only after existing whole-run worker admission."""
    validate_wasserstein_task(task)
    if backend != "cpu" or precision != "float64" or precision != task["precision"]:
        raise ValueError("The existing Wasserstein solver requires its own CPU-float64 task")
    if task.get("resume_from") is not None:
        raise ValueError("Wasserstein comparison attempts restart atomically; frame-prefix resume is unsupported")
    from .pipeline.run_io import for_task
    binding = for_task(task)
    if binding is None:
        raise ValueError("Wasserstein execution requires its exact admitted per-task I/O binding")
    binding.require_output(output)
    from .mass.n1_pipeline import DataReader, _write_json, asset
    from .pipeline.job_queue import retry_transient
    reader = DataReader(unc_only=False)
    config = reader.json(binding.configuration_descriptor)
    plan = reader.json(config["plan"])
    contract = validate_operation(config, plan)
    if (config["run"] != task["run"] or config["comparison"] != task["comparison"]
            or contract["content_sha256"] != task["wasserstein_contract_sha256"]):
        raise ValueError("Consumed Wasserstein operation differs from the owned task")
    output = Path(output)
    retry_transient(lambda: output.mkdir(parents=True, exist_ok=True))
    if retry_transient(lambda: any(output.iterdir())):
        raise FileExistsError("A Wasserstein comparison attempt must be fresh")

    def progress(stage):
        if on_progress:
            on_progress(dict(state="wasserstein_progress", run=task["run"], stage=stage,
                             comparison=config["comparison"]))

    started = time.perf_counter()
    progress("comparison_started")
    row, runtime = _solve_capped(reader, plan, config["comparison"])
    # Keep real diagnostic output if convergence fails; such an attempt raises
    # and cannot acquire a successful shared queue completion.
    files = {"comparison.json": _write_json(output / "comparison.json", row)}
    validate_solver_row(row, config["comparison"])
    from .wasserstein_materialization import _write_csv, _write_xlsx
    for filename, writer in (("wasserstein_distances.csv", _write_csv), ("wasserstein_distances.xlsx", _write_xlsx)):
        writer(output / filename, [row])
        files[filename] = asset(output / filename)
    progress("comparison_computed")
    return dict(format=RESULT_FORMAT, complete=True, pipeline=PIPELINE, run=task["run"],
        backend=backend, precision=precision, host=socket.gethostname(),
        comparison=config["comparison"], wasserstein_plan=task["wasserstein_plan"],
        wasserstein_contract_sha256=contract["content_sha256"], files=files,
        solver_status=row["status"], solver_converged=row["solver_converged"],
        comparisons=1, frames=0, runtime=runtime, wall_seconds=time.perf_counter() - started,
        worker_references_closed=True,
        validation=dict(state="not_scored", reason="Queue completion is not scientific acceptance"))
