"""Materialize reproducible CPU-float64 Wasserstein comparisons."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openpyxl import Workbook

import darsia
from darsia.fluidflower.execution import (
    AdapterDescriptor,
    TaskExecutionContext,
    load_task_plan,
    verify_file_reference,
)
from darsia.fluidflower.mass_artifacts import IndexedFieldProduct, load_field_index
from darsia.fluidflower.project import FluidFlowerProject, sha256_file, utc_now
from darsia.fluidflower.wasserstein import (
    MASS_ANALYSIS_RESULT_FORMAT,
    WASSERSTEIN_MASS_COMPONENTS,
    WASSERSTEIN_REVISION_FORMAT,
    WASSERSTEIN_TASK_CONTRACT_FORMAT,
)

WASSERSTEIN_MATERIALIZER_VERSION = "1.0.0"
WASSERSTEIN_RESULT_FORMAT = "darsia-fluidflower-wasserstein-result-v1"
WASSERSTEIN_PROVENANCE_FORMAT = "darsia-fluidflower-wasserstein-provenance-v1"
WASSERSTEIN_INTERMEDIATE_INDEX_FORMAT = (
    "darsia-fluidflower-wasserstein-intermediate-index-v1"
)

RESULT_COLUMNS = (
    "run_1",
    "run_2",
    "time_index",
    "time_hours",
    "domain_id",
    "mass_component",
    "status",
    "solver_converged",
    "solver_iterations",
    "solver_warning_count",
    "solver_warnings",
    "runtime_seconds",
    "source_mass_1_g",
    "source_mass_2_g",
    "pair_mean_detected_mass_g",
    "absolute_mass_difference_g",
    "relative_mass_difference",
    "transport_cost_mass_m",
    "mass_based_distance_m",
    "rig_based_normalized_distance",
    "rig_diagonal_normalized_distance",
    "plume_based_normalized_distance",
    "injected_mass_reference_g",
    "reference_plume_radius_m",
    "rig_mean_dimension_m",
    "rig_diagonal_m",
    "domain_width_m",
    "domain_height_m",
    "grid_rows",
    "grid_columns",
    "cell_width_m",
    "cell_height_m",
    "transport_mass_balancing",
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from .mass_artifacts import _atomic_json as _write_metadata
    return _write_metadata(path, payload)


def _safe_identifier(value: str) -> str:
    result = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in str(value)
    ).strip("-")
    return result or "value"


def _reference_map(
    contract: Mapping[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    rows = (contract.get("mass_result") or {}).get("product_artifacts") or []
    references: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        row = deepcopy(dict(raw))
        run_id = str(row.get("run_id") or "")
        product = str(row.get("product") or row.get("component") or "")
        if run_id and product:
            references[(run_id, product)] = row
    return references


def _load_mass_manifest(contract: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    reference = dict(contract.get("mass_result") or {})
    path = verify_file_reference(
        str(reference.get("path") or ""),
        str(reference.get("sha256") or ""),
        label="Mass-analysis result manifest",
    )
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(payload, dict)
        or payload.get("format") != MASS_ANALYSIS_RESULT_FORMAT
    ):
        raise ValueError("Unsupported mass-analysis result manifest")
    if payload.get("complete") is not True:
        raise ValueError("Mass-analysis result is incomplete")
    return path, payload


def _float_series(row: Mapping[str, Any], key: str, count: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in row.get(key) or [])
    if len(values) != count or any(
        not math.isfinite(value) or value < 0.0 for value in values
    ):
        raise ValueError(f"Invalid {key} for {row.get('run_id')}")
    return values


@dataclass(frozen=True)
class RunInputs:
    run_id: str
    times_hours: tuple[float, ...]
    injected_mass_g: tuple[float, ...]
    reference_plume_radius_m: tuple[float, ...]
    width_m: float
    height_m: float
    fields: dict[str, IndexedFieldProduct]
    transport_weight: IndexedFieldProduct

    @property
    def shape_rc(self) -> tuple[int, int]:
        return self.fields[WASSERSTEIN_MASS_COMPONENTS[0]].shape_rc


def _load_run_inputs(
    row: Mapping[str, Any],
    references: Mapping[tuple[str, str], Mapping[str, Any]],
    selected_components: Sequence[str],
) -> RunInputs:
    run_id = str(row.get("run_id") or "")
    times = tuple(float(value) for value in row.get("target_times_hours") or [])
    geometry = dict(row.get("geometry") or {})
    width_m = float(geometry.get("width_m") or 0.0)
    height_m = float(geometry.get("height_m") or 0.0)
    if not run_id or not times or width_m <= 0.0 or height_m <= 0.0:
        raise ValueError(f"Incomplete Wasserstein input metadata for {run_id}")

    fields: dict[str, IndexedFieldProduct] = {}
    for product in selected_components:
        reference = dict(references.get((run_id, product)) or {})
        field = load_field_index(
            str(reference.get("path") or ""),
            str(reference.get("sha256") or ""),
            run_id=run_id,
            product=product,
            expected_times_hours=times,
        )
        if field.unit.casefold().replace(" ", "") not in {
            "gpixel-1",
            "g/pixel",
            "gpx-1",
        }:
            raise ValueError(f"{run_id}/{product} must contain mass in g per pixel")
        fields[product] = field

    reference = dict(references.get((run_id, "transport_weight")) or {})
    transport_weight = load_field_index(
        str(reference.get("path") or ""),
        str(reference.get("sha256") or ""),
        run_id=run_id,
        product="transport_weight",
    )
    if transport_weight.unit.casefold() not in {"1", "dimensionless"}:
        raise ValueError(f"{run_id}/transport_weight must be dimensionless")
    if len(transport_weight.frames) not in {1, len(times)}:
        raise ValueError(
            f"{run_id}/transport_weight must be static or use the mass time grid"
        )
    if (
        len(transport_weight.frames) == len(times)
        and transport_weight.times_hours != times
    ):
        raise ValueError(
            f"{run_id}/transport_weight time grid does not match mass fields"
        )
    shapes = {field.shape_rc for field in fields.values()}
    shapes.add(transport_weight.shape_rc)
    if len(shapes) != 1:
        raise ValueError(f"Wasserstein fields for {run_id} do not share one grid")
    return RunInputs(
        run_id=run_id,
        times_hours=times,
        injected_mass_g=_float_series(row, "injected_mass_g", len(times)),
        reference_plume_radius_m=_float_series(
            row, "reference_plume_radius_m", len(times)
        ),
        width_m=width_m,
        height_m=height_m,
        fields=fields,
        transport_weight=transport_weight,
    )


@dataclass(frozen=True)
class DomainGrid:
    domain_id: str
    row_slice: slice
    column_slice: slice
    width_m: float
    height_m: float
    target_shape_rc: tuple[int, int]

    @property
    def cell_area_m2(self) -> float:
        rows, columns = self.target_shape_rc
        return (self.width_m / columns) * (self.height_m / rows)


def _domain_grid(
    domain_id: str,
    domain_rows: Mapping[str, Mapping[str, Any]],
    run: RunInputs,
    target_cell_size_m: float,
) -> DomainGrid:
    rows, columns = run.shape_rc
    if domain_id == "whole_active_domain":
        normalized = (0.0, 0.0, 1.0, 1.0)
    else:
        raw = domain_rows.get(domain_id)
        if raw is None:
            raise ValueError(f"Unknown analysis domain: {domain_id}")
        normalized = tuple(
            float(value) for value in raw.get("normalized_xyxy_top_left") or []
        )
        if len(normalized) != 4:
            raise ValueError(f"Analysis domain {domain_id} has no normalized rectangle")
    x0, y0, x1, y1 = normalized
    row_0 = max(0, min(rows - 1, round(y0 * rows)))
    row_1 = max(row_0 + 1, min(rows, round(y1 * rows)))
    column_0 = max(0, min(columns - 1, round(x0 * columns)))
    column_1 = max(column_0 + 1, min(columns, round(x1 * columns)))
    source_rows = row_1 - row_0
    source_columns = column_1 - column_0
    width_m = run.width_m * source_columns / columns
    height_m = run.height_m * source_rows / rows
    target_rows = max(1, round(height_m / target_cell_size_m))
    target_columns = max(1, round(width_m / target_cell_size_m))
    if target_rows > source_rows or target_columns > source_columns:
        raise ValueError(
            f"Target grid would upsample {run.run_id}/{domain_id}: "
            f"{source_rows}x{source_columns} to {target_rows}x{target_columns}"
        )
    return DomainGrid(
        domain_id=domain_id,
        row_slice=slice(row_0, row_1),
        column_slice=slice(column_0, column_1),
        width_m=width_m,
        height_m=height_m,
        target_shape_rc=(target_rows, target_columns),
    )


def _nonnegative(array: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(values))))
    if float(np.min(values)) < -1.0e-10 * scale:
        raise ValueError(f"{label} contains negative mass")
    return np.maximum(values, 0.0)


def _conservative_resize(
    array: np.ndarray, target_shape_rc: tuple[int, int]
) -> np.ndarray:
    source = np.asarray(array, dtype=np.float64)
    source_mass = float(np.sum(source, dtype=np.float64))
    if source.shape == target_shape_rc:
        return source.copy()
    target = cv2.resize(
        source,
        (target_shape_rc[1], target_shape_rc[0]),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float64, copy=False)
    target_mass = float(np.sum(target, dtype=np.float64))
    if source_mass == 0.0:
        target.fill(0.0)
    elif target_mass <= 0.0:
        raise ValueError("Mass-preserving resize lost a non-zero distribution")
    else:
        target *= source_mass / target_mass
    return target


def _nearest_resize(array: np.ndarray, target_shape_rc: tuple[int, int]) -> np.ndarray:
    source = np.asarray(array, dtype=np.float64)
    if source.shape == target_shape_rc:
        return source.copy()
    return cv2.resize(
        source,
        (target_shape_rc[1], target_shape_rc[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float64, copy=False)


def _reduced_field(
    values: np.ndarray,
    domain: DomainGrid,
    *,
    label: str,
) -> np.ndarray:
    cropped = _nonnegative(
        values[domain.row_slice, domain.column_slice],
        label,
    )
    return _conservative_resize(cropped, domain.target_shape_rc)


def _reduced_weight(
    values: np.ndarray, domain: DomainGrid, *, label: str
) -> np.ndarray:
    cropped = np.asarray(
        values[domain.row_slice, domain.column_slice], dtype=np.float64
    )
    if not np.all(np.isfinite(cropped)) or float(np.min(cropped)) <= 0.0:
        raise ValueError(f"{label} must contain finite positive transport weights")
    return _nearest_resize(cropped, domain.target_shape_rc)


def _mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / len(values))


@dataclass(frozen=True)
class SolveRequest:
    order: int
    run_1: RunInputs
    run_2: RunInputs
    time_index: int
    domain: DomainGrid
    component: str
    mass_1: np.ndarray
    mass_2: np.ndarray
    weight: np.ndarray


def _solver_options() -> dict[str, Any]:
    return {
        "mobility_mode": darsia.MobilityMode.CELL_BASED,
        "l1_mode": darsia.L1Mode.RAVIART_THOMAS,
        "bregman_update": lambda iteration: iteration in {0, 10, 20},
        "num_iter": 20,
        "tol_distance": 1.0e-4,
        "tol_increment": 1.0e-2,
        "linear_solver": "direct",
        "return_info": True,
        "verbose": False,
    }


def _normalized(cost: float, denominator: float, *, zero_cost: bool) -> float | None:
    if denominator > 0.0:
        return float(cost / denominator)
    return 0.0 if zero_cost else None


def _solve(request: SolveRequest) -> dict[str, Any]:
    from darsia._opencv_runtime import scientific_thread, completed_frame
    ipp_before = scientific_thread("wasserstein-solve:before")
    started = time.perf_counter()
    mass_1_g = float(np.sum(request.mass_1, dtype=np.float64))
    mass_2_g = float(np.sum(request.mass_2, dtype=np.float64))
    pair_mean_g = 0.5 * (mass_1_g + mass_2_g)
    absolute_difference = abs(mass_1_g - mass_2_g)
    relative_difference = (
        absolute_difference / pair_mean_g if pair_mean_g > 0.0 else 0.0
    )
    injected_mass_g = _mean(
        (
            request.run_1.injected_mass_g[request.time_index],
            request.run_2.injected_mass_g[request.time_index],
        )
    )
    plume_radius_m = _mean(
        (
            request.run_1.reference_plume_radius_m[request.time_index],
            request.run_2.reference_plume_radius_m[request.time_index],
        )
    )
    rig_mean_dimension_m = 0.25 * (
        request.run_1.width_m
        + request.run_1.height_m
        + request.run_2.width_m
        + request.run_2.height_m
    )
    rig_diagonal_m = _mean(
        (
            math.hypot(request.run_1.width_m, request.run_1.height_m),
            math.hypot(request.run_2.width_m, request.run_2.height_m),
        )
    )

    status = "success"
    converged = True
    iterations = 0
    solver_warnings: list[str] = []
    cost: float | None
    if np.isclose(mass_1_g, 0.0, atol=1.0e-12) and np.isclose(
        mass_2_g, 0.0, atol=1.0e-12
    ):
        cost = 0.0
        status = "both_zero_mass"
    elif np.isclose(mass_1_g, 0.0, atol=1.0e-12) or np.isclose(
        mass_2_g, 0.0, atol=1.0e-12
    ):
        cost = None
        status = "single_zero_mass"
        converged = False
    else:
        balanced_1 = request.mass_1 * (pair_mean_g / mass_1_g)
        balanced_2 = request.mass_2 * (pair_mean_g / mass_2_g)
        density_1 = balanced_1 / request.domain.cell_area_m2
        density_2 = balanced_2 / request.domain.cell_area_m2
        metadata = {
            "width": request.domain.width_m,
            "height": request.domain.height_m,
            "space_dim": 2,
            "scalar": True,
        }
        source = darsia.Image(density_1, **metadata)
        destination = darsia.Image(density_2, **metadata)
        weight = darsia.Image(request.weight, **metadata)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            raw_cost, info = darsia.wasserstein_distance(
                source,
                destination,
                method="bregman",
                weight=weight,
                options=_solver_options(),
            )
        solver_warnings = sorted({str(warning.message) for warning in caught_warnings})
        cost = float(raw_cost)
        converged = bool(info.get("converged"))
        iterations = int(info.get("number_iterations") or 0) + 1
        if not math.isfinite(cost) or cost < 0.0:
            raise RuntimeError("Wasserstein solver returned an invalid transport cost")
        if not converged:
            status = "solver_not_converged"

    zero_cost = cost == 0.0
    mass_based = (
        None if cost is None else _normalized(cost, pair_mean_g, zero_cost=zero_cost)
    )
    rig_based = (
        None
        if cost is None
        else _normalized(
            cost,
            injected_mass_g * rig_mean_dimension_m,
            zero_cost=zero_cost,
        )
    )
    rig_diagonal = (
        None
        if mass_based is None
        else _normalized(mass_based, rig_diagonal_m, zero_cost=zero_cost)
    )
    plume_based = (
        None
        if cost is None
        else _normalized(
            cost,
            injected_mass_g * plume_radius_m,
            zero_cost=zero_cost,
        )
    )
    rows, columns = request.domain.target_shape_rc
    return {
        "opencv_ipp": completed_frame(ipp_before, "wasserstein-solve:after"),
        "run_1": request.run_1.run_id,
        "run_2": request.run_2.run_id,
        "time_index": request.time_index,
        "time_hours": request.run_1.times_hours[request.time_index],
        "domain_id": request.domain.domain_id,
        "mass_component": request.component,
        "status": status,
        "solver_converged": converged,
        "solver_iterations": iterations,
        "solver_warning_count": len(solver_warnings),
        "solver_warnings": " | ".join(solver_warnings),
        "runtime_seconds": time.perf_counter() - started,
        "source_mass_1_g": mass_1_g,
        "source_mass_2_g": mass_2_g,
        "pair_mean_detected_mass_g": pair_mean_g,
        "absolute_mass_difference_g": absolute_difference,
        "relative_mass_difference": relative_difference,
        "transport_cost_mass_m": cost,
        "mass_based_distance_m": mass_based,
        "rig_based_normalized_distance": rig_based,
        "rig_diagonal_normalized_distance": rig_diagonal,
        "plume_based_normalized_distance": plume_based,
        "injected_mass_reference_g": injected_mass_g,
        "reference_plume_radius_m": plume_radius_m,
        "rig_mean_dimension_m": rig_mean_dimension_m,
        "rig_diagonal_m": rig_diagonal_m,
        "domain_width_m": request.domain.width_m,
        "domain_height_m": request.domain.height_m,
        "grid_rows": rows,
        "grid_columns": columns,
        "cell_width_m": request.domain.width_m / columns,
        "cell_height_m": request.domain.height_m / rows,
        "transport_mass_balancing": "each_distribution_scaled_to_pair_mean",
        "order": request.order,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(RESULT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in RESULT_COLUMNS})


def _write_xlsx(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    workbook = Workbook(write_only=True)
    worksheet = workbook.create_sheet("Wasserstein distances")
    worksheet.append(list(RESULT_COLUMNS))
    for row in rows:
        worksheet.append([row.get(key) for key in RESULT_COLUMNS])
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def _artifact(staging: Path, path: Path, product: str) -> dict[str, Any]:
    return {
        "product": product,
        "path": path.relative_to(staging).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _input_digest(runs: Sequence[RunInputs]) -> str:
    payload = [
        {
            "run_id": run.run_id,
            "fields": {
                product: sha256_file(field.index_path)
                for product, field in sorted(run.fields.items())
            },
            "transport_weight": sha256_file(run.transport_weight.index_path),
        }
        for run in runs
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class WassersteinAnalysisAdapter:
    """Compare synchronized mass fields using DarSIA's weighted W1 solver."""

    descriptor = AdapterDescriptor(
        key="wasserstein-cpu-float64-v1",
        version=WASSERSTEIN_MATERIALIZER_VERSION,
        title="Weighted Wasserstein analysis",
        modules=("wasserstein",),
        backends=("cpu-float64",),
        description=(
            "Computes mass-preserving, permeability-weighted DarSIA Wasserstein "
            "distances and FF-UM-compatible normalized metrics."
        ),
    )

    def execute(
        self,
        task: Mapping[str, Any],
        context: TaskExecutionContext,
    ) -> Mapping[str, Any]:
        if task.get("module") != "wasserstein":
            raise ValueError("The Wasserstein adapter only accepts Wasserstein tasks")
        if (task.get("backend") or {}).get("name") != "cpu-float64":
            raise ValueError("Wasserstein analysis uses the CPU float64 contract")
        plan, plan_path, plan_file_sha256 = load_task_plan(task)
        if plan.get("format") != WASSERSTEIN_REVISION_FORMAT:
            raise ValueError("The task does not reference a Wasserstein revision")
        contract = dict(plan.get("task_contract") or {})
        if contract.get("format") != WASSERSTEIN_TASK_CONTRACT_FORMAT:
            raise ValueError("Unsupported Wasserstein task contract")
        if not bool((plan.get("validation") or {}).get("execution_ready")):
            raise RuntimeError("The saved Wasserstein plan is not execution-ready")

        project = FluidFlowerProject.load(Path(str(task["project"]["root"])))
        if project.project_id != str(task["project"].get("project_id") or ""):
            raise RuntimeError("Task and project identities do not match")
        expected_source = str(task["project"].get("source_manifest_sha256") or "")
        if expected_source:
            verify_file_reference(
                project.source_manifest_path,
                expected_source,
                label="Project source manifest",
            )

        mass_manifest_path, mass_manifest = _load_mass_manifest(contract)
        manifest_rows = {
            str(row.get("run_id") or ""): dict(row)
            for row in mass_manifest.get("run_results") or []
            if isinstance(row, Mapping)
        }
        references = _reference_map(contract)
        run_ids = [str(value) for value in contract.get("run_ids") or []]
        comparison = dict(contract.get("comparison") or {})
        components = [str(value) for value in comparison.get("mass_components") or []]
        runs = [
            _load_run_inputs(manifest_rows[run_id], references, components)
            for run_id in run_ids
        ]
        if len(runs) < 2:
            raise ValueError("Wasserstein analysis requires at least two runs")
        times = runs[0].times_hours
        if any(run.times_hours != times for run in runs[1:]):
            raise ValueError("Wasserstein runs do not share one target-time grid")
        resolved_selection = dict(contract.get("resolved_time_selection") or {})
        selected_time_indices = [
            int(value)
            for value in resolved_selection.get("selected_time_indices") or []
        ]
        if (
            not selected_time_indices
            or selected_time_indices != sorted(set(selected_time_indices))
            or selected_time_indices[0] < 0
            or selected_time_indices[-1] >= len(times)
        ):
            raise ValueError("Wasserstein task has an invalid resolved time selection")
        selected_times = [times[index] for index in selected_time_indices]
        if selected_times != [
            float(value)
            for value in resolved_selection.get("selected_times_hours") or []
        ]:
            raise ValueError("Resolved Wasserstein time identities do not match inputs")

        domain_rows = {
            str(row.get("domain_id") or ""): dict(row)
            for row in (contract.get("analysis_domains") or {}).get("domains") or []
            if isinstance(row, Mapping)
        }
        domain_ids = [str(value) for value in comparison.get("domains") or []]
        target_cell_size = float(
            (contract.get("spatial_discretization") or {}).get("target_cell_size_m")
        )
        domains_by_run = {
            run.run_id: {
                domain_id: _domain_grid(domain_id, domain_rows, run, target_cell_size)
                for domain_id in domain_ids
            }
            for run in runs
        }
        for domain_id in domain_ids:
            signatures = {
                (
                    grid.target_shape_rc,
                    round(grid.width_m, 12),
                    round(grid.height_m, 12),
                )
                for grid in (domains_by_run[run.run_id][domain_id] for run in runs)
            }
            if len(signatures) != 1:
                raise ValueError(
                    f"Runs do not share a physical target grid for {domain_id}"
                )

        transport_weights: dict[tuple[str, str], np.ndarray] = {}
        for run in runs:
            values, _ = run.transport_weight.load_frame(0)
            for domain_id in domain_ids:
                domain = domains_by_run[run.run_id][domain_id]
                transport_weights[(run.run_id, domain_id)] = _reduced_weight(
                    values,
                    domain,
                    label=f"{run.run_id}/{domain_id}/transport_weight",
                )
        common_weights: dict[str, np.ndarray] = {}
        for domain_id in domain_ids:
            candidates = [transport_weights[(run.run_id, domain_id)] for run in runs]
            reference_weight = candidates[0]
            if any(
                not np.allclose(reference_weight, candidate, rtol=1.0e-9, atol=1.0e-12)
                for candidate in candidates[1:]
            ):
                raise ValueError(
                    f"Transport weights differ between runs for {domain_id}; "
                    "a symmetric distance requires one shared physical cost field"
                )
            common_weights[domain_id] = reference_weight

        pairs = [
            (str(row.get("run_1") or ""), str(row.get("run_2") or ""))
            for row in comparison.get("pairs") or []
        ]
        run_map = {run.run_id: run for run in runs}
        workers = min(
            24,
            max(
                1,
                int(
                    (task.get("resources") or {}).get("cpu_workers")
                    or (contract.get("execution_profile") or {}).get("workers")
                    or 1
                ),
            ),
        )
        total_requests = (
            len(selected_time_indices) * len(pairs) * len(domain_ids) * len(components)
        )
        context.checkpoint(
            {
                "stage": "inputs_verified",
                "run_count": len(runs),
                "comparison_count": total_requests,
            }
        )

        intermediate_root = context.staging / "intermediate"
        intermediate_root.mkdir(parents=True, exist_ok=True)
        completed_rows: list[dict[str, Any]] = []
        intermediate_references: list[dict[str, Any]] = []
        for selection_index, time_index in enumerate(selected_time_indices):
            time_hours = times[time_index]
            chunk_id = f"wasserstein-time-{time_index:06d}"
            committed = context.committed_chunk(chunk_id)
            committed_state = dict(committed.get("state") or {})
            saved_rows = committed_state.get("rows")
            saved_references = committed_state.get("intermediate_references")
            if (
                committed
                and isinstance(saved_rows, list)
                and isinstance(saved_references, list)
            ):
                context.restore_chunk(chunk_id, context.staging)
                completed_rows.extend(
                    dict(row) for row in saved_rows if isinstance(row, Mapping)
                )
                intermediate_references.extend(
                    dict(row) for row in saved_references if isinstance(row, Mapping)
                )
                context.checkpoint(
                    {
                        "stage": "time_restored",
                        "time_index": time_index,
                        "time_hours": time_hours,
                        "completed_comparisons": len(completed_rows),
                        "comparison_count": total_requests,
                    }
                )
                context.raise_if_interrupted()
                continue

            if context.cancellation_requested():
                raise RuntimeError("Wasserstein analysis was cancelled")
            reduced: dict[tuple[str, str, str], np.ndarray] = {}
            for run in runs:
                for component in components:
                    values, _ = run.fields[component].load_frame(time_index)
                    for domain_id in domain_ids:
                        domain = domains_by_run[run.run_id][domain_id]
                        reduced[(run.run_id, component, domain_id)] = _reduced_field(
                            values,
                            domain,
                            label=(
                                f"{run.run_id}/{component}/{domain_id}/"
                                f"time-{time_index}"
                            ),
                        )

            requests: list[SolveRequest] = []
            for pair_index, (run_1_id, run_2_id) in enumerate(pairs):
                for domain_index, domain_id in enumerate(domain_ids):
                    domain = domains_by_run[run_1_id][domain_id]
                    for component_index, component in enumerate(components):
                        order = (
                            (selection_index * len(pairs) + pair_index)
                            * len(domain_ids)
                            + domain_index
                        ) * len(components) + component_index
                        requests.append(
                            SolveRequest(
                                order=order,
                                run_1=run_map[run_1_id],
                                run_2=run_map[run_2_id],
                                time_index=time_index,
                                domain=domain,
                                component=component,
                                mass_1=reduced[(run_1_id, component, domain_id)],
                                mass_2=reduced[(run_2_id, component, domain_id)],
                                weight=common_weights[domain_id],
                            )
                        )

            batch_rows: list[dict[str, Any]] = []
            chunk_staging = context.prepare_chunk(chunk_id)
            chunk_intermediate_root = chunk_staging / "intermediate"
            chunk_intermediate_root.mkdir(parents=True, exist_ok=True)
            chunk_references: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=min(workers, len(requests))) as pool:
                futures = {
                    pool.submit(_solve, request): request for request in requests
                }
                for future in as_completed(futures):
                    row = future.result()
                    batch_rows.append(row)
                    filename = (
                        f"{row['time_index']:04d}-"
                        f"{_safe_identifier(row['run_1'])}--"
                        f"{_safe_identifier(row['run_2'])}-"
                        f"{_safe_identifier(row['domain_id'])}-"
                        f"{_safe_identifier(row['mass_component'])}.json"
                    )
                    path = chunk_intermediate_root / filename
                    _atomic_json(path, row)
                    chunk_references.append(
                        {
                            "order": int(row["order"]),
                            "path": path.relative_to(chunk_staging).as_posix(),
                            "sha256": sha256_file(path),
                            "size_bytes": path.stat().st_size,
                        }
                    )
            batch_rows.sort(key=lambda row: int(row["order"]))
            chunk_references.sort(key=lambda row: int(row["order"]))
            context.commit_chunk(
                chunk_id,
                chunk_staging,
                state={
                    "time_index": time_index,
                    "time_hours": time_hours,
                    "rows": batch_rows,
                    "intermediate_references": chunk_references,
                },
            )
            context.restore_chunk(chunk_id, context.staging)
            completed_rows.extend(batch_rows)
            intermediate_references.extend(chunk_references)
            completed_count = len(completed_rows)
            context.checkpoint(
                {
                    "stage": "time_complete",
                    "time_index": time_index,
                    "time_hours": time_hours,
                    "completed_comparisons": completed_count,
                    "comparison_count": total_requests,
                }
            )
            context.raise_if_interrupted()

        completed_rows.sort(key=lambda row: int(row["order"]))
        intermediate_references.sort(key=lambda row: int(row["order"]))
        if len(completed_rows) != total_requests:
            raise RuntimeError(
                "Wasserstein output coverage does not match the immutable task contract"
            )
        csv_path = context.staging / "wasserstein_distances.csv"
        xlsx_path = context.staging / "wasserstein_distances.xlsx"
        _write_csv(csv_path, completed_rows)
        _write_xlsx(xlsx_path, completed_rows)

        intermediate_index_path = intermediate_root / "index.json"
        _atomic_json(
            intermediate_index_path,
            {
                "schema": 1,
                "format": WASSERSTEIN_INTERMEDIATE_INDEX_FORMAT,
                "task_id": str(task["task_id"]),
                "comparison_count": total_requests,
                "rows": intermediate_references,
            },
        )
        provenance_path = context.staging / "provenance.json"
        provenance = {
            "schema": 1,
            "format": WASSERSTEIN_PROVENANCE_FORMAT,
            "generated_utc": utc_now(),
            "adapter": {
                "key": self.descriptor.key,
                "version": self.descriptor.version,
            },
            "task_id": str(task["task_id"]),
            "task_sha256": str(task.get("content_sha256") or ""),
            "plan_id": str(plan.get("plan_id") or ""),
            "plan_path": str(plan_path),
            "plan_file_sha256": plan_file_sha256,
            "analysis_lineage": deepcopy(dict(plan.get("upstream") or {})),
            "mass_result_manifest_path": str(mass_manifest_path),
            "mass_result_manifest_sha256": sha256_file(mass_manifest_path),
            "input_index_digest": _input_digest(runs),
            "backend": deepcopy(dict(task.get("backend") or {})),
            "hardware": deepcopy(dict(context.inventory)),
            "workers": workers,
            "solver": {
                "method": "bregman",
                "mobility_mode": "cell_based",
                "l1_mode": "raviart_thomas",
                "bregman_update_iterations": [0, 10, 20],
                "num_iter": 20,
                "tol_distance": 1.0e-4,
                "tol_increment": 1.0e-2,
                "linear_solver": "direct",
                "precision": "float64",
            },
            "scientific_contract": {
                "source_fields": "mass_per_pixel_g",
                "solver_fields": "mass_density_g_per_m2",
                "downsampling": "conservative_area_with_exact_mass_rescaling",
                "transport_weight_downsampling": "nearest_neighbor",
                "transport_weight": "inverse_relative_permeability",
                "raw_mass_difference_preserved": True,
                "equal_mass_scaling_scope": "transport_solver_only",
                "dimensionless_resize_factor_used": False,
                "normalization_reference_scope": "full_run_injection_protocol",
                "time_selection": deepcopy(resolved_selection),
            },
        }
        _atomic_json(provenance_path, provenance)
        artifacts = [
            _artifact(context.staging, csv_path, "wasserstein_csv"),
            _artifact(context.staging, xlsx_path, "wasserstein_xlsx"),
            _artifact(
                context.staging,
                intermediate_index_path,
                "wasserstein_intermediate_index",
            ),
            _artifact(context.staging, provenance_path, "provenance_json"),
        ]
        result_manifest = {
            "schema": 1,
            "format": WASSERSTEIN_RESULT_FORMAT,
            "wasserstein_plan_id": str(plan.get("plan_id") or ""),
            "wasserstein_task_contract_sha256": str(
                contract.get("content_sha256") or ""
            ),
            "task_id": str(task["task_id"]),
            "analysis_lineage": deepcopy(dict(plan.get("upstream") or {})),
            "adapter": {
                "key": self.descriptor.key,
                "version": self.descriptor.version,
            },
            "complete": True,
            "comparison_count": total_requests,
            "all_solver_rows_converged": all(
                bool(row["solver_converged"]) or row["status"] == "both_zero_mass"
                for row in completed_rows
            ),
            "status_counts": {
                status: sum(row["status"] == status for row in completed_rows)
                for status in sorted({str(row["status"]) for row in completed_rows})
            },
            "solver_warning_row_count": sum(
                int(row["solver_warning_count"]) > 0 for row in completed_rows
            ),
            "artifacts": artifacts,
        }
        _atomic_json(
            context.staging / "wasserstein_result.json",
            result_manifest,
        )
        _atomic_json(context.staging / "task.json", task)
        _atomic_json(context.staging / "plan.json", plan)
        context.checkpoint(
            {
                "stage": "wasserstein_ready",
                "comparison_count": total_requests,
                "all_solver_rows_converged": result_manifest[
                    "all_solver_rows_converged"
                ],
            }
        )
        return {
            "result_manifest": "wasserstein_result.json",
            "comparison_count": total_requests,
            "run_count": len(runs),
            "time_count": len(selected_time_indices),
            "all_solver_rows_converged": result_manifest["all_solver_rows_converged"],
            "scientific_results_created": True,
        }
