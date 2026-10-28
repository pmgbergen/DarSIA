"""Materialize generic FluidFlower mass fields with CPU float64 semantics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import darsia
from darsia.fluidflower.execution import (
    AdapterDescriptor,
    TaskExecutionContext,
    load_task_plan,
    verify_file_reference,
)
from darsia.fluidflower.mass_acceleration import (
    ACCELERATED_MASS_BACKENDS,
    NONREFERENCE_MASS_BACKENDS,
    MassFrameResult,
    MassKernelConfiguration,
    compare_mass_frame_results,
    create_mass_compute_session,
)
from darsia.fluidflower.mass_analysis import (
    MASS_ANALYSIS_REVISION_FORMAT,
    MASS_ANALYSIS_TASK_CONTRACT_FORMAT,
)
from darsia.fluidflower.mass_artifacts import (
    write_field_frame,
    write_field_index,
    write_preview_index,
)
from darsia.fluidflower.mass_inputs import (
    MASS_INPUT_MANIFEST_FORMAT,
    MASS_PARAMETER_ARTIFACT_FORMAT,
    load_mass_input_products,
    validate_mass_parameter_artifact,
)
from darsia.fluidflower.project import FluidFlowerProject, sha256_file, utc_now
from darsia.fluidflower.production_preflight import (
    validate_mass_input_revision_link,
    validate_mass_production_task_photometry,
)
from darsia.fluidflower.quality_budget import (
    numerical_dtype_for_backend,
    resolve_analysis_shape,
    summarize_quality_contract,
)
from darsia.fluidflower.wasserstein import MASS_ANALYSIS_RESULT_FORMAT

MASS_MATERIALIZER_VERSION = "1.2.0"
MASS_PROVENANCE_FORMAT = "darsia-fluidflower-mass-provenance-v1"

INTEGRATED_COLUMNS = (
    "run_id",
    "frame_id",
    "time_hours",
    "injected_mass_g",
    "detected_mass_total_g",
    "detected_mass_aqueous_g",
    "detected_mass_free_gas_g",
    "free_gas_mass_fraction",
    "reference_plume_radius_m",
    "diffusion_dilution_factor",
    "source_time_index",
    "selected_time_index",
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from .mass_artifacts import _atomic_json as _write_metadata
    return _write_metadata(path, payload)


def _safe_identifier(value: str) -> str:
    result = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in str(value)
    ).strip("-")
    return result or "run"


def _resample_analysis_field(
    values: np.ndarray,
    target_shape: tuple[int, int],
    *,
    material: bool = False,
) -> np.ndarray:
    """Downscale an aligned field under the recorded analysis-grid contract."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape == target_shape:
        return array
    interpolation = cv2.INTER_NEAREST if material else cv2.INTER_AREA
    resized = cv2.resize(
        array,
        (int(target_shape[1]), int(target_shape[0])),
        interpolation=interpolation,
    )
    return np.asarray(resized, dtype=np.float64)


def _artifact_reference(staging: Path, path: Path, product: str) -> dict[str, Any]:
    return {
        "product": product,
        "path": path.relative_to(staging).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _field_reference(staging: Path, reference: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(reference["path"])).resolve()
    return {
        **deepcopy(dict(reference)),
        "path": path.relative_to(staging.resolve()).as_posix(),
    }


def _strict_nonnegative(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain one finite 2D field")
    scale = max(1.0, float(np.max(np.abs(array))))
    if float(np.min(array)) < -1.0e-10 * scale:
        raise ValueError(f"{label} contains negative values")
    return np.maximum(array, 0.0)


def _probability(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain one finite 2D field")
    if float(np.min(array)) < -1.0e-6 or float(np.max(array)) > 1.0 + 1.0e-6:
        raise ValueError(f"{label} lies outside [0, 1]")
    return np.clip(array, 0.0, 1.0)


def _transfer(
    values: np.ndarray,
    low: float,
    high: float,
    shape: str,
) -> np.ndarray:
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError(f"Invalid transfer interval [{low}, {high}]")
    scaled = np.clip(
        (np.asarray(values, dtype=np.float64) - low) / (high - low), 0.0, 1.0
    )
    if shape == "linear":
        return scaled
    if shape == "smoothstep":
        return scaled * scaled * (3.0 - 2.0 * scaled)
    raise ValueError(f"Unsupported transfer shape: {shape}")


def _image(values: np.ndarray, width_m: float, height_m: float) -> darsia.Image:
    return darsia.Image(
        np.asarray(values, dtype=np.float64),
        width=width_m,
        height=height_m,
        space_dim=2,
        scalar=True,
    )


def _titration_kwargs(parameters: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {
        "initial_pH": "initial_pH",
        "alkalinity_M": "alkalinity_M",
        "btb_M": "btb_M",
        "pKa1": "pKa1",
        "pKa2": "pKa2",
        "pKaB": "pKaB",
        "co2_sat_M": "co2_sat_M",
        "n_lut": "n_lut",
        "aqueous_blend": "aqueous_blend",
    }
    return {
        target: parameters[source]
        for source, target in aliases.items()
        if source in parameters and parameters[source] is not None
    }


def _mass_kernel_configuration(
    model: str,
    parameters: Mapping[str, Any],
) -> MassKernelConfiguration:
    if model == "simple_flash":
        lut_signal = np.asarray([0.0, 1.0], dtype=np.float64)
        lut_concentration = lut_signal.copy()
    elif model == "titration":
        flash = darsia.TitrationFlash(
            0.0,
            1.0,
            0.0,
            1.0,
            **_titration_kwargs(parameters),
        )
        lut_signal = np.asarray(flash._lut_y, dtype=np.float64)
        physical = np.asarray(flash._lut_caq, dtype=np.float64)
        blend = float(flash.aqueous_blend)
        lut_concentration = (1.0 - blend) * physical + blend * lut_signal
    else:
        raise ValueError(f"Unsupported chemistry model: {model}")
    return MassKernelConfiguration(
        model=model,
        transfer_shape=str(parameters["transfer_shape"]),
        acid_low=float(parameters["acid_low"]),
        acid_high=float(parameters["acid_high"]),
        co2_onset=float(parameters["co2_onset"]),
        co2_high=float(parameters["co2_high"]),
        gas_onset=float(parameters["gas_onset"]),
        gas_high=float(parameters["gas_high"]),
        max_gas_saturation=float(parameters["max_gas_saturation"]),
        lut_signal=lut_signal,
        lut_concentration=lut_concentration,
    )


def _aqueous_concentration(
    acidification: np.ndarray,
    co2_probability: np.ndarray,
    *,
    model: str,
    parameters: Mapping[str, Any],
    width_m: float,
    height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    shape = str(parameters["transfer_shape"])
    acid_fraction = _transfer(
        acidification,
        float(parameters["acid_low"]),
        float(parameters["acid_high"]),
        shape,
    )
    normalized_signal = _image(acid_fraction, width_m, height_m)
    if model == "simple_flash":
        flash: darsia.SimpleFlash = darsia.SimpleFlash(0.0, 1.0, 0.0, 1.0)
    elif model == "titration":
        flash = darsia.TitrationFlash(
            0.0,
            1.0,
            0.0,
            1.0,
            **_titration_kwargs(parameters),
        )
    else:
        raise ValueError(f"Unsupported chemistry model: {model}")
    concentration, _unused_gas = flash(normalized_signal)
    co2_gate = _transfer(
        co2_probability,
        float(parameters["co2_onset"]),
        float(parameters["co2_high"]),
        shape,
    )
    return (
        np.clip(np.asarray(concentration.img, dtype=np.float64) * co2_gate, 0.0, 1.0),
        co2_gate,
    )


def _gas_saturation(
    gas_probability: np.ndarray,
    parameters: Mapping[str, Any],
) -> np.ndarray:
    normalized = _transfer(
        gas_probability,
        float(parameters["gas_onset"]),
        float(parameters["gas_high"]),
        str(parameters["transfer_shape"]),
    )
    return np.clip(
        float(parameters["max_gas_saturation"]) * normalized,
        0.0,
        1.0,
    )


def _mass_analysis(
    shape: tuple[int, int],
    width_m: float,
    height_m: float,
    pressure_bar: float,
    temperature_c: float,
    pressure_gradient_bar_per_m: float,
    temperature_gradient_c_per_m: float,
) -> darsia.CO2MassAnalysis:
    baseline = _image(np.zeros(shape, dtype=np.float64), width_m, height_m)
    return darsia.CO2MassAnalysis(
        baseline=baseline,
        atmospheric_pressure=pressure_bar,
        atmospheric_temperature=temperature_c,
        atmospheric_pressure_gradient=pressure_gradient_bar_per_m,
        atmospheric_temperature_gradient=temperature_gradient_c_per_m,
    )


def _update_mass_state(
    analysis: darsia.CO2MassAnalysis,
    state: Mapping[str, Sequence[float]],
    index: int,
) -> None:
    analysis.update_state(
        atmospheric_pressure=float(state["pressure_bar"][index]),
        atmospheric_temperature=float(state["temperature_c"][index]),
        atmospheric_pressure_gradient=float(
            state["pressure_gradient_bar_per_m"][index]
        ),
        atmospheric_temperature_gradient=float(
            state["temperature_gradient_c_per_m"][index]
        ),
    )


def _cpu_mass_frame(
    *,
    acidification: np.ndarray,
    co2_probability: np.ndarray,
    gas_probability: np.ndarray,
    active: np.ndarray,
    voxel_volume_m3: np.ndarray,
    diffusion_factor: float,
    model: str,
    parameters: Mapping[str, Any],
    width_m: float,
    height_m: float,
    analysis: darsia.CO2MassAnalysis,
) -> MassFrameResult:
    """Evaluate the authoritative DarSIA CPU-float64 frame contract."""

    concentration, co2_gate = _aqueous_concentration(
        acidification,
        co2_probability,
        model=model,
        parameters=parameters,
        width_m=width_m,
        height_m=height_m,
    )
    concentration *= diffusion_factor
    concentration[~active] = 0.0
    saturation = _gas_saturation(gas_probability, parameters)
    saturation[~active] = 0.0
    result = analysis.mass_analysis(
        _image(concentration, width_m, height_m),
        _image(saturation, width_m, height_m),
    )
    mass_total = (
        np.asarray(result.mass.img, dtype=np.float64) * voxel_volume_m3 * 1000.0
    )
    mass_aqueous = (
        np.asarray(result.mass_aq.img, dtype=np.float64) * voxel_volume_m3 * 1000.0
    )
    mass_gas = (
        np.asarray(result.mass_g.img, dtype=np.float64) * voxel_volume_m3 * 1000.0
    )
    return MassFrameResult(
        concentration_aqueous=np.asarray(concentration, dtype=np.float64),
        saturation_free_gas=np.asarray(saturation, dtype=np.float64),
        co2_gate=np.asarray(co2_gate, dtype=np.float64),
        mass_aqueous=mass_aqueous,
        mass_free_gas=mass_gas,
        mass_total=mass_total,
    )


def _parity_source_indices(
    selected_indices: Sequence[int],
    fraction: float,
) -> set[int]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("CPU parity fraction must lie in (0, 1]")
    count = max(
        1, min(len(selected_indices), math.ceil(len(selected_indices) * fraction))
    )
    positions = np.linspace(0, len(selected_indices) - 1, count, dtype=np.int64)
    return {int(selected_indices[int(position)]) for position in positions}


def _transport_weight(permeability: np.ndarray, active: np.ndarray) -> np.ndarray:
    positive = permeability[active]
    if not positive.size:
        raise ValueError("Permeability field has no active positive values")
    minimum = float(np.min(positive))
    maximum = float(np.max(positive))
    clip_value = max(minimum * 1.0e-6, np.finfo(np.float64).tiny)
    safe = np.where(active, np.maximum(permeability, clip_value), clip_value)
    return np.asarray(maximum / safe, dtype=np.float64)


def _diffusion_factors(
    *,
    beta: float,
    injected_mass_g: np.ndarray,
    capacities: np.ndarray,
) -> np.ndarray:
    factors = np.ones_like(injected_mass_g, dtype=np.float64)
    if beta <= 0.0 or not len(factors):
        return factors
    final_injected = float(np.max(injected_mass_g))
    if final_injected <= 0.0:
        return factors
    completed = np.flatnonzero(injected_mass_g >= 0.9999 * final_injected)
    if not completed.size:
        return factors
    injection_end = int(completed[0])
    reference = max(float(capacities[injection_end]), 1.0e-15)
    post = np.arange(len(factors)) > injection_end
    growth = np.maximum(capacities[post] / reference, 1.0)
    factors[post] = np.clip(np.power(growth, -beta), 0.02, 1.0)
    return factors


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(INTEGRATED_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in INTEGRATED_COLUMNS})


def _copy_previews(
    staging: Path,
    run_root: Path,
    run_id: str,
    preview: Any,
    selected_time_indices: Sequence[int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    output = run_root / "preview_images"
    output.mkdir(parents=True, exist_ok=True)
    for source_index in selected_time_indices:
        row = preview.frames[source_index]
        image = preview.load_frame(source_index)
        path = output / f"{source_index:04d}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not write corrected preview for {run_id}")
        rows.append(
            {
                "frame_id": str(row.get("frame_id") or f"frame-{source_index:04d}"),
                "time_hours": float(row["time_hours"]),
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    index_path = output / "index.json"
    reference = write_preview_index(index_path, run_id=run_id, frames=rows)
    return _field_reference(staging, reference)


def _input_digest(manifest: Mapping[str, Any]) -> str:
    payload = {
        "upstream": manifest.get("upstream"),
        "runs": [
            {
                "run_id": row.get("run_id"),
                "phase_candidate_id": row.get("phase_candidate_id"),
                "products": row.get("products"),
                "preview_images": row.get("preview_images"),
                **({"physical_pixel_area": (row.get("geometry") or {})["pixel_area_frames"]}
                   if (row.get("geometry") or {}).get("pixel_area_frames") is not None else {}),
            }
            for row in manifest.get("run_inputs") or []
            if isinstance(row, Mapping)
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _materialize_run(
    *,
    staging: Path,
    manifest_path: Path,
    row: Mapping[str, Any],
    model: str,
    parameters: Mapping[str, Any],
    products_requested: set[str],
    cancelled: Any,
    selected_time_indices: Sequence[int] | None = None,
    quality_profile: Mapping[str, Any] | None = None,
    numerical_dtype: str = "float64",
    backend_name: str = "cpu-float64",
    device_id: str = "",
    device_choice: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = str(row["run_id"])
    run_root = staging / "runs" / _safe_identifier(run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    from darsia._opencv_runtime import scientific_thread, completed_frame, require_ipp_off
    ipp_start = scientific_thread("mass-materialization:setup")
    fields, preview = load_mass_input_products(manifest_path, row)
    dynamic = (
        fields["optical_acidification"],
        fields["phase_co2_probability"],
        fields["phase_free_gas_probability"],
    )
    source_shape = dynamic[0].shape_rc
    if any(field.shape_rc != source_shape for field in fields.values()):
        raise ValueError(f"Mass inputs for {run_id} do not share one grid")
    shape = resolve_analysis_shape(source_shape, quality_profile or {})
    if numerical_dtype not in {"float32", "float64"}:
        raise ValueError(f"Unsupported numerical dtype: {numerical_dtype}")
    expected_dtype = numerical_dtype_for_backend(backend_name)
    if numerical_dtype != expected_dtype:
        raise ValueError(
            f"Backend {backend_name} requires {expected_dtype}, not {numerical_dtype}"
        )
    nonreference = backend_name in NONREFERENCE_MASS_BACKENDS
    if backend_name != "cpu-float64" and not nonreference:
        raise ValueError(f"Unsupported mass-analysis backend: {backend_name}")
    output_dtype = np.float32 if numerical_dtype == "float32" else np.float64
    frame_ids = [
        tuple(str(field.frames[index]["frame_id"]) for field in dynamic)
        for index in range(len(dynamic[0].frames))
    ]
    if any(len(set(ids)) != 1 for ids in frame_ids):
        raise ValueError(f"Dynamic field identities differ for {run_id}")

    depth = _strict_nonnegative(
        _resample_analysis_field(fields["depth"].load_frame(0)[0], shape),
        f"{run_id}/depth",
    )
    porosity = _strict_nonnegative(
        _resample_analysis_field(fields["porosity"].load_frame(0)[0], shape),
        f"{run_id}/porosity",
    )
    permeability = _strict_nonnegative(
        _resample_analysis_field(
            fields["permeability"].load_frame(0)[0], shape, material=True
        ),
        f"{run_id}/permeability",
    )
    if float(np.max(porosity)) > 1.0 + 1.0e-9:
        raise ValueError(f"{run_id}/porosity exceeds one")
    geometry = dict(row["geometry"])
    width_m = float(geometry["width_m"])
    height_m = float(geometry["height_m"])
    pixel_area_m2 = width_m * height_m / float(shape[0] * shape[1])
    fixed_pixel_area_m2 = pixel_area_m2
    variable_area = geometry.get("pixel_area_frames") is not None
    if variable_area and nonreference:
        raise ValueError("Frame-varying rig-plane area currently requires the CPU float64 mass backend")
    active = (depth > 0.0) & (porosity > 0.0) & (permeability > 0.0)
    voxel_volume_m3 = pixel_area_m2 * depth * porosity * active
    transport = _transport_weight(permeability, active)

    source_times = np.asarray(row["target_times_hours"], dtype=np.float64)
    source_injected = np.asarray(row["injected_mass_g"], dtype=np.float64)
    selected_indices = (
        list(range(len(source_times)))
        if selected_time_indices is None
        else [int(value) for value in selected_time_indices]
    )
    if (
        not selected_indices
        or selected_indices != sorted(set(selected_indices))
        or selected_indices[0] < 0
        or selected_indices[-1] >= len(source_times)
    ):
        raise ValueError("Mass time selection is empty, duplicated, or out of range")
    times = source_times[selected_indices]
    injected = source_injected[selected_indices]
    state = dict(row["thermodynamic_state"])
    analysis = _mass_analysis(
        shape,
        width_m,
        height_m,
        float(state["pressure_bar"][0]),
        float(state["temperature_c"][0]),
        float(state["pressure_gradient_bar_per_m"][0]),
        float(state["temperature_gradient_c_per_m"][0]),
    )
    kernel_configuration = _mass_kernel_configuration(model, parameters)

    capacities = np.zeros(len(source_times), dtype=np.float64)
    for index in range(len(source_times)):
        if variable_area:
            from .phase_pixel_area import mass_frame_pixel_area
            capacity_area_m2, _ = mass_frame_pixel_area(
                geometry, index, source_shape, shape, base_dir=manifest_path.parent)
            capacity_volume_m3 = capacity_area_m2 * depth * porosity * active
        else:
            capacity_volume_m3 = voxel_volume_m3
        co2 = _probability(
            _resample_analysis_field(
                fields["phase_co2_probability"].load_frame(index)[0], shape
            ),
            f"{run_id}/phase_co2_probability/{index}",
        )
        _update_mass_state(analysis, state, index)
        co2_gate = _transfer(
            co2,
            float(parameters["co2_onset"]),
            float(parameters["co2_high"]),
            str(parameters["transfer_shape"]),
        )
        capacities[index] = float(
            np.sum(
                capacity_volume_m3
                * np.asarray(analysis.solubility_co2, dtype=np.float64)
                * co2_gate,
                dtype=np.float64,
            )
        )
    diffusion = _diffusion_factors(
        beta=float(parameters.get("diffusion_beta", 0.0)),
        injected_mass_g=source_injected,
        capacities=capacities,
    )

    precision = dict((quality_profile or {}).get("precision") or {})
    parity_fraction = float(precision.get("cpu_float64_parity_fraction") or 0.05)
    absolute_tolerance = float(
        precision.get("absolute_parity_tolerance")
        or (1.0e-10 if numerical_dtype == "float64" else 1.0e-5)
    )
    relative_tolerance = float(
        precision.get("relative_parity_tolerance")
        or (1.0e-9 if numerical_dtype == "float64" else 1.0e-4)
    )
    parity_indices = (
        _parity_source_indices(selected_indices, parity_fraction)
        if nonreference
        else set()
    )
    output_frames: dict[str, list[dict[str, Any]]] = {
        product: []
        for product in (
            "mass_total",
            "mass_aqueous",
            "mass_free_gas",
            "concentration_aqueous",
            "saturation_free_gas",
        )
        if product in products_requested
    }
    integrated_rows: list[dict[str, Any]] = []
    reference_radii: list[float] = []
    parity_rows: list[dict[str, Any]] = []
    session = (
        create_mass_compute_session(
            backend=backend_name,
            active=active,
            voxel_volume_m3=voxel_volume_m3,
            configuration=kernel_configuration,
            device_id=device_id,
        )
        if nonreference
        else None
    )
    backend_execution = (
        session.provenance()
        if session is not None
        else {
            "backend": "cpu-float64",
            "runtime": "darsia-numpy",
            "numpy": np.__version__,
            "device_id": "cpu",
            "dtype": "float64",
            "fused_phase_mass_kernel": False,
        }
    )
    if session is not None and device_choice is not None:
        from .device_choice import verify_session_device

        try:
            verify_session_device(backend_execution, device_choice)
        except RuntimeError:
            session.close()
            raise
        backend_execution["device_choice"] = deepcopy(dict(device_choice))
    try:
        for output_index, source_index in enumerate(selected_indices):
            ipp_before = scientific_thread("mass-materialization:frame-before")
            time_hours = source_times[source_index]
            if cancelled():
                raise RuntimeError("Mass analysis was cancelled")
            area_binding = None
            if variable_area:
                from .phase_pixel_area import mass_frame_pixel_area
                pixel_area_m2, area_binding = mass_frame_pixel_area(
                    geometry, source_index, source_shape, shape, base_dir=manifest_path.parent)
                voxel_volume_m3 = pixel_area_m2 * depth * porosity * active
            acidification = np.asarray(
                _resample_analysis_field(
                    fields["optical_acidification"].load_frame(source_index)[0], shape
                ),
                dtype=np.float64,
            )
            if not np.all(np.isfinite(acidification)):
                raise ValueError(
                    f"{run_id}/optical_acidification/{source_index} is non-finite"
                )
            co2_probability = _probability(
                _resample_analysis_field(
                    fields["phase_co2_probability"].load_frame(source_index)[0], shape
                ),
                f"{run_id}/phase_co2_probability/{source_index}",
            )
            gas_probability = _probability(
                _resample_analysis_field(
                    fields["phase_free_gas_probability"].load_frame(source_index)[0],
                    shape,
                ),
                f"{run_id}/phase_free_gas_probability/{source_index}",
            )
            if np.any(gas_probability > co2_probability + 1.0e-5):
                raise ValueError(
                    f"{run_id}/frame-{source_index} has free-gas probability outside CO2 support"
                )
            _update_mass_state(analysis, state, source_index)
            if session is None:
                frame_result = _cpu_mass_frame(
                    acidification=acidification,
                    co2_probability=co2_probability,
                    gas_probability=gas_probability,
                    active=active,
                    voxel_volume_m3=voxel_volume_m3,
                    diffusion_factor=float(diffusion[source_index]),
                    model=model,
                    parameters=parameters,
                    width_m=width_m,
                    height_m=height_m,
                    analysis=analysis,
                )
            else:
                frame_result = session.compute_frame(
                    acidification=acidification,
                    co2_probability=co2_probability,
                    gas_probability=gas_probability,
                    solubility_co2=np.asarray(
                        analysis.solubility_co2, dtype=np.float64
                    ),
                    density_gaseous_co2=np.asarray(
                        analysis.density_gaseous_co2, dtype=np.float64
                    ),
                    diffusion_factor=float(diffusion[source_index]),
                )
                if source_index in parity_indices:
                    reference = _cpu_mass_frame(
                        acidification=acidification,
                        co2_probability=co2_probability,
                        gas_probability=gas_probability,
                        active=active,
                        voxel_volume_m3=voxel_volume_m3,
                        diffusion_factor=float(diffusion[source_index]),
                        model=model,
                        parameters=parameters,
                        width_m=width_m,
                        height_m=height_m,
                        analysis=analysis,
                    )
                    parity = compare_mass_frame_results(
                        frame_result,
                        reference,
                        absolute_tolerance=absolute_tolerance,
                        relative_tolerance=relative_tolerance,
                    )
                    parity.update(
                        {
                            "candidate_backend": backend_name,
                            "run_id": run_id,
                            "frame_id": frame_ids[source_index][0],
                            "source_time_index": source_index,
                        }
                    )
                    parity_rows.append(parity)
                    if not parity["passed"]:
                        failed_fields = [
                            name
                            for name, field in parity["fields"].items()
                            if not field.get("passed")
                        ]
                        raise RuntimeError(
                            f"{backend_name} failed CPU-float64 parity for {run_id}/"
                            f"frame-{source_index}: {', '.join(failed_fields)}"
                        )
            values_by_product = {
                "mass_total": frame_result.mass_total,
                "mass_aqueous": frame_result.mass_aqueous,
                "mass_free_gas": frame_result.mass_free_gas,
                "concentration_aqueous": frame_result.concentration_aqueous,
                "saturation_free_gas": frame_result.saturation_free_gas,
            }
            ipp_frame = completed_frame(ipp_before, "mass-materialization:frame-after")
            frame_id = frame_ids[source_index][0]
            for product, frames in output_frames.items():
                product_root = run_root / product
                frame_path = product_root / f"{source_index:04d}.npz"
                frames.append(
                    write_field_frame(
                        frame_path,
                        np.asarray(values_by_product[product], dtype=output_dtype),
                        run_id=run_id,
                        product=product,
                        frame_id=frame_id,
                        time_hours=float(time_hours),
                    )
                )
            projected_area_m2 = float(
                np.sum(frame_result.co2_gate * active * pixel_area_m2, dtype=np.float64)
                if variable_area else
                np.sum(frame_result.co2_gate * active, dtype=np.float64) * pixel_area_m2
            )
            radius = math.sqrt(max(projected_area_m2, 0.0) / math.pi)
            reference_radii.append(radius)
            total_g = float(np.sum(frame_result.mass_total, dtype=np.float64))
            aqueous_g = float(np.sum(frame_result.mass_aqueous, dtype=np.float64))
            gas_g = float(np.sum(frame_result.mass_free_gas, dtype=np.float64))
            integrated_rows.append(
                {
                    "run_id": run_id,
                    "frame_id": frame_id,
                    "time_hours": float(time_hours),
                    "injected_mass_g": float(source_injected[source_index]),
                    "detected_mass_total_g": total_g,
                    "detected_mass_aqueous_g": aqueous_g,
                    "detected_mass_free_gas_g": gas_g,
                    "free_gas_mass_fraction": (
                        gas_g / total_g if total_g > 0.0 else 0.0
                    ),
                    "reference_plume_radius_m": radius,
                    "diffusion_dilution_factor": float(diffusion[source_index]),
                    "source_time_index": source_index,
                    "selected_time_index": output_index,
                    "opencv_ipp": ipp_frame,
                    **({"pixel_area": area_binding} if area_binding is not None else {}),
                    **({"active_pore_volume_m3": float(np.sum(voxel_volume_m3, dtype=np.float64))}
                       if variable_area else {}),
                }
            )
    finally:
        if session is not None:
            session.close()

    grid = {
        "coordinate_system": "corrected_registered_rig_grid",
        "shape_rc": list(shape),
        "source_shape_rc": list(source_shape),
        "resolution_policy": str(
            ((quality_profile or {}).get("numerical_resolution") or {}).get(
                "policy", "native_registered_grid"
            )
        ),
        "width_m": width_m,
        "height_m": height_m,
        "cell_width_m": width_m / shape[1],
        "cell_height_m": height_m / shape[0],
        "pixel_area_m2": None if variable_area else pixel_area_m2,
        **({"pixel_area_m2_fixed_grid": fixed_pixel_area_m2,
            "pixel_area_policy": "source-bound per-frame rig-plane Jacobian",
            "pixel_area_frames": geometry["pixel_area_frames"]} if variable_area else {}),
    }
    product_references: dict[str, dict[str, Any]] = {}
    field_metadata = {
        "mass_total": ("co2_mass_per_pixel", "g pixel-1"),
        "mass_aqueous": ("aqueous_co2_mass_per_pixel", "g pixel-1"),
        "mass_free_gas": ("free_gas_co2_mass_per_pixel", "g pixel-1"),
        "concentration_aqueous": ("normalized_aqueous_co2_concentration", "1"),
        "saturation_free_gas": ("free_gas_saturation", "1"),
    }
    for product, frames in output_frames.items():
        index_path = run_root / product / "index.json"
        quantity, unit = field_metadata[product]
        reference = write_field_index(
            index_path,
            run_id=run_id,
            product=product,
            quantity=quantity,
            unit=unit,
            frames=frames,
            grid=grid,
        )
        product_references[product] = _field_reference(staging, reference)

    transport_root = run_root / "transport_weight"
    transport_frame = write_field_frame(
        transport_root / "0000.npz",
        np.asarray(transport, dtype=output_dtype),
        run_id=run_id,
        product="transport_weight",
        frame_id="static-geometry",
        time_hours=float(times[0]),
    )
    transport_reference = write_field_index(
        transport_root / "index.json",
        run_id=run_id,
        product="transport_weight",
        quantity="inverse_relative_permeability_transport_cost",
        unit="1",
        frames=[transport_frame],
        grid=grid,
    )
    product_references["transport_weight"] = _field_reference(
        staging, transport_reference
    )

    integrated_path = run_root / "integrated_mass.csv"
    _write_csv(integrated_path, integrated_rows)
    product_references["integrated_mass_csv"] = _artifact_reference(
        staging, integrated_path, "integrated_mass_csv"
    )
    if "preview_images" in products_requested:
        if preview is None:
            raise ValueError(f"{run_id} has no corrected preview images")
        product_references["preview_images"] = _copy_previews(
            staging,
            run_root,
            run_id,
            preview,
            selected_indices,
        )
    if "diagnostic_injected_mass_projection" in products_requested:
        diagnostic_path = run_root / "injected_mass_projection.json"
        _atomic_json(
            diagnostic_path,
            {
                "schema": 1,
                "run_id": run_id,
                "policy": "diagnostic_only_no_field_rescaling",
                "rows": [
                    {
                        "frame_id": record["frame_id"],
                        "time_hours": record["time_hours"],
                        "injected_mass_g": record["injected_mass_g"],
                        "detected_mass_total_g": record["detected_mass_total_g"],
                        "projection_factor": (
                            record["injected_mass_g"] / record["detected_mass_total_g"]
                            if record["detected_mass_total_g"] > 0.0
                            else None
                        ),
                    }
                    for record in integrated_rows
                ],
            },
        )
        product_references["diagnostic_injected_mass_projection"] = _artifact_reference(
            staging,
            diagnostic_path,
            "diagnostic_injected_mass_projection",
        )

    return {
        "run_id": run_id,
        "phase_candidate_id": str(row["phase_candidate_id"]),
        "complete": True,
        "target_times_hours": times.tolist(),
        "injected_mass_g": injected.tolist(),
        "source_time_indices": selected_indices,
        "reference_plume_radius_m": reference_radii,
        "domains": ["whole_active_domain"],
        "geometry": {
            "width_m": width_m,
            "height_m": height_m,
            "shape_rc": list(shape),
            "source_shape_rc": list(source_shape),
            "resolution_policy": str(
                ((quality_profile or {}).get("numerical_resolution") or {}).get(
                    "policy", "native_registered_grid"
                )
            ),
            "numerical_dtype": numerical_dtype,
            "native_cell_size_m": None if variable_area else math.sqrt(pixel_area_m2),
            "cell_width_m": width_m / shape[1],
            "cell_height_m": height_m / shape[0],
            "active_pixel_count": int(np.count_nonzero(active)),
            "active_pore_volume_m3": None if variable_area else float(np.sum(voxel_volume_m3)),
        },
        "chemistry_model": model,
        "parameters": deepcopy(dict(parameters)),
        "backend_execution": backend_execution,
        "opencv_ipp": dict(start=ipp_start, completed=require_ipp_off("mass-materialization:completed")),
        "cpu_float64_parity": {
            "required": nonreference,
            "passed": all(row["passed"] for row in parity_rows),
            "sample_fraction": parity_fraction if nonreference else 1.0,
            "sample_count": len(parity_rows),
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "frames": parity_rows,
        },
        "products": product_references,
        "integrated_mass": integrated_rows,
    }


def materialize_mass_run(
    *,
    staging: Path,
    manifest_path: Path,
    row: Mapping[str, Any],
    model: str,
    parameters: Mapping[str, Any],
    products_requested: set[str],
    cancelled: Any = lambda: False,
    selected_time_indices: Sequence[int] | None = None,
    quality_profile: Mapping[str, Any] | None = None,
    numerical_dtype: str = "float64",
    backend_name: str = "cpu-float64",
    device_id: str = "",
    device_choice: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Public single-run entry point used by pilots and trusted adapters."""

    return _materialize_run(
        staging=staging,
        manifest_path=manifest_path,
        row=row,
        model=model,
        parameters=parameters,
        products_requested=products_requested,
        cancelled=cancelled,
        selected_time_indices=selected_time_indices,
        quality_profile=quality_profile,
        numerical_dtype=numerical_dtype,
        backend_name=backend_name,
        device_id=device_id,
        device_choice=device_choice,
    )


def analysis_shape_for_run(
    manifest_path: Path,
    row: Mapping[str, Any],
    quality_profile: Mapping[str, Any],
) -> tuple[int, int]:
    """Return the analysis grid of one mass-input run from its field indexes only."""

    fields, _ = load_mass_input_products(manifest_path, row)
    return resolve_analysis_shape(
        fields["optical_acidification"].shape_rc, quality_profile
    )


class MassAnalysisAdapter:
    """Convert reviewed visible phases to physical mass with DarSIA."""

    descriptor = AdapterDescriptor(
        key="mass-analysis-cpu-float64-v1",
        version=MASS_MATERIALIZER_VERSION,
        title="Generic DarSIA mass analysis",
        modules=("mass_analysis",),
        backends=("cpu-float64",),
        description=(
            "Combines corrected optical evidence, reviewed phase probabilities, "
            "DarSIA chemistry, and approved local geometry without forcing mass balance."
        ),
    )

    def execute(
        self,
        task: Mapping[str, Any],
        context: TaskExecutionContext,
    ) -> Mapping[str, Any]:
        if task.get("module") != "mass_analysis":
            raise ValueError("The mass adapter only accepts mass-analysis tasks")
        backend_name = str((task.get("backend") or {}).get("name") or "")
        if backend_name not in self.descriptor.backends:
            raise ValueError(
                f"Adapter {self.descriptor.key} does not support {backend_name}"
            )
        plan, plan_path, plan_file_sha256 = load_task_plan(task)
        if plan.get("format") != MASS_ANALYSIS_REVISION_FORMAT:
            raise ValueError("The task does not reference a mass-analysis revision")
        contract = dict(plan.get("task_contract") or {})
        if contract.get("format") != MASS_ANALYSIS_TASK_CONTRACT_FORMAT:
            raise ValueError("Unsupported mass-analysis task contract")
        if not bool((plan.get("validation") or {}).get("execution_ready")):
            raise RuntimeError("The saved mass-analysis plan is not execution-ready")
        production_photometry = validate_mass_production_task_photometry(
            task, plan, plan_file_sha256=plan_file_sha256
        )

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

        input_reference = dict(contract.get("mass_inputs") or {})
        input_path = verify_file_reference(
            str(input_reference.get("path") or ""),
            str(input_reference.get("sha256") or ""),
            label="Prepared mass-input manifest",
        )
        input_manifest = json.loads(input_path.read_text(encoding="utf-8-sig"))
        if (
            not isinstance(input_manifest, dict)
            or input_manifest.get("format") != MASS_INPUT_MANIFEST_FORMAT
        ):
            raise ValueError("Unsupported prepared mass-input manifest")
        production_photometry["mass_input_revision_link"] = (
            validate_mass_input_revision_link(plan, input_manifest)
        )

        calibration_reference = dict(contract.get("calibration_result") or {})
        calibration_path = verify_file_reference(
            str(calibration_reference.get("path") or ""),
            str(calibration_reference.get("sha256") or ""),
            label="Calibration-result manifest",
        )
        calibration_result = json.loads(
            calibration_path.read_text(encoding="utf-8-sig")
        )
        if not isinstance(calibration_result, dict):
            raise TypeError("Calibration-result manifest must contain one object")
        parameter_reference = dict(
            calibration_reference.get("parameter_artifact") or {}
        )
        parameter_path = verify_file_reference(
            str(parameter_reference.get("path") or ""),
            str(parameter_reference.get("sha256") or ""),
            label="Calibrated mass parameters",
        )
        parameter_payload = json.loads(parameter_path.read_text(encoding="utf-8-sig"))
        if (
            not isinstance(parameter_payload, dict)
            or parameter_payload.get("format") != MASS_PARAMETER_ARTIFACT_FORMAT
        ):
            raise ValueError("Unsupported calibrated mass-parameter artifact")

        run_bindings = [dict(row) for row in contract.get("run_bindings") or []]
        run_ids = [str(row["run_id"]) for row in run_bindings]
        model = str(calibration_result.get("selected_model") or "")
        parameter_report = validate_mass_parameter_artifact(
            parameter_payload,
            run_ids=run_ids,
            selected_model=model,
        )
        if not parameter_report["valid"]:
            messages = [
                str(issue.get("message") or "")
                for issue in parameter_report.get("issues") or []
                if issue.get("severity") == "error"
            ]
            raise ValueError(
                "Invalid calibrated mass parameters: " + " | ".join(messages)
            )
        parameters_by_run = dict(parameter_report["resolved_parameters"])

        rows_by_id = {
            str(row.get("run_id") or ""): dict(row)
            for row in input_manifest.get("run_inputs") or []
            if isinstance(row, Mapping)
        }
        if sorted(rows_by_id) != sorted(run_ids):
            raise ValueError("Prepared mass-input runs differ from the task contract")
        products_requested = {
            str(value)
            for value in (contract.get("exports") or {}).get("products") or []
        }
        quality_profile = dict(contract.get("quality_profile") or {})
        numerical_dtype = str(
            (contract.get("exports") or {}).get("numerical_dtype") or "float64"
        )
        planned_backend = str(
            (contract.get("execution_profile") or {}).get("primary_backend") or ""
        )
        if planned_backend != backend_name:
            raise ValueError(
                f"Saved mass plan requires {planned_backend}; task selected {backend_name}"
            )
        expected_dtype = numerical_dtype_for_backend(backend_name)
        if numerical_dtype != expected_dtype:
            raise ValueError(
                f"Saved mass plan uses {numerical_dtype}; {backend_name} requires "
                f"{expected_dtype}"
            )
        resolved_selection = dict(contract.get("resolved_time_selection") or {})
        selected_time_indices = [
            int(value)
            for value in resolved_selection.get("selected_time_indices") or []
        ]
        if not selected_time_indices:
            raise ValueError("Mass task contract has no resolved time selection")
        resource_key = (
            "cpu_workers" if backend_name.startswith("cpu-") else "accelerator_workers"
        )
        workers = min(
            len(run_ids),
            24,
            max(
                1,
                int(
                    (task.get("resources") or {}).get(resource_key)
                    or (contract.get("execution_profile") or {}).get("workers")
                    or 1
                ),
            ),
        )
        device_id = str(
            ((task.get("adapter") or {}).get("settings") or {}).get("device_id")
            or (task.get("resources") or {}).get("device_id")
            or ""
        )
        context.checkpoint(
            {
                "stage": "inputs_verified",
                "run_count": len(run_ids),
                "worker_count": workers,
                "backend": backend_name,
                "device_id": device_id,
            }
        )
        run_results: list[dict[str, Any]] = []
        missing_run_ids: list[str] = []
        for run_id in run_ids:
            chunk_id = f"mass-run-{_safe_identifier(run_id)}"
            committed = context.committed_chunk(chunk_id)
            state = dict(committed.get("state") or {})
            run_result = state.get("run_result")
            if committed and isinstance(run_result, Mapping):
                context.restore_chunk(chunk_id, context.staging)
                run_results.append(dict(run_result))
            else:
                missing_run_ids.append(run_id)
        device_choice = None
        if backend_name.startswith(("cuda-", "opencl-")) and run_ids:
            # One device per job, chosen before its first run and kept on resume.
            from .device_choice import resolve_job_device, task_device_request

            requested, source = task_device_request(task)
            device_choice = resolve_job_device(
                backend=backend_name,
                shape=analysis_shape_for_run(
                    input_path, rows_by_id[run_ids[0]], quality_profile
                ),
                workers=workers,
                explicit=requested,
                explicit_source=source,
                committed=[
                    dict(row.get("backend_execution") or {}) for row in run_results
                ],
            )
            device_id = str(device_choice["device_id"])
            context.checkpoint(
                {
                    "stage": "device_chosen",
                    "backend": backend_name,
                    "device_id": device_id,
                    "basis": device_choice["basis"],
                }
            )

        def materialize_chunk(run_id: str) -> tuple[str, dict[str, Any]]:
            chunk_id = f"mass-run-{_safe_identifier(run_id)}"
            chunk_staging = context.prepare_chunk(chunk_id)
            run_result = _materialize_run(
                staging=chunk_staging,
                manifest_path=input_path,
                row=rows_by_id[run_id],
                model=model,
                parameters=parameters_by_run[run_id],
                products_requested=products_requested,
                cancelled=context.cancellation_requested,
                selected_time_indices=selected_time_indices,
                quality_profile=quality_profile,
                numerical_dtype=numerical_dtype,
                backend_name=backend_name,
                device_id=device_id,
                device_choice=device_choice,
            )
            context.commit_chunk(
                chunk_id,
                chunk_staging,
                state={"run_id": run_id, "run_result": run_result},
            )
            return chunk_id, run_result

        for offset in range(0, len(missing_run_ids), workers):
            batch = missing_run_ids[offset : offset + workers]
            with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as pool:
                futures = {
                    pool.submit(materialize_chunk, run_id): run_id for run_id in batch
                }
                for future in as_completed(futures):
                    run_id = futures[future]
                    chunk_id, run_result = future.result()
                    context.restore_chunk(chunk_id, context.staging)
                    run_results.append(run_result)
                    context.checkpoint(
                        {
                            "stage": "run_complete",
                            "run_id": run_id,
                            "completed_runs": len(run_results),
                            "run_count": len(run_ids),
                        }
                    )
            context.raise_if_interrupted()
        run_results.sort(key=lambda row: run_ids.index(str(row["run_id"])))
        parity_reports = [
            deepcopy(dict(row.get("cpu_float64_parity") or {})) for row in run_results
        ]
        if backend_name != "cpu-float64" and (
            not parity_reports
            or not all(
                report.get("passed") is True
                and int(report.get("sample_count") or 0) > 0
                for report in parity_reports
            )
        ):
            raise RuntimeError(
                f"{backend_name} results cannot be published without CPU-float64 parity"
            )

        provenance_path = context.staging / "provenance.json"
        provenance = {
            "schema": 1,
            "format": MASS_PROVENANCE_FORMAT,
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
            "production_photometry": production_photometry,
            "mass_input_manifest_path": str(input_path),
            "mass_input_manifest_sha256": sha256_file(input_path),
            "mass_input_transitive_digest": _input_digest(input_manifest),
            "calibration_result_path": str(calibration_path),
            "calibration_result_sha256": sha256_file(calibration_path),
            "parameter_artifact_path": str(parameter_path),
            "parameter_artifact_sha256": sha256_file(parameter_path),
            "backend": deepcopy(dict(task.get("backend") or {})),
            "backend_execution": [
                deepcopy(dict(row.get("backend_execution") or {}))
                for row in run_results
            ],
            "hardware": deepcopy(dict(context.inventory)),
            "workers": workers,
            "cpu_float64_parity": {
                "required": backend_name != "cpu-float64",
                "passed": all(
                    report.get("passed") is True for report in parity_reports
                ),
                "sample_count": sum(
                    int(report.get("sample_count") or 0) for report in parity_reports
                ),
                "runs": parity_reports,
            },
            "scientific_contract": {
                "phase_model_role": "visible_2d_support_and_probability_only",
                "continuous_signal": "radiometrically_corrected_optical_acidification",
                "chemistry_engine": f"darsia_{model}",
                "gas_density_and_solubility": "darsia_co2_mass_analysis",
                "phase_partition": "aqueous_mass_uses_one_minus_gas_saturation",
                "physical_volume": "pixel_area_times_depth_times_porosity",
                "raw_detected_mass_authoritative": True,
                "injected_mass_projection": "diagnostic_only",
                "mass_conservation_forced": False,
                "quality_profile": deepcopy(quality_profile),
                "quality_summary": summarize_quality_contract(
                    quality_profile,
                    backend=backend_name,
                ),
                "numerical_dtype": numerical_dtype,
                "time_selection": deepcopy(resolved_selection),
                "visible_slice_caveat": (
                    "The corrected camera plane is treated as representative across "
                    "the measured local depth; out-of-plane topology remains uncertain."
                ),
            },
        }
        if device_choice is not None:
            provenance["device_choice"] = deepcopy(device_choice)
        _atomic_json(provenance_path, provenance)
        result_manifest = {
            "schema": 1,
            "format": MASS_ANALYSIS_RESULT_FORMAT,
            "mass_analysis_plan_id": str(plan.get("plan_id") or ""),
            "mass_task_contract_sha256": str(contract.get("content_sha256") or ""),
            "task_id": str(task["task_id"]),
            "analysis_lineage": deepcopy(dict(plan.get("upstream") or {})),
            "adapter": {
                "key": self.descriptor.key,
                "version": self.descriptor.version,
            },
            "complete": True,
            "raw_mass_is_authoritative": True,
            "mass_conservation_forced": False,
            "chemistry_model": model,
            "quality_contract": summarize_quality_contract(
                quality_profile,
                backend=backend_name,
            ),
            "run_results": run_results,
            "artifacts": [
                _artifact_reference(context.staging, provenance_path, "provenance_json")
            ],
        }
        _atomic_json(context.staging / "mass_analysis_result.json", result_manifest)
        _atomic_json(context.staging / "task.json", task)
        _atomic_json(context.staging / "plan.json", plan)
        context.checkpoint(
            {
                "stage": "mass_analysis_ready",
                "run_count": len(run_results),
            }
        )
        return {
            "result_manifest": "mass_analysis_result.json",
            "run_count": len(run_results),
            "time_count": len(run_results[0]["target_times_hours"]),
            "scientific_results_created": True,
        }


class AcceleratedMassAnalysisAdapter(MassAnalysisAdapter):
    """Execute the same mass contract on CUDA, OpenCL, or Metal."""

    descriptor = AdapterDescriptor(
        key="mass-analysis-accelerated-v1",
        version=MASS_MATERIALIZER_VERSION,
        title="Accelerated DarSIA mass analysis",
        modules=("mass_analysis",),
        backends=ACCELERATED_MASS_BACKENDS,
        description=(
            "Runs phase transfer, chemistry LUT interpolation, phase partition, and "
            "mass algebra on CUDA, OpenCL, or Metal. Deterministic frames must pass "
            "the authoritative CPU-float64 parity check before publication."
        ),
    )


class CpuFloat32MassAnalysisAdapter(MassAnalysisAdapter):
    """Execute mass analysis with CPU float32 and CPU-float64 parity samples."""

    descriptor = AdapterDescriptor(
        key="mass-analysis-cpu-float32-v1",
        version=MASS_MATERIALIZER_VERSION,
        title="CPU float32 DarSIA mass analysis",
        modules=("mass_analysis",),
        backends=("cpu-float32",),
        description=(
            "Runs the phase and mass algebra with NumPy float32 for memory- or "
            "throughput-constrained CPU hosts. Deterministic frames must pass the "
            "authoritative CPU-float64 parity check before publication."
        ),
    )


__all__ = [
    "AcceleratedMassAnalysisAdapter",
    "CpuFloat32MassAnalysisAdapter",
    "MASS_MATERIALIZER_VERSION",
    "MASS_PROVENANCE_FORMAT",
    "MassAnalysisAdapter",
]
