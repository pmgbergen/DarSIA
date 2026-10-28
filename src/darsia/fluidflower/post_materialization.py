"""Generic post-analysis materializer for immutable FluidFlower mass products."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from matplotlib import pyplot as plt

from darsia.fluidflower.execution import (
    AdapterDescriptor,
    TaskExecutionContext,
    load_task_plan,
    verify_file_reference,
)
from darsia.fluidflower.mass_artifacts import (
    IndexedFieldProduct,
    PreviewProduct,
    load_field_index,
    load_preview_index,
)
from darsia.fluidflower.post_analysis import (
    POST_ANALYSIS_RESULT_FORMAT,
    POST_ANALYSIS_REVISION_FORMAT,
    POST_ANALYSIS_TASK_CONTRACT_FORMAT,
    expected_post_analysis_artifacts,
)
from darsia.fluidflower.project import FluidFlowerProject, sha256_file, utc_now

plt.switch_backend("Agg")


POST_MATERIALIZER_VERSION = "1.0.0"

PRODUCT_FILENAMES = {
    "phase_map_movie_corrected_background": "phase_map_corrected.mp4",
    "phase_map_movie_black_background": "phase_map_black.mp4",
    "phase_contour_movie": "phase_contours.mp4",
    "phase_contour_contact_sheet": "phase_contours_contact_sheet.png",
    "mass_curve_full_domain": "mass_curve_full_domain.png",
    "mass_curve_analysis_domains": "mass_curve_analysis_domains.png",
    "cross_run_mass_grid": "cross_run_mass_grid.png",
    "cross_run_phase_grid_movie": "cross_run_phase_grid.mp4",
    "integrated_mass_csv": "integrated_mass.csv",
    "per_frame_diagnostics_csv": "per_frame_diagnostics.csv",
    "provenance_json": "provenance.json",
}

PRODUCT_MEDIA_TYPES = {
    "phase_map_movie_corrected_background": "video/mp4",
    "phase_map_movie_black_background": "video/mp4",
    "phase_contour_movie": "video/mp4",
    "phase_contour_contact_sheet": "image/png",
    "mass_curve_full_domain": "image/png",
    "mass_curve_analysis_domains": "image/png",
    "cross_run_mass_grid": "image/png",
    "cross_run_phase_grid_movie": "video/mp4",
    "integrated_mass_csv": "text/csv",
    "per_frame_diagnostics_csv": "text/csv",
    "provenance_json": "application/json",
}

VISUAL_PRODUCTS = {
    "phase_map_movie_corrected_background",
    "phase_map_movie_black_background",
    "phase_contour_movie",
    "phase_contour_contact_sheet",
    "cross_run_phase_grid_movie",
}

MASS_PRODUCTS = ("mass_total", "mass_aqueous", "mass_free_gas")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from .mass_artifacts import _atomic_json as _write_metadata
    return _write_metadata(path, payload)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty post-analysis table: {path}")
    fieldnames = list(rows[0])
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except (FileNotFoundError, OSError):
            pass


def _load_mass_manifest(contract: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    reference = dict(contract.get("mass_result") or {})
    path = verify_file_reference(
        str(reference.get("path") or ""),
        str(reference.get("sha256") or ""),
        label="Mass-analysis result manifest",
    )
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError("Mass-analysis result manifest must be a JSON object")
    if payload.get("format") != "darsia-fluidflower-mass-analysis-result-v1":
        raise ValueError("Unsupported mass-analysis result manifest")
    if payload.get("complete") is not True:
        raise ValueError("Mass-analysis result is not complete")
    return path, payload


def _reference_map(
    contract: Mapping[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    rows.extend((contract.get("mass_result") or {}).get("product_artifacts") or [])
    rows.extend(contract.get("input_artifacts") or [])
    references: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        run_id = str(row.get("run_id") or "")
        product = str(row.get("product") or row.get("component") or "")
        key = (run_id, product)
        if not run_id or not product or key in references:
            raise ValueError(f"Duplicate or incomplete mass artifact reference: {key}")
        references[key] = row
    return references


def _run_manifest_rows(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("run_id") or ""): dict(row)
        for row in payload.get("run_results") or []
        if isinstance(row, Mapping) and row.get("run_id")
    }


@dataclass
class RunInputs:
    run_id: str
    times_hours: list[float]
    injected_mass_g: list[float] | None
    fields: dict[str, IndexedFieldProduct]
    preview: PreviewProduct | None

    @property
    def shape_rc(self) -> tuple[int, int]:
        return self.fields["mass_total"].shape_rc


def _load_run_inputs(
    run_id: str,
    run_manifest: Mapping[str, Any],
    references: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    visual_required: bool,
    corrected_background_required: bool,
) -> RunInputs:
    times = [float(value) for value in run_manifest.get("target_times_hours") or []]
    if not times:
        raise ValueError(f"Mass result has no target times for {run_id}")
    required = [*MASS_PRODUCTS]
    if visual_required:
        required.extend(["concentration_aqueous", "saturation_free_gas"])
    fields: dict[str, IndexedFieldProduct] = {}
    for product in required:
        reference = dict(references.get((run_id, product)) or {})
        if not reference:
            raise ValueError(f"Missing indexed product {run_id}/{product}")
        fields[product] = load_field_index(
            str(reference.get("path") or ""),
            str(reference.get("sha256") or ""),
            run_id=run_id,
            product=product,
            expected_times_hours=times,
        )
    shapes = {product.shape_rc for product in fields.values()}
    if len(shapes) != 1:
        raise ValueError(f"Mass and phase fields have different grids for {run_id}")
    for product in MASS_PRODUCTS:
        if fields[product].unit not in {"g pixel-1", "g/pixel"}:
            raise ValueError(
                f"{run_id}/{product} must store physical grams per pixel, "
                f"not {fields[product].unit!r}"
            )

    preview: PreviewProduct | None = None
    if corrected_background_required:
        reference = dict(references.get((run_id, "preview_images")) or {})
        if not reference:
            raise ValueError(f"Missing corrected preview image index for {run_id}")
        preview = load_preview_index(
            str(reference.get("path") or ""),
            str(reference.get("sha256") or ""),
            run_id=run_id,
            expected_times_hours=times,
        )
        if preview.load_frame(0).shape[:2] != next(iter(shapes)):
            raise ValueError(f"Preview and numerical grids differ for {run_id}")

    raw_injected = run_manifest.get("injected_mass_g")
    injected: list[float] | None = None
    if isinstance(raw_injected, list) and len(raw_injected) == len(times):
        candidate = [float(value) for value in raw_injected]
        if all(math.isfinite(value) and value >= 0.0 for value in candidate):
            injected = candidate
    return RunInputs(run_id, times, injected, fields, preview)


def _domain_slices(
    contract: Mapping[str, Any], shape_rc: tuple[int, int]
) -> dict[str, tuple[slice, slice]]:
    rows, columns = shape_rc
    result = {"whole_active_domain": (slice(0, rows), slice(0, columns))}
    available = {
        str(row.get("domain_id") or ""): row
        for row in (contract.get("analysis_domains") or {}).get("domains") or []
        if isinstance(row, Mapping) and row.get("domain_id")
    }
    for domain_id in contract.get("domains") or []:
        domain_id = str(domain_id)
        if domain_id == "whole_active_domain":
            continue
        row = available.get(domain_id)
        if row is None:
            raise ValueError(f"Analysis domain is unavailable: {domain_id}")
        normalized = [
            float(value) for value in row.get("normalized_xyxy_top_left") or []
        ]
        if len(normalized) != 4:
            raise ValueError(f"Analysis domain has no normalized bounds: {domain_id}")
        x0, y0, x1, y1 = normalized
        left = max(0, min(columns - 1, round(x0 * columns)))
        right = max(left + 1, min(columns, round(x1 * columns)))
        top = max(0, min(rows - 1, round(y0 * rows)))
        bottom = max(top + 1, min(rows, round(y1 * rows)))
        result[domain_id] = (slice(top, bottom), slice(left, right))
    return result


def _integrate_run(
    run: RunInputs,
    domains: Mapping[str, tuple[slice, slice]],
    *,
    include_phase_diagnostics: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    integrated: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for frame_index, time_hours in enumerate(run.times_hours):
        mass = {
            product: run.fields[product].load_frame(frame_index)[0]
            for product in MASS_PRODUCTS
        }
        for domain_id, bounds in domains.items():
            integrated.append(
                {
                    "run_id": run.run_id,
                    "frame_index": frame_index,
                    "time_h": time_hours,
                    "domain_id": domain_id,
                    "mass_total_g": float(np.sum(mass["mass_total"][bounds])),
                    "mass_aqueous_g": float(np.sum(mass["mass_aqueous"][bounds])),
                    "mass_free_gas_g": float(np.sum(mass["mass_free_gas"][bounds])),
                    "injected_mass_reference_g": (
                        run.injected_mass_g[frame_index]
                        if run.injected_mass_g is not None
                        else ""
                    ),
                }
            )
        whole = integrated[-len(domains)]
        aqueous_pixels = ""
        gas_pixels = ""
        maximum_aqueous = ""
        maximum_gas = ""
        if include_phase_diagnostics:
            aqueous, aqueous_display = run.fields["concentration_aqueous"].load_frame(
                frame_index
            )
            gas, gas_display = run.fields["saturation_free_gas"].load_frame(frame_index)
            aqueous_view = aqueous_display if aqueous_display is not None else aqueous
            gas_view = gas_display if gas_display is not None else gas
            aqueous_pixels = int(np.count_nonzero(aqueous_view > 0.0))
            gas_pixels = int(np.count_nonzero(gas_view > 0.0))
            maximum_aqueous = float(np.max(aqueous_view))
            maximum_gas = float(np.max(gas_view))
        diagnostics.append(
            {
                "run_id": run.run_id,
                "frame_index": frame_index,
                "time_h": time_hours,
                "mass_total_g": whole["mass_total_g"],
                "mass_aqueous_g": whole["mass_aqueous_g"],
                "mass_free_gas_g": whole["mass_free_gas_g"],
                "phase_sum_minus_total_g": (
                    whole["mass_aqueous_g"]
                    + whole["mass_free_gas_g"]
                    - whole["mass_total_g"]
                ),
                "aqueous_nonzero_pixels": aqueous_pixels,
                "free_gas_nonzero_pixels": gas_pixels,
                "maximum_aqueous_display_value": maximum_aqueous,
                "maximum_free_gas_saturation": maximum_gas,
            }
        )
    return integrated, diagnostics


def _rows_for_domain(
    rows: Sequence[Mapping[str, Any]], domain_id: str
) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("domain_id") == domain_id]


def _plot_mass_curve(
    path: Path,
    run: RunInputs,
    integrated: Sequence[Mapping[str, Any]],
    *,
    domain_id: str,
) -> None:
    rows = _rows_for_domain(integrated, domain_id)
    figure, axis = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
    axis.plot(
        [row["time_h"] for row in rows],
        [row["mass_total_g"] for row in rows],
        color="#20262c",
        linewidth=2.2,
        label="Detected total",
    )
    axis.plot(
        [row["time_h"] for row in rows],
        [row["mass_aqueous_g"] for row in rows],
        color="#24a98b",
        linewidth=1.8,
        label="Detected aqueous",
    )
    axis.plot(
        [row["time_h"] for row in rows],
        [row["mass_free_gas_g"] for row in rows],
        color="#d3468a",
        linewidth=1.8,
        label="Detected free gas",
    )
    if run.injected_mass_g is not None and domain_id == "whole_active_domain":
        axis.plot(
            run.times_hours,
            run.injected_mass_g,
            color="#2e6aa3",
            linewidth=1.5,
            linestyle="--",
            label="Injected reference",
        )
    axis.set_title(f"{run.run_id} | {domain_id}")
    axis.set_xlabel("Time [h]")
    axis.set_ylabel("CO2 mass [g]")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_domain_curves(
    path: Path,
    run: RunInputs,
    integrated: Sequence[Mapping[str, Any]],
    domain_ids: Sequence[str],
) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
    for domain_id in domain_ids:
        rows = _rows_for_domain(integrated, domain_id)
        axis.plot(
            [row["time_h"] for row in rows],
            [row["mass_total_g"] for row in rows],
            linewidth=1.8,
            label=domain_id,
        )
    axis.set_title(f"{run.run_id} | detected total mass by analysis domain")
    axis.set_xlabel("Time [h]")
    axis.set_ylabel("CO2 mass [g]")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_cross_run_grid(
    path: Path,
    runs: Sequence[RunInputs],
    integrated_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    columns = 2
    rows_count = math.ceil(len(runs) / columns)
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(12.5, max(4.0, rows_count * 4.2)),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, run in zip(axes.flat, runs, strict=False):
        rows = _rows_for_domain(integrated_by_run[run.run_id], "whole_active_domain")
        axis.plot(
            [row["time_h"] for row in rows],
            [row["mass_total_g"] for row in rows],
            color="#20262c",
            label="Total",
        )
        axis.plot(
            [row["time_h"] for row in rows],
            [row["mass_aqueous_g"] for row in rows],
            color="#24a98b",
            label="Aqueous",
        )
        axis.plot(
            [row["time_h"] for row in rows],
            [row["mass_free_gas_g"] for row in rows],
            color="#d3468a",
            label="Free gas",
        )
        if run.injected_mass_g is not None:
            axis.plot(
                run.times_hours,
                run.injected_mass_g,
                color="#2e6aa3",
                linestyle="--",
                label="Injected reference",
            )
        axis.set_title(run.run_id)
        axis.set_xlabel("Time [h]")
        axis.set_ylabel("CO2 mass [g]")
        axis.grid(True, alpha=0.25)
    for axis in axes.flat[len(runs) :]:
        axis.set_visible(False)
    axes.flat[0].legend(loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _injector_pixels(
    project: FluidFlowerProject,
    shape_rc: tuple[int, int],
    grid: Mapping[str, Any],
) -> list[tuple[int, int, str]]:
    rows, columns = shape_rc
    width_m = float(grid.get("width_m") or 0.0)
    height_m = float(grid.get("height_m") or 0.0)
    if width_m <= 0.0 or height_m <= 0.0:
        return []
    points: list[tuple[int, int, str]] = []
    for index, raw in enumerate(
        (project.manifest.get("rig") or {}).get("injectors") or []
    ):
        if not isinstance(raw, Mapping):
            continue
        try:
            x_m = float(raw.get("x_m"))
            y_m = float(raw.get("y_m"))
        except (TypeError, ValueError):
            continue
        column = round(np.clip(x_m / width_m, 0.0, 1.0) * (columns - 1))
        row = round((1.0 - np.clip(y_m / height_m, 0.0, 1.0)) * (rows - 1))
        points.append((column, row, str(raw.get("name") or f"Injector {index + 1}")))
    return points


def _resize_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    interpolation = cv2.INTER_AREA if width < image.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _draw_contours(
    image: np.ndarray,
    values: np.ndarray,
    levels: Sequence[float],
    colors: Sequence[tuple[int, int, int]],
) -> None:
    for index, level in enumerate(levels):
        mask = np.asarray(values >= float(level), dtype=np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(
            image,
            contours,
            -1,
            colors[min(index, len(colors) - 1)],
            max(1, round(min(image.shape[:2]) / 480)),
            cv2.LINE_AA,
        )


def _render_phase_frame(
    run: RunInputs,
    frame_index: int,
    *,
    mode: str,
    project: FluidFlowerProject,
    rendering: Mapping[str, Any],
) -> np.ndarray:
    concentration, concentration_display = run.fields[
        "concentration_aqueous"
    ].load_frame(frame_index)
    saturation, saturation_display = run.fields["saturation_free_gas"].load_frame(
        frame_index
    )
    aqueous = (
        concentration_display if concentration_display is not None else concentration
    )
    gas = saturation_display if saturation_display is not None else saturation
    if mode == "black":
        image = np.zeros((*run.shape_rc, 3), dtype=np.uint8)
    else:
        if run.preview is None:
            raise ValueError(f"Corrected background is unavailable for {run.run_id}")
        image = cv2.cvtColor(run.preview.load_frame(frame_index), cv2.COLOR_RGB2BGR)

    aqueous_levels = [float(value) for value in rendering["aqueous_levels_c_over_csat"]]
    gas_levels = [float(value) for value in rendering["free_gas_levels_saturation"]]
    if mode in {"overlay", "black"}:
        gas_mask = gas >= min(gas_levels)
        aqueous_mask = (aqueous >= min(aqueous_levels)) & ~gas_mask
        alpha = float(rendering.get("phase_overlay_alpha") or 0.58)
        if mode == "black":
            alpha = 1.0
        result = image.astype(np.float32)
        for mask, color in (
            (aqueous_mask, np.asarray([135, 205, 67], dtype=np.float32)),
            (gas_mask, np.asarray([143, 63, 224], dtype=np.float32)),
        ):
            result[mask] = (1.0 - alpha) * result[mask] + alpha * color
        image = np.clip(result, 0, 255).astype(np.uint8)
    elif mode == "contour":
        _draw_contours(
            image,
            aqueous,
            aqueous_levels,
            [(210, 210, 60), (190, 170, 30), (160, 120, 0)],
        )
        _draw_contours(image, gas, gas_levels, [(143, 63, 224)])
    else:
        raise ValueError(f"Unknown phase rendering mode: {mode}")

    if bool(rendering.get("show_injectors", True)):
        grid = run.fields["mass_total"].grid
        for column, row, name in _injector_pixels(project, run.shape_rc, grid):
            cv2.circle(image, (column, row), 7, (35, 35, 230), 2, cv2.LINE_AA)
            cv2.putText(
                image,
                name,
                (column + 10, row - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (35, 35, 230),
                1,
                cv2.LINE_AA,
            )
    if bool(rendering.get("show_elapsed_time", True)):
        label = f"{run.run_id} | {run.times_hours[frame_index]:.2f} h"
        cv2.putText(
            image,
            label,
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (245, 245, 245),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return _resize_width(image, int(rendering["movie_width_pixels"]))


def _render_run_cache(
    run: RunInputs,
    *,
    modes: Sequence[str],
    cache_root: Path,
    project: FluidFlowerProject,
    rendering: Mapping[str, Any],
    workers: int,
) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for mode in modes:
        mode_root = cache_root / run.run_id / mode
        mode_root.mkdir(parents=True, exist_ok=True)

        def render(index: int) -> Path:
            destination = mode_root / f"frame-{index:05d}.png"
            if destination.is_file():
                return destination
            image = _render_phase_frame(
                run,
                index,
                mode=mode,
                project=project,
                rendering=rendering,
            )
            if not cv2.imwrite(str(destination), image):
                raise RuntimeError(f"Could not write rendered frame: {destination}")
            return destination

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            result[mode] = list(pool.map(render, range(len(run.times_hours))))
    return result


def _video_writer(path: Path, shape: tuple[int, int], fps: float) -> tuple[Any, str]:
    width, height = shape
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError("No compatible MP4 video encoder is available")


def _write_movie(frame_paths: Sequence[Path], path: Path, fps: float) -> str:
    if not frame_paths:
        raise ValueError("A movie requires at least one frame")
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"Could not decode rendered frame: {frame_paths[0]}")
    writer, codec = _video_writer(path, (first.shape[1], first.shape[0]), fps)
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None or frame.shape != first.shape:
                raise ValueError(f"Rendered movie frame is invalid: {frame_path}")
            writer.write(frame)
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Video encoder did not publish a movie: {path}")
    return codec


def _contact_sheet(frame_paths: Sequence[Path], path: Path) -> None:
    count = min(12, len(frame_paths))
    indexes = sorted(
        set(np.linspace(0, len(frame_paths) - 1, count).round().astype(int))
    )
    images = [
        cv2.imread(str(frame_paths[index]), cv2.IMREAD_COLOR) for index in indexes
    ]
    if any(image is None for image in images):
        raise ValueError("Could not decode contact-sheet frames")
    valid = [image for image in images if image is not None]
    tile_width = min(720, valid[0].shape[1])
    resized = [_resize_width(image, tile_width) for image in valid]
    columns = min(4, len(resized))
    rows = math.ceil(len(resized) / columns)
    tile_height = resized[0].shape[0]
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, image in enumerate(resized):
        top = (index // columns) * tile_height
        left = (index % columns) * tile_width
        sheet[top : top + tile_height, left : left + tile_width] = image
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"Could not write contact sheet: {path}")


def _tile_images(images: Sequence[np.ndarray], labels: Sequence[str]) -> np.ndarray:
    columns = min(2, len(images))
    rows = math.ceil(len(images) / columns)
    tile_width = max(image.shape[1] for image in images)
    tile_height = max(image.shape[0] for image in images)
    canvas = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        top = (index // columns) * tile_height
        left = (index % columns) * tile_width
        canvas[top : top + image.shape[0], left : left + image.shape[1]] = image
        cv2.putText(
            canvas,
            label,
            (left + 14, top + tile_height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (left + 14, top + tile_height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _cross_run_movie(
    cached: Mapping[str, Sequence[Path]],
    path: Path,
    fps: float,
    cache_root: Path,
) -> str:
    run_ids = list(cached)
    frame_count = len(cached[run_ids[0]])
    if any(len(cached[run_id]) != frame_count for run_id in run_ids):
        raise ValueError("Cross-run phase grids require identical frame counts")
    grid_root = cache_root / "cross_run_grid"
    grid_root.mkdir(parents=True, exist_ok=True)
    grid_paths: list[Path] = []
    for frame_index in range(frame_count):
        destination = grid_root / f"frame-{frame_index:05d}.png"
        images = [
            cv2.imread(str(cached[run_id][frame_index]), cv2.IMREAD_COLOR)
            for run_id in run_ids
        ]
        if any(image is None for image in images):
            raise ValueError("Could not decode a cross-run phase frame")
        grid = _tile_images([image for image in images if image is not None], run_ids)
        if not cv2.imwrite(str(destination), grid):
            raise RuntimeError(f"Could not write cross-run frame: {destination}")
        grid_paths.append(destination)
    return _write_movie(grid_paths, path, fps)


def _artifact(
    staging: Path,
    *,
    artifact_id: str,
    product: str,
    run_ids: Sequence[str],
    path: Path,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "product": product,
        "run_ids": list(run_ids),
        "path": path.relative_to(staging).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "media_type": PRODUCT_MEDIA_TYPES[product],
    }


class PostAnalysisAdapter:
    """Render planned diagnostics from standardized, immutable mass products."""

    descriptor = AdapterDescriptor(
        key="post-analysis-v1",
        version=POST_MATERIALIZER_VERSION,
        title="Generic post-analysis",
        modules=("post_analysis",),
        backends=("cpu-float64",),
        description=(
            "Streams standardized mass and phase artifacts into verified tables, "
            "figures, phase movies, cross-run grids, and provenance."
        ),
    )

    def execute(
        self,
        task: Mapping[str, Any],
        context: TaskExecutionContext,
    ) -> Mapping[str, Any]:
        if task.get("module") != "post_analysis":
            raise ValueError(
                "The post-analysis adapter only accepts post-analysis tasks"
            )
        if (task.get("backend") or {}).get("name") != "cpu-float64":
            raise ValueError("Post-analysis uses the cpu-float64 backend contract")
        plan, plan_path, plan_file_sha256 = load_task_plan(task)
        if plan.get("format") != POST_ANALYSIS_REVISION_FORMAT:
            raise ValueError("The task does not reference a post-analysis revision")
        contract = dict(plan.get("task_contract") or {})
        if contract.get("format") != POST_ANALYSIS_TASK_CONTRACT_FORMAT:
            raise ValueError("The post-analysis task contract is unsupported")
        if not bool((plan.get("validation") or {}).get("execution_ready")):
            raise RuntimeError("The saved post-analysis plan is not execution-ready")

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
        references = _reference_map(contract)
        run_rows = _run_manifest_rows(mass_manifest)
        run_ids = [str(value) for value in contract.get("run_ids") or []]
        products = [str(value) for value in contract.get("products") or []]
        visual_required = bool(set(products) & VISUAL_PRODUCTS)
        corrected_required = bool(
            set(products)
            & {
                "phase_map_movie_corrected_background",
                "phase_contour_movie",
                "phase_contour_contact_sheet",
                "cross_run_phase_grid_movie",
            }
        )
        runs = [
            _load_run_inputs(
                run_id,
                run_rows.get(run_id) or {},
                references,
                visual_required=visual_required,
                corrected_background_required=corrected_required,
            )
            for run_id in run_ids
        ]
        if not runs:
            raise ValueError("Post-analysis has no selected runs")
        shared_times = runs[0].times_hours
        if any(run.times_hours != shared_times for run in runs[1:]):
            raise ValueError("Cross-run post-analysis requires one shared time grid")

        rendering = dict(contract.get("rendering") or {})
        frame_workers = min(
            24,
            max(
                1,
                int(
                    (task.get("resources") or {}).get("cpu_workers")
                    or (contract.get("execution_profile") or {}).get("frame_workers")
                    or 1
                ),
            ),
        )
        context.checkpoint(
            {
                "stage": "inputs_verified",
                "run_count": len(runs),
                "frame_count": len(shared_times),
            }
        )

        artifacts: list[dict[str, Any]] = []
        integrated_by_run: dict[str, list[dict[str, Any]]] = {}
        diagnostics_by_run: dict[str, list[dict[str, Any]]] = {}
        overlay_cache: dict[str, list[Path]] = {}
        encoder_records: list[dict[str, Any]] = []
        cache_root = context.workspace / "post_render_cache"

        for run_index, run in enumerate(runs, start=1):
            if context.cancellation_requested():
                raise RuntimeError("Post-analysis was cancelled")
            domains = _domain_slices(contract, run.shape_rc)
            integrated, diagnostics = _integrate_run(
                run,
                domains,
                include_phase_diagnostics=visual_required,
            )
            integrated_by_run[run.run_id] = integrated
            diagnostics_by_run[run.run_id] = diagnostics
            output = context.staging / "runs" / run.run_id
            output.mkdir(parents=True, exist_ok=True)

            if "integrated_mass_csv" in products:
                path = output / PRODUCT_FILENAMES["integrated_mass_csv"]
                _write_csv(path, integrated)
                artifacts.append(
                    _artifact(
                        context.staging,
                        artifact_id=f"integrated-mass-{run.run_id}",
                        product="integrated_mass_csv",
                        run_ids=[run.run_id],
                        path=path,
                    )
                )
            if "per_frame_diagnostics_csv" in products:
                path = output / PRODUCT_FILENAMES["per_frame_diagnostics_csv"]
                _write_csv(path, diagnostics)
                artifacts.append(
                    _artifact(
                        context.staging,
                        artifact_id=f"frame-diagnostics-{run.run_id}",
                        product="per_frame_diagnostics_csv",
                        run_ids=[run.run_id],
                        path=path,
                    )
                )
            if "mass_curve_full_domain" in products:
                path = output / PRODUCT_FILENAMES["mass_curve_full_domain"]
                _plot_mass_curve(path, run, integrated, domain_id="whole_active_domain")
                artifacts.append(
                    _artifact(
                        context.staging,
                        artifact_id=f"mass-curve-full-{run.run_id}",
                        product="mass_curve_full_domain",
                        run_ids=[run.run_id],
                        path=path,
                    )
                )
            if "mass_curve_analysis_domains" in products:
                path = output / PRODUCT_FILENAMES["mass_curve_analysis_domains"]
                _plot_domain_curves(path, run, integrated, list(domains))
                artifacts.append(
                    _artifact(
                        context.staging,
                        artifact_id=f"mass-curves-domains-{run.run_id}",
                        product="mass_curve_analysis_domains",
                        run_ids=[run.run_id],
                        path=path,
                    )
                )

            modes: list[str] = []
            if {
                "phase_map_movie_corrected_background",
                "cross_run_phase_grid_movie",
            } & set(products):
                modes.append("overlay")
            if "phase_map_movie_black_background" in products:
                modes.append("black")
            if {
                "phase_contour_movie",
                "phase_contour_contact_sheet",
            } & set(products):
                modes.append("contour")
            cached: dict[str, list[Path]] = {}
            if modes:
                cached = _render_run_cache(
                    run,
                    modes=modes,
                    cache_root=cache_root,
                    project=project,
                    rendering=rendering,
                    workers=frame_workers,
                )
                if "overlay" in cached:
                    overlay_cache[run.run_id] = cached["overlay"]
            movie_requests = {
                "phase_map_movie_corrected_background": "overlay",
                "phase_map_movie_black_background": "black",
                "phase_contour_movie": "contour",
            }
            for product, mode in movie_requests.items():
                if product not in products:
                    continue
                path = output / PRODUCT_FILENAMES[product]
                codec = _write_movie(
                    cached[mode], path, float(rendering.get("movie_fps") or 6.0)
                )
                encoder_records.append(
                    {"product": product, "run_ids": [run.run_id], "codec": codec}
                )
                artifacts.append(
                    _artifact(
                        context.staging,
                        artifact_id=f"{product}-{run.run_id}",
                        product=product,
                        run_ids=[run.run_id],
                        path=path,
                    )
                )
            if "phase_contour_contact_sheet" in products:
                path = output / PRODUCT_FILENAMES["phase_contour_contact_sheet"]
                _contact_sheet(cached["contour"], path)
                artifacts.append(
                    _artifact(
                        context.staging,
                        artifact_id=f"phase-contact-sheet-{run.run_id}",
                        product="phase_contour_contact_sheet",
                        run_ids=[run.run_id],
                        path=path,
                    )
                )
            context.checkpoint(
                {
                    "stage": "run_complete",
                    "run_id": run.run_id,
                    "completed_runs": run_index,
                    "run_count": len(runs),
                }
            )

        if "cross_run_mass_grid" in products:
            path = (
                context.staging / "cross_run" / PRODUCT_FILENAMES["cross_run_mass_grid"]
            )
            _plot_cross_run_grid(path, runs, integrated_by_run)
            artifacts.append(
                _artifact(
                    context.staging,
                    artifact_id="cross-run-mass-grid",
                    product="cross_run_mass_grid",
                    run_ids=run_ids,
                    path=path,
                )
            )
        if "cross_run_phase_grid_movie" in products:
            path = (
                context.staging
                / "cross_run"
                / PRODUCT_FILENAMES["cross_run_phase_grid_movie"]
            )
            codec = _cross_run_movie(
                overlay_cache,
                path,
                float(rendering.get("movie_fps") or 6.0),
                cache_root,
            )
            encoder_records.append(
                {
                    "product": "cross_run_phase_grid_movie",
                    "run_ids": run_ids,
                    "codec": codec,
                }
            )
            artifacts.append(
                _artifact(
                    context.staging,
                    artifact_id="cross-run-phase-grid",
                    product="cross_run_phase_grid_movie",
                    run_ids=run_ids,
                    path=path,
                )
            )

        if "provenance_json" in products:
            path = context.staging / PRODUCT_FILENAMES["provenance_json"]
            provenance = {
                "schema": 1,
                "format": "darsia-fluidflower-post-analysis-provenance-v1",
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
                "backend": deepcopy(dict(task.get("backend") or {})),
                "hardware": deepcopy(dict(context.inventory)),
                "frame_workers": frame_workers,
                "video_encoders": encoder_records,
                "scientific_contract": {
                    "raw_detected_mass_is_authoritative": True,
                    "injected_mass_is_reference_only": True,
                    "force_mass_conservation": False,
                    "phase_map_scope": "camera_visible_2d_slice",
                    "phase_map_representation": rendering.get(
                        "phase_map_representation"
                    ),
                },
                "input_references": [
                    deepcopy(dict(row)) for row in references.values()
                ],
            }
            _atomic_json(path, provenance)
            artifacts.append(
                _artifact(
                    context.staging,
                    artifact_id="post-analysis-provenance",
                    product="provenance_json",
                    run_ids=[],
                    path=path,
                )
            )

        expected = {
            (row["product"], tuple(row["run_ids"]))
            for row in expected_post_analysis_artifacts(
                dict(plan.get("configuration") or {})
            )
        }
        observed = {
            (row["product"], tuple(sorted(row["run_ids"]))) for row in artifacts
        }
        if expected != observed:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise RuntimeError(
                "Post-analysis did not materialize the exact planned artifact "
                f"coverage: missing={missing}, extra={extra}"
            )
        result_manifest = {
            "schema": 1,
            "format": POST_ANALYSIS_RESULT_FORMAT,
            "post_analysis_plan_id": str(plan.get("plan_id") or ""),
            "post_analysis_task_contract_sha256": str(
                contract.get("content_sha256") or ""
            ),
            "task_id": str(task["task_id"]),
            "analysis_lineage": deepcopy(dict(plan.get("upstream") or {})),
            "adapter": {
                "key": self.descriptor.key,
                "version": self.descriptor.version,
            },
            "complete": True,
            "artifacts": artifacts,
        }
        _atomic_json(context.staging / "post_analysis_result.json", result_manifest)
        _atomic_json(context.staging / "task.json", task)
        _atomic_json(context.staging / "plan.json", plan)
        if cache_root.is_dir():
            shutil.rmtree(cache_root)
        context.checkpoint(
            {
                "stage": "post_analysis_ready",
                "artifact_count": len(artifacts),
            }
        )
        return {
            "result_manifest": "post_analysis_result.json",
            "artifact_count": len(artifacts),
            "run_count": len(runs),
            "frame_count": len(shared_times),
            "scientific_results_created": False,
        }
