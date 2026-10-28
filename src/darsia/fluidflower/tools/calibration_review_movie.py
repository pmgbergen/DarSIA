"""Reusable calibration-review movies with mass curves and phase contours."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


FRAME_WIDTH = 1600
FRAME_HEIGHT = 900
DEFAULT_FPS = 5.0
DEFAULT_PLOT_SECONDS = 4.0
DEFAULT_IMAGE_SECONDS = 1.4
AQUEOUS_LEVELS = (0.01, 0.05, 0.30, 0.50)
GAS_LEVELS = (0.05,)
MULTILEVEL_GAS_LEVELS = (0.005, 0.01, 0.03, 0.05)
_GAS_COLORS = (
    (255, 205, 70),
    (255, 145, 45),
    (245, 70, 60),
    (215, 45, 150),
)


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if np.issubdtype(values.dtype, np.floating):
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        if values.size and float(np.max(values)) <= 1.5:
            values = values * 255.0
    values = np.clip(values, 0, 255).astype(np.uint8)
    if values.ndim == 2:
        values = np.repeat(values[..., None], 3, axis=2)
    return np.ascontiguousarray(values[..., :3])


def _draw_level(
    image: np.ndarray,
    field: np.ndarray,
    level: float,
    color: tuple[int, ...],
    thickness: int,
) -> None:
    mask = np.asarray(np.isfinite(field) & (field >= level), dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    outline = (15, 18, 20, 255) if image.shape[2] == 4 else (15, 18, 20)
    cv2.drawContours(
        image,
        contours,
        -1,
        outline,
        thickness + 2,
        lineType=cv2.LINE_AA,
    )
    cv2.drawContours(
        image,
        contours,
        -1,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def _review_geometry(
    image_shape: Sequence[int],
    *,
    width: int,
    height: int,
    preserve_native_resolution: bool,
) -> tuple[float, int, int, int, int, int, int]:
    """Return scalable review layout and image placement geometry."""

    source_height, source_width = int(image_shape[0]), int(image_shape[1])
    layout_scale = min(width / FRAME_WIDTH, height / FRAME_HEIGHT)
    header_height = max(1, int(round(126 * layout_scale)))
    margin = max(1, int(round(18 * layout_scale)))
    available_width = width - 2 * margin
    available_height = height - header_height - margin
    if preserve_native_resolution:
        if source_width > available_width or source_height > available_height:
            raise ValueError(
                "Native image does not fit inside the requested review frame"
            )
        target_width, target_height = source_width, source_height
    else:
        image_scale = min(
            available_width / source_width,
            available_height / source_height,
        )
        target_width = max(2, int(round(source_width * image_scale)))
        target_height = max(2, int(round(source_height * image_scale)))
    x0 = (width - target_width) // 2
    y0 = header_height + (available_height - target_height) // 2
    return (
        layout_scale,
        header_height,
        margin,
        x0,
        y0,
        target_width,
        target_height,
    )


def _resize_field(
    field: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    values = np.asarray(field, dtype=np.float32)
    if values.shape[:2] == (target_height, target_width):
        return values
    return cv2.resize(
        values,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )


def _scaled(value: float, layout_scale: float, *, minimum: int = 1) -> int:
    return max(minimum, int(round(value * layout_scale)))


def render_contour_frame(
    *,
    image: np.ndarray,
    concentration_aq: np.ndarray,
    saturation_g: np.ndarray,
    run: str,
    variant: str,
    frame_index: int,
    frame_count: int,
    elapsed_h: float,
    injected_mass_kg: float,
    detected_total_mass_kg: float,
    detected_gas_mass_kg: float,
    detected_aqueous_mass_kg: float,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    gas_fill_alpha: float = 0.20,
    show_aqueous: bool = True,
    show_gas: bool = True,
    gas_levels: Sequence[float] | None = None,
    preserve_native_resolution: bool = False,
    header_title: str | None = None,
    header_mass: str | None = None,
    status_text: str | None = None,
) -> np.ndarray:
    """Render one corrected calibration image with phase contours."""

    canvas = np.full((height, width, 3), (19, 23, 27), dtype=np.uint8)
    rgb = _to_uint8_rgb(image)
    (
        layout_scale,
        header_height,
        margin,
        x0,
        y0,
        target_width,
        target_height,
    ) = _review_geometry(
        rgb.shape,
        width=width,
        height=height,
        preserve_native_resolution=preserve_native_resolution,
    )
    if rgb.shape[:2] != (target_height, target_width):
        rgb = cv2.resize(
            rgb,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
    aqueous = _resize_field(concentration_aq, target_width, target_height)
    gas = _resize_field(saturation_g, target_width, target_height)

    selected_gas_levels = tuple(
        float(level) for level in (GAS_LEVELS if gas_levels is None else gas_levels)
    )
    if show_gas and (
        not selected_gas_levels
        or not all(np.isfinite(level) and level > 0.0 for level in selected_gas_levels)
        or any(
            right <= left
            for left, right in zip(
                selected_gas_levels,
                selected_gas_levels[1:],
                strict=False,
            )
        )
    ):
        raise ValueError("Gas contour levels must be finite, positive, and increasing.")

    fill_alpha = float(gas_fill_alpha)
    if not 0.0 <= fill_alpha <= 1.0:
        raise ValueError("gas_fill_alpha must be in [0, 1].")
    gas_fill_level = selected_gas_levels[0] if selected_gas_levels else 1.0
    gas_mask = np.isfinite(gas) & (gas >= gas_fill_level)
    if not show_gas:
        gas_mask[:] = False
    if fill_alpha > 0.0 and np.any(gas_mask):
        gas_color = np.asarray((245, 70, 60), dtype=np.float32)
        rgb_float = rgb.astype(np.float32)
        rgb_float[gas_mask] = (
            (1.0 - fill_alpha) * rgb_float[gas_mask] + fill_alpha * gas_color
        )
        rgb = np.clip(rgb_float, 0, 255).astype(np.uint8)

    aqueous_styles = (
        (0.01, (40, 195, 230), _scaled(1, layout_scale)),
        (0.05, (30, 225, 115), _scaled(2, layout_scale)),
        (0.30, (115, 245, 170), _scaled(2, layout_scale)),
        (0.50, (220, 255, 235), _scaled(2, layout_scale)),
    )
    if show_aqueous:
        for level, color, thickness in aqueous_styles:
            _draw_level(rgb, aqueous, level, color, thickness)
    # Draw gas last: in coupled models its boundary can nearly coincide with
    # high aqueous levels, which otherwise paint over and hide the gas contour.
    gas_styles = tuple(
        (
            level,
            _GAS_COLORS[
                round(index * (len(_GAS_COLORS) - 1) / max(1, len(selected_gas_levels) - 1))
            ],
            _scaled(3 if index == len(selected_gas_levels) - 1 else 2, layout_scale),
        )
        for index, level in enumerate(selected_gas_levels)
    )
    if show_gas:
        for level, color, thickness in gas_styles:
            _draw_level(rgb, gas, level, color, thickness)

    canvas[y0 : y0 + target_height, x0 : x0 + target_width] = rgb

    ratio = (
        detected_total_mass_kg / injected_mass_kg
        if injected_mass_kg > 0.0
        else float("nan")
    )
    title = header_title or (
        f"{run.upper()}  |  {variant}  |  calibration frame "
        f"{frame_index + 1}/{frame_count}  |  t={elapsed_h:.3f} h"
    )
    cv2.putText(
        canvas,
        title,
        (_scaled(24, layout_scale), _scaled(34, layout_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68 * layout_scale,
        (245, 247, 249),
        _scaled(2, layout_scale),
        cv2.LINE_AA,
    )
    ratio_label = f"{ratio:.3f}" if np.isfinite(ratio) else "n/a"
    mass_line = header_mass or (
        f"Injected {1000.0 * injected_mass_kg:.4f} g    "
        f"Detected {1000.0 * detected_total_mass_kg:.4f} g    "
        f"Aqueous {1000.0 * detected_aqueous_mass_kg:.4f} g    "
        f"Gas {1000.0 * detected_gas_mass_kg:.4f} g    ratio {ratio_label}"
    )
    cv2.putText(
        canvas,
        mass_line,
        (_scaled(24, layout_scale), _scaled(70, layout_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58 * layout_scale,
        (218, 223, 228),
        _scaled(1, layout_scale),
        cv2.LINE_AA,
    )
    legend_items = []
    if show_aqueous:
        legend_items.extend(
            (
                ("Aq c/csat = 0.01", (40, 195, 230), 1),
                ("Aq c/csat = 0.05", (30, 225, 115), 2),
                ("Aq c/csat = 0.30", (115, 245, 170), 2),
                ("Aq c/csat = 0.50", (220, 255, 235), 2),
            )
        )
    if show_gas:
        legend_items.extend(
            (
                f"Gas Sg = {level:g}",
                color,
                thickness,
            )
            for level, color, thickness in gas_styles
        )
    if legend_items:
        legend_width = (width - _scaled(48, layout_scale)) // len(legend_items)
        for index, (label, color, thickness) in enumerate(legend_items):
            legend_x = _scaled(24, layout_scale) + index * legend_width
            cv2.line(
                canvas,
                (legend_x, _scaled(103, layout_scale)),
                (legend_x + _scaled(25, layout_scale), _scaled(103, layout_scale)),
                color,
                thickness + _scaled(1, layout_scale),
                cv2.LINE_AA,
            )
            legend_font_scale = 0.36 if len(legend_items) > 5 else 0.43
            cv2.putText(
                canvas,
                label,
                (legend_x + _scaled(34, layout_scale), _scaled(109, layout_scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                legend_font_scale * layout_scale,
                (225, 230, 234),
                _scaled(1, layout_scale),
                cv2.LINE_AA,
            )
    else:
        message = (
            "Corrected image | phase contours hidden"
            if status_text is None
            else status_text
        )
        if message:
            cv2.putText(
                canvas,
                message,
                (_scaled(24, layout_scale), _scaled(109, layout_scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46 * layout_scale,
                (174, 182, 188),
                _scaled(1, layout_scale),
                cv2.LINE_AA,
            )
    return canvas


def render_contour_overlay(
    *,
    image_shape: Sequence[int],
    concentration_aq: np.ndarray,
    saturation_g: np.ndarray,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    show_aqueous: bool = True,
    show_gas: bool = True,
    gas_levels: Sequence[float] | None = None,
    preserve_native_resolution: bool = False,
) -> np.ndarray:
    """Render phase contours as a transparent RGBA review overlay."""

    (
        layout_scale,
        _header_height,
        _margin,
        x0,
        y0,
        target_width,
        target_height,
    ) = _review_geometry(
        image_shape,
        width=width,
        height=height,
        preserve_native_resolution=preserve_native_resolution,
    )
    aqueous = _resize_field(concentration_aq, target_width, target_height)
    gas = _resize_field(saturation_g, target_width, target_height)
    overlay = np.zeros((target_height, target_width, 4), dtype=np.uint8)

    aqueous_styles = (
            (0.01, (40, 195, 230, 255), _scaled(1, layout_scale)),
            (0.05, (30, 225, 115, 255), _scaled(2, layout_scale)),
            (0.30, (115, 245, 170, 255), _scaled(2, layout_scale)),
            (0.50, (220, 255, 235, 255), _scaled(2, layout_scale)),
    )
    if show_aqueous:
        for level, color, thickness in aqueous_styles:
            _draw_level(overlay, aqueous, level, color, thickness)

    selected_gas_levels = tuple(
        float(level) for level in (GAS_LEVELS if gas_levels is None else gas_levels)
    )
    if show_gas and (
        not selected_gas_levels
        or not all(np.isfinite(level) and level > 0.0 for level in selected_gas_levels)
        or any(
            right <= left
            for left, right in zip(
                selected_gas_levels,
                selected_gas_levels[1:],
                strict=False,
            )
        )
    ):
        raise ValueError("Gas contour levels must be finite, positive, and increasing.")
    gas_styles = []
    if show_gas:
        for index, level in enumerate(selected_gas_levels):
            color = _GAS_COLORS[
                round(
                    index
                    * (len(_GAS_COLORS) - 1)
                    / max(1, len(selected_gas_levels) - 1)
                )
            ] + (255,)
            thickness = _scaled(
                3 if index == len(selected_gas_levels) - 1 else 2,
                layout_scale,
            )
            _draw_level(overlay, gas, level, color, thickness)
            gas_styles.append((level, color, thickness))

    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    canvas[y0 : y0 + target_height, x0 : x0 + target_width] = overlay
    legend_items = []
    if show_aqueous:
        legend_items.extend(
            (f"Aq c/csat = {level:g}", color, thickness)
            for level, color, thickness in aqueous_styles
        )
    legend_items.extend(
        (f"Gas Sg = {level:g}", color, thickness)
        for level, color, thickness in gas_styles
    )
    if legend_items:
        legend_width = (width - _scaled(48, layout_scale)) // len(legend_items)
        for index, (label, color, thickness) in enumerate(legend_items):
            legend_x = _scaled(24, layout_scale) + index * legend_width
            cv2.line(
                canvas,
                (legend_x, _scaled(103, layout_scale)),
                (legend_x + _scaled(25, layout_scale), _scaled(103, layout_scale)),
                color,
                thickness + _scaled(1, layout_scale),
                cv2.LINE_AA,
            )
            font_scale = (0.36 if len(legend_items) > 5 else 0.43) * layout_scale
            cv2.putText(
                canvas,
                label,
                (legend_x + _scaled(34, layout_scale), _scaled(109, layout_scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (225, 230, 234, 255),
                _scaled(1, layout_scale),
                cv2.LINE_AA,
            )
    return canvas


def render_mass_plot(
    records: Sequence[Mapping[str, Any]],
    *,
    run: str,
    variant: str,
    objective_full_scale: float | None = None,
    mass_objective_full_scale: float | None = None,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
) -> np.ndarray:
    """Render the opening mass-balance figure as an RGB movie frame."""

    elapsed = np.asarray([float(row["elapsed_h"]) for row in records])
    injected = 1000.0 * np.asarray(
        [float(row["injected_mass_kg"]) for row in records]
    )
    total = 1000.0 * np.asarray(
        [float(row["detected_total_mass_kg"]) for row in records]
    )
    aqueous = 1000.0 * np.asarray(
        [float(row["detected_aqueous_mass_kg"]) for row in records]
    )
    gas = 1000.0 * np.asarray(
        [float(row["detected_gas_mass_kg"]) for row in records]
    )
    valid_ratio = injected > 0.0
    relative = np.full_like(injected, np.nan)
    relative[valid_ratio] = np.abs(total[valid_ratio] / injected[valid_ratio] - 1.0)
    median_relative = float(np.nanmedian(relative)) if np.any(valid_ratio) else float("nan")

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    axis = fig.add_subplot(1, 1, 1)
    fig.patch.set_facecolor("#f6f7f8")
    axis.set_facecolor("#ffffff")
    axis.plot(elapsed, injected, color="#2463a6", linewidth=2.5, label="Injected")
    axis.plot(
        elapsed,
        total,
        color="#202124",
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label="Detected total",
    )
    axis.plot(
        elapsed,
        aqueous,
        color="#16845b",
        linewidth=1.9,
        marker="o",
        markersize=3.5,
        label="Dissolved CO2",
    )
    axis.plot(
        elapsed,
        gas,
        color="#d64541",
        linewidth=1.9,
        marker="o",
        markersize=3.5,
        label="Free CO2",
    )
    for boundary, label in ((4.1, "Early redistribution"), (8.0, "Late redistribution")):
        if elapsed.min() <= boundary <= elapsed.max():
            axis.axvline(boundary, color="#aeb4ba", linewidth=1.0, linestyle="--")
            axis.text(
                boundary,
                0.98,
                label,
                rotation=90,
                va="top",
                ha="right",
                color="#6b7278",
                fontsize=8,
                transform=axis.get_xaxis_transform(),
            )
    if objective_full_scale is None:
        objective_label = "Objective: unavailable"
    elif mass_objective_full_scale is None:
        objective_label = f"Objective: {1000.0 * objective_full_scale:.4f} g"
    else:
        penalty = max(0.0, objective_full_scale - mass_objective_full_scale)
        objective_label = (
            f"Composite objective: {1000.0 * objective_full_scale:.4f} g "
            f"(mass L1 {1000.0 * mass_objective_full_scale:.4f} g + "
            f"penalties {1000.0 * penalty:.4f} g)"
        )
    axis.set_title(
        f"{run.upper()} calibration: injected and detected CO2 mass\n{variant}",
        loc="left",
        fontsize=17,
        pad=18,
    )
    axis.text(
        0.995,
        1.025,
        f"{objective_label}   |   median absolute relative error: {100.0 * median_relative:.1f}%",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#4d5358",
    )
    axis.set_xlabel("Elapsed time (h)")
    axis.set_ylabel("CO2 mass (g)")
    axis.grid(color="#d9dde1", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    fig.subplots_adjust(left=0.085, right=0.97, top=0.84, bottom=0.18)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = np.ascontiguousarray(rgba[..., :3])
    plt.close(fig)
    return frame


def write_calibration_movie(
    *,
    output: Path,
    plot_frame: np.ndarray,
    contour_frames: Sequence[np.ndarray],
    fps: float = DEFAULT_FPS,
    plot_seconds: float = DEFAULT_PLOT_SECONDS,
    image_seconds: float = DEFAULT_IMAGE_SECONDS,
    gas_levels: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Write an atomic MP4 with a plot slide followed by contour frames."""

    if not contour_frames:
        raise ValueError("At least one contour frame is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.mp4")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(plot_frame.shape[1]), int(plot_frame.shape[0])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not initialize MP4 writer for {output}")
    plot_repeats = max(1, int(round(float(fps) * float(plot_seconds))))
    image_repeats = max(1, int(round(float(fps) * float(image_seconds))))
    try:
        for _ in range(plot_repeats):
            writer.write(cv2.cvtColor(plot_frame, cv2.COLOR_RGB2BGR))
        for frame in contour_frames:
            if frame.shape != plot_frame.shape:
                raise ValueError("All calibration movie frames must have equal shape")
            encoded = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for _ in range(image_repeats):
                writer.write(encoded)
    finally:
        writer.release()
    if not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Calibration movie is empty: {temporary}")
    os.replace(temporary, output)
    return {
        "schema": 1,
        "movie": str(output),
        "fps": float(fps),
        "plot_seconds": float(plot_seconds),
        "image_seconds_per_calibration_frame": float(image_seconds),
        "calibration_frame_count": len(contour_frames),
        "aqueous_contour_levels": list(AQUEOUS_LEVELS),
        "gas_contour_levels": list(
            GAS_LEVELS if gas_levels is None else tuple(float(level) for level in gas_levels)
        ),
        "duration_seconds": (plot_repeats + image_repeats * len(contour_frames))
        / float(fps),
    }


def write_movie_provenance(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    os.replace(temporary, path)
