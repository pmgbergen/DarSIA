"""Utils for creating protocol-time ordered videos/GIFs from configured image sources."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

import darsia
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig

logger = logging.getLogger(__name__)


def _resolve_source_folder(config: FluidFlowerConfig) -> Path:
    assert config.data is not None
    assert config.video is not None
    src = config.video.source.folder
    if src is None:
        raise ValueError("Missing required [video.source].folder.")
    return src if src.is_absolute() else config.data.results / src


def _scan_source_images(
    folder: Path,
    pattern: str | None,
    extensions: list[str],
    recursive: bool,
) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Video source folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Video source folder is not a directory: {folder}")

    raw_paths = (
        list(folder.rglob(pattern or "*"))
        if recursive
        else list(folder.glob(pattern or "*"))
    )
    ext_set = {e.lower() for e in extensions}
    return sorted({p for p in raw_paths if p.is_file() and p.suffix.lower() in ext_set})


def _protocol_sort_frames(
    experiment: darsia.ProtocolledExperiment, image_paths: list[Path]
) -> list[tuple[Path, datetime, float]]:
    rows: list[tuple[Path, datetime, float]] = []
    for path in image_paths:
        try:
            if experiment.imaging_protocol.is_blacklisted(path):
                continue
            dt = experiment.imaging_protocol.get_datetime(path)
        except ValueError:
            continue
        elapsed_time_h = experiment.time_since_start(dt)
        rows.append((path, dt, elapsed_time_h))
    rows.sort(key=lambda item: item[1])
    return rows


def _apply_overlay(
    frame: np.ndarray,
    elapsed_time_h: float,
    overlay_config,
) -> np.ndarray:
    lines: list[str] = []
    if overlay_config.show_elapsed_time:
        lines.append(overlay_config.elapsed_time_format.format(elapsed_time_h))
    if overlay_config.show_note and overlay_config.note.strip():
        lines.append(overlay_config.note.strip())
    if not lines:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = overlay_config.font_scale
    thickness = overlay_config.thickness
    line_spacing = overlay_config.line_spacing
    padding = overlay_config.box_padding

    text_sizes = [
        cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines
    ]
    text_height = sum(size[1] for size in text_sizes) + line_spacing * (len(lines) - 1)
    text_width = max((size[0] for size in text_sizes), default=0)
    box_w = text_width + 2 * padding
    box_h = text_height + 2 * padding

    h, w = frame.shape[:2]
    x = min(max(int(overlay_config.position[0]), 0), max(w - box_w, 0))
    y = min(max(int(overlay_config.position[1]), 0), max(h - box_h, 0))

    if overlay_config.box_enabled:
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + box_w, y + box_h),
            _rgb_to_bgr(overlay_config.box_color),
            thickness=-1,
        )
        frame = cv2.addWeighted(
            overlay,
            overlay_config.box_alpha,
            frame,
            1 - overlay_config.box_alpha,
            0,
        )

    cursor_y = y + padding
    for line, size in zip(lines, text_sizes):
        baseline_y = cursor_y + size[1]
        cv2.putText(
            frame,
            line,
            (x + padding, baseline_y),
            font,
            font_scale,
            _rgb_to_bgr(overlay_config.text_color),
            thickness,
            cv2.LINE_AA,
        )
        cursor_y = baseline_y + line_spacing
    return frame


def _rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return color[2], color[1], color[0]


def _read_and_process_frames(
    ordered_frames: list[tuple[Path, datetime, float]],
    resolution: tuple[int, int] | None,
    overlay_config,
) -> list[np.ndarray]:
    processed: list[np.ndarray] = []
    target_resolution = resolution
    for idx, (path, _, elapsed_time_h) in enumerate(ordered_frames):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning("Failed to read source frame '%s'; skipping.", path)
            continue
        if idx == 0 and target_resolution is None:
            target_resolution = (frame.shape[1], frame.shape[0])
        assert target_resolution is not None
        frame = cv2.resize(frame, target_resolution, interpolation=cv2.INTER_AREA)
        frame = _apply_overlay(frame, elapsed_time_h, overlay_config)
        processed.append(frame)
    if not processed:
        raise ValueError("No readable frames available for media generation.")
    return processed


def _write_mp4(
    frames: list[np.ndarray],
    output_path: Path,
    fps: float,
    codec: str,
) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for '{output_path}'.")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _write_gif(
    frames: list[np.ndarray],
    output_path: Path,
    fps: float,
) -> None:
    duration_ms = max(int(round(1000.0 / fps)), 1)
    pil_frames = [
        PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames
    ]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def build_media(path: Path | list[Path]) -> dict[str, Path]:
    """Build protocol-time ordered MP4/GIF from configured analysis outputs."""
    config = FluidFlowerConfig(path, require_data=True, require_results=False)
    config.check("data", "protocol", "video")
    assert config.video is not None
    assert config.data is not None

    experiment = darsia.ProtocolledExperiment.init_from_config(config)
    source_folder = _resolve_source_folder(config)
    source_paths = _scan_source_images(
        folder=source_folder,
        pattern=config.video.source.pattern,
        extensions=config.video.source.extensions,
        recursive=bool(config.video.source.recursive),
    )
    if not source_paths:
        raise FileNotFoundError(f"No source images found in '{source_folder}'.")
    ordered = _protocol_sort_frames(experiment, source_paths)
    if not ordered:
        raise ValueError(
            "No source images could be matched to the imaging protocol (or all were "
            "blacklisted)."
        )
    logger.info(
        "Matched %d/%d source images to protocol and sorted by protocol datetime.",
        len(ordered),
        len(source_paths),
    )

    output_folder = config.video.folder
    output_folder.mkdir(parents=True, exist_ok=True)
    stem = config.video.output.filename or config.video.analysis
    frames = _read_and_process_frames(
        ordered_frames=ordered,
        resolution=config.video.output.resolution,
        overlay_config=config.video.overlay,
    )

    outputs: dict[str, Path] = {}
    if "mp4" in config.video.output.formats:
        out = output_folder / f"{stem}.mp4"
        _write_mp4(
            frames=frames,
            output_path=out,
            fps=config.video.output.fps,
            codec=config.video.output.codec,
        )
        outputs["mp4"] = out
    if "gif" in config.video.output.formats:
        out = output_folder / f"{stem}.gif"
        _write_gif(
            frames=frames,
            output_path=out,
            fps=config.video.output.fps,
        )
        outputs["gif"] = out

    for kind, out_path in outputs.items():
        logger.info("Saved %s to %s", kind.upper(), out_path)
    return outputs
