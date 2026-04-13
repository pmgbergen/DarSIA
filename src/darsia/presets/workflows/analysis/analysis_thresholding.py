"""Template for thresholding analysis."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from darsia.presets.workflows.analysis.analysis_context import (
    AnalysisContext,
    prepare_analysis_context,
)
from darsia.presets.workflows.analysis.streaming import _to_bgr_array, publish_stream_images
from darsia.presets.workflows.config.analysis import AnalysisThresholdingConfig
from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)


def _to_scalar_array(image_like: Any) -> np.ndarray:
    array = np.asarray(image_like.img if hasattr(image_like, "img") else image_like)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 1:
        return array[..., 0]
    raise ValueError(f"Thresholding requires scalar images, got shape {array.shape}.")


def _rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return (int(color[2]), int(color[1]), int(color[0]))


def _apply_legend(
    frame: np.ndarray,
    *,
    text: str,
    legend_config,
) -> np.ndarray:
    if not legend_config.show:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = legend_config.font_scale
    thickness = legend_config.thickness
    line_spacing = legend_config.line_spacing
    padding = legend_config.box_padding
    baseline = 0
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    box_w = text_w + 2 * padding
    box_h = text_h + baseline + 2 * padding + line_spacing
    h, w = frame.shape[:2]
    x = min(max(int(legend_config.position[0]), 0), max(w - box_w, 0))
    y = min(max(int(legend_config.position[1]), 0), max(h - box_h, 0))

    if legend_config.box_enabled:
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + box_w, y + box_h),
            _rgb_to_bgr(legend_config.box_color),
            thickness=-1,
        )
        cv2.addWeighted(
            overlay,
            legend_config.box_alpha,
            frame,
            1 - legend_config.box_alpha,
            0.0,
            dst=frame,
        )

    text_org = (x + padding, y + padding + text_h)
    cv2.putText(
        frame,
        text,
        text_org,
        font,
        font_scale,
        _rgb_to_bgr(legend_config.text_color),
        thickness,
        cv2.LINE_AA,
    )
    return frame


def _extract_mode_images(result: Any) -> dict[str, Any]:
    return {
        "concentration_aq": result.concentration_aq,
        "saturation_g": result.saturation_g,
        "mass_total": result.mass,
        "mass_g": result.mass_g,
        "mass_aq": result.mass_aq,
    }


def analysis_thresholding_from_context(
    ctx: AnalysisContext,
    show: bool = False,
    stream_callback: Callable[[dict[str, bytes] | None], None] | None = None,
) -> None:
    """Thresholding analysis using pre-prepared context."""
    assert ctx.config.data is not None
    assert ctx.config.analysis is not None
    assert ctx.color_to_mass_analysis is not None

    config = ctx.config
    fluidflower = ctx.fluidflower
    image_paths = ctx.image_paths
    color_to_mass_analysis = ctx.color_to_mass_analysis

    if config.analysis.thresholding is None:
        config.analysis.thresholding = AnalysisThresholdingConfig().load(
            sec={"thresholding": {}},
            results=config.data.results,
        )

    thresholding_config = config.analysis.thresholding
    thresholding_config.folder.mkdir(parents=True, exist_ok=True)

    # Storage folder organization by mode.
    mode_folders = {
        mode: thresholding_config.folder / mode for mode in thresholding_config.modes
    }
    for mode_folder in mode_folders.values():
        mode_folder.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        img = fluidflower.read_image(path)
        result = color_to_mass_analysis(img)
        mode_images = _extract_mode_images(result)
        stream_payload: dict[str, Any] = {"thresholding_source_image": img}

        for mode in thresholding_config.modes:
            scalar = _to_scalar_array(mode_images[mode])
            threshold = float(thresholding_config.thresholds[mode])
            mask = (scalar >= threshold).astype(np.uint8)

            np.savez_compressed(
                mode_folders[mode] / f"{path.stem}.npz",
                mask=mask,
                threshold=threshold,
                mode=mode,
            )

            preview = np.where(mask > 0, 255, 0).astype(np.uint8)
            preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            preview = _apply_legend(
                preview,
                text=f"{mode} >= {threshold:g}",
                legend_config=thresholding_config.legend,
            )
            cv2.imwrite(str(mode_folders[mode] / f"{path.stem}.jpg"), preview)
            stream_payload[f"thresholding_{mode}"] = cv2.cvtColor(
                preview, cv2.COLOR_BGR2RGB
            )

        if show:
            import matplotlib.pyplot as plt

            img.show(title=f"Image at {path.stem}", delay=True)
            for mode in thresholding_config.modes:
                mode_preview = _to_bgr_array(stream_payload[f"thresholding_{mode}"])
                plt.figure()
                plt.title(f"Thresholding {mode} at {path.stem}")
                plt.imshow(cv2.cvtColor(mode_preview, cv2.COLOR_BGR2RGB))
                plt.axis("off")
            plt.show()

        publish_stream_images(
            stream_callback=stream_callback,
            image_payload=stream_payload,
            logger=logger,
            error_message=f"Failed to stream thresholding previews for image '{path}'.",
        )


def analysis_thresholding(
    cls: type[Rig],
    path: Path | list[Path],
    all: bool = False,
    show: bool = False,
    stream_callback: Callable[[dict[str, bytes] | None], None] | None = None,
) -> None:
    """Thresholding analysis (standalone entry point)."""
    ctx = prepare_analysis_context(
        cls=cls,
        path=path,
        all=all,
        require_color_to_mass=True,
    )
    analysis_thresholding_from_context(ctx, show=show, stream_callback=stream_callback)
