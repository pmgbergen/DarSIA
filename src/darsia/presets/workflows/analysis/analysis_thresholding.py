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
from darsia.presets.workflows.analysis.streaming import (
    _to_bgr_array,
    publish_stream_images,
)
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


def _overlay_layer(
    base_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    fill: tuple[int, int, int],
    stroke: tuple[int, int, int],
    fill_alpha: float,
    stroke_width: int,
) -> np.ndarray:
    overlay = base_bgr.copy()
    mask_u8 = (mask > 0).astype(np.uint8)

    # Fill on active mask region.
    fill_bgr = np.array(_rgb_to_bgr(fill), dtype=np.uint8)
    color_layer = np.zeros_like(overlay, dtype=np.uint8)
    color_layer[mask_u8.astype(bool)] = fill_bgr
    if fill_alpha > 0.0:
        overlay = cv2.addWeighted(color_layer, fill_alpha, overlay, 1.0, 0.0)

    # Stroke around contours.
    if stroke_width > 0:
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            overlay,
            contours,
            -1,
            _rgb_to_bgr(stroke),
            thickness=stroke_width,
            lineType=cv2.LINE_AA,
        )

    return overlay


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

    layer_names = list(thresholding_config.layers.keys())
    has_jpg = "jpg" in thresholding_config.formats
    has_npz = "npz" in thresholding_config.formats

    jpg_folder = thresholding_config.folder / "jpg"
    npz_folder = thresholding_config.folder / "npz"
    if has_jpg:
        for layer_name in layer_names:
            (jpg_folder / layer_name).mkdir(parents=True, exist_ok=True)
        (jpg_folder / "all").mkdir(parents=True, exist_ok=True)
    if has_npz:
        for layer_name in layer_names:
            (npz_folder / layer_name).mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        img = fluidflower.read_image(path)
        result = color_to_mass_analysis(img)
        mode_images = _extract_mode_images(result)
        stream_payload: dict[str, Any] = {"thresholding_source_image": img}
        img_bgr = _to_bgr_array(img)
        master_preview = img_bgr.copy()

        for layer_name, layer in thresholding_config.layers.items():
            scalar = _to_scalar_array(mode_images[layer.mode])
            threshold = float(layer.threshold)
            mask = (scalar >= threshold).astype(np.uint8)

            if has_npz:
                np.savez_compressed(
                    npz_folder / layer_name / f"{path.stem}.npz",
                    mask=mask,
                    threshold=threshold,
                    mode=layer.mode,
                    layer=layer_name,
                )

            preview = _overlay_layer(
                img_bgr,
                mask,
                fill=layer.fill,
                stroke=layer.stroke,
                fill_alpha=layer.fill_alpha,
                stroke_width=layer.stroke_width,
            )
            preview = _apply_legend(
                preview,
                text=f"{layer.label} ({layer.mode} >= {threshold:g})",
                legend_config=thresholding_config.legend,
            )

            if has_jpg:
                cv2.imwrite(str(jpg_folder / layer_name / f"{path.stem}.jpg"), preview)

            stream_payload[f"thresholding_{layer_name}"] = cv2.cvtColor(
                preview, cv2.COLOR_BGR2RGB
            )
            master_preview = _overlay_layer(
                master_preview,
                mask,
                fill=layer.fill,
                stroke=layer.stroke,
                fill_alpha=layer.fill_alpha,
                stroke_width=layer.stroke_width,
            )

        master_preview = _apply_legend(
            master_preview,
            text="All layers",
            legend_config=thresholding_config.legend,
        )
        if has_jpg:
            cv2.imwrite(str(jpg_folder / "all" / f"{path.stem}.jpg"), master_preview)
        stream_payload["thresholding_all"] = cv2.cvtColor(
            master_preview, cv2.COLOR_BGR2RGB
        )

        if show:
            import matplotlib.pyplot as plt

            img.show(title=f"Image at {path.stem}", delay=True)
            for layer_name in layer_names:
                plt.figure()
                plt.title(f"Thresholding {layer_name} at {path.stem}")
                plt.imshow(stream_payload[f"thresholding_{layer_name}"])
                plt.axis("off")
            plt.figure()
            plt.title(f"Thresholding all at {path.stem}")
            plt.imshow(stream_payload["thresholding_all"])
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
