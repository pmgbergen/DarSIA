"""Template for thresholding analysis."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import darsia
from darsia.presets.workflows.analysis.analysis_context import (
    AnalysisContext,
    infer_require_color_to_mass_from_config,
    prepare_analysis_context,
)
from darsia.presets.workflows.analysis.image_export_formats import ImageExportFormats
from darsia.presets.workflows.analysis.progress import (
    AnalysisProgressEvent,
    publish_image_progress,
)
from darsia.presets.workflows.analysis.scalar_products import (
    analysis_scalar_products,
    requires_rescaled_modes,
)
from darsia.presets.workflows.analysis.streaming import (
    _to_bgr_array,
    publish_stream_images,
)
from darsia.presets.workflows.config.analysis import AnalysisThresholdingConfig
from darsia.presets.workflows.mode_resolution import (
    mode_requires_color_to_mass,
    resolve_mode_image,
)
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
    progress_callback: Callable[[AnalysisProgressEvent], None] | None = None,
) -> None:
    """Thresholding analysis using pre-prepared context."""
    assert ctx.config.data is not None
    assert ctx.config.analysis is not None

    config = ctx.config
    experiment = ctx.experiment
    fluidflower = ctx.fluidflower
    image_paths = ctx.image_paths

    if config.analysis.thresholding is None:
        config.analysis.thresholding = AnalysisThresholdingConfig().load(
            sec={"thresholding": {}},
            results=config.data.results,
        )

    thresholding_config = config.analysis.thresholding
    thresholding_config.folder.mkdir(parents=True, exist_ok=True)
    requires_color_to_mass = any(
        mode_requires_color_to_mass(layer.mode)
        for layer in thresholding_config.layers.values()
    )
    if requires_color_to_mass and ctx.color_to_mass_analysis is None:
        raise ValueError(
            "Thresholding config uses color-to-mass modes, but color-to-mass analysis "
            "is not initialized."
        )
    color_to_mass_analysis = ctx.color_to_mass_analysis

    layer_names = list(thresholding_config.layers.keys())
    requested_modes = {layer.mode for layer in thresholding_config.layers.values()}
    need_rescaled = requires_rescaled_modes(requested_modes)
    exporter = ImageExportFormats.from_analysis_config(
        config,
        fallback_formats=thresholding_config.formats,
    )

    step_started_at = time.monotonic()
    image_total = len(image_paths)
    for image_index, path in enumerate(image_paths, start=1):
        image_started_at = time.monotonic()
        img = fluidflower.read_image(path)
        result = color_to_mass_analysis(img) if requires_color_to_mass else None
        mode_images = {}
        if result is not None:
            scalar_kwargs = {}
            if need_rescaled:
                co2_mass_analysis = None
                if hasattr(color_to_mass_analysis, "co2_mass_analysis"):
                    co2_mass_analysis = color_to_mass_analysis.co2_mass_analysis
                scalar_kwargs = {
                    "geometry": fluidflower.geometry,
                    "injection_protocol": experiment.injection_protocol,
                    "co2_mass_analysis": co2_mass_analysis,
                    "date": img.date,
                }
            mode_images, _ = analysis_scalar_products(
                mass_analysis_result=result,
                requested_modes=requested_modes,
                expert_knowledge_adapter=ctx.expert_knowledge_adapter,
                **scalar_kwargs,
            )
        stream_payload: dict[str, Any] = {"thresholding_source_image": img}
        img_bgr = _to_bgr_array(img)
        master_preview = img_bgr.copy()

        for layer_name, layer in thresholding_config.layers.items():
            mode_image = resolve_mode_image(
                layer.mode,
                img,
                mass_analysis_result=result,
                colorrange_config=getattr(config, "colorrange", None),
                scalar_products=mode_images,
            )
            scalar = _to_scalar_array(mode_image)
            threshold_min = layer.threshold_min
            threshold_max = layer.threshold_max
            if threshold_min is not None and threshold_max is not None:
                mask = ((scalar >= threshold_min) & (scalar <= threshold_max)).astype(
                    np.uint8
                )
            elif threshold_min is not None:
                mask = (scalar >= threshold_min).astype(np.uint8)
            elif threshold_max is not None:
                mask = (scalar <= threshold_max).astype(np.uint8)

            mask_meta = {}
            if hasattr(mode_image, "metadata"):
                mask_meta = mode_image.metadata()
            elif hasattr(img, "metadata"):
                mask_meta = img.metadata()
            mask_image = darsia.ScalarImage(mask.astype(np.float32), **mask_meta)
            exporter.export_image(
                mask_image,
                thresholding_config.folder,
                path.stem,
                supported_types={"npz", "npy", "csv"},
                subfolder=layer_name,
            )

            preview = _overlay_layer(
                img_bgr,
                mask,
                fill=layer.fill,
                stroke=layer.stroke,
                fill_alpha=layer.fill_alpha,
                stroke_width=layer.stroke_width,
            )
            if threshold_min is not None and threshold_max is not None:
                legend_text = f"{layer.label} ({layer.mode} in"
                legend_text += f" [{threshold_min:g}, {threshold_max:g}])"
            elif threshold_min is not None:
                legend_text = f"{layer.label} ({layer.mode} >= {threshold_min:g})"
            elif threshold_max is not None:
                legend_text = f"{layer.label} ({layer.mode} <= {threshold_max:g})"

            preview = _apply_legend(
                preview,
                text=legend_text,
                legend_config=thresholding_config.legend,
            )

            preview_image = darsia.OpticalImage(
                cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), color_space="RGB"
            )
            exporter.export_image(
                preview_image,
                thresholding_config.folder,
                path.stem,
                supported_types={"jpg", "png"},
                subfolder=layer_name,
            )

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
        master_preview_image = darsia.OpticalImage(
            cv2.cvtColor(master_preview, cv2.COLOR_BGR2RGB), color_space="RGB"
        )
        exporter.export_image(
            master_preview_image,
            thresholding_config.folder,
            path.stem,
            supported_types={"jpg", "png"},
            subfolder="all",
        )
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
        publish_image_progress(
            progress_callback,
            step="thresholding",
            image_path=str(path.resolve()),
            image_index=image_index,
            image_total=image_total,
            image_duration_s=time.monotonic() - image_started_at,
            step_elapsed_s=time.monotonic() - step_started_at,
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
        require_color_to_mass=infer_require_color_to_mass_from_config(
            path,
            include_thresholding=True,
        ),
    )
    analysis_thresholding_from_context(ctx, show=show, stream_callback=stream_callback)
