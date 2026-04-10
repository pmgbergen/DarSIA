"""Template for segmentation analysis."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2

from darsia.presets.workflows.analysis.analysis_context import (
    AnalysisContext,
    prepare_analysis_context,
)
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.segmentation_contours import SegmentationContours

logger = logging.getLogger(__name__)


def _encode_low_resolution_png(
    contour_image: Any,
    max_width: int = 640,
    max_height: int = 480,
) -> bytes:
    """Encode an image as low-resolution PNG bytes.

    Args:
        contour_image: Image-like object containing contour visualization.
        max_width: Maximum output width in pixels.
        max_height: Maximum output height in pixels.

    Returns:
        PNG-encoded bytes for streaming.

    """
    bgr_image = contour_image.to_trichromatic("BGR", return_image=True)
    bgr_array = bgr_image.img

    height, width = bgr_array.shape[:2]
    if width == 0 or height == 0:
        raise ValueError(
            f"Cannot encode an image with zero dimensions: width={width}, "
            f"height={height}."
        )
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1.0:
        bgr_array = cv2.resize(
            bgr_array,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    success, encoded = cv2.imencode(".png", bgr_array)
    if not success:
        raise ValueError("Failed to encode segmentation stream image.")
    return encoded.tobytes()


def analysis_segmentation_from_context(
    ctx: AnalysisContext,
    show: bool = False,
    stream_callback: Callable[[dict[str, bytes] | None], None] | None = None,
) -> None:
    """Segmentation analysis using pre-prepared context.

    Args:
        ctx: Pre-prepared analysis context with color_to_mass_analysis initialized.
        show: Whether to show the images.
        stream_callback: Optional callback receiving streamed segmentation previews.

    """
    assert ctx.config.analysis is not None
    assert ctx.config.analysis.segmentation is not None
    assert ctx.color_to_mass_analysis is not None

    fluidflower = ctx.fluidflower
    image_paths = ctx.image_paths
    color_to_mass_analysis = ctx.color_to_mass_analysis

    # Extract segmentation config (asserted not None above)
    segmentation_config = ctx.config.analysis.segmentation

    # ! ---- CONTOUR PLOTTING ----
    segmentation_contours = SegmentationContours(segmentation_config.config)

    # Loop over images and analyze
    for path in image_paths:
        # Extract color signal and assign mass
        img = fluidflower.read_image(path)
        mass_analysis_result = color_to_mass_analysis(img)

        # Produce contour images
        contour_image = segmentation_contours(
            img,
            saturation_g=mass_analysis_result.saturation_g,
            concentration_aq=mass_analysis_result.concentration_aq,
            mass=mass_analysis_result.mass,
        )

        if show:
            contour_image.show(
                title=f"Contours for {path.stem} | {img.time} seconds", delay=False
            )

        contour_path = segmentation_config.folder / f"{path.stem}.jpg"
        contour_image.write(contour_path, quality=80)

        if stream_callback is not None:
            try:
                stream_callback(
                    {"segmentation": _encode_low_resolution_png(contour_image)}
                )
            except Exception:
                logger.exception(
                    "Failed to stream segmentation preview for image '%s'.", path
                )
                try:
                    stream_callback(None)
                except Exception:
                    pass


def analysis_segmentation(
    cls: type[Rig],
    path: Path | list[Path],
    show: bool = False,
    all: bool = False,
    stream_callback: Callable[[dict[str, bytes] | None], None] | None = None,
):
    """Segmentation analysis (standalone entry point).

    Args:
        cls: Rig class.
        path: Path or list of paths to config files.
        show: Whether to show the images.
        all: Whether to use all images.
        stream_callback: Optional callback receiving streamed segmentation previews.

    """
    ctx = prepare_analysis_context(
        cls=cls,
        path=path,
        all=all,
        require_color_to_mass=True,
    )
    analysis_segmentation_from_context(
        ctx,
        show=show,
        stream_callback=stream_callback,
    )
