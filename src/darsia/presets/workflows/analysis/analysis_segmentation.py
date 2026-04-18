"""Template for segmentation analysis."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from darsia.presets.workflows.analysis.analysis_context import (
    AnalysisContext,
    infer_require_color_to_mass_from_config,
    prepare_analysis_context,
)
from darsia.presets.workflows.analysis.progress import (
    AnalysisProgressEvent,
    publish_image_progress,
)
from darsia.presets.workflows.analysis.scalar_products import (
    analysis_scalar_products,
    requires_rescaled_modes,
)
from darsia.presets.workflows.analysis.streaming import (
    publish_stream_images,
)
from darsia.presets.workflows.mode_resolution import mode_requires_color_to_mass
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.segmentation_contours import SegmentationContours

logger = logging.getLogger(__name__)


def analysis_segmentation_from_context(
    ctx: AnalysisContext,
    show: bool = False,
    stream_callback: Callable[[dict[str, bytes] | None], None] | None = None,
    progress_callback: Callable[[AnalysisProgressEvent], None] | None = None,
) -> None:
    """Segmentation analysis using pre-prepared context.

    Args:
        ctx: Pre-prepared analysis context with color_to_mass_analysis initialized.
        show: Whether to show the images.
        stream_callback: Optional callback receiving streamed segmentation previews.

    """
    assert ctx.config.analysis is not None
    assert ctx.config.analysis.segmentation is not None

    fluidflower = ctx.fluidflower
    experiment = ctx.experiment
    image_paths = ctx.image_paths

    # Extract segmentation config (asserted not None above)
    segmentation_config = ctx.config.analysis.segmentation
    _config = segmentation_config.config
    if isinstance(_config, dict):
        modes = [cfg.mode for cfg in _config.values() if cfg.mode is not None]
    else:
        modes = [_config.mode] if _config.mode is not None else []
    requires_color_to_mass = any(mode_requires_color_to_mass(mode) for mode in modes)
    if requires_color_to_mass and ctx.color_to_mass_analysis is None:
        raise ValueError(
            "Segmentation config uses color-to-mass modes, but color-to-mass analysis "
            "is not initialized."
        )
    color_to_mass_analysis = ctx.color_to_mass_analysis

    # ! ---- CONTOUR PLOTTING ----
    segmentation_contours = SegmentationContours(segmentation_config.config)
    requested_modes = segmentation_contours.requested_modes()
    need_rescaled = requires_rescaled_modes(requested_modes)

    # Loop over images and analyze
    step_started_at = time.monotonic()
    image_total = len(image_paths)
    for image_index, path in enumerate(image_paths, start=1):
        image_started_at = time.monotonic()
        # Extract color signal and assign mass
        img = fluidflower.read_image(path)
        mass_analysis_result = (
            color_to_mass_analysis(img) if requires_color_to_mass else None
        )
        scalar_products = {}
        if mass_analysis_result is not None:
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
            scalar_products, _ = analysis_scalar_products(
                mass_analysis_result=mass_analysis_result,
                requested_modes=requested_modes,
                expert_knowledge_adapter=ctx.expert_knowledge_adapter,
                **scalar_kwargs,
            )

        # Produce contour images
        contour_image = segmentation_contours(
            img,
            scalar_products=scalar_products,
            mass_analysis_result=mass_analysis_result,
            colorrange_config=getattr(ctx.config, "colorrange", None),
        )

        if show:
            contour_image.show(
                title=f"Contours for {path.stem} | {img.time} seconds", delay=False
            )

        contour_path = segmentation_config.folder / f"{path.stem}.jpg"
        contour_image.write(contour_path, quality=80)

        publish_stream_images(
            stream_callback=stream_callback,
            image_payload={"segmentation": contour_image},
            logger=logger,
            error_message=f"Failed to stream segmentation preview for image '{path}'.",
        )
        publish_image_progress(
            progress_callback,
            step="segmentation",
            image_path=str(path.resolve()),
            image_index=image_index,
            image_total=image_total,
            image_duration_s=time.monotonic() - image_started_at,
            step_elapsed_s=time.monotonic() - step_started_at,
        )


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
        require_color_to_mass=infer_require_color_to_mass_from_config(
            path,
            include_segmentation=True,
        ),
    )
    analysis_segmentation_from_context(
        ctx,
        show=show,
        stream_callback=stream_callback,
    )
