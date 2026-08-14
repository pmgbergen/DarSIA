"""Template for cropping/reading images."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import darsia
from darsia.presets.workflows.analysis.analysis_context import (
    AnalysisContext,
    prepare_analysis_context,
)
from darsia.presets.workflows.analysis.image_export_formats import ImageExportFormats
from darsia.presets.workflows.analysis.progress import (
    AnalysisProgressEvent,
    publish_image_progress,
)
from darsia.presets.workflows.analysis.streaming import publish_stream_images
from darsia.presets.workflows.rig import Rig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def analysis_cropping_from_context(
    ctx: AnalysisContext,
    show: bool = False,
    stream_callback: Callable[[dict[str, bytes] | None], None] | None = None,
    progress_callback: Callable[[AnalysisProgressEvent], None] | None = None,
) -> None:
    """Cropping analysis using pre-prepared context.

    Args:
        ctx: Pre-prepared analysis context.
        show: Whether to show the images.

    """
    # Sanity checks
    assert ctx.config.data is not None
    assert ctx.config.analysis is not None

    fluidflower = ctx.fluidflower
    image_paths = ctx.image_paths

    cropping_config = ctx.config.analysis.cropping
    legacy_formats = cropping_config.formats if cropping_config is not None else ["jpg"]
    exporter = ImageExportFormats.from_analysis_config(
        ctx.config, fallback_formats=legacy_formats
    )

    if not (show or len(exporter.formats) > 0):
        raise ValueError(
            "Cropping requires output selection via [analysis.cropping].formats, "
            'e.g. ["jpg"], ["npz"], or ["jpg", "npz"], or --show.'
        )

    # ! ---- CROPPING ----
    plot_folder = ctx.config.data.results / "cropping"
    plot_folder.mkdir(parents=True, exist_ok=True)

    # Loop over images
    step_started_at = time.monotonic()
    image_total = len(image_paths)

    # Random shuffle for faster preview of analysis.
    if ctx.config.analysis.random_traverse:
        random.shuffle(image_paths)

    for image_index, path in enumerate(image_paths, start=1):
        image_started_at = time.monotonic()
        # Read image
        try:
            img = fluidflower.read_image(path)
        except Exception as e:
            logger.error(f"Failed to read image '{path}': {e}")
            continue

        # Convert image to darsia.OpticalImage
        img = darsia.OpticalImage(img.img, **img.metadata())

        if show:
            img.show()

        export_image = img.img_as(np.uint8)
        export_image.original_dtype = np.uint8  # Hack to allow plotting
        exporter.export_image(export_image, plot_folder, path.stem)

        publish_stream_images(
            stream_callback=stream_callback,
            image_payload={"cropping": img},
            logger=logger,
            error_message=f"Failed to stream cropped preview for image '{path}'.",
        )
        publish_image_progress(
            progress_callback,
            step="cropping",
            image_path=str(path.resolve()),
            image_index=image_index,
            image_total=image_total,
            image_duration_s=time.monotonic() - image_started_at,
            step_elapsed_s=time.monotonic() - step_started_at,
        )

    print("Done. Analysis.")


def analysis_cropping(
    cls: type[Rig],
    path: Path | list[Path],
    show: bool = False,
    all: bool = False,
    stream_callback: Callable[[dict[str, bytes] | None], None] | None = None,
) -> None:
    """Cropping analysis (standalone entry point).

    Note: Output is configured via `[analysis.cropping].formats`.

    Args:
        cls: Rig class.
        path: Path or list of Paths to the images.
        show: Whether to show the images.
        all: Whether to use all images or only the ones specified in the config.

    """
    ctx = prepare_analysis_context(
        cls=cls,
        path=path,
        all=all,
        require_color_to_mass=False,
    )

    analysis_cropping_from_context(
        ctx,
        show=show,
        stream_callback=stream_callback,
    )
