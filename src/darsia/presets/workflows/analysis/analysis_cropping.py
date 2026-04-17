"""Template for cropping/reading images."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import darsia
from darsia.presets.workflows.analysis.analysis_context import (
    AnalysisContext,
    prepare_analysis_context,
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
) -> None:
    """Cropping analysis using pre-prepared context.

    Args:
        ctx: Pre-prepared analysis context.
        show: Whether to show the images.

    """
    # Sanity checks
    assert ctx.config.data is not None
    assert ctx.config.analysis is not None

    cropping_config = ctx.config.analysis.cropping
    formats = cropping_config.formats if cropping_config is not None else ["jpg"]

    save_jpg = "jpg" in formats
    save_npz = "npz" in formats

    if not (show or save_jpg or save_npz):
        raise ValueError(
            "Cropping requires output selection via [analysis.cropping].formats, "
            'e.g. ["jpg"], ["npz"], or ["jpg", "npz"], or --show.'
        )

    # ! ---- CROPPING ----
    plot_folder = ctx.config.data.results / "cropping"
    plot_folder.mkdir(parents=True, exist_ok=True)

    for path in ctx.image_paths:
        # Read image
        img = ctx.fluidflower.read_image(path)

        # Convert image to darsia.OpticalImage
        img = darsia.OpticalImage(img.img, **img.metadata())

        if show:
            img.show()

        if save_npz:
            img.save(plot_folder / f"{path.stem}.npz")

        if save_jpg:
            img = img.img_as(np.uint8)
            img.original_dtype = np.uint8  # Hack to allow plotting
            img.write(plot_folder / f"{path.stem}.jpg", quality=50)

        publish_stream_images(
            stream_callback=stream_callback,
            image_payload={"cropping": img},
            logger=logger,
            error_message=f"Failed to stream cropped preview for image '{path}'.",
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
