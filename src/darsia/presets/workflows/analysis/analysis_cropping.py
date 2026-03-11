"""Template for cropping/reading images."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import darsia
from darsia.presets.workflows.analysis.analysis_context import (
    AnalysisContext,
    prepare_analysis_context,
)
from darsia.presets.workflows.rig import Rig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def analysis_cropping_from_context(
    ctx: AnalysisContext,
    show: bool = False,
    save_jpg: bool = False,
    save_npz: bool = False,
    interactive: bool | None = None,
) -> None:
    """Cropping analysis using pre-prepared context.

    Args:
        ctx: Pre-prepared analysis context.
        show: Whether to show the images.
        save_jpg: Whether to save the images as JPG.
        save_npz: Whether to save the images as NPZ.
        interactive: Whether to prompt for output options when none are set.
            Defaults to ``True`` when stdin is a TTY, ``False`` otherwise.
            When ``False`` and no output option is set, a ``ValueError`` is raised.

    Raises:
        ValueError: If no output options are set and not in interactive mode.

    Note:
        The interactive prompt is only suitable for direct terminal use. In
        non-interactive contexts (CI pipelines, notebooks, tests, redirected
        stdin) ``interactive`` should be left as ``None`` (auto-detected) or
        explicitly set to ``False`` to receive a clear error instead of a hang
        or ``EOFError``. Similar considerations apply to other workflow
        functions that use ``input()``; see ``setup_rig.delete_rig`` and
        ``heterogeneous_color_analysis`` for examples.

    """
    # Default: interactive only when stdin is a real terminal
    if interactive is None:
        interactive = sys.stdin.isatty()
    # Sanity checks
    assert ctx.config.data is not None

    # Require that output options are selected
    if not (show or save_jpg or save_npz):
        if not interactive:
            raise ValueError(
                "No output options selected and not running interactively. "
                "Set at least one of show, save_jpg, or save_npz to True."
            )
        logger.warning(
            "\033[33mNo output options selected. At least one output option must be chosen "
            "before proceeding. Please select one or more options below.\033[0m"
        )
        try:
            user_input = input(
                "Enter a number to select output options (1=show, 2=save_jpg, 3=save_npz, e.g. 13 for show and save_npz): "
            )
        except EOFError:
            raise ValueError(
                "No output options selected and stdin is not available for interactive input. "
                "Set at least one of show, save_jpg, or save_npz to True."
            )
        show = "1" in user_input
        save_jpg = "2" in user_input
        save_npz = "3" in user_input

    # ! ---- CROPPING ----
    plot_folder = ctx.config.data.results / "cropped_images"
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

    print("Done. Analysis.")


def analysis_cropping(
    cls: type[Rig],
    path: Path | list[Path],
    show: bool = False,
    save_jpg: bool = False,
    save_npz: bool = False,
    all: bool = False,
    interactive: bool | None = None,
) -> None:
    """Cropping analysis (standalone entry point).

    Note: If no options are set, the images are only read and no output is saved.

    Args:
        cls: FluidFlower rig class.
        path: Path or list of Paths to the images.
        show: Whether to show the images.
        save_jpg: Whether to save the images as JPG.
        save_npz: Whether to save the images as NPZ.
        all: Whether to use all images or only the ones specified in the config.
        interactive: Whether to prompt for output options when none are set.
            Defaults to ``True`` when stdin is a TTY, ``False`` otherwise.

    """
    ctx = prepare_analysis_context(
        cls=cls,
        path=path,
        all=all,
        use_facies=False,
        require_color_to_mass=False,
    )
    analysis_cropping_from_context(
        ctx,
        show=show,
        save_jpg=save_jpg,
        save_npz=save_npz,
        interactive=interactive,
    )
