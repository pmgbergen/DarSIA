"""User interface to standardized analysis workflows."""

import argparse
import logging
import time
from collections.abc import Callable

from darsia.presets.workflows.analysis.analysis_context import prepare_analysis_context
from darsia.presets.workflows.analysis.analysis_cropping import (
    analysis_cropping_from_context,
)
from darsia.presets.workflows.analysis.analysis_fingers import (
    analysis_fingers_from_context,
)
from darsia.presets.workflows.analysis.analysis_mass import analysis_mass_from_context
from darsia.presets.workflows.analysis.analysis_segmentation import (
    analysis_segmentation_from_context,
)
from darsia.presets.workflows.analysis.analysis_thresholding import (
    analysis_thresholding_from_context,
)
from darsia.presets.workflows.analysis.analysis_volume import (
    analysis_volume_from_context,
)
from darsia.presets.workflows.analysis.progress import (
    AnalysisProgressEvent,
    publish_step_complete,
    publish_step_start,
)
from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)


def build_parser_for_analysis():
    parser = argparse.ArgumentParser(description="Setup run.")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to config file(s). Multiple files can be specified.",
    )
    parser.add_argument(
        "--cropping", action="store_true", help="Apply cropping to analysis images."
    )
    parser.add_argument(
        "--segmentation", action="store_true", help="Perform segmentation analysis."
    )
    parser.add_argument(
        "--fingers", action="store_true", help="Perform finger analysis."
    )
    parser.add_argument(
        "--mass", action="store_true", help="Perform color to mass analysis."
    )
    parser.add_argument(
        "--volume", action="store_true", help="Perform color to volume analysis."
    )
    parser.add_argument(
        "--thresholding", action="store_true", help="Perform thresholding analysis."
    )
    parser.add_argument(
        "--all", action="store_true", help="Perform analysis on entire dataset."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the labels after each step.",
    )
    parser.add_argument(
        "--info", action="store_true", help="Provide help for activated flags."
    )
    return parser


def print_help_for_flags(args, parser):
    if args.info:
        if args.all:
            parser.print_help()
        if args.cropping:
            print(
                "Cropping analysis: Applies cropping to specified images "
                "based on configuration."
            )
        if args.segmentation:
            print(
                "Segmentation analysis: Performs segmentation on images "
                "according to configuration."
            )
        print("To run the analysis, remove the '--info' flag.")
        import sys

        sys.exit(0)


def run_analysis(
    rig_cls: type[Rig],
    args,
    stream_callback: Callable[[dict[str, bytes] | None], None] | None = None,
    progress_callback: Callable[[AnalysisProgressEvent], None] | None = None,
    **kwargs,
):
    if not (
        args.cropping
        or args.mass
        or args.volume
        or args.segmentation
        or args.fingers
        or args.thresholding
    ):
        raise ValueError(
            """No analysis type specified. Please select at least one analysis."""
            """Choose from --cropping, --mass, --volume, --segmentation, """
            """--fingers, --thresholding."""
        )

    # Determine if we need color-to-mass analysis (expensive initialization)
    require_color_to_mass = (
        args.mass
        or args.volume
        or args.segmentation
        or args.fingers
        or args.thresholding
    )

    # Prepare shared context once for all analyses
    ctx = prepare_analysis_context(
        cls=rig_cls,
        path=args.config,
        all=args.all,
        require_color_to_mass=require_color_to_mass,
    )

    # Run requested analyses using shared context
    if args.cropping:
        step_started_at = time.monotonic()
        publish_step_start(
            progress_callback, step="cropping", image_total=len(ctx.image_paths)
        )
        analysis_cropping_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
            progress_callback=progress_callback,
        )
        publish_step_complete(
            progress_callback,
            step="cropping",
            image_total=len(ctx.image_paths),
            step_elapsed_s=time.monotonic() - step_started_at,
        )

    if args.mass:
        step_started_at = time.monotonic()
        publish_step_start(
            progress_callback, step="mass", image_total=len(ctx.image_paths)
        )
        analysis_mass_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
            progress_callback=progress_callback,
        )
        publish_step_complete(
            progress_callback,
            step="mass",
            image_total=len(ctx.image_paths),
            step_elapsed_s=time.monotonic() - step_started_at,
        )

    if args.volume:
        step_started_at = time.monotonic()
        publish_step_start(
            progress_callback, step="volume", image_total=len(ctx.image_paths)
        )
        analysis_volume_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
            progress_callback=progress_callback,
        )
        publish_step_complete(
            progress_callback,
            step="volume",
            image_total=len(ctx.image_paths),
            step_elapsed_s=time.monotonic() - step_started_at,
        )

    if args.segmentation:
        step_started_at = time.monotonic()
        publish_step_start(
            progress_callback, step="segmentation", image_total=len(ctx.image_paths)
        )
        analysis_segmentation_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
            progress_callback=progress_callback,
        )
        publish_step_complete(
            progress_callback,
            step="segmentation",
            image_total=len(ctx.image_paths),
            step_elapsed_s=time.monotonic() - step_started_at,
        )

    if args.fingers:
        step_started_at = time.monotonic()
        publish_step_start(
            progress_callback, step="fingers", image_total=len(ctx.image_paths)
        )
        analysis_fingers_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
            progress_callback=progress_callback,
        )
        publish_step_complete(
            progress_callback,
            step="fingers",
            image_total=len(ctx.image_paths),
            step_elapsed_s=time.monotonic() - step_started_at,
        )

    if args.thresholding:
        step_started_at = time.monotonic()
        publish_step_start(
            progress_callback, step="thresholding", image_total=len(ctx.image_paths)
        )
        analysis_thresholding_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
            progress_callback=progress_callback,
        )
        publish_step_complete(
            progress_callback,
            step="thresholding",
            image_total=len(ctx.image_paths),
            step_elapsed_s=time.monotonic() - step_started_at,
        )


def preset_analysis(rig_cls: type[Rig], **kwargs):
    parser = build_parser_for_analysis()
    args = parser.parse_args()
    print_help_for_flags(args, parser)
    run_analysis(rig_cls, args, **kwargs)
