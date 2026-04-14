"""User interface to standardized analysis workflows."""

import argparse
import logging
from collections.abc import Callable

from darsia.presets.workflows.analysis.analysis_context import (
    infer_require_color_to_mass_from_config,
    prepare_analysis_context,
)
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
    require_color_to_mass = infer_require_color_to_mass_from_config(
        args.config,
        include_segmentation=args.segmentation,
        include_fingers=args.fingers,
        include_thresholding=args.thresholding,
        include_mass=args.mass,
        include_volume=args.volume,
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
        analysis_cropping_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
        )

    if args.mass:
        analysis_mass_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
        )

    if args.volume:
        analysis_volume_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
        )

    if args.segmentation:
        analysis_segmentation_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
        )

    if args.fingers:
        analysis_fingers_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
        )

    if args.thresholding:
        analysis_thresholding_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
        )


def preset_analysis(rig_cls: type[Rig], **kwargs):
    parser = build_parser_for_analysis()
    args = parser.parse_args()
    print_help_for_flags(args, parser)
    run_analysis(rig_cls, args, **kwargs)
