"""User interface to standardized analysis workflows."""

import argparse
import logging

from darsia.presets.workflows.analysis.analysis_context import prepare_analysis_context
from darsia.presets.workflows.analysis.analysis_cropping import (
    analysis_cropping_from_context,
)
from darsia.presets.workflows.analysis.analysis_segmentation import (
    analysis_segmentation_from_context,
)
from darsia.presets.workflows.analysis.analysis_mass import (
    analysis_mass_from_context,
)
from darsia.presets.workflows.analysis.analysis_volume import (
    analysis_volume_from_context,
)

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
        "--mass", action="store_true", help="Perform color to mass analysis."
    )
    parser.add_argument(
        "--volume", action="store_true", help="Perform color to volume analysis."
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
        "--save-jpg",
        action="store_true",
        help="Save output figures as JPG.",
    )
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Save output figures as NPZ.",
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


def run_analysis(rig, args, **kwargs):
    if not (args.cropping or args.mass or args.volume or args.segmentation):
        raise ValueError(
            """No analysis type specified. Please select at least one analysis."""
            """Choose from --cropping, --mass, --volume, or --segmentation."""
        )

    # Determine if we need color-to-mass analysis (expensive initialization)
    require_color_to_mass = args.mass or args.volume or args.segmentation

    # Determine if we need facies (only for mass/volume/segmentation)
    use_facies = require_color_to_mass

    # Prepare shared context once for all analyses
    ctx = prepare_analysis_context(
        cls=rig,
        path=args.config,
        all=args.all,
        use_facies=use_facies,
        require_color_to_mass=require_color_to_mass,
    )

    # Run requested analyses using shared context
    if args.cropping:
        analysis_cropping_from_context(
            ctx,
            show=args.show,
            save_jpg=args.save_jpg,
            save_npz=args.save_npz,
        )

    if args.mass:
        analysis_mass_from_context(
            ctx,
            show=args.show,
        )

    if args.volume:
        analysis_volume_from_context(
            ctx,
            show=args.show,
        )

    if args.segmentation:
        analysis_segmentation_from_context(
            ctx,
            show=args.show,
        )


def preset_analysis(rig, **kwargs):
    parser = build_parser_for_analysis()
    args = parser.parse_args()
    print_help_for_flags(args, parser)
    run_analysis(rig, args, **kwargs)
