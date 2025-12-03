"""User interface to standardized analysis workflows."""

import argparse

import logging

from darsia.presets.workflows.analysis.analysis_cropping import analysis_cropping
from darsia.presets.workflows.analysis.analysis_segmentation import (
    analysis_segmentation,
)
from darsia.presets.workflows.analysis.analysis_color_to_mass import (
    analysis_color_to_mass,
)
from darsia.presets.workflows.analysis.analysis_volume import analysis_volume

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
                "Cropping analysis: Applies cropping to specified images based on configuration."
            )
        if args.segmentation:
            print(
                "Segmentation analysis: Performs segmentation on images according to configuration."
            )
        print("To run the analysis, remove the '--info' flag.")
        import sys

        sys.exit(0)


def run_analysis(rig, args, **kwargs):
    if args.cropping:
        analysis_cropping(
            rig,
            args.config,
            args.show,
            args.save_jpg,
            args.save_npz,
            args.all,
        )

    if args.mass:
        analysis_color_to_mass(
            rig,
            args.config,
            show=args.show,
            rois=kwargs.get("rois"),
            rois_and_labels=kwargs.get("rois_and_labels"),
            all=args.all,
        )

    if args.volume:
        analysis_volume(
            rig,
            args.config,
            rois=kwargs.get("rois"),
            show=args.show,
            all=args.all,
        )

    if args.segmentation:
        analysis_segmentation(
            rig,
            args.config,
            show=args.show,
            all=args.all,
        )


def preset_analysis(rig, **kwargs):
    parser = build_parser_for_analysis()
    args = parser.parse_args()
    print_help_for_flags(args, parser)
    run_analysis(rig, args, **kwargs)
