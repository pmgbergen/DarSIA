"""Batch analysis for cropping images."""

import argparse
from pathlib import Path

import logging

from darsia.presets.workflows.analysis.analysis_cropping import analysis_cropping
from darsia.presets.workflows.analysis.analysis_color_signal import (
    analysis_color_signal,
)
from darsia.presets.workflows.analysis.analysis_segmentation import (
    analysis_segmentation,
)
from darsia.presets.workflows.analysis.analysis_mass import (
    analysis_mass,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_parser_for_analysis():
    parser = argparse.ArgumentParser(description="Setup run.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Activate all analysis steps."
    )
    parser.add_argument(
        "--cropping", action="store_true", help="Apply cropping to analysis images."
    )
    parser.add_argument(
        "--color-signal", action="store_true", help="Perform signal analysis."
    )
    parser.add_argument(
        "--segmentation", action="store_true", help="Perform segmentation analysis."
    )
    parser.add_argument("--mass", action="store_true", help="Perform mass analysis.")
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
        if args.color_signal:
            print(
                "Color signal analysis: Analyzes color signals in the images as per configuration."
            )
        if args.segmentation:
            print(
                "Segmentation analysis: Performs segmentation on images according to configuration."
            )
        if args.mass:
            print(
                "Mass analysis: Conducts mass-related analysis based on the provided configuration."
            )
        print("To run the analysis, remove the '--info' flag.")
        import sys

        sys.exit(0)


def run_analysis(rig, args):
    if args.all or args.cropping:
        analysis_cropping(
            rig,
            Path(args.config),
            args.show,
            args.save_jpg,
            args.save_npz,
        )

    if args.all or args.color_signal:
        analysis_color_signal(
            rig,
            Path(args.config),
            args.show,
            args.save_jpg,
            args.save_npz,
        )

    if args.all or args.segmentation:
        analysis_segmentation(
            rig,
            Path(args.config),
            args.show,
            args.save_jpg,
            args.save_npz,
        )

    if args.all or args.mass:
        analysis_mass(
            rig,
            Path(args.config),
            args.show,
            args.save_jpg,
            args.save_npz,
        )


def preset_analysis(rig):
    parser = build_parser_for_analysis()
    args = parser.parse_args()
    print_help_for_flags(args, parser)
    run_analysis(rig, args)
