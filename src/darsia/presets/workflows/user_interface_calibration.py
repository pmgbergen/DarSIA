"""User interface to preset calibration workflows.

Available:
- Color path setup
- Color signal calibration
- Mass calibration

"""

import argparse
from pathlib import Path
import logging

from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.calibration.calibration_color_paths import (
    calibration_color_paths,
)
from darsia.presets.workflows.calibration.calibration_mass_analysis import (
    calibration_mass_analysis,
)
from darsia.presets.workflows.calibration.calibration_color_signal import (
    calibration_color_signal,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_parser_for_calibration():
    parser = argparse.ArgumentParser(description="Setup run.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Activate all calibration steps."
    )
    parser.add_argument(
        "--color-path", action="store_true", help="Calibrate color paths."
    )
    parser.add_argument(
        "--color-signal", action="store_true", help="Calibrate color signal."
    )
    parser.add_argument("--mass", action="store_true", help="Calibrate mass.")
    parser.add_argument(
        "--show", action="store_true", help="Show the labels after each step."
    )
    parser.add_argument(
        "--info", action="store_true", help="Provide help for activated flags."
    )
    return parser


def print_help_for_flags(args, parser):
    if args.info:
        if args.all:
            parser.print_help()
        if args.color_path:
            print(
                "The --color-path flag activates the setup of color paths. "
                "This involves segmenting colored images and calibrating the color paths "
                "based on the provided configuration."
            )
        if args.color_signal:
            print(
                "The --color-signal flag activates the calibration of color signal. "
                "This step calibrates the color signal based on the color paths and "
                "the provided configuration."
            )
        if args.mass:
            print(
                "The --mass flag activates the calibration of mass. "
                "This step calibrates the mass analysis based on the provided configuration."
            )


def preset_calibration(rig=Rig):
    parser = build_parser_for_calibration()
    args = parser.parse_args()

    print_help_for_flags(args, parser)

    if args.all or args.color_path:
        calibration_color_paths(rig, Path(args.config), args.show)

    if args.all or args.color_signal:
        calibration_color_signal(rig, Path(args.config), args.show)

    if args.all or args.mass:
        calibration_mass_analysis(rig, Path(args.config), args.show)
