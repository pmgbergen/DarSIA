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
from darsia.presets.workflows.calibration.calibration_color_analysis import (
    calibration_color_analysis,
)
from darsia.presets.workflows.calibration.calibration_color_to_mass_analysis import (
    calibration_color_to_mass_analysis,
)
from darsia.presets.workflows.calibration.calibration_flash import calibration_flash

logger = logging.getLogger(__name__)


def build_parser_for_calibration():
    parser = argparse.ArgumentParser(description="Calibration run.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--ref-config",
        type=str,
        help="Path to reference/global config file.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Activate all calibration steps."
    )
    parser.add_argument(
        "--color-paths", action="store_true", help="Calibrate color paths (step 1)."
    )
    parser.add_argument(
        "--color-signal", action="store_true", help="Calibrate color signal (step 2)."
    )
    parser.add_argument(
        "--color-analysis",
        action="store_true",
        help="Calibrate color analysis (step 2).",
    )
    parser.add_argument(
        "--color-to-mass-analysis",
        action="store_true",
        help="Calibrate color to mass analysis (step 2).",
    )
    parser.add_argument("--mass", action="store_true", help="Calibrate mass (step 3).")
    parser.add_argument(
        "--flash",
        action="store_true",
        help="Calibrate flash settings (step 2/3 combined).",
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset existing calibration data."
    )
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
        if args.flash:
            print(
                "The --flash flag activates the calibration of flash settings. "
                "This step combines the calibration of color signal and mass based on the provided configuration."
            )


def preset_calibration(rig=Rig):
    parser = build_parser_for_calibration()
    args = parser.parse_args()

    print_help_for_flags(args, parser)

    if args.all or args.color_paths:
        calibration_color_paths(rig, Path(args.config), args.show)

    if args.all or args.color_to_mass_analysis:
        ref_config = Path(args.ref_config) if args.ref_config else None
        calibration_color_to_mass_analysis(
            rig,
            Path(args.config),
            ref_path=ref_config,
            reset=args.reset,
            show=args.show,
        )

    if args.all or args.color_analysis:
        calibration_color_analysis(rig, Path(args.config), args.show)

    if args.all or args.color_signal:
        calibration_color_signal(rig, Path(args.config), args.show)

    if args.all or args.mass:
        calibration_mass_analysis(rig, Path(args.config), args.show)

    if args.all or args.flash:
        calibration_flash(rig, Path(args.config), args.show)
