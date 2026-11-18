"""User interface to preset calibration workflows.

Available:
- Color path setup/calibration
- Mass calibration

"""

import argparse
import logging
import sys
from pathlib import Path

import darsia
from darsia.presets.workflows.analysis.analysis_context import prepare_analysis_context
from darsia.presets.workflows.calibration.calibration_color_paths import (
    calibration_color_paths_from_context,
    delete_calibration,
)
from darsia.presets.workflows.calibration.calibration_color_to_mass_analysis import (
    calibration_color_to_mass_analysis_from_context,
)
from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def build_parser_for_calibration():
    parser = argparse.ArgumentParser(description="Calibration run.")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to config file(s). Multiple files can be specified.",
    )
    parser.add_argument(
        "--ref-config",
        type=str,
        help="Path to reference/global config file.",
    )
    parser.add_argument(
        "--color-embedding", action="store_true", help="Calibrate color embedding."
    )
    parser.add_argument("--mass", action="store_true", help="Calibrate mass.")
    parser.add_argument(
        "--default-mass", action="store_true", help="Calibrate default mass."
    )
    parser.add_argument("--volume", action="store_true", help="Calibrate volume.")
    parser.add_argument(
        "--reset", action="store_true", help="Reset existing calibration data."
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete existing calibration files and cached images. Use with caution.",
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
        if args.color_embedding:
            print(
                "The --color-embedding flag activates calibration of the configured "
                "color embedding. "
                "This involves segmenting colored images and calibrating the color path "
                "based on the provided configuration."
            )
        if args.mass:
            print("The --mass flag activates the calibration of mass. ")
        if args.volume:
            print("The --volume flag activates the calibration of volume. ")


def report_timings(config, /, **info) -> None:
    """Log the timed stages of the run and append them to the timing history.

    The history is `<results>/log/timings.jsonl` (one line per run); it is skipped
    without a results folder. Reporting never fails a run.

    """
    try:
        logger.info("Timings of the run:\n%s", darsia.timings.report())
        results = getattr(getattr(config, "data", None), "results", None)
        if results is not None and Path(results).is_dir():
            darsia.timings.write(Path(results) / "log" / "timings.jsonl", **info)
    except Exception as error:
        logger.debug("Could not report the timings: %s", error)


def preset_calibration(rig=Rig, **kwargs):
    parser = build_parser_for_calibration()
    args = parser.parse_args()

    print_help_for_flags(args, parser)

    if args.delete:
        delete_calibration(args.config)
        return

    # Prepare shared context once for all analyses
    ctx = prepare_analysis_context(
        cls=rig,
        path=args.config,
        all=False,
        require_color_to_mass=False,
        section="calibration",
        require_results=False,
    )

    if args.color_embedding:
        print("Running color embedding calibration...", flush=True)
        try:
            calibration_color_paths_from_context(
                ctx,
                args.show,
            )
        finally:
            report_timings(
                ctx.config, workflow="calibration colour paths", config=args.config
            )

    if args.mass or args.default_mass:
        print("Running mass calibration...", flush=True)
        ref_config = Path(args.ref_config) if args.ref_config else None
        try:
            calibration_color_to_mass_analysis_from_context(
                ctx,
                ref_path=ref_config,
                reset=args.reset,
                show=args.show,
                default=args.default_mass,
            )
        finally:
            report_timings(ctx.config, workflow="calibration mass", config=args.config)


if __name__ == "__main__":
    preset_calibration()
