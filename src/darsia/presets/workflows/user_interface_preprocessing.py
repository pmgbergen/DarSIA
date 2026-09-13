"""CLI entrypoint for preprocessing steps: protocols, depth measurements, crop
correction.

These run ahead of, and independently from, the main setup routines (depth
map, segmentation, facies, rig) covered by ``user_interface_setup``.

Usage (for more information run with --help):
    python user_interface_preprocessing.py --all

Advanced usage (activate specific steps):
    python user_interface_preprocessing.py --protocol
    python user_interface_preprocessing.py --depth-measurements
    python user_interface_preprocessing.py --crop

"""

import argparse
import logging
import sys

from darsia.presets.workflows.setup.setup_crop import setup_crop_correction
from darsia.presets.workflows.setup.setup_depth import setup_depth_measurements
from darsia.presets.workflows.setup.setup_protocols import setup_imaging_protocol

# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def build_parser_for_preprocessing():
    parser = argparse.ArgumentParser(description="Preprocessing run.")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to config file(s). Multiple files can be specified.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Activate protocol and depth-measurements setup. Excludes crop "
            "correction, which is always interactive and opt-in via --crop."
        ),
    )
    parser.add_argument(
        "--protocol",
        action="store_true",
        help="Generate imaging/injection/pressure-temperature protocol CSV files.",
    )
    parser.add_argument(
        "--depth-measurements",
        action="store_true",
        help=(
            "Generate a depth-measurements CSV from a constant value, if "
            "[depth].measurements_mode is 'constant'."
        ),
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Activate interactive setup of crop correction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite when generating protocol/depth-measurements files.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show intermediate results.",
    )
    return parser


def preset_preprocessing():
    parser = build_parser_for_preprocessing()
    args = parser.parse_args()

    if args.all or args.depth_measurements:
        print("Running depth-measurements setup...", flush=True)
        setup_depth_measurements(args.config, force=args.force, show=args.show)
    if args.all or args.protocol:
        print("Running protocol setup...", flush=True)
        setup_imaging_protocol(args.config, force=args.force, show=args.show)
    if args.crop:
        print("Running crop correction setup...", flush=True)
        setup_crop_correction(args.config, show=args.show)


if __name__ == "__main__":
    preset_preprocessing()
