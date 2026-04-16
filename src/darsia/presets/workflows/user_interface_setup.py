"""Unified interface to the various preset workflows supported by DarSIA.

Setup routines:
- depth map setup
- labeling setup
- protocol setup
- rig setup

Usage (for more information run with --help and/or --info flag):
    python setup.py --all

Advanced usage (activate specific steps):
    python setup.py --depth
    python setup.py --segmentation
    python setup.py --facies
    python setup.py --protocol
    python setup.py --rig

"""

import argparse
import logging

from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.setup.setup_depth import setup_depth_map
from darsia.presets.workflows.setup.setup_facies import setup_facies
from darsia.presets.workflows.setup.setup_labeling import segment_colored_image
from darsia.presets.workflows.setup.setup_protocols import setup_imaging_protocol
from darsia.presets.workflows.setup.setup_rig import delete_rig, setup_rig

# Set logging level
logger = logging.getLogger(__name__)


def build_parser_for_setup():
    parser = argparse.ArgumentParser(description="Setup run.")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to config file(s). Multiple files can be specified.",
    )
    parser.add_argument("--all", action="store_true", help="Activate all setup steps.")
    parser.add_argument("--depth", action="store_true", help="Activate setup of depth.")
    parser.add_argument(
        "--segmentation", action="store_true", help="Activate labeling."
    )
    parser.add_argument(
        "--facies", action="store_true", help="Activate setup of facies."
    )
    parser.add_argument(
        "--protocol",
        action="store_true",
        help="Generate imaging/injection/pressure-temperature protocol CSV files.",
    )
    parser.add_argument("--rig", action="store_true", help="Activate setup of rig.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite when generating protocol CSV files.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete existing results. Use with caution as this will delete existing results.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show intermediate results.",
    )
    parser.add_argument(
        "--info", action="store_true", help="Provide help for activated flags."
    )
    return parser


def preset_setup(rig=Rig):
    parser = build_parser_for_setup()
    args = parser.parse_args()

    if args.all or args.depth:
        setup_depth_map(args.config, key="depth", show=args.show)
    if args.all or args.segmentation:
        segment_colored_image(args.config, args.show)
    if args.all or args.facies:
        setup_facies(rig, args.config, args.show)
    if args.all or args.rig:
        setup_rig(rig, args.config, args.show)
    if args.protocol:
        setup_imaging_protocol(args.config, force=args.force, show=args.show)
    if args.delete:
        delete_rig(rig, args.config, args.show)
