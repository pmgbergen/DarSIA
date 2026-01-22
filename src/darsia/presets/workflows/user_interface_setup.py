"""Unified interface to the various preset workflows supported by DarSIA.

Setup routines:
- depth map setup
- labeling setup
- rig setup

Usage (for more information run with --help and/or --info flag):
    python setup.py --all
    python setup.py --depth
    python setup.py --segmentation
    python setup.py --rig

"""

import argparse
import logging

from darsia.presets.workflows.setup.setup_rig import setup_rig
from darsia.presets.workflows.setup.setup_depth import setup_depth_map
from darsia.presets.workflows.setup.setup_labeling import segment_colored_image
from darsia.presets.workflows.setup.setup_facies import setup_facies
from darsia.presets.workflows.rig import Rig

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
    parser.add_argument("--rig", action="store_true", help="Activate setup of rig.")
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
