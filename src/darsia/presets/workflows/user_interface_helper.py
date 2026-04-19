"""User interface to helper workflows."""

import argparse
import logging
from pathlib import Path

from darsia.presets.workflows.helper.helper_roi import helper_roi
from darsia.presets.workflows.helper.helper_roi_viewer import helper_roi_viewer
from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)


def build_parser_for_helper():
    parser = argparse.ArgumentParser(description="Helper run.")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to config file(s). Multiple files can be specified.",
    )
    parser.add_argument(
        "--roi",
        action="store_true",
        help="Run ROI helper workflow.",
    )
    parser.add_argument(
        "--roi-viewer",
        action="store_true",
        help="Run ROI viewer workflow.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Reserved for parity with other workflow interfaces.",
    )
    parser.add_argument(
        "--info", action="store_true", help="Provide help for activated flags."
    )
    return parser


def print_help_for_flags(args, parser):
    if args.info:
        if args.roi:
            print(
                "ROI helper: interactive ROI selection with copy-ready TOML template."
            )
        if args.roi_viewer:
            print(
                "ROI viewer: interactive ROI browsing for ROI registry entries "
                "on selected data."
            )
        if not args.roi and not args.roi_viewer:
            parser.print_help()


def run_helper(rig_cls: type[Rig], args):
    if not args.roi and not args.roi_viewer:
        raise ValueError("No helper type specified. Choose from --roi, --roi-viewer.")
    config_paths = [Path(p) for p in args.config]
    if args.roi:
        helper_roi(rig_cls, config_paths, show=args.show)
    if args.roi_viewer:
        helper_roi_viewer(rig_cls, config_paths, show=args.show)


def preset_helper(rig_cls: type[Rig], **kwargs):
    del kwargs
    parser = build_parser_for_helper()
    args = parser.parse_args()
    print_help_for_flags(args, parser)
    run_helper(rig_cls, args)
