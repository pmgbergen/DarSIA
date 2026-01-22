"""User interface to comparative analysis workflows."""

import argparse
from pathlib import Path

import logging

from darsia.presets.workflows.comparison.comparison_events import (
    comparison_events,
)
from darsia.presets.workflows.comparison.comparison_wasserstein import (
    comparison_wasserstein,
)
from darsia.presets.workflows.rig import Rig


logger = logging.getLogger(__name__)


def build_parser_for_comparison():
    parser = argparse.ArgumentParser(description="Setup run.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path(s) to config file(s). Multiple files can be specified.",
    )
    parser.add_argument("--events", action="store_true", help="Determine events.")
    parser.add_argument(
        "--wasserstein-compute", action="store_true", help="Determine W1 over time."
    )
    parser.add_argument(
        "--wasserstein-assemble", action="store_true", help="Determine W1 over time."
    )
    parser.add_argument(
        "--wasserstein-check", action="store_true", help="Determine W1 over time."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the labels after each step.",
    )
    parser.add_argument(
        "--info", action="store_true", help="Provide help for activated flags."
    )
    return parser


def print_help_for_flags(args, parser):
    if args.info:
        import sys

        sys.exit(0)


def run_comparison(rig: type[Rig], args, **kwargs):
    # Only allow one option at a time
    assert (
        args.events
        + args.wasserstein_compute
        + args.wasserstein_assemble
        + args.wasserstein_check
    ) == 1, (
        "Exactly one of events, wasserstein_compute, wasserstein_assemble, or wasserstein_check must be True."
    )
    # Check if none is chosen
    if not (
        args.events
        or args.wasserstein_compute
        or args.wasserstein_assemble
        or args.wasserstein_check
    ):
        parser.print_help()
        sys.exit(1)

    if args.events:
        comparison_events(args.config, **kwargs)

    if args.wasserstein_compute:
        comparison_wasserstein(
            cls=Rig,
            path=args.config,
            compute=True,
        )

    if args.wasserstein_assemble:
        comparison_wasserstein(
            path=args.config,
            assemble=True,
        )

    if args.wasserstein_check:
        comparison_wasserstein(
            path=args.config,
            check=True,
        )


def preset_comparison(rig: type[Rig], **kwargs):
    parser = build_parser_for_comparison()
    args = parser.parse_args()
    print_help_for_flags(args, parser)
    run_comparison(rig, args, **kwargs)
