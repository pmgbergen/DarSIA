"""User interface to comparative analysis workflows."""

import argparse
from pathlib import Path

import logging

from darsia.presets.workflows.comparison.comparison_events import (
    comparison_events,
)

logger = logging.getLogger(__name__)


def build_parser_for_comparison():
    parser = argparse.ArgumentParser(description="Setup run.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to config file.",
    )
    parser.add_argument("--events", action="store_true", help="Determine events.")
    parser.add_argument(
        "--wasserstein", action="store_true", help="Determine W1 over time."
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


def run_comparison(args, **kwargs):
    if args.events:
        comparison_events(Path(args.config), args.show, **kwargs)


def preset_comparison(**kwargs):
    parser = build_parser_for_comparison()
    args = parser.parse_args()
    print_help_for_flags(args, parser)
    run_comparison(args, **kwargs)
