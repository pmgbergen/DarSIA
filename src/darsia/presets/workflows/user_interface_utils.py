"""Unified interface to the various preset workflows supported by DarSIA.

To cache raw images:
    python setup.py --download-data

"""

import argparse
import logging
from pathlib import Path

from darsia.presets.workflows.utils.utils_download import download_data

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
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Activate caching of raw images. This can be useful if you want to run the setup"
        " on a different machine than the one where the raw data is stored.",
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


def preset_utils():
    parser = build_parser_for_setup()
    args = parser.parse_args()

    if args.download_data:
        logger.info("Downloading raw data...")
        download_data([Path(p) for p in args.config])
