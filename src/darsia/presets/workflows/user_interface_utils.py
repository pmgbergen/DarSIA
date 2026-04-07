"""Unified interface to the various preset workflows supported by DarSIA.

To cache raw images:
    python setup.py --download-data

"""

import argparse
import logging
from pathlib import Path

from darsia.presets.workflows.utils.calibration_bundle import (
    export_calibration_bundle,
    import_calibration_bundle,
)
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
        "--export-calibration",
        action="store_true",
        help="Export calibration artifacts to a zip bundle for reuse on another machine.",
    )
    parser.add_argument(
        "--import-calibration",
        action="store_true",
        help="Import a calibration bundle and generate a config snippet with paths.",
    )
    parser.add_argument(
        "--calibration-bundle",
        type=str,
        default=None,
        help="Path to calibration bundle zip. Used as output for export and input for import.",
    )
    parser.add_argument(
        "--calibration-target",
        type=str,
        default=None,
        help="Folder where imported calibration is extracted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow import to overwrite an existing calibration target folder.",
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
        download_data(args.config)
    if args.export_calibration:
        bundle_path = (
            Path(args.calibration_bundle) if args.calibration_bundle is not None else None
        )
        exported = export_calibration_bundle(args.config, bundle=bundle_path)
        logger.info("Calibration bundle exported to %s", exported)
    if args.import_calibration:
        if args.calibration_bundle is None:
            raise ValueError(
                "Import requires --calibration-bundle pointing to a zip file."
            )
        target = (
            Path(args.calibration_target) if args.calibration_target is not None else None
        )
        imported = import_calibration_bundle(
            args.config,
            bundle=Path(args.calibration_bundle),
            target_folder=target,
            overwrite=args.overwrite,
        )
        logger.info("Imported calibration artifacts: %s", imported)
