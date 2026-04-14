"""User interface to standardized analysis workflows."""

import argparse
import logging
from collections.abc import Callable

from darsia.presets.workflows.analysis.analysis_context import prepare_analysis_context
from darsia.presets.workflows.analysis.analysis_cropping import (
    analysis_cropping_from_context,
)
from darsia.presets.workflows.analysis.analysis_fingers import (
    analysis_fingers_from_context,
)
from darsia.presets.workflows.analysis.analysis_mass import analysis_mass_from_context
from darsia.presets.workflows.analysis.analysis_segmentation import (
    analysis_segmentation_from_context,
)
from darsia.presets.workflows.analysis.analysis_thresholding import (
    analysis_thresholding_from_context,
)
from darsia.presets.workflows.analysis.analysis_volume import (
    analysis_volume_from_context,
)
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.mode_resolution import mode_requires_color_to_mass
from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)


def _collect_requested_modes(config: FluidFlowerConfig, args) -> list[str]:
    if config.analysis is None:
        return []

    modes: list[str] = []
    if args.segmentation and config.analysis.segmentation is not None:
        segmentation_config = config.analysis.segmentation.config
        if isinstance(segmentation_config, dict):
            modes.extend(
                cfg.mode for cfg in segmentation_config.values() if cfg.mode is not None
            )
        elif segmentation_config.mode is not None:
            modes.append(segmentation_config.mode)

    if args.fingers and config.analysis.fingers is not None:
        fingers_config = config.analysis.fingers.config
        if isinstance(fingers_config, dict):
            modes.extend(
                cfg.mode for cfg in fingers_config.values() if cfg.mode is not None
            )
        elif fingers_config.mode is not None:
            modes.append(fingers_config.mode)

    if args.thresholding and config.analysis.thresholding is not None:
        modes.extend(layer.mode for layer in config.analysis.thresholding.layers.values())

    return modes


def _infer_require_color_to_mass(args) -> bool:
    # Always needed for explicit mass/volume analysis.
    if args.mass or args.volume:
        return True

    # No potentially dependent analyses requested.
    if not (args.segmentation or args.fingers or args.thresholding):
        return False

    config = FluidFlowerConfig(args.config, require_results=True, require_data=True)
    modes = _collect_requested_modes(config, args)
    if len(modes) == 0:
        # Keep previous behavior for incomplete configs.
        return True
    return any(mode_requires_color_to_mass(mode) for mode in modes)


def build_parser_for_analysis():
    parser = argparse.ArgumentParser(description="Setup run.")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to config file(s). Multiple files can be specified.",
    )
    parser.add_argument(
        "--cropping", action="store_true", help="Apply cropping to analysis images."
    )
    parser.add_argument(
        "--segmentation", action="store_true", help="Perform segmentation analysis."
    )
    parser.add_argument(
        "--fingers", action="store_true", help="Perform finger analysis."
    )
    parser.add_argument(
        "--mass", action="store_true", help="Perform color to mass analysis."
    )
    parser.add_argument(
        "--volume", action="store_true", help="Perform color to volume analysis."
    )
    parser.add_argument(
        "--thresholding", action="store_true", help="Perform thresholding analysis."
    )
    parser.add_argument(
        "--all", action="store_true", help="Perform analysis on entire dataset."
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
        if args.all:
            parser.print_help()
        if args.cropping:
            print(
                "Cropping analysis: Applies cropping to specified images "
                "based on configuration."
            )
        if args.segmentation:
            print(
                "Segmentation analysis: Performs segmentation on images "
                "according to configuration."
            )
        print("To run the analysis, remove the '--info' flag.")
        import sys

        sys.exit(0)


def run_analysis(
    rig_cls: type[Rig],
    args,
    stream_callback: Callable[[dict[str, bytes] | None], None] | None = None,
    **kwargs,
):
    if not (
        args.cropping
        or args.mass
        or args.volume
        or args.segmentation
        or args.fingers
        or args.thresholding
    ):
        raise ValueError(
            """No analysis type specified. Please select at least one analysis."""
            """Choose from --cropping, --mass, --volume, --segmentation, """
            """--fingers, --thresholding."""
        )

    # Determine if we need color-to-mass analysis (expensive initialization)
    require_color_to_mass = _infer_require_color_to_mass(args)

    # Prepare shared context once for all analyses
    ctx = prepare_analysis_context(
        cls=rig_cls,
        path=args.config,
        all=args.all,
        require_color_to_mass=require_color_to_mass,
    )

    # Run requested analyses using shared context
    if args.cropping:
        analysis_cropping_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
        )

    if args.mass:
        analysis_mass_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
        )

    if args.volume:
        analysis_volume_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
        )

    if args.segmentation:
        analysis_segmentation_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
        )

    if args.fingers:
        analysis_fingers_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
        )

    if args.thresholding:
        analysis_thresholding_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
        )


def preset_analysis(rig_cls: type[Rig], **kwargs):
    parser = build_parser_for_analysis()
    args = parser.parse_args()
    print_help_for_flags(args, parser)
    run_analysis(rig_cls, args, **kwargs)
