"""User interface to standardized analysis workflows."""

import argparse
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

from darsia.presets.workflows.analysis.analysis_context import (
    infer_require_color_to_mass_from_config,
    prepare_analysis_context,
    select_image_paths,
)
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
from darsia.presets.workflows.analysis.progress import (
    AnalysisProgressEvent,
    encode_progress_line,
    publish_step_complete,
    publish_step_start,
)
from darsia.presets.workflows.analysis.streaming import build_file_cache_stream_callback
from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


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
        "--stream",
        action="store_true",
        help="Publish low-res preview images to --stream-cache-dir as they're produced.",
    )
    parser.add_argument(
        "--stream-cache-dir",
        type=str,
        default=None,
        help="Directory to write streamed preview PNGs to. Required with --stream.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Emit per-image progress events (step/image count) as stdout lines.",
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
    progress_callback: Callable[[AnalysisProgressEvent], None] | None = None,
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
    require_color_to_mass = infer_require_color_to_mass_from_config(
        args.config,
        include_segmentation=args.segmentation,
        include_fingers=args.fingers,
        include_thresholding=args.thresholding,
        include_mass=args.mass,
        include_volume=args.volume,
    )

    # Prepare shared context once for all analyses
    ctx = prepare_analysis_context(
        cls=rig_cls,
        path=args.config,
        all=args.all,
        require_color_to_mass=require_color_to_mass,
    )

    # Run requested analyses using shared context
    if args.cropping:
        step_started_at = time.monotonic()
        # TODO: This is a bit of a hack, but it works for now.
        # We should refactor the analysis functions to return the image paths they process.
        # This will allow us to avoid this duplication and make the code cleaner.
        cropping_image_paths = (
            select_image_paths(
                ctx.config,
                ctx.experiment,
                all=args.all,
                sub_config=ctx.config.analysis.cropping,
            )
            if ctx.config.analysis.cropping
            else []
        )
        publish_step_start(
            progress_callback, step="cropping", image_total=len(cropping_image_paths)
        )
        analysis_cropping_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
            progress_callback=progress_callback,
        )
        publish_step_complete(
            progress_callback,
            step="cropping",
            image_total=len(cropping_image_paths),
            step_elapsed_s=time.monotonic() - step_started_at,
        )

    if args.mass:
        step_started_at = time.monotonic()
        mass_image_paths = (
            select_image_paths(
                ctx.config,
                ctx.experiment,
                all=args.all,
                sub_config=ctx.config.analysis.mass,
            )
            if ctx.config.analysis.mass
            else []
        )
        publish_step_start(
            progress_callback, step="mass", image_total=len(mass_image_paths)
        )
        analysis_mass_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
            progress_callback=progress_callback,
        )
        publish_step_complete(
            progress_callback,
            step="mass",
            image_total=len(mass_image_paths),
            step_elapsed_s=time.monotonic() - step_started_at,
        )

    if args.volume:
        step_started_at = time.monotonic()
        volume_image_paths = (
            select_image_paths(
                ctx.config,
                ctx.experiment,
                all=args.all,
                sub_config=ctx.config.analysis.volume,
            )
            if ctx.config.analysis.volume
            else []
        )
        publish_step_start(
            progress_callback, step="volume", image_total=len(volume_image_paths)
        )
        analysis_volume_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
            progress_callback=progress_callback,
        )
        publish_step_complete(
            progress_callback,
            step="volume",
            image_total=len(volume_image_paths),
            step_elapsed_s=time.monotonic() - step_started_at,
        )

    if args.segmentation:
        step_started_at = time.monotonic()
        segmentation_image_paths = (
            select_image_paths(
                ctx.config,
                ctx.experiment,
                all=args.all,
                sub_config=ctx.config.analysis.segmentation,
            )
            if ctx.config.analysis.segmentation
            else []
        )
        publish_step_start(
            progress_callback,
            step="segmentation",
            image_total=len(segmentation_image_paths),
        )
        analysis_segmentation_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
            progress_callback=progress_callback,
        )
        publish_step_complete(
            progress_callback,
            step="segmentation",
            image_total=len(segmentation_image_paths),
            step_elapsed_s=time.monotonic() - step_started_at,
        )

    if args.fingers:
        step_started_at = time.monotonic()
        fingers_image_paths = (
            select_image_paths(
                ctx.config,
                ctx.experiment,
                all=args.all,
                sub_config=ctx.config.analysis.fingers,
            )
            if ctx.config.analysis.fingers
            else []
        )
        publish_step_start(
            progress_callback, step="fingers", image_total=len(fingers_image_paths)
        )
        analysis_fingers_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
            progress_callback=progress_callback,
        )
        publish_step_complete(
            progress_callback,
            step="fingers",
            image_total=len(fingers_image_paths),
            step_elapsed_s=time.monotonic() - step_started_at,
        )

    if args.thresholding:
        step_started_at = time.monotonic()
        thresholding_image_paths = (
            select_image_paths(
                ctx.config,
                ctx.experiment,
                all=args.all,
                sub_config=ctx.config.analysis.thresholding,
            )
            if ctx.config.analysis.thresholding
            else []
        )
        publish_step_start(
            progress_callback,
            step="thresholding",
            image_total=len(thresholding_image_paths),
        )
        analysis_thresholding_from_context(
            ctx,
            show=args.show,
            stream_callback=stream_callback,
            progress_callback=progress_callback,
        )
        publish_step_complete(
            progress_callback,
            step="thresholding",
            image_total=len(thresholding_image_paths),
            step_elapsed_s=time.monotonic() - step_started_at,
        )


def _progress_wrapper(
    seq_cell: dict[str, int] | None = None,
) -> Callable[[AnalysisProgressEvent], None]:
    """Progress wrapper that prints every progress event as a stdout line.

    Used both for the minimal status-bar batch monitor (all event types:
    step_start carries image_total before any image has been processed,
    image_progress carries the running count, step_complete marks a mode
    finishing) and, when seq_cell is given, to additionally correlate
    streamed frames with their real image index/datetime for the Streaming
    Preview panel's timeline slider — only "image_progress" events are
    tagged with the same seq value the immediately-preceding stream_callback
    call just used.
    """

    def _callback(event: AnalysisProgressEvent) -> None:
        payload = event
        if seq_cell is not None and event.get("event") == "image_progress":
            payload = {**event, "seq": seq_cell["n"]}
        print(encode_progress_line(payload), flush=True)

    return _callback


def preset_analysis(rig_cls: type[Rig], **kwargs):
    parser = build_parser_for_analysis()
    args = parser.parse_args()
    print_help_for_flags(args, parser)
    streaming = getattr(args, "stream", False)
    seq_cell = {"n": 0} if streaming else None
    if streaming and "stream_callback" not in kwargs:
        if not args.stream_cache_dir:
            raise ValueError("--stream requires --stream-cache-dir.")
        kwargs["stream_callback"] = build_file_cache_stream_callback(
            Path(args.stream_cache_dir), seq_cell
        )
    if (
        streaming or getattr(args, "progress", False)
    ) and "progress_callback" not in kwargs:
        kwargs["progress_callback"] = _progress_wrapper(seq_cell)
    run_analysis(rig_cls, args, **kwargs)


if __name__ == "__main__":
    preset_analysis(Rig)
