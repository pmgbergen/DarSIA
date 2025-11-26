"""Template for color signal analysis."""

import logging
from pathlib import Path

import darsia
from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.heterogeneous_color_analysis import (
    HeterogeneousColorAnalysis,
)

logger = logging.getLogger(__name__)


def analysis_color_signal(
    cls,
    path: Path,
    show: bool = False,
    save_jpg: bool = False,
    save_npz: bool = False,
    **kwargs,
):
    # ! ---- LOAD RUN AND RIG ----
    config = FluidFlowerConfig(path)
    config.check("analysis", "protocol", "data", "color_paths", "color_signal", "rig")

    # Mypy type checking
    for c in [
        config.color_signal,
        config.color_paths,
        config.data,
        config.protocol,
        config.analysis,
        config.rig,
    ]:
        assert c is not None

    fluidflower = cls()
    fluidflower.load(config.rig.path)

    # Load experiment
    experiment = darsia.ProtocolledExperiment(
        data=config.data.data,
        imaging_protocol=config.protocol.imaging,
        injection_protocol=config.protocol.injection,
        pressure_temperature_protocol=config.protocol.pressure_temperature,
        blacklist_protocol=config.protocol.blacklist,
        pad=config.data.pad,
    )
    fluidflower.load_experiment(experiment)

    # ! ---- CONCENTRATION ANALYSIS ---- ! #

    color_signal_analysis = HeterogeneousColorAnalysis(
        baseline=fluidflower.baseline,
        labels=fluidflower.labels,
        # restoration=fluidflower.restoration,
        ignore_labels=config.color_paths.ignore_labels,
    )
    color_signal_analysis.load(config.color_signal.calibration_file)

    # ! ---- VISUALIZATION ----

    # Pick a reference color path - merely for visualization
    reference_label = config.color_paths.reference_label
    ref_color_path = darsia.ColorPath()
    ref_color_path.load(
        config.color_paths.calibration_file / f"color_path_{reference_label}.json"
    )
    custom_cmap = ref_color_path.get_color_map()

    # ! ---- ANALYSIS ----

    # assert False, "make this part of the config?"

    if len(config.analysis.image_paths) == 0:
        image_times = config.analysis.image_times
        image_datetimes = [
            experiment.experiment_start + darsia.timedelta(hours=t) for t in image_times
        ]
        image_paths = experiment.imaging_protocol.find_images_for_datetimes(
            paths=config.data.data,
            datetimes=image_datetimes,
        )
    else:
        image_paths = config.analysis.image_paths

    # Loop over images and analyze
    for path in image_paths:
        # Extract color signal
        img = fluidflower.read_image(path)
        color_signal = color_signal_analysis(img)

        if show:
            img.show(title=f"Image at {path.stem}", delay=True)
            color_signal.show(
                title=f"Concentration at {path.stem}", cmap=custom_cmap, delay=False
            )

        if save_npz:
            path = config.data.results / "color_signal_analysis" / f"{path.stem}.npz"
            color_signal.save(path)

        if save_jpg:
            path = config.data.results / "color_signal_analysis" / f"{path.stem}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            color_signal.write(path, cmap=custom_cmap, quality=80)
