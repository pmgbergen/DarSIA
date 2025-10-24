"""Template for mass analysis."""

import logging
from pathlib import Path

import darsia
from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.heterogeneous_color_analysis import (
    HeterogeneousColorAnalysis,
)
from darsia.presets.workflows.mass_computation import MassComputation

logger = logging.getLogger(__name__)


def analysis_mass(
    cls,
    path: Path,
    show: bool = False,
    save_jpg: bool = False,
    save_npz: bool = False,
):
    # ! ---- LOAD RUN AND RIG ----
    config = FluidFlowerConfig(path)
    fluidflower = cls()
    fluidflower.load(config.data.results / "fluidflower")

    # Load experiment
    experiment = darsia.ProtocolledExperiment(
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

    # ! ---- MASS ----

    experiment_start = experiment.experiment_start
    flash = darsia.SimpleFlash()
    co2_mass_analysis = darsia.CO2MassAnalysis(
        baseline=fluidflower.baseline,
        atmospheric_pressure=experiment.pressure_temperature_protocol.get_state(
            experiment_start
        ).pressure,
        temperature=experiment.pressure_temperature_protocol.get_state(
            experiment_start
        ).temperature,
    )
    fluidflower.co2_mass_analysis = co2_mass_analysis
    mass_computation = MassComputation(
        baseline=fluidflower.baseline,
        geometry=fluidflower.geometry,
        flash=flash,
        co2_mass_analysis=fluidflower.co2_mass_analysis,
    )
    mass_computation.transformation.load(config.mass.calibration_file)

    # ! ---- VISUALIZATION ----

    # Pick a reference color path - merely for visualization
    reference_label = config.color_paths.reference_label
    ref_color_path = darsia.ColorPath()
    ref_color_path.load(
        config.color_paths.calibration_file / f"color_path_{reference_label}.json"
    )
    custom_cmap = ref_color_path.get_color_map()

    # ! ---- ANALYSIS ----

    if len(config.analysis.image_paths) > 0:
        image_paths = [config.data.folder / p for p in config.analysis.image_paths]
    else:
        image_times = config.analysis.image_times
        image_datetimes = [
            experiment.experiment_start + darsia.timedelta(hours=t) for t in image_times
        ]
        image_paths = experiment.imaging_protocol.find_images_for_datetimes(
            paths=config.data.data,
            datetimes=image_datetimes,
        )

    # Loop over images and analyze
    for path in image_paths:
        # Extract color signal and assign mass
        img = fluidflower.read_image(path)
        color_signal = color_signal_analysis(img)
        mass = mass_computation(color_signal)

        if show:
            img.show(title=f"Image at {path.stem}", delay=True)
            mass.show(title=f"Mass at {path.stem}", cmap=custom_cmap, delay=False)

        if save_npz:
            path = config.data.results / "mass_analysis" / f"{path.stem}.npz"
            mass.save(path)

        if save_jpg:
            path = config.data.results / "mass_analysis" / f"{path.stem}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            mass.write(path, cmap=custom_cmap, quality=80)
