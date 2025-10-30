import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import darsia
from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.heterogeneous_color_analysis import (
    HeterogeneousColorAnalysis,
)
from darsia.presets.workflows.mass_computation import MassComputation

logger = logging.getLogger(__name__)


def calibration_mass_analysis(cls, path: Path, show: bool = False) -> None:
    # ! ---- LOAD RUN AND RIG ----

    config = FluidFlowerConfig(path)
    config.check("rig", "data", "protocol", "color_paths", "color_signal", "mass")

    # Mypy type checking
    for c in [
        config.mass,
        config.color_signal,
        config.color_paths,
        config.data,
        config.protocol,
        config.rig,
    ]:
        assert c is not None

    fluidflower = cls()
    fluidflower.load(config.rig.path)

    # Load experiment
    experiment = darsia.ProtocolledExperiment(
        imaging_protocol=config.protocol.imaging,
        injection_protocol=config.protocol.injection,
        pressure_temperature_protocol=config.protocol.pressure_temperature,
        blacklist_protocol=config.protocol.blacklist,
        pad=config.data.pad,
    )
    fluidflower.load_experiment(experiment)

    # ! ---- READ COLOR PATHS FROM FILE ----

    color_paths = {
        label: darsia.ColorPath() for label in np.unique(fluidflower.labels.img)
    }
    for label in np.unique(fluidflower.labels.img):
        path = config.color_paths.calibration_file / f"color_path_{label}.json"
        if path.exists():
            color_paths[label].load(path)
            color_paths[label].name = f"Original Color path for label {label}"
        else:
            logger.warning(f"No color path found for label {label}, using empty path.")

    # Pick a reference color path - merely for visualization
    reference_label = config.color_paths.reference_label
    ref_color_path = color_paths[reference_label]
    custom_cmap = ref_color_path.get_color_map()

    # ! ---- CONCENTRATION ANALYSIS ---- ! #

    concentration_analysis = HeterogeneousColorAnalysis(
        baseline=fluidflower.baseline,
        labels=fluidflower.labels,
        # restoration=fluidflowerrestoration, # TODO try!
        ignore_labels=config.color_paths.ignore_labels,
    )
    concentration_analysis.load(config.color_signal.calibration_file)

    # ! ---- MASS ----

    experiment_start = experiment.experiment_start
    flash = darsia.SimpleFlash()  # TODO add restoration?
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
    if config.mass.calibration_file.exists():
        mass_computation.transformation.load(config.mass.calibration_file)

    # ! ---- CALIBRATION ----

    # Determine calibration images based on times
    calibration_times = config.mass.calibration_image_times
    calibration_datetimes = [
        experiment.experiment_start + darsia.timedelta(hours=t)
        for t in calibration_times
    ]
    calibration_images_paths = experiment.imaging_protocol.find_images_for_datetimes(
        paths=config.data.data,
        datetimes=calibration_datetimes,
    )

    # Read images
    calibration_images = []
    for img in calibration_images_paths:
        logger.info(f"Reading calibration image {img}.")
        calibration_images.append(fluidflower.read_image(img))

    # Convert to concentrations
    concentration_images = []
    for img in calibration_images:
        logger.info(f"Converting calibration image to concentration for {img.name}.")
        concentration_images.append(concentration_analysis(img))

    # Fit calibration
    mass_computation.fit(
        concentration_images,
        experiment,
    )

    # Save calibration
    mass_computation.transformation.save(config.mass.calibration_file)

    # ! ---- TEST CALIBRATION ----
    # TODO include calibration of scaling!
    # Aim at reproducing the background values through linear scaling (or RGB shift)
    # Can also use a patch analysis based on background images.

    # Test run
    mass_images = [mass_computation(img) for img in concentration_images]

    for i in range(len(calibration_images)):
        calibration_images[i].show(title=f"Calibration image {i}", delay=True)
        mass_images[i].mass.show(title=f"Mass image {i}", cmap=custom_cmap, delay=True)
    plt.show()

    print("Done. Analysis.")
