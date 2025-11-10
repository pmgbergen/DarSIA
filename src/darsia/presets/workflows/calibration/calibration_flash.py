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
import time

logger = logging.getLogger(__name__)


def calibration_flash(cls, path: Path, show: bool = False):
    # ! ---- LOAD RUN AND RIG ----

    config = FluidFlowerConfig(path)
    config.check("color_signal", "color_paths", "rig", "data", "protocol")

    # Mypy type checking
    assert config.color_signal is not None
    assert config.color_paths is not None
    assert config.rig is not None
    assert config.data is not None
    assert config.data.data is not None
    assert config.protocol is not None
    assert config.protocol.imaging is not None
    assert config.protocol.injection is not None
    assert config.protocol.pressure_temperature is not None
    assert config.protocol.blacklist is not None

    # ! ---- LOAD EXPERIMENT ----
    experiment = darsia.ProtocolledExperiment(
        data=config.data.data,
        imaging_protocol=config.protocol.imaging,
        injection_protocol=config.protocol.injection,
        pressure_temperature_protocol=config.protocol.pressure_temperature,
        blacklist_protocol=config.protocol.blacklist,
        pad=config.data.pad,
    )

    # ! ---- LOAD RIG ----
    fluidflower = cls()
    fluidflower.load(config.rig.path)
    fluidflower.load_experiment(experiment)

    # ! ---- LOAD IMAGES ----
    # Store cached versions of calibration images to speed up development
    # TODO Use calibration.flash.image_paths or so.
    calibration_image_paths = config.color_paths.calibration_image_paths
    if len(calibration_image_paths) == 0:
        calibration_image_paths = experiment.find_images_for_times(
            times=config.color_paths.calibration_image_times
        )
    calibration_images = []
    for p in calibration_image_paths:
        cache_path = Path(".") / "tmp" / f"cache_{p.stem}.npz"
        if not cache_path.exists():
            calibration_image = fluidflower.read_image(p)
            calibration_image.save(cache_path)
        else:
            calibration_image = darsia.imread(cache_path)
        calibration_images.append(calibration_image)

    # ! ---- COLOR PATH TOOL ----
    # color_path_regression = darsia.LabelColorPathMapRegression(
    #    labels=fluidflower.labels,
    #    mask=fluidflower.boolean_porosity,
    #    color_range=color_range,
    #    resolution=config.color_paths.resolution,
    #    ignore_labels=config.color_paths.ignore_labels,  # TODO move somewhere else?
    # )

    # ! ---- LOAD COLOR PATHS ----
    color_paths = darsia.LabelColorPathMap.load(config.color_paths.calibration_file)

    # ! ---- LOAD COLOR RANGE ----
    color_range = darsia.ColorRange.load(config.color_paths.color_range_file)
    color_paths.parametrize_color_range(
        color_range, config.color_paths.resolution, relative=True
    )

    # Pick a reference color path - merely for visualization
    reference_color_path = color_paths[config.color_paths.reference_label]
    custom_cmap = reference_color_path.get_color_map()
    if show and False:
        reference_color_path.show()

    # ! ---- DEACTIVATE INSENSITIVE COLOR PATHS ----

    # TODO move this to another calibration function.

    # Metric I.
    # Determine distance from color path to baseline spectrum (consider the furthest
    # away color to measure sensitivity)
    baseline_color_spectrum = darsia.LabelColorSpectrumMap.load(
        config.color_paths.baseline_color_spectrum_file
    )
    distances = {}
    for label, color_path in color_paths.items():
        distances[label] = max(
            [baseline_color_spectrum[label].distance(c) for c in color_path.colors]
        )

    # Metric II.
    # Determine distance from color path to reference color path.
    reference_interpolation = darsia.ColorPathInterpolation(
        color_path=reference_color_path, interpolation="relative"
    )
    interpolation_values = {}
    for label, color_path in color_paths.items():
        interpolation_values[label] = max(
            [
                max(0.0, float(reference_interpolation(c)))
                for c in color_path.relative_colors
            ]
        )

    # Decide which labels to ignore based on the two metrics
    threshold = 0.5  # TODO include in config.
    ignore_labels = []
    for label in np.unique(fluidflower.labels.img):
        relative_distance = distances[label] / max(distances.values())
        relative_max_interpolation = interpolation_values[label] / max(
            interpolation_values.values()
        )
        if min(relative_distance, relative_max_interpolation) < threshold:
            ignore_labels.append(label)

    # Rescale color paths based on reference interpolation
    for label, color_path in color_paths.items():
        color_paths[label].values = (
            np.array(color_path.values)
            * interpolation_values[label]
            / max(color_path.values)
        ).tolist()

    # Overwrite the color paths with updated interpolation values
    for label in np.unique(fluidflower.labels.img):
        if label in config.color_paths.ignore_labels or label in ignore_labels:
            color_paths[label] = reference_color_path
            # TODO: Set an empty color path based on the mean background color.
            # color_paths[label] = darsia.ColorPath()
        else:
            print(color_paths[label].values)
    if show and False:
        color_paths.show()

    # ! ---- CONCENTRATION ANALYSIS ---- ! #

    concentration_analysis = HeterogeneousColorAnalysis(
        baseline=fluidflower.baseline,
        labels=fluidflower.labels,
        # restoration=fluidflower.restoration,
        label_color_path_map=color_paths,
        ignore_labels=config.color_paths.ignore_labels + ignore_labels,
    )

    # ! ---- MASS ANALYSIS ---- ! #

    # TODO connect to fluidflower?
    # fluidflower.co2_mass_analysis = co2_mass_analysis

    # TODO
    # * Use identify transformation here, i.e., aim at signal analysis to map to [0,2]
    # * Consider the entire color paths with 2 for initialization?
    current_time = experiment.experiment_start
    co2_mass_analysis = darsia.CO2MassAnalysis(
        baseline=fluidflower.baseline,
        atmospheric_pressure=experiment.pressure_temperature_protocol.get_state(
            current_time
        ).pressure,
        temperature=experiment.pressure_temperature_protocol.get_state(
            current_time
        ).temperature,
    )
    mass_computation = MassComputation(
        baseline=fluidflower.baseline,
        geometry=fluidflower.geometry,
        flash=darsia.SimpleFlash(),  # TODO add restoration?
        # co2_mass_analysis=fluidflower.co2_mass_analysis,
        co2_mass_analysis=co2_mass_analysis,
    )
    # if config.mass.calibration_file.exists():
    #    mass_computation.transformation.load(config.mass.calibration_file)

    # ! ---- INTERACTIVE CALIBRATION ---- ! #

    # TODO include! one should allow for manual color signal tuning!

    ## Start from existing calibration if available
    # if config.color_signal.calibration_file.exists():
    #    concentration_analysis.load(config.color_signal.calibration_file)

    # Perform global and local calibration
    concentration_analysis.global_calibration_flash(
        mass_computation=mass_computation,
        mask=fluidflower.boolean_porosity,
        calibration_images=calibration_images,
        experiment=experiment,
        cmap=custom_cmap,
        show=show,
    )

    assert False
    concentration_analysis.local_calibration_flash(
        mass_computation=mass_computation,
        mask=fluidflower.boolean_porosity,
        calibration_images=calibration_images,
        cmap=custom_cmap,
        show=show,
    )

    assert False

    # Free memory for performance
    del calibration_images

    # Use reference color path for ignored labels (as a de)
    for label in np.unique(fluidflower.labels.img):
        if label in config.color_paths.ignore_labels or label in ignore_labels:
            color_paths[label] = reference_color_path

    # Store calibration of concentration analysis and mass
    concentration_analysis.save(config.color_signal.calibration_file)
    # mass_computation.transformation.save(config.mass.calibration_file)

    # Test run
    concentration_images = [concentration_analysis(img) for img in calibration_images]

    for i in range(len(calibration_images)):
        calibration_images[i].show(title=f"Calibration image {i}", delay=True)
        concentration_images[i].show(
            title=f"Concentration image {i}", cmap=custom_cmap, delay=True
        )
    plt.show()
