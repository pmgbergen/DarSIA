# TODO: File may be obsolete - consider removing.
# Currently, the functionality is not used anywhere.
# This an initial version of a calibration workflow for the FluidFlower experiment.
# It is still a work in progress and may be subject to significant changes.
# The main goal is to set up a calibration workflow that allows for interactive tuning
# of color paths and mass analysis based on the provided configuration and calibration
# images.
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import darsia
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.heterogeneous_color_analysis import (
    HeterogeneousColorAnalysis,
)
from darsia.presets.workflows.mass_computation import MassComputation

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
    experiment = darsia.ProtocolledExperiment.init_from_config(config)

    # ! ---- LOAD RIG ----
    fluidflower = cls.load(config.rig.path)
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
        cache_path = config.data.cache / f"{p.stem}.npz"
        if not cache_path.exists():
            calibration_image = fluidflower.read_image(p)
            calibration_image.save(cache_path)
        else:
            calibration_image = fluidflower.read_image(p)
        calibration_images.append(calibration_image)

    # ! ---- LOAD COLOR PATHS ----
    color_paths = darsia.LabelColorPathMap.load(config.color_paths.calibration_file)

    # ! ---- LOAD COLOR RANGE ----
    # TODO: rm? Not used?
    # color_range = darsia.ColorRange.load(config.color_paths.color_range_file)
    # discrete_color_range = darsia.DiscreteColorRange(
    #    color_range, config.color_paths.resolution
    # )

    # ! ---- ALLOCATE EMPTY INTERPOLATIONS ----
    color_path_interpolation = {
        label: darsia.ColorPathInterpolation(
            color_path=color_path,
            color_mode=darsia.ColorMode.RELATIVE,
            values=color_path.relative_distances,
        )
        for label, color_path in color_paths.items()
    }

    # Pick a reference color path - merely for visualization
    reference_label = config.color_paths.reference_label
    reference_color_path = color_paths[reference_label]
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
    distances = {
        label: max(
            [baseline_color_spectrum[label].distance(c) for c in color_path.colors]
        )
        for label, color_path in color_paths.items()
    }

    # Metric II.
    # Determine distance from color path to reference color path.
    reference_interpolation = darsia.ColorPathInterpolation(
        color_path=reference_color_path,
        color_mode=darsia.ColorMode.RELATIVE,
        values=reference_color_path.relative_distances,
    )
    interpolation_values = {
        label: max(
            [
                max(0.0, float(reference_interpolation(c)))
                for c in color_path.relative_colors
            ]
        )
        for label, color_path in color_paths.items()
    }

    # Decide which labels to ignore based on the two metrics
    threshold = 0.5  # TODO include in config.
    ignore_labels = []
    for label in np.unique(fluidflower.labels.img):
        relative_distance = distances[label] / max(distances.values())
        relative_max_interpolation = interpolation_values[label] / max(
            interpolation_values.values()
        )
        print(label, relative_distance, relative_max_interpolation)
        if min(relative_distance, relative_max_interpolation) < threshold:
            ignore_labels.append(label)

    print("Ignoring labels:", ignore_labels)

    # Illustrate the ignored labels for the calibration images throgh grayscaling
    for img in calibration_images[-1:]:
        _img = img.copy()
        for mask, label in darsia.Masks(fluidflower.labels, return_label=True):
            if label not in ignore_labels:
                continue
            _img.img[mask.img] = np.mean(_img.img[mask.img], axis=1, keepdims=True)
        _img.show(cmap=custom_cmap, title="Ignored labels")

    # Rescale color paths based on reference interpolation
    for label in color_path_interpolation:
        color_path_interpolation[label].values *= interpolation_values[label]
        # color_path = color_paths[label]
        # color_path_interpolation[label].values = (
        #    np.array(color_path.relative_distances)
        #    * interpolation_values[label]
        #    / max(
        #        color_path.relative_distances
        #    )  # Normalization not needed as relative distances are normalized # TODO
        # ).tolist()

    # Overwrite the color paths with updated interpolation values
    for label in np.unique(fluidflower.labels.img):
        if label in config.color_paths.ignore_labels or label in ignore_labels:
            color_path_interpolation[label] = color_path_interpolation[reference_label]
            # TODO: Set an empty color path based on the mean background color.
            # color_paths[label] = darsia.ColorPath()
        elif False:
            print(color_path_interpolation[label].values)
            color_paths[label].show()

    # ! ---- CONCENTRATION ANALYSIS ---- ! #

    # TODO: rm? Not used?
    # color_path_interpolation_lut = {
    #    label: darsia.ColorPathLUT(
    #        color_path_interpolation[label], discrete_color_range
    #    )
    #    for label in color_path_interpolation
    # }

    # Plain color path interpretation - basis for concentration analysis
    color_path_interpretation = {
        label: darsia.ColorPathInterpolation(
            color_path=color_path,
            color_mode=darsia.ColorMode.RELATIVE,
            # values=color_path.equidistant_distances,
            values=color_path.relative_distances,
        )
        for label, color_path in color_paths.items()
    }
    color_analysis = HeterogeneousColorAnalysis(
        baseline=fluidflower.baseline,
        labels=fluidflower.labels,
        color_mode=darsia.ColorMode.RELATIVE,
        color_path_functions=color_path_interpretation,
        # restoration=fluidflower.restoration,
        ignore_labels=config.color_paths.ignore_labels + ignore_labels,
    )

    color_images = []
    for img in calibration_images:
        color_image = color_analysis(img)
        color_image.show(title="Color analysis", cmap=custom_cmap, delay=False)
        color_images.append(color_image)

    # TODO: rm? Not used?
    # signal_analysis = HeterogeneousSignalAnalysis()

    concentration_analysis = HeterogeneousColorAnalysis(
        baseline=fluidflower.baseline,
        labels=fluidflower.labels,
        color_mode=darsia.ColorMode.RELATIVE,
        color_path_functions=color_path_interpolation,
        # restoration=fluidflower.restoration,
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

    # Start from existing calibration if available
    if config.color_signal.calibration_file.exists():
        concentration_analysis.load(config.color_signal.calibration_file)

    # Perform global and local calibration
    concentration_analysis.global_calibration_flash(
        mass_computation=mass_computation,
        mask=fluidflower.boolean_porosity,
        calibration_images=calibration_images,
        experiment=experiment,
        cmap=custom_cmap,
        show=show,
    )

    concentration_analysis.local_calibration_flash(
        mass_computation=mass_computation,
        mask=fluidflower.boolean_porosity,
        calibration_images=calibration_images,
        cmap=custom_cmap,
        show=show,
    )

    # Free memory for performance
    # del calibration_images

    # Use reference color path for ignored labels (as a de)
    for label in np.unique(fluidflower.labels.img):
        if label in config.color_paths.ignore_labels or label in ignore_labels:
            color_paths[label] = reference_color_path

    # Store calibration of concentration analysis and mass
    concentration_analysis.save(config.color_signal.calibration_file)
    # mass_computation.transformation.save(config.mass.calibration_file)

    ###################################################
    # Test run
    ###################################################
    concentration_images = [concentration_analysis(img) for img in calibration_images]

    for i in range(len(calibration_images)):
        calibration_images[i].show(title=f"Calibration image {i}", delay=True)
        concentration_images[i].show(
            title=f"Concentration image {i}", cmap=custom_cmap, delay=True
        )
    plt.show()
