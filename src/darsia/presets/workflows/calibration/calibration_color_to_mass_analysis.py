import logging
from pathlib import Path

import numpy as np

import darsia
from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.heterogeneous_color_to_mass_analysis import (
    HeterogeneousColorToMassAnalysis,
)

logger = logging.getLogger(__name__)


def calibration_color_to_mass_analysis(
    cls, path: Path, reset: bool = False, show: bool = False
):
    # ! ---- LOAD RUN AND RIG ----

    config = FluidFlowerConfig(path)
    config.check("color_paths", "rig", "data", "protocol", "color_to_mass")

    # Mypy type checking
    assert config.data is not None
    assert config.protocol is not None
    assert config.color_paths is not None
    assert config.rig is not None
    assert config.color_to_mass is not None

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

    # ! ---- LOAD COLOR PATHS ----
    pre_color_paths = darsia.LabelColorPathMap.load(config.color_paths.calibration_file)

    # ! ---- REFINE COLOR PATHS ----
    color_paths = darsia.LabelColorPathMap.refine(
        pre_color_paths,
        num_segments=8,
    )

    # ! ---- LOAD BASELINE COLOR SPECTRUM ----
    baseline_color_spectrum = darsia.LabelColorSpectrumMap.load(
        config.color_paths.baseline_color_spectrum_file
    )

    # ! ---- LOAD IMAGES ----
    # Store cached versions of calibration images to speed up development
    # TODO Use calibration.flash.image_paths or so.
    calibration_image_paths = config.color_to_mass.calibration_image_paths
    if len(calibration_image_paths) == 0:
        calibration_image_paths = experiment.find_images_for_times(
            times=config.color_to_mass.calibration_image_times
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

    # Pick a reference color path - merely for visualization
    reference_label = config.color_paths.reference_label
    reference_color_path = color_paths[reference_label]
    custom_cmap = reference_color_path.get_color_map()
    if show and False:
        reference_color_path.show()

    # ! ---- ALLOCATE EMPTY INTERPOLATIONS ----

    color_path_interpolation = {
        label: darsia.ColorPathInterpolation(
            color_path=color_path,
            color_mode=darsia.ColorMode.RELATIVE,
            values=color_path.relative_distances,
            ignore_spectrum=baseline_color_spectrum[label],
        )
        for label, color_path in color_paths.items()
    }

    # ! ---- DEACTIVATE INSENSITIVE COLOR PATHS ----

    # TODO move this to another calibration function.

    # Util 1.
    threshold = 0.2  # TODO include in config.

    # Metric I.
    # Determine distance from color path to baseline spectrum (consider the furthest
    # away color to measure sensitivity)
    distances = {
        label: max(
            [
                float(baseline_color_spectrum[label].distance(c))
                for c in color_path.colors
            ]
        )
        for label, color_path in color_paths.items()
    }
    reference_distance = max(distances.values())

    # Metric II.
    # Determine distance from color path to reference color path.
    reference_interpolation = color_path_interpolation[reference_label]
    interpolation_values = {
        label: max(
            [
                max(0.0, float(reference_interpolation(c)))
                for c in color_path.relative_colors
            ]
        )
        for label, color_path in color_paths.items()
    }
    reference_interpolation_value = max(interpolation_values.values())

    # Decide which labels to ignore based on the two metrics
    ignore_labels = []
    for label in np.unique(fluidflower.labels.img):
        relative_distance = distances[label] / reference_distance
        relative_max_interpolation = (
            interpolation_values[label] / reference_interpolation_value
        )
        if min(relative_distance, relative_max_interpolation) < threshold:
            ignore_labels.append(label)

    print("Ignoring labels:", ignore_labels)

    # Illustrate the ignored labels for the calibration images throgh grayscaling
    if False:
        for img in calibration_images[-1:]:
            _img = img.copy()
            for mask, label in darsia.Masks(fluidflower.labels, return_label=True):
                if label not in ignore_labels:
                    continue
                _img.img[mask.img] = np.mean(_img.img[mask.img], axis=1, keepdims=True)
            _img.show(cmap=custom_cmap, title="Ignored labels")

    # TODO deactivate?

    # Utils I. Scale refined color paths to match the original color paths (scale by 2)
    # for label in color_path_interpolation:
    #    color_path_interpolation[label].values *= 2

    ## Utils II. Determine distance to baseline spectrum and use that for the first interpolation value
    ## Helps to tone down fluctuations in the baseline color spectrum
    # for label in baseline_color_spectrum:
    #    color_path = color_paths[label]
    #    min_distance = baseline_color_spectrum[label].distance(
    #        color_path.relative_colors[1]
    #    )
    #    if color_path.relative_distances[1] is not np.nan:
    #        color_path.relative_distances[1] = min_distance

    # Utils III. Adapt the interpolation values based on the reference color path

    # Rescale color paths based on reference interpolation
    for label in color_path_interpolation:
        color_path_interpolation[label].values *= interpolation_values[label]

    # Overwrite the color paths with updated interpolation values
    for label in np.unique(fluidflower.labels.img):
        if label in config.color_paths.ignore_labels or label in ignore_labels:
            color_path_interpolation[label] = color_path_interpolation[reference_label]

    # ! ---- COLOR PATH INTERPRETATION ---- ! #

    color_path_interpretation = {
        label: darsia.ColorPathInterpolation(
            color_path=color_path,
            color_mode=darsia.ColorMode.RELATIVE,
            values=color_path.equidistant_distances,
            # values=color_path.relative_distances,
            ignore_spectrum=baseline_color_spectrum[label],
        )
        for label, color_path in color_paths.items()
    }

    # ! ---- SIGNAL FUNCTIONS ---- ! #

    signal_functions = {
        label: darsia.PWTransformation(
            color_paths[label].equidistant_distances,
            color_path_interpretation[label].values,
        )
        for label in color_path_interpretation
    }

    # ! ---- POROSITY-INFORMED AVERAGING ---- ! #
    # TODO: holes in the segmentation?
    image_porosity = fluidflower.image_porosity
    # boolean_porosity = image_porosity > 0.9
    restoration = darsia.VolumeAveraging(
        rev=darsia.REV(size=0.005, img=fluidflower.baseline),
        mask=image_porosity,
        # labels=fluidflower.labels,
    )

    # ! ---- FROM COLOR PATH TO MASS ----

    experiment_start = experiment.experiment_start
    flash = darsia.SimpleFlash(cut_off=0.8, max_value=0.95, restoration=None)
    state = experiment.pressure_temperature_protocol.get_state(experiment_start)
    gradient = experiment.pressure_temperature_protocol.get_gradient(experiment_start)
    co2_mass_analysis = darsia.CO2MassAnalysis(
        baseline=fluidflower.baseline,
        atmospheric_pressure=state.pressure,
        atmospheric_temperature=state.temperature,
        atmospheric_pressure_gradient=gradient.pressure,
        atmospheric_temperature_gradient=gradient.temperature,
    )

    if not reset and config.color_to_mass.calibration_folder.exists():
        # Start from existing calibration if available
        color_analysis = HeterogeneousColorToMassAnalysis.load(
            folder=config.color_to_mass.calibration_folder,
            baseline=fluidflower.baseline,
            labels=fluidflower.labels,
            co2_mass_analysis=co2_mass_analysis,
            geometry=fluidflower.geometry,
            restoration=restoration,
        )
    else:
        color_analysis = HeterogeneousColorToMassAnalysis(
            baseline=fluidflower.baseline,
            labels=fluidflower.labels,
            color_mode=darsia.ColorMode.RELATIVE,
            # color_path_interpretation=color_path_interpretation,
            # signal_functions=signal_functions,
            color_path_interpretation=color_path_interpretation,
            signal_functions=signal_functions,
            flash=flash,
            co2_mass_analysis=co2_mass_analysis,
            geometry=fluidflower.geometry,
            restoration=restoration,
            ignore_labels=config.color_paths.ignore_labels + ignore_labels,
        )

    # ! ---- INTERACTIVE CALIBRATION ---- ! #

    # Perform local calibration
    color_analysis.manual_calibration(
        images=calibration_images, experiment=experiment, cmap=custom_cmap
    )

    # TODO use reference color path to ignore labels?
    # for label in np.unique(fluidflower.labels.img):
    #    if label in config.color_paths.ignore_labels or label in ignore_labels:
    #        # TODO color_paths[label] = reference_color_path

    # Store calibration
    color_analysis.save(config.color_to_mass.calibration_folder)

    # Test run
    for img in calibration_images:
        mass = color_analysis(img).mass
        mass.show(title=f"Mass image {img.time}", cmap=custom_cmap, delay=False)
