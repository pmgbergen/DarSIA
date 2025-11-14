import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

import darsia
from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.heterogeneous_color_analysis import (
    HeterogeneousColorAnalysis,
)

logger = logging.getLogger(__name__)


def calibration_color_analysis(cls, path: Path, show: bool = False):
    # ! ---- LOAD RUN AND RIG ----

    config = FluidFlowerConfig(path)
    config.check("color_signal", "color_paths", "rig", "data", "protocol")

    # Mypy type checking
    for c in [
        config.color_signal,
        config.color_paths,
        config.rig,
        config.data,
        config.protocol,
    ]:
        assert c is not None

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
    color_paths = darsia.LabelColorPathMap.load(config.color_paths.calibration_file)

    # ! ---- LOAD COLOR RANGE ----
    color_range = darsia.ColorRange.load(config.color_paths.color_range_file)

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
        )
        for label, color_path in color_paths.items()
    }

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
    if False:
        for img in calibration_images[-1:]:
            _img = img.copy()
            for mask, label in darsia.Masks(fluidflower.labels, return_label=True):
                if label not in ignore_labels:
                    continue
                _img.img[mask.img] = np.mean(_img.img[mask.img], axis=1, keepdims=True)
            _img.show(cmap=custom_cmap, title="Ignored labels")

    # Utils I. Determine distance to baseline spectrum
    for label in baseline_color_spectrum:
        color_path = color_paths[label]
        # Determine the minimum distance
        min_distance = baseline_color_spectrum[label].distance(
            color_path.relative_colors[1]
        )
        # Corresponding relative distance
        relative_distance = color_path.relative_distances[1]
        # Adjust all relative distances accordingly through additive shift
        if relative_distance is not np.nan:
            color_path.relative_distances[1:] += min_distance - relative_distance

    # Utils II.

    # Rescale color paths based on reference interpolation
    # for label in color_path_interpolation:
    #     color_path_interpolation[label].values *= interpolation_values[label]
    #     # color_path = color_paths[label]
    #     # color_path_interpolation[label].values = (
    #     #    np.array(color_path.relative_distances)
    #     #    * interpolation_values[label]
    #     #    / max(
    #     #        color_path.relative_distances
    #     #    )  # Normalization not needed as relative distances are normalized # TODO
    #     # ).tolist()

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

    # ! ---- INTERACTIVE CALIBRATION ---- ! #

    ## Start from existing calibration if available
    # if config.color_signal.calibration_file.exists():
    #    concentration_analysis.load(config.color_signal.calibration_file)

    color_paths[23].show_path()

    # Perform local calibration
    color_analysis.local_calibration_values(
        images=calibration_images, mask=fluidflower.boolean_porosity, cmap=custom_cmap
    )

    # TODO use reference color path to ignore labels?
    # for label in np.unique(fluidflower.labels.img):
    #    if label in config.color_paths.ignore_labels or label in ignore_labels:
    #        # TODO color_paths[label] = reference_color_path

    # Store calibration
    color_analysis.save(config.color_signal.calibration_file)

    # Test run
    concentration_images = [color_analysis(img) for img in calibration_images]

    for i in range(len(calibration_images)):
        calibration_images[i].show(title=f"Calibration image {i}", delay=True)
        concentration_images[i].show(
            title=f"Concentration image {i}", cmap=custom_cmap, delay=True
        )
    plt.show()
