import logging
from pathlib import Path

import darsia
from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig

logger = logging.getLogger(__name__)


def calibration_color_paths(cls, path: Path, show: bool = False) -> None:
    """Calibration of color paths for a given fluidflower class and configuration.

    Args:
        cls: The fluidflower class to use (e.g., ffum.MuseumRig).
        path: The path to the configuration file.
        show: Whether to display plots during processing.

    """
    config = FluidFlowerConfig(path)
    config.check("rig", "data", "protocol", "color_paths")

    # Mypy type checking
    assert config.color_paths is not None
    assert config.rig is not None
    assert config.data is not None
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
    baseline_images = []
    for p in config.color_paths.baseline_image_paths:
        cache_path = Path(".") / "tmp" / f"cache_{p.stem}.npz"
        if not cache_path.exists():
            baseline_image = fluidflower.read_image(p)
            baseline_image.save(cache_path)
        else:
            baseline_image = darsia.imread(cache_path)
        baseline_images.append(baseline_image)

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

    # ! ---- IDENTIFY AND STORE COLOR RANGE ----

    tracer_color_range = darsia.ColorRange(
        images=calibration_images,
        baseline=fluidflower.baseline,
        mask=fluidflower.boolean_porosity,
    )
    tracer_color_range.save(config.color_paths.color_range_file)

    # ! ---- COLOR PATH TOOL ----

    color_path_regression = darsia.LabelColorPathMapRegression(
        labels=fluidflower.labels,
        mask=fluidflower.boolean_porosity,
        color_range=tracer_color_range,
        resolution=config.color_paths.resolution,
        ignore_labels=config.color_paths.ignore_labels,
    )

    # ! ---- ANALYZE FLUCTUATIONS IN BASELINE IMAGES ----

    baseline_color_spectrum: darsia.LabelColorSpectrumMap = (
        color_path_regression.get_label_color_spectrum_map(
            images=baseline_images,
            baseline=fluidflower.baseline,
            threshold_significant=config.color_paths.threshold_baseline,
            verbose=False,
        )
    )
    # Free memory for performance
    del baseline_images

    # Expand the baseline color spectrum through linear regression
    expanded_baseline_color_spectrum: darsia.LabelColorSpectrumMap = (
        color_path_regression.expand_label_color_spectrum_map(
            label_color_spectrum_map=baseline_color_spectrum,
            verbose=False,
        )
    )

    # ! ---- EXTRACT COLOR SPECTRUM OF TRACERS ---- ! #

    tracer_color_spectrum = color_path_regression.get_label_color_spectrum_map(
        images=calibration_images,
        baseline=fluidflower.baseline,
        ignore=expanded_baseline_color_spectrum,
        threshold_significant=config.color_paths.threshold_calibration,
        verbose=False,
    )
    # Free memory for performance
    del calibration_images

    # Find a relative color path through the significant boxes
    label_color_path_map: darsia.LabelColorPathMap = (
        color_path_regression.find_label_color_path_map(
            label_color_spectrum_map=tracer_color_spectrum,
            ignore=expanded_baseline_color_spectrum,
            num_segments=config.color_paths.num_segments,
            directory=config.color_paths.calibration_file,
            verbose=show,
        )
    )

    # Store the color paths to file
    label_color_path_map.save(config.color_paths.calibration_file)

    # Display the color paths
    if show:
        print(label_color_path_map)
        label_color_path_map.show()

    # TODO: Provide advanced plotting - mostly for paper publication.

    logger.info("Calibration of color paths completed.")
