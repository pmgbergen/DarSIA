import logging
from pathlib import Path

import darsia
from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig

logger = logging.getLogger(__name__)


def setup_color_paths(cls, path: Path, show: bool = False) -> None:
    """Setup color paths for a given fluidflower class and configuration.

    Args:
        cls: The fluidflower class to use (e.g., ffum.MuseumRig).
        path: The path to the configuration file.
        show: Whether to display plots during processing.

    """
    config = FluidFlowerConfig(path)

    # ! ---- LOAD RUN AND RIG ----

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

    # ! ---- COLOR PATH TOOL ----
    color_path_regression = darsia.ColorPathRegression(
        labels=fluidflower.labels,
        ignore_labels=config.color_paths.ignore_labels,
        mask=fluidflower.boolean_porosity,
    )

    # ! ---- ANALYZE FLUCTUATIONS IN 10 BASELINE IMAGES ----
    baseline_images = []
    for path in config.color_paths.baseline_images:
        logger.info("Reading baseline image: %s", path)
        baseline_image = fluidflower.read_image(path)
        baseline_images.append(baseline_image)

    baseline_color_spectrum = color_path_regression.get_color_spectrum(
        images=baseline_images,
        baseline=fluidflower.baseline,
        resolution=config.color_paths.resolution,
        threshold_significant=config.color_paths.threshold_baseline,
        verbose=show,
    )

    # Expand the baseline color spectrum through linear regression
    baseline_color_spectrum = color_path_regression.expand_color_spectrum(
        color_spectrum=baseline_color_spectrum, verbose=show
    )

    # ! ---- EXTRACT COLOR SPECTRUM OF TRACERS ---- ! #

    calibration_images = []
    for path in config.color_paths.calibration_images:
        logger.info("Reading calibration image: %s", path)
        calibration_image = fluidflower.read_image(path)
        calibration_images.append(calibration_image)

    tracer_color_spectrum = color_path_regression.get_color_spectrum(
        images=calibration_images,
        baseline=fluidflower.baseline,
        resolution=config.color_paths.resolution,
        ignore_color_spectrum=baseline_color_spectrum,
        threshold_significant=config.color_paths.threshold_calibration,
        verbose=show,
    )

    # Find a relative color path through the significant boxes
    color_path: dict[int, darsia.ColorPath] = {}
    for label, spectrum in tracer_color_spectrum.items():
        color_path[label] = color_path_regression.find_relative_color_path(
            spectrum=spectrum,
            base_color_spectrum=baseline_color_spectrum.get(label, None),
            verbose=show,
            plot_title=f"Color Path Analysis - Label {label}",
        )

    config.color_paths.calibration_file.mkdir(parents=True, exist_ok=True)
    for label, path in color_path.items():
        print(f"Label {label}:")
        print(f"  Base color: {path.base_color}")
        print(f"  Relative colors: {path.relative_colors}")
        print(f"  Values: {path.values}")
        if show:
            path.show()
        path.save(config.color_paths.calibration_file / f"color_path_{label}.json")
