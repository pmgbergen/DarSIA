"""Template for cropping/reading images."""

import logging
from pathlib import Path

import numpy as np

import darsia
from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def analysis_cropping(
    cls,
    path: Path,
    show: bool = False,
    save_jpg: bool = False,
    save_npz: bool = False,
    **kwargs,
):
    # Read data from meta
    config = FluidFlowerConfig(path)
    config.check("analysis", "protocol", "data", "rig")

    # Mypy type checking
    for c in [
        config.rig,
        config.data,
        config.protocol,
        config.analysis,
    ]:
        assert c is not None

    # ! ---- LOAD RIG AND RUN ----

    fluidflower = cls()
    fluidflower.load(config.rig.path)

    # Load run
    experiment = darsia.ProtocolledExperiment(
        data=config.data.data,
        imaging_protocol=config.protocol.imaging,
        injection_protocol=config.protocol.injection,
        pressure_temperature_protocol=config.protocol.pressure_temperature,
        blacklist_protocol=config.protocol.blacklist,
        pad=config.data.pad,
    )
    fluidflower.load_experiment(experiment)

    # Plotting
    plot_folder = config.data.results / "cropped_images"
    plot_folder.mkdir(parents=True, exist_ok=True)

    # Make selection of images to analyze
    if len(config.analysis.image_paths) > 0:
        image_paths = [config.data.folder / p for p in config.analysis.image_paths]
    else:
        image_times = config.analysis.image_times
        image_datetimes = [
            experiment.experiment_start + darsia.timedelta(hours=t) for t in image_times
        ]
        image_paths = experiment.imaging_protocol.find_images_for_datetimes(
            paths=config.data.data, datetimes=image_datetimes
        )

    for path in image_paths:
        # Update
        fluidflower.update(path)

        # Read image
        img = fluidflower.read_image(path)

        # Convert image to darsia.OpticalImage
        img = darsia.OpticalImage(img.img, **img.metadata())

        if show:
            img.show()

        if save_npz:
            img.save(plot_folder / f"{path.stem}.npz")

        if save_jpg:
            img = img.img_as(np.uint8)
            img.original_dtype = np.uint8  # Hack to allow plotting
            img.write(plot_folder / f"{path.stem}.jpg", quality=50)

    print("Done. Analysis.")
