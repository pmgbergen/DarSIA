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
    path: Path | list[Path],
    show: bool = False,
    save_jpg: bool = False,
    save_npz: bool = False,
    all: bool = False,
) -> None:
    """Cropping analysis.

    Note: If no options are set, the images are only read and no output is saved.

    Args:
        cls: FluidFlower rig class.
        path: Path or list of Paths to the images.
        show: Whether to show the images.
        save_jpg: Whether to save the images as JPG.
        save_npz: Whether to save the images as NPZ.
        all: Whether to use all images or only the ones specified in the config.

    """
    # Make sure that conig is a Path or list of Path
    if not isinstance(path, list):
        path = [path]
    path = [Path(p) for p in path]

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

    # ! ---- LOAD EXPERIMENT ----
    experiment = darsia.ProtocolledExperiment.init_from_config(config)

    # ! ---- LOAD RIG ----
    fluidflower = cls()
    fluidflower.load(config.rig.path)
    fluidflower.load_experiment(experiment)

    # ! ---- LOAD IMAGES ----
    if all:
        image_paths = config.data.data
    elif len(config.analysis.image_paths) > 0:
        image_paths = config.analysis.image_paths
    else:
        image_paths = experiment.find_images_for_times(
            times=config.analysis.image_times
        )

    # ! ---- CROPPING ----
    plot_folder = config.data.results / "cropped_images"
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
