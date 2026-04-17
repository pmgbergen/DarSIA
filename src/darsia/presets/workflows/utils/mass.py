"""Load data from FluidFlowerConfig experiments."""

import os
from typing import Literal

import darsia
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig


def load_data(
    config: FluidFlowerConfig,
    data: Literal["mass"],
    time: float,
    tol: float | None = None,
) -> darsia.Image | None:
    """Load data from a given FluidFlowerConfig.

    Args:
        config (FluidFlowerConfig): Configuration of the experiment.
        data (Literal["mass"]): Type of data to load. Currently only "mass" is supported.
        time (float): Time point at which to load the data (in seconds).
        tol (float | None): Tolerance for time matching (in hours). If None, exact match is
            required.

    Returns:
        darsia.Image: Loaded data image.

    """

    # Fetch available analyzed mass files
    if data == "mass":
        folder = config.analysis.mass.folder
    else:
        raise ValueError(f"Data type {data} not recognized.")
    available_files = sorted(
        folder / name for name in os.listdir(folder) if name.endswith(".npz")
    )

    # Fetch experiment protocol
    experiment = darsia.ProtocolledExperiment.init_from_config(config)

    # Find closest time point
    path = experiment.find_images_for_times(
        times=time,
        tol=tol * 3600 if tol is not None else None,
        # tol=None,
        data=available_files,
    )

    if path is None:
        return None
    else:
        return darsia.imread(path)
