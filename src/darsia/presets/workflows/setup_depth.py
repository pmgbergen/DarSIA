"""Standard workflow step to set up depth map from measurements."""

from pathlib import Path

import darsia
import numpy as np

from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig


def setup_depth_map(path: Path, key="mean") -> None:
    """Set up depth map from measurements.

    NOTE: This function stores the depth map in npz format according to the
    specifications in the config file.

    Args:
        path: Path to configuration file (needs to comply with FluidFlowerConfig).
        key: Column identifier in the csv file to use for interpolation (default: "mean").

    """
    print("Setting up depth map...")

    # ! ---- READ CONFIG ---- ! #
    config = FluidFlowerConfig(path)

    # Convert to depth map by interpolation
    proxy_image = darsia.Image(
        img=np.zeros(config.specs.resolution),
        width=config.specs.width,
        height=config.specs.height,
        scalar=True,
        space_dim=config.specs.dim,
    )
    depth_map = darsia.interpolate_to_image_from_csv(
        csv_file=config.depth.measurements,
        key=key,
        image=proxy_image,
        method="rbf",
    )
    depth_map.save(config.depth.depth_map.with_suffix(".npz"))
