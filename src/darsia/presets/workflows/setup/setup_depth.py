"""Standard workflow step to set up depth map from measurements."""

import logging
from pathlib import Path

import numpy as np

import darsia
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.setup.illustrations import save_scalar_map_illustration

logger = logging.getLogger(__name__)


def setup_depth_map(path: Path | list[Path], key="mean", show: bool = False) -> None:
    """Set up depth map from measurements.

    NOTE: This function stores the depth map in npz format according to the
    specifications in the config file.

    Args:
        path: Path to configuration file (needs to comply with FluidFlowerConfig).
        key: Column identifier in the csv file to use for interpolation (default: "mean").
        show: Whether to show the resulting depth map.

    """
    logger.info("\033[92mSetting up depth map from measurements...\033[0m")

    # ! ---- READ CONFIG ---- ! #
    config = FluidFlowerConfig(path, require_data=False, require_results=False)
    config.check("depth", "rig")

    # Mypy type checking
    for c in [
        config.depth,
        config.rig,
    ]:
        assert c is not None

    # Convert to depth map by interpolation
    proxy_image = darsia.Image(
        img=np.zeros(config.rig.resolution),
        width=config.rig.width,
        height=config.rig.height,
        scalar=True,
        space_dim=config.rig.dim,
    )
    depth_map = darsia.interpolate_to_image_from_csv(
        csv_file=config.depth.measurements,
        key=key,
        image=proxy_image,
        method="rbf",
    )
    depth_map_path = config.depth.depth_map.with_suffix(".npz")
    depth_map.save(depth_map_path)
    save_scalar_map_illustration(
        depth_map.img,
        config.depth.depth_map.with_suffix(".jpg"),
        title="Depth map",
        colorbar_label="Depth",
    )

    if show:
        depth_map.show(title="Depth map")
