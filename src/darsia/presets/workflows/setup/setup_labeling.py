"""Standardized workflow to setup geometric labels for FluidFlower data."""

import logging
from pathlib import Path

import numpy as np

import darsia
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.setup.illustrations import save_discrete_map_illustration

logger = logging.getLogger(__name__)


def segment_colored_image(path: Path, show: bool = False):
    """Segment colored image based on config file."""

    logger.info("\033[92mSegmenting colored image...\033[0m")

    config = FluidFlowerConfig(path, require_data=False, require_results=False)
    config.check("labeling", "rig")

    # Mypy type checking
    for c in [
        config.rig,
        config.labeling,
    ]:
        assert c is not None

    # Step 1: Unsupervised labeling of manually colored image
    manually_colored_image = darsia.imread(
        config.labeling.colored_image,
        dim=config.rig.dim,
        width=config.rig.width,
        height=config.rig.height,
    )

    labels = darsia.label_image(
        manually_colored_image,
        significance=config.labeling.rtol,  # ignore labels which cover too little area
        ensure_connectivity=config.labeling.ensure_connectivity,
    )
    logger.info("Num unique labels: %d", len(np.unique(labels.img)))

    # Step 2: Unite same labels based on TOML file
    if show:
        labels.show(title="Labeled image")
    if config.labeling.unite_labels is not None:
        unite_labels = [tuple(group) for group in config.labeling.unite_labels]
        labels = darsia.group_labels(labels, unite_labels)
        labels = darsia.make_consecutive(labels)
        logger.info("Number unique labels: %d", len(np.unique(labels.img)))
        if show:
            labels.show()

    # Step 3: Save labels
    labels.save(config.labeling.labels)
    save_discrete_map_illustration(
        labels.img,
        config.labeling.labels.with_suffix(".jpg"),
        title="Labels",
        colorbar_label="Label id",
    )
