"""Standardized workflow to setup geometric labels for FluidFlower data."""

from pathlib import Path
import darsia
from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig
import numpy as np
import logging

logger = logging.getLogger(__name__)


def segment_colored_image(config_path: Path, show: bool = False):
    """Segment colored image based on config file."""

    config = FluidFlowerConfig(Path(config_path))

    # Step 1: Unsupervised labeling of manually colored image
    manually_colored_image = darsia.imread(
        config.labeling.colored_image,
        dim=config.specs.dim,
        width=config.specs.width,
        height=config.specs.height,
    )

    labels = darsia.label_image(
        manually_colored_image,
        significance=0.001,  # ignore labels which cover less than 0.1% of the image area
        ensure_connectivity=True,  # ensure that labels are connected regions
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
