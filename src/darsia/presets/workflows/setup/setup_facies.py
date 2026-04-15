"""Connect labels to physical facies."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import darsia
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.setup.illustrations import save_discrete_map_illustration
from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)


def setup_facies(cls: Rig, path: Path, show: bool = False):
    """Setup facies based on config file."""

    logger.info("\033[92mSetting up facies...\033[0m")

    config = FluidFlowerConfig(path, require_data=False, require_results=False)
    config.check("facies", "labeling")

    # Mypy type checking
    assert config.labeling is not None
    assert config.facies is not None

    # ! ---- LOAD PRE-DEFINED LABELS ----
    labels = darsia.imread(config.labeling.labels)

    # ! ---- CONNECT LABELS TO FACIES ----

    # Read facies groups from config
    all_label_ids = set(np.unique(labels.img))
    label_to_facies_map = config.facies.label_to_facies_map

    # Fill in missing labels with canonical facies ids.
    for label_id in all_label_ids:
        if label_id not in label_to_facies_map:
            label_to_facies_map[label_id] = label_id

    # Reassign labels to facies ids.
    facies = darsia.reassign_labels(labels, label_to_facies_map)

    # Sanity check - ids. Check that all facies ids are defined in props.
    facies_props = pd.read_excel(config.facies.props)
    facies_ids = facies_props["id"].tolist()
    for facies_id in np.unique(facies.img):
        assert (
            facies_id in facies_ids
        ), f"Facies id {facies_id} not found in facies properties."

    # ! ---- SAVE FACIES ----
    facies.save(config.facies.path)
    save_discrete_map_illustration(
        facies.img,
        config.facies.path.with_suffix(".jpg"),
        title="Facies",
        colorbar_label="Facies id",
    )
    if show:
        facies.show(title="Facies")
