"""Connect labels to physical facies."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import darsia
from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)


def setup_facies(cls: Rig, path: Path, show: bool = False):
    """Setup facies based on config file."""

    config = FluidFlowerConfig(path, require_data=False, require_results=False)
    config.check("facies", "labeling")

    # Mypy type checking
    assert config.labeling is not None
    assert config.facies is not None

    # ! ---- LOAD PRE-DEFINED LABELS ----

    labels = darsia.imread(config.labeling.labels)

    # ! ---- CONNECT LABELS TO FACIES ----
    if show:
        # Used for defining the config.
        labels.show(title="Labels")

    # Read facies groups from config
    id_label_map = config.facies.id_label_map
    groups = list(id_label_map.values())
    ids = list(id_label_map.keys())
    facies = darsia.group_labels(labels, groups=groups, values=ids)

    # Sanity check - ids. Check that all facies ids are defined in props.
    facies_props = pd.read_excel(config.facies.props)
    facies_ids = facies_props["id"].tolist()
    for facies_id in ids:
        assert facies_id in facies_ids, (
            f"Facies id {facies_id} not found in facies properties."
        )

    # Sanity check - groups. Check that all labels are assigned to a facies.
    unique_labels = set(np.unique(labels.img))
    assigned_labels = set([label for group in groups for label in group])
    assert unique_labels == assigned_labels, (
        "Some labels are not assigned to any facies. "
        f"Labels which are not set: {unique_labels - assigned_labels}. "
        f"Assigned labels which are not in label: {assigned_labels - unique_labels}"
    )

    if show:
        facies.show(title="Facies")

    # ! ---- SAVE FACIES ----
    facies.save(config.facies.path)
