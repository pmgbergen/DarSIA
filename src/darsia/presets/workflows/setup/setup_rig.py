"""Step 3 of workflow.

Setup and storing of fluidflower object. This routine is time consuming and should be run only once per run.
It stores the baseline image, necessary corrections, the image porosity and the depth map,
adapted to the corrected baseline image etc.

"""

import logging
import time
from pathlib import Path
from typing import Type

import darsia
from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.rig import Rig

# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def setup_rig(cls: Type[Rig], path: Path | list[Path], show: bool = False) -> None:
    """Setup and store fluidflower object.

    Args:
        cls: Class of the rig to be setup, e.g. ffum.MuseumRig
        path (Path): Path to the config file.
        show (bool): Whether to show intermediate results.

    """
    # Monitoring time of execution
    tic = time.time()

    # ! ---- LOAD RUN AND RIG ----
    config = FluidFlowerConfig(path, require_data=False, require_results=False)
    config.check("data", "depth", "labeling", "protocol")

    # Mypy type checking
    assert config.data is not None
    assert config.depth is not None
    assert config.labeling is not None
    assert config.facies is not None
    assert config.protocol is not None
    assert config.protocol.imaging is not None
    assert config.protocol.injection is not None
    assert config.protocol.pressure_temperature is not None

    # Load imaging protocol
    experiment = darsia.ProtocolledExperiment(
        data=config.data.data,
        imaging_protocol=config.protocol.imaging,
        injection_protocol=config.protocol.injection,
        pressure_temperature_protocol=config.protocol.pressure_temperature,
        blacklist_protocol=config.protocol.blacklist,
        pad=config.data.pad,
    )

    # Setup and save rig
    rig = cls()
    rig.setup(
        experiment=experiment,
        baseline_path=config.data.baseline,
        depth_map_path=config.depth.depth_map,
        labels_path=config.labeling.labels,
        facies_path=config.facies.path,
        facies_props_path=config.facies.props,
        config_path=path,  # TODO replace with actual config file read from toml
        log=config.data.results,
    )
    rig.save(config.data.results / "setup" / "rig")

    # Monitoring time of execution
    logger.info(f"Rig setup in {time.time() - tic:.2f} s.")

    if show:
        rig.baseline.show()
