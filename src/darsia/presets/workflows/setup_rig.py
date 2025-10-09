"""Step 3 of workflow.

Setup and storing of fluidflower object. This routine is time consuming and should be run only once per run.
It stores the baseline image, necessary corrections, the image porosity and the depth map,
adapted to the corrected baseline image etc.

"""

import logging
import time
from pathlib import Path
import darsia

from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig
# from darsia.presets.workflows.rig import Rig

# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# TODO use class Rig here.
def setup_rig(cls, path: Path, show: bool = False) -> None:
    """Setup and store fluidflower object.

    Args:
        cls: Class of the rig to be setup, e.g. ffum.MuseumRig
        path (Path): Path to the config file.
        show (bool): Whether to show intermediate results.

    """
    # Monitoring time of execution
    tic = time.time()

    # ! ---- LOAD RUN AND RIG ----
    config = FluidFlowerConfig(path)

    # Load imaging protocol
    experiment = darsia.ProtocolledExperiment(
        imaging_protocol=config.protocol.imaging,
        injection_protocol=config.protocol.injection,
        pressure_temperature_protocol=config.protocol.pressure_temperature,
        blacklist_protocol=config.protocol.blacklist,
        pad=config.data.pad,
    )

    # Setup fluidflower
    fluidflower = cls()
    # ffum.MuseumRig()
    fluidflower.setup(
        experiment=experiment,
        baseline_path=config.data.baseline,
        depth_map_path=config.depth.depth_map,
        labels_path=config.labeling.labels,
        correction_config_path=path,  # TODO replace with actual config file read from toml
        log=config.data.log,
    )

    # Save fluidflower
    fluidflower.save(config.data.results / "fluidflower")

    # Monitoring time of execution
    logger.info(f"Fluidflower setup in {time.time() - tic:.2f} s.")

    if show:
        fluidflower.baseline.show()
