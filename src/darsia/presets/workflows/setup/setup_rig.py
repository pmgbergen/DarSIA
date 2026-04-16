"""Setup/storing/deleting main rig object. This routine is time consuming and should be run
only once per run. It stores the baseline image, necessary corrections, the image porosity
and the depth map, adapted to the corrected baseline image etc.

"""

import logging
import shutil
import time
from pathlib import Path
from typing import Type

import darsia
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.rig import Rig

# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def setup_rig(cls: Type[Rig], path: Path | list[Path], show: bool = False) -> None:
    """Setup and store rig object.

    Args:
        cls: Class of the rig to be setup
        path (Path): Path to the config file.
        show (bool): Whether to show intermediate results.

    """

    logger.info("\033[92mSetting up rig...\033[0m")

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

    # Determine effective baseline path, using cache when enabled.
    baseline_path = config.data.baseline
    if config.data.use_cache:
        assert config.data.raw_cache is not None
        baseline_relative = None
        for folder in config.data.folders:
            try:
                baseline_relative = baseline_path.resolve().relative_to(folder.resolve())
                break
            except ValueError:
                continue
        if baseline_relative is None:
            baseline_relative = Path(baseline_path.name)
        cache_path = config.data.raw_cache / baseline_relative.with_suffix(".npz")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            baseline_path = cache_path
        else:
            original_baseline = darsia.imread(baseline_path)
            original_baseline.save(cache_path)
            baseline_path = cache_path

    # Setup and save rig
    rig = cls()
    rig.setup(
        experiment=experiment,
        baseline_path=baseline_path,
        depth_map_path=config.depth.depth_map,
        labels_path=config.labeling.labels,
        facies_path=config.facies.path,
        facies_props_path=config.facies.props,
        corrections_config=config.corrections,
        image_porosity_config=config.image_porosity,
        log=config.rig.path / "log",
        show_plot=show,
    )
    rig.save(config.rig.path)

    # Monitoring time of execution
    logger.info(f"Rig setup in {time.time() - tic:.2f} s.")

    if show:
        rig.baseline.show()


def delete_rig(cls: Type[Rig], path: Path | list[Path], show: bool = False) -> None:
    """Reset rig by deleting existing results and re-running setup."""
    logger.warning(
        """\033[91mResetting existing results. Use with caution as this will delete """
        """existing results.\033[0m"""
    )
    rig_path = FluidFlowerConfig(
        path, require_data=False, require_results=True
    ).rig.path
    if rig_path.exists():
        logger.info("Deleting existing rig...")
        user_input = input(
            """\033[91mAre you sure you want to delete the existing rig? """
            """This action cannot be undone. (y/n): \033[0m"""
        )
        if user_input.lower() == "y":
            shutil.rmtree(rig_path, ignore_errors=True)
            logger.info("Rig deleted.")
        else:
            logger.info("Rig deletion aborted.")
    else:
        logger.info("No existing rig found to reset.")
