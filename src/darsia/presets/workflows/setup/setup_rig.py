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
from darsia.presets.workflows.config.sections import (
    list_required_sections,
    required_sections,
)
from darsia.presets.workflows.rig import Rig

# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@required_sections(
    "data",
    "depth",
    "labeling",
    "facies",
    "protocols",
    "rig",
    "corrections",
    "image_porosity",
)
def setup_rig(cls: Type[Rig], path: Path | list[Path], show: bool = False,
              *, raw_build_request: dict | None = None) -> None:
    """Setup and store rig object.

    Parameters
    ----------
    cls : Type[Rig]
        Class of the rig to be setup.
    path : Path
        Path to the config file.
    show : bool
        Whether to show intermediate results.
    """

    logger.info("\033[92mSetting up rig...\033[0m")

    # Monitoring time of execution
    tic = time.time()

    # ! ---- LOAD RUN AND RIG ----
    from darsia.fluidflower.rig_raw_build import capture_configuration_sources

    configuration_sources = capture_configuration_sources(path)
    from darsia.presets.workflows.config.utils import _get_section_from_toml
    data_section = _get_section_from_toml(path, "data")
    raw_requested = data_section.get("raw_camera") is not None
    if raw_requested:
        from darsia.fluidflower.rig_raw_build import read_build_request
        read_build_request(raw_build_request, artifact_kind="raw_rig_baseline")
        if data_section.get("use_cache", False) is not False:
            raise ValueError("RAW setup must not create or write a data cache")
    config = FluidFlowerConfig(path, require_data=False, require_results=raw_requested)
    config.check(*list_required_sections(setup_rig))

    # Mypy type checking
    assert config.data is not None
    assert config.depth is not None
    assert config.labeling is not None
    assert config.facies is not None
    assert config.protocols is not None
    assert config.protocols.imaging is not None
    assert config.protocols.injection is not None
    assert config.protocols.pressure_temperature is not None

    if raw_requested:
        from darsia.fluidflower.rig_raw_basis import require_protocol_sources
        require_protocol_sources(
            read_build_request(raw_build_request, artifact_kind="raw_rig_baseline")[0],
            config.protocols)

    # Load imaging protocol
    experiment = darsia.ProtocolledExperiment(
        data=config.data.data,
        imaging_protocol=config.protocols.imaging,
        injection_protocol=config.protocols.injection,
        pressure_temperature_protocol=config.protocols.pressure_temperature,
        blacklist_protocol=config.protocols.blacklist,
    )

    # Determine effective baseline path, using cache when enabled.
    baseline_path = config.data.baseline
    raw_camera = getattr(config.data, "raw_camera", None)
    analysis_exclusions = getattr(config.rig, "analysis_exclusions", None)
    native_product = getattr(config.rig, "native_product", None)
    raw_build = None
    if raw_camera is not None:
        from darsia.fluidflower.rig_raw_build import begin_raw_rig_build

        raw_build = begin_raw_rig_build(
            path, baseline_path, raw_camera, config.rig.path,
            configuration_sources=configuration_sources,
            request_binding=raw_build_request,
            **({"analysis_exclusions": analysis_exclusions} if analysis_exclusions is not None else {}),
            **({"native_product": native_product} if native_product is not None else {}),
        )
    if config.data.use_cache and raw_camera is None:
        assert config.data.raw_cache is not None
        baseline_relative = None
        for folder in config.data.folders:
            try:
                baseline_relative = baseline_path.resolve().relative_to(
                    folder.resolve()
                )
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
        roi_registry=config.roi_registry,
        image_porosity_config=config.image_porosity,
        log=config.rig.path / "figures",
        show_plot=show,
        **({"raw_camera": raw_camera,
            "raw_baseline_sha256": raw_build.raw_source["sha256"],
            "raw_build_id": raw_build.operation_id,
            "raw_build": raw_build} if raw_build is not None else {}),
        **({"analysis_exclusions": analysis_exclusions} if analysis_exclusions is not None else {}),
        **({"native_product": native_product} if native_product is not None else {}),
    )
    rig.save(config.rig.path)
    if raw_build is not None:
        raw_build.publish(rig)

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
