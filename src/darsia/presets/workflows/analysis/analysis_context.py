"""Common analysis context for all analysis workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import darsia
from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.heterogeneous_color_to_mass_analysis import (
    HeterogeneousColorToMassAnalysis,
)
from darsia.presets.workflows.rig import Rig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class AnalysisContext:
    """Common objects prepared for analysis workflows.

    This context is shared across all analysis types (cropping, mass, volume,
    segmentation) to avoid redundant initialization of heavy objects.

    Attributes:
        config: The FluidFlower configuration.
        experiment: The protocolled experiment.
        fluidflower: The loaded rig instance.
        image_paths: List of image paths to analyze.
        restoration: Volume averaging restoration (for mass/volume/segmentation).
        color_to_mass_analysis: The color to mass analysis pipeline
            (for mass/volume/segmentation).

    """

    config: FluidFlowerConfig
    experiment: darsia.ProtocolledExperiment
    fluidflower: Rig
    image_paths: list[Path]

    # Optional - only initialized for mass/volume/segmentation analyses
    restoration: darsia.VolumeAveraging | None = None
    color_to_mass_analysis: HeterogeneousColorToMassAnalysis | None = None


def select_image_paths(
    config: FluidFlowerConfig,
    experiment: darsia.ProtocolledExperiment,
    all: bool = False,
    sub_config=None,
) -> list[Path]:
    """Select image paths based on configuration and flags.

    Args:
        config: The FluidFlower configuration.
        experiment: The protocolled experiment.
        all: Whether to use all images.

    Returns:
        List of image paths to analyze.

    """
    assert config.data is not None

    if all or sub_config is None:
        assert config.data.data is not None
        image_paths = config.data.data
    elif (
        hasattr(sub_config, "data")
        and sub_config.data is not None
        and len(sub_config.data.image_paths) > 0
    ):
        image_paths = sub_config.data.image_paths
    elif (
        hasattr(sub_config, "image_paths")
        and sub_config.image_paths is not None
        and len(sub_config.image_paths) > 0
    ):
        # Fallback for cropping which uses sub_config.image_paths directly
        image_paths = sub_config.image_paths
    else:
        # Use times from config
        if hasattr(sub_config, "data") and sub_config.data is not None:
            times = sub_config.data.image_times
        elif hasattr(sub_config, "image_times"):
            times = sub_config.image_times
        else:
            raise ValueError("No image paths or times specified in config.")
        image_paths = experiment.find_images_for_times(times=times)

    assert len(image_paths) > 0, "No images found for analysis."
    return image_paths


def prepare_analysis_context(
    cls: type[Rig],
    path: Path | list[Path],
    all: bool = False,
    use_facies: bool = True,
    require_color_to_mass: bool = False,
) -> AnalysisContext:
    """Prepare common analysis context.

    This function initializes all shared objects needed for analysis workflows.
    When `require_color_to_mass` is True, it also initializes the restoration
    and color_to_mass_analysis pipelines.

    Args:
        cls: FluidFlower rig class.
        path: Path or list of paths to config files.
        all: Whether to use all images.
        use_facies: Whether to use facies as labels.
        require_color_to_mass: Whether to initialize the color-to-mass pipeline.

    Returns:
        AnalysisContext with all common objects initialized.

    """
    # ! ---- LOAD CONFIG ----
    config = FluidFlowerConfig(path, require_results=True, require_data=True)
    config.check("analysis", "protocol", "data", "rig")

    # Mypy type checking
    assert config.rig is not None
    assert config.rig.path is not None
    assert config.data is not None
    assert config.protocol is not None
    assert config.analysis is not None

    # ! ---- LOAD EXPERIMENT ----
    experiment = darsia.ProtocolledExperiment.init_from_config(config)

    # ! ---- LOAD RIG ----
    fluidflower = cls.load(config.rig.path)
    fluidflower.load_experiment(experiment)
    if use_facies:
        fluidflower.labels = fluidflower.facies.copy()

    # ! ---- SELECT IMAGE PATHS ----
    image_paths = select_image_paths(
        config, experiment, all=all, sub_config=config.analysis
    )

    # Initialize optional components
    restoration = None
    color_to_mass_analysis = None

    if require_color_to_mass:
        # ! ---- POROSITY-INFORMED AVERAGING ----
        image_porosity = fluidflower.image_porosity
        restoration = darsia.VolumeAveraging(
            rev=darsia.REV(size=0.005, img=fluidflower.baseline),
            mask=image_porosity,
        )

        # ! ---- FROM COLOR PATH TO MASS ----
        assert config.color_to_mass is not None

        experiment_start = experiment.experiment_start
        state = experiment.pressure_temperature_protocol.get_state(experiment_start)
        gradient = experiment.pressure_temperature_protocol.get_gradient(
            experiment_start
        )
        co2_mass_analysis = darsia.CO2MassAnalysis(
            baseline=fluidflower.baseline,
            atmospheric_pressure=state.pressure,
            atmospheric_temperature=state.temperature,
            atmospheric_pressure_gradient=gradient.pressure,
            atmospheric_temperature_gradient=gradient.temperature,
        )

        color_to_mass_analysis = HeterogeneousColorToMassAnalysis.load(
            folder=config.color_to_mass.calibration_folder,
            baseline=fluidflower.baseline,
            labels=fluidflower.labels,
            co2_mass_analysis=co2_mass_analysis,
            geometry=fluidflower.geometry,
            restoration=restoration,
        )

    return AnalysisContext(
        config=config,
        experiment=experiment,
        fluidflower=fluidflower,
        image_paths=image_paths,
        restoration=restoration,
        color_to_mass_analysis=color_to_mass_analysis,
    )
