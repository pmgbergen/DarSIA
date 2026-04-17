"""Common analysis context for all analysis workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import darsia
from darsia.presets.workflows.basis import select_labels_for_basis
from darsia.presets.workflows.config.data_registry import DataRegistry
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.heterogeneous_color_to_mass_analysis import (
    HeterogeneousColorToMassAnalysis,
)
from darsia.presets.workflows.analysis.expert_knowledge import ExpertKnowledgeAdapter
from darsia.presets.workflows.restoration import build_restoration
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
        restoration: Restoration model (e.g. VolumeAveraging or TVD), or None
            if no restoration is configured.  Available to all analysis workflows.
        color_to_mass_analysis: The color to mass analysis pipeline
            (for mass/volume/segmentation).

    """

    config: FluidFlowerConfig
    experiment: darsia.ProtocolledExperiment
    fluidflower: Rig
    analysis_labels: darsia.Image
    image_paths: list[Path]

    # Optional - only initialized for mass/volume/segmentation analyses
    restoration: darsia.VolumeAveraging | darsia.TVD | None = None
    color_to_mass_analysis: HeterogeneousColorToMassAnalysis | None = None
    expert_knowledge_adapter: ExpertKnowledgeAdapter | None = None


def select_image_paths(
    config: FluidFlowerConfig,
    experiment: darsia.ProtocolledExperiment,
    all: bool = False,
    sub_config=None,
    source: Path | None = None,
    data_registry: DataRegistry | None = None,
) -> list[Path]:
    """Select image paths based on configuration and flags.

    Args:
        config: The FluidFlower configuration.
        experiment: The protocolled experiment.
        all: Whether to use all images.
        sub_config: Optional sub-configuration for the analysis.
        source: Optional source path for time-based image lookup.
        data_registry: Optional global data registry for resolving registry-based
            data references in the sub-configuration.

    Returns:
        List of image paths to analyze.

    """
    assert config.data is not None

    if all or sub_config is None:
        assert config.data.data is not None
        paths = config.data.data
        image_paths = experiment.find_images_for_paths(paths=paths)
    elif hasattr(sub_config, "data") and isinstance(sub_config.data, (str, list)):
        # Resolve registry reference if sub_config.data is a raw registry key
        if data_registry is not None:
            resolved = data_registry.resolve(sub_config.data)
            if len(resolved.image_paths) > 0:
                image_paths = experiment.find_images_for_paths(
                    paths=resolved.image_paths
                )
            else:
                image_paths = experiment.find_images_for_times(
                    times=resolved.image_times, data=source
                )
        else:
            raise ValueError(
                "sub_config.data is a registry key reference but no data_registry "
                "was provided to resolve it."
            )
    elif (
        hasattr(sub_config, "data")
        and sub_config.data is not None
        and len(sub_config.data.image_paths) > 0
    ):
        paths = sub_config.data.image_paths
        image_paths = experiment.find_images_for_paths(paths=paths)
    elif (
        hasattr(sub_config, "image_paths")
        and sub_config.image_paths is not None
        and len(sub_config.image_paths) > 0
    ):
        # Fallback for cropping which uses sub_config.image_paths directly
        paths = sub_config.image_paths
        image_paths = experiment.find_images_for_paths(paths=paths)
    else:
        # Use times from config
        if hasattr(sub_config, "data") and sub_config.data is not None:
            times = sub_config.data.image_times
        elif hasattr(sub_config, "image_times"):
            times = sub_config.image_times
        else:
            raise ValueError("No image paths or times specified in config.")
        image_paths = experiment.find_images_for_times(times=times, data=source)

    assert len(image_paths) > 0, "No images found for analysis."
    return image_paths


def prepare_analysis_context(
    cls: type[Rig],
    path: Path | list[Path],
    all: bool = False,
    require_color_to_mass: bool = False,
) -> AnalysisContext:
    """Prepare common analysis context.

    This function initializes all shared objects needed for analysis workflows.
    Restoration is always built from the config (if a [restoration] section is
    present) so it is available to all analysis workflows.  When
    `require_color_to_mass` is True, the color_to_mass_analysis pipeline is
    also initialized and receives the restoration object.

    Args:
        cls: Rig class.
        path: Path or list of paths to config files.
        all: Whether to use all images.
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
    fluidflower = cls.load(config.rig.path, config.corrections)
    fluidflower.load_experiment(experiment)
    if require_color_to_mass:
        assert config.color_to_mass is not None
        selected_basis, analysis_labels = select_labels_for_basis(
            fluidflower, config.color_to_mass.basis
        )
    else:
        analysis_labels = fluidflower.labels

    # ! ---- SELECT IMAGE PATHS ----
    image_paths = select_image_paths(
        config,
        experiment,
        all=all,
        sub_config=config.analysis,
        data_registry=config.data.registry,
    )

    # ! ---- RESTORATION ----
    # Always build restoration so it is available to all analysis workflows.
    restoration = build_restoration(config.restoration, fluidflower)

    color_to_mass_analysis = None
    expert_knowledge_adapter = ExpertKnowledgeAdapter.from_config(
        config=config.analysis.expert_knowledge if config.analysis is not None else None,
        roi_registry=config.roi_registry,
    )

    if require_color_to_mass:
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
            labels=analysis_labels,
            co2_mass_analysis=co2_mass_analysis,
            geometry=fluidflower.geometry,
            restoration=restoration,
            basis=config.color_to_mass.basis,
            expert_knowledge_adapter=expert_knowledge_adapter,
        )

    return AnalysisContext(
        config=config,
        experiment=experiment,
        fluidflower=fluidflower,
        analysis_labels=analysis_labels,
        image_paths=image_paths,
        restoration=restoration,
        color_to_mass_analysis=color_to_mass_analysis,
        expert_knowledge_adapter=expert_knowledge_adapter,
    )
