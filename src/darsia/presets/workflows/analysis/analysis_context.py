"""Common analysis context for all analysis workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

import darsia
from darsia.presets.workflows.analysis.expert_knowledge import ExpertKnowledgeAdapter
from darsia.presets.workflows.color_embedding import (
    ColorEmbeddingRuntime,
    ColorPathEmbedding,
)
from darsia.presets.workflows.config.data_registry import DataRegistry
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.config.time_data import TimeData
from darsia.presets.workflows.heterogeneous_color_to_mass_analysis import (
    HeterogeneousColorToMassAnalysis,
)
from darsia.presets.workflows.mode_resolution import mode_requires_color_to_mass
from darsia.presets.workflows.restoration import build_restoration
from darsia.presets.workflows.rig import Rig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def infer_require_color_to_mass_from_config(
    path: Path | list[Path],
    *,
    include_segmentation: bool = False,
    include_fingers: bool = False,
    include_thresholding: bool = False,
    include_mass: bool = False,
    include_volume: bool = False,
) -> bool:
    """Infer if color-to-mass initialization is required for selected analyses."""
    if include_mass or include_volume:
        return True

    config = FluidFlowerConfig(path, require_results=True, require_data=True)
    if config.analysis is None:
        return True

    modes: list[str] = []
    if include_segmentation and config.analysis.segmentation is not None:
        segmentation_config = config.analysis.segmentation.config
        if isinstance(segmentation_config, dict):
            modes.extend(
                cfg.mode for cfg in segmentation_config.values() if cfg.mode is not None
            )
        elif segmentation_config.mode is not None:
            modes.append(segmentation_config.mode)

    if include_fingers and config.analysis.fingers is not None:
        fingers_config = config.analysis.fingers.config
        if isinstance(fingers_config, dict):
            modes.extend(
                cfg.mode for cfg in fingers_config.values() if cfg.mode is not None
            )
        elif fingers_config.mode is not None:
            modes.append(fingers_config.mode)

    if include_thresholding and config.analysis.thresholding is not None:
        modes.extend(
            layer.mode for layer in config.analysis.thresholding.layers.values()
        )

    if len(modes) == 0:
        return True
    return any(mode_requires_color_to_mass(mode) for mode in modes)


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
    color_embedding_runtime: ColorEmbeddingRuntime | None = None


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
        # Resolve registry reference if sub_config.data is a (list of) raw registry key
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
    elif hasattr(sub_config, "data") and isinstance(sub_config.data, TimeData):
        image_paths = []
        image_paths += experiment.find_images_for_paths(
            paths=sub_config.data.image_paths
        )
        image_paths += experiment.find_images_for_times(
            times=sub_config.data.image_times, data=source
        )
    else:
        # Support legacy format, but throw deprecation warning.
        warn("Using legacy image_paths format in sub_config.", DeprecationWarning)
        image_paths = []
        if hasattr(sub_config, "image_paths") and sub_config.image_paths is not None:
            paths = sub_config.image_paths
            image_paths += experiment.find_images_for_paths(paths=paths)
        if (
            hasattr(sub_config, "data")
            and hasattr(sub_config.data, "image_paths")
            and sub_config.data.image_paths is not None
        ):
            paths = sub_config.data.image_paths
            image_paths += experiment.find_images_for_paths(paths=paths)
        if hasattr(sub_config, "image_times") and sub_config.image_times is not None:
            times = sub_config.image_times
            image_paths += experiment.find_images_for_times(times=times, data=source)
        if (
            hasattr(sub_config, "data")
            and hasattr(sub_config.data, "image_times")
            and sub_config.data.image_times is not None
        ):
            times = sub_config.data.image_times
            image_paths += experiment.find_images_for_times(times=times, data=source)

    assert len(image_paths) > 0, "No images found for analysis."
    return image_paths


def _build_color_to_mass_analysis(
    config: FluidFlowerConfig,
    experiment: darsia.ProtocolledExperiment,
    rig: Rig,
    restoration: darsia.VolumeAveraging | darsia.TVD | None,
    expert_knowledge_adapter: ExpertKnowledgeAdapter | None,
) -> HeterogeneousColorToMassAnalysis:
    # ! ---- COLOR EMBEDDING ----
    assert config.color is not None
    assert config.analysis is not None
    assert config.analysis.mass is not None
    embedding = config.color.resolve(config.analysis.mass.color)
    if not isinstance(embedding, ColorPathEmbedding):
        raise NotImplementedError(
            "Mass analysis currently only supports color-path embeddings."
        )

    # ! ---- ANALYSIS LABELS ----
    analysis_labels = embedding.get_labels(rig)

    # ! ---- CO2 MASS ANALYSIS ----
    experiment_start = experiment.experiment_start
    state = experiment.pressure_temperature_protocol.get_state(experiment_start)
    gradient = experiment.pressure_temperature_protocol.get_gradient(experiment_start)
    co2_mass_analysis = darsia.CO2MassAnalysis(
        baseline=rig.baseline,
        atmospheric_pressure=state.pressure,
        atmospheric_temperature=state.temperature,
        atmospheric_pressure_gradient=gradient.pressure,
        atmospheric_temperature_gradient=gradient.temperature,
    )

    # ! ---- COLOR TO MASS ANALYSIS ----
    color_to_mass_analysis = HeterogeneousColorToMassAnalysis.load(
        folder=embedding.color_to_mass_folder,
        baseline=rig.baseline,
        labels=analysis_labels,
        co2_mass_analysis=co2_mass_analysis,
        geometry=rig.geometry,
        restoration=restoration,
        basis=embedding.basis,
        expert_knowledge_adapter=expert_knowledge_adapter,
    )
    return color_to_mass_analysis


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

    # ! ---- SELECT IMAGE PATHS ----
    image_paths = select_image_paths(
        config,
        experiment,
        all=all,
        sub_config=config.analysis,
        data_registry=config.data.registry,
    )

    # ! ---- RESTORATION ----
    restoration = build_restoration(config.restoration, fluidflower)

    # ! ---- EXPERT KNOWLEDGE ADAPTER (for all analyses) ----
    expert_knowledge_adapter = ExpertKnowledgeAdapter.from_config(
        config=(
            config.analysis.expert_knowledge if config.analysis is not None else None
        ),
        roi_registry=config.roi_registry,
    )

    # ! ---- COLOR EMBEDDING RUNTIME CONTEXT (for all analyses) ----
    color_embedding_runtime = ColorEmbeddingRuntime(rig=fluidflower)

    # ! ---- COLOR-TO-MASS ANALYSIS (only if required) ----
    if require_color_to_mass:
        color_to_mass_analysis = _build_color_to_mass_analysis(
            config=config,
            experiment=experiment,
            rig=fluidflower,
            restoration=restoration,
            expert_knowledge_adapter=expert_knowledge_adapter,
        )

        # TODO: refactor and remove analysis_labels.
        embedding = config.color.resolve(config.analysis.mass.color)
        analysis_labels = embedding.get_labels(fluidflower)
    else:
        color_to_mass_analysis = None
        analysis_labels = None

    return AnalysisContext(
        config=config,
        experiment=experiment,
        fluidflower=fluidflower,
        analysis_labels=analysis_labels,  # TODO: remove, not used much
        image_paths=image_paths,
        restoration=restoration,
        color_to_mass_analysis=color_to_mass_analysis,
        expert_knowledge_adapter=expert_knowledge_adapter,
        color_embedding_runtime=color_embedding_runtime,
    )
