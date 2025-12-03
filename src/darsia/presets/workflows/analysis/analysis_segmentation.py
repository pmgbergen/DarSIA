"""Template for segmentation analysis."""

import logging
from pathlib import Path

import darsia
from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.heterogeneous_color_to_mass_analysis import (
    HeterogeneousColorToMassAnalysis,
)
from darsia.presets.workflows.segmentation_contours import SegmentationContours
from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)


def analysis_segmentation(
    cls: type[Rig],
    path: Path | list[Path],
    show: bool = False,
    all: bool = False,
    use_facies: bool = True,
):
    # ! ---- LOAD RUN AND RIG ----
    config = FluidFlowerConfig(path, require_data=True, require_results=True)
    config.check(
        "data",
        "rig",
        "protocol",
        "analysis",
        "analysis.segmentation",
    )

    # Mypy type checking
    assert config.data is not None
    assert config.rig is not None
    assert config.rig.path is not None
    assert config.protocol is not None
    assert config.analysis is not None

    # ! ---- Load experiment
    experiment = darsia.ProtocolledExperiment.init_from_config(config)

    # ! ---- LOAD RIG ----
    fluidflower = cls.load(config.rig.path)
    fluidflower.load_experiment(experiment)
    if use_facies:
        fluidflower.labels = fluidflower.facies.copy()

    # ! ---- POROSITY-INFORMED AVERAGING ---- ! #
    image_porosity = fluidflower.image_porosity
    restoration = darsia.VolumeAveraging(
        rev=darsia.REV(size=0.005, img=fluidflower.baseline),
        mask=image_porosity,
    )

    # ! ---- FROM COLOR PATH TO MASS ----

    experiment_start = experiment.experiment_start
    state = experiment.pressure_temperature_protocol.get_state(experiment_start)
    gradient = experiment.pressure_temperature_protocol.get_gradient(experiment_start)
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

    # ! ---- IMAGES ----

    if all:
        image_paths = config.data.data
    elif len(config.analysis.data.image_paths) > 0:
        image_paths = config.analysis.data.image_paths
    else:
        image_paths = experiment.find_images_for_times(
            times=config.analysis.data.image_times
        )
    assert len(image_paths) > 0, "No images found for analysis."

    # ! ---- CONTOUR PLOTTING ---- ! #

    segmentation_contours = SegmentationContours(config.analysis.segmentation.config)

    # Loop over images and analyze
    for path in image_paths:
        # Extract color signal and assign mass
        img = fluidflower.read_image(path)
        mass_analysis_result = color_to_mass_analysis(img)

        # Produce contour images
        contour_image = segmentation_contours(
            img,
            saturation_g=mass_analysis_result.saturation_g,
            concentration_aq=mass_analysis_result.concentration_aq,
            mass=mass_analysis_result.mass,
        )

        if show:
            contour_image.show(
                title=f"Contours for {path.stem} | {img.time} seconds", delay=False
            )

        contour_path = config.analysis.segmentation.folder / f"{path.stem}.jpg"
        contour_image.write(contour_path, quality=80)
