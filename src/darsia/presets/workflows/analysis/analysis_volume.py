"""Template for color signal analysis."""

import logging
from pathlib import Path

import pandas as pd

import darsia
from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.heterogeneous_color_to_mass_analysis import (
    HeterogeneousColorToMassAnalysis,
)
from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)


def analysis_volume(
    cls: type[Rig],
    path: Path,
    show: bool = False,
    all: bool = False,
    use_facies: bool = True,
    rois: dict[str, darsia.CoordinateArray] | None = None,
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
    # ! ---- GEOMETRY FOR INTEGRATION ---- ! #

    # Define default ROIs
    rois = rois or {
        "full": darsia.CoordinateArray(
            [fluidflower.baseline.origin, fluidflower.baseline.opposite_corner]
        )
    }
    geometry = {key: fluidflower.geometry.subregion(roi) for key, roi in rois.items()}

    # Initialize DataFrame for storing integrated masses
    detected_volume_g_cols = [f"{key}_detected_volume_g" for key in rois.keys()]
    detected_volume_aq_cols = [f"{key}_detected_volume_aq" for key in rois.keys()]
    columns = (
        ["time", "datetime", "stem"] + detected_volume_g_cols + detected_volume_aq_cols
    )
    volume_df = pd.DataFrame(columns=columns)
    csv_path = config.data.results / "sparse_data" / "integrated_volume.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Storing and plotting
    folder_saturation_g = config.data.results / "saturation_g"
    folder_concentration_aq = config.data.results / "concentration_aq"

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

    # ! ---- ANALYSIS ----

    # Loop over images and analyze
    for path in image_paths:
        # Extract color signal and assign mass
        img = fluidflower.read_image(path)
        mass_analysis_result = color_to_mass_analysis(img)

        # Log time
        time = mass_analysis_result.time

        # Fetch results
        saturation_g = mass_analysis_result.saturation_g
        concentration_aq = mass_analysis_result.concentration_aq

        # Store coarse data to disk
        saturation_g.save(folder_saturation_g / f"{path.stem}.npz")
        concentration_aq.save(folder_concentration_aq / f"{path.stem}.npz")

        # Prepare row data for DataFrame
        row_data = {"time": time, "datetime": img.date, "stem": path.stem}

        # Compute exact mass in ROIs and add to row data
        for key, roi in rois.items():
            # Build effective aqueous saturation
            saturation_aq = concentration_aq.copy()
            saturation_aq.img *= 1 - saturation_g.img

            # Integrate over chosen roi
            volume_g_roi = geometry[key].integrate(saturation_g.subregion(roi))
            volume_aq_roi = geometry[key].integrate(saturation_aq.subregion(roi))

            # Store
            row_data[f"{key}_detected_volume_g"] = volume_g_roi
            row_data[f"{key}_detected_volume_aq"] = volume_aq_roi

        # Add row to DataFrame using pd.concat for better performance
        new_row = pd.DataFrame([row_data])
        volume_df = pd.concat([volume_df, new_row], ignore_index=True)

        # Save DataFrame to CSV after each image analysis
        volume_df.to_csv(csv_path, index=False)

        # Log the current analysis results
        logger.info(f"Processed {path.stem} at time {time}")
        for key in rois.keys():
            detected_g = row_data[f"{key}_detected_volume_g"]
            detected_aq = row_data[f"{key}_detected_volume_aq"]
            logger.info(
                f"  {key}: detected_g={detected_g:.6f}, detected_aq={detected_aq:.6f}"
            )
