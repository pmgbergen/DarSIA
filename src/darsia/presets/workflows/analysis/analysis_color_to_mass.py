"""Template for mass analysis."""

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


def analysis_color_to_mass(
    cls: type[Rig],
    path: Path | list[Path],
    rois: dict[str, darsia.CoordinateArray] | None = None,
    rois_and_labels: dict[str, tuple[int, darsia.CoordinateArray]] | None = None,
    show: bool = False,
    all: bool = False,
    use_facies: bool = True,
):
    # ! ---- LOAD RUN AND RIG ----
    config = FluidFlowerConfig(path, require_data=True)
    config.check(
        "protocol",
        "data",
        "color_paths",
        "rig",
    )

    # Mypy type checking
    assert config.protocol is not None
    assert config.protocol.imaging is not None
    assert config.protocol.injection is not None
    assert config.protocol.pressure_temperature is not None
    assert config.protocol.blacklist is not None
    assert config.data is not None
    assert config.color_paths is not None
    assert config.analysis is not None
    assert config.rig is not None
    assert config.color_to_mass is not None

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
    rois_and_labels = rois_and_labels or {}

    geometry = {key: fluidflower.geometry.subregion(roi) for key, roi in rois.items()}
    geometry.update(
        {
            key: fluidflower.geometry.subregion(roi)
            for key, (_, roi) in rois_and_labels.items()
        }
    )

    # ! ---- ANALYSIS ----

    if all:
        image_paths = config.data.data
    elif len(config.analysis.image_paths) > 0:
        image_paths = config.analysis.image_paths
    else:
        image_paths = experiment.find_images_for_times(
            times=config.analysis.image_times
        )
    assert len(image_paths) > 0, "No images found for analysis."

    # Plotting
    folder_mass = config.data.results / "mass"
    folder_mass.mkdir(parents=True, exist_ok=True)

    # Initialize DataFrame for storing integrated masses
    detected_cols = [f"{key}_detected_mass" for key in rois.keys()] + [
        f"{key}_detected_mass" for key in rois_and_labels.keys()
    ]
    detected_cols_g = [f"{key}_detected_mass_g" for key in rois.keys()] + [
        f"{key}_detected_mass_g" for key in rois_and_labels.keys()
    ]
    detected_cols_aq = [f"{key}_detected_mass_aq" for key in rois.keys()] + [
        f"{key}_detected_mass_aq" for key in rois_and_labels.keys()
    ]
    exact_cols = [f"{key}_exact_mass" for key in rois.keys()]
    columns = (
        ["time", "datetime", "image_stem"]
        + exact_cols
        + detected_cols
        + detected_cols_g
        + detected_cols_aq
    )
    mass_df = pd.DataFrame(columns=columns)
    # TODO control this path through toml/config.
    csv_path = config.data.results / "sparse_data" / "integrated_mass.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Loop over images and analyze
    for path in image_paths:
        # Extract color signal and assign mass
        img = fluidflower.read_image(path)
        mass_analysis_result = color_to_mass_analysis(img)

        # Log time
        time = mass_analysis_result.time

        # Fetch results
        mass = mass_analysis_result.mass
        mass_g = mass_analysis_result.mass_g
        mass_aq = mass_analysis_result.mass_aq

        # Store data to disk
        mass.save(folder_mass / f"{path.stem}.npz")  # Prepare row data for DataFrame
        row_data = {"time": time, "datetime": img.date, "image_stem": path.stem}

        # Compute exact mass in ROIs and add to row data
        for key, roi in rois.items():
            # Fetch exact mass from injection protocol
            exact_mass_roi = experiment.injection_protocol.injected_mass(img.date, roi)

            # Integrate over chosen roi
            mass_roi = geometry[key].integrate(mass.subregion(roi))
            mass_g_roi = geometry[key].integrate(mass_g.subregion(roi))
            mass_aq_roi = geometry[key].integrate(mass_aq.subregion(roi))

            # Store
            row_data[f"{key}_exact_mass"] = exact_mass_roi
            row_data[f"{key}_detected_mass"] = mass_roi
            row_data[f"{key}_detected_mass_g"] = mass_g_roi
            row_data[f"{key}_detected_mass_aq"] = mass_aq_roi

        # Compute integrated mass, mass_g, mass_aq in sub-ROIs and add to row data
        for key, (label, roi) in rois_and_labels.items():
            # Restrict mass arrays to labeled area.
            _mass = mass.copy()
            _mass.img[fluidflower.labels.img != label] = 0.0
            _mass_g = mass_g.copy()
            _mass_g.img[fluidflower.labels.img != label] = 0.0
            _mass_aq = mass_aq.copy()
            _mass_aq.img[fluidflower.labels.img != label] = 0.0

            # Integrate over chosen roi
            mass_roi = geometry[key].integrate(_mass.subregion(roi))
            mass_g_roi = geometry[key].integrate(_mass_g.subregion(roi))
            mass_aq_roi = geometry[key].integrate(_mass_aq.subregion(roi))

            # Store
            row_data[f"{key}_detected_mass"] = mass_roi
            row_data[f"{key}_detected_mass_g"] = mass_g_roi
            row_data[f"{key}_detected_mass_aq"] = mass_aq_roi

        # Add row to DataFrame using pd.concat for better performance
        new_row = pd.DataFrame([row_data])
        mass_df = pd.concat([mass_df, new_row], ignore_index=True)

        # Save DataFrame to CSV after each image analysis
        mass_df.to_csv(csv_path, index=False)  # Log the current analysis results
        logger.info(f"Processed {path.stem} at time {time}")
        for key in rois.keys():
            exact = row_data[f"{key}_exact_mass"]
            detected = row_data[f"{key}_detected_mass"]
            error = (detected - exact) / max(exact, 1e-8) if exact else 0
            logger.info(
                f"  {key}: detected={detected:.6f}, exact={exact:.6f}, "
                f"error={error:.2%}"
            )

        if show:
            import matplotlib.pyplot as plt

            img.show(title=f"Image at {path.stem}", delay=True)
            mass.show(title=f"Mass at {path.stem}", delay=True)
            mass_g.show(title=f"Mass G at {path.stem}", delay=True)
            mass_aq.show(title=f"Mass AQ at {path.stem}", delay=True)
            plt.show()
