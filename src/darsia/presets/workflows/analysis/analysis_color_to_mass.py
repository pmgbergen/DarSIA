"""Template for mass analysis."""

import logging
from pathlib import Path

import pandas as pd
import darsia
from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.heterogeneous_color_to_mass_analysis import (
    HeterogeneousColorToMassAnalysis,
)


logger = logging.getLogger(__name__)


def analysis_color_to_mass(
    cls,
    path: Path,
    rois: dict[str, darsia.CoordinateArray] | None = None,
    rois_and_labels: dict[str, tuple[int, darsia.CoordinateArray]] | None = None,
    test_run: bool = False,
    show: bool = False,
    use_facies: bool = True,
    **kwargs,
):
    # ! ---- LOAD RUN AND RIG ----
    config = FluidFlowerConfig(path)
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
    experiment = darsia.ProtocolledExperiment(
        data=config.data.data,
        imaging_protocol=config.protocol.imaging,
        injection_protocol=config.protocol.injection,
        pressure_temperature_protocol=config.protocol.pressure_temperature,
        blacklist_protocol=config.protocol.blacklist,
        pad=config.data.pad,
    )

    # ! ---- LOAD RIG ----
    fluidflower = cls()
    fluidflower.load(config.rig.path)
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
    # ! ---- ANALYSIS ----

    if test_run:
        image_paths = config.color_to_mass.calibration_image_paths
        if len(image_paths) == 0:
            image_paths = experiment.find_images_for_times(
                times=config.color_to_mass.calibration_image_times
            )
    else:
        image_paths = config.data.data

    # Plotting
    (config.data.results / "mass").mkdir(parents=True, exist_ok=True)
    (config.data.results / "saturation_g").mkdir(parents=True, exist_ok=True)
    (config.data.results / "concentration_aq").mkdir(parents=True, exist_ok=True)

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

    # Initialize DataFrame for storing integrated masses
    detected_cols = [f"{key}_detected_mass" for key in rois.keys()]
    exact_cols = [f"{key}_exact_mass" for key in rois.keys()]
    columns = ["time", "datetime", "image_stem"] + detected_cols + exact_cols
    mass_df = pd.DataFrame(columns=columns)
    csv_path = config.data.results / "sparse_data" / "integrated_masses.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Loop over images and analyze
    for path in image_paths:
        # Extract color signal and assign mass
        img = fluidflower.read_image(path)
        img.show()
        print(img.time, img.date)
        mass_analysis_result = color_to_mass_analysis(img)

        # Log time
        time = mass_analysis_result.time

        # Fetch results
        mass = mass_analysis_result.mass
        mass_g = mass_analysis_result.mass_g
        mass_aq = mass_analysis_result.mass_aq
        saturation = mass_analysis_result.saturation_g
        concentration = mass_analysis_result.concentration_aq

        # Store coarse data to disk
        mass.save(config.data.results / "mass" / f"{path.stem}.npz")
        saturation.save(config.data.results / "saturation_g" / f"{path.stem}.npz")
        concentration.save(
            config.data.results / "concentration_aq" / f"{path.stem}.npz"
        )

        # Prepare row data for DataFrame
        row_data = {"time": time, "datetime": img.date, "path": path.stem}

        # Compute exact mass in ROIs and add to row data
        for key, roi in rois.items():
            # Compute mass in the analysed data
            mass_roi = geometry[key].integrate(mass.subregion(roi))
            row_data[f"{key}_detected_mass"] = mass_roi

            # Fetch exact mass from injection protocol
            exact_mass_roi = experiment.injection_protocol.injected_mass(img.date, roi)
            row_data[f"{key}_exact_mass"] = exact_mass_roi

        # Compute integrated mass, mass_g, mass_aq in sub-ROIs and add to row data
        for key, (label, roi) in rois_and_labels.items():
            # Compute mass in the analysed data
            _mass = mass.copy()
            _mass.img[fluidflower.labels.img != label] = 0.0
            _mass_g = mass_g.copy()
            _mass_g.img[fluidflower.labels.img != label] = 0.0
            _mass_aq = mass_aq.copy()
            _mass_aq.img[fluidflower.labels.img != label] = 0.0
            mass_roi = geometry[key].integrate(_mass.subregion(roi))
            mass_g_roi = geometry[key].integrate(_mass_g.subregion(roi))
            mass_aq_roi = geometry[key].integrate(_mass_aq.subregion(roi))
            row_data[f"{key}_detected_mass"] = mass_roi
            row_data[f"{key}_detected_mass_g"] = mass_g_roi
            row_data[f"{key}_detected_mass_aq"] = mass_aq_roi

        # Add row to DataFrame using pd.concat for better performance
        new_row = pd.DataFrame([row_data])
        mass_df = pd.concat([mass_df, new_row], ignore_index=True)

        # Save DataFrame to CSV after each image analysis
        mass_df.to_csv(csv_path, index=False)

        # Log the current analysis results
        logger.info(f"Processed {path.stem} at time {time}")
        for key in rois.keys():
            detected = row_data[f"{key}_detected_mass"]
            exact = row_data[f"{key}_exact_mass"]
            error = abs(detected - exact) / max(exact, 1e-8) if exact else 0
            logger.info(
                f"  {key}: detected={detected:.6f}, exact={exact:.6f}, "
                f"error={error:.2%}"
            )

        if show:
            import matplotlib.pyplot as plt

            img.show(title=f"Image at {path.stem}", delay=True)
            mass.show(title=f"Mass at {path.stem}", delay=True)
            saturation.show(title=f"Saturation at {path.stem}", delay=True)
            concentration.show(title=f"Concentration at {path.stem}", delay=True)
            plt.show()
