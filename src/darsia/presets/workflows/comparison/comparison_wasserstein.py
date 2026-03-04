"""Comparison of two runs using Wasserstein distance."""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import darsia
from darsia.presets.workflows.fluidflower_config import MultiFluidFlowerConfig
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.utils.mass import load_data

logger = logging.getLogger(__name__)


@dataclass
class WassersteinDistanceResult:
    """Dataclass to store Wasserstein result information."""

    run_1: str
    """Name of the first run."""
    run_2: str
    """Name of the second run."""
    time: float
    """Time at which the Wasserstein distance was computed (in hours)."""
    time_1: float | None
    """Time in run 1 corresponding to the computation time (optional)."""
    time_2: float | None
    """Time in run 2 corresponding to the computation time (optional)."""
    roi_name: str
    """Name of the region of interest."""
    roi_exact_mass: float
    """Exact mass of the region of interest."""
    roi_detected_mass_1: float
    """Detected mass of the region of interest in run 1."""
    roi_detected_mass_2: float
    """Detected mass of the region of interest in run 2."""
    distance: float
    """Computed Wasserstein distance."""
    normalized_distance: float | dict[str, float]
    """Normalized Wasserstein distance."""
    computation_time: float
    """Time taken to compute the Wasserstein distance (in seconds)."""
    timestamp: str
    """Timestamp of the computation."""
    status: Literal["success", "missing"] | None = None
    """Status of the computation (optional)."""

    @staticmethod
    def get_filename(
        run_1: str,
        run_2: str,
        time: float,
        roi_name: str,
    ) -> Path:
        """Generate standardized filename for intermediate results."""
        # Sanitize time for filename (replace . and : with _)
        time_str = f"{time:.3f}".replace(".", "_").replace(":", "_")

        filename = f"wasserstein_{run_1}_{run_2}_{time_str}_{roi_name}.json".replace(
            " ", "_"
        )

        return Path(filename)

    def get_result_filename(self) -> Path:
        """Get the filename for this result."""
        return self.get_filename(
            self.run_1,
            self.run_2,
            self.time,
            self.roi_name,
        )

    def save(self, path: Path) -> None:
        """Save result to JSON file at specified path."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def save_to_dir(self, dir: Path) -> Path:
        """Save result to JSON file in results directory with auto-generated name.

        Args:
            dir: The directory where the results should be saved.

        Returns:
            The path to the saved result file."""
        filename = self.get_result_filename()
        full_path = dir / filename
        self.save(full_path)
        logger.info(
            f"Saved Wasserstein result to: file:///{full_path.resolve().as_posix()}"
        )
        return full_path

    @classmethod
    def load(cls, filename: Path) -> "WassersteinDistanceResult":
        """Load result from JSON file."""
        with open(filename, "r") as f:
            data = json.load(f)
        return cls(**data)


def comparison_wasserstein(
    cls: type[Rig],
    path: Path | list[Path],
    compute: bool = False,
    assemble: bool = False,
    skip_existing: bool = False,
):
    # Only allow one of compute, assemble, check_compleletion
    assert (compute + assemble) == 1, (
        "Exactly one of compute, assemble, or check_completion must be True."
    )

    config = MultiFluidFlowerConfig(path, require_data=False, require_results=True)
    assert config.wasserstein is not None

    if compute:
        _compute_wasserstein_distances(cls, config, skip_existing)

    if assemble:
        # _assemble_wasserstein_results(config)
        _assemble_all_wasserstein_results(config)


def _load_and_process_mass(
    run_name: str,
    config: MultiFluidFlowerConfig,
    time: float,
    uncertainty: float,
    porosity_times_depth: darsia.Image,
    ref_image: darsia.Image | None = None,
) -> darsia.Image | None:
    """Load and resize mass data for a given run and time.

    Args:
        run_name: Name of the run.
        config: Multi-run FluidFlower configuration.
        time: Time at which to load mass data.
        uncertainty: Uncertainty tolerance for time matching.
        porosity_times_depth: Geometry weight (porosity × depth).
        ref_image: Reference image for resizing (optional).

    Returns:
        coarse_mass: The processed coarse mass image, or None if mass data is not found.

    """
    # Load mass data
    mass = load_data(
        config.runs.config[run_name],
        data="mass",
        time=time,
        tol=uncertainty,
    )

    if mass is None:
        logger.warning(
            f"Mass data for run {run_name} at time {time} not found. Skipping."
        )
        return None, None

    # Apply weight for 3D effect and porosity
    dimensionally_reduced_mass = mass * porosity_times_depth

    # Resize
    if ref_image is None:
        dimensionally_reduced_coarse_mass = darsia.resize(
            dimensionally_reduced_mass,
            fx=config.wasserstein.resize_factor,
            fy=config.wasserstein.resize_factor,
            interpolation="inter_area",
        )
    else:
        dimensionally_reduced_coarse_mass = darsia.resize(
            dimensionally_reduced_mass,
            ref_image=ref_image,
            interpolation="inter_area",
        )

    # Rescale coarse mass to preserve total mass
    integrated_mass = dimensionally_reduced_mass.integral()
    integrated_coarse_mass = dimensionally_reduced_coarse_mass.integral()
    coarse_mass = dimensionally_reduced_coarse_mass * (
        integrated_mass / integrated_coarse_mass
    )

    return coarse_mass


def _compute_wasserstein_distances(
    cls: type[Rig], config: MultiFluidFlowerConfig, skip_existing: bool
) -> None:
    """Compute all Wasserstein distances with intermediate file storage."""

    # Set some 'standard' options for computing the Wasserstein distance.
    options = {
        # L1 options for discretization
        "mobility_mode": darsia.MobilityMode.CELL_BASED,
        "l1_mode": darsia.L1Mode.RAVIART_THOMAS,  # Accuarate
        # Nonlinear solver
        "bregman_update": lambda iter: iter in [0, 10, 20],
        # Performance control
        "num_iter": 20,
        "tol_distance": 1e-4,
        "tol_increment": 1e-2,
        # Linear solver
        "linear_solver": "direct",
        # Output
        "return_info": True,
        "return_status": True,
        "verbose": True,
    }

    times = config.wasserstein.times
    active_runs = config.wasserstein.runs
    results_dir = config.wasserstein.results

    total_computations = 0
    completed_computations = 0

    # Fetch geometry weighting factor
    example_config = config.runs.config[active_runs[0]]
    fluidflower = cls.load(example_config.rig.path)
    geometry = fluidflower.geometry
    porosity_times_depth = geometry.weight

    # Determine a weight for the Wasserstein distance based on the permeability.
    permeability = fluidflower.permeability
    # Make sure to clip values to avoid division by zero.
    non_zero_min_permeability = np.min(
        permeability.img[~np.isclose(permeability.img, 0)]
    )
    clip_value = (
        1e-6 * non_zero_min_permeability
        if np.isclose(np.min(fluidflower.permeability.img), 0)
        else 0
    )
    # Use the inverse of the permeability divided by its maximum.
    max_permeability = np.max(permeability.img)
    wasserstein_weight = darsia.full_like(
        permeability, max_permeability / np.clip(permeability.img, clip_value, None)
    )

    # Fetch experimental protocol for defining reference values based on injection profile
    example_experiment = darsia.ProtocolledExperiment.init_from_config(example_config)

    # Determine characteristic solubility distriburion
    state = example_experiment.pressure_temperature_protocol.get_state(
        example_experiment.experiment_start
    )
    gradient = example_experiment.pressure_temperature_protocol.get_gradient(
        example_experiment.experiment_start
    )
    co2_mass_analysis = darsia.CO2MassAnalysis(
        baseline=fluidflower.baseline,
        atmospheric_pressure=state.pressure,
        atmospheric_temperature=state.temperature,
        atmospheric_pressure_gradient=gradient.pressure,
        atmospheric_temperature_gradient=gradient.temperature,
    )
    example_solubility = co2_mass_analysis.solubility_co2
    avg_solubility = np.average(example_solubility)

    # Count total computations for progress tracking
    for time, uncertainty in times:
        for run_1 in active_runs:
            for run_2 in active_runs:
                if run_1 >= run_2:  # Only compute upper triangle
                    continue
                for roi_config in config.wasserstein.roi.values():
                    total_computations += 1
    logger.info(f"Starting computation of {total_computations} Wasserstein distances")

    for time, uncertainty in times:
        logger.info(f"Processing time {time} with uncertainty {uncertainty}")

        # Obtain expected mass from injection profile at given time
        exact_mass = example_experiment.injection_protocol.injected_mass(time=time)

        # Determine refernce distance based on plume expansion
        avg_depth = np.average(geometry.depth.img)
        avg_volume = exact_mass / avg_solubility
        avg_plume_area = avg_volume / avg_depth
        avg_plume_radius = np.sqrt(avg_plume_area / np.pi)

        plume_based_reference_distance = avg_plume_radius * exact_mass

        # Determine reference distance based on rig dimensions
        rig_length = geometry.dimensions[0]
        rig_width = geometry.dimensions[1]
        avg_rig_dimension = 0.5 * (rig_length + rig_width)

        rig_based_reference_distance = exact_mass * avg_rig_dimension

        print(
            f"""Example reference distances {plume_based_reference_distance=} """
            f"""and {rig_based_reference_distance=}"""
        )

        # Outer loop over runs (only load mass data once per run and time)
        for run_1 in active_runs:
            # Load mass data for run_1 once per time
            coarse_mass_1 = _load_and_process_mass(
                run_name=run_1,
                config=config,
                time=time,
                uncertainty=uncertainty,
                porosity_times_depth=porosity_times_depth,
            )

            # Add weight accounting for 3D effect and porosity
            if coarse_mass_1 is None:
                coarse_wasserstein_weight = None
            else:
                coarse_wasserstein_weight = darsia.resize(
                    wasserstein_weight,
                    ref_image=coarse_mass_1,
                    interpolation="inter_nearest",
                )

            # Nested loop over run pairs (only upper triangle)
            for run_2 in active_runs:
                if run_1 >= run_2:
                    # Only compute upper triangle
                    # TODO fetch and copy existing result? or keep this for the assembly?
                    continue  # Load mass data for run_2

                coarse_mass_2 = _load_and_process_mass(
                    run_name=run_2,
                    config=config,
                    time=time,
                    uncertainty=uncertainty,
                    porosity_times_depth=porosity_times_depth,
                    ref_image=coarse_mass_1,
                )

                # Perform wasserstein computations for all ROIs
                for roi_config in config.wasserstein.roi.values():
                    # Main ROI computation
                    # TODO condense to single function?
                    # distance, info = _compute_single_wasserstein(...)
                    roi_name = roi_config.name
                    # sub_roi_name = None # TODO rm?

                    # Obtain expected mass from injection profile at given time
                    roi_exact_mass = (
                        example_experiment.injection_protocol.injected_mass(
                            time=time, roi=roi_config.roi
                        )
                    )

                    # Check if result already exists
                    result_file = WassersteinDistanceResult.get_filename(
                        run_1, run_2, time, roi_name
                    )
                    if skip_existing and (results_dir / result_file).exists():
                        result = WassersteinDistanceResult.load(
                            results_dir / result_file
                        )
                        previous_status = result.status
                    else:
                        previous_status = None

                    if previous_status == "success":
                        # Stop if previous computation was successful
                        logger.debug(
                            f"Skipping successful Wasserstein distance: {result_file.name}"
                        )
                    elif (coarse_mass_1 is None) or (coarse_mass_2 is None):
                        # Stop if mass data is missing
                        logger.warning(
                            f"""Mass data for run {run_1} or {run_2} at time {time} """
                            """not found. Skipping."""
                        )
                        result = WassersteinDistanceResult(
                            run_1=run_1,
                            run_2=run_2,
                            time=time,
                            time_1=np.nan,
                            time_2=np.nan,
                            roi_name=roi_name,
                            roi_exact_mass=np.nan,
                            roi_detected_mass_1=np.nan,
                            roi_detected_mass_2=np.nan,
                            distance=np.nan,
                            normalized_distance={
                                "plume_based": np.nan,
                                "rig_based": np.nan,
                                "mass_based": np.nan,
                            },
                            computation_time=0.0,
                            timestamp=datetime.now().isoformat(),
                            status="missing",
                        )
                        result.save_to_dir(results_dir)
                        logger.debug(
                            f"Skipping unsuccessful Wasserstein distance: {result_file.name}"
                        )
                    else:
                        # Extract roi.
                        roi_mass_1 = coarse_mass_1.subregion(roi_config.roi)
                        roi_mass_2 = coarse_mass_2.subregion(roi_config.roi)
                        roi_wasserstein_weight = coarse_wasserstein_weight.subregion(
                            roi_config.roi
                        )
                        roi_integral_1 = roi_mass_1.integral()
                        roi_integral_2 = roi_mass_2.integral()

                        # Compute relative mass difference and average for control and
                        # normalization.
                        roi_integral_average = 0.5 * (roi_integral_1 + roi_integral_2)
                        relative_mass_difference = (
                            abs(roi_integral_1 - roi_integral_2) / roi_integral_average
                        )

                        # React to zero mass case I (both masses are zero).
                        if np.isclose(roi_integral_1, 0) and np.isclose(
                            roi_integral_2, 0
                        ):
                            wasserstein_distance = 0.0
                            normalized_wasserstein_distance = {
                                "plume_based": 0.0,
                                "rig_based": 0.0,
                                "mass_based": 0.0,
                            }
                            computation_time = 0.0

                        # React to zero mass case II (one mass is zero)
                        elif np.isclose(roi_integral_1, 0) or np.isclose(
                            roi_integral_2, 0
                        ):
                            wasserstein_distance = np.nan
                            normalized_wasserstein_distance = {
                                "plume_based": np.nan,
                                "rig_based": np.nan,
                                "mass_based": np.nan,
                            }
                            computation_time = 0.0

                        # React to large relative differences in mass
                        elif (config.wasserstein.relative_tol is not None) and (
                            relative_mass_difference > config.wasserstein.relative_tol
                        ):
                            # raise NotImplementedError
                            wasserstein_distance = np.nan
                            normalized_wasserstein_distance = {
                                "plume_based": np.nan,
                                "rig_based": np.nan,
                                "mass_based": np.nan,
                            }
                            computation_time = 0.0

                        else:
                            # Actual Wasserstein distance computation

                            # Perform computation
                            import time as time_

                            start_time = time_.time()

                            # Normalize roi masses by their average
                            roi_mass_1 *= roi_integral_average / roi_integral_1
                            roi_mass_2 *= roi_integral_average / roi_integral_2
                            assert np.isclose(
                                roi_mass_1.integral(), roi_mass_2.integral()
                            )

                            # Compute Wasserstein distance
                            wasserstein_distance, info = darsia.wasserstein_distance(
                                roi_mass_1,
                                roi_mass_2,
                                weight=roi_wasserstein_weight,
                                method="bregman",
                                options=options,
                            )

                            # Fetch status.
                            # if info["status"] == "diverged": ...
                            # if not info["converged"]:
                            #    raise NotImplementedError

                            # Print the result to vtk.
                            vtk_path = (
                                Path("./new_weighted_w1")
                                / f"roi_{roi_name}"
                                / f"run_{run_1}_to_{run_2}"
                                / f"w1_{time}"
                            )
                            darsia.wasserstein_distance_to_vtk(vtk_path, info)

                            # Normalize Wasserstein distance.
                            normalized_wasserstein_distance = {
                                "plume_based": wasserstein_distance
                                / plume_based_reference_distance,
                                "rig_based": wasserstein_distance
                                / rig_based_reference_distance,
                                "mass_based": wasserstein_distance
                                / roi_integral_average,
                            }

                            # Control evaluation of the transport density over sub-rois
                            # weighted_transport_density = darsia.full_like(
                            #    roi_mass_1,
                            #    info.get("weighted_transport_density"),
                            # )
                            # transport_density.show()
                            # for sub_roi_name in roi_config.sub_roi:
                            #    sub_roi_config = config.wasserstein.sub_roi[sub_roi_name]
                            # subroi_distance = _evaluate_single_wasserstein_roi(
                            #    distance,
                            #    info,
                            #    sub_roi_name,
                            #    sub_roi_config,
                            #    results_dir,
                            #    skip_existing,
                            # )

                            # sub_roi_result = None
                            # ...

                            computation_time = time_.time() - start_time

                        # Create and save result
                        result = WassersteinDistanceResult(
                            run_1=run_1,
                            run_2=run_2,
                            time=time,
                            time_1=coarse_mass_1.time,
                            time_2=coarse_mass_2.time,
                            roi_name=roi_name,
                            roi_exact_mass=roi_exact_mass,
                            roi_detected_mass_1=roi_integral_1,
                            roi_detected_mass_2=roi_integral_2,
                            distance=wasserstein_distance,
                            normalized_distance=normalized_wasserstein_distance,
                            computation_time=computation_time,
                            timestamp=datetime.now().isoformat(),
                            status="success",
                        )

                        result.save_to_dir(results_dir)
                        logger.debug(
                            f"Computed Wasserstein distance: {result_file.name}"
                        )

                    completed_computations += 1


# TODO include helper functions here
# def _compute_single_wasserstein(...)

# try:
#    # Your actual Wasserstein computation here
# ...
# except Exception as e:
#    logger.error(f"Failed to compute {result_file.name}: {str(e)}")
#    # Optionally save error info
#    error_file = result_file.with_suffix(".error")
#    with open(error_file, "w") as f:
#        json.dump(
#            {
#                "error": str(e),
#                "timestamp": datetime.now().isoformat(),
#                "parameters": {
#                    "run_1": run_1,
#                    "run_2": run_2,
#                    "time": time,
#                    "roi_name": roi_name,
#                    "sub_roi_name": sub_roi_name,
#                },
#            },
#            f,
#            indent=2,
#        )


def _assemble_wasserstein_results(config):
    """Assemble all intermediate results into final tables."""

    # Fetch configuration details
    times = config.wasserstein.times
    active_runs = config.wasserstein.runs
    results_dir = config.wasserstein.results
    assert results_dir.exists(), (
        f"Results directory {results_dir} does not exist. Run computation first."
    )
    roi_keys = list(config.wasserstein.roi.keys())
    roi_names = [config.wasserstein.roi[key].name for key in roi_keys]

    # Create output directory
    output_dir = results_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    for time, _ in times:
        # Ignore uncertainty for assembly and associate time with matched results as done
        # during computation.
        logger.info(f"Assembling results for time {time}")

        # Create DataFrame for this time
        # Rows: run pairs, Columns: ROI names
        run_pairs = [(r1, r2) for r1 in active_runs for r2 in active_runs if r1 < r2]

        df = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(run_pairs, names=["run_1", "run_2"]),
            columns=roi_names,
        )

        # Keep track of missing results for reporting
        missing_results = []

        # Fill DataFrame with results
        for run_1, run_2 in run_pairs:
            for roi_name in roi_names:
                result_file = WassersteinDistanceResult.get_filename(
                    run_1, run_2, time, roi_name
                )

                try:
                    if (results_dir / result_file).exists():
                        result = WassersteinDistanceResult.load(
                            results_dir / result_file
                        )
                        df.loc[(run_1, run_2), roi_name] = result.distance
                    else:
                        missing_results.append(str(result_file.name))
                        df.loc[(run_1, run_2), roi_name] = np.nan

                except Exception as e:
                    logger.warning(f"Failed to load {results_dir / result_file}: {e}")
                    df.loc[(run_1, run_2), roi_name] = np.nan

        # Save assembled table
        output_file = output_dir / f"wasserstein_distances_{time:.3f}.csv"
        df.to_csv(output_file)

        # Save missing results log
        missing_file = output_dir / f"missing_results_{time:.3f}.txt"
        with open(missing_file, "w") as f:
            f.write("\n".join(missing_results))

        # Report statistics
        total_expected = len(run_pairs) * len(roi_names)
        missing_count = len(missing_results)
        completion_rate = (total_expected - missing_count) / total_expected * 100

        logger.info(
            f"Time {time}: {completion_rate:.1f}% complete "
            f"({total_expected - missing_count}/{total_expected} results)"
        )


def _assemble_all_wasserstein_results(config: MultiFluidFlowerConfig) -> None:
    """Assemble all Wasserstein results into a single CSV file.

    Collects all results across all times, ROIs, and run pairs into a single
    DataFrame with columns: time, roi_name, run_1, run_2, normalization, distance.

    For each result, the raw distance is stored with normalization=None, and
    each normalized distance is stored with its key as the normalization value.

    Args:
        config: Multi-run FluidFlower configuration.

    """
    # Fetch configuration details
    times = config.wasserstein.times
    active_runs = config.wasserstein.runs
    results_dir = config.wasserstein.results
    assert results_dir.exists(), (
        f"Results directory {results_dir} does not exist. Run computation first."
    )
    roi_keys = list(config.wasserstein.roi.keys())
    roi_names = [config.wasserstein.roi[key].name for key in roi_keys]

    # Create output directory
    output_dir = results_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all rows, aiming at creating DataFrame df further below.
    rows = []

    for time, _ in times:
        run_pairs = [(r1, r2) for r1 in active_runs for r2 in active_runs if r1 < r2]

        for run_1, run_2 in run_pairs:
            for roi_name in roi_names:
                result_file = WassersteinDistanceResult.get_filename(
                    run_1, run_2, time, roi_name
                )
                result_path = results_dir / result_file

                if not result_path.exists():
                    logger.warning(f"Missing result: {result_file.name}")
                    continue

                try:
                    result = WassersteinDistanceResult.load(result_path)
                except Exception as e:
                    logger.warning(f"Failed to load {result_file.name}: {e}")
                    continue

                # Raw (unnormalized) distance
                rows.append(
                    {
                        "time": result.time,
                        "roi_name": result.roi_name,
                        "run_1": result.run_1,
                        "run_2": result.run_2,
                        "normalization": "None",
                        "distance": result.distance,
                    }
                )

                # Normalized distances
                normalized = result.normalized_distance
                if isinstance(normalized, dict):
                    for norm_key, norm_value in normalized.items():
                        rows.append(
                            {
                                "time": result.time,
                                "roi_name": result.roi_name,
                                "run_1": result.run_1,
                                "run_2": result.run_2,
                                "normalization": norm_key,
                                "distance": norm_value,
                            }
                        )
                else:
                    # Single normalized value (float)
                    rows.append(
                        {
                            "time": result.time,
                            "roi_name": result.roi_name,
                            "run_1": result.run_1,
                            "run_2": result.run_2,
                            "normalization": "normalized",
                            "distance": normalized,
                        }
                    )

    # Assemble DataFrame and save
    df = pd.DataFrame(
        rows,
        columns=["time", "roi_name", "run_1", "run_2", "normalization", "distance"],
    )

    output_file = output_dir / "wasserstein_distances_all.csv"
    df.to_csv(output_file, index=False, na_rep="NaN")

    logger.info(
        f"Assembled {len(df)} rows into {output_file} "
        f"({df['normalization'].nunique()} normalization types, "
        f"{df['time'].nunique()} times, "
        f"{df['roi_name'].nunique()} ROIs)"
    )
