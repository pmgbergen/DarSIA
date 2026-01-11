"""Comparison of two runs using Wasserstein distance."""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import darsia
from darsia.presets.workflows.fluidflower_config import MultiFluidFlowerConfig
from darsia.presets.workflows.utils.mass import load_data
from darsia.presets.workflows.rig import Rig


import json
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class WassersteinResult:
    """Dataclass to store Wasserstein result information."""

    run_1: str
    """Name of the first run."""
    run_2: str
    """Name of the second run."""
    time: float
    """Time at which the Wasserstein distance was computed (in hours)."""
    roi_name: str
    """Name of the region of interest."""
    wasserstein_distance: float
    """Computed Wasserstein distance."""
    normalized_wasserstein_distance: float
    """Normalized Wasserstein distance."""
    roi_exact_mass_1: float
    """Exact mass of the region of interest in run 1."""
    roi_exact_mass_2: float
    """Exact mass of the region of interest in run 2."""
    roi_detected_mass_1: float
    """Detected mass of the region of interest in run 1."""
    roi_detected_mass_2: float
    """Detected mass of the region of interest in run 2."""
    computation_time: float
    """Time taken to compute the Wasserstein distance (in seconds)."""
    timestamp: str
    """Timestamp of the computation."""
    sub_roi_name: Optional[str] = None
    """Name of the sub-region of interest, if applicable."""
    subroi_normalized_wasserstein_distance: Optional[float] = None
    """Normalized Wasserstein distance for sub-ROI, if applicable."""
    success: Optional[bool] = None
    """Whether the computation was successful."""

    @staticmethod
    def get_filename(
        results_dir: Path,
        run_1: str,
        run_2: str,
        time: float,
        roi_name: str,
        sub_roi_name: Optional[str] = None,
    ) -> Path:
        """Generate standardized filename for intermediate results."""
        # Sanitize time for filename (replace . and : with _)
        time_str = f"{time:.3f}".replace(".", "_").replace(":", "_")

        if sub_roi_name:
            filename = (
                f"wasserstein_{run_1}_{run_2}_{time_str}_{roi_name}_{sub_roi_name}.json"
            )
        else:
            filename = f"wasserstein_{run_1}_{run_2}_{time_str}_{roi_name}.json"

        return results_dir / filename

    def get_result_filename(self, results_dir: Path) -> Path:
        """Get the filename for this result."""
        return self.get_filename(
            results_dir,
            self.run_1,
            self.run_2,
            self.time,
            self.roi_name,
            self.sub_roi_name,
        )

    def save(self, path: Path) -> None:
        """Save result to JSON file at specified path."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def save_to_dir(self, results_dir: Path) -> Path:
        """Save result to JSON file in results directory with auto-generated name."""
        filename = self.get_result_filename(results_dir)
        self.save(filename)
        return filename

    @classmethod
    def load(cls, filename: Path) -> "WassersteinResult":
        """Load result from JSON file."""
        with open(filename, "r") as f:
            data = json.load(f)
        return cls(**data)


def comparison_wasserstein(
    cls: type[Rig],
    path: Path | list[Path],
    compute: bool = False,
    assemble: bool = False,
    check: bool = False,
    skip_existing: bool = False,
):
    # Only allow one of compute, assemble, check_compleletion
    assert (compute + assemble + check) == 1, (
        "Exactly one of compute, assemble, or check_completion must be True."
    )

    config = MultiFluidFlowerConfig(path, require_data=False, require_results=True)
    assert config.wasserstein is not None

    if compute:
        _compute_wasserstein_distances(cls, config, skip_existing)

    if assemble:
        _assemble_wasserstein_results(config)

    if check:
        _check_completion_status(config)


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
            f"Example reference distances {plume_based_reference_distance=} and {rig_based_reference_distance=}"
        )

        for run_1 in active_runs:
            # Load mass data for run_1 once per time
            mass_1 = load_data(
                config.runs.config[run_1],
                data="mass",
                time=time,
                tol=uncertainty,
            )
            if mass_1 is None:
                logger.warning(
                    f"Mass data for run {run_1} at time {time} not found. Skipping."
                )
                continue

            # Add weight accounting for 3D effect and porosity
            dimensionally_reduced_mass_1 = mass_1 * porosity_times_depth

            # Resize and rescale mass distributions
            dimensionally_reduced_coarse_mass_1 = darsia.resize(
                dimensionally_reduced_mass_1,
                fx=config.wasserstein.resize_factor,
                fy=config.wasserstein.resize_factor,
                interpolation="inter_area",
            )
            integrated_mass_1 = dimensionally_reduced_mass_1.integral()
            integrated_coarse_mass_1 = dimensionally_reduced_coarse_mass_1.integral()
            coarse_mass_1 = dimensionally_reduced_coarse_mass_1 * (
                integrated_mass_1 / integrated_coarse_mass_1
            )

            # Same for the weight
            coarse_wasserstein_weight = darsia.resize(
                wasserstein_weight,
                ref_image=dimensionally_reduced_coarse_mass_1,
                interpolation="inter_nearest",
            )

            # dimensionally_reduced_mass_1.show(title="mass", delay=True)
            # coarse_mass_1.show(title="coarse mass", delay=False)

            for run_2 in active_runs:
                if run_1 >= run_2:
                    # Only compute upper triangle
                    # TODO fetch and copy existing result? or keep this for the assembly?
                    continue  # Load mass data for run_2
                mass_2 = load_data(
                    config.runs.config[run_2],
                    data="mass",
                    time=time,
                    tol=uncertainty,
                )
                if mass_2 is None:
                    logger.warning(
                        f"Mass data for run {run_2} at time {time} not found. Skipping."
                    )
                    continue
                dimensionally_reduced_mass_2 = mass_2 * porosity_times_depth

                # Resize and rescale mass distributions
                dimensionally_reduced_coarse_mass_2 = darsia.resize(
                    dimensionally_reduced_mass_2,
                    ref_image=dimensionally_reduced_coarse_mass_1,
                    interpolation="inter_area",
                )
                integrated_mass_2 = dimensionally_reduced_mass_2.integral()
                integrated_coarse_mass_2 = (
                    dimensionally_reduced_coarse_mass_2.integral()
                )
                coarse_mass_2 = dimensionally_reduced_coarse_mass_2 * (
                    integrated_mass_2 / integrated_coarse_mass_2
                )

                # Perform wasserstein computations for all ROIs
                for roi_config in config.wasserstein.roi.values():
                    # Main ROI computation
                    # TODO condense to single function?
                    # distance, info = _compute_single_wasserstein(...)
                    roi_name = roi_config.name
                    sub_roi_name = None

                    # Check if result already exists
                    result_file = WassersteinResult.get_filename(
                        results_dir, run_1, run_2, time, roi_name, sub_roi_name
                    )
                    if skip_existing and result_file.exists():
                        logger.debug(f"Skipping existing result: {result_file.name}")
                        raise NotImplementedError

                    import time as time_

                    start_time = time_.time()

                    # Your actual Wasserstein computation here
                    roi_mass_1 = coarse_mass_1.subregion(roi_config.roi)
                    roi_mass_2 = coarse_mass_2.subregion(roi_config.roi)
                    roi_wasserstein_weight = coarse_wasserstein_weight.subregion(
                        roi_config.roi
                    )

                    roi_integral_1 = roi_mass_1.integral()
                    roi_integral_2 = roi_mass_2.integral()

                    # React to zero mass cases
                    if np.isclose(integrated_mass_1, 0) or np.isclose(
                        integrated_mass_2, 0
                    ):
                        raise NotImplementedError

                    # React to large relative differences in mass
                    roi_integral_average = 0.5 * (roi_integral_1 + roi_integral_2)
                    relative_mass_difference = (
                        abs(roi_integral_1 - roi_integral_2) / roi_integral_average
                    )
                    if config.wasserstein.relative_tol is not None:
                        if relative_mass_difference > config.wasserstein.relative_tol:
                            raise NotImplementedError

                    # Normalize roi masses by their average
                    roi_mass_1 *= roi_integral_average / roi_integral_1
                    roi_mass_2 *= roi_integral_average / roi_integral_2
                    assert np.isclose(roi_mass_1.integral(), roi_mass_2.integral())

                    # Compute Wasserstein distance
                    wasserstein_distance, info = darsia.wasserstein_distance(
                        roi_mass_1,
                        roi_mass_2,
                        weight=None,  # roi_wasserstein_weight,
                        method="bregman",
                        options=options,
                    )

                    # Fetch status.
                    # if info["status"] == "diverged": ...
                    # if not info["converged"]:
                    #    raise NotImplementedError

                    # Print the result to vtk.
                    vtk_path = (
                        Path("./unweighted_w1")
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
                        "mass_based": wasserstein_distance / roi_integral_average,
                    }

                    print(f"{normalized_wasserstein_distance=}")

                    # Sub-ROI evaluation
                    # Evaluate Wasserstein flux over subroi, if given.
                    transport_density = darsia.full_like(
                        roi_mass_1, info["transport_density"]
                    )
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

                    computation_time = time_.time() - start_time

                    # Create and save result
                    # result = WassersteinResult(
                    #    run_1=run_1,
                    #    run_2=run_2,
                    #    time=time,
                    #    roi_name=roi_name,
                    #    sub_roi_name=sub_roi_name,
                    #    wasserstein_distance=wasserstein_distance,
                    #    normalized_wasserstein_distance=normalized_wasserstein_distance,
                    #    roi_exact_mass_1=roi_exact_mass_1,
                    #    roi_exact_mass_2=roi_exact_mass_2,
                    #    roi_detected_mass_1=float(roi_detected_mass_1.sum()),
                    #    roi_detected_mass_2=float(roi_detected_mass_2.sum()),
                    #    computation_time=computation_time,
                    #    timestamp=datetime.now().isoformat(),
                    # )

                    # result.save_to_dir(results_dir)
                    logger.debug(f"Computed Wasserstein distance: {result_file.name}")

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

    times = config.wasserstein.times
    active_runs = config.wasserstein.runs
    roi_names = list(config.wasserstein.roi.keys())
    results_dir = config.wasserstein.results

    # Create output directory
    output_dir = Path("wasserstein_tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    for time in times:
        logger.info(f"Assembling results for time {time}")

        # Create DataFrame for this time
        # Rows: run pairs, Columns: ROI names
        run_pairs = [(r1, r2) for r1 in active_runs for r2 in active_runs if r1 < r2]

        df = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(run_pairs, names=["run_1", "run_2"]),
            columns=roi_names,
        )

        missing_results = []
        # Fill DataFrame with results
        for run_1, run_2 in run_pairs:
            for roi_name in roi_names:
                result_file = WassersteinResult.get_filename(
                    results_dir, run_1, run_2, time, roi_name, None
                )

                try:
                    if result_file.exists():
                        result = WassersteinResult.load(result_file)
                        df.loc[(run_1, run_2), roi_name] = (
                            result.normalized_wasserstein_distance
                        )
                    else:
                        missing_results.append(str(result_file.name))
                        df.loc[(run_1, run_2), roi_name] = np.nan

                except Exception as e:
                    logger.warning(f"Failed to load {result_file}: {e}")
                    df.loc[(run_1, run_2), roi_name] = np.nan

        # Save assembled table
        output_file = output_dir / f"wasserstein_distances_{time:.3f}.csv"
        df.to_csv(output_file)

        # Report statistics
        total_expected = len(run_pairs) * len(roi_names)
        missing_count = len(missing_results)
        completion_rate = (total_expected - missing_count) / total_expected * 100

        logger.info(
            f"Time {time}: {completion_rate:.1f}% complete "
            f"({total_expected - missing_count}/{total_expected} results)"
        )

        if missing_results:
            missing_file = output_dir / f"missing_results_{time:.3f}.txt"
            with open(missing_file, "w") as f:
                f.write("\n".join(missing_results))


def _check_completion_status(config):
    """Check how many computations are complete."""
    results_dir = config.wasserstein.results
    # Implementation to scan results_dir and report completion statistics
    completed = 0
    total = 0

    for result_file in results_dir.glob("*.json"):
        total += 1
        try:
            result = WassersteinResult.load(result_file)
            if result:
                completed += 1
        except Exception as e:
            logger.warning(f"Failed to load {result_file}: {e}")

    logger.info(f"Completed {completed}/{total} computations.")
    return completed, total
