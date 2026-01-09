"""Comparison of two runs using Wasserstein distance."""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import darsia
from darsia.presets.workflows.fluidflower_config import MultiFluidFlowerConfig
from darsia.presets.workflows.utils.mass import load_mass_data


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
    path: Path,
    compute: bool = False,
    assemble: bool = False,
    check: bool = False,
    skip_existing: bool = False,
):
    # Only allow one of compute, assemble, check_compleletion
    assert (compute + assemble + check) == 1, (
        "Exactly one of compute, assemble, or check_completion must be True."
    )

    config = MultiFluidFlowerConfig(path, require_results=True)
    assert config.wasserstein is not None

    if compute:
        _compute_wasserstein_distances(config, skip_existing)

    if assemble:
        _assemble_wasserstein_results(config)

    if check:
        _check_completion_status(config)


def _compute_wasserstein_distances(config, skip_existing: bool) -> None:
    """Compute all Wasserstein distances with intermediate file storage."""

    times_with_uncertainty = config.wasserstein.times_with_uncertainty
    active_runs = config.wasserstein.runs
    results_dir = config.wasserstein.results

    total_computations = 0
    completed_computations = 0

    # Count total computations for progress tracking
    for time, uncertainty in times_with_uncertainty:
        for run_1 in active_runs:
            for run_2 in active_runs:
                if run_1 >= run_2:  # Only compute upper triangle
                    continue
                for roi_config in config.wasserstein.roi.values():
                    total_computations += 1
    logger.info(f"Starting computation of {total_computations} Wasserstein distances")

    for time, uncertainty in times_with_uncertainty:
        logger.info(f"Processing time {time} with uncertainty {uncertainty}")

        for run_1 in active_runs:
            # Load mass data for run_1 once per time
            mass_1 = load_mass_data(
                config.sub_config[str(run_1)],
                time=time,
                tol=uncertainty,
            )
            coarse_mass_1 = darsia.resize(
                mass_1,
                fx=config.wasserstein.resize_factor,
                fy=config.wasserstein.resize_factor,
                interpolation="area",
            )

            for run_2 in active_runs:
                if run_1 >= run_2:
                    # Only compute upper triangle
                    # TODO fetch and copy existing result? or keep this for the assembly?
                    continue  # Load mass data for run_2
                mass_2 = load_mass_data(
                    config.sub_config[run_2],
                    time=time,
                    tol=uncertainty,
                )
                coarse_mass_2 = darsia.resize(mass_2, ref_image=coarse_mass_1)

                for roi_config in config.wasserstein.roi.values():
                    # Main ROI computation
                    distance, info = _compute_single_wasserstein(
                        run_1,
                        run_2,
                        time,
                        roi_config.name,
                        None,
                        roi_config,
                        coarse_mass_1,
                        coarse_mass_2,
                        results_dir,
                        skip_existing,
                    )

                    # Sub-ROI evaluation
                    for sub_roi_name in roi_config.sub_roi:
                        sub_roi_config = config.wasserstein.sub_roi[sub_roi_name]
                        subroi_distance = _evaluate_single_wasserstein_roi(
                            distance,
                            info,
                            sub_roi_name,
                            sub_roi_config,
                            results_dir,
                            skip_existing,
                        )

                    completed_computations += 1


def _compute_single_wasserstein(
    run_1,
    run_2,
    time,
    roi_name,
    sub_roi_name,
    roi_config,
    mass_1,
    mass_2,
    coarse_mass_1,
    coarse_mass_2,
    results_dir,
    skip_existing,
):
    """Compute single Wasserstein distance with error handling."""

    # Check if result already exists
    result_file = WassersteinResult.get_filename(
        results_dir, run_1, run_2, time, roi_name, sub_roi_name
    )
    if skip_existing and result_file.exists():
        logger.debug(f"Skipping existing result: {result_file.name}")
        return

    try:
        start_time = time.time()

        # Your actual Wasserstein computation here
        roi_geometry = _compute_roi_geometry(roi_config.roi)
        roi_exact_mass_1 = _extract_exact_mass(mass_1, roi_config.roi, time)
        roi_exact_mass_2 = _extract_exact_mass(mass_2, roi_config.roi, time)
        roi_detected_mass_1 = _extract_detected_mass(coarse_mass_1, roi_config.roi)
        roi_detected_mass_2 = _extract_detected_mass(coarse_mass_2, roi_config.roi)

        # Normalize
        roi_detected_mass_1 /= roi_exact_mass_1 if roi_exact_mass_1 else 1
        roi_detected_mass_2 /= roi_exact_mass_2 if roi_exact_mass_2 else 1

        # Compute Wasserstein distance
        wasserstein_distance = _compute_wasserstein_distance(
            roi_detected_mass_1, roi_detected_mass_2
        )
        normalized_wasserstein_distance = _normalize_wasserstein_distance(
            wasserstein_distance, roi_geometry
        )

        # Evaluate Wasserstein flux over subroi, if given.
        ...

        computation_time = time.time() - start_time

        # Create and save result
        result = WassersteinResult(
            run_1=run_1,
            run_2=run_2,
            time=time,
            roi_name=roi_name,
            sub_roi_name=sub_roi_name,
            wasserstein_distance=wasserstein_distance,
            normalized_wasserstein_distance=normalized_wasserstein_distance,
            roi_exact_mass_1=roi_exact_mass_1,
            roi_exact_mass_2=roi_exact_mass_2,
            roi_detected_mass_1=float(roi_detected_mass_1.sum()),
            roi_detected_mass_2=float(roi_detected_mass_2.sum()),
            computation_time=computation_time,
            timestamp=datetime.now().isoformat(),
        )

        result.save_to_dir(results_dir)
        logger.debug(f"Computed Wasserstein distance: {result_file.name}")

    except Exception as e:
        logger.error(f"Failed to compute {result_file.name}: {str(e)}")
        # Optionally save error info
        error_file = result_file.with_suffix(".error")
        with open(error_file, "w") as f:
            json.dump(
                {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "parameters": {
                        "run_1": run_1,
                        "run_2": run_2,
                        "time": time,
                        "roi_name": roi_name,
                        "sub_roi_name": sub_roi_name,
                    },
                },
                f,
                indent=2,
            )


def _compute_roi_geometry(roi):
    """Compute geometry for a region of interest.

    TODO: Implement this function.
    """
    raise NotImplementedError("_compute_roi_geometry not yet implemented")


def _extract_exact_mass(mass, roi, time):
    """Extract exact mass at given time and ROI.

    TODO: Implement this function.
    """
    raise NotImplementedError("_extract_exact_mass not yet implemented")


def _extract_detected_mass(coarse_mass, roi):
    """Extract detected mass at given ROI.

    TODO: Implement this function.
    """
    raise NotImplementedError("_extract_detected_mass not yet implemented")


def _compute_wasserstein_distance(mass_1, mass_2):
    """Compute Wasserstein distance between two mass distributions.

    TODO: Implement this function.
    """
    raise NotImplementedError("_compute_wasserstein_distance not yet implemented")


def _normalize_wasserstein_distance(wasserstein_distance, roi_geometry):
    """Normalize Wasserstein distance by ROI geometry.

    TODO: Implement this function.
    """
    raise NotImplementedError("_normalize_wasserstein_distance not yet implemented")


def _evaluate_single_wasserstein_roi(
    distance, info, sub_roi_name, sub_roi_config, results_dir, skip_existing
):
    """Evaluate Wasserstein distance for a sub-ROI.

    TODO: Implement this function.
    """
    raise NotImplementedError("_evaluate_single_wasserstein_roi not yet implemented")


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
