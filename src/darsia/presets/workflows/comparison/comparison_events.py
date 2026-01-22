"""Determine events of earliest occurence."""

import logging
from pathlib import Path

import pandas as pd
from darsia.presets.workflows.fluidflower_config import MultiFluidFlowerConfig
import numpy as np

logger = logging.getLogger(__name__)


def comparison_events(
    path: Path,
):
    # ! ---- LOAD CONFIG FOR COMPARISON OF RUNS ----

    # This script does not depend on raw data, only results
    config = MultiFluidFlowerConfig(path, require_results=True)

    # Safety checks
    assert config.events is not None

    # Prepare data frame to store results
    events = pd.DataFrame(columns=["run"] + list(config.events.events.keys()))

    # Prepare overall data frame file
    for run, run_config in config.sub_config.items():
        # Safety checks
        assert run_config.data is not None

        # Loop over events
        for event in config.events.events.values():
            if event.mode in ["mass", "mass_g", "mass_aq"]:
                # Fetch mass analysis results
                mass_path = (
                    run_config.data.results / "sparse_data" / "integrated_mass.csv"
                )
                mass_df = pd.read_csv(mass_path)
                times = mass_df["time"]
                # Find a column with "exact_mass" to get total injected mass
                exact_mass_cols = [
                    col for col in mass_df.columns if "exact_mass" in col
                ]
                if exact_mass_cols:
                    total_mass = np.max(mass_df[exact_mass_cols[0]])
                else:
                    total_mass = 1.0  # fallback if no exact mass column found

                # Fetch detected mass for the event's ROI
                if event.mode == "mass":
                    key = f"{event.roi_name}_detected_mass"
                elif event.mode == "mass_g":
                    key = f"{event.roi_name}_detected_mass_g"
                elif event.mode == "mass_aq":
                    key = f"{event.roi_name}_detected_mass_aq"
                columns = mass_df.columns
                assert key in columns, f"Key {key} not found in mass results."
                detected_mass = mass_df[key]

                # Determine event time based on relative threshold
                all_event_times = times[
                    detected_mass >= event.relative_threshold * total_mass
                ]
                earliest_event_time = np.min(all_event_times)
                events.at[run, event.event_id] = earliest_event_time

                print(detected_mass)

            else:
                raise NotImplementedError(f"Event type {event.mode} not implemented.")
            logger.info(
                f"Event {event.event_id} for run {run} occurred at time {events.at[run, event.event_id]}."
            )

    # Store to file.
    events.to_csv(config.events.path)
