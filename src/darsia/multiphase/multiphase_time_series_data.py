"""Dataclass for managing time series multiphase data."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class TimeSeriesData:
    time: List[float] = field(default_factory=list)
    """Time at each data point"""
    name: List[str] = field(default_factory=list)
    """Name for each data point, e.g. name of raw image."""


@dataclass
class MultiphaseTimeSeriesData(TimeSeriesData):
    mass_g: List[float] = field(default_factory=list)
    """Mass of the gaseous phase at each time point"""
    mass_aq: List[float] = field(default_factory=list)
    """Mass of the aqueous phase at each time point"""
    mass_tot: List[float] = field(default_factory=list)
    """Total mass (gaseous + aqueous) at each time point"""
    exact_mass_tot: List[Optional[float]] = field(default_factory=list)
    """Exact/expected total mass at each time point, if available"""
    volume_g: List[float] = field(default_factory=list)
    """Volume of the gaseous phase at each time point"""
    volume_aq: List[float] = field(default_factory=list)
    """Volume of the aqueous phase at each time point"""
    volume_tot: List[float] = field(default_factory=list)
    """Total volume (gaseous + aqueous) at each time point"""

    # ! ----- DATA MANAGEMENT -----

    def append(
        self,
        time: float,
        name: str,
        mass_g: float,
        mass_aq: float,
        exact_mass_tot: Optional[float],
        volume_g: float,
        volume_aq: float,
    ) -> None:
        """Append a new data point to the multiphase data.

        Args:
            time (float): Time at which the data was recorded.
            name (str): Name for the data point, e.g. name of raw image.
            mass_g (float): Mass of the gaseous phase at this time point.
            mass_aq (float): Mass of the aqueous phase at this time point.
            exact_mass_tot (Optional[float]): Exact/expected total mass.
            volume_g (float): Volume of the gaseous phase at this time point.
            volume_aq (float): Volume of the aqueous phase at this time point.
        """
        self.time.append(time)
        self.name.append(name)
        self.mass_g.append(mass_g)
        self.mass_aq.append(mass_aq)
        self.mass_tot.append(mass_g + mass_aq)
        self.exact_mass_tot.append(exact_mass_tot)
        self.volume_g.append(volume_g)
        self.volume_aq.append(volume_aq)
        self.volume_tot.append(volume_g + volume_aq)

    def reset(self) -> None:
        """Reset the multiphase data to empty lists."""
        for attr in [
            "time",
            "name",
            "mass_g",
            "mass_aq",
            "mass_tot",
            "exact_mass_tot",
            "volume_g",
            "volume_aq",
            "volume_tot",
        ]:
            setattr(self, attr, [])

    def clean(self, tol: float = np.inf) -> None:
        """Remove data points outside a absolute/relative mass difference threshold.

        The comparison is drawn based on the total and exact mass (reference).

        Args:
            tol (float): Absolute or relative threshold for the mass difference.
                Default is np.inf, which means no cleaning.

        """
        # Determine indices where the relative error is below the threshold (to be kept)
        error = np.abs(
            np.array(self.mass_tot) - np.array(self.exact_mass_tot)
        )  # Absolute error
        keep_indices = np.where(error / (1 + np.array(self.exact_mass_tot)) < tol)[0]

        # Remove data points that do not meet the threshold
        for attr in [
            "time",
            "name",
            "mass_g",
            "mass_aq",
            "mass_tot",
            "exact_mass_tot",
            "volume_g",
            "volume_aq",
            "volume_tot",
        ]:
            attr_list = getattr(self, attr)
            setattr(self, attr, [attr_list[i] for i in keep_indices])

        # Monitor how many indices are removed
        num_all_indices = len(self.time)
        num_kept_indices = len(keep_indices)
        print(
            f"""Removed {num_all_indices - num_kept_indices} out of"""
            f""" {num_all_indices} data points."""
        )

    # ! ----- I/O -----

    def save(self, path: Path) -> None:
        """Save the multiphase data to a csv file."""

        df = pd.DataFrame(
            dict((key, getattr(self, key)) for key in self.__dataclass_fields__)
        )

        df.to_csv(path, index=False)

    def load(self, path: Path) -> None:
        """Load the multiphase data from a csv file."""

        df = pd.read_csv(path)
        for attr in [
            "time",
            "mass_g",
            "mass_aq",
            "mass_tot",
            "exact_mass_tot",
            "volume_g",
            "volume_aq",
            "volume_tot",
        ]:
            setattr(self, attr, df[attr].tolist())
        self.name = df["name"].astype(str).tolist()  # Convert to string

    # ! ----- PLOTTING -----

    def plot_mass_over_time(
        self,
        time_max: Optional[float] = None,
        show: bool = False,
        path: Optional[Path] = None,
    ) -> None:
        """Plot mass of gaseous, aqueous and total phases over time."""

        # Determine the index up to which to plot
        if time_max:
            ind = np.argmax(np.where(np.array(self.time) < time_max)[0]) + 1
        else:
            ind = len(self.time)

        # Plot mass over time
        plt.figure("Mass over time")
        plt.plot(self.time[:ind], self.mass_tot[:ind], color="blue", label="total")
        plt.plot(self.time[:ind], self.mass_g[:ind], color="green", label="gas")
        plt.plot(self.time[:ind], self.mass_aq[:ind], color="orange", label="aqueous")
        if all([m is not None for m in self.exact_mass_tot]):
            plt.plot(
                self.time[:ind],
                self.exact_mass_tot[:ind],
                color="red",
                label="exact",
                linestyle="--",
            )
        # Add dots for data points
        plt.scatter(self.time[:ind], self.mass_tot[:ind], color="blue")
        plt.scatter(self.time[:ind], self.mass_g[:ind], color="green")
        plt.scatter(self.time[:ind], self.mass_aq[:ind], color="orange")

        # Annotations and labels
        plt.xlabel("Time [hrs]")
        plt.ylabel("Mass [kg]")
        plt.title("Mass over time")
        plt.legend()
        plt.tight_layout()

        # Store the plot if a path is provided and display it if requested
        if path:
            plt.savefig(path)
        if show:
            plt.show()
        plt.close()

    def plot_volume_over_time(
        self,
        time_max: Optional[float] = None,
        show: bool = False,
        path: Optional[Path] = None,
    ) -> None:
        """Plot volume of gaseous, aqueous and total phases over time."""

        # Determine the index up to which to plot
        if time_max:
            ind = np.argmax(np.where(np.array(self.time) < time_max)[0]) + 1
        else:
            ind = len(self.time)

        # Plot volume over time
        plt.figure("Volume over time")
        plt.plot(self.time[:ind], self.volume_tot[:ind], color="blue", label="total")
        plt.plot(self.time[:ind], self.volume_g[:ind], color="green", label="gas")
        plt.plot(self.time[:ind], self.volume_aq[:ind], color="orange", label="aqueous")

        # Add dots for data points
        plt.scatter(self.time[:ind], self.volume_tot[:ind], color="blue")
        plt.scatter(self.time[:ind], self.volume_g[:ind], color="green")
        plt.scatter(self.time[:ind], self.volume_aq[:ind], color="orange")

        # Annotations and labels
        plt.xlabel("Time [hrs]")
        plt.ylabel("Volume [m³]")
        plt.title("Volume over time")
        plt.legend()
        plt.tight_layout()

        # Store the plot if a path is provided and display it if requested
        if path:
            plt.savefig(path)
        if show:
            plt.show()
        plt.close()
