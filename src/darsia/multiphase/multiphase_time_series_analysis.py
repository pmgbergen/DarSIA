from abc import abstractmethod
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

import darsia
from darsia.utils.augmented_plotting import plot_contour_on_image


class MultiphaseTimeSeriesAnalysis:
    """Class for analyzing a single run of the multiphase flow experiment.
    Includes data tracking, time series data management, and plotting of results.

    """

    def __init__(self, geometry: darsia.Geometry):
        self.geometry = geometry
        """Geometry for integration of mass and volume."""
        self.data = darsia.MultiphaseTimeSeriesData()
        """Time series data for multiphase mass and volume tracking."""
        self.color_aq = (255, 0, 0)
        """Color for aqueous phase in plots."""
        self.color_g = (0, 255, 0)
        """Color for gaseous phase in plots."""
        self.color_mass = (0, 0, 255)
        """Color for mass in plots."""

    # ! ----- MANAGING TIME SERIES DATA -----

    def save(self, path: Path) -> None:
        """Save the multiphase time series data to a csv file.

        Args:
            path (Path): Path to the csv file to save the time series data.

        """
        self.data.save(path)

    def load(self, path: Path) -> None:
        """Load the multiphase time series data from a csv file.

        Args:
            path (Path): Path to the csv file containing the time series data.

        """
        self.data.load(path)

    def reset(self) -> None:
        """Reset the tracking time series data to empty lists."""
        self.data.reset()

    @abstractmethod
    def track(self, mass_analysis_result: darsia.MassAnalysisResults) -> None:
        """Track the mass analysis result and add to time series data.

        Use: self.data.append(...) to add data to the time series.

        Args:
            mass_analysis_result (darsia.MassAnalysisResults): The mass analysis results containing the component data.
            This should include the mass and volume data for gaseous and aqueous phases.

        """
        ...

    def clean(self, threshold) -> None:
        """Remove faulty data from tracked time series data.

        Args:
            threshold (float): Threshold for cleaning the data. Data points with absolute values below this threshold will be removed.

        """
        self.data.clean(tol=threshold)

    # ! ---- PLOTTING TIME SERIES DATA ----

    def plot_mass_over_time(self, path: Path, **kwargs) -> None:
        """Plot the time series mass data of gaseous and aqueous phases.

        Args:
            path (Path): Path to save the plot.
            **kwargs: Additional keyword arguments, e.g., 'upper_time_limit' to limit the time range.

        """
        self.data.plot_mass_over_time(
            time_max=kwargs.get("upper_time_limit", None), path=path, show=False
        )

    def plot_volume_over_time(self, path: Path, **kwargs) -> None:
        """Plot the time series volume of gaseous and aqueous phases.

        Args:
            path (Path): Path to save the plot.
            **kwargs: Additional keyword arguments, e.g., 'upper_time_limit' to limit the time range.

        """
        self.data.plot_volume_over_time(
            time_max=kwargs.get("upper_time_limit", None), path=path, show=False
        )

    # ! ---- IMAGE AND CONTOUR PLOTTING ----

    def plot_result(
        self,
        mass_analysis_result: darsia.MassAnalysisResults,
        component,
        path: darsia.Path,
        vmax: Optional[float] = None,
    ) -> None:
        """Plot the result of the mass analysis for a specific component.

        Args:
            mass_analysis_result (darsia.MassAnalysisResults): The mass analysis results containing the component data.
            component (str): The component to plot, e.g., 'normalized_signal_aq', 'normalized_signal_g', or 'mass'.
            path (darsia.Path): Path to save the plot.
            vmax (Optional[float]): Maximum value for the color scale. If None, the maximum value of the image is used.

        """
        plt.figure()
        if vmax is not None:
            plt.imshow(getattr(mass_analysis_result, component).img)
        else:
            plt.imshow(getattr(mass_analysis_result, component).img, vmax=vmax)
        plt.savefig(path)
        plt.close()

    def plot_contour_signal(
        self,
        img,
        mass_analysis_result: darsia.MassAnalysisResults,
        values_aq: list[float],
        values_g: list[float],
        path: Path,
        thickness: int = 5,
    ) -> darsia.Image:
        """Plot contours of the aqueous and gaseous signals on the image.

        Args:
            img (darsia.Image): The image on which to plot the contours.
            mass_analysis_result (darsia.MassAnalysisResults): The mass analysis results containing the signals.
            values_aq (list[float]): List of aqueous signal values to create contours for.
            values_g (list[float]): List of gaseous signal values to create contours for.
            path (Path): Path to save the contour image.
            thickness (int, optional): Thickness of the contour lines. Defaults to 5.

        Returns:
            darsia.Image: The contour image with aqueous and gaseous signal contours plotted.

        """
        contour_image = plot_contour_on_image(
            img=img,
            mask=[
                mass_analysis_result.normalized_signal_aq > value for value in values_aq
            ]
            + [mass_analysis_result.normalized_signal_g > value for value in values_g],
            color=[self.color_aq for _ in values_aq] + [self.color_g for _ in values_g],
            alpha=values_aq + values_g,
            thickness=thickness,
            path=path,
            show_plot=False,
            return_image=True,
        )
        return contour_image

    def plot_contour_mass(
        self,
        img: darsia.Image,
        mass_analysis_result: darsia.MassAnalysisResults,
        values: list[float],
        path: Path,
        thickness: int = 5,
    ) -> darsia.Image:
        """Plot contours of the mass on the image.

        Args:
            img (darsia.Image): The image on which to plot the contours.
            mass_analysis_result (darsia.MassAnalysisResults): The mass analysis results containing the mass data.
            values (list[float]): List of mass values to create contours for.
            path (Path): Path to save the contour image.
            thickness (int, optional): Thickness of the contour lines. Defaults to 5.

        Returns:
            darsia.Image: The contour image with mass contours plotted.

        """
        # Map values onto 0.1..1 through linear transformation
        mapped_values = [
            (value - min(values)) / (max(values) - min(values)) * 0.9 + 0.1
            for value in values
        ]
        contour_image = plot_contour_on_image(
            img=img,
            mask=[mass_analysis_result.mass > value for value in values],
            color=[self.color_mass for _ in values],
            alpha=mapped_values,
            thickness=thickness,
            path=path,
            show_plot=False,
            return_image=True,
        )
        return contour_image
