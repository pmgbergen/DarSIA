from pathlib import Path
from typing import Literal

import darsia
from darsia.utils.augmented_plotting import plot_contour_on_image
import numpy as np
import skimage
from dataclasses import dataclass, field
import pandas as pd
from datetime import datetime


@dataclass
class SimpleMassAnalysisResults:
    """Summary object."""

    name: str
    """Name for the mass analysis result, e.g., name of raw image."""
    date: datetime | None
    """Date of the mass analysis result."""
    time: float | None
    """Relative time (e.g., since injection start) of the mass analysis result."""
    mass: darsia.Image
    """Total mass of phase."""
    mass_g: darsia.Image
    """Mass of gaseous phase."""
    mass_aq: darsia.Image
    """Mass of aqueous phase."""
    saturation_g: darsia.Image
    """Saturation of gaseous phase."""
    color_signal: darsia.Image
    """Signal of color analysis."""
    concentration_aq: darsia.Image
    """Concentration in aqueous phase."""

    def subregion(self, roi: darsia.CoordinateArray) -> "SimpleMassAnalysisResults":
        return SimpleMassAnalysisResults(
            name=self.name,
            date=self.date,
            time=self.time,
            mass=self.mass.subregion(roi),
            mass_g=self.mass_g.subregion(roi),
            mass_aq=self.mass_aq.subregion(roi),
            saturation_g=self.saturation_g.subregion(roi),
            color_signal=self.color_signal.subregion(roi),
            concentration_aq=self.concentration_aq.subregion(roi),
        )


@dataclass
class SimpleMultiphaseTimeSeriesData(darsia.TimeSeriesData):
    mass_g: list[float] = field(default_factory=list)
    """Mass of the gaseous phase at each time point"""
    mass_aq: list[float] = field(default_factory=list)
    """Mass of the aqueous phase at each time point"""
    mass_tot: list[float] = field(default_factory=list)
    """Total mass (gaseous + aqueous) at each time point"""
    exact_mass_tot: list[float | None] = field(default_factory=list)
    """Exact/expected total mass at each time point, if available"""

    # ! ----- DATA MANAGEMENT -----

    def append(
        self,
        time: float,
        name: str,
        mass_g: float,
        mass_aq: float,
        exact_mass_tot: float | None,
    ) -> None:
        """Append a new data point to the multiphase data.

        Args:
            time (float): Time at which the data was recorded.
            name (str): Name for the data point, e.g. name of raw image.
            mass_g (float): Mass of the gaseous phase at this time point.
            mass_aq (float): Mass of the aqueous phase at this time point.
            exact_mass_tot (Optional[float]): Exact/expected total mass.
        """
        self.time.append(time)
        self.name.append(name)
        self.mass_g.append(mass_g)
        self.mass_aq.append(mass_aq)
        self.mass_tot.append(mass_g + mass_aq)
        self.exact_mass_tot.append(exact_mass_tot)

    def reset(self) -> None:
        """Reset the multiphase data to empty lists."""
        for attr in [
            "time",
            "name",
            "mass_g",
            "mass_aq",
            "mass_tot",
            "exact_mass_tot",
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
        ]:
            setattr(self, attr, df[attr].tolist())
        self.name = df["name"].astype(str).tolist()  # Convert to string


class SimpleRunAnalysis(darsia.MultiphaseTimeSeriesAnalysis):
    """Customized class for analyzing a single run of the FFUM experiment."""

    geometry: darsia.Geometry
    """Geometry for integration of mass."""

    def __init__(self, geometry: darsia.Geometry, colors: dict = {}) -> None:
        """Initialize the SimpleRunAnalysis class.

        Args:
            geometry (darsia.Geometry): Geometry for integration of mass.
            colors (dict, optional): Dictionary specifying colors for plotting.
                Keys are 'aqueous', 'gaseous', and 'mass'. Values are RGB tuples.
                Defaults to None, which uses preset colors.

        """

        super().__init__(geometry=geometry)
        # Overwrite colors for contour plotting
        self.color_aq = colors.get("aqueous", (50, 190, 0))
        """Color for aqueous phase in plots."""
        self.color_g = colors.get("gaseous", (207, 35, 35))
        """Color for gaseous phase in plots."""
        self.color_mass = colors.get("mass", (255, 75, 128))
        """Color for mass in plots."""

        # Use a simple data structure for time series data
        self.data = SimpleMultiphaseTimeSeriesData()
        """Time series data for multiphase mass and volume tracking."""

    # ! ----- MANAGING TIME SERIES DATA -----

    def track(
        self,
        mass_analysis_result: darsia.MassAnalysisResults,
        exact_mass: float | None = None,
    ) -> None:
        """Track the mass analysis result and add to time series data.

        Args:
            mass_analysis_result (darsia.MassAnalysisResults): The mass analysis results
                containing the component data.
            exact_mass (float): The exact injected mass at the time of the analysis result.
            time (float): Time in hours since the start of the injection.

        """
        self.data.append(
            time=mass_analysis_result.time,
            name=mass_analysis_result.name,
            mass_g=self.geometry.integrate(mass_analysis_result.mass_g),
            mass_aq=self.geometry.integrate(mass_analysis_result.mass_aq),
            exact_mass_tot=exact_mass,
        )

    # ! ---- IMAGE AND CONTOUR PLOTTING ----

    def plot_contour_signal(
        self,
        img,
        mass_analysis_result: darsia.MassAnalysisResults,
        path: Path,
    ) -> darsia.Image:
        """Customized contour plot of the signal analysis.

        Args:
            img (darsia.Image): The image on which to plot the contours.
            mass_analysis_result (darsia.MassAnalysisResults): The mass analysis results containing the component data.
            path (Path): Path to save the contour plot image.

        Returns:
            darsia.Image: The contour plot image with mass contours.

        """
        return super().plot_contour_signal(
            img=img,
            mass_analysis_result=mass_analysis_result,
            values_aq=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
            values_g=[0.3, 0.6, 0.9],
            path=path,
            thickness=5,
        )

    def plot_contour_mass(
        self,
        img: darsia.Image,
        mass_analysis_result: darsia.MassAnalysisResults,
        path: Path,
    ) -> darsia.Image:
        """Customized contour plot of the mass analysis.

        Args:
            img (darsia.Image): The image on which to plot the contours.
            mass_analysis_result (darsia.MassAnalysisResults): The mass analysis results containing the component data.
            path (Path): Path to save the contour plot image.

        Returns:
            darsia.Image: The contour plot image with mass contours.

        """
        ref_value = 3
        values = [
            0.03 * ref_value,
            0.075 * ref_value,
            0.1667 * ref_value,
            0.5 * ref_value,
            0.8333 * ref_value,
            0.97 * ref_value,
        ]
        return super().plot_contour_mass(
            img=img,
            mass_analysis_result=mass_analysis_result,
            values=values,
            path=path,
            thickness=5,
        )

    # ! ---- OBSOLETE METHODS - SOON TO BE REMOVED ----

    def plot_pure_contour_signal(
        self,
        img,
        mass_analysis_result: darsia.MassAnalysisResults,
        mode: Literal["aqueous", "gaseous"],
        threshold: float,
        path: Path,
        thickness: int = 5,
    ) -> darsia.Image:
        contour_image = plot_contour_on_image(
            img=darsia.zeros_like(img),
            mask=[
                (
                    mass_analysis_result.normalized_signal_aq
                    if mode == "aqueous"
                    else mass_analysis_result.normalized_signal_g
                )
                > threshold,
            ],
            color=[[255, 255, 255]],
            alpha=[1],
            thickness=thickness,
            path=path,
            show_plot=False,
            return_image=True,
        )
        return contour_image

    def plot_simple_contour_signal(
        self,
        img,
        mass_analysis_result: darsia.MassAnalysisResults,
        path: Path,
        thickness: int = 5,
    ) -> darsia.Image:
        contour_image = plot_contour_on_image(
            img=img,
            mask=[
                mass_analysis_result.normalized_signal_aq > 0.1,
                mass_analysis_result.normalized_signal_g > 0.3,
            ],
            color=[self.color_aq] + [self.color_g],
            alpha=[1] + [0.8],
            thickness=thickness,
            path=path,
            show_plot=False,
            return_image=True,
        )
        return contour_image

    def plot_contour_saturation_concentration(
        self,
        img: darsia.Image,
        mass_analysis_result: darsia.MassAnalysisResults,
        path: Path,
        thickness: int = 5,
    ) -> darsia.Image:
        contour_image = plot_contour_on_image(
            img=img,
            mask=[
                mass_analysis_result.saturation_g > 0.3,
                mass_analysis_result.saturation_g > 0.6,
                mass_analysis_result.saturation_g > 0.9,
                mass_analysis_result.concentration_co2_aq > 0.05,
                mass_analysis_result.concentration_co2_aq > 0.1,
                mass_analysis_result.concentration_co2_aq > 0.3,
                mass_analysis_result.concentration_co2_aq > 0.5,
                mass_analysis_result.concentration_co2_aq > 0.7,
                mass_analysis_result.concentration_co2_aq > 0.9,
            ],
            color=3 * [self.color_g] + 6 * [self.color_aq],
            alpha=[0.3, 0.6, 0.9] + [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            thickness=thickness,
            path=path,
            show_plot=False,
            return_image=True,
        )
        return contour_image

    def plot_contour_saturation(
        self,
        img: darsia.Image,
        mass_analysis_result: darsia.MassAnalysisResults,
        path: Path,
        thickness: int = 5,
    ) -> darsia.Image:
        contour_image = plot_contour_on_image(
            img=img,
            mask=[
                mass_analysis_result.saturation_g > 0.3,
                mass_analysis_result.saturation_g > 0.6,
                mass_analysis_result.saturation_g > 0.9,
            ],
            color=3 * [self.color_g],
            alpha=[0.3, 0.6, 0.9],
            thickness=thickness,
            path=path,
            show_plot=False,
            return_image=True,
        )
        return contour_image

    def plot_contour_concentration(
        self,
        img: darsia.Image,
        mass_analysis_result: darsia.MassAnalysisResults,
        path: Path,
        thickness: int = 5,
    ) -> darsia.Image:
        contour_image = plot_contour_on_image(
            img=img,
            mask=[
                mass_analysis_result.concentration_co2_aq > 0.05,
                mass_analysis_result.concentration_co2_aq > 0.1,
                mass_analysis_result.concentration_co2_aq > 0.3,
                mass_analysis_result.concentration_co2_aq > 0.5,
                mass_analysis_result.concentration_co2_aq > 0.7,
                mass_analysis_result.concentration_co2_aq > 0.9,
            ],
            color=6 * [self.color_aq],
            alpha=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
            thickness=thickness,
            path=path,
            show_plot=False,
            return_image=True,
        )
        return contour_image

    def plot_dissolved_CO2(
        self,
        background: darsia.Image,
        img: darsia.Image,
        mass_analysis_result: darsia.MassAnalysisResults,
        path: Path,
        thickness: int = 5,
    ) -> darsia.Image:
        # Track dissolved CO2
        # Remove gas
        mask_co2 = mass_analysis_result.concentration_co2_aq.img > 0.05
        mask_g = mass_analysis_result.saturation_g.img > 0.3
        mask = mask_co2 & ~mask_g

        # Start with the original image
        original_background = np.clip(np.copy(background.img), 0, 1)
        original_background = skimage.img_as_ubyte(original_background)
        original_img = np.clip(np.copy(img.img), 0, 1)
        original_img = skimage.img_as_ubyte(original_img)

        # Plot background.img and on top of it img, only where mask is True
        original_background[mask] = original_img[mask]
        original_background[mask_g] = (
            0.5 * original_background[mask_g] + 0.5 * original_img[mask_g]
        ).astype(np.uint8)

        # Save the image
        # bgr_canvas = cv2.cvtColor(original_background, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(
        #    str(path),
        #    bgr_canvas,
        #    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        # )
        plot_contour_on_image(
            img=original_background,
            mask=[
                mass_analysis_result.normalized_signal_aq > 0.05,
                mass_analysis_result.normalized_signal_aq > 0.1,
                mass_analysis_result.normalized_signal_aq > 0.3,
                mass_analysis_result.normalized_signal_aq > 0.5,
                mass_analysis_result.normalized_signal_aq > 0.7,
                mass_analysis_result.normalized_signal_aq > 0.9,
                mass_analysis_result.saturation_g > 0.3,
                # mass_analysis_result.normalized_signal_g > 0.3,
                # mass_analysis_result.normalized_signal_g > 0.6,
                # mass_analysis_result.normalized_signal_g > 0.9,
            ],
            color=7 * [self.color_aq],  # + 3 * [self.color_g],
            alpha=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],  # + [0.3, 0.7, 0.9],
            thickness=thickness,
            path=path,
            show_plot=False,
            return_image=False,
        )

    def plot_gas(
        self,
        background: darsia.Image,
        img: darsia.Image,
        mass_analysis_result: darsia.MassAnalysisResults,
        path: Path,
        thickness: int = 5,
    ) -> darsia.Image:
        # Track gas
        mask_co2 = mass_analysis_result.concentration_co2_aq.img > 0.05
        mask_g = mass_analysis_result.saturation_g.img > 0.3
        mask_dissolved = mask_co2 & ~mask_g

        # Start with the original image
        original_background = np.clip(np.copy(background.img), 0, 1)
        original_background = skimage.img_as_ubyte(original_background)
        original_img = np.clip(np.copy(img.img), 0, 1)
        original_img = skimage.img_as_ubyte(original_img)

        # Plot background.img and on top of it img, only where mask is True
        original_background[mask_dissolved] = (
            0.5 * original_background[mask_dissolved]
            + 0.5 * original_img[mask_dissolved]
        ).astype(np.uint8)
        original_background[mask_g] = original_img[mask_g]

        ## Save the image
        # bgr_canvas = cv2.cvtColor(original_background, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(
        #    str(path),
        #    bgr_canvas,
        #    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        # )
        plot_contour_on_image(
            img=original_background,
            mask=[
                # mass_analysis_result.normalized_signal_aq > 0.05,
                # mass_analysis_result.normalized_signal_aq > 0.1,
                # mass_analysis_result.normalized_signal_aq > 0.3,
                # mass_analysis_result.normalized_signal_aq > 0.5,
                # mass_analysis_result.normalized_signal_aq > 0.7,
                # mass_analysis_result.normalized_signal_aq > 0.9,
                mass_analysis_result.saturation_g > 0.3,
                mass_analysis_result.normalized_signal_g > 0.3,
                mass_analysis_result.normalized_signal_g > 0.6,
                mass_analysis_result.normalized_signal_g > 0.9,
            ],
            color=4 * [self.color_g],
            alpha=[0.1, 0.3, 0.7, 0.9],
            thickness=thickness,
            path=path,
            show_plot=False,
            return_image=False,
        )
