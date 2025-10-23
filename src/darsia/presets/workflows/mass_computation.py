import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, Slider

import darsia
from darsia.presets.workflows.simple_run_analysis import (
    SimpleMassAnalysisResults,
    SimpleRunAnalysis,
)
# from darsia.presets.workflows.heterogeneous_color_analysis import (
#    HeterogeneousColorAnalysis,
# )
# from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)

#
#  class MassCalibration:
#
#     def __init__(
#
#         self,
#
#         mass_computation: "MassComputation",
#
#         concentration_analysis: HeterogeneousColorAnalysis,
#
#         fluidflower: Rig,
#
#     ):
#
#         self.mass_computation = mass_computation
#
#         self.concentration_analysis = concentration_analysis
#        self.fluidflower = fluidflower


class MassComputation:
    def __init__(
        self,
        baseline: darsia.Image,
        geometry: darsia.Geometry,
        flash: darsia.SimpleFlash,
        co2_mass_analysis: darsia.CO2MassAnalysis,
    ):
        self.baseline = baseline
        self.geometry = geometry
        self.flash = flash
        self.co2_mass_analysis = co2_mass_analysis
        self.transformation = darsia.PWTransformation(
            supports=[-1, 0, 0.1, 0.25] + np.linspace(0.5, 1.0, 11).tolist() + [10.0],
            values=[0, 0, 0.1, 0.25] + np.linspace(0.5, 2, 11).tolist() + [2],
        )

    def fit(
        self,
        untransformed_images: list[darsia.Image],
        experiment: darsia.ProtocolledExperiment,
    ) -> None:
        # Check expected status
        times = [experiment.time_since_start(img.date) for img in untransformed_images]
        expected_mass = [
            experiment.injection_protocol.injected_mass(img.date)
            for img in untransformed_images
        ]

        # Cache number of values in transformation
        num_values = len(self.transformation.values)

        # Step 1: pre-mass analysis (nothing to do as images already converted to signals)
        analysis = SimpleRunAnalysis(self.geometry)
        folder = Path("calibration_mass")
        # Remove everything in the folder
        if folder.exists():
            for file in folder.iterdir():
                if file.is_file():
                    file.unlink()
        folder.mkdir(parents=True, exist_ok=True)

        # Step 2: Initialize multiphase time series
        integrated_mass = []
        integrated_mass_g = []
        integrated_mass_aq = []
        square_error = []

        def update_analysis():
            """Auxiliary function to update multiphase time series analysis."""
            nonlocal analysis
            nonlocal integrated_mass
            nonlocal integrated_mass_g
            nonlocal integrated_mass_aq
            nonlocal square_error

            analysis.reset()
            for img, exact_mass, time in zip(
                untransformed_images, expected_mass, times
            ):
                # Mass analysis
                mass_analysis_result: SimpleMassAnalysisResults = self(img)

                # Track result
                analysis.track(mass_analysis_result, exact_mass=exact_mass, time=time)

                # # Clean data - TODO?
                # analysis.clean(threshold=1.0)

            # Monitor mass evolution over time
            integrated_mass = analysis.data.mass_tot
            integrated_mass_g = analysis.data.mass_g
            integrated_mass_aq = analysis.data.mass_aq

            # Errors
            square_error = np.square(
                np.array(integrated_mass) - np.array(expected_mass)
            )

        # Log errors and calibration iteration
        log_calibration_iteration = 0
        log_error = []
        log_transformation_supports = np.empty((0, num_values))
        log_transformation_values = np.empty((0, num_values))

        def log_iteration():
            """Auxiliary function to log calibration iteration."""
            nonlocal num_values  # not modified
            nonlocal log_error
            nonlocal log_transformation_supports
            nonlocal log_transformation_values

            log_error.append(np.sqrt(np.sum(square_error)))

            # Log calibration iteration (supports and values)
            log_transformation_supports = np.vstack(
                (log_transformation_supports, self.transformation.supports)
            )
            log_transformation_values = np.vstack(
                (log_transformation_values, self.transformation.values)
            )

            # Log the results in a csv file
            df_dict = {
                "error": log_error,
            }
            for i in range(num_values):
                df_dict[f"support_{i}"] = log_transformation_supports[:, i]
                df_dict[f"value_{i}"] = log_transformation_values[:, i]
            df = pd.DataFrame(df_dict)
            df.to_csv(folder / "transformation.csv", index=False)

            # Log the current active transformations
            self.transformation.log(folder / "transformation.png")

        # Update multiphase time series analysis and log iteration
        update_analysis()
        log_iteration()

        # Make one annotation and four plots
        # 0. Error as text in the top left corner
        # 1. PWTransformation for gas with sliders
        # 2. PWTransformation for aqueous with sliders
        # 3. Integrated mass over time, entire run, updated upon activation
        # 4. Integrated mass over time, first 12 hours, updated upon activation

        # Set up the figure layout: sliders on the left, plots on the right
        fig = plt.figure(figsize=(18, 10))
        # Use gridspec to allocate space for sliders and plots
        import matplotlib.gridspec as gridspec

        gs = gridspec.GridSpec(
            nrows=1,
            ncols=2,
            width_ratios=[1, 2],  # Sliders:Plots
            left=0.05,
            right=0.95,
            bottom=0.05,
            top=0.95,
            wspace=0.3,
        )
        # Sliders area (left)
        slider_area = plt.subplot(gs[0, 0])
        slider_area.axis("off")
        # Plots area (right, 2x1)
        gs_plots = gridspec.GridSpecFromSubplotSpec(
            nrows=2,
            ncols=1,
            subplot_spec=gs[0, 1],
            height_ratios=[1, 1],
            hspace=0.3,
        )
        ax = [plt.subplot(gs_plots[0, 0]), plt.subplot(gs_plots[1, 0])]

        # ! ---- ANNOTATION 0: ERROR ----

        # Print the error in the top left corner
        error_txt = fig.text(
            0.5,
            1.25,
            f"""Error: {np.sqrt(np.sum(square_error)):.2e}""",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax[0].transAxes,
        )

        # ! ---- PLOT 1: PW TRANSFORMATION FOR GAS ----

        # Pre-define a signal domain [0, 2] for plotting of transformations
        signal_domain = np.arange(0.0, 2.0, 0.001)

        # Set the size of the subfigure to leave space for the sliders
        [pw_transformation] = ax[0].plot(
            signal_domain,
            self.transformation(signal_domain),
            color="green",
            label="transformation",
        )
        pw_transformation_scatter = ax[0].scatter(
            self.transformation.supports,
            self.transformation.values,
            color="green",
        )
        ax[0].set_xlabel("Signal")
        ax[0].set_ylabel("Converted signal")
        ax[0].set_title("PWTransformation")
        ax[0].set_xlim(0, 1)
        ax[0].set_ylim(0, 2)
        ax[0].legend()

        # ! ---- SLIDERS FOR TRANSFORMATION ----

        # Define an axes area and draw sliders for gas transformation
        slider_ax = {}
        slider = {}
        for i in range(num_values):
            slider_ax[i] = fig.add_axes(
                [
                    0.07,  # left (relative to fig)
                    0.85 - i * 0.05,  # bottom (stacked vertically)
                    0.18,  # width
                    0.03,  # height
                ]
            )
            slider[i] = Slider(
                slider_ax[i],
                f"Support {i}\n{self.transformation.supports[i]:.2f}",
                0.0,
                2.0,
                valinit=self.transformation.values[i],
            )

        # Define an action for modifying the line when any slider's value changes
        def sliders_on_changed(val):
            """Update the transformation based on slider values and redraw the plot."""
            self.transformation.update(
                values=[slider[i].val for i in range(num_values)]
            )
            pw_transformation.set_ydata(self.transformation(signal_domain))
            pw_transformation_scatter.set_offsets(
                np.c_[
                    self.transformation.supports,
                    self.transformation.values,
                ]
            )
            fig.canvas.draw_idle()

        # Connect the sliders to the update function
        for i in range(len(slider)):
            slider[i].on_changed(sliders_on_changed)

        # ! ---- PLOT 2: PW TRANSFORMATION FOR AQUEOUS ----

        ...

        # ! ---- PLOT 3: INTEGRATED MASS OVER TIME, ENTIRE RUN ----

        # Combine plot and scatter for integrated mass over time.
        # Decompose into total, gas, and aqueous mass, and add
        # expected mass using a dashed line.
        ax[1].set_xlabel("Time (h)")
        ax[1].set_ylabel("Mass (g)")
        [integrated_mass_plot] = ax[1].plot(
            times,
            integrated_mass,
            color="blue",
            label="total",
        )
        [integrated_mass_plot_g] = ax[1].plot(
            times,
            integrated_mass_g,
            color="green",
            label="gas",
        )
        [integrated_mass_plot_aq] = ax[1].plot(
            times,
            integrated_mass_aq,
            color="orange",
            label="aqueous",
        )
        ax[1].plot(
            times,
            expected_mass,
            linestyle="--",
            color="red",
            label="injected",
        )
        integrated_mass_scatter = ax[1].scatter(
            times,
            integrated_mass,
            color="blue",
        )
        integrated_mass_scatter_g = ax[1].scatter(
            times,
            integrated_mass_g,
            color="green",
        )
        integrated_mass_scatter_aq = ax[1].scatter(
            times,
            integrated_mass_aq,
            color="orange",
        )
        ax[1].set_ylim(0.0, 0.01)
        ax[1].legend()
        ax[1].set_title("Integrated mass over time, entire run")

        # ! ---- UPDATING THE MASS PLOTS ----

        # Add a button for resetting the parameters
        update_button_ax = fig.add_axes([0.85, 0.95, 0.1, 0.04])
        update_button = Button(
            update_button_ax,
            "Update",
            color="lightgoldenrodyellow",
            hovercolor="0.975",
        )

        def update_button_on_clicked(mouse_event):
            """Update the mass plots based on the current transformations."""
            nonlocal log_calibration_iteration
            log_calibration_iteration += 1

            # Deactivate the mass plots by plotting additional text 'Loading...' in the center of the plot
            loading_txt = ax[1].text(
                np.max(times) // 2,
                0.008,
                "Loading...",
                horizontalalignment="center",
                # color="r",
            )
            fig.canvas.draw_idle()

            # Update the multiphase time series analysis
            update_analysis()

            # Update the mass plots (Plot 3)
            integrated_mass_plot.set_ydata(integrated_mass)
            integrated_mass_plot_g.set_ydata(integrated_mass_g)
            integrated_mass_plot_aq.set_ydata(integrated_mass_aq)

            integrated_mass_scatter.set_offsets(np.c_[times, integrated_mass])
            integrated_mass_scatter_g.set_offsets(np.c_[times, integrated_mass_g])
            integrated_mass_scatter_aq.set_offsets(np.c_[times, integrated_mass_aq])

            # Log the results
            log_iteration()

            # Replace the error text with the updated error
            error_txt.set_text(f"""Error: {np.sqrt(np.sum(square_error)):.2e}""")

            # Remove the text again
            loading_txt.remove()

            # Redraw the canvas
            fig.canvas.draw_idle()

            # Plot the results in a png file
            plt.savefig(
                folder / f"pw_transformation_{log_calibration_iteration}.png",
                dpi=500,
            )

        # Connect the button to the update function
        update_button.on_clicked(update_button_on_clicked)

        # Show the plot
        plt.show()

        # Update multiphase time series analysis and log iteration
        update_analysis()
        log_iteration()

    def __call__(self, untransformed_img: darsia.Image) -> SimpleMassAnalysisResults:
        """Compute mass based on concentration image.

        Args:
            img (darsia.Image): Untransformed signal [0, 2].

        """
        img = self.transformation(untransformed_img)
        c_aq, s_g = self.flash(img)

        gas_density_array = self.co2_mass_analysis.density_gaseous_co2
        solubility_array = self.co2_mass_analysis.solubility_co2
        mass_g_array = gas_density_array * s_g.img
        mass_aq_array = solubility_array * c_aq.img * (1 - s_g.img)
        return SimpleMassAnalysisResults(
            name=img.name,
            date=img.date,
            mass=darsia.full_like(img, mass_g_array + mass_aq_array),
            mass_g=darsia.full_like(img, mass_g_array),
            mass_aq=darsia.full_like(img, mass_aq_array),
            saturation_g=s_g,
            color_signal=img,
            concentration_aq=c_aq,
        )

    def compute_total_mass(self, img: darsia.Image) -> float:
        """Compute total mass in the image.

        Args:
            img (darsia.Image): Mass image.

        Returns:
            float: Total mass in the image.
        """
        mass = self(img)
        total_mass = self.geometry.integrate(mass)

        return total_mass

    def calibration(self, calibration_data: dict):
        # Calibrate transformations based on provided data
        for label, data in calibration_data.items():
            # Example: simple linear scaling based on mean values
            target_mean = data["target_mean"]
            current_mean = data["current_mean"]
            scale = target_mean / current_mean if current_mean != 0 else 1.0
            self.transformations[label] = lambda x, s=scale: x * s

    def load(self, path: Path):
        with open(path, "r") as f:
            self.transformations = json.load(f)

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.transformations, f)
            f.close()

    def show(self):
        # Visualize the transformations
        ...
