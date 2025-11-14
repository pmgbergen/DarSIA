import copy
import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from darsia.presets.workflows.simple_run_analysis import (
    SimpleMassAnalysisResults,
    SimpleRunAnalysis,
)
import time

import darsia

logger = logging.getLogger(__name__)


class HeterogeneousColorToMassAnalysis(darsia.ConcentrationAnalysis):
    """Hetereogeneous analysis converting colors to mass."""

    def __init__(
        self,
        baseline: darsia.Image,
        labels: darsia.Image,
        color_mode: darsia.ColorMode,
        color_path_interpretation: dict[int, darsia.ColorPathInterpolation],
        signal_functions: dict[int, darsia.ColorPathFunction],
        flash: darsia.SimpleFlash,
        co2_mass_analysis: darsia.CO2MassAnalysis,
        geometry: darsia.Geometry,
        restoration: darsia.Model | None = None,
        ignore_labels: list[int] | None = None,
    ):
        base_model = darsia.CombinedModel(
            [
                darsia.HeterogeneousModel(
                    color_path_interpretation,
                    labels,
                    ignore_labels=ignore_labels,
                ),
                # restoration ...
            ]
        )

        # Define general config options
        config = {
            "diff option": "plain",
            "restoration -> model": False,
        }

        # Define general ConcentrationAnalysis.
        super().__init__(
            base=baseline if color_mode == darsia.ColorMode.RELATIVE else None,
            restoration=restoration,
            labels=labels,
            model=base_model,
            **config,
        )

        self.signal_model = darsia.HeterogeneousModel(
            signal_functions,
            labels,
            ignore_labels=ignore_labels,
        )
        """Model converting color interpretation to pH."""

        self.flash = flash
        """Flash model for converting pH to concentrations."""

        self.co2_mass_analysis = co2_mass_analysis
        """Mass computation tool."""

        self.geometry = geometry
        """Geometry of the experiment - for the integration of mass during calibration."""

    def call_color_interpretation(self, image: darsia.Image) -> darsia.Image:
        return super().__call__(image)

    def call_pH_analysis(self, color_interpretation: darsia.Image) -> darsia.Image:
        """Run the mass analysis on a single image."""
        return self.signal_model(color_interpretation)

    def call_flash_and_mass_analysis(
        self, pH: darsia.Image
    ) -> SimpleMassAnalysisResults:
        """Run the mass analysis on a single image."""
        c_aq, s_g = self.flash(pH)
        mass_analysis_result: SimpleMassAnalysisResults = (
            self.co2_mass_analysis.mass_analysis(
                c_aq=c_aq,
                s_g=s_g,
            )
        )
        return mass_analysis_result

    def __call__(self, image: darsia.Image) -> darsia.Image:
        """Run the analysis on a single image."""
        color_interpretation = self.call_color_interpretation(image)
        pH = self.call_pH_analysis(color_interpretation)
        mass_analysis_result = self.call_flash_and_mass_analysis(pH)
        return mass_analysis_result

    def manual_calibration(
        self,
        images: list[darsia.Image],
        experiment: darsia.ProtocolledExperiment,
        cmap=None,
    ) -> None:
        """Define a local color path for a specific label.

        Instructions:
            - Pick a label in the image using the mouse and select a rectangle.
            - Define a new color path just for this label.
            - Tune the values for the color path.

        Args:
            image (darsia.Image): The image from which to define the local color path.

        """
        # ! ---- IMAGES -----

        # Set up parameters for coarse visualization
        coarse_rows = max(200, self.labels.img.shape[0] // 4)
        coarse_cols = int(
            self.labels.img.shape[1] / self.labels.img.shape[0] * coarse_rows
        )
        coarse_shape = (coarse_rows, coarse_cols)

        # Coarsen the image and labels for better visualization
        coarse_labels = darsia.resize(
            self.labels,
            shape=coarse_shape,
            interpolation="inter_nearest",
        )
        coarse_images = [darsia.resize(image, shape=coarse_shape) for image in images]

        # ! ---- COLOR INTERPRETATIONS ----
        color_interpretations = [
            self.call_color_interpretation(image) for image in images
        ]

        # Allocate variable for image selector
        image_idx = 0

        def image_idx_selector() -> int:
            """Interactive selection of plotted images - allow to click for showing the next coarse images - keep track of the idx in the list and return when finishing."""
            nonlocal image_idx

            # Create a simple figure for image selection
            fig, ax = plt.subplots(figsize=(8, 6))
            current_image = coarse_images[image_idx]
            img_display = ax.imshow(current_image.img)
            ax.set_title(
                f"Image {image_idx + 1}/{len(images)} - Click to cycle through images, close window to finish"
            )
            ax.axis("off")

            def on_click(event):
                nonlocal image_idx
                if event.inaxes == ax:
                    # Cycle to next image
                    image_idx = (image_idx + 1) % len(images)
                    current_image = coarse_images[image_idx]
                    img_display.set_data(current_image.img)
                    ax.set_title(
                        f"Image {image_idx + 1}/{len(images)} - Click to cycle through images, close window to finish"
                    )
                    fig.canvas.draw_idle()

            # Connect click event
            fig.canvas.mpl_connect("button_press_event", on_click)

            plt.show()
            return image_idx

        image_idx = image_idx_selector()

        # Allocate variables for mass analysis
        # Step 2: Initialize multiphase time series
        times = []
        expected_mass = []
        integrated_mass = []
        integrated_mass_g = []
        integrated_mass_aq = []
        square_error = []

        analysis = SimpleRunAnalysis(self.geometry)

        def update_analysis():
            """Auxiliary function to update multiphase time series analysis."""
            # nonlocal analysis
            nonlocal times
            nonlocal expected_mass
            nonlocal integrated_mass
            nonlocal integrated_mass_g
            nonlocal integrated_mass_aq
            nonlocal square_error

            analysis.reset()
            for img in images:
                # Preliminaries
                state = experiment.pressure_temperature_protocol.get_state(img.time)
                self.co2_mass_analysis.update_state(
                    atmospheric_pressure=state.pressure, temperature=state.temperature
                )

                # Signal analysis
                tic_local = time.time()
                pH = self.call_pH_analysis(color_interpretation)
                print(f"Color analysis took {time.time() - tic_local} seconds")

                # Mass computation
                tic_local = time.time()
                mass_analysis_result = self.call_flash_and_mass_analysis(pH)
                print(f"Mass computation took {time.time() - tic_local} seconds")

                # Compute expected mass
                exact_mass = experiment.injection_protocol.injected_mass(img.date)

                # Track result
                tic_local = time.time()
                analysis.track(
                    mass_analysis_result,
                    exact_mass=exact_mass,
                )
                print(f"Tracking took {time.time() - tic_local} seconds")

                # # Clean data - TODO?
                # analysis.clean(threshold=1.0)

            # Monitor mass evolution over time
            times = analysis.data.time
            expected_mass = analysis.data.exact_mass_tot
            integrated_mass = analysis.data.mass_tot
            integrated_mass_g = analysis.data.mass_g
            integrated_mass_aq = analysis.data.mass_aq

            # Errors
            square_error = np.square(
                np.array(integrated_mass) - np.array(expected_mass)
            )

        tic = time.time()
        update_analysis()
        print(f"Update analysis took {time.time() - tic} seconds")

        # Interactive tuning of values for each color path (chosen by user)
        done_picking_new_labels = False
        while not done_picking_new_labels:
            # Pick image
            # image = images[image_idx]
            coarse_image = coarse_images[image_idx]
            color_interpretation = color_interpretations[image_idx]

            # Identify mass density
            pH = self.call_pH_analysis(color_interpretation)
            mass_analysis_result = self.call_flash_and_mass_analysis(pH)
            density = mass_analysis_result.mass

            # Pick label interactively
            assistant = darsia.RectangleSelectionAssistant(
                density, labels=self.labels, cmap=cmap
            )

            # Identify the label of interest
            label_box: Tuple[slice, slice] = assistant()
            label = np.argmax(np.bincount(self.labels.img[label_box].ravel()))

            done_tuning_values = False
            while not done_tuning_values:

                def show_tuner(idx):
                    nonlocal done_tuning_values, done_picking_new_labels
                    fig, ax_conc = plt.subplots(figsize=(8, 4))
                    ax_image = plt.axes([0.05, 0.5, 0.15, 0.4])
                    plt.subplots_adjust(left=0.25, bottom=0.25)
                    ax_image.imshow(coarse_image.img)
                    mask = np.zeros_like(coarse_labels.img, dtype=np.uint8)
                    mask[coarse_labels.img == idx] = 1
                    ax_image.imshow(mask, alpha=0.5, cmap="gray", vmin=0, vmax=1)
                    ax_conc.set_title(f"Tune values for color path #{idx}")

                    sliders_color_to_signal = []
                    slider_height = 0.03
                    for i, val in enumerate(self.signal_model[idx].values):
                        ax_slider = plt.axes(
                            [0.25, 0.15 - i * slider_height, 0.65, slider_height]
                        )
                        slider = Slider(
                            ax_slider,
                            f"Value {i}",
                            -0.5,
                            1.5,
                            valinit=val,
                            valstep=0.05,
                        )
                        sliders_color_to_signal.append(slider)

                    # Determine number for positioning of flash sliders
                    num_sliders = len(self.signal_model[idx].values)

                    # Add two sliders for thresholding flash model
                    ax_slider_phase = plt.axes(
                        [
                            0.25,
                            0.15 - (num_sliders + 2) * slider_height,
                            0.65,
                            slider_height,
                        ]
                    )
                    slider_phase = Slider(
                        ax_slider_phase,
                        "Flash cut-off",
                        0.0,
                        1.5,
                        valinit=self.flash.cut_off,
                        valstep=0.05,
                    )
                    ax_slider_max = plt.axes(
                        [
                            0.25,
                            0.15 - (num_sliders + 3) * slider_height,
                            0.65,
                            slider_height,
                        ]
                    )
                    slider_max = Slider(
                        ax_slider_max,
                        "Flash max",
                        0.0,
                        1.5,
                        valinit=self.flash.max_value,
                        valstep=0.05,
                    )
                    sliders_flash = [slider_phase, slider_max]

                    ax_update = plt.axes([0.8, 0.925, 0.1, 0.04])
                    btn_update = Button(ax_update, "Update values")
                    ax_close = plt.axes([0.68, 0.925, 0.1, 0.04])
                    btn_close = Button(ax_close, "Next layer")
                    ax_next_image = plt.axes([0.56, 0.925, 0.1, 0.04])
                    btn_next_image = Button(ax_next_image, "Switch image")
                    ax_finish = plt.axes([0.44, 0.925, 0.1, 0.04])
                    btn_finish = Button(ax_finish, "Finish")

                    # Determine  and plot density
                    pH = self.call_pH_analysis(color_interpretation)
                    c_aq, s_g = self.flash(pH)
                    mass_analysis_result: SimpleMassAnalysisResults = (
                        self.co2_mass_analysis.mass_analysis(
                            c_aq=c_aq,
                            s_g=s_g,
                        )
                    )
                    density = mass_analysis_result.mass
                    coarse_density = darsia.resize(density, shape=coarse_shape)

                    # Extract contours for s_w and c_aq if available, added to the coarse image.
                    # TODO
                    # c_aq ...
                    # s_g ...
                    coarse_contour = coarse_image.img.copy()

                    # Plot
                    density_img = ax_conc.imshow(coarse_density.img, cmap=cmap)
                    # contour_img = ax_contour.imshow(coarse_contour.img)

                    def update_density(event=None):
                        # Update parameters
                        self.signal_model.update_model_parameters(
                            [slider.val for slider in sliders_color_to_signal]
                        )
                        self.flash.update(
                            cutoff=sliders_flash[0].val,
                            max_value=sliders_flash[1].val,
                        )

                        # Update images
                        pH = self.call_pH_analysis(color_interpretation)
                        c_aq, s_g = self.flash(pH)
                        mass_analysis_result: SimpleMassAnalysisResults = (
                            self.co2_mass_analysis.mass_analysis(
                                c_aq=c_aq,
                                s_g=s_g,
                            )
                        )
                        density = mass_analysis_result.mass
                        coarse_density = darsia.resize(density, shape=coarse_shape)

                        # Update contours TODO

                        # Update plots
                        density_img.set_data(coarse_density.img)
                        # contour_img.set_data(coarse_contour.img)
                        fig.canvas.draw_idle()

                    btn_update.on_clicked(update_density)

                    def close(event):
                        nonlocal done_tuning_values
                        done_tuning_values = True
                        plt.close("all")

                    btn_close.on_clicked(close)

                    def finish(event):
                        nonlocal done_tuning_values, done_picking_new_labels
                        done_tuning_values = True
                        done_picking_new_labels = True
                        plt.close("all")

                    btn_finish.on_clicked(finish)

                    def next_image(event):
                        nonlocal image_idx, done_tuning_values
                        image_idx = (image_idx + 1) % len(images)
                        done_tuning_values = True
                        plt.close("all")

                    btn_next_image.on_clicked(next_image)

                    plt.show()

                show_tuner(label)

    def global_calibration_flash(
        self,
        mass_computation: MassComputation,
        mask: darsia.Image,
        calibration_images: list[darsia.Image],
        experiment: darsia.ProtocolledExperiment,
        cmap=None,
        show: bool = False,
    ) -> None:
        """Coarse tuning of the color signal analysis."""

        # Check expected status

        # Step 1: pre-mass analysis (nothing to do as images already converted to signals)
        analysis = SimpleRunAnalysis(mass_computation.geometry)

        # # Tracking...
        # folder = Path("calibration_mass")  # TODO?
        # # Remove everything in the folder
        # if folder.exists():
        #     for file in folder.iterdir():
        #         if file.is_file():
        #             file.unlink()
        # folder.mkdir(parents=True, exist_ok=True)

        # Step 2: Initialize multiphase time series
        times = []
        expected_mass = []
        integrated_mass = []
        integrated_mass_g = []
        integrated_mass_aq = []
        square_error = []

        def update_analysis():
            """Auxiliary function to update multiphase time series analysis."""
            nonlocal analysis
            nonlocal times
            nonlocal expected_mass
            nonlocal integrated_mass
            nonlocal integrated_mass_g
            nonlocal integrated_mass_aq
            nonlocal square_error

            analysis.reset()
            for img in calibration_images:
                # Preliminaries
                img_time = img.time
                # TODO update co2_mass_analysis/flash etc based on img_time.

                # Signal analysis
                tic_local = time.time()
                signal = self(img)
                print(f"Color analysis took {time.time() - tic_local} seconds")

                # Mass computation
                tic_local = time.time()
                mass_analysis_result: SimpleMassAnalysisResults = mass_computation(
                    signal
                )
                print(f"Mass computation took {time.time() - tic_local} seconds")

                # Compute expected mass
                exact_mass = experiment.injection_protocol.injected_mass(img.date)

                # Track result
                tic_local = time.time()
                analysis.track(
                    mass_analysis_result,
                    exact_mass=exact_mass,
                )
                print(f"Tracking took {time.time() - tic_local} seconds")

                # # Clean data - TODO?
                # analysis.clean(threshold=1.0)

            # Monitor mass evolution over time
            times = analysis.data.time
            expected_mass = analysis.data.exact_mass_tot
            integrated_mass = analysis.data.mass_tot
            integrated_mass_g = analysis.data.mass_g
            integrated_mass_aq = analysis.data.mass_aq

            # Errors
            square_error = np.square(
                np.array(integrated_mass) - np.array(expected_mass)
            )

        tic = time.time()
        update_analysis()
        toc = time.time()

        from icecream import ic

        ic(f"First analysis took {toc - tic} seconds")

        # TODO log iteration?

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

        plt.show()

    def save(self, path: Path) -> None:
        """Save the calibration data to json file.

        Args:
            path (Path): The path to save the calibration data.

        """
        raise NotImplementedError("continue here - check HeterogeneousColorAnalysis")

    @staticmethod
    def load(self, path: Path) -> "HeterogeneousColorToMassAnalysis":
        """Load the calibration data from json file.

        Args:
            path (Path): path to load the model

        """
        raise NotImplementedError("continue here - check HeterogeneousColorAnalysis")
