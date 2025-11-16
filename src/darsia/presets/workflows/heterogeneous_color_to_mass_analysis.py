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
import matplotlib.gridspec as gridspec

import darsia

logger = logging.getLogger(__name__)


class HeterogeneousColorToMassAnalysis:
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

        # TODO make use of ignore_labels
        # TODO make use of restoration

        # Define general ConcentrationAnalysis.
        self.color_analysis = darsia.ConcentrationAnalysis(
            base=baseline if color_mode == darsia.ColorMode.RELATIVE else None,
            restoration=restoration,
            labels=labels,
            model=base_model,
            **config,
        )

        signal_model = darsia.CombinedModel(
            [
                darsia.ClipModel(0.0, 1.0),  # TODO fetch from signal_functions...
                darsia.HeterogeneousModel(
                    signal_functions,
                    labels,
                    ignore_labels=ignore_labels,
                ),
            ],
        )

        self.signal_model = darsia.ConcentrationAnalysis(
            base=None,
            restoration=None,
            labels=labels,
            model=signal_model,
            **config,
        )
        """Model converting color interpretation to pH."""

        self.flash = flash
        """Flash model for converting pH to concentrations."""

        self.co2_mass_analysis = co2_mass_analysis
        """Mass computation tool."""

        self.geometry = geometry
        """Geometry of the experiment - for the integration of mass during calibration."""

    @property
    def labels(self) -> darsia.Image:
        """Labels image - refer to attribute of color analysis."""
        assert self.color_analysis.labels is not None
        return self.color_analysis.labels

    def call_color_interpretation(self, image: darsia.Image) -> darsia.Image:
        return self.color_analysis(image)

    @darsia.timing_decorator
    def call_pH_analysis(self, color_interpretation: darsia.Image) -> darsia.Image:
        """Run the mass analysis on a single image."""
        return self.signal_model(color_interpretation)

    @darsia.timing_decorator
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

    @darsia.timing_decorator
    def __call__(self, image: darsia.Image) -> SimpleMassAnalysisResults:
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
            image (darsia.Image): The image from which to define the local color path."""

        # ! ---- SETUP ----

        # Initialize analysis tool
        analysis = SimpleRunAnalysis(self.geometry)

        # Allocate identifiers
        image_idx = 0
        label_idx = 0

        # Allocate variables for mass analysis
        times = []
        expected_mass = []
        integrated_mass = []
        integrated_mass_g = []
        integrated_mass_aq = []
        square_error = []

        # ! ---- IMAGES -----

        # Set up parameters for coarse visualization
        coarse_rows = max(200, self.labels.img.shape[0] // 4)
        coarse_cols = int(
            self.labels.img.shape[1] / self.labels.img.shape[0] * coarse_rows
        )
        coarse_shape = tuple[int]((coarse_rows, coarse_cols))

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

        # ! ----- IMAGE SELECTION -----

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

        # ! ---- LABEL SELECTION ----

        def label_idx_selector(density: darsia.Image) -> int:
            nonlocal label_idx
            assistant = darsia.RectangleSelectionAssistant(
                density, labels=self.labels, cmap=cmap
            )
            label_box: Tuple[slice, slice] = assistant()
            label_idx = np.argmax(np.bincount(self.labels.img[label_box].ravel()))
            return label_idx

        # ! ---- MASS ANALYSIS ----

        # Perform detailed mass analysis
        def detailed_mass_analysis(
            _color_interpretation: darsia.Image,
        ) -> Tuple[darsia.Image, darsia.Image, darsia.Image]:
            pH = self.call_pH_analysis(color_interpretation)
            c_aq, s_g = self.flash(pH)
            mass_analysis_result: SimpleMassAnalysisResults = (
                self.co2_mass_analysis.mass_analysis(
                    c_aq=c_aq,
                    s_g=s_g,
                )
            )
            density = mass_analysis_result.mass
            return density, c_aq, s_g

        # ! ---- MASS ANALYSIS SETUP ----

        @darsia.timing_decorator
        def update_mass_analysis() -> None:
            """Auxiliary function to update multiphase time series analysis."""
            # nonlocal analysis
            nonlocal times
            nonlocal expected_mass
            nonlocal integrated_mass
            nonlocal integrated_mass_g
            nonlocal integrated_mass_aq
            nonlocal square_error

            analysis.reset()
            for img, color_interpretation in zip(images, color_interpretations):
                # Update thermodynamic state
                # state = experiment.pressure_temperature_protocol.get_state(img.time)
                # self.co2_mass_analysis.update_state(
                #    atmospheric_pressure=state.pressure, temperature=state.temperature
                # )

                # Signal analysis
                pH = self.call_pH_analysis(color_interpretation)

                # Mass computation
                mass_analysis_result = self.call_flash_and_mass_analysis(pH)

                # Compute expected mass
                exact_mass = experiment.injection_protocol.injected_mass(img.date)

                # Track result
                analysis.track(mass_analysis_result, exact_mass=exact_mass)

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

        # ! ---- GRID FOR PLOTTING ----

        # Interactive tuning of values for each color path (chosen by user)
        done_calibration = False
        need_to_pick_new_image = True
        need_to_pick_new_label = True
        first_time_plotting = True
        while not done_calibration:
            # General design:
            # 1. User needs to pick a focus image (interactively)
            # 2. User picks a label in the image (interactively)
            # 3. Mass analysis is computed for current settings
            # 4. User tunes values for the color path for this label (interactively)
            # 5. User can control to rerun any of the above steps through buttons.

            # Pick image
            if need_to_pick_new_image:
                image_idx = image_idx_selector()
                need_to_pick_new_image = False
                need_to_pick_new_label = True

            # Fetch the current image and its color interpretation
            coarse_image = coarse_images[image_idx]
            color_interpretation = color_interpretations[image_idx]

            # Identify mass density for chosen image
            density, c_aq, s_g = detailed_mass_analysis(color_interpretation)

            # Pick label interactively
            if need_to_pick_new_label:
                label_idx = label_idx_selector(density)

            # Compute mass analysis
            update_mass_analysis()

            # Coarsen outputs for visualization
            coarse_density = darsia.resize(density, shape=coarse_shape)
            coarse_c_aq = darsia.resize(c_aq, shape=coarse_shape)
            coarse_s_g = darsia.resize(s_g, shape=coarse_shape)

            # Determine contours
            coarse_contour = darsia.plot_contour_on_image(
                coarse_image,
                mask=[
                    coarse_c_aq > 0.05,
                    coarse_s_g > 0.05,
                ],  # TODO make thresholds adjustable
                color=[(255, 0, 0), (0, 255, 0)],  # TODO make colors adjustable
                return_image=True,
            )

            # Use gridspec to create a 2x2 grid with proper ratios
            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(
                nrows=2,
                ncols=2,
                width_ratios=[1, 1],  # Bottom: sliders(1) : mass plot(2)
                height_ratios=[1, 1],  # Top row larger(2) : bottom row smaller(1)
                left=0.08,
                right=0.95,
                bottom=0.08,
                top=0.92,
                wspace=0.25,
                hspace=0.25,
            )

            # Create the 2x2 grid of subplots
            # Top left: Density/Image visualization
            ax_image = plt.subplot(gs[0, 0])
            ax_image.axis("off")
            ax_image.set_title("Current Density")

            # Top right: Contour/Secondary visualization
            ax_contour = plt.subplot(gs[0, 1])
            ax_contour.axis("off")
            ax_contour.set_title("Current Image & Phase segmentation")

            # Bottom left: Mass evolution plots
            ax_mass = plt.subplot(gs[1, 0])
            ax_mass.set_title("Mass Evolution")

            # Bottom right: Parameter sliders (will be subdivided)
            ax_sliders = plt.subplot(gs[1, 1])
            ax_sliders.axis("off")  # Turn off axis for slider area
            ax_sliders.set_title("Parameter Controls")

            # ! ---- PLOTTING ----

            # if first_time_plotting:
            #    # Turn off setup for next call
            #    first_time_plotting = False

            # Plot the coarse density in ax_image
            coarse_density_img = ax_image.imshow(coarse_density.img, cmap=cmap)
            coarse_contour_img = ax_contour.imshow(coarse_contour.img)

            # Combine plot and scatter for integrated mass over time.
            # Decompose into total, gas, and aqueous mass, and add
            # expected mass using a dashed line.
            ax_mass.set_xlabel("Time (h)")
            ax_mass.set_ylabel("Mass (g)")
            [integrated_mass_plot] = ax_mass.plot(
                times,
                integrated_mass,
                color="blue",
                label="total",
            )
            [integrated_mass_plot_g] = ax_mass.plot(
                times,
                integrated_mass_g,
                color="green",
                label="gas",
            )
            [integrated_mass_plot_aq] = ax_mass.plot(
                times,
                integrated_mass_aq,
                color="orange",
                label="aqueous",
            )
            ax_mass.plot(
                times,
                expected_mass,
                linestyle="--",
                color="red",
                label="injected",
            )
            integrated_mass_scatter = ax_mass.scatter(
                times,
                integrated_mass,
                color="blue",
            )
            integrated_mass_scatter_g = ax_mass.scatter(
                times,
                integrated_mass_g,
                color="green",
            )
            integrated_mass_scatter_aq = ax_mass.scatter(
                times,
                integrated_mass_aq,
                color="orange",
            )
            ax_mass.set_ylim(0.0, 0.01)
            ax_mass.legend()
            ax_mass.grid(True, alpha=0.3)

            # Get slider area position in figure coordinates
            slider_bbox = ax_sliders.get_position()
            slider_left = slider_bbox.x0
            slider_bottom = slider_bbox.y0
            slider_width = slider_bbox.width
            slider_height_total = slider_bbox.height

            # Calculate total number of sliders needed
            num_value_sliders = len(self.signal_model.model[1][label_idx].values)
            num_flash_sliders = 2
            total_sliders = num_value_sliders + num_flash_sliders

            # Calculate dimensions for vertical sliders
            # Leave some spacing between sliders
            slider_width_individual = slider_width / total_sliders * 0.8
            # Leave space for labels at bottom
            slider_height_individual = slider_height_total * 0.6
            spacing_width = slider_width / total_sliders * 0.2
            spacing = spacing_width / (total_sliders - 1) if total_sliders > 1 else 0

            sliders_color_to_signal = []

            # Create vertical sliders for signal model values
            for i, val in enumerate(self.signal_model.model[1][label_idx].values):
                slider_x = slider_left + i * (slider_width_individual + spacing)
                # Leave space at bottom for labels
                slider_y = slider_bottom + slider_height_total * 0.2

                ax_slider = fig.add_axes(
                    [
                        slider_x,
                        slider_y,
                        slider_width_individual,
                        slider_height_individual,
                    ]
                )

                slider = Slider(
                    ax_slider,
                    f"Val {i}",
                    -0.5,
                    1.5,
                    valinit=val,
                    valstep=0.05,
                    orientation="vertical",
                )
                sliders_color_to_signal.append(slider)

            # Add two vertical sliders for thresholding flash model
            flash_slider_start_idx = num_value_sliders

            # Flash cut-off slider
            slider_x = slider_left + flash_slider_start_idx * (
                slider_width_individual + spacing
            )
            slider_y = slider_bottom + slider_height_total * 0.2

            ax_slider_phase = fig.add_axes(
                [
                    slider_x,
                    slider_y,
                    slider_width_individual,
                    slider_height_individual,
                ]
            )
            slider_phase = Slider(
                ax_slider_phase,
                "Flash\ncut-off",
                0.0,
                1.5,
                valinit=self.flash.cut_off,
                valstep=0.05,
                orientation="vertical",
            )

            # Flash max slider
            slider_x = slider_left + (flash_slider_start_idx + 1) * (
                slider_width_individual + spacing
            )

            ax_slider_max = fig.add_axes(
                [
                    slider_x,
                    slider_y,
                    slider_width_individual,
                    slider_height_individual,
                ]
            )
            slider_max = Slider(
                ax_slider_max,
                "Flash\nmax",
                0.0,
                1.5,
                valinit=self.flash.max_value,
                valstep=0.05,
                orientation="vertical",
            )
            sliders_flash = [slider_phase, slider_max]

            # Sliders for thresholding flash model for c_aq and s_g, start with value 0.05

            slider_x = slider_left + (flash_slider_start_idx + 2) * (
                slider_width_individual + spacing
            )
            ax_slider_c_aq = fig.add_axes(
                [
                    slider_x,
                    slider_y,
                    slider_width_individual,
                    slider_height_individual,
                ]
            )
            slider_c_aq = Slider(
                ax_slider_c_aq,
                "threshold\nc_aq",
                0.0,
                1.0,
                valinit=0.05,
                valstep=0.01,
                orientation="vertical",
            )
            slider_x = slider_left + (flash_slider_start_idx + 3) * (
                slider_width_individual + spacing
            )
            ax_slider_s_g = fig.add_axes(
                [
                    slider_x,
                    slider_y,
                    slider_width_individual,
                    slider_height_individual,
                ]
            )
            slider_s_g = Slider(
                ax_slider_s_g,
                "threshold\ns_g",
                0.0,
                1.0,
                valinit=0.05,
                valstep=0.01,
                orientation="vertical",
            )
            sliders_flash.extend([slider_c_aq, slider_s_g])

            # Position buttons at the top of the figure
            button_width = 0.08
            button_height = 0.04
            button_y = 0.95
            button_spacing = 0.01

            ax_update = plt.axes(
                [0.85 - button_width, button_y, button_width, button_height]
            )
            btn_update = Button(ax_update, "Update")

            ax_next_label = plt.axes(
                [
                    0.85 - 2 * button_width - button_spacing,
                    button_y,
                    button_width,
                    button_height,
                ]
            )
            btn_next_label = Button(ax_next_label, "New Label")

            ax_next_image = plt.axes(
                [
                    0.85 - 3 * button_width - 2 * button_spacing,
                    button_y,
                    button_width,
                    button_height,
                ]
            )
            btn_next_image = Button(ax_next_image, "New Image")

            ax_finish = plt.axes(
                [
                    0.85 - 4 * button_width - 3 * button_spacing,
                    button_y,
                    button_width,
                    button_height,
                ]
            )
            btn_finish = Button(ax_finish, "Finish")

            # Tune values for this label
            done_tuning_values = False
            while not done_tuning_values:

                def show_tuner(_label_idx):
                    nonlocal done_tuning_values, done_calibration

                    def update_analysis(event=None):
                        nonlocal \
                            done_tuning_values, \
                            sliders_color_to_signal, \
                            sliders_flash, \
                            _label_idx

                        ## Update parameters
                        # self.signal_model.model[1][_label_idx].update_model_parameters(
                        #    [slider.val for slider in sliders_color_to_signal]
                        # )
                        self.flash.update(
                            cut_off=sliders_flash[0].val,
                            max_value=sliders_flash[1].val,
                        )
                        # Identify mass density for chosen image
                        density, c_aq, s_g = detailed_mass_analysis(
                            color_interpretation
                        )

                        # Coarsen outputs for visualization
                        coarse_density = darsia.resize(density, shape=coarse_shape)
                        coarse_c_aq = darsia.resize(c_aq, shape=coarse_shape)
                        coarse_s_g = darsia.resize(s_g, shape=coarse_shape)

                        # Determine contours
                        coarse_contour = darsia.plot_contour_on_image(
                            coarse_image,
                            mask=[
                                coarse_c_aq > slider_c_aq.val,
                                coarse_s_g > slider_s_g.val,
                            ],
                            color=[
                                (255, 0, 0),
                                (0, 255, 0),
                            ],  # TODO make colors adjustable
                            return_image=True,
                        )

                        # Update mass analysis
                        update_mass_analysis()

                        # Update plots
                        coarse_density_img.set_data(coarse_density.img)
                        coarse_contour_img.set_data(coarse_contour.img)
                        integrated_mass_plot.set_data(times, integrated_mass)
                        integrated_mass_plot_g.set_data(times, integrated_mass_g)
                        integrated_mass_plot_aq.set_data(times, integrated_mass_aq)
                        integrated_mass_scatter.set_offsets(
                            np.c_[times, integrated_mass]
                        )
                        integrated_mass_scatter_g.set_offsets(
                            np.c_[times, integrated_mass_g]
                        )
                        integrated_mass_scatter_aq.set_offsets(
                            np.c_[times, integrated_mass_aq]
                        )
                        fig.canvas.draw_idle()

                        # done_tuning_values = True

                    btn_update.on_clicked(update_analysis)

                    def next_label(event):
                        nonlocal done_tuning_values, need_to_pick_new_label
                        done_tuning_values = True
                        need_to_pick_new_label = True
                        plt.close("all")

                    btn_next_label.on_clicked(next_label)

                    def next_image(event):
                        nonlocal done_tuning_values, need_to_pick_new_image
                        done_tuning_values = True
                        need_to_pick_new_image = True
                        plt.close("all")

                    btn_next_image.on_clicked(next_image)

                    def finish(event):
                        nonlocal done_tuning_values, done_calibration
                        done_tuning_values = True
                        done_calibration = True
                        plt.close("all")

                    btn_finish.on_clicked(finish)

                    plt.show()

                show_tuner(label_idx)

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
