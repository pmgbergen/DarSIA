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

        self.color_path_interpretation = color_path_interpretation
        """Color path interpretations for different labels."""

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

        # Choices for signal illustration
        signal_options = {
            "pH": "pH",
            "density": "CO2 Density",
            "c_aq": "CO2(aq) concentration",
            "s_g": "CO2(g) saturation",
        }
        signal_option_idx = 0
        signal_option_ptr = list(signal_options.keys())

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

        def label_idx_selector(pH: darsia.Image) -> int:
            nonlocal label_idx
            assistant = darsia.RectangleSelectionAssistant(
                pH, labels=self.labels, cmap=cmap
            )
            label_box: Tuple[slice, slice] = assistant()
            label_idx = np.argmax(np.bincount(self.labels.img[label_box].ravel()))
            return label_idx

        # ! ---- MASS ANALYSIS ----

        # Perform detailed mass analysis
        def detailed_mass_analysis(
            _color_interpretation: darsia.Image,
        ) -> Tuple[darsia.Image, darsia.Image, darsia.Image, darsia.Image]:
            pH = self.call_pH_analysis(color_interpretation)
            c_aq, s_g = self.flash(pH)
            mass_analysis_result: SimpleMassAnalysisResults = (
                self.co2_mass_analysis.mass_analysis(
                    c_aq=c_aq,
                    s_g=s_g,
                )
            )
            density = mass_analysis_result.mass
            return pH, density, c_aq, s_g

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
            pH, density, c_aq, s_g = detailed_mass_analysis(color_interpretation)

            # Pick label interactively
            if need_to_pick_new_label:
                label_idx = label_idx_selector(pH)

            # Compute mass analysis
            update_mass_analysis()

            # Coarsen outputs for visualization
            coarse_pH = darsia.resize(pH, shape=coarse_shape)
            coarse_density = darsia.resize(density, shape=coarse_shape)
            coarse_c_aq = darsia.resize(c_aq, shape=coarse_shape)
            coarse_s_g = darsia.resize(s_g, shape=coarse_shape)
            coarse_signal_dict = {
                "pH": coarse_pH,
                "density": coarse_density,
                "c_aq": coarse_c_aq,
                "s_g": coarse_s_g,
            }
            coarse_signal = coarse_signal_dict[signal_option_ptr[signal_option_idx]]

            # Determine contours
            coarse_contour = darsia.plot_contour_on_image(
                coarse_image,
                mask=[
                    coarse_c_aq > 0.05,
                    coarse_s_g > 0.05,
                ],  # TODO make thresholds adjustable
                color=[(255, 0, 0), (0, 255, 0)],  # TODO make colors adjustable
                return_image=True,
            )  # Use gridspec to create a 2x3 grid with proper ratios
            fig = plt.figure(figsize=(20, 12))
            gs = gridspec.GridSpec(
                nrows=2,
                ncols=3,
                width_ratios=[1, 1, 1],  # Three equal columns
                height_ratios=[1, 1],  # Two equal rows
                left=0.05,
                right=0.95,
                bottom=0.08,
                top=0.92,
                wspace=0.2,
                hspace=0.25,
            )

            # Create the 2x3 grid of subplots
            # Top left: Coarse image
            ax_coarse_image = plt.subplot(gs[0, 0])
            ax_coarse_image.axis("off")
            ax_coarse_image.set_title(
                f"Current Image {image_idx} and Label {label_idx}"
            )

            # Top middle: Density visualization
            ax_signal = plt.subplot(gs[0, 1])
            ax_signal.axis("off")
            ax_signal.set_title(signal_options[signal_option_ptr[signal_option_idx]])

            # Top right: Contour/Secondary visualization
            ax_contour = plt.subplot(gs[0, 2])
            ax_contour.axis("off")
            ax_contour.set_title("Phase Segmentation")

            # Bottom left: Signal function plot
            ax_signal_function = plt.subplot(gs[1, 0])
            ax_signal_function.set_title(f"Signal Function - Label {label_idx}")

            # Bottom middle: Mass evolution plots
            ax_mass = plt.subplot(gs[1, 1])
            ax_mass.set_title("Mass Evolution")

            # Bottom right: Parameter sliders (will be subdivided)
            ax_sliders = plt.subplot(gs[1, 2])
            ax_sliders.axis("off")  # Turn off axis for slider area
            ax_sliders.set_title("Parameter Controls")

            # ! ---- PLOTTING ----

            # Plot the coarse image with highlighted label - turn the rest to gray
            labeled_coarse_image = coarse_image.img.copy()
            labeled_coarse_image[coarse_labels.img != label_idx] = (
                0.7
                * np.mean(
                    labeled_coarse_image[coarse_labels.img != label_idx],
                    axis=-1,
                    keepdims=True,
                )
                + 0.3 * coarse_image.img[coarse_labels.img != label_idx]
            )
            coarse_image_img = ax_coarse_image.imshow(labeled_coarse_image)

            # Plot the coarse signal
            coarse_signal_img = ax_signal.imshow(
                coarse_signal.img, cmap=cmap
            )  # Add colorbar for the signal
            coarse_signal_colorbar = fig.colorbar(
                coarse_signal_img, ax=ax_signal, shrink=1.0
            )

            # Set initial color scale for the signal
            initial_signal_min = np.nanmin(coarse_signal.img)
            initial_signal_max = np.nanmax(coarse_signal.img)
            coarse_signal_img.set_clim(vmin=initial_signal_min, vmax=initial_signal_max)
            coarse_signal_colorbar.update_normal(coarse_signal_img)

            # Plot the contour
            coarse_contour_img = ax_contour.imshow(coarse_contour.img)

            # Plot the signal function for the selected label
            signal_func: darsia.PWTransformation = self.signal_model.model[1][label_idx]
            signal_function_line = None
            if hasattr(signal_func, "values") and len(signal_func.values) > 0:
                # Get the color path interpretation for gradient background
                _color_path_interpretation: darsia.ColorPathInterpolation = (
                    self.color_path_interpretation[label_idx]
                )

                # Create background gradient using the color path colormap
                x_gradient = np.linspace(0, 1, 256).reshape(1, -1)
                y_gradient = np.ones((10, 1))  # Height for visualization
                gradient_data = x_gradient * y_gradient

                # Get the colormap from the color path
                color_path_cmap = _color_path_interpretation.color_path.get_color_map()

                # Display the gradient as background
                ax_signal_function.imshow(
                    gradient_data,
                    extent=[
                        0,
                        1,
                        0,
                        1,
                    ],  # TODO update extent if extending the pw transforms
                    aspect="auto",
                    cmap=color_path_cmap,
                    alpha=0.3,  # Semi-transparent
                    zorder=0,  # Behind everything else
                )

                # Create x values for plotting the piecewise linear function
                x_vals = signal_func.supports
                y_vals = signal_func.values.copy()
                [signal_function_line] = ax_signal_function.plot(
                    x_vals, y_vals, "gray", linewidth=2, zorder=2
                )

                colors = _color_path_interpretation.color_path.colors
                signal_function_scatter = ax_signal_function.scatter(
                    x_vals,
                    y_vals,
                    c=np.clip(colors, 0, 1),
                    zorder=3,
                    s=50,
                    edgecolors="black",
                    linewidth=1,
                )

                # Indicate cut-off and max value from flash model
                signal_function_cut_off_y = ax_signal_function.axhline(
                    y=self.flash.cut_off,
                    color="k",
                    linestyle="--",
                    label="cut-off",
                    zorder=1,
                )
                signal_function_max_value_y = ax_signal_function.axhline(
                    y=self.flash.max_value,
                    color="k",
                    linestyle="--",
                    label="max value",
                    zorder=1,
                )

                # Find the corresponding x-values for the cut-off and max value lines as solution to the signal_function(x) = cut_off and signal_function(x) = max_value
                cut_off_x = signal_func.inverse(self.flash.cut_off)
                max_value_x = signal_func.inverse(self.flash.max_value)

                # Add vertical lines for cut-off and max value
                signal_function_cut_off_x = ax_signal_function.axvline(
                    x=cut_off_x, color="k", linestyle="--", label="cut-off", zorder=1
                )
                signal_function_max_value_x = ax_signal_function.axvline(
                    x=max_value_x,
                    color="k",
                    linestyle="--",
                    label="max value",
                    zorder=1,
                )

                # Make text annotations "CO2(aq)" and "CO2(g)" below and above the cut-off line
                ax_signal_function.text(
                    x=cut_off_x - 0.05,
                    y=self.flash.cut_off - 0.05,
                    s="CO2(aq)",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="k",
                )
                ax_signal_function.text(
                    x=cut_off_x + 0.05,
                    y=self.flash.cut_off + 0.05,
                    s="CO2(g)",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="k",
                )

                ax_signal_function.set_xlabel("Color Interpretation")
                ax_signal_function.set_ylabel("Signal Value")
                ax_signal_function.grid(True, alpha=0.3, zorder=1)
                ax_signal_function.set_xlim(0, 1)
                ax_signal_function.set_ylim(0, 1)

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
            num_threshold_sliders = 2
            total_sliders = (
                num_value_sliders + num_flash_sliders + num_threshold_sliders
            )

            # Calculate dimensions for vertical sliders
            # Leave some spacing between sliders - ensure they all fit in the box
            available_width = slider_width * 0.95  # Use 95% of available width
            slider_width_individual = available_width / total_sliders * 0.85
            # Leave space for labels at bottom
            slider_height_individual = slider_height_total * 0.6
            spacing = available_width / total_sliders * 0.15 if total_sliders > 1 else 0

            sliders_color_to_signal = []  # Create vertical sliders for signal model values
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
                    0.0,
                    2.5,
                    valinit=val,
                    valstep=0.05,
                    orientation="vertical",
                )
                sliders_color_to_signal.append(slider)

            # Add two vertical sliders for thresholding flash model
            flash_slider_start_idx = num_value_sliders  # Flash cut-off slider
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
            sliders_threshold = [slider_c_aq, slider_s_g]

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

            ax_next_signal = plt.axes(
                [
                    0.85 - 5 * button_width - 4 * button_spacing,
                    button_y,
                    button_width,
                    button_height,
                ]
            )
            btn_next_signal = Button(ax_next_signal, "Next Signal")

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
                            sliders_threshold, \
                            _label_idx

                        # Check if any slider value has changed that requires updating mass analysis
                        need_update = False
                        need_update = need_update or any(
                            [
                                sliders_color_to_signal[i].val
                                != self.signal_model.model[1][_label_idx].values[i]
                                for i in range(
                                    len(self.signal_model.model[1][_label_idx].values)
                                )
                            ]
                        )
                        need_update = need_update or any(
                            [
                                sliders_flash[0].val != self.flash.cut_off,
                                sliders_flash[1].val != self.flash.max_value,
                            ]
                        )

                        ## Update parameters
                        self.signal_model.model[1][_label_idx].update(
                            values=[slider.val for slider in sliders_color_to_signal]
                        )
                        self.flash.update(
                            cut_off=sliders_flash[0].val,
                            max_value=sliders_flash[1].val,
                        )
                        # Identify mass density for chosen image
                        pH, density, c_aq, s_g = detailed_mass_analysis(
                            color_interpretation
                        )

                        # Coarsen outputs for visualization
                        coarse_pH = darsia.resize(pH, shape=coarse_shape)
                        coarse_density = darsia.resize(density, shape=coarse_shape)
                        coarse_c_aq = darsia.resize(c_aq, shape=coarse_shape)
                        coarse_s_g = darsia.resize(s_g, shape=coarse_shape)
                        coarse_signal_dict = {
                            "pH": coarse_pH,
                            "density": coarse_density,
                            "c_aq": coarse_c_aq,
                            "s_g": coarse_s_g,
                        }
                        coarse_signal = coarse_signal_dict[
                            signal_option_ptr[signal_option_idx]
                        ]

                        # Determine contours
                        coarse_contour = darsia.plot_contour_on_image(
                            coarse_image,
                            mask=[
                                coarse_c_aq > sliders_threshold[0].val,
                                coarse_s_g > sliders_threshold[1].val,
                            ],
                            color=[
                                (255, 0, 0),
                                (0, 255, 0),
                            ],  # TODO make colors adjustable
                            return_image=True,
                        )

                        # Update mass analysis
                        if need_update:
                            update_mass_analysis()

                        # Update signal function plot
                        if signal_function_line is not None:
                            signal_func = self.signal_model.model[1][_label_idx]
                            x_vals = signal_func.supports
                            y_vals = signal_func.values.copy()
                            signal_function_line.set_data(x_vals, y_vals)
                            signal_function_scatter.set_offsets(np.c_[x_vals, y_vals])

                        # Update flash cut-off and max value lines
                        signal_function_cut_off_y.set_ydata(self.flash.cut_off)
                        signal_function_max_value_y.set_ydata(self.flash.max_value)
                        cut_off_x = signal_func.inverse(self.flash.cut_off)
                        max_value_x = signal_func.inverse(self.flash.max_value)
                        signal_function_cut_off_x.set_xdata(cut_off_x)
                        signal_function_max_value_x.set_xdata(max_value_x)

                        # Update the position of the annotaion
                        ax_signal_function.texts[0].set_position(
                            (cut_off_x - 0.05, self.flash.cut_off - 0.05)
                        )
                        ax_signal_function.texts[1].set_position(
                            (cut_off_x + 0.05, self.flash.cut_off + 0.05)
                        )

                        # Update dense plots
                        coarse_signal_img.set_data(coarse_signal.img)
                        coarse_contour_img.set_data(coarse_contour.img)

                        # Update color scale (min/max values) based on current signal
                        signal_min = np.nanmin(coarse_signal.img)
                        signal_max = np.nanmax(coarse_signal.img)
                        coarse_signal_img.set_clim(vmin=signal_min, vmax=signal_max)
                        coarse_signal_colorbar.update_normal(coarse_signal_img)

                        # Update mass plots
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

                        # Redraw
                        fig.canvas.draw_idle()

                        # TODO remove if everything works
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

                    def next_signal(event):
                        nonlocal signal_option_idx
                        signal_option_idx = (signal_option_idx + 1) % len(
                            signal_option_ptr
                        )

                        # Identify mass density for chosen image
                        pH, density, c_aq, s_g = detailed_mass_analysis(
                            color_interpretation
                        )

                        # Coarsen outputs for visualization
                        coarse_pH = darsia.resize(pH, shape=coarse_shape)
                        coarse_density = darsia.resize(density, shape=coarse_shape)
                        coarse_c_aq = darsia.resize(c_aq, shape=coarse_shape)
                        coarse_s_g = darsia.resize(s_g, shape=coarse_shape)
                        coarse_signal_dict = {
                            "pH": coarse_pH,
                            "density": coarse_density,
                            "c_aq": coarse_c_aq,
                            "s_g": coarse_s_g,
                        }
                        coarse_signal = coarse_signal_dict[
                            signal_option_ptr[signal_option_idx]
                        ]

                        # Update the image and title
                        coarse_signal_img.set_data(coarse_signal.img)
                        ax_signal.set_title(
                            signal_options[signal_option_ptr[signal_option_idx]]
                        )  # Update color scale (min/max values) based on current signal
                        signal_min = np.nanmin(coarse_signal.img)
                        signal_max = np.nanmax(coarse_signal.img)
                        coarse_signal_img.set_clim(vmin=signal_min, vmax=signal_max)
                        coarse_signal_colorbar.update_normal(coarse_signal_img)

                        # Redraw
                        fig.canvas.draw_idle()

                    btn_next_signal.on_clicked(next_signal)

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
