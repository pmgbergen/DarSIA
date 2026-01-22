import logging
from pathlib import Path
from typing import Tuple
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy.optimize import minimize
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
        geometry: darsia.ExtrudedPorousGeometry,
        restoration: darsia.Model | None = None,
        ignore_labels: list[int] | None = None,
    ):
        base_model = darsia.CombinedModel(
            [
                darsia.HeterogeneousModel(
                    color_path_interpretation,
                    labels,
                    ignore_labels=ignore_labels,
                )
            ]
        )

        # Define general config options
        config = {
            "diff option": "plain",
            "restoration -> model": False,
        }

        # TODO make use of ignore_labels

        # Define general ConcentrationAnalysis.
        self.color_analysis = darsia.ConcentrationAnalysis(
            base=baseline if color_mode == darsia.ColorMode.RELATIVE else None,
            restoration=restoration,
            model=base_model,
            labels=labels,
            **config,
        )

        # To make sure the signal functions can be evaluated (within their domain of support)
        # explicitly clip the arguments of the signal model
        min_domain_signal_functions = max(
            min(func.supports) for func in signal_functions.values()
        )
        max_domain_signal_functions = min(
            max(func.supports) for func in signal_functions.values()
        )
        min_range_signal_functions = min(
            min(func.values) for func in signal_functions.values()
        )
        max_range_signal_functions = max(
            max(func.values) for func in signal_functions.values()
        )
        self.signal_model_extents = (
            (min_domain_signal_functions, max_domain_signal_functions),
            (min_range_signal_functions, max_range_signal_functions),
        )

        signal_model = darsia.CombinedModel(
            [
                darsia.ClipModel(
                    min_domain_signal_functions, max_domain_signal_functions
                ),
                darsia.HeterogeneousModel(
                    signal_functions,
                    labels,
                    ignore_labels=ignore_labels,
                ),
            ],
        )

        self.signal_model = darsia.ConcentrationAnalysis(
            model=signal_model,
            labels=labels,
            **config,
        )
        """Model converting color interpretation to pH."""

        self.flash = flash
        """Flash model for converting pH to concentrations."""

        self.co2_mass_analysis = co2_mass_analysis
        """Mass computation tool."""

        self.geometry = geometry
        """Geometry of the experiment - for the integration of mass during calibration."""

        self.original_depth = geometry.depth.copy()
        """Original depth image."""

        self.analysis = SimpleRunAnalysis(self.geometry)
        """Analysis tool for tracking mass evolution."""

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
        rois: dict[str, darsia.VoxelArray | darsia.CoordinateArray] | None = None,
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

        # Fill up rois with "entire frame"
        rois = rois or {}
        assert "" not in rois
        rois.update(
            {
                "full": darsia.CoordinateArray(
                    [
                        self.color_analysis.base.origin,
                        self.color_analysis.base.opposite_corner,
                    ]
                )
            }
        )

        # ! ---- SETUP ----

        # Allocate identifiers
        image_idx = 0
        label_idx = 0

        # Allocate variables for mass analysis
        times = {r: [] for r in rois}
        expected_mass = {r: [] for r in rois}
        integrated_mass = {r: [] for r in rois}
        integrated_mass_g = {r: [] for r in rois}
        integrated_mass_aq = {r: [] for r in rois}

        # Choices for signal illustration
        signal_options = {
            "pH": "pH",
            "color": "Color Interpretation",
            "rho_co2": "CO2 Density",
            "c_aq": "CO2(aq) concentration",
            "s_g": "CO2(g) saturation",
            "labels": "Labels",
            "depth": "Depth",
            "rho_co2(g)": "CO2(g) Density",
            "solubility_co2(aq)": "CO2(aq) Solubility",
            "porosity": "Porosity",
            "effective_depth": "Effective Depth",
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

        coarse_depth = darsia.resize(
            self.geometry.depth,
            shape=coarse_shape,
        )

        coarse_porosity = darsia.resize(
            self.geometry.porosity,
            shape=coarse_shape,
        )

        coarse_effective_depth = darsia.resize(
            self.geometry.weight,
            shape=coarse_shape,
        )

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
            _color_interpretation: darsia.Image, roi: darsia.Image | None = None
        ) -> Tuple[darsia.Image, darsia.Image, darsia.Image, darsia.Image]:
            # TODO:color_interpretation: darsia.Image restrivuted to the separate boxes
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

            analysis = {
                roi_key: SimpleRunAnalysis(self.geometry.subregion(roi))
                for roi_key, roi in rois.items()
            }
            for img, color_interpretation in zip(images, color_interpretations):
                # Update thermodynamic state
                state = experiment.pressure_temperature_protocol.get_state(img.date)
                gradient = experiment.pressure_temperature_protocol.get_gradient(
                    img.date
                )
                self.co2_mass_analysis.update_state(
                    atmospheric_pressure=state.pressure,
                    atmospheric_temperature=state.temperature,
                    atmospheric_pressure_gradient=gradient.pressure,
                    atmospheric_temperature_gradient=gradient.temperature,
                )

                # Signal analysis
                pH = self.call_pH_analysis(color_interpretation)

                # Mass computation
                mass_analysis_result = self.call_flash_and_mass_analysis(pH)

                for roi_key, roi in rois.items():
                    # Subregion mass
                    roi_mass_analysis_result = mass_analysis_result.subregion(roi)

                    # Compute expected mass
                    assert isinstance(roi, darsia.CoordinateArray)
                    roi_exact_mass = experiment.injection_protocol.injected_mass(
                        date=img.date, roi=roi
                    )

                    # Track result
                    analysis[roi_key].track(
                        roi_mass_analysis_result, exact_mass=roi_exact_mass
                    )

                    # # Clean data - TODO?
                    # analysis.clean(threshold=5.0)

            # Monitor mass evolution over time
            for roi_key in rois:
                times[roi_key] = [t / 3600 for t in analysis[roi_key].data.time]
                expected_mass[roi_key] = analysis[roi_key].data.exact_mass_tot
                integrated_mass[roi_key] = analysis[roi_key].data.mass_tot
                integrated_mass_g[roi_key] = analysis[roi_key].data.mass_g
                integrated_mass_aq[roi_key] = analysis[roi_key].data.mass_aq

                # Errors
                relative_error = (
                    np.array(integrated_mass[roi_key])
                    - np.array(expected_mass[roi_key])
                ) / np.abs(expected_mass[roi_key]).clip(min=1e-8)

                logging.info(
                    f"Mass analysis for roi {roi_key} updated.\nNew relative error: {relative_error}"
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
            coarse_color_interpretation = darsia.resize(
                color_interpretation, shape=coarse_shape
            )

            # Identify mass density for chosen image
            pH, density, c_aq, s_g = detailed_mass_analysis(color_interpretation)

            # Pick label interactively
            if need_to_pick_new_label:
                label_idx = label_idx_selector(pH)

            # Compute mass analysis
            update_mass_analysis()

            # Fetch thermodynamic infos for current image
            state = experiment.pressure_temperature_protocol.get_state(
                coarse_image.date
            )
            gradient = experiment.pressure_temperature_protocol.get_gradient(
                coarse_image.date
            )
            self.co2_mass_analysis.update_state(
                atmospheric_pressure=state.pressure,
                atmospheric_temperature=state.temperature,
                atmospheric_pressure_gradient=gradient.pressure,
                atmospheric_temperature_gradient=gradient.temperature,
            )
            co2_g_density = darsia.full_like(
                pH, self.co2_mass_analysis.density_gaseous_co2
            )
            co2_aq_solubility = darsia.full_like(
                pH, self.co2_mass_analysis.solubility_co2
            )

            # Coarsen outputs for visualization
            coarse_pH = darsia.resize(pH, shape=coarse_shape)
            coarse_density = darsia.resize(density, shape=coarse_shape)
            coarse_c_aq = darsia.resize(c_aq, shape=coarse_shape)
            coarse_s_g = darsia.resize(s_g, shape=coarse_shape)
            coarse_co2_g_density = darsia.resize(co2_g_density, shape=coarse_shape)
            coarse_co2_aq_solubility = darsia.resize(
                co2_aq_solubility, shape=coarse_shape
            )
            coarse_signal_dict = {
                "color": coarse_color_interpretation,
                "pH": coarse_pH,
                "rho_co2": coarse_density,
                "c_aq": coarse_c_aq,
                "s_g": coarse_s_g,
                "labels": coarse_labels,
                "depth": coarse_depth,
                "rho_co2(g)": coarse_co2_g_density,
                "solubility_co2(aq)": coarse_co2_aq_solubility,
                "porosity": coarse_porosity,
                "effective_depth": coarse_effective_depth,
            }
            coarse_signal = coarse_signal_dict[signal_option_ptr[signal_option_idx]]

            # Determine contours
            coarse_contour = darsia.plot_contour_on_image(
                coarse_image,
                mask=[
                    coarse_c_aq > 0.05,
                    coarse_c_aq > 0.1,
                    coarse_c_aq > 0.25,
                    coarse_c_aq > 0.50,
                    coarse_c_aq > 0.9,
                    coarse_s_g > 0.05,
                ],
                color=[
                    (255, 0, 0),
                    (255, 0, 0),
                    (255, 0, 0),
                    (255, 0, 0),
                    (255, 0, 0),
                    (0, 255, 0),
                ],
                alpha=[0.2, 0.3, 0.4, 0.5, 1.0, 1.0],
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

            # Plot the coarse signal with rois
            # TODO
            coarse_signal_img = ax_signal.imshow(coarse_signal.img, cmap=cmap)

            # Add colorbar
            signal_pos = ax_signal.get_position()
            colorbar_width = 0.01
            colorbar_height = signal_pos.height
            colorbar_x = signal_pos.x1 + 0.01
            colorbar_y = signal_pos.y0
            colorbar_ax = fig.add_axes(
                [colorbar_x, colorbar_y, colorbar_width, colorbar_height]
            )
            coarse_signal_colorbar = fig.colorbar(coarse_signal_img, cax=colorbar_ax)

            # Set initial color scale for the signal
            if signal_option_ptr[signal_option_idx] == "pH":
                initial_signal_min = 0.0
                initial_signal_max = 1.0
            else:
                initial_signal_min = np.nanmin(coarse_signal.img)
                initial_signal_max = np.nanmax(coarse_signal.img)
            coarse_signal_img.set_clim(vmin=initial_signal_min, vmax=initial_signal_max)
            coarse_signal_colorbar.update_normal(coarse_signal_img)  # Plot the contour
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
                x_gradient = np.linspace(
                    self.signal_model_extents[0][0],
                    self.signal_model_extents[0][1],
                    256,
                ).reshape(1, -1)
                y_gradient = np.ones((10, 1))  # Height for visualization
                gradient_data = x_gradient * y_gradient

                # Get the colormap from the color path
                color_path_cmap = _color_path_interpretation.color_path.get_color_map()

                # Display the gradient as background
                signal_function_background = ax_signal_function.imshow(
                    gradient_data,
                    extent=[
                        self.signal_model_extents[0][0],
                        self.signal_model_extents[0][1],
                        self.signal_model_extents[1][0],
                        self.signal_model_extents[1][1],
                    ],
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
                signal_function_aq_min_y = ax_signal_function.axhline(
                    y=self.flash.min_value_aq,
                    color="k",
                    linestyle="--",
                    label="aq_min",
                    zorder=1,
                )
                signal_function_aq_max_y = ax_signal_function.axhline(
                    y=self.flash.max_value_aq,
                    color="k",
                    linestyle="--",
                    label="aq_max",
                    zorder=1,
                )
                signal_function_g_max_y = ax_signal_function.axhline(
                    y=self.flash.max_value_g,
                    color="k",
                    linestyle="--",
                    label="g_max",
                    zorder=1,
                )

                signal_function_g_min_y = ax_signal_function.axhline(
                    y=self.flash.min_value_g,
                    color="k",
                    linestyle="--",
                    label="g_min",
                    zorder=1,
                )

                # Find the corresponding x-values for the cut-off and max value lines as solution to the signal_function(x) = cut_off and signal_function(x) = max_value
                aq_min_x = signal_func.inverse(self.flash.min_value_aq)
                aq_max_x = signal_func.inverse(self.flash.max_value_aq)
                g_max_x = signal_func.inverse(self.flash.max_value_g)
                g_min_x = signal_func.inverse(self.flash.min_value_g)

                # Add vertical lines for cut-off and max value
                signal_function_aq_min_x = ax_signal_function.axvline(
                    x=aq_min_x,
                    color="k",
                    linestyle="--",
                    label="min value aq",
                    zorder=1,
                )
                signal_function_aq_max_x = ax_signal_function.axvline(
                    x=aq_max_x,
                    color="k",
                    linestyle="--",
                    label="max value aq",
                    zorder=1,
                )
                signal_function_g_max_x = ax_signal_function.axvline(
                    x=g_max_x,
                    color="k",
                    linestyle="--",
                    label="max value g",
                    zorder=1,
                )
                signal_function_g_min_x = ax_signal_function.axvline(
                    x=g_min_x,
                    color="k",
                    linestyle="--",
                    label="min value g",
                    zorder=1,
                )

                # Make text annotations "CO2(aq)" and "CO2(g)" below and above the cut-off line
                signal_function_co2_aq = ax_signal_function.text(
                    x=aq_max_x - 0.05,
                    y=0.04,
                    s="CO2(aq)",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="k",
                )
                signal_function_co2_g = ax_signal_function.text(
                    x=aq_max_x + 0.05,
                    y=0.04,
                    s="CO2(g)",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="k",
                )
                signal_function_co2_g_full = ax_signal_function.text(
                    x=g_max_x + 0.1,
                    y=0.04,
                    s="CO2(g) - 100%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="k",
                )

                ax_signal_function.set_xlabel("Color Interpretation")
                ax_signal_function.set_ylabel("Signal Value")
                ax_signal_function.grid(True, alpha=0.3, zorder=1)
                ax_signal_function.set_xlim(
                    self.signal_model_extents[0][0], self.signal_model_extents[0][1]
                )
                ax_signal_function.set_ylim(
                    self.signal_model_extents[1][0], self.signal_model_extents[1][1]
                )  # Combine plot and scatter for integrated mass over time.
            # Decompose into total, gas, and aqueous mass, and add
            # expected mass using a dashed line.
            ax_mass.set_xlabel("Time (h)")
            ax_mass.set_ylabel("Mass (g)")

            # Store plot objects for updates
            integrated_mass_plots = {}
            integrated_mass_plots_g = {}
            integrated_mass_plots_aq = {}
            expected_mass_plots = {}
            integrated_mass_scatters = {}
            integrated_mass_scatters_g = {}
            integrated_mass_scatters_aq = {}

            # Generate distinct colors for each ROI
            num_rois = len(rois)
            _cmap = plt.cm.get_cmap("tab10" if num_rois <= 10 else "tab20")
            roi_colors = [_cmap(i / max(1, num_rois - 1)) for i in range(num_rois)]

            # Plot each ROI as separate series
            max_expected = 0
            for i, (roi_key, roi_times) in enumerate(times.items()):
                if len(roi_times) > 0:
                    color = roi_colors[i % len(roi_colors)]
                    roi_label = roi_key if roi_key else "entire_frame"

                    # Plot lines
                    [integrated_mass_plots[roi_key]] = ax_mass.plot(
                        roi_times,
                        integrated_mass[roi_key],
                        color=color,
                        label=f"total ({roi_label})",
                        linewidth=2,
                    )
                    [integrated_mass_plots_g[roi_key]] = ax_mass.plot(
                        roi_times,
                        integrated_mass_g[roi_key],
                        color=color,
                        label=f"gas ({roi_label})",
                        linestyle=":",
                        linewidth=2,
                    )
                    [integrated_mass_plots_aq[roi_key]] = ax_mass.plot(
                        roi_times,
                        integrated_mass_aq[roi_key],
                        color=color,
                        label=f"aqueous ({roi_label})",
                        linestyle="-.",
                        linewidth=2,
                    )
                    [expected_mass_plots[roi_key]] = ax_mass.plot(
                        roi_times,
                        expected_mass[roi_key],
                        linestyle="--",
                        color=color,
                        label=f"injected ({roi_label})",
                        alpha=0.7,
                    )

                    # Plot scatter points
                    integrated_mass_scatters[roi_key] = ax_mass.scatter(
                        roi_times,
                        integrated_mass[roi_key],
                        color=color,
                        s=20,
                        alpha=0.8,
                    )
                    integrated_mass_scatters_g[roi_key] = ax_mass.scatter(
                        roi_times,
                        integrated_mass_g[roi_key],
                        color=color,
                        s=15,
                        alpha=0.6,
                        marker="^",
                    )
                    integrated_mass_scatters_aq[roi_key] = ax_mass.scatter(
                        roi_times,
                        integrated_mass_aq[roi_key],
                        color=color,
                        s=15,
                        alpha=0.6,
                        marker="s",
                    )

                    # Track maximum for y-axis scaling
                    if len(expected_mass[roi_key]) > 0:
                        max_expected = max(max_expected, max(expected_mass[roi_key]))

            # Set y-axis limits based on maximum expected mass across all ROIs
            if max_expected > 0:
                ax_mass.set_ylim(0.0, 2.0 * max_expected)
            else:
                ax_mass.set_ylim(0.0, 1.0)
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
            num_flash_sliders = 4
            num_threshold_sliders = 2
            num_depth_sliders = 1  # Depth scaling slider
            total_sliders = (
                num_value_sliders
                + num_flash_sliders
                + num_threshold_sliders
                + num_depth_sliders
            )

            # Calculate dimensions for vertical sliders
            # Leave some spacing between sliders - ensure they all fit in the box
            available_width = slider_width * 0.95  # Use 95% of available width
            slider_width_individual = available_width / total_sliders * 0.85
            # Leave space for labels at bottom
            slider_height_individual = slider_height_total * 0.6
            spacing = available_width / total_sliders * 0.15 if total_sliders > 1 else 0

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
                    0.0,
                    2.0,
                    valinit=val,
                    valstep=0.01,
                    color=np.clip(
                        self.color_path_interpretation[label_idx].color_path.colors[i],
                        0,
                        1,
                    ),
                    orientation="vertical",
                )
                sliders_color_to_signal.append(slider)

            # Add two rows of arrow buttons below the signal sliders
            arrow_button_height = (
                slider_height_total * 0.05
            )  # Small height for arrow buttons
            arrow_button_spacing = (
                slider_height_total * 0.02
            )  # Small spacing between up and down buttons

            # Calculate positions for arrow buttons
            up_button_y = slider_bottom + slider_height_total * 0.1
            down_button_y = up_button_y - arrow_button_height - arrow_button_spacing

            # Create functions for signal slider arrow button callbacks
            def slider_up_arrow(slider_index):
                """Increment signal slider by one step.

                Args:
                    slider_index (int): Index of the signal slider
                """
                # Get the slider object
                slider = sliders_color_to_signal[slider_index]

                # Get current value and slider parameters
                # current_val = slider.val
                # step_size = slider.valstep
                max_val = slider.valmax

                # # Calculate new value with step increment, capped at maximum
                # slider.set_val(new_val)
                # Fetch all values of all color_path_interpolations for all labels
                available_labels = np.sort(list(self.signal_model.model[1].keys()))
                current_values = [
                    self.signal_model.model[1][_label_idx].values[slider_index]
                    for _label_idx in available_labels
                ]

                # Fetch all values of next slider index for all labels
                if slider_index + 1 < len(self.signal_model.model[1][0].values):
                    next_slider_index = slider_index + 1
                    next_values = {
                        _label_idx: self.signal_model.model[1][_label_idx].values[
                            next_slider_index
                        ]
                        for _label_idx in available_labels
                    }
                    ten_percent_increase = {
                        _label_idx: 0.1
                        * (next_values[_label_idx] - current_values[_label_idx])
                        for _label_idx in available_labels
                    }
                    new_values = {
                        _label_idx: min(
                            current_values[_label_idx]
                            + ten_percent_increase[_label_idx],
                            max_val,
                        )
                        for _label_idx in available_labels
                    }

                    # Update the current slider
                    slider.set_val(new_values[label_idx])

                    # Update parameters

                    for _label_idx in available_labels:
                        _old_values = self.signal_model.model[1][_label_idx].values
                        _new_values = _old_values.copy()
                        _new_values[slider_index] = new_values[_label_idx]
                        self.signal_model.model[1][_label_idx].update(
                            values=_new_values
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
                        "color": coarse_color_interpretation,
                        "pH": coarse_pH,
                        "rho_co2": coarse_density,
                        "c_aq": coarse_c_aq,
                        "s_g": coarse_s_g,
                        "labels": coarse_labels,
                        "depth": coarse_depth,
                        "rho_co2(g)": coarse_co2_g_density,
                        "solubility_co2(aq)": coarse_co2_aq_solubility,
                        "porosity": coarse_porosity,
                        "effective_depth": coarse_effective_depth,
                    }
                    coarse_signal = coarse_signal_dict[
                        signal_option_ptr[signal_option_idx]
                    ]

                    # Determine contours
                    coarse_contour = darsia.plot_contour_on_image(
                        coarse_image,
                        mask=[
                            coarse_c_aq > sliders_threshold[0].val,
                            coarse_c_aq > 0.1,
                            coarse_c_aq > 0.25,
                            coarse_c_aq > 0.50,
                            coarse_c_aq > 0.9,
                            coarse_s_g > sliders_threshold[1].val,
                        ],
                        color=[
                            (255, 0, 0),
                            (255, 0, 0),
                            (255, 0, 0),
                            (255, 0, 0),
                            (255, 0, 0),
                            (0, 255, 0),
                        ],
                        alpha=[0.2, 0.3, 0.4, 0.5, 1.0, 1.0],
                        return_image=True,
                    )

                    # Update mass analysis
                    update_mass_analysis()

                    # Update signal function plot
                    if signal_function_line is not None:
                        signal_func = self.signal_model.model[1][label_idx]
                        x_vals = signal_func.supports
                        y_vals = signal_func.values.copy()
                        signal_function_line.set_data(x_vals, y_vals)
                        signal_function_scatter.set_offsets(np.c_[x_vals, y_vals])

                    # Update flash cut-off and max value lines
                    signal_function_aq_max_y.set_ydata(self.flash.max_value_aq)
                    signal_function_aq_min_y.set_ydata(self.flash.min_value_aq)
                    signal_function_g_max_y.set_ydata(self.flash.max_value_g)
                    signal_function_g_min_y.set_ydata(self.flash.min_value_g)
                    aq_max_x = signal_func.inverse(self.flash.max_value_aq)
                    aq_min_x = signal_func.inverse(self.flash.min_value_aq)
                    g_max_x = signal_func.inverse(self.flash.max_value_g)
                    g_min_x = signal_func.inverse(self.flash.min_value_g)
                    signal_function_aq_max_x.set_xdata(aq_max_x)
                    signal_function_aq_min_x.set_xdata(aq_min_x)
                    signal_function_g_max_x.set_xdata(g_max_x)
                    signal_function_g_min_x.set_xdata(g_min_x)

                    # Update the position of the annotaion
                    ax_signal_function.texts[0].set_position((aq_max_x - 0.05, 0.04))
                    ax_signal_function.texts[1].set_position((aq_max_x + 0.05, 0.04))
                    ax_signal_function.texts[2].set_position((g_max_x + 0.1, 0.04))

                    # Update dense plots
                    coarse_signal_img.set_data(coarse_signal.img)
                    coarse_contour_img.set_data(coarse_contour.img)

                    # Update color scale (min/max values) based on current signal
                    if signal_option_ptr[signal_option_idx] == "pH":
                        signal_min = 0.0
                        signal_max = 1.0
                    else:
                        signal_min = np.nanmin(coarse_signal.img)
                        signal_max = np.nanmax(coarse_signal.img)
                    coarse_signal_img.set_clim(vmin=signal_min, vmax=signal_max)
                    coarse_signal_colorbar.update_normal(coarse_signal_img)

                    # Update mass plots
                    for roi_key in rois:
                        integrated_mass_plots[roi_key].set_data(
                            times[roi_key], integrated_mass[roi_key]
                        )
                        integrated_mass_plots_g[roi_key].set_data(
                            times[roi_key], integrated_mass_g[roi_key]
                        )
                        integrated_mass_plots_aq[roi_key].set_data(
                            times[roi_key], integrated_mass_aq[roi_key]
                        )
                        integrated_mass_scatters[roi_key].set_offsets(
                            np.c_[times[roi_key], integrated_mass[roi_key]]
                        )
                        integrated_mass_scatters_g[roi_key].set_offsets(
                            np.c_[times[roi_key], integrated_mass_g[roi_key]]
                        )
                        integrated_mass_scatters_aq[roi_key].set_offsets(
                            np.c_[times[roi_key], integrated_mass_aq[roi_key]]
                        )

                    # Redraw
                    fig.canvas.draw_idle()

            def slider_down_arrow(slider_index):
                """Decrement signal slider by one step.

                Args:
                    slider_index (int): Index of the signal slider
                """
                # Get the slider object
                slider = sliders_color_to_signal[slider_index]

                # Get current value and slider parameters
                # current_val = slider.val
                # step_size = slider.valstep
                max_val = slider.valmax

                # # Calculate new value with step increment, capped at maximum
                # slider.set_val(new_val)
                # Fetch all values of all color_path_interpolations for all labels
                available_labels = np.sort(list(self.signal_model.model[1].keys()))
                current_values = [
                    self.signal_model.model[1][_label_idx].values[slider_index]
                    for _label_idx in available_labels
                ]

                # Fetch all values of next slider index for all labels
                if slider_index - 1 >= 0:
                    previous_slider_index = slider_index - 1
                    previous_values = {
                        _label_idx: self.signal_model.model[1][_label_idx].values[
                            previous_slider_index
                        ]
                        for _label_idx in available_labels
                    }
                    fifty_percent_decrease = {
                        _label_idx: 0.5
                        * (current_values[_label_idx] - previous_values[_label_idx])
                        for _label_idx in available_labels
                    }
                    new_values = {
                        _label_idx: min(
                            current_values[_label_idx]
                            - fifty_percent_decrease[_label_idx],
                            max_val,
                        )
                        for _label_idx in available_labels
                    }

                    # Update the current slider
                    slider.set_val(new_values[label_idx])

                    # Update parameters

                    for _label_idx in available_labels:
                        _old_values = self.signal_model.model[1][_label_idx].values
                        _new_values = _old_values.copy()
                        _new_values[slider_index] = new_values[_label_idx]
                        self.signal_model.model[1][_label_idx].update(
                            values=_new_values
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
                        "color": coarse_color_interpretation,
                        "pH": coarse_pH,
                        "rho_co2": coarse_density,
                        "c_aq": coarse_c_aq,
                        "s_g": coarse_s_g,
                        "labels": coarse_labels,
                        "depth": coarse_depth,
                        "rho_co2(g)": coarse_co2_g_density,
                        "solubility_co2(aq)": coarse_co2_aq_solubility,
                        "porosity": coarse_porosity,
                        "effective_depth": coarse_effective_depth,
                    }
                    coarse_signal = coarse_signal_dict[
                        signal_option_ptr[signal_option_idx]
                    ]

                    # Determine contours
                    coarse_contour = darsia.plot_contour_on_image(
                        coarse_image,
                        mask=[
                            coarse_c_aq > sliders_threshold[0].val,
                            coarse_c_aq > 0.1,
                            coarse_c_aq > 0.25,
                            coarse_c_aq > 0.5,
                            coarse_c_aq > 0.9,
                            coarse_s_g > sliders_threshold[1].val,
                        ],
                        color=[
                            (255, 0, 0),
                            (255, 0, 0),
                            (255, 0, 0),
                            (255, 0, 0),
                            (255, 0, 0),
                            (0, 255, 0),
                        ],
                        alpha=[0.2, 0.3, 0.4, 0.5, 1.0, 1.0],
                        return_image=True,
                    )

                    # Update mass analysis
                    update_mass_analysis()

                    # Update signal function plot
                    if signal_function_line is not None:
                        signal_func = self.signal_model.model[1][label_idx]
                        x_vals = signal_func.supports
                        y_vals = signal_func.values.copy()
                        signal_function_line.set_data(x_vals, y_vals)
                        signal_function_scatter.set_offsets(np.c_[x_vals, y_vals])

                    # Update flash cut-off and max value lines
                    signal_function_aq_max_y.set_ydata(self.flash.max_value_aq)
                    signal_function_aq_min_y.set_ydata(self.flash.min_value_aq)
                    signal_function_g_max_y.set_ydata(self.flash.max_value_g)
                    signal_function_g_min_y.set_ydata(self.flash.min_value_g)
                    aq_max_x = signal_func.inverse(self.flash.max_value_aq)
                    aq_min_x = signal_func.inverse(self.flash.min_value_aq)
                    g_max_x = signal_func.inverse(self.flash.max_value_g)
                    g_min_x = signal_func.inverse(self.flash.min_value_g)
                    signal_function_aq_max_x.set_xdata(aq_max_x)
                    signal_function_aq_min_x.set_xdata(aq_min_x)
                    signal_function_g_max_x.set_xdata(g_max_x)
                    signal_function_g_min_x.set_xdata(g_min_x)

                    # Update the position of the annotaion
                    ax_signal_function.texts[0].set_position((aq_max_x - 0.05, 0.04))
                    ax_signal_function.texts[1].set_position((aq_max_x + 0.05, 0.04))
                    ax_signal_function.texts[2].set_position((g_max_x + 0.1, 0.04))

                    # Update dense plots
                    coarse_signal_img.set_data(coarse_signal.img)
                    coarse_contour_img.set_data(coarse_contour.img)

                    # Update color scale (min/max values) based on current signal
                    if signal_option_ptr[signal_option_idx] == "pH":
                        signal_min = 0.0
                        signal_max = 1.0
                    else:
                        signal_min = np.nanmin(coarse_signal.img)
                        signal_max = np.nanmax(coarse_signal.img)
                    coarse_signal_img.set_clim(vmin=signal_min, vmax=signal_max)
                    coarse_signal_colorbar.update_normal(coarse_signal_img)

                    # Update mass plots
                    for roi_key in rois:
                        integrated_mass_plots[roi_key].set_data(
                            times[roi_key], integrated_mass[roi_key]
                        )
                        integrated_mass_plots_g[roi_key].set_data(
                            times[roi_key], integrated_mass_g[roi_key]
                        )
                        integrated_mass_plots_aq[roi_key].set_data(
                            times[roi_key], integrated_mass_aq[roi_key]
                        )
                        integrated_mass_scatters[roi_key].set_offsets(
                            np.c_[times[roi_key], integrated_mass[roi_key]]
                        )
                        integrated_mass_scatters_g[roi_key].set_offsets(
                            np.c_[times[roi_key], integrated_mass_g[roi_key]]
                        )
                        integrated_mass_scatters_aq[roi_key].set_offsets(
                            np.c_[times[roi_key], integrated_mass_aq[roi_key]]
                        )

                    # Redraw
                    fig.canvas.draw_idle()

            # Create arrow buttons for each signal slider
            arrow_buttons_up = []
            arrow_buttons_down = []

            for i in range(num_value_sliders):
                slider_x = slider_left + i * (slider_width_individual + spacing)

                # Up arrow button
                ax_up_button = fig.add_axes(
                    [
                        slider_x,
                        up_button_y,
                        slider_width_individual,
                        arrow_button_height,
                    ]
                )
                btn_up = Button(ax_up_button, "▲")
                arrow_buttons_up.append(btn_up)

                # Down arrow button
                ax_down_button = fig.add_axes(
                    [
                        slider_x,
                        down_button_y,
                        slider_width_individual,
                        arrow_button_height,
                    ]
                )
                btn_down = Button(ax_down_button, "▼")
                arrow_buttons_down.append(btn_down)

            # Add four vertical sliders for thresholding flash model
            flash_slider_start_idx = num_value_sliders  # Flash min value aq slider
            slider_x = slider_left + flash_slider_start_idx * (
                slider_width_individual + spacing
            )
            slider_y = slider_bottom + slider_height_total * 0.2

            ax_slider_min_aq = fig.add_axes(
                [
                    slider_x,
                    slider_y,
                    slider_width_individual,
                    slider_height_individual,
                ]
            )

            slider_min_aq = Slider(
                ax_slider_min_aq,
                "Flash\nmin aq",
                0.0,
                2.0,
                valinit=self.flash.min_value_aq,
                valstep=0.01,
                color="darkgray",
                orientation="vertical",
            )

            # Flash max slider
            slider_x = slider_left + (flash_slider_start_idx + 1) * (
                slider_width_individual + spacing
            )

            ax_slider_max_aq = fig.add_axes(
                [
                    slider_x,
                    slider_y,
                    slider_width_individual,
                    slider_height_individual,
                ]
            )
            slider_max_aq = Slider(
                ax_slider_max_aq,
                "Flash\nmax aq",
                0.0,
                2.0,
                valinit=self.flash.max_value_aq,
                valstep=0.01,
                color="darkgray",
                orientation="vertical",
            )
            slider_x = slider_left + (flash_slider_start_idx + 2) * (
                slider_width_individual + spacing
            )
            ax_slider_min_g = fig.add_axes(
                [
                    slider_x,
                    slider_y,
                    slider_width_individual,
                    slider_height_individual,
                ]
            )

            slider_min_g = Slider(
                ax_slider_min_g,
                "Flash\nmin g",
                0.0,
                2.0,
                valinit=self.flash.min_value_g,
                valstep=0.01,
                color="darkgray",
                orientation="vertical",
            )

            # Flash max slider
            slider_x = slider_left + (flash_slider_start_idx + 3) * (
                slider_width_individual + spacing
            )

            ax_slider_max_g = fig.add_axes(
                [
                    slider_x,
                    slider_y,
                    slider_width_individual,
                    slider_height_individual,
                ]
            )
            slider_max_g = Slider(
                ax_slider_max_g,
                "Flash\nmax g",
                0.0,
                2.0,
                valinit=self.flash.max_value_g,
                valstep=0.01,
                color="darkgray",
                orientation="vertical",
            )
            sliders_flash = [slider_min_aq, slider_max_aq, slider_min_g, slider_max_g]

            # Sliders for thresholding flash model for c_aq and s_g, start with value 0.05
            slider_x = slider_left + (flash_slider_start_idx + 4) * (
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
                "Cont.\nc_aq",
                0.0,
                1.0,
                valinit=0.05,
                valstep=0.01,
                color=(1, 0, 0),
                orientation="vertical",
            )
            slider_x = slider_left + (flash_slider_start_idx + 5) * (
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
                "Cont.\ns_g",
                0.0,
                1.0,
                valinit=0.05,
                valstep=0.01,
                color=(0, 1, 0),
                orientation="vertical",
            )
            sliders_threshold = [
                slider_c_aq,
                slider_s_g,
            ]

            # Add depth scaling slider
            slider_x = slider_left + (flash_slider_start_idx + 6) * (
                slider_width_individual + spacing
            )
            ax_slider_depth = fig.add_axes(
                [
                    slider_x,
                    slider_y,
                    slider_width_individual,
                    slider_height_individual,
                ]
            )
            slider_depth = Slider(
                ax_slider_depth,
                "Depth\ncorr.(+/-)",
                -0.01,
                0.01,
                valinit=0.0,
                valstep=0.001,
                color=(0, 0, 1),  # Blue color
                orientation="vertical",
            )
            sliders_depth = [slider_depth]

            # Add arrow buttons for flash sliders (cut-off and max)
            flash_arrow_buttons_up = []
            flash_arrow_buttons_down = []

            for i in range(num_flash_sliders):
                slider_x = slider_left + (flash_slider_start_idx + i) * (
                    slider_width_individual + spacing
                )

                # Up arrow button for flash sliders
                ax_flash_up_button = fig.add_axes(
                    [
                        slider_x,
                        up_button_y,
                        slider_width_individual,
                        arrow_button_height,
                    ]
                )
                btn_flash_up = Button(ax_flash_up_button, "▲")
                flash_arrow_buttons_up.append(btn_flash_up)

                # Down arrow button for flash sliders
                ax_flash_down_button = fig.add_axes(
                    [
                        slider_x,
                        down_button_y,
                        slider_width_individual,
                        arrow_button_height,
                    ]
                )
                btn_flash_down = Button(ax_flash_down_button, "▼")
                flash_arrow_buttons_down.append(btn_flash_down)

            # Add arrow buttons for threshold sliders (c_aq and s_g)
            threshold_arrow_buttons_up = []
            threshold_arrow_buttons_down = []

            for i in range(num_threshold_sliders):
                slider_x = slider_left + (
                    flash_slider_start_idx + num_flash_sliders + i
                ) * (slider_width_individual + spacing)

                # Up arrow button for threshold sliders
                ax_threshold_up_button = fig.add_axes(
                    [
                        slider_x,
                        up_button_y,
                        slider_width_individual,
                        arrow_button_height,
                    ]
                )
                btn_threshold_up = Button(ax_threshold_up_button, "▲")
                threshold_arrow_buttons_up.append(btn_threshold_up)

                # Down arrow button for threshold sliders
                ax_threshold_down_button = fig.add_axes(
                    [
                        slider_x,
                        down_button_y,
                        slider_width_individual,
                        arrow_button_height,
                    ]
                )
                btn_threshold_down = Button(ax_threshold_down_button, "▼")
                threshold_arrow_buttons_down.append(btn_threshold_down)

            # Add arrow buttons for depth correction slider
            depth_arrow_buttons_up = []
            depth_arrow_buttons_down = []

            for i in range(num_depth_sliders):
                slider_x = slider_left + (
                    flash_slider_start_idx
                    + num_flash_sliders
                    + num_threshold_sliders
                    + i
                ) * (slider_width_individual + spacing)

                # Up arrow button for depth correction slider
                ax_depth_up_button = fig.add_axes(
                    [
                        slider_x,
                        up_button_y,
                        slider_width_individual,
                        arrow_button_height,
                    ]
                )
                btn_depth_up = Button(ax_depth_up_button, "▲")
                depth_arrow_buttons_up.append(btn_depth_up)

                # Down arrow button for depth correction slider
                ax_depth_down_button = fig.add_axes(
                    [
                        slider_x,
                        down_button_y,
                        slider_width_individual,
                        arrow_button_height,
                    ]
                )
                btn_depth_down = Button(ax_depth_down_button, "▼")
                depth_arrow_buttons_down.append(btn_depth_down)

            # Create empty functions for flash slider arrow button callbacks
            def flash_slider_up_arrow(slider_index):
                """Increment flash slider by one step. And all signal function sliders across all labels.

                Args:
                    slider_index (int): Index of the flash slider (0=cut-off, 1=max)

                """
                # Get the slider object
                slider = sliders_flash[slider_index]

                # Get current value and slider parameters
                current_val = slider.val
                step_size = slider.valstep
                max_val = slider.valmax

                # Calculate new value with step increment, capped at maximum
                new_val = min(current_val + step_size, max_val)
                slider.set_val(new_val)

            def flash_slider_down_arrow(slider_index):
                """Decrement flash slider by one step.

                Args:
                    slider_index (int): Index of the flash slider (0=cut-off, 1=max)
                """
                # Get the slider object
                slider = sliders_flash[slider_index]

                # Get current value and slider parameters
                current_val = slider.val
                step_size = slider.valstep
                min_val = slider.valmin

                # Calculate new value with step decrement, capped at minimum
                new_val = max(current_val - step_size, min_val)
                slider.set_val(new_val)

            # Create empty functions for threshold slider arrow button callbacks
            def threshold_slider_up_arrow(slider_index):
                """Increment threshold slider by one step.

                Args:
                    slider_index (int): Index of the threshold slider (0=c_aq, 1=s_g)
                """
                # Get the slider object
                slider = sliders_threshold[slider_index]

                # Get current value and slider parameters
                current_val = slider.val
                step_size = slider.valstep
                max_val = slider.valmax

                # Calculate new value with step increment, capped at maximum
                new_val = min(current_val + step_size, max_val)
                slider.set_val(new_val)

            def threshold_slider_down_arrow(slider_index):
                """Decrement threshold slider by one step.

                Args:
                    slider_index (int): Index of the threshold slider (0=c_aq, 1=s_g)
                """
                # Get the slider object
                slider = sliders_threshold[slider_index]

                # Get current value and slider parameters
                current_val = slider.val
                step_size = slider.valstep
                min_val = slider.valmin

                # Calculate new value with step decrement, capped at minimum
                new_val = max(current_val - step_size, min_val)
                slider.set_val(new_val)

            # Create empty functions for depth scaling slider arrow button callbacks
            def depth_slider_up_arrow(slider_index):
                """Increment depth scaling slider by one step.

                Args:
                    slider_index (int): Index of the depth scaling slider (always 0)
                """
                pass

            #    nonlocal \
            #        coarse_depth, \
            #        times, \
            #        integrated_mass, \
            #        integrated_mass_g, \
            #        integrated_mass_aq
            #    # Get the slider object
            #    slider = sliders_depth[slider_index]

            #    # Get current value and slider parameters
            #    current_val = slider.val
            #    step_size = slider.valstep
            #    max_val = slider.valmax

            #    # Calculate new value with step increment, capped at maximum
            #    new_val = min(current_val + step_size, max_val)
            #    slider.set_val(new_val)

            #    # Update the depth in the geometry
            #    new_depth = self.original_depth.copy()
            #    new_depth.img += new_val
            #    self.geometry.update(depth=new_depth)
            #    self.analysis.geometry.update(depth=new_depth)

            #    # Update coarse depth for visualization
            #    coarse_depth = darsia.resize(self.geometry.depth, shape=coarse_shape)

            #    # Update coarse signal plot if active
            #    if signal_option_ptr[signal_option_idx] == "depth":
            #        coarse_signal = coarse_depth
            #        coarse_signal_img.set_data(coarse_signal.img)

            #        # Update color scale (min/max values) based on current signal
            #        signal_min = np.nanmin(coarse_signal.img)
            #        signal_max = np.nanmax(coarse_signal.img)
            #        coarse_signal_img.set_clim(vmin=signal_min, vmax=signal_max)
            #        coarse_signal_colorbar.update_normal(coarse_signal_img)

            #    # Update mass analysis
            #    update_mass_analysis()

            #    # Update mass plots
            #    integrated_mass_plot.set_data(times, integrated_mass)
            #    integrated_mass_plot_g.set_data(times, integrated_mass_g)
            #    integrated_mass_plot_aq.set_data(times, integrated_mass_aq)
            #    integrated_mass_scatter.set_offsets(np.c_[times, integrated_mass])
            #    integrated_mass_scatter_g.set_offsets(np.c_[times, integrated_mass_g])
            #    integrated_mass_scatter_aq.set_offsets(np.c_[times, integrated_mass_aq])

            #    # Redraw
            #    fig.canvas.draw_idle()

            def depth_slider_down_arrow(slider_index):
                """Decrement depth scaling slider by one step.

                Args:
                    slider_index (int): Index of the depth scaling slider (always 0)
                """
                pass

            #    nonlocal \
            #        coarse_depth, \
            #        times, \
            #        integrated_mass, \
            #        integrated_mass_g, \
            #        integrated_mass_aq
            #    # Get the slider object
            #    slider = sliders_depth[slider_index]

            #    # Get current value and slider parameters
            #    current_val = slider.val
            #    step_size = slider.valstep
            #    min_val = slider.valmin

            #    # Calculate new value with step decrement, capped at minimum
            #    new_val = max(current_val - step_size, min_val)
            #    slider.set_val(new_val)

            #    # Update the depth in the geometry
            #    new_depth = self.original_depth.copy()
            #    new_depth.img += new_val
            #    self.geometry.update(depth=new_depth)
            #    self.analysis.geometry.update(depth=new_depth)

            #    # Update coarse depth for visualization
            #    coarse_depth = darsia.resize(self.geometry.depth, shape=coarse_shape)

            #    # Update coarse signal plot if active
            #    if signal_option_ptr[signal_option_idx] == "depth":
            #        coarse_signal = coarse_depth
            #        coarse_signal_img.set_data(coarse_signal.img)

            #        # Update color scale (min/max values) based on current signal
            #        signal_min = np.nanmin(coarse_signal.img)
            #        signal_max = np.nanmax(coarse_signal.img)
            #        coarse_signal_img.set_clim(vmin=signal_min, vmax=signal_max)
            #        coarse_signal_colorbar.update_normal(coarse_signal_img)

            #    # Update mass analysis
            #    update_mass_analysis()

            #    # Update mass plots
            #    integrated_mass_plot.set_data(times, integrated_mass)
            #    integrated_mass_plot_g.set_data(times, integrated_mass_g)
            #    integrated_mass_plot_aq.set_data(times, integrated_mass_aq)
            #    integrated_mass_scatter.set_offsets(np.c_[times, integrated_mass])
            #    integrated_mass_scatter_g.set_offsets(np.c_[times, integrated_mass_g])
            #    integrated_mass_scatter_aq.set_offsets(np.c_[times, integrated_mass_aq])

            #    # Redraw
            #    fig.canvas.draw_idle()

            # Position buttons at the top of the figure
            button_width = 0.08
            button_height = 0.04
            button_y = 0.95
            button_spacing = 0.01

            # Button order from left to right (Update moved to bottom right):
            # - Next Signal
            # - Next Label
            # - New Label
            # - Next Image
            # - New Image
            # - Finish
            # Calculate positions for 6 buttons with proper spacing
            total_buttons = 5
            button_start_x = (
                0.98
                - total_buttons * button_width
                - (total_buttons - 1) * button_spacing
            )

            # Position Update button at bottom right corner
            update_button_x = 0.98 - button_width  # Right side with small margin
            update_button_y = 0.02  # Bottom with small margin

            # Add "Update label" button to the left of "Update all"
            update_label_button_x = update_button_x - button_width - button_spacing
            ax_update_label = plt.axes(
                [
                    update_label_button_x,
                    update_button_y,
                    button_width,
                    button_height,
                ]
            )
            btn_update_label = Button(ax_update_label, "Update label")

            # Add "Update all" button
            ax_update = plt.axes(
                [
                    update_button_x,
                    update_button_y,
                    button_width,
                    button_height,
                ]
            )
            btn_update = Button(ax_update, "Update all")

            ax_next_signal = plt.axes(
                [
                    button_start_x + 0 * (button_width + button_spacing),
                    button_y,
                    button_width,
                    button_height,
                ]
            )
            btn_next_signal = Button(ax_next_signal, "Next Signal")

            ax_next_label = plt.axes(
                [
                    button_start_x + 1 * (button_width + button_spacing),
                    button_y,
                    button_width,
                    button_height,
                ]
            )
            btn_next_label = Button(ax_next_label, "Next Label")

            ax_new_label = plt.axes(
                [
                    button_start_x + 2 * (button_width + button_spacing),
                    button_y,
                    button_width,
                    button_height,
                ]
            )
            btn_new_label = Button(ax_new_label, "New Label")

            ax_next_image = plt.axes(
                [
                    button_start_x + 3 * (button_width + button_spacing),
                    button_y,
                    button_width,
                    button_height,
                ]
            )
            btn_next_image = Button(ax_next_image, "Next Image")

            ax_new_image = plt.axes(
                [
                    button_start_x + 4 * (button_width + button_spacing),
                    button_y,
                    button_width,
                    button_height,
                ]
            )
            btn_new_image = Button(ax_new_image, "New Image")

            finish_button_x = 0.02  # Right side with small margin
            ax_finish = plt.axes(
                [
                    finish_button_x,
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
                                sliders_flash[0].val != self.flash.min_value_aq,
                                sliders_flash[1].val != self.flash.max_value_aq,
                                sliders_flash[2].val != self.flash.min_value_g,
                                sliders_flash[3].val != self.flash.max_value_g,
                            ]
                        )

                        ## Update parameters
                        self.signal_model.model[1][_label_idx].update(
                            values=[slider.val for slider in sliders_color_to_signal]
                        )
                        self.flash.update(
                            min_value_aq=sliders_flash[0].val,
                            max_value_aq=sliders_flash[1].val,
                            min_value_g=sliders_flash[2].val,
                            max_value_g=sliders_flash[3].val,
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
                            "color": coarse_color_interpretation,
                            "pH": coarse_pH,
                            "rho_co2": coarse_density,
                            "c_aq": coarse_c_aq,
                            "s_g": coarse_s_g,
                            "labels": coarse_labels,
                            "depth": coarse_depth,
                            "rho_co2(g)": coarse_co2_g_density,
                            "solubility_co2(aq)": coarse_co2_aq_solubility,
                            "porosity": coarse_porosity,
                            "effective_depth": coarse_effective_depth,
                        }
                        coarse_signal = coarse_signal_dict[
                            signal_option_ptr[signal_option_idx]
                        ]

                        # Determine contours
                        coarse_contour = darsia.plot_contour_on_image(
                            coarse_image,
                            mask=[
                                coarse_c_aq > sliders_threshold[0].val,
                                coarse_c_aq > 0.1,
                                coarse_c_aq > 0.25,
                                coarse_c_aq > 0.5,
                                coarse_c_aq > 0.9,
                                coarse_s_g > sliders_threshold[1].val,
                            ],
                            color=[
                                (255, 0, 0),
                                (255, 0, 0),
                                (255, 0, 0),
                                (255, 0, 0),
                                (255, 0, 0),
                                (0, 255, 0),
                            ],
                            alpha=[0.2, 0.3, 0.4, 0.5, 1.0, 1.0],
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
                        signal_function_aq_max_y.set_ydata(self.flash.max_value_aq)
                        signal_function_g_max_y.set_ydata(self.flash.max_value_g)
                        signal_function_g_min_y.set_ydata(self.flash.min_value_g)
                        aq_max_x = signal_func.inverse(self.flash.max_value_aq)
                        aq_min_x = signal_func.inverse(self.flash.min_value_aq)
                        g_max_x = signal_func.inverse(self.flash.max_value_g)
                        g_min_x = signal_func.inverse(self.flash.min_value_g)
                        signal_function_aq_max_x.set_xdata(aq_max_x)
                        signal_function_aq_min_x.set_xdata(aq_min_x)
                        signal_function_g_max_x.set_xdata(g_max_x)
                        signal_function_g_min_x.set_xdata(g_min_x)

                        # Update the position of the annotaion
                        ax_signal_function.texts[0].set_position(
                            (aq_max_x - 0.05, 0.04)
                        )
                        ax_signal_function.texts[1].set_position(
                            (aq_max_x + 0.05, 0.04)
                        )
                        ax_signal_function.texts[2].set_position((g_max_x + 0.1, 0.04))

                        # Update dense plots
                        coarse_signal_img.set_data(coarse_signal.img)
                        coarse_contour_img.set_data(coarse_contour.img)

                        # Update color scale (min/max values) based on current signal
                        if signal_option_ptr[signal_option_idx] == "pH":
                            signal_min = 0.0
                            signal_max = 1.0
                        else:
                            signal_min = np.nanmin(coarse_signal.img)
                            signal_max = np.nanmax(coarse_signal.img)
                        coarse_signal_img.set_clim(vmin=signal_min, vmax=signal_max)
                        coarse_signal_colorbar.update_normal(coarse_signal_img)

                        # Update mass plots
                        for roi_key in rois:
                            integrated_mass_plots[roi_key].set_data(
                                times[roi_key], integrated_mass[roi_key]
                            )
                            integrated_mass_plots_g[roi_key].set_data(
                                times[roi_key], integrated_mass_g[roi_key]
                            )
                            integrated_mass_plots_aq[roi_key].set_data(
                                times[roi_key], integrated_mass_aq[roi_key]
                            )
                            integrated_mass_scatters[roi_key].set_offsets(
                                np.c_[times[roi_key], integrated_mass[roi_key]]
                            )
                            integrated_mass_scatters_g[roi_key].set_offsets(
                                np.c_[times[roi_key], integrated_mass_g[roi_key]]
                            )
                            integrated_mass_scatters_aq[roi_key].set_offsets(
                                np.c_[times[roi_key], integrated_mass_aq[roi_key]]
                            )

                        # Redraw
                        fig.canvas.draw_idle()

                    def update_label_analysis(event=None):
                        """Update analysis for current label only (copy of update_analysis for now)"""
                        nonlocal \
                            done_tuning_values, \
                            sliders_color_to_signal, \
                            sliders_flash, \
                            sliders_threshold, \
                            _label_idx

                        # Update parameters
                        self.signal_model.model[1][_label_idx].update(
                            values=[slider.val for slider in sliders_color_to_signal]
                        )
                        self.flash.update(
                            min_value_aq=sliders_flash[0].val,
                            max_value_aq=sliders_flash[1].val,
                            min_value_g=sliders_flash[2].val,
                            max_value_g=sliders_flash[3].val,
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
                            "color": coarse_color_interpretation,
                            "pH": coarse_pH,
                            "rho_co2": coarse_density,
                            "c_aq": coarse_c_aq,
                            "s_g": coarse_s_g,
                            "labels": coarse_labels,
                            "depth": coarse_depth,
                            "rho_co2(g)": coarse_co2_g_density,
                            "solubility_co2(aq)": coarse_co2_aq_solubility,
                            "porosity": coarse_porosity,
                            "effective_depth": coarse_effective_depth,
                        }
                        coarse_signal = coarse_signal_dict[
                            signal_option_ptr[signal_option_idx]
                        ]

                        # Determine contours
                        coarse_contour = darsia.plot_contour_on_image(
                            coarse_image,
                            mask=[
                                coarse_c_aq > sliders_threshold[0].val,
                                coarse_c_aq > 0.1,
                                coarse_c_aq > 0.25,
                                coarse_c_aq > 0.5,
                                coarse_c_aq > 0.9,
                                coarse_s_g > sliders_threshold[1].val,
                            ],
                            color=[
                                (255, 0, 0),
                                (255, 0, 0),
                                (255, 0, 0),
                                (255, 0, 0),
                                (255, 0, 0),
                                (0, 255, 0),
                            ],
                            alpha=[0.2, 0.3, 0.4, 0.5, 1.0, 1.0],
                            return_image=True,
                        )

                        # Update signal function plot
                        if signal_function_line is not None:
                            signal_func = self.signal_model.model[1][_label_idx]
                            x_vals = signal_func.supports
                            y_vals = signal_func.values.copy()
                            signal_function_line.set_data(x_vals, y_vals)
                            signal_function_scatter.set_offsets(np.c_[x_vals, y_vals])

                        # Update flash cut-off and max value lines
                        signal_function_aq_max_y.set_ydata(self.flash.max_value_aq)
                        signal_function_g_max_y.set_ydata(self.flash.max_value_g)
                        signal_function_g_min_y.set_ydata(self.flash.min_value_g)
                        aq_max_x = signal_func.inverse(self.flash.max_value_aq)
                        aq_min_x = signal_func.inverse(self.flash.min_value_aq)
                        g_max_x = signal_func.inverse(self.flash.max_value_g)
                        g_min_x = signal_func.inverse(self.flash.min_value_g)
                        signal_function_aq_max_x.set_xdata(aq_max_x)
                        signal_function_aq_min_x.set_xdata(aq_min_x)
                        signal_function_g_max_x.set_xdata(g_max_x)
                        signal_function_g_min_x.set_xdata(g_min_x)

                        # Update the position of the annotation
                        ax_signal_function.texts[0].set_position(
                            (aq_max_x - 0.05, 0.04)
                        )
                        ax_signal_function.texts[1].set_position(
                            (aq_max_x + 0.05, 0.04)
                        )
                        ax_signal_function.texts[2].set_position((g_max_x + 0.1, 0.04))

                        # Update dense plots
                        coarse_signal_img.set_data(coarse_signal.img)
                        coarse_contour_img.set_data(coarse_contour.img)

                        # Update color scale (min/max values) based on current signal
                        if signal_option_ptr[signal_option_idx] == "pH":
                            signal_min = 0.0
                            signal_max = 1.0
                        else:
                            signal_min = np.nanmin(coarse_signal.img)
                            signal_max = np.nanmax(coarse_signal.img)
                        coarse_signal_img.set_clim(vmin=signal_min, vmax=signal_max)
                        coarse_signal_colorbar.update_normal(coarse_signal_img)

                        # Redraw
                        fig.canvas.draw_idle()

                    # Connect button callbacks
                    btn_update.on_clicked(update_analysis)
                    btn_update_label.on_clicked(update_label_analysis)

                    def new_label(event):
                        nonlocal done_tuning_values, need_to_pick_new_label
                        done_tuning_values = True
                        need_to_pick_new_label = True
                        plt.close("all")

                    btn_new_label.on_clicked(new_label)

                    def new_image(event):
                        nonlocal done_tuning_values, need_to_pick_new_image
                        done_tuning_values = True
                        need_to_pick_new_image = True
                        plt.close("all")

                    btn_new_image.on_clicked(new_image)

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
                            "color": coarse_color_interpretation,
                            "pH": coarse_pH,
                            "rho_co2": coarse_density,
                            "c_aq": coarse_c_aq,
                            "s_g": coarse_s_g,
                            "labels": coarse_labels,
                            "depth": coarse_depth,
                            "rho_co2(g)": coarse_co2_g_density,
                            "solubility_co2(aq)": coarse_co2_aq_solubility,
                            "porosity": coarse_porosity,
                            "effective_depth": coarse_effective_depth,
                        }
                        coarse_signal = coarse_signal_dict[
                            signal_option_ptr[signal_option_idx]
                        ]

                        # Update the image and title
                        coarse_signal_img.set_data(
                            coarse_signal.img
                            if isinstance(coarse_signal, darsia.Image)
                            else coarse_signal
                        )
                        ax_signal.set_title(
                            signal_options[signal_option_ptr[signal_option_idx]]
                        )  # Update color scale (min/max values) based on current signal
                        if signal_option_ptr[signal_option_idx] == "pH":
                            signal_min = 0.0
                            signal_max = 1.0
                        else:
                            signal_min = np.nanmin(coarse_signal.img)
                            signal_max = np.nanmax(coarse_signal.img)
                        coarse_signal_img.set_clim(vmin=signal_min, vmax=signal_max)
                        coarse_signal_colorbar.update_normal(coarse_signal_img)
                        # Redraw
                        fig.canvas.draw_idle()

                    btn_next_signal.on_clicked(next_signal)

                    def next_label(event):
                        nonlocal label_idx, _label_idx
                        # Get available labels from the labels image
                        available_labels = np.sort(
                            list(self.signal_model.model[1].keys())
                        )
                        available_labels = available_labels[
                            available_labels >= 0
                        ]  # Remove negative labels if any

                        # Find current position and move to next
                        if label_idx in available_labels:
                            current_pos = np.where(available_labels == label_idx)[0][0]
                            next_pos = (current_pos + 1) % len(available_labels)
                            label_idx = available_labels[next_pos]
                        else:
                            label_idx = available_labels[
                                0
                            ]  # Default to first available label
                        _label_idx = label_idx

                        # Update coarse image plot that depends on label_idx
                        ax_coarse_image.set_title(
                            f"Current Image {image_idx} and Label {label_idx}"
                        )
                        # Update coarse image highlighting
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
                        coarse_image_img.set_data(labeled_coarse_image)

                        # Update signal function
                        ax_signal_function.set_title(
                            f"Signal Function - Label {label_idx}"
                        )
                        if signal_function_line is not None:
                            signal_func = self.signal_model.model[1][label_idx]
                            if (
                                hasattr(signal_func, "values")
                                and len(signal_func.values) > 0
                            ):
                                x_vals = signal_func.supports
                                y_vals = signal_func.values.copy()
                                signal_function_line.set_data(x_vals, y_vals)
                                signal_function_scatter.set_offsets(
                                    np.c_[x_vals, y_vals]
                                )

                        # Plot the signal function for the selected label
                        signal_func: darsia.PWTransformation = self.signal_model.model[
                            1
                        ][label_idx]

                        # Get the color path interpretation for gradient background
                        _color_path_interpretation: darsia.ColorPathInterpolation = (
                            self.color_path_interpretation[label_idx]
                        )

                        # Get the colormap from the color path
                        color_path_cmap = (
                            _color_path_interpretation.color_path.get_color_map()
                        )

                        # Update the cmap in signal_function_background
                        signal_function_background.set_cmap(color_path_cmap)

                        # Create x values for plotting the piecewise linear function
                        x_vals = signal_func.supports
                        y_vals = signal_func.values.copy()
                        signal_function_line.set_data(x_vals, y_vals)
                        signal_function_scatter.set_offsets(np.c_[x_vals, y_vals])

                        # Update flash cut-off and max value lines
                        signal_function_aq_max_y.set_ydata(self.flash.max_value_aq)
                        signal_function_g_max_y.set_ydata(self.flash.max_value_g)
                        signal_function_g_min_y.set_ydata(self.flash.min_value_g)
                        aq_max_x = signal_func.inverse(self.flash.max_value_aq)
                        aq_min_x = signal_func.inverse(self.flash.min_value_aq)
                        g_max_x = signal_func.inverse(self.flash.max_value_g)
                        g_min_x = signal_func.inverse(self.flash.min_value_g)
                        signal_function_aq_max_x.set_xdata(aq_max_x)
                        signal_function_aq_min_x.set_xdata(aq_min_x)
                        signal_function_g_max_x.set_xdata(g_max_x)
                        signal_function_g_min_x.set_xdata(g_min_x)

                        # Update the position of the annotaion
                        ax_signal_function.texts[0].set_position(
                            (aq_max_x - 0.05, 0.04)
                        )
                        ax_signal_function.texts[1].set_position(
                            (aq_max_x + 0.05, 0.04)
                        )
                        ax_signal_function.texts[2].set_position((g_max_x + 0.1, 0.04))

                        # Update values of slides_signal
                        for i, slider in enumerate(sliders_color_to_signal):
                            slider.set_val(
                                self.signal_model.model[1][label_idx].values[i]
                            )

                        # Update colors of sliders_signal
                        for i, slider in enumerate(sliders_color_to_signal):
                            new_color = np.clip(
                                self.color_path_interpretation[
                                    label_idx
                                ].color_path.colors[i],
                                0,
                                1,
                            )
                            slider.color = new_color
                            # Update the visual elements of the slider
                            if hasattr(slider, "poly") and slider.poly is not None:
                                slider.poly.set_facecolor(new_color)
                            if hasattr(slider, "hline") and slider.hline is not None:
                                slider.hline.set_color(new_color)
                            # Force redraw of the slider's axes
                            slider.ax.figure.canvas.draw_idle()

                        # Redraw
                        fig.canvas.draw_idle()

                    btn_next_label.on_clicked(next_label)

                    def next_image(event):
                        nonlocal image_idx, coarse_image, color_interpretation
                        # Cycle to next image
                        image_idx = (image_idx + 1) % len(images)

                        # Update image-dependent variables
                        coarse_image = coarse_images[image_idx]
                        color_interpretation = color_interpretations[image_idx]

                        # Recalculate mass analysis for new image
                        pH, density, c_aq, s_g = detailed_mass_analysis(
                            color_interpretation
                        )

                        # Coarsen outputs for visualization
                        coarse_pH = darsia.resize(pH, shape=coarse_shape)
                        coarse_density = darsia.resize(density, shape=coarse_shape)
                        coarse_c_aq = darsia.resize(c_aq, shape=coarse_shape)
                        coarse_s_g = darsia.resize(s_g, shape=coarse_shape)
                        coarse_signal_dict = {
                            "color": coarse_color_interpretation,
                            "pH": coarse_pH,
                            "rho_co2": coarse_density,
                            "c_aq": coarse_c_aq,
                            "s_g": coarse_s_g,
                            "labels": coarse_labels,
                            "depth": coarse_depth,
                            "rho_co2(g)": coarse_co2_g_density,
                            "solubility_co2(aq)": coarse_co2_aq_solubility,
                            "porosity": coarse_porosity,
                            "effective_depth": coarse_effective_depth,
                        }
                        coarse_signal = coarse_signal_dict[
                            signal_option_ptr[signal_option_idx]
                        ]

                        # Update contours
                        coarse_contour = darsia.plot_contour_on_image(
                            coarse_image,
                            mask=[
                                coarse_c_aq > sliders_threshold[0].val,
                                coarse_c_aq > 0.1,
                                coarse_c_aq > 0.25,
                                coarse_c_aq > 0.5,
                                coarse_c_aq > 0.9,
                                coarse_s_g > sliders_threshold[1].val,
                            ],
                            color=[
                                (255, 0, 0),
                                (255, 0, 0),
                                (255, 0, 0),
                                (255, 0, 0),
                                (255, 0, 0),
                                (0, 255, 0),
                            ],
                            alpha=[0.2, 0.3, 0.4, 0.5, 1.0, 1.0],
                            return_image=True,
                        )

                        # Update plot titles
                        ax_coarse_image.set_title(
                            f"Current Image {image_idx} and Label {label_idx}"
                        )

                        # Update coarse image highlighting
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
                        coarse_image_img.set_data(labeled_coarse_image)

                        # Update signal image and colorbar
                        coarse_signal_img.set_data(coarse_signal.img)
                        if signal_option_ptr[signal_option_idx] == "pH":
                            signal_min = 0.0
                            signal_max = 1.0
                        else:
                            signal_min = np.nanmin(coarse_signal.img)
                            signal_max = np.nanmax(coarse_signal.img)
                        coarse_signal_img.set_clim(vmin=signal_min, vmax=signal_max)
                        coarse_signal_colorbar.update_normal(coarse_signal_img)

                        # Update contour image
                        coarse_contour_img.set_data(coarse_contour.img)

                        # Redraw
                        fig.canvas.draw_idle()

                    btn_next_image.on_clicked(next_image)

                    # Connect signal slider arrow button callbacks
                    for i, btn_up in enumerate(arrow_buttons_up):
                        btn_up.on_clicked(lambda event, idx=i: slider_up_arrow(idx))

                    for i, btn_down in enumerate(arrow_buttons_down):
                        btn_down.on_clicked(lambda event, idx=i: slider_down_arrow(idx))

                    # Connect flash arrow button callbacks
                    for i, btn_flash_up in enumerate(flash_arrow_buttons_up):
                        btn_flash_up.on_clicked(
                            lambda event, idx=i: flash_slider_up_arrow(idx)
                        )

                    for i, btn_flash_down in enumerate(flash_arrow_buttons_down):
                        btn_flash_down.on_clicked(
                            lambda event, idx=i: flash_slider_down_arrow(idx)
                        )

                    # Connect threshold arrow button callbacks
                    for i, btn_threshold_up in enumerate(threshold_arrow_buttons_up):
                        btn_threshold_up.on_clicked(
                            lambda event, idx=i: threshold_slider_up_arrow(idx)
                        )

                    for i, btn_threshold_down in enumerate(
                        threshold_arrow_buttons_down
                    ):
                        btn_threshold_down.on_clicked(
                            lambda event, idx=i: threshold_slider_down_arrow(idx)
                        )

                    # Connect depth scaling arrow button callbacks
                    for i, btn_depth_up in enumerate(depth_arrow_buttons_up):
                        btn_depth_up.on_clicked(
                            lambda event, idx=i: depth_slider_up_arrow(idx)
                        )

                    for i, btn_depth_down in enumerate(depth_arrow_buttons_down):
                        btn_depth_down.on_clicked(
                            lambda event, idx=i: depth_slider_down_arrow(idx)
                        )

                    plt.show()

                show_tuner(label_idx)

    def automatic_calibration(
        self,
        images: list[darsia.Image],
        experiment: darsia.ProtocolledExperiment,
        rois: dict[str, darsia.VoxelArray | darsia.CoordinateArray] | None = None,
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
        # Fill up rois with "entire frame"
        rois = rois or {}
        assert "" not in rois
        rois.update(
            {
                "full": darsia.CoordinateArray(
                    [
                        self.color_analysis.base.origin,
                        self.color_analysis.base.opposite_corner,
                    ]
                )
            }
        )

        # ! ---- COLOR INTERPRETATIONS ----
        color_interpretations = [
            self.call_color_interpretation(image) for image in images
        ]

        # ! ---- INITIAL DOFS ----
        available_labels = np.sort(list(self.signal_model.model[1].keys()))
        ignore_labels = [1, 3, 4]
        initial_dofs = np.hstack(
            [
                np.diff(self.signal_model.model[1][label].values)
                for label in available_labels
                if label not in ignore_labels
            ]
            + [
                self.flash.min_value_aq,
                self.flash.max_value_aq - self.flash.min_value_aq,
                self.flash.min_value_g,
                self.flash.max_value_g - self.flash.min_value_g,
            ]
        )
        logging.info(f"Number of DOFs for optimization: {len(initial_dofs)}")

        # ! ---- OBJECTIVE FUNCTION ----

        def objective_function(dofs: np.ndarray) -> float:
            logging.info(f"DOFs {dofs}")
            logging.info(f"DOFs hash: {hash(tuple(dofs))}")
            # Update the signal model
            idx = 0
            for label in available_labels:
                if label in ignore_labels:
                    continue
                num_values = len(self.signal_model.model[1][label].values)
                new_values = np.cumsum(
                    np.hstack(
                        [
                            0.0,
                            dofs[idx : idx + num_values - 1],
                        ]
                    )
                )
                logging.info(f"Label {label}: {new_values}")
                self.signal_model.model[1][label].update(values=new_values)
                idx += num_values - 1
            # Update the flash model
            self.flash.update(
                min_value_aq=dofs[-4],
                max_value_aq=dofs[-4] + dofs[-3],
                min_value_g=dofs[-2],
                max_value_g=dofs[-1] + dofs[-2],
            )
            logging.info(
                f"Flash min: {self.flash.min_value_aq}, max: {self.flash.max_value_aq}, min g: {self.flash.min_value_g}, max g: {self.flash.max_value_g}"
            )

            # Allocate variables for mass analysis
            times = {r: [] for r in rois}
            expected_mass = {r: [] for r in rois}
            integrated_mass = {r: [] for r in rois}
            integrated_mass_g = {r: [] for r in rois}
            integrated_mass_aq = {r: [] for r in rois}
            error = 0.0

            analysis = {
                roi_key: SimpleRunAnalysis(self.geometry.subregion(roi))
                for roi_key, roi in rois.items()
            }
            for img, color_interpretation in zip(images, color_interpretations):
                # Update thermodynamic state
                state = experiment.pressure_temperature_protocol.get_state(img.date)
                gradient = experiment.pressure_temperature_protocol.get_gradient(
                    img.date
                )
                self.co2_mass_analysis.update_state(
                    atmospheric_pressure=state.pressure,
                    atmospheric_temperature=state.temperature,
                    atmospheric_pressure_gradient=gradient.pressure,
                    atmospheric_temperature_gradient=gradient.temperature,
                )

                # Signal analysis
                pH = self.call_pH_analysis(color_interpretation)

                # Mass computation
                mass_analysis_result = self.call_flash_and_mass_analysis(pH)

                for roi_key, roi in rois.items():
                    # Subregion mass
                    roi_mass_analysis_result = mass_analysis_result.subregion(roi)

                    # Compute expected mass
                    assert isinstance(roi, darsia.CoordinateArray)
                    roi_exact_mass = experiment.injection_protocol.injected_mass(
                        img.date, roi
                    )

                    # Track result
                    analysis[roi_key].track(
                        roi_mass_analysis_result, exact_mass=roi_exact_mass
                    )

            # Monitor mass evolution over time
            for roi_key in rois:
                times[roi_key] = analysis[roi_key].data.time
                expected_mass[roi_key] = analysis[roi_key].data.exact_mass_tot
                integrated_mass[roi_key] = analysis[roi_key].data.mass_tot
                integrated_mass_g[roi_key] = analysis[roi_key].data.mass_g
                integrated_mass_aq[roi_key] = analysis[roi_key].data.mass_aq

                # Errors
                _error = np.abs(
                    np.array(integrated_mass[roi_key])
                    - np.array(expected_mass[roi_key])
                ) / np.abs(expected_mass[roi_key]).clip(min=1e-8)

                logging.info(
                    f"Mass analysis for roi {roi_key} updated.\nNew relative error: {_error}"
                )
                error += np.sum(_error)
            logging.info(f"Current total error: {error}")

            return error

        # ! ---- OPTIMIZATION ----
        # Use Nelder-Mead method for derivative-free optimization
        # Configure comprehensive optimization options with detailed explanations
        optimization_options = {
            "maxiter": 10,  # Maximum number of iterations to prevent infinite loops
            # "maxfev": 5000,  # Maximum function evaluations (safety limit)
            "disp": True,  # Display convergence messages during optimization
            "xtol": 1e-6,  # Absolute tolerance for parameter convergence
            "ftol": 1e-6,  # Absolute tolerance for function value convergence
            # "adaptive": True,  # Use adaptive algorithm parameters for better performance
            # "return_all": False,  # Don't return all intermediate results (saves memory)
        }

        # Create initial simplex for Nelder-Mead method with custom increment sizes
        # The simplex needs n+1 vertices where n is the number of parameters
        n_params = len(initial_dofs)
        initial_simplex = np.zeros((n_params + 1, n_params))

        # First vertex is the initial guess
        initial_simplex[0] = initial_dofs

        # Create remaining vertices by adding increments to each parameter
        for i in range(n_params):
            initial_simplex[i + 1] = initial_dofs.copy()

            # Use different increment sizes based on parameter type
            if i < n_params - 2:  # Signal function parameters
                increment = 0.2
            else:  # Last two parameters (flash cut-off and max_value)
                increment = 0.01

            initial_simplex[i + 1, i] += increment

        logging.info(f"Initial simplex shape: {initial_simplex.shape}")
        logging.info(f"Initial simplex:\n{initial_simplex}")

        # Add initial simplex to optimization options
        optimization_options["initial_simplex"] = initial_simplex

        result = minimize(
            objective_function,
            initial_dofs,
            method="Nelder-Mead",
            # method="Powell",
            # method="L-BFGS-B",
            bounds=[(0, 1)] * len(initial_dofs),
            options=optimization_options,
        )
        logging.info(f"Optimization result: {result}")

        # Final update of the signal model with optimized DOFs
        idx = 0
        for label in available_labels:
            num_values = len(self.signal_model.model[1][label].values)
            new_values = np.cumsum(
                np.hstack(
                    [
                        0.0,
                        result.x[idx : idx + num_values - 1],
                    ]
                )
            )
            logging.info(f"Label {label}: {new_values}")
            self.signal_model.model[1][label].update(values=new_values)
            idx += num_values - 1
        # Update the flash model
        self.flash.update(
            min_value_aq=result.x[-4],
            max_value_aq=result.x[-4] + result.x[-3],
            min_value_g=result.x[-2],
            max_value_g=result.x[-1] + result.x[-2],
        )
        logging.info(
            f"Flash min: {self.flash.min_value_aq}, max: {self.flash.max_value_aq}, min g: {self.flash.min_value_g}, max g: {self.flash.max_value_g}"
        )

    def save(self, folder: Path) -> None:
        """Save the calibration data to json file.

        Args:
            path (Path): The path to save the calibration data.

        """
        # Save the color path interpretation
        for label in np.unique(self.labels.img):
            if label < 0:
                continue
            self.color_path_interpretation[label].save(
                folder
                / "color_path_interpretation"
                / f"color_path_interpretation_{label}"
            )

            # Save the signal model (color interpretation to signal)
            self.signal_model.model[1][label].save(
                folder / "signal_model" / f"signal_model_{label}"
            )

        # Save the flash
        self.flash.save(folder / "flash" / "flash")

        # Save remaining meta data (implicit and explicit)
        color_mode = (
            darsia.ColorMode.ABSOLUTE
            if self.color_analysis.base is None
            else darsia.ColorMode.RELATIVE
        )
        ignore_labels = [
            int(label) for label in self.signal_model.model[1].ignore_labels
        ]

        metadata = {
            "color_mode": color_mode,
            "ignore_labels": ignore_labels,
        }
        with open(folder / "metadata.json", "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(
        cls,
        folder: Path,
        baseline: darsia.Image,
        labels: darsia.Image,
        co2_mass_analysis: darsia.CO2MassAnalysis,
        geometry: darsia.ExtrudedPorousGeometry,
        restoration: darsia.Model | None = None,
    ) -> "HeterogeneousColorToMassAnalysis":
        """Load the calibration data from json file.

        Args:
            path (Path): path to load the model

        """
        color_path_interpretation = {
            label: darsia.ColorPathInterpolation.load(
                folder
                / "color_path_interpretation"
                / f"color_path_interpretation_{label}"
            )
            for label in np.unique(labels.img)
            if label >= 0
        }
        signal_functions = {
            label: darsia.PWTransformation.load(
                folder / "signal_model" / f"signal_model_{label}"
            )
            for label in np.unique(labels.img)
            if label >= 0
        }
        flash = darsia.SimpleFlash.load(folder / "flash" / "flash")

        metadata_path = folder / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            color_mode = darsia.ColorMode(metadata["color_mode"])
            ignore_labels = metadata["ignore_labels"]

        return cls(
            baseline=baseline,
            labels=labels,
            color_mode=color_mode,
            color_path_interpretation=color_path_interpretation,
            signal_functions=signal_functions,
            flash=flash,
            co2_mass_analysis=co2_mass_analysis,
            geometry=geometry,
            restoration=restoration,
            ignore_labels=ignore_labels,
        )
