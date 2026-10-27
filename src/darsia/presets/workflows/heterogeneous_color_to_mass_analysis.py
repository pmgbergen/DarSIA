# TODO review noqa F824 usage

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox
from matplotlib.widgets import Button, Slider
from scipy.optimize import minimize

import darsia
from darsia.presets.workflows.analysis.expert_knowledge import ExpertKnowledgeAdapter
from darsia.presets.workflows.basis import label_ids_from_image
from darsia.presets.workflows.calibration.metadata import (
    read_calibration_metadata,
    validate_basis_metadata,
    write_calibration_metadata,
)
from darsia.presets.workflows.simple_run_analysis import (
    SimpleMassAnalysisResults,
    SimpleRunAnalysis,
)
from darsia.signals.color.color_embedding import (
    ColorEmbeddingBasis,
    parse_color_embedding_basis,
)
from darsia.utils.units import (
    choose_mass_unit,
    choose_time_unit,
    convert_mass,
    convert_time,
    mass_label,
    time_label,
)

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
        basis: ColorEmbeddingBasis = ColorEmbeddingBasis.LABELS,
        expert_knowledge_adapter: ExpertKnowledgeAdapter | None = None,
        contour_smoother: darsia.ContourSmoother | None = None,
        mass_unit: str = "auto",
        time_unit: str = "auto",
    ):
        self.mass_unit = mass_unit
        """Unit for mass axes/values in calibration plots ('auto' or a fixed unit)."""
        self.time_unit = time_unit
        """Unit for time axes/values in calibration plots ('auto' or a fixed unit)."""

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
            restoration=None,
            model=base_model,
            labels=labels,
            **config,
        )

        # Labels may have different numbers of key colors, but all share the domain
        # [0, 1] (equidistant abscissas), so the intersection of domains is nonempty.
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
            base=None,
            restoration=restoration,
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

        self.color_path_interpretation = color_path_interpretation
        """Color path interpretations for different labels."""
        self.basis = parse_color_embedding_basis(basis)
        """Label-space basis used during calibration."""
        self.expert_knowledge_adapter = expert_knowledge_adapter
        """Optional adapter constraining saturation/concentration via expert ROIs."""
        self.contour_smoother = contour_smoother
        """Optional contour smoother for better visualization of ROIs during calibration."""

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
        if self.expert_knowledge_adapter is not None:
            c_aq = self.expert_knowledge_adapter.apply(c_aq, "concentration_aq")
            s_g = self.expert_knowledge_adapter.apply(s_g, "saturation_g")
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

        Parameters
        ----------
        images : list[darsia.Image]
            The images to use for calibration.
        experiment : darsia.ProtocolledExperiment
            The protocolled experiment.
        rois : dict[str, darsia.VoxelArray or darsia.CoordinateArray], optional
            Named regions of interest; defaults to the entire frame.
        cmap : str or matplotlib.colors.Colormap, optional
            Optional colormap for visualization.
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
        coarse_color_interpretations = [
            darsia.resize(color_interpretation, shape=coarse_shape)
            for color_interpretation in color_interpretations
        ]

        # ! ----- IMAGE SELECTION -----

        def image_idx_selector() -> int:
            """Interactive selection of plotted images - allow to click for showing the next
            coarse images - keep track of the idx in the list and return when finishing.

            """
            nonlocal image_idx  # noqa: F824

            # Create a simple figure for image selection
            fig, ax = plt.subplots(figsize=(8, 6))
            current_image = coarse_images[image_idx]
            img_display = ax.imshow(current_image.img)
            ax.set_title(
                f"""Image {image_idx + 1}/{len(images)} - Click to cycle through """
                f"""images, close window to finish"""
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
                        f"""Image {image_idx + 1}/{len(images)} - Click to cycle through """
                        f"""images, close window to finish"""
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
            pH = self.call_pH_analysis(_color_interpretation)
            c_aq, s_g = self.flash(pH)
            if self.expert_knowledge_adapter is not None:
                c_aq = self.expert_knowledge_adapter.apply(c_aq, "concentration_aq")
                s_g = self.expert_knowledge_adapter.apply(s_g, "saturation_g")
            mass_analysis_result: SimpleMassAnalysisResults = (
                self.co2_mass_analysis.mass_analysis(
                    c_aq=c_aq,
                    s_g=s_g,
                )
            )
            density = mass_analysis_result.mass
            return pH, density, c_aq, s_g

        # ! ---- DISPLAY HELPERS ----

        def compute_maps() -> dict:
            """Analyze the current image at full resolution, coarsened for display.

            The thermodynamic state is set for the date of the current image.

            """
            date = images[image_idx].date
            state = experiment.pressure_temperature_protocol.get_state(date)
            gradient = experiment.pressure_temperature_protocol.get_gradient(date)
            self.co2_mass_analysis.update_state(
                atmospheric_pressure=state.pressure,
                atmospheric_temperature=state.temperature,
                atmospheric_pressure_gradient=gradient.pressure,
                atmospheric_temperature_gradient=gradient.temperature,
            )
            pH, density, c_aq, s_g = detailed_mass_analysis(
                color_interpretations[image_idx]
            )
            co2_g_density = darsia.full_like(
                pH, self.co2_mass_analysis.density_gaseous_co2
            )
            co2_aq_solubility = darsia.full_like(
                pH, self.co2_mass_analysis.solubility_co2
            )

            def coarsen(img: darsia.Image) -> darsia.Image:
                return darsia.resize(img, shape=coarse_shape)

            return {
                "image_idx": image_idx,
                "pH": pH,
                "c_aq": coarsen(c_aq),
                "s_g": coarsen(s_g),
                "signals": {
                    "color": coarse_color_interpretations[image_idx],
                    "pH": coarsen(pH),
                    "rho_co2": coarsen(density),
                    "c_aq": coarsen(c_aq),
                    "s_g": coarsen(s_g),
                    "labels": coarse_labels,
                    "depth": coarse_depth,
                    "rho_co2(g)": coarsen(co2_g_density),
                    "solubility_co2(aq)": coarsen(co2_aq_solubility),
                    "porosity": coarse_porosity,
                    "effective_depth": coarse_effective_depth,
                },
            }

        def contour_overlay(
            coarse_image: darsia.Image,
            coarse_c_aq: darsia.Image,
            coarse_s_g: darsia.Image,
            thresholds: list[float],
        ) -> darsia.Image:
            """Contours of the phase segmentation on top of the coarse image."""
            return darsia.plot_contour_on_image(
                coarse_image,
                mask=[
                    coarse_c_aq > thresholds[0],
                    coarse_c_aq > 0.1,
                    coarse_c_aq > 0.25,
                    coarse_c_aq > 0.5,
                    coarse_c_aq > 0.9,
                    coarse_s_g > thresholds[1],
                ],
                contour_smoother=self.contour_smoother,
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

        def cached_contour_overlay(maps: dict, thresholds: list[float]) -> darsia.Image:
            """Contour overlay of the maps; recomputed only for new thresholds."""
            key = tuple(float(threshold) for threshold in thresholds)
            if maps.get("contour_key") != key:
                maps["contour"] = contour_overlay(
                    coarse_images[maps["image_idx"]],
                    maps["c_aq"],
                    maps["s_g"],
                    thresholds,
                )
                maps["contour_key"] = key
            return maps["contour"]

        def label_highlight(coarse_image: darsia.Image, label: int) -> np.ndarray:
            """Coarse image with everything outside the label turned (mostly) gray."""
            highlighted = coarse_image.img.copy()
            outside = coarse_labels.img != label
            highlighted[outside] = (
                0.7 * np.mean(highlighted[outside], axis=-1, keepdims=True)
                + 0.3 * coarse_image.img[outside]
            )
            return highlighted

        # ! ---- MASS ANALYSIS SETUP ----

        @darsia.timing_decorator
        def update_mass_analysis() -> None:
            """Auxiliary function to update multiphase time series analysis."""
            nonlocal times  # noqa: F824
            nonlocal expected_mass  # noqa: F824
            nonlocal integrated_mass  # noqa: F824
            nonlocal integrated_mass_g  # noqa: F824
            nonlocal integrated_mass_aq  # noqa: F824

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
                    f"""Mass analysis for roi {roi_key} updated.\nNew relative """
                    f"""error: {relative_error}"""
                )

        # ! ---- INITIAL STATE (restored by the Reset button) ----

        initial_values = {
            label: np.array(self.signal_model.model[1][label].values, dtype=float)
            for label in self.signal_model.model[1].keys()
        }
        initial_flash = (
            self.flash.min_value_aq,
            self.flash.max_value_aq,
            self.flash.min_value_g,
            self.flash.max_value_g,
        )
        default_contour_threshold = 0.05
        # Current contour thresholds (c_aq, s_g); kept when the figure is rebuilt
        contour_thresholds = [default_contour_threshold, default_contour_threshold]

        # ! ---- GRID FOR PLOTTING ----

        # Interactive tuning of values for each color path (chosen by user)
        done_calibration = False
        need_to_pick_new_image = True
        need_to_pick_new_label = True
        # Label-independent results are kept when the figure is rebuilt (e.g. for a
        # new label): the maps of the current image, and the mass series. The maps
        # are replaced whenever the image or the parameters change (refresh), and
        # the mass series whenever the parameters change (update_analysis).
        maps = None
        mass_analysis_done = False
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

            # Fetch the current image and analyze it
            coarse_image = coarse_images[image_idx]
            if maps is None or maps["image_idx"] != image_idx:
                maps = compute_maps()

            # Pick label interactively
            if need_to_pick_new_label:
                label_idx = label_idx_selector(maps["pH"])
                need_to_pick_new_label = False

            # Compute mass analysis
            if not mass_analysis_done:
                update_mass_analysis()
                mass_analysis_done = True

            coarse_signal = maps["signals"][signal_option_ptr[signal_option_idx]]
            coarse_contour = cached_contour_overlay(maps, contour_thresholds)

            # Use gridspec to create a 2x3 grid with proper ratios
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
            coarse_image_img = ax_coarse_image.imshow(
                label_highlight(coarse_image, label_idx)
            )

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
                ax_signal_function.imshow(
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

                # Find the corresponding x-values for the cut-off and max value lines as
                # solution to the signal_function(x) = cut_off and
                # signal_function(x) = max_value
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
                ax_signal_function.text(
                    x=aq_max_x - 0.05,
                    y=0.04,
                    s="CO2(aq)",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="k",
                )
                ax_signal_function.text(
                    x=aq_max_x + 0.05,
                    y=0.04,
                    s="CO2(g)",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="k",
                )
                ax_signal_function.text(
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

            # Determine a representative mass/time magnitude up front, so the
            # unit (fixed, or auto-chosen to keep displayed numbers readable)
            # is known before any series is drawn.
            max_expected = max(
                (max(expected_mass[k]) for k in rois if len(expected_mass[k]) > 0),
                default=0.0,
            )
            max_time = max(
                (max(times[k]) for k in rois if len(times[k]) > 0), default=0.0
            )
            mass_unit = (
                choose_mass_unit(max_expected)
                if self.mass_unit == "auto"
                else self.mass_unit
            )
            time_unit = (
                choose_time_unit(max_time)
                if self.time_unit == "auto"
                else self.time_unit
            )
            ax_mass.set_xlabel(time_label(time_unit))
            ax_mass.set_ylabel(mass_label(mass_unit))

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
            import matplotlib

            _cmap = matplotlib.colormaps["tab10" if num_rois <= 10 else "tab20"]
            roi_colors = [_cmap(i / max(1, num_rois - 1)) for i in range(num_rois)]

            # Plot each ROI as separate series
            for i, (roi_key, roi_times) in enumerate(times.items()):
                if len(roi_times) > 0:
                    color = roi_colors[i % len(roi_colors)]
                    roi_label = roi_key if roi_key else "entire_frame"
                    plot_times = convert_time(roi_times, time_unit)

                    # Plot lines
                    [integrated_mass_plots[roi_key]] = ax_mass.plot(
                        plot_times,
                        convert_mass(integrated_mass[roi_key], mass_unit),
                        color=color,
                        label=f"total ({roi_label})",
                        linewidth=2,
                    )
                    [integrated_mass_plots_g[roi_key]] = ax_mass.plot(
                        plot_times,
                        convert_mass(integrated_mass_g[roi_key], mass_unit),
                        color=color,
                        label=f"gas ({roi_label})",
                        linestyle=":",
                        linewidth=2,
                    )
                    [integrated_mass_plots_aq[roi_key]] = ax_mass.plot(
                        plot_times,
                        convert_mass(integrated_mass_aq[roi_key], mass_unit),
                        color=color,
                        label=f"aqueous ({roi_label})",
                        linestyle="-.",
                        linewidth=2,
                    )
                    [expected_mass_plots[roi_key]] = ax_mass.plot(
                        plot_times,
                        convert_mass(expected_mass[roi_key], mass_unit),
                        linestyle="--",
                        color=color,
                        label=f"injected ({roi_label})",
                        alpha=0.7,
                    )

                    # Plot scatter points
                    integrated_mass_scatters[roi_key] = ax_mass.scatter(
                        plot_times,
                        convert_mass(integrated_mass[roi_key], mass_unit),
                        color=color,
                        s=20,
                        alpha=0.8,
                    )
                    integrated_mass_scatters_g[roi_key] = ax_mass.scatter(
                        plot_times,
                        convert_mass(integrated_mass_g[roi_key], mass_unit),
                        color=color,
                        s=15,
                        alpha=0.6,
                        marker="^",
                    )
                    integrated_mass_scatters_aq[roi_key] = ax_mass.scatter(
                        plot_times,
                        convert_mass(integrated_mass_aq[roi_key], mass_unit),
                        color=color,
                        s=15,
                        alpha=0.6,
                        marker="s",
                    )

            # Set y-axis limits based on maximum expected mass across all ROIs
            if max_expected > 0:
                ax_mass.set_ylim(0.0, convert_mass(2.0 * max_expected, mass_unit))
            else:
                ax_mass.autoscale()
            ax_mass.legend()
            ax_mass.grid(True, alpha=0.3)

            # ! ---- PARAMETER CONTROLS ----

            # Grouped rows (label | slider | value box). The number of signal value
            # rows follows the number of key colors of the selected label.
            controls_box = ax_sliders.get_position()
            signal_values = self.signal_model.model[1][label_idx].values
            path_colors = self.color_path_interpretation[label_idx].color_path.colors
            num_values = len(signal_values)
            row_height = controls_box.height / (num_values + 9)

            def make_row(row, name, valinit, vmin, vmax, color):
                """Create a labeled slider (with value text) in a row."""
                x0, width = controls_box.x0, controls_box.width
                y0 = controls_box.y1 - (row + 1) * row_height
                fig.text(
                    x0, y0 + 0.5 * row_height, name, fontsize=8, ha="left", va="center"
                )
                ax_row_slider = fig.add_axes(
                    [
                        x0 + 0.30 * width,
                        y0 + 0.2 * row_height,
                        0.55 * width,
                        0.6 * row_height,
                    ]
                )
                slider = Slider(
                    ax_row_slider,
                    "",
                    vmin,
                    vmax,
                    valinit=valinit,
                    valstep=0.01,
                    color=color,
                )
                slider.valtext.set_fontsize(8)
                return slider

            def section_title(row, title):
                fig.text(
                    controls_box.x0,
                    controls_box.y1 - (row + 0.5) * row_height,
                    title,
                    fontsize=9,
                    fontweight="bold",
                    ha="left",
                    va="center",
                )

            section_title(0, f"Signal values (label {label_idx})")
            sliders_color_to_signal = [
                make_row(
                    1 + i,
                    f"Value {i}",
                    val,
                    0.0,
                    2.0,
                    np.clip(path_colors[i], 0, 1) if i < len(path_colors) else "gray",
                )
                for i, val in enumerate(signal_values)
            ]

            flash_row = num_values + 1
            section_title(flash_row, "Flash")
            sliders_flash = [
                make_row(flash_row + 1 + i, name, value, 0.0, 2.0, "darkgray")
                for i, (name, value) in enumerate(
                    [
                        ("Min aq", self.flash.min_value_aq),
                        ("Max aq", self.flash.max_value_aq),
                        ("Min g", self.flash.min_value_g),
                        ("Max g", self.flash.max_value_g),
                    ]
                )
            ]

            contour_row = flash_row + 5
            section_title(contour_row, "Contours")
            sliders_threshold = [
                make_row(
                    contour_row + 1,
                    "Threshold c_aq",
                    contour_thresholds[0],
                    0.0,
                    1.0,
                    (1, 0, 0),
                ),
                make_row(
                    contour_row + 2,
                    "Threshold s_g",
                    contour_thresholds[1],
                    0.0,
                    1.0,
                    (0, 1, 0),
                ),
            ]

            for i, slider in enumerate(sliders_threshold):
                slider.on_changed(
                    lambda val, i=i: contour_thresholds.__setitem__(i, val)
                )

            # Slider updates only repaint the sliders (blitting). Otherwise, every
            # slider step re-renders the entire figure, including all images. The
            # moving parts of the sliders are animated artists, which are excluded
            # from the stored background and drawn on top of it.
            all_sliders = sliders_color_to_signal + sliders_flash + sliders_threshold
            animated_artists = [
                artist
                for slider in all_sliders
                for artist in (slider.poly, slider._handle, slider.valtext)
            ]
            for artist in animated_artists:
                artist.set_animated(True)
            background = None

            def store_background(event):
                nonlocal background
                background = fig.canvas.copy_from_bbox(fig.bbox)
                for artist in animated_artists:
                    fig.draw_artist(artist)

            fig.canvas.mpl_connect("draw_event", store_background)

            def blit_slider(slider):
                if background is None or not fig.canvas.supports_blit:
                    fig.canvas.draw_idle()
                    return
                handle_radius = 0.5 * slider._handle.get_markersize() * fig.dpi / 72
                region = Bbox.union(
                    [slider.ax.bbox, slider.valtext.get_window_extent()]
                ).padded(handle_radius + 1)
                fig.canvas.restore_region(background)
                for artist in animated_artists:
                    fig.draw_artist(artist)
                fig.canvas.blit(region)

            for slider in all_sliders:
                slider.drawon = False
                slider.on_changed(lambda val, slider=slider: blit_slider(slider))

            # Position buttons at the top of the figure
            button_width = 0.08
            button_height = 0.04
            button_y = 0.95
            button_spacing = 0.01

            # Button order from left to right (Reset, Update at bottom right):
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

            # Add "Reset" button to the left of "Update"
            reset_button_x = update_button_x - button_width - button_spacing
            ax_reset = plt.axes(
                [reset_button_x, update_button_y, button_width, button_height]
            )
            btn_reset = Button(ax_reset, "Reset")

            # Add "Update" button
            ax_update = plt.axes(
                [
                    update_button_x,
                    update_button_y,
                    button_width,
                    button_height,
                ]
            )
            btn_update = Button(ax_update, "Update")

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
                    nonlocal done_tuning_values  # noqa: F824
                    nonlocal done_calibration  # noqa: F824

                    def update_signal_function_plot():
                        """Update the signal function plot (curve, flash lines)."""
                        # Update signal function plot
                        if signal_function_line is not None:
                            signal_func = self.signal_model.model[1][_label_idx]
                            x_vals = signal_func.supports
                            y_vals = signal_func.values.copy()
                            signal_function_line.set_data(x_vals, y_vals)
                            signal_function_scatter.set_offsets(np.c_[x_vals, y_vals])

                            # Rescale y-axis to fit the curve, floored at the
                            # current flash max (never zoom in tighter than the
                            # flash thresholds, but grow if the curve exceeds
                            # them)
                            flash_max = max(
                                self.flash.max_value_aq, self.flash.max_value_g
                            )
                            ax_signal_function.set_ylim(
                                self.signal_model_extents[1][0],
                                max(flash_max, float(np.max(y_vals))),
                            )

                        # Update flash cut-off and max value lines
                        signal_function_aq_min_y.set_ydata([self.flash.min_value_aq])
                        signal_function_aq_max_y.set_ydata([self.flash.max_value_aq])
                        signal_function_g_max_y.set_ydata([self.flash.max_value_g])
                        signal_function_g_min_y.set_ydata([self.flash.min_value_g])
                        aq_max_x = signal_func.inverse(self.flash.max_value_aq)
                        aq_min_x = signal_func.inverse(self.flash.min_value_aq)
                        g_max_x = signal_func.inverse(self.flash.max_value_g)
                        g_min_x = signal_func.inverse(self.flash.min_value_g)
                        signal_function_aq_max_x.set_xdata([aq_max_x])
                        signal_function_aq_min_x.set_xdata([aq_min_x])
                        signal_function_g_max_x.set_xdata([g_max_x])
                        signal_function_g_min_x.set_xdata([g_min_x])

                        # Update the position of the annotaion
                        ax_signal_function.texts[0].set_position(
                            (aq_max_x - 0.05, 0.04)
                        )
                        ax_signal_function.texts[1].set_position(
                            (aq_max_x + 0.05, 0.04)
                        )
                        ax_signal_function.texts[2].set_position((g_max_x + 0.1, 0.04))

                    def update_mass_plot():
                        """Update the mass evolution plots."""
                        # Update mass plots
                        max_expected = max(
                            (
                                max(expected_mass[k])
                                for k in rois
                                if len(expected_mass[k]) > 0
                            ),
                            default=0.0,
                        )
                        max_time = max(
                            (max(times[k]) for k in rois if len(times[k]) > 0),
                            default=0.0,
                        )
                        mass_unit = (
                            choose_mass_unit(max_expected)
                            if self.mass_unit == "auto"
                            else self.mass_unit
                        )
                        time_unit = (
                            choose_time_unit(max_time)
                            if self.time_unit == "auto"
                            else self.time_unit
                        )
                        for roi_key in rois:
                            plot_times = convert_time(times[roi_key], time_unit)
                            integrated_mass_plots[roi_key].set_data(
                                plot_times,
                                convert_mass(integrated_mass[roi_key], mass_unit),
                            )
                            integrated_mass_plots_g[roi_key].set_data(
                                plot_times,
                                convert_mass(integrated_mass_g[roi_key], mass_unit),
                            )
                            integrated_mass_plots_aq[roi_key].set_data(
                                plot_times,
                                convert_mass(integrated_mass_aq[roi_key], mass_unit),
                            )
                            integrated_mass_scatters[roi_key].set_offsets(
                                np.c_[
                                    plot_times,
                                    convert_mass(integrated_mass[roi_key], mass_unit),
                                ]
                            )
                            integrated_mass_scatters_g[roi_key].set_offsets(
                                np.c_[
                                    plot_times,
                                    convert_mass(integrated_mass_g[roi_key], mass_unit),
                                ]
                            )
                            integrated_mass_scatters_aq[roi_key].set_offsets(
                                np.c_[
                                    plot_times,
                                    convert_mass(
                                        integrated_mass_aq[roi_key], mass_unit
                                    ),
                                ]
                            )
                        if max_expected > 0:
                            ax_mass.set_ylim(
                                0.0, convert_mass(2.0 * max_expected, mass_unit)
                            )
                        else:
                            ax_mass.autoscale()
                        ax_mass.set_xlabel(time_label(time_unit))
                        ax_mass.set_ylabel(mass_label(mass_unit))

                    def refresh(recompute=True, signal_function=False, mass_plot=False):
                        """Update the displayed panels.

                        Parameters
                        ----------
                        recompute : bool
                            Analyze the current image anew (after changes of the
                            parameters or the image); otherwise reuse the last maps.
                        signal_function : bool
                            Also update the signal function plot.
                        mass_plot : bool
                            Also update the mass evolution plot.

                        """
                        nonlocal maps
                        if recompute:
                            maps = compute_maps()
                        current_image = coarse_images[image_idx]

                        # Image panel and signal panel
                        ax_coarse_image.set_title(
                            f"Current Image {image_idx} and Label {_label_idx}"
                        )
                        coarse_image_img.set_data(
                            label_highlight(current_image, _label_idx)
                        )
                        coarse_signal = maps["signals"][
                            signal_option_ptr[signal_option_idx]
                        ]
                        coarse_signal_img.set_data(coarse_signal.img)
                        ax_signal.set_title(
                            signal_options[signal_option_ptr[signal_option_idx]]
                        )
                        if signal_option_ptr[signal_option_idx] == "pH":
                            signal_min = 0.0
                            signal_max = 1.0
                        else:
                            signal_min = np.nanmin(coarse_signal.img)
                            signal_max = np.nanmax(coarse_signal.img)
                        coarse_signal_img.set_clim(vmin=signal_min, vmax=signal_max)
                        coarse_signal_colorbar.update_normal(coarse_signal_img)

                        # Phase segmentation panel
                        coarse_contour_img.set_data(
                            cached_contour_overlay(
                                maps, [slider.val for slider in sliders_threshold]
                            ).img
                        )

                        if signal_function and signal_function_line is not None:
                            update_signal_function_plot()
                        if mass_plot:
                            update_mass_plot()
                        fig.canvas.draw_idle()

                    def update_analysis(event=None):
                        """Apply the slider values and update everything."""
                        # Check if any slider value has changed that requires updating
                        # mass analysis.
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
                        # Update mass analysis
                        if need_update:
                            update_mass_analysis()
                        refresh(recompute=True, signal_function=True, mass_plot=True)

                    # Connect button callbacks
                    btn_update.on_clicked(update_analysis)

                    def reset(event):
                        """Restore the initial values of this label, flash, contours."""
                        for slider, value in zip(
                            sliders_color_to_signal, initial_values[_label_idx]
                        ):
                            slider.set_val(value)
                        for slider, value in zip(sliders_flash, initial_flash):
                            slider.set_val(value)
                        for slider in sliders_threshold:
                            slider.set_val(default_contour_threshold)
                        update_analysis()

                    btn_reset.on_clicked(reset)

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
                        refresh(recompute=False)

                    btn_next_signal.on_clicked(next_signal)

                    def next_label(event):
                        """Move to the next label and rebuild the figure for it."""
                        nonlocal label_idx, done_tuning_values, need_to_pick_new_label
                        available_labels = np.sort(
                            list(self.signal_model.model[1].keys())
                        )
                        available_labels = available_labels[available_labels >= 0]
                        position = np.where(available_labels == label_idx)[0]
                        next_position = (
                            (position[0] + 1) % len(available_labels)
                            if len(position) > 0
                            else 0
                        )
                        label_idx = available_labels[next_position]
                        need_to_pick_new_label = False
                        done_tuning_values = True
                        plt.close("all")

                    btn_next_label.on_clicked(next_label)

                    def next_image(event):
                        nonlocal image_idx
                        image_idx = (image_idx + 1) % len(images)
                        refresh(recompute=True)

                    btn_next_image.on_clicked(next_image)

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

        Parameters
        ----------
        images : list[darsia.Image]
            The images to use for calibration.
        experiment : darsia.ProtocolledExperiment
            The protocolled experiment.
        rois : dict[str, darsia.VoxelArray or darsia.CoordinateArray], optional
            Named regions of interest; defaults to the entire frame.
        cmap : str or matplotlib.colors.Colormap, optional
            Optional colormap for visualization.
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
                f"""Flash min: {self.flash.min_value_aq}, max: {self.flash.max_value_aq}, """
                f"""min g: {self.flash.min_value_g}, max g: {self.flash.max_value_g}"""
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
            f"""Flash min: {self.flash.min_value_aq}, max: {self.flash.max_value_aq}, """
            f"""min g: {self.flash.min_value_g}, max g: {self.flash.max_value_g}"""
        )

    def save(self, folder: Path) -> None:
        """Save the calibration data to json file.

        Parameters
        ----------
        folder : Path
            The folder to save the calibration data to.
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

        label_ids = label_ids_from_image(self.labels)
        metadata = {
            "color_mode": color_mode,
            "ignore_labels": ignore_labels,
            "basis": self.basis.value,
            "label_ids": label_ids,
        }
        write_calibration_metadata(
            folder / "metadata.json",
            basis=self.basis,
            label_ids=label_ids,
            extra=metadata,
        )

    @staticmethod
    def _resample_signal_functions(
        signal_functions: dict[int, darsia.PWTransformation],
        color_path_interpretation: dict[int, darsia.ColorPathInterpolation],
    ) -> None:
        """Match stored signal functions to the current color paths (in place).

        The number of key colors can differ per label and change when color paths
        are recalibrated. Mismatching signal functions are resampled onto the
        supports of the current color path.

        """
        for label, interpretation in color_path_interpretation.items():
            function = signal_functions.get(label)
            if function is None:
                continue
            supports = np.asarray(interpretation.values, dtype=float)
            old_supports = np.asarray(function.supports, dtype=float)
            if len(old_supports) == len(supports) and np.allclose(
                old_supports, supports
            ):
                continue
            logger.warning(
                f"Label {label}: stored signal model has {len(old_supports)} values, "
                f"but the color path has {len(supports)} key colors; resampling."
            )
            values = np.interp(supports, old_supports, function.values)
            signal_functions[label] = darsia.PWTransformation(
                supports, np.maximum.accumulate(values)
            )

    @classmethod
    def load(
        cls,
        folder: Path,
        baseline: darsia.Image,
        labels: darsia.Image,
        co2_mass_analysis: darsia.CO2MassAnalysis,
        geometry: darsia.ExtrudedPorousGeometry,
        restoration: darsia.Model | None = None,
        basis: ColorEmbeddingBasis = ColorEmbeddingBasis.LABELS,
        expert_knowledge_adapter: ExpertKnowledgeAdapter | None = None,
        contour_smoother: darsia.ContourSmoother | None = None,
        mass_unit: str = "auto",
        time_unit: str = "auto",
        color_path_interpretation: (
            dict[int, darsia.ColorPathInterpolation] | None
        ) = None,
    ) -> "HeterogeneousColorToMassAnalysis":
        """Load the calibration data from json file.

        Parameters
        ----------
        folder : Path
            Folder to load the model from.
        baseline : darsia.Image
            Baseline image.
        labels : darsia.Image
            Labeled image.
        co2_mass_analysis : darsia.CO2MassAnalysis
            The CO2 mass analysis tool.
        geometry : darsia.ExtrudedPorousGeometry
            The geometry used for mass computation.
        restoration : darsia.Model, optional
            Restoration model applied prior to analysis.
        basis : ColorEmbeddingBasis, optional
            Basis used for the color embedding.
        expert_knowledge_adapter : ExpertKnowledgeAdapter, optional
            Adapter for injecting expert-knowledge corrections.
        contour_smoother : darsia.ContourSmoother, optional
            Optional smoother for the label contours.
        color_path_interpretation : dict, optional
            Current color path interpretations. If provided, they replace the stored
            ones, and stored signal models whose supports do not match the (possibly
            recalibrated) color path of their label are resampled onto its supports.
        """
        stored_interpretation = color_path_interpretation is None
        if color_path_interpretation is None:
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
        if not stored_interpretation:
            cls._resample_signal_functions(signal_functions, color_path_interpretation)
        flash = darsia.SimpleFlash.load(folder / "flash" / "flash")

        metadata_path = folder / "metadata.json"
        metadata = read_calibration_metadata(metadata_path)
        validate_basis_metadata(
            metadata=metadata,
            expected_basis=parse_color_embedding_basis(basis),
            expected_label_ids=label_ids_from_image(labels),
            artifact="color_to_mass",
        )
        color_mode = darsia.ColorMode.RELATIVE
        ignore_labels = []
        parsed_basis = parse_color_embedding_basis(basis)
        if metadata is not None:
            if "color_mode" in metadata:
                color_mode = darsia.ColorMode(metadata["color_mode"])
            if "ignore_labels" in metadata:
                ignore_labels = metadata["ignore_labels"]
            if "basis" in metadata:
                parsed_basis = parse_color_embedding_basis(metadata["basis"])

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
            basis=parsed_basis,
            expert_knowledge_adapter=expert_knowledge_adapter,
            contour_smoother=contour_smoother,
            mass_unit=mass_unit,
            time_unit=time_unit,
        )
