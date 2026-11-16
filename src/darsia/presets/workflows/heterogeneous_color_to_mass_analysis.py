# TODO review noqa F824 usage

import dataclasses
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba_array
from matplotlib.transforms import Bbox
from matplotlib.widgets import Button, Slider

import darsia
import darsia.multiphase.mass_kernels as mass_kernels
import darsia.signals.color.color_path_kernels as color_kernels
import darsia.signals.models.signal_kernels as signal_kernels
from darsia.multiphase.flash import load_flash
from darsia.presets.workflows.analysis.expert_knowledge import ExpertKnowledgeAdapter
from darsia.presets.workflows.basis import label_ids_from_image
from darsia.presets.workflows.calibration.manual_calibration import CalibrationState
from darsia.presets.workflows.calibration.metadata import (
    calibration_inputs,
    read_calibration_metadata,
    validate_basis_metadata,
    validate_interpretation_metadata,
    write_calibration_metadata,
)
from darsia.presets.workflows.config.physics import Interpretation
from darsia.presets.workflows.simple_run_analysis import SimpleMassAnalysisResults
from darsia.signals.color.color_embedding import (
    ColorEmbeddingBasis,
    parse_color_embedding_basis,
)
from darsia.signals.models.evaluation import curve_inverse, signal_curve
from darsia.utils.figure_window import ask_yes_no, maximize_figure_window
from darsia.utils.units import (
    choose_mass_unit,
    choose_time_unit,
    choose_volume_unit,
    convert_mass,
    convert_time,
    convert_volume,
    mass_label,
    time_label,
    volume_label,
)

logger = logging.getLogger(__name__)


def _dot_values(signal_func) -> np.ndarray:
    """Values shown as dots in the signal function plot.

    Key colors which do not influence the signal (unused supports of a kernel) are not
    shown (NaN).

    """
    values = np.asarray(signal_func.values, dtype=float).copy()
    used = getattr(signal_func, "support_indices", None)
    if used is not None:
        values[[i for i in range(len(values)) if i not in used]] = np.nan
    return values


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
        fluid_model: darsia.FluidMassAnalysis,
        geometry: darsia.ExtrudedPorousGeometry,
        restoration: darsia.Model | None = None,
        ignore_labels: list[int] | None = None,
        basis: ColorEmbeddingBasis = ColorEmbeddingBasis.LABELS,
        expert_knowledge_adapter: ExpertKnowledgeAdapter | None = None,
        contour_smoother: darsia.ContourSmoother | None = None,
        mass_unit: str = "auto",
        time_unit: str = "auto",
        evaluation_mode: darsia.EvaluationMode | None = None,
        volume_unit: str = "auto",
    ):
        self.evaluation_mode = (
            darsia.LinearEvaluation() if evaluation_mode is None else evaluation_mode
        )
        """How colors are turned into a signal (stored with the calibration)."""
        self.mass_unit = mass_unit
        """Unit for mass axes/values in calibration plots ('auto' or a fixed unit)."""
        self.volume_unit = volume_unit
        """Unit for volume axes/values in tracer calibration plots ('auto' or fixed)."""
        self.time_unit = time_unit
        """Unit for time axes/values in calibration plots ('auto' or a fixed unit)."""

        self.color_path_interpretation = color_path_interpretation
        """Color path interpretations for different labels."""
        self._baseline = baseline
        self._labels = labels
        self._color_mode = color_mode
        self._restoration = restoration
        self._ignore_labels = ignore_labels
        self._build_stages(signal_functions)

        self.flash = flash
        """Flash model for converting pH to concentrations."""

        self.fluid_model = fluid_model
        """Mass computation tool (fluid model)."""

        self.pressure_temperature_protocol: (
            darsia.PressureTemperatureProtocol | None
        ) = None
        """Protocol setting the state of the fluid model for each analyzed image."""

        self.has_gas = bool(
            getattr(getattr(fluid_model, "system", None), "is_multiphase", True)
        )
        """Whether there is a gas phase; single-phase (tracer) has no flash parameters."""

        self.geometry = geometry
        """Geometry of the experiment - for the integration of mass during calibration."""

        self.basis = parse_color_embedding_basis(basis)
        """Label-space basis used during calibration."""
        self.expert_knowledge_adapter = expert_knowledge_adapter
        """Optional adapter constraining saturation/concentration via expert ROIs."""
        self.contour_smoother = contour_smoother
        """Optional contour smoother for better visualization of ROIs during calibration."""

    def _build_stages(self, signal_functions: dict) -> None:
        """Set up the two stages from colors to the signal (pH).

        Stage 1 (color interpretation) projects the colors onto the color path, if the
        evaluation mode requires it, and otherwise only provides the baseline-relative
        colors. Stage 2 evaluates the signal functions of the labels.

        """
        labels = self._labels
        ignore_labels = self._ignore_labels
        projects = self.evaluation_mode.projects_onto_path

        base_model = (
            darsia.CombinedModel(
                [
                    darsia.HeterogeneousModel(
                        self.color_path_interpretation,
                        labels,
                        ignore_labels=ignore_labels,
                    )
                ]
            )
            if projects
            else None
        )

        # Define general config options
        config = {
            "diff option": "plain",
            "restoration -> model": False,
        }

        # TODO make use of ignore_labels

        # Define general ConcentrationAnalysis.
        self.color_analysis = darsia.ConcentrationAnalysis(
            base=(
                self._baseline
                if self._color_mode == darsia.ColorMode.RELATIVE
                else None
            ),
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

        # NOTE: Without projection, the first model is a placeholder (no clipping), so
        # that the signal functions are always the second model.
        signal_model = darsia.CombinedModel(
            [
                (
                    darsia.ClipModel(
                        min_domain_signal_functions, max_domain_signal_functions
                    )
                    if projects
                    else darsia.ClipModel(-np.inf, np.inf)
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
            restoration=self._restoration,
            model=signal_model,
            labels=labels,
            **config,
        )
        """Model converting color interpretation to pH."""

    def set_evaluation_mode(self, evaluation_mode: darsia.EvaluationMode) -> None:
        """Switch the evaluation mode, keeping the values at the key colors.

        Parameters
        ----------
        evaluation_mode : darsia.EvaluationMode
            New mode; signal functions of all labels are created anew with it.

        """
        if evaluation_mode.same_as(self.evaluation_mode):
            return
        logger.info(
            "Switching the evaluation mode from '%s' to '%s', keeping the signal "
            "values.",
            self.evaluation_mode.type,
            evaluation_mode.type,
        )
        model = self.signal_model.model[1]
        signal_functions = {}
        for label in list(model.keys()):
            interpretation = self.color_path_interpretation[label]
            old = model[label]
            signal_functions[label] = evaluation_mode.signal_function(
                interpretation.color_path,
                interpretation.color_mode,
                old.supports,
                old.values,
                label=label,
            )
        self.evaluation_mode = evaluation_mode
        self._build_stages(signal_functions)

    @property
    def co2_mass_analysis(self) -> darsia.FluidMassAnalysis:
        """Deprecated name of `fluid_model` (read-only), kept for older scripts."""
        return self.fluid_model

    @property
    def labels(self) -> darsia.Image:
        """Labels image - refer to attribute of color analysis."""
        assert self.color_analysis.labels is not None
        return self.color_analysis.labels

    @property
    def supports_reduction(self) -> bool:
        """Whether :meth:`reduced` is available (not with a restoration)."""
        return self._restoration is None

    def reduced(self, factor: int) -> "HeterogeneousColorToMassAnalysis":
        """Copy of the analysis for images reduced by a factor per axis.

        Baseline, depth and porosity are averaged over the reduced pixels, a reduced
        pixel gets the label at its center, and the fluid properties are evaluated on
        the reduced grid. The copy shares the color paths and the flash with this
        analysis, and starts from its signal values (the signal functions are
        copies). Used for a faster, approximate calibration.

        Parameters
        ----------
        factor : int
            Reduction per axis.

        Raises
        ------
        NotImplementedError
            With a restoration, which is set up for the full resolution.

        """
        if not self.supports_reduction:
            raise NotImplementedError(
                "A reduced analysis is not available with a restoration."
            )
        shape = tuple(max(1, int(n) // int(factor)) for n in self.labels.img.shape[:2])

        def average(quantity):
            if isinstance(quantity, darsia.Image):
                return darsia.resize(quantity, shape=shape, interpolation="inter_area")
            return quantity

        def nearest(image):
            if image is None:
                return None
            return darsia.resize(image, shape=shape, interpolation="inter_nearest")

        baseline = average(self._baseline)
        geometry = darsia.ExtrudedPorousGeometry(
            depth=average(self.geometry.depth),
            porosity=average(self.geometry.porosity),
            **baseline.shape_metadata(),
        )
        fluid_model = darsia.FluidMassAnalysis(
            baseline=None if self.fluid_model.baseline is None else baseline,
            atmospheric_pressure=self.fluid_model.atmospheric_pressure,
            atmospheric_temperature=self.fluid_model.atmospheric_temperature,
            atmospheric_pressure_gradient=self.fluid_model.atmospheric_pressure_gradient,
            atmospheric_temperature_gradient=(
                self.fluid_model.atmospheric_temperature_gradient
            ),
            system=self.fluid_model.system,
        )
        expert_knowledge_adapter = self.expert_knowledge_adapter
        if expert_knowledge_adapter is not None:
            # Label and facies based regions are masks at the image resolution
            expert_knowledge_adapter = dataclasses.replace(
                expert_knowledge_adapter,
                labels_image=nearest(expert_knowledge_adapter.labels_image),
                facies_image=nearest(expert_knowledge_adapter.facies_image),
            )
        functions = self.signal_model.model[1]
        reduced = HeterogeneousColorToMassAnalysis(
            baseline=baseline,
            labels=nearest(self.labels),
            color_mode=self._color_mode,
            color_path_interpretation=self.color_path_interpretation,
            signal_functions={label: functions[label] for label in functions.keys()},
            flash=self.flash,
            fluid_model=fluid_model,
            geometry=geometry,
            restoration=None,
            ignore_labels=self._ignore_labels,
            basis=self.basis,
            expert_knowledge_adapter=expert_knowledge_adapter,
            contour_smoother=self.contour_smoother,
            mass_unit=self.mass_unit,
            time_unit=self.time_unit,
            evaluation_mode=self.evaluation_mode,
            volume_unit=self.volume_unit,
        )
        reduced.pressure_temperature_protocol = self.pressure_temperature_protocol
        return reduced

    @darsia.timed("color interpretation")
    def call_color_interpretation(self, image: darsia.Image) -> darsia.Image:
        fast = self._fast_color_interpretation(image)
        if fast is not None:
            return fast
        # The generic analysis works on floats (the image may be in its native type).
        return self.color_analysis(image.as_float())

    @darsia.timed("signal model")
    @darsia.timing_decorator
    def call_pH_analysis(self, color_interpretation: darsia.Image) -> darsia.Image:
        """Run the mass analysis on a single image."""
        fast = self._fast_signal(color_interpretation)
        if fast is not None:
            return fast
        return self.signal_model(color_interpretation)

    # ! ---- Single-pass evaluation of the stages ----

    # The three stages are per-pixel arithmetic. If the analysis is set up in the
    # standard way, each stage is evaluated by a compiled kernel in one pass over the
    # pixels, with the same arithmetic (and results) as the generic implementation,
    # which is used in all other cases. The kernels state which code they mirror.
    # Whether a component is in its standard form is decided by the component itself
    # (`ConcentrationAnalysis.is_plain`, `HeterogeneousModel.color_path_labels`,
    # `PWTransformation.linear_table`, `SimpleFlash.ramp_bounds`,
    # `FluidMassAnalysis.row_properties`), next to its options.

    @staticmethod
    def _wrap(array: np.ndarray, image: darsia.Image) -> darsia.Image:
        """Image of a result, as returned by `ConcentrationAnalysis.__call__`."""
        metadata = image.metadata()
        if len(array.shape) == len(image.shape) - 1:
            return darsia.ScalarImage(array, **metadata)
        if array.shape[-1] == 1:
            return darsia.ScalarImage(array[..., 0], **metadata)
        return type(image)(array, **metadata)

    def _fast_color_interpretation(self, image: darsia.Image) -> darsia.Image | None:
        """Color interpretation in one pass (None if not applicable).

        The colors relative to the baseline are projected onto the color path of the
        label of each pixel, without forming the difference image.

        """
        analysis = self.color_analysis
        model = analysis.model
        if not (
            analysis.is_plain()
            and analysis.restoration is None
            and type(model) is darsia.CombinedModel
            and len(model.models) == 1
            and type(model.models[0]) is darsia.HeterogeneousModel
        ):
            return None
        array = image.img
        base = None if analysis.base is None else analysis.base.img
        if base is not None and (base.dtype != np.float64 or base.shape != array.shape):
            return None
        # The image may be held in the type of the corrections (see `Rig.read_image`,
        # native), if the kernel reads that type exactly.
        if color_kernels.image_scale(array.dtype, with_base=base is not None) is None:
            return None
        interpretations = model.models[0]
        active = interpretations.color_path_labels(array.shape)
        if active is None:
            return None
        values = color_kernels.evaluate_labeled_image(
            array,
            interpretations.masks.labels.img,
            {label: interpretations[label].kernel_data() for label in active},
            base=base,
        )
        return self._wrap(values, image)

    def _fast_signal(self, color_interpretation: darsia.Image) -> darsia.Image | None:
        """Signal in one pass (None if not applicable).

        The position on the color path is clipped to the common domain of the signal
        functions and converted by the piecewise linear function of the label of each
        pixel (linear evaluation); a restoration is applied afterwards.

        """
        analysis = self.signal_model
        model = analysis.model
        if not (
            signal_kernels.NUMBA_AVAILABLE
            and analysis.is_plain()
            and analysis.base is None
            and type(model) is darsia.CombinedModel
            and len(model.models) == 2
            and type(model.models[0]) is darsia.ClipModel
            and type(model.models[1]) is darsia.HeterogeneousModel
        ):
            return None
        clip, functions = model.models
        array = color_interpretation.img
        labels = functions.masks.labels.img
        low, high = clip.bounds()
        if not (
            array.dtype == np.float64
            and array.shape == labels.shape
            and np.issubdtype(labels.dtype, np.integer)
            and low is not None
            and high is not None
            and low <= high
        ):
            return None
        tables = {}
        for label in functions.masks.unique_labels:
            if label in functions.ignore_labels:
                continue
            linear_table = getattr(functions[label], "linear_table", None)
            table = None if linear_table is None else linear_table()
            # The function must be defined on the whole (clipped) domain.
            if table is None or table[0][0] > low or table[0][-1] < high:
                return None
            tables[label] = table
        if len(tables) == 0:
            return None
        signal = signal_kernels.evaluate_labeled_image(array, labels, tables, low, high)
        return self._wrap(analysis._restore_signal(signal), color_interpretation)

    def _fast_flash_and_mass(
        self, pH: darsia.Image
    ) -> SimpleMassAnalysisResults | None:
        """Flash and mass in one pass (None if not applicable).

        Concentration and saturation are linear ramps of the signal; density and
        aqueous capacity only depend on the pixel row (pressure and temperature vary
        with height).

        """
        flash, fluid_model = self.flash, self.fluid_model
        signal = pH.img
        if not (
            mass_kernels.NUMBA_AVAILABLE
            and signal.dtype == np.float64
            and signal.ndim == 2
        ):
            return None
        ramp_bounds = getattr(flash, "ramp_bounds", None)
        row_properties = getattr(fluid_model, "row_properties", None)
        if ramp_bounds is None or row_properties is None:
            return None
        bounds = ramp_bounds()
        properties = row_properties(signal.shape)
        if bounds is None or properties is None:
            return None
        density_gas, aqueous_capacity = properties
        keep_aq = keep_g = None
        if self.expert_knowledge_adapter is not None:
            keep_aq = self.expert_knowledge_adapter.mask_for(pH, "concentration_aq")
            keep_g = self.expert_knowledge_adapter.mask_for(pH, "saturation_g")
            if any(
                keep is not None and keep.shape != signal.shape
                for keep in (keep_aq, keep_g)
            ):
                return None
        c_aq, s_g, mass, mass_g, mass_aq = mass_kernels.flash_and_mass(
            signal,
            bounds,
            density_gas,
            aqueous_capacity,
            keep_aq,
            keep_g,
        )
        metadata = pH.metadata()

        def image(array: np.ndarray) -> darsia.Image:
            return type(pH)(array, **metadata)

        return SimpleMassAnalysisResults(
            name=pH.name,
            date=pH.date,
            time=pH.time,
            mass=image(mass),
            mass_g=image(mass_g),
            mass_aq=image(mass_aq),
            saturation_g=image(s_g),
            concentration_aq=image(c_aq),
            color_signal=None,
        )

    @darsia.timed("flash and mass")
    @darsia.timing_decorator
    def call_flash_and_mass_analysis(
        self, pH: darsia.Image
    ) -> SimpleMassAnalysisResults:
        """Run the mass analysis on a single image."""
        fast = self._fast_flash_and_mass(pH)
        if fast is not None:
            return fast
        c_aq, s_g = self.flash(pH)
        if self.expert_knowledge_adapter is not None:
            c_aq = self.expert_knowledge_adapter.apply(c_aq, "concentration_aq")
            s_g = self.expert_knowledge_adapter.apply(s_g, "saturation_g")
        mass_analysis_result: SimpleMassAnalysisResults = (
            self.fluid_model.mass_analysis(
                c_aq=c_aq,
                s_g=s_g,
            )
        )
        return mass_analysis_result

    def update_state(
        self,
        date: datetime | None,
        protocol: "darsia.PressureTemperatureProtocol | None" = None,
    ) -> None:
        """Set pressure and temperature (and gradients) of the fluid model at a date.

        Densities and solubility depend on the state, which may change during the
        experiment (weather, room temperature). Nothing happens without a protocol or
        a date; an unchanged state is not evaluated again.

        Parameters
        ----------
        date : datetime, optional
            Date of the image.
        protocol : darsia.PressureTemperatureProtocol, optional
            Protocol to use; default is `pressure_temperature_protocol`.
        """
        protocol = self.pressure_temperature_protocol if protocol is None else protocol
        if protocol is None or date is None:
            return
        state = protocol.get_state(date)
        gradient = protocol.get_gradient(date)
        self.fluid_model.update_state(
            atmospheric_pressure=state.pressure,
            atmospheric_temperature=state.temperature,
            atmospheric_pressure_gradient=gradient.pressure,
            atmospheric_temperature_gradient=gradient.temperature,
        )

    @darsia.timing_decorator
    def __call__(self, image: darsia.Image) -> SimpleMassAnalysisResults:
        """Run the analysis on a single image, at the state of its date."""
        self.update_state(image.date)
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
        save: Callable[[], None] | None = None,
        resolution_level: int = 1,
        stored: bool = False,
    ) -> bool:
        """Define a local color path for a specific label.

        Instructions:
            - Pick a label in the image using the mouse and select a rectangle.
            - Define a new color path just for this label.
            - Tune the values for the color path.
            - Save with the Save button (the window stays open). Closing the window
              ends the calibration; with unsaved changes, it asks whether to save
              them (without a dialog, e.g. in a script, they are saved).
            - The resolution slider reduces the images per axis for the calculations
              (applied by Update): fast tuning at a reduced level, check at level 1.

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
        save : Callable, optional
            Stores the current calibration; called by the Save button, and when the
            window is closed with unsaved changes which the user wants to keep.
        resolution_level : int
            Reduction of the images per axis for the calculations at the start (1, 2,
            4, 8 or 16; 1 is the full resolution). The stored calibration does not
            depend on it.
        stored : bool
            Whether the calibration at the start is the stored one; otherwise (e.g. a
            new calibration) it counts as unsaved.

        Returns
        -------
        bool
            Whether the stored calibration equals the calibration at the end (False
            if changes were discarded or saving failed).
        """

        # Data and calculations (no plotting)
        state = CalibrationState(
            self, images, experiment, rois, resolution_level, stored=stored
        )
        rois = state.rois
        # Mass series per roi; updated in place by `state.update_mass_analysis`
        times = state.times
        expected_mass = state.expected_mass
        integrated_mass = state.integrated_mass
        integrated_mass_g = state.integrated_mass_g
        integrated_mass_aq = state.integrated_mass_aq
        # Values restored by the Reset button
        initial_values = state.initial_values
        initial_flash = state.initial_flash
        default_contour_threshold = state.default_contour_threshold
        # Current contour thresholds; kept when the figure is rebuilt
        contour_thresholds = state.contour_thresholds

        # ! ----- IMAGE SELECTION -----

        def image_idx_selector() -> int:
            """Interactive selection of plotted images - allow to click for showing the next
            coarse images - keep track of the idx in the list and return when finishing.

            """
            # Create a simple figure for image selection
            fig, ax = plt.subplots(figsize=(8, 6))
            img_display = ax.imshow(state.coarse_image.img)
            ax.set_title(
                f"""Image {state.image_idx + 1}/{len(images)} - Click to cycle through """
                f"""images, close window to finish"""
            )
            ax.axis("off")

            def on_click(event):
                if event.inaxes == ax:
                    # Cycle to next image
                    state.next_image()
                    img_display.set_data(state.coarse_image.img)
                    ax.set_title(
                        f"""Image {state.image_idx + 1}/{len(images)} - Click to cycle """
                        f"""through images, close window to finish"""
                    )
                    fig.canvas.draw_idle()

            # Connect click event
            fig.canvas.mpl_connect("button_press_event", on_click)

            plt.show()
            return state.image_idx

        # ! ---- LABEL SELECTION ----

        def label_idx_selector(pH: darsia.Image) -> int:
            assistant = darsia.RectangleSelectionAssistant(
                pH, labels=state.labels, cmap=cmap
            )
            label_box: Tuple[slice, slice] = assistant()
            return state.select_label_in(label_box)

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
                image_idx_selector()
                need_to_pick_new_image = False
                need_to_pick_new_label = True

            # Fetch the current image and analyze it
            coarse_image = state.coarse_image
            if state.maps is None or state.maps["image_idx"] != state.image_idx:
                state.compute_maps()

            # Pick label interactively
            if need_to_pick_new_label:
                label_idx_selector(state.maps["pH"])
                need_to_pick_new_label = False

            # Compute mass analysis
            if not state.mass_analysis_done:
                state.update_mass_analysis()

            # The figure is built for this image and label
            image_idx = state.image_idx
            label_idx = state.label_idx
            coarse_signal = state.coarse_signal
            coarse_contour = state.contour_overlay()

            # Use gridspec to create a 2x3 grid with proper ratios
            fig = plt.figure(figsize=(20, 12))
            maximize_figure_window(fig)
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
            ax_signal.set_title(state.signal_title)

            # Top right: Contour/Secondary visualization
            ax_contour = plt.subplot(gs[0, 2])
            ax_contour.axis("off")
            ax_contour.set_title(
                "Phase Segmentation" if self.has_gas else "Concentration Contours"
            )

            # Bottom left: Signal function plot
            ax_signal_function = plt.subplot(gs[1, 0])
            ax_signal_function.set_title(f"Signal Function - Label {label_idx}")

            # Bottom middle: Mass evolution plots
            ax_mass = plt.subplot(gs[1, 1])
            ax_mass.set_title("Mass Evolution" if self.has_gas else "Volume Evolution")

            # Bottom right: Parameter sliders (will be subdivided)
            ax_sliders = plt.subplot(gs[1, 2])
            ax_sliders.axis("off")  # Turn off axis for slider area
            ax_sliders.set_title("Parameter Controls")

            # ! ---- PLOTTING ----

            # Plot the coarse image with highlighted label - turn the rest to gray
            coarse_image_img = ax_coarse_image.imshow(
                state.label_highlight(coarse_image, label_idx)
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
            initial_signal_min, initial_signal_max = state.signal_range()
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

                # Plot the signal function evaluated on a fine grid along the color
                # path, with the values at the key colors as dots
                x_vals = signal_func.supports
                t_fine, y_fine = signal_curve(signal_func)
                [signal_function_line] = ax_signal_function.plot(
                    t_fine, y_fine, "gray", linewidth=2, zorder=2
                )

                colors = _color_path_interpretation.color_path.colors
                facecolors = to_rgba_array(np.clip(colors, 0, 1))
                signal_function_scatter = ax_signal_function.scatter(
                    x_vals,
                    _dot_values(signal_func),
                    c=facecolors,
                    zorder=3,
                    s=50,
                    edgecolors="black",
                    linewidth=1,
                )

                if self.has_gas:
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
                    aq_min_x = curve_inverse(t_fine, y_fine, self.flash.min_value_aq)
                    aq_max_x = curve_inverse(t_fine, y_fine, self.flash.max_value_aq)
                    g_max_x = curve_inverse(t_fine, y_fine, self.flash.max_value_g)
                    g_min_x = curve_inverse(t_fine, y_fine, self.flash.min_value_g)

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

                    # Annotate "CO2(aq)" and "CO2(g)" below and above the cut-off
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
                # Single-phase (tracer): no phase transition, hence no flash lines;
                # the signal is the concentration itself.
                # NOTE: Keep after the annotations above, which are accessed by index
                used_indices = getattr(signal_func, "support_indices", None)
                if used_indices is not None and len(used_indices) < len(
                    signal_func.values
                ):
                    ax_signal_function.text(
                        x=0.01,
                        y=0.99,
                        s="Unused key colors (no dot): the signal is affine in the "
                        "color and bends at every key color.",
                        transform=ax_signal_function.transAxes,
                        ha="left",
                        va="top",
                        fontsize=7,
                        color="k",
                        zorder=4,
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
            max_expected = state.max_expected_mass()
            max_time = state.max_time()
            # Quantity compared with the injection protocol: the mass (CO2), or the
            # volume (tracer; the "mass" series below then hold volumes in m3).
            if self.has_gas:
                mass_unit = (
                    choose_mass_unit(max_expected)
                    if self.mass_unit == "auto"
                    else self.mass_unit
                )
                convert_quantity, quantity_label = convert_mass, mass_label
                total_name = "total"
            else:
                mass_unit = (
                    choose_volume_unit(max_expected)
                    if self.volume_unit == "auto"
                    else self.volume_unit
                )
                convert_quantity, quantity_label = convert_volume, volume_label
                total_name = "detected"
            time_unit = (
                choose_time_unit(max_time)
                if self.time_unit == "auto"
                else self.time_unit
            )
            ax_mass.set_xlabel(time_label(time_unit))
            ax_mass.set_ylabel(quantity_label(mass_unit))

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
                        convert_quantity(integrated_mass[roi_key], mass_unit),
                        color=color,
                        label=f"{total_name} ({roi_label})",
                        linewidth=2,
                    )
                    if self.has_gas:
                        [integrated_mass_plots_g[roi_key]] = ax_mass.plot(
                            plot_times,
                            convert_quantity(integrated_mass_g[roi_key], mass_unit),
                            color=color,
                            label=f"gas ({roi_label})",
                            linestyle=":",
                            linewidth=2,
                        )
                        [integrated_mass_plots_aq[roi_key]] = ax_mass.plot(
                            plot_times,
                            convert_quantity(integrated_mass_aq[roi_key], mass_unit),
                            color=color,
                            label=f"aqueous ({roi_label})",
                            linestyle="-.",
                            linewidth=2,
                        )
                    [expected_mass_plots[roi_key]] = ax_mass.plot(
                        plot_times,
                        convert_quantity(expected_mass[roi_key], mass_unit),
                        linestyle="--",
                        color=color,
                        label=f"injected ({roi_label})",
                        alpha=0.7,
                    )

                    # Plot scatter points
                    integrated_mass_scatters[roi_key] = ax_mass.scatter(
                        plot_times,
                        convert_quantity(integrated_mass[roi_key], mass_unit),
                        color=color,
                        s=20,
                        alpha=0.8,
                    )
                    if self.has_gas:
                        integrated_mass_scatters_g[roi_key] = ax_mass.scatter(
                            plot_times,
                            convert_quantity(integrated_mass_g[roi_key], mass_unit),
                            color=color,
                            s=15,
                            alpha=0.6,
                            marker="^",
                        )
                        integrated_mass_scatters_aq[roi_key] = ax_mass.scatter(
                            plot_times,
                            convert_quantity(integrated_mass_aq[roi_key], mass_unit),
                            color=color,
                            s=15,
                            alpha=0.6,
                            marker="s",
                        )

            # Set y-axis limits based on maximum expected mass across all ROIs
            if max_expected > 0:
                ax_mass.set_ylim(0.0, convert_quantity(2.0 * max_expected, mass_unit))
            else:
                ax_mass.autoscale()
            ax_mass.legend()
            ax_mass.grid(True, alpha=0.3)

            # ! ---- PARAMETER CONTROLS ----

            # Grouped rows (label | slider | value box). The number of signal value
            # rows follows the number of key colors of the selected label.
            controls_box = ax_sliders.get_position()
            signal_values = self.signal_model.model[1][label_idx].values
            used_values = getattr(
                self.signal_model.model[1][label_idx],
                "support_indices",
                range(len(signal_values)),
            )
            path_colors = self.color_path_interpretation[label_idx].color_path.colors
            num_values = len(signal_values)
            # Rows: signal values (title, values), flash (title, 4 sliders; CO2 only),
            # contours (title, one threshold per phase), resolution (title, level)
            levels = state.available_levels
            num_rows = num_values + (9 if self.has_gas else 3)
            if len(levels) > 1:
                num_rows += 2
            row_height = controls_box.height / num_rows

            def make_row(
                row, name, valinit, vmin, vmax, color, active=True, valstep=0.01
            ):
                """Create a labeled slider (with value text) in a row."""
                x0, width = controls_box.x0, controls_box.width
                y0 = controls_box.y1 - (row + 1) * row_height
                label_text = fig.text(
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
                    valstep=valstep,
                    color=color,
                )
                slider.valtext.set_fontsize(8)
                if not active:
                    # Values without influence cannot be changed and appear faded
                    slider.set_active(False)
                    for artist in (
                        label_text,
                        slider.poly,
                        slider._handle,
                        slider.valtext,
                    ):
                        artist.set_alpha(0.35)
                return slider

            def section_title(row, title):
                return fig.text(
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
                    f"Value {i}" + ("" if i in used_values else " (unused)"),
                    val,
                    0.0,
                    2.0 if self.has_gas else 1.0,
                    np.clip(path_colors[i], 0, 1) if i < len(path_colors) else "gray",
                    active=i in used_values,
                )
                for i, val in enumerate(signal_values)
            ]

            flash_row = num_values + 1
            if self.has_gas:
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
            else:
                # Single-phase: no flash parameters, the signal is the concentration
                sliders_flash = []
                contour_row = flash_row
            section_title(contour_row, "Contours")
            sliders_threshold = [
                make_row(
                    contour_row + 1,
                    "Threshold c_aq",
                    contour_thresholds[0],
                    0.0,
                    1.0,
                    (1, 0, 0),
                )
            ]
            if self.has_gas:
                sliders_threshold.append(
                    make_row(
                        contour_row + 2,
                        "Threshold s_g",
                        contour_thresholds[1],
                        0.0,
                        1.0,
                        (0, 1, 0),
                    )
                )

            for i, slider in enumerate(sliders_threshold):
                slider.on_changed(
                    lambda val, i=i: contour_thresholds.__setitem__(i, val)
                )

            # Slider updates only repaint the sliders (blitting). Otherwise, every
            # slider step re-renders the entire figure, including all images. The
            # moving parts of the sliders are animated artists, which are excluded
            # from the stored background and drawn on top of it.
            all_sliders = sliders_color_to_signal + sliders_flash + sliders_threshold

            # Resolution of the calculations: a discrete slider over the levels. Like
            # the parameters, the selected level is applied by the Update button.
            slider_resolution = None
            if len(levels) > 1:
                resolution_row = contour_row + 1 + len(sliders_threshold)

                def resolution_title() -> str:
                    return f"Resolution (in use: 1/{state.resolution_level})"

                resolution_status = section_title(resolution_row, resolution_title())
                slider_resolution = make_row(
                    resolution_row + 1,
                    "Level",
                    levels.index(state.resolution_level),
                    0,
                    len(levels) - 1,
                    "steelblue",
                    valstep=1,
                )
                # Show the level instead of its position
                slider_resolution._format = lambda val: f"1/{levels[int(round(val))]}"
                slider_resolution.valtext.set_text(
                    slider_resolution._format(slider_resolution.val)
                )

            def selected_level() -> int:
                """Resolution level chosen with the slider."""
                if slider_resolution is None:
                    return state.resolution_level
                return levels[int(round(slider_resolution.val))]

            blit_sliders = all_sliders + (
                [] if slider_resolution is None else [slider_resolution]
            )
            animated_artists = [
                artist
                for slider in blit_sliders
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

            for slider in blit_sliders:
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
            # - Save
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
            busy_status = fig.text(
                update_button_x + button_width,
                update_button_y + button_height + 0.004,
                "",
                fontsize=8,
                color="darkorange",
                ha="right",
                va="bottom",
            )
            """Message shown while a calculation blocks the window."""
            values_status = fig.text(
                controls_box.x0,
                update_button_y + 0.5 * button_height,
                "",
                fontsize=8,
                color="red",
                ha="left",
                va="center",
            )
            """Message about invalid (decreasing) signal values."""

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

            save_button_x = 0.02  # Left side with small margin
            ax_save = plt.axes(
                [
                    save_button_x,
                    button_y,
                    button_width,
                    button_height,
                ]
            )
            btn_save = Button(ax_save, "Save")
            save_status = fig.text(
                save_button_x + button_width + button_spacing,
                button_y + 0.5 * button_height,
                "",
                fontsize=9,
                color="green",
                va="center",
            )
            rebuilding = False  # figure is closed to be rebuilt, not to leave

            # Tune values for this label
            done_tuning_values = False
            while not done_tuning_values:

                def show_tuner(_label_idx):
                    nonlocal done_tuning_values  # noqa: F824
                    nonlocal done_calibration  # noqa: F824

                    def update_signal_function_plot():
                        """Update the signal function plot (curve, flash lines)."""
                        signal_func = self.signal_model.model[1][_label_idx]
                        t_fine, y_fine = signal_curve(signal_func)
                        # Update signal function plot
                        if signal_function_line is not None:
                            x_vals = signal_func.supports
                            y_vals = signal_func.values.copy()
                            signal_function_line.set_data(t_fine, y_fine)
                            signal_function_scatter.set_offsets(
                                np.c_[x_vals, _dot_values(signal_func)]
                            )

                            # Rescale y-axis to fit the curve, floored at the
                            # current flash max (never zoom in tighter than the
                            # flash thresholds, but grow if the curve exceeds
                            # them)
                            flash_max = (
                                max(self.flash.max_value_aq, self.flash.max_value_g)
                                if self.has_gas
                                else 1.0
                            )
                            ax_signal_function.set_ylim(
                                self.signal_model_extents[1][0],
                                max(
                                    flash_max,
                                    float(np.max(y_fine)),
                                    float(np.max(y_vals)),
                                ),
                            )

                        if self.has_gas:
                            # Update flash cut-off and max value lines
                            signal_function_aq_min_y.set_ydata(
                                [self.flash.min_value_aq]
                            )
                            signal_function_aq_max_y.set_ydata(
                                [self.flash.max_value_aq]
                            )
                            signal_function_g_max_y.set_ydata([self.flash.max_value_g])
                            signal_function_g_min_y.set_ydata([self.flash.min_value_g])
                            aq_max_x = curve_inverse(
                                t_fine, y_fine, self.flash.max_value_aq
                            )
                            aq_min_x = curve_inverse(
                                t_fine, y_fine, self.flash.min_value_aq
                            )
                            g_max_x = curve_inverse(
                                t_fine, y_fine, self.flash.max_value_g
                            )
                            g_min_x = curve_inverse(
                                t_fine, y_fine, self.flash.min_value_g
                            )
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
                            ax_signal_function.texts[2].set_position(
                                (g_max_x + 0.1, 0.04)
                            )

                    def update_mass_plot():
                        """Update the mass evolution plots."""
                        # Update mass plots
                        max_expected = state.max_expected_mass()
                        max_time = state.max_time()
                        if self.has_gas:
                            mass_unit = (
                                choose_mass_unit(max_expected)
                                if self.mass_unit == "auto"
                                else self.mass_unit
                            )
                            convert_quantity, quantity_label = convert_mass, mass_label
                        else:
                            mass_unit = (
                                choose_volume_unit(max_expected)
                                if self.volume_unit == "auto"
                                else self.volume_unit
                            )
                            convert_quantity, quantity_label = (
                                convert_volume,
                                volume_label,
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
                                convert_quantity(integrated_mass[roi_key], mass_unit),
                            )
                            if self.has_gas:
                                integrated_mass_plots_g[roi_key].set_data(
                                    plot_times,
                                    convert_quantity(
                                        integrated_mass_g[roi_key], mass_unit
                                    ),
                                )
                                integrated_mass_plots_aq[roi_key].set_data(
                                    plot_times,
                                    convert_quantity(
                                        integrated_mass_aq[roi_key], mass_unit
                                    ),
                                )
                            integrated_mass_scatters[roi_key].set_offsets(
                                np.c_[
                                    plot_times,
                                    convert_quantity(
                                        integrated_mass[roi_key], mass_unit
                                    ),
                                ]
                            )
                            if self.has_gas:
                                integrated_mass_scatters_g[roi_key].set_offsets(
                                    np.c_[
                                        plot_times,
                                        convert_quantity(
                                            integrated_mass_g[roi_key], mass_unit
                                        ),
                                    ]
                                )
                                integrated_mass_scatters_aq[roi_key].set_offsets(
                                    np.c_[
                                        plot_times,
                                        convert_quantity(
                                            integrated_mass_aq[roi_key], mass_unit
                                        ),
                                    ]
                                )
                        if max_expected > 0:
                            ax_mass.set_ylim(
                                0.0, convert_quantity(2.0 * max_expected, mass_unit)
                            )
                        else:
                            ax_mass.autoscale()
                        ax_mass.set_xlabel(time_label(time_unit))
                        ax_mass.set_ylabel(quantity_label(mass_unit))

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
                        if recompute:
                            state.compute_maps()

                        # Image panel and signal panel
                        ax_coarse_image.set_title(
                            f"Current Image {state.image_idx} and Label {_label_idx}"
                        )
                        coarse_image_img.set_data(
                            state.label_highlight(state.coarse_image, _label_idx)
                        )
                        coarse_signal_img.set_data(state.coarse_signal.img)
                        ax_signal.set_title(state.signal_title)
                        signal_min, signal_max = state.signal_range()
                        coarse_signal_img.set_clim(vmin=signal_min, vmax=signal_max)
                        coarse_signal_colorbar.update_normal(coarse_signal_img)

                        # Phase segmentation panel
                        coarse_contour_img.set_data(
                            state.contour_overlay(
                                [slider.val for slider in sliders_threshold]
                            ).img
                        )

                        if signal_function and signal_function_line is not None:
                            update_signal_function_plot()
                        if mass_plot:
                            update_mass_plot()
                        fig.canvas.draw_idle()

                    def check_values() -> str | None:
                        """Message if the signal values of the label decrease, else None."""
                        values = [slider.val for slider in sliders_color_to_signal]
                        return state.check_values(values, _label_idx)

                    def show_values_status(message: str | None) -> None:
                        """Show (or clear) the message about invalid signal values."""
                        text = (
                            "" if message is None else message.replace(": ", ":\n", 1)
                        )
                        if values_status.get_text() != text:
                            values_status.set_text(text)
                            fig.canvas.draw_idle()

                    for slider in sliders_color_to_signal:
                        slider.on_changed(
                            lambda val: show_values_status(check_values())
                        )

                    def apply_parameters() -> bool | None:
                        """Apply the slider values.

                        Returns
                        -------
                        bool or None
                            Whether any value changed, or None if the values are
                            invalid, in which case nothing is applied.

                        """
                        nonlocal pending
                        message = check_values()
                        show_values_status(message)
                        if message is not None:
                            return None
                        changed = state.apply_parameters(
                            [slider.val for slider in sliders_color_to_signal],
                            [slider.val for slider in sliders_flash],
                            _label_idx,
                        )
                        pending = False
                        return changed

                    # Whether a value or flash slider was moved since the last
                    # applied values. The sliders round to their step, so their
                    # untouched values are not applied (they would change the
                    # stored calibration).
                    pending = False

                    def on_value_slider(*args):
                        nonlocal pending
                        pending = True

                    for slider in sliders_color_to_signal + sliders_flash:
                        slider.on_changed(on_value_slider)

                    unsaved_text = "Changes not saved."

                    def mark_unsaved(*args):
                        """Show that the displayed values differ from the saved ones."""
                        if save_status.get_text() != unsaved_text:
                            save_status.set_text(unsaved_text)
                            save_status.set_color("darkorange")
                            fig.canvas.draw_idle()

                    for slider in all_sliders:
                        slider.on_changed(mark_unsaved)

                    def show_busy(message: str) -> None:
                        """Show (or clear) a message and draw it right away.

                        The calculations run in the window's event loop, so the
                        window does not react until they are done.

                        """
                        if busy_status.get_text() == message:
                            return
                        busy_status.set_text(message)
                        fig.canvas.draw()
                        fig.canvas.flush_events()

                    def update_analysis(event=None):
                        """Apply the slider values and update everything."""
                        changed = apply_parameters()
                        if changed is None:
                            return
                        level = selected_level()
                        new_level = level != state.resolution_level
                        if changed:
                            mark_unsaved()
                        if changed or new_level:
                            show_busy(
                                f"Computing at resolution 1/{level} ..."
                                if new_level
                                else "Computing ..."
                            )
                        if new_level:
                            state.set_resolution(level)
                            resolution_status.set_text(resolution_title())
                        if changed or new_level:
                            state.update_mass_analysis()
                        refresh(recompute=True, signal_function=True, mass_plot=True)
                        show_busy("")

                    # Connect button callbacks
                    btn_update.on_clicked(update_analysis)

                    def reset(event):
                        """Restore the initial values of this label, flash, contours."""
                        for slider, value in zip(
                            sliders_color_to_signal, initial_values[_label_idx]
                        ):
                            slider.set_val(value)
                        if self.has_gas:
                            for slider, value in zip(sliders_flash, initial_flash):
                                slider.set_val(value)
                        for slider in sliders_threshold:
                            slider.set_val(default_contour_threshold)
                        update_analysis()

                    btn_reset.on_clicked(reset)

                    def new_label(event):
                        nonlocal done_tuning_values, need_to_pick_new_label
                        nonlocal rebuilding
                        done_tuning_values = True
                        need_to_pick_new_label = True
                        rebuilding = True
                        plt.close("all")

                    btn_new_label.on_clicked(new_label)

                    def new_image(event):
                        nonlocal done_tuning_values, need_to_pick_new_image
                        nonlocal rebuilding
                        done_tuning_values = True
                        need_to_pick_new_image = True
                        rebuilding = True
                        plt.close("all")

                    btn_new_image.on_clicked(new_image)

                    def save_calibration() -> bool:
                        """Store the calibration; return if it was successful."""
                        if save is None:
                            logger.warning("No way to save the calibration provided.")
                            return False
                        try:
                            save()
                        except Exception:
                            logger.exception("Saving the calibration failed.")
                            return False
                        state.mark_saved()
                        logger.info("Saved the calibration.")
                        return True

                    def on_save(event):
                        """Apply the current values and save; the window stays open."""
                        message = check_values()
                        if message is not None:
                            show_values_status(message)
                            save_status.set_text("Not saved: fix the values first.")
                            save_status.set_color("red")
                            fig.canvas.draw_idle()
                            return
                        update_analysis()
                        saved = save_calibration()
                        save_status.set_text(
                            "Saved." if saved else "Saving failed, see log."
                        )
                        save_status.set_color("green" if saved else "red")
                        fig.canvas.draw_idle()

                    btn_save.on_clicked(on_save)

                    def on_close(event):
                        """Closing the window ends the calibration; unsaved changes
                        are saved or discarded, as the user decides."""
                        nonlocal done_tuning_values, done_calibration
                        if rebuilding:
                            return
                        try:
                            message = check_values() if pending else None
                            if pending and message is None:
                                apply_parameters()
                            elif pending:
                                logger.warning(
                                    f"{message}. The values of the sliders are not "
                                    "applied."
                                )
                            if state.has_unsaved_changes():
                                keep = ask_yes_no(
                                    "Unsaved calibration",
                                    "The calibration has unsaved changes.",
                                    yes="Save",
                                    no="Discard",
                                )
                                # Without a dialog (e.g. a script), keep the changes
                                if keep is None or keep:
                                    save_calibration()
                                else:
                                    logger.info(
                                        "Discarded the unsaved changes of the "
                                        "calibration."
                                    )
                        except Exception:
                            logger.exception("Closing the calibration window failed.")
                        finally:
                            done_tuning_values = True
                            done_calibration = True

                    fig.canvas.mpl_connect("close_event", on_close)

                    def next_signal(event):
                        state.next_signal()
                        refresh(recompute=False)

                    btn_next_signal.on_clicked(next_signal)

                    def next_label(event):
                        """Move to the next label and rebuild the figure for it."""
                        nonlocal done_tuning_values, need_to_pick_new_label
                        nonlocal rebuilding
                        state.next_label()
                        need_to_pick_new_label = False
                        done_tuning_values = True
                        rebuilding = True
                        plt.close("all")

                    btn_next_label.on_clicked(next_label)

                    def next_image(event):
                        show_busy("Computing ...")
                        state.next_image()
                        refresh(recompute=True)
                        show_busy("")

                    btn_next_image.on_clicked(next_image)

                    plt.show()

                show_tuner(label_idx)

        return not state.has_unsaved_changes()

    def save(self, folder: Path, interpretation: Interpretation) -> None:
        """Save the calibration data to json file.

        Parameters
        ----------
        folder : Path
            The folder to save the calibration data to.
        interpretation : Interpretation
            Quantity and closure of the calibration; stored in the metadata
            and checked on load.
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

        # Save the evaluation mode
        self.evaluation_mode.save(folder)

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
            "interpretation": interpretation.to_dict(),
            # Fingerprints of the inputs, for the missing/valid/outdated status
            "inputs": calibration_inputs(folder),
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
        evaluation_mode: darsia.EvaluationMode,
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
            signal_functions[label] = evaluation_mode.signal_function(
                interpretation.color_path,
                interpretation.color_mode,
                supports,
                np.maximum.accumulate(values),
                label=label,
            )

    @classmethod
    def load(
        cls,
        folder: Path,
        baseline: darsia.Image,
        labels: darsia.Image,
        interpretation: Interpretation,
        fluid_model: darsia.FluidMassAnalysis,
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
        volume_unit: str = "auto",
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
        fluid_model : darsia.FluidMassAnalysis
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
        volume_unit : str, optional
            Unit for volume axes in tracer calibration plots ('auto' or a fixed unit).
        interpretation : Interpretation
            The expected quantity and closure; must match the stored
            calibration (checked first, before anything else is read).
        """
        metadata = read_calibration_metadata(folder / "metadata.json")
        validate_interpretation_metadata(
            metadata=metadata, expected=interpretation, artifact="color_to_mass"
        )
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
        evaluation_mode = darsia.EvaluationMode.load(folder)
        signal_functions = {}
        for label in np.unique(labels.img):
            if label < 0:
                continue
            stored = darsia.PWTransformation.load(
                folder / "signal_model" / f"signal_model_{label}"
            )
            path_interpretation = color_path_interpretation[label]
            signal_functions[label] = evaluation_mode.signal_function(
                path_interpretation.color_path,
                path_interpretation.color_mode,
                stored.supports,
                stored.values,
                label=label,
            )
        if not stored_interpretation:
            cls._resample_signal_functions(
                signal_functions, color_path_interpretation, evaluation_mode
            )
        flash = load_flash(interpretation.closure, folder / "flash" / "flash")

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
            fluid_model=fluid_model,
            geometry=geometry,
            restoration=restoration,
            ignore_labels=ignore_labels,
            basis=parsed_basis,
            expert_knowledge_adapter=expert_knowledge_adapter,
            contour_smoother=contour_smoother,
            mass_unit=mass_unit,
            time_unit=time_unit,
            evaluation_mode=evaluation_mode,
            volume_unit=volume_unit,
        )
