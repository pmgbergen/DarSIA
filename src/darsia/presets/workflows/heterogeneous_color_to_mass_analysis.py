# TODO review noqa F824 usage

import dataclasses
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

import darsia
import darsia.multiphase.mass_kernels as mass_kernels
import darsia.signals.color.color_path_kernels as color_kernels
import darsia.signals.models.signal_kernels as signal_kernels
from darsia.multiphase.flash import load_flash
from darsia.presets.workflows.analysis.expert_knowledge import ExpertKnowledgeAdapter
from darsia.presets.workflows.basis import label_ids_from_image
from darsia.presets.workflows.calibration.calibration_controller import (
    CalibrationController,
)
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
        masks = darsia.Masks(labels)

        base_model = (
            darsia.CombinedModel(
                [
                    darsia.HeterogeneousModel(
                        self.color_path_interpretation,
                        labels,
                        ignore_labels=ignore_labels,
                        masks=masks,
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
                    masks=masks,
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

    def has_fast_color_interpretation(self, image: darsia.Image) -> bool:
        """Whether the color interpretation of `image` runs in one compiled pass.

        Then it is cheap enough to be computed again instead of being kept.

        """
        return self._fast_color_interpretation_arguments(image) is not None

    def _fast_color_interpretation(self, image: darsia.Image) -> darsia.Image | None:
        """Color interpretation in one pass (None if not applicable).

        The colors relative to the baseline are projected onto the color path of the
        label of each pixel, without forming the difference image.

        """
        arguments = self._fast_color_interpretation_arguments(image)
        if arguments is None:
            return None
        array, labels, data, base = arguments
        values = color_kernels.evaluate_labeled_image(array, labels, data, base=base)
        return self._wrap(values, image)

    def _fast_color_interpretation_arguments(self, image: darsia.Image) -> tuple | None:
        """Arguments of `color_kernels.evaluate_labeled_image` for `image` (None if the
        compiled pass does not apply)."""
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
        return (
            array,
            interpretations.masks.labels.img,
            {label: interpretations[label].kernel_data() for label in active},
            base,
        )

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

    def _flash_and_mass_arguments(self, pH: darsia.Image) -> tuple | None:
        """Inputs of the flash and mass kernels (None if they do not apply).

        Signal, flash bounds, gas density and aqueous capacity per row, and the
        expert-knowledge masks (or None).

        """
        signal = pH.img
        if not (
            mass_kernels.NUMBA_AVAILABLE
            and signal.dtype == np.float64
            and signal.ndim == 2
        ):
            return None
        ramp_bounds = getattr(self.flash, "ramp_bounds", None)
        row_properties = getattr(self.fluid_model, "row_properties", None)
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
        return signal, bounds, density_gas, aqueous_capacity, keep_aq, keep_g

    def integrated_flash_and_mass(
        self, pH: darsia.Image, geometry: darsia.Geometry
    ) -> tuple[float, float] | None:
        """Integrated gas mass and dissolved mass of a signal, without the fields.

        Gives `geometry.integrate` of the gas and dissolved mass of
        `call_flash_and_mass_analysis(pH)` to the last digit, in a fraction of the
        time (e.g. for a series over many images, where only integrals are needed).

        Parameters
        ----------
        pH : darsia.Image
            Signal.
        geometry : darsia.Geometry
            Geometry of the signal's shape to integrate over.

        Returns
        -------
        tuple or None
            Integrated mass in the gas and in the aqueous phase; None if the setup is
            not the standard one (then use `call_flash_and_mass_analysis`).

        """
        arguments = self._flash_and_mass_arguments(pH)
        if arguments is None or not hasattr(geometry, "voxel_volumes"):
            return None
        voxel_volume = geometry.voxel_volumes(pH.img.shape)
        if not (
            isinstance(voxel_volume, np.ndarray)
            and voxel_volume.shape == pH.img.shape
            and voxel_volume.dtype == np.float64
        ):
            return None
        signal, bounds, density_gas, aqueous_capacity, keep_aq, keep_g = arguments
        return mass_kernels.flash_and_mass_integrals(
            signal, bounds, density_gas, aqueous_capacity, voxel_volume, keep_aq, keep_g
        )

    def _fast_flash_and_mass(
        self, pH: darsia.Image
    ) -> SimpleMassAnalysisResults | None:
        """Flash and mass in one pass (None if not applicable).

        Concentration and saturation are linear ramps of the signal; density and
        aqueous capacity only depend on the pixel row (pressure and temperature vary
        with height).

        """
        arguments = self._flash_and_mass_arguments(pH)
        if arguments is None:
            return None
        c_aq, s_g, mass, mass_g, mass_aq = mass_kernels.flash_and_mass(*arguments)
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
            - The window opens on the first image; Next Image cycles the images.
            - Click in the image to select a label (Next Label cycles the labels).
            - Tune the values for the color path of the label and apply them with
              Update; values not applied are dropped when the label changes.
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

        state = CalibrationState(
            self, images, experiment, rois, resolution_level, stored=stored
        )
        return CalibrationController(state, self, cmap, save).run()

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
