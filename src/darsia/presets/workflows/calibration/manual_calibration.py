"""State of the manual calibration of the color interpretation (no plotting).

The calibration tunes, per label, the values of the signal function at the key colors
(and for fluids with a gas phase the flash parameters), such that the detected mass
(single-phase: volume) in each region of interest matches the injected amount. The
state holds what the calibration window shows and the calculations behind it; the
window itself lives in `HeterogeneousColorToMassAnalysis.manual_calibration`.

"""

import logging
from typing import TYPE_CHECKING, Tuple

import numpy as np

import darsia
from darsia.presets.workflows.simple_run_analysis import (
    SimpleMassAnalysisResults,
    SimpleRunAnalysis,
)
from darsia.signals.models.pwtransformation import (
    monotonicity_message,
    monotonicity_violation,
)

if TYPE_CHECKING:
    from darsia.presets.workflows.heterogeneous_color_to_mass_analysis import (
        HeterogeneousColorToMassAnalysis,
    )

DEFAULT_CONTOUR_THRESHOLD = 0.05
"""Start value (and reset value) of the contour thresholds."""


class CalibrationState:
    """Data and calculations of the manual calibration, independent of the figure.

    The calibrated parameters themselves are those of the analysis (signal functions
    of the labels, flash); they are changed in place by :meth:`apply_parameters`.

    Parameters
    ----------
    analysis : HeterogeneousColorToMassAnalysis
        The analysis to calibrate.
    images : list[darsia.Image]
        The calibration images.
    experiment : darsia.ProtocolledExperiment
        Provides the injection protocol and the pressure-temperature protocol.
    rois : dict, optional
        Named regions of interest; the entire frame is added as "full".

    """

    def __init__(
        self,
        analysis: "HeterogeneousColorToMassAnalysis",
        images: list[darsia.Image],
        experiment: darsia.ProtocolledExperiment,
        rois: dict[str, darsia.VoxelArray | darsia.CoordinateArray] | None = None,
    ) -> None:
        self.analysis = analysis
        self.images = images
        self.experiment = experiment

        # Fill up rois with "entire frame"
        rois = rois or {}
        assert "" not in rois
        rois.update(
            {
                "full": darsia.CoordinateArray(
                    [
                        analysis.color_analysis.base.origin,
                        analysis.color_analysis.base.opposite_corner,
                    ]
                )
            }
        )
        self.rois = rois

        # ! ---- SELECTION ----

        self.image_idx = 0
        """Index of the displayed image."""
        self.label_idx = 0
        """Label whose signal function is tuned."""

        # ! ---- SERIES OF THE MASS ANALYSIS (per roi) ----

        self.times: dict[str, list] = {r: [] for r in rois}
        """Times in hours."""
        self.expected_mass: dict[str, list] = {r: [] for r in rois}
        """Injected mass (single-phase: volume)."""
        self.integrated_mass: dict[str, list] = {r: [] for r in rois}
        """Detected mass (single-phase: volume)."""
        self.integrated_mass_g: dict[str, list] = {r: [] for r in rois}
        self.integrated_mass_aq: dict[str, list] = {r: [] for r in rois}

        # ! ---- CHOICES FOR THE SIGNAL PANEL ----

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
        if not analysis.has_gas:
            # Single-phase (tracer): no gas saturation or CO2 properties; the signal
            # is the aqueous concentration.
            signal_options = {
                key: label
                for key, label in signal_options.items()
                if key not in ("s_g", "rho_co2(g)", "solubility_co2(aq)")
            }
            signal_options["rho_co2"] = "Mass Density"
            signal_options["c_aq"] = "Concentration"
        self.signal_options = signal_options
        """Keys and titles of the maps which can be shown in the signal panel."""
        self.signal_option_idx = 0
        self.signal_option_ptr = list(signal_options.keys())

        # ! ---- IMAGES -----

        # Set up parameters for coarse visualization
        labels = analysis.labels
        coarse_rows = max(200, labels.img.shape[0] // 4)
        coarse_cols = int(labels.img.shape[1] / labels.img.shape[0] * coarse_rows)
        self.coarse_shape = tuple[int]((coarse_rows, coarse_cols))

        # Coarsen the image and labels for better visualization
        self.coarse_labels = darsia.resize(
            labels,
            shape=self.coarse_shape,
            interpolation="inter_nearest",
        )
        self.coarse_images = [
            darsia.resize(image, shape=self.coarse_shape) for image in images
        ]
        self.coarse_depth = darsia.resize(
            analysis.geometry.depth, shape=self.coarse_shape
        )
        self.coarse_porosity = darsia.resize(
            analysis.geometry.porosity, shape=self.coarse_shape
        )
        self.coarse_effective_depth = darsia.resize(
            analysis.geometry.weight, shape=self.coarse_shape
        )

        # ! ---- COLOR INTERPRETATIONS ----

        # The projection onto the color path is expensive, so the color interpretations
        # of the images are cached. Without projection, they are merely the baseline
        # relative colors (3 channels, i.e., larger), cheap to compute and not cached.
        self._color_interpretations = (
            [analysis.call_color_interpretation(image) for image in images]
            if analysis.evaluation_mode.projects_onto_path
            else None
        )
        self.coarse_color_interpretations = [
            self._display_color_interpretation(self.color_interpretation_of(idx))
            for idx in range(len(images))
        ]

        # ! ---- INITIAL STATE (restored by the Reset button) ----

        functions = analysis.signal_model.model[1]
        self.initial_values = {
            label: np.array(functions[label].values, dtype=float)
            for label in functions.keys()
        }
        """Signal values of all labels at the start."""
        self.initial_flash = (
            (
                analysis.flash.min_value_aq,
                analysis.flash.max_value_aq,
                analysis.flash.min_value_g,
                analysis.flash.max_value_g,
            )
            if analysis.has_gas
            else None
        )
        """Flash parameters at the start (None without a gas phase)."""
        self.default_contour_threshold = DEFAULT_CONTOUR_THRESHOLD
        self.contour_thresholds = (
            [DEFAULT_CONTOUR_THRESHOLD, DEFAULT_CONTOUR_THRESHOLD]
            if analysis.has_gas
            else [DEFAULT_CONTOUR_THRESHOLD]
        )
        """Current contour thresholds (c_aq, and s_g if there is a gas phase)."""

        # ! ---- RESULTS ----

        # Label-independent results: the maps of the current image, and the mass
        # series. The maps are replaced whenever the image or the parameters change,
        # and the mass series whenever the parameters change.
        self.maps: dict | None = None
        """Maps of the current image, see :meth:`compute_maps`."""
        self.mass_analysis_done = False
        """Whether the mass series have been computed at least once."""

    # ! ---- COLOR INTERPRETATIONS ----

    def color_interpretation_of(self, idx: int) -> darsia.Image:
        """Color interpretation of an image (cached if expensive)."""
        if self._color_interpretations is not None:
            return self._color_interpretations[idx]
        return self.analysis.call_color_interpretation(self.images[idx])

    def _display_color_interpretation(self, color_interpretation: darsia.Image):
        """Scalar color interpretation for display (colors: size of the color)."""
        if color_interpretation.img.ndim == 3:
            color_interpretation = darsia.ScalarImage(
                np.linalg.norm(color_interpretation.img, axis=-1),
                **color_interpretation.metadata(),
            )
        return darsia.resize(color_interpretation, shape=self.coarse_shape)

    # ! ---- SELECTION ----

    @property
    def coarse_image(self) -> darsia.Image:
        """Coarse version of the current image."""
        return self.coarse_images[self.image_idx]

    @property
    def signal_key(self) -> str:
        """Key of the map shown in the signal panel."""
        return self.signal_option_ptr[self.signal_option_idx]

    @property
    def signal_title(self) -> str:
        """Title of the map shown in the signal panel."""
        return self.signal_options[self.signal_key]

    @property
    def coarse_signal(self) -> darsia.Image:
        """Map shown in the signal panel (requires :meth:`compute_maps`)."""
        assert self.maps is not None
        return self.maps["signals"][self.signal_key]

    def signal_range(self) -> Tuple[float, float]:
        """Color scale of the signal panel: fixed for the signal, else min-max."""
        if self.signal_key == "pH":
            return 0.0, 1.0
        coarse_signal = self.coarse_signal
        return np.nanmin(coarse_signal.img), np.nanmax(coarse_signal.img)

    def next_image(self) -> int:
        """Move to the next image (cyclic)."""
        self.image_idx = (self.image_idx + 1) % len(self.images)
        return self.image_idx

    def next_signal(self) -> int:
        """Move to the next map for the signal panel (cyclic)."""
        self.signal_option_idx = (self.signal_option_idx + 1) % len(
            self.signal_option_ptr
        )
        return self.signal_option_idx

    def next_label(self) -> int:
        """Move to the next label with a signal function (cyclic)."""
        available_labels = np.sort(list(self.analysis.signal_model.model[1].keys()))
        available_labels = available_labels[available_labels >= 0]
        position = np.where(available_labels == self.label_idx)[0]
        next_position = (
            (position[0] + 1) % len(available_labels) if len(position) > 0 else 0
        )
        self.label_idx = available_labels[next_position]
        return self.label_idx

    def select_label_in(self, box: Tuple[slice, slice] | np.ndarray) -> int:
        """Select the most frequent label within a region of the labels image."""
        self.label_idx = np.argmax(np.bincount(self.analysis.labels.img[box].ravel()))
        return self.label_idx

    # ! ---- MAPS OF THE CURRENT IMAGE ----

    def detailed_mass_analysis(
        self, _color_interpretation: darsia.Image
    ) -> Tuple[darsia.Image, darsia.Image, darsia.Image, darsia.Image]:
        """Signal, mass density, aqueous concentration and gas saturation."""
        analysis = self.analysis
        pH = analysis.call_pH_analysis(_color_interpretation)
        c_aq, s_g = analysis.flash(pH)
        if analysis.expert_knowledge_adapter is not None:
            c_aq = analysis.expert_knowledge_adapter.apply(c_aq, "concentration_aq")
            s_g = analysis.expert_knowledge_adapter.apply(s_g, "saturation_g")
        mass_analysis_result: SimpleMassAnalysisResults = (
            analysis.fluid_model.mass_analysis(
                c_aq=c_aq,
                s_g=s_g,
            )
        )
        density = mass_analysis_result.mass
        return pH, density, c_aq, s_g

    def compute_maps(self) -> dict:
        """Analyze the current image at full resolution, coarsened for display.

        The thermodynamic state is set for the date of the current image.

        """
        analysis = self.analysis
        image_idx = self.image_idx
        analysis.update_state(
            self.images[image_idx].date, self.experiment.pressure_temperature_protocol
        )
        pH, density, c_aq, s_g = self.detailed_mass_analysis(
            self.color_interpretation_of(image_idx)
        )

        def coarsen(img: darsia.Image) -> darsia.Image:
            return darsia.resize(img, shape=self.coarse_shape)

        signals = {
            "color": self.coarse_color_interpretations[image_idx],
            "pH": coarsen(pH),
            "rho_co2": coarsen(density),
            "c_aq": coarsen(c_aq),
            "labels": self.coarse_labels,
            "depth": self.coarse_depth,
            "porosity": self.coarse_porosity,
            "effective_depth": self.coarse_effective_depth,
        }
        if analysis.has_gas:
            co2_g_density = darsia.full_like(pH, analysis.fluid_model.density_gas)
            co2_aq_solubility = darsia.full_like(
                pH, analysis.fluid_model.aqueous_capacity
            )
            signals["s_g"] = coarsen(s_g)
            signals["rho_co2(g)"] = coarsen(co2_g_density)
            signals["solubility_co2(aq)"] = coarsen(co2_aq_solubility)

        self.maps = {
            "image_idx": image_idx,
            "pH": pH,
            "c_aq": coarsen(c_aq),
            "s_g": coarsen(s_g),
            "signals": signals,
        }
        return self.maps

    def _contour_overlay(
        self,
        coarse_image: darsia.Image,
        coarse_c_aq: darsia.Image,
        coarse_s_g: darsia.Image,
        thresholds: list[float],
    ) -> darsia.Image:
        """Contours of the phase segmentation on top of the coarse image."""
        masks = [
            coarse_c_aq > thresholds[0],
            coarse_c_aq > 0.1,
            coarse_c_aq > 0.25,
            coarse_c_aq > 0.5,
            coarse_c_aq > 0.9,
        ]
        colors = [(255, 0, 0)] * 5
        alphas = [0.2, 0.3, 0.4, 0.5, 1.0]
        if self.analysis.has_gas:
            masks.append(coarse_s_g > thresholds[1])
            colors.append((0, 255, 0))
            alphas.append(1.0)
        return darsia.plot_contour_on_image(
            coarse_image,
            mask=masks,
            contour_smoother=self.analysis.contour_smoother,
            color=colors,
            alpha=alphas,
            return_image=True,
        )

    def contour_overlay(self, thresholds: list[float] | None = None) -> darsia.Image:
        """Contour overlay of the maps; recomputed only for new thresholds.

        Parameters
        ----------
        thresholds : list[float], optional
            Thresholds (c_aq, and s_g if there is a gas phase); default are the
            current `contour_thresholds`.

        """
        maps = self.maps
        assert maps is not None
        if thresholds is None:
            thresholds = self.contour_thresholds
        key = tuple(float(threshold) for threshold in thresholds)
        if maps.get("contour_key") != key:
            maps["contour"] = self._contour_overlay(
                self.coarse_images[maps["image_idx"]],
                maps["c_aq"],
                maps["s_g"],
                thresholds,
            )
            maps["contour_key"] = key
        return maps["contour"]

    def label_highlight(self, coarse_image: darsia.Image, label: int) -> np.ndarray:
        """Coarse image with everything outside the label turned (mostly) gray."""
        highlighted = coarse_image.img.copy()
        outside = self.coarse_labels.img != label
        highlighted[outside] = (
            0.7 * np.mean(highlighted[outside], axis=-1, keepdims=True)
            + 0.3 * coarse_image.img[outside]
        )
        return highlighted

    # ! ---- MASS ANALYSIS ----

    @darsia.timing_decorator
    def update_mass_analysis(self) -> None:
        """Update the series (per roi) of detected and injected mass over all images."""
        analysis = self.analysis
        experiment = self.experiment
        rois = self.rois

        run_analysis = {
            roi_key: SimpleRunAnalysis(analysis.geometry.subregion(roi))
            for roi_key, roi in rois.items()
        }
        # Single-phase (tracer): the reference is the injected volume, compared
        # with the integrated aqueous concentration (a volume).
        volume_series = {
            roi_key: {"time": [], "exact": [], "detected": []} for roi_key in rois
        }
        for idx, img in enumerate(self.images):
            color_interpretation = self.color_interpretation_of(idx)

            # Update thermodynamic state
            analysis.update_state(img.date, experiment.pressure_temperature_protocol)

            # Signal analysis
            pH = analysis.call_pH_analysis(color_interpretation)

            # Mass computation
            mass_analysis_result = analysis.call_flash_and_mass_analysis(pH)

            for roi_key, roi in rois.items():
                # Subregion mass
                roi_mass_analysis_result = mass_analysis_result.subregion(roi)

                assert isinstance(roi, darsia.CoordinateArray)
                if not analysis.has_gas:
                    series = volume_series[roi_key]
                    series["time"].append(roi_mass_analysis_result.time)
                    series["exact"].append(
                        experiment.injection_protocol.injected_volume(
                            date=img.date, roi=roi
                        )
                    )
                    series["detected"].append(
                        run_analysis[roi_key].geometry.integrate(
                            roi_mass_analysis_result.concentration_aq
                        )
                    )
                    continue

                # Compute expected mass
                roi_exact_mass = experiment.injection_protocol.injected_mass(
                    date=img.date, roi=roi
                )

                # Track result
                run_analysis[roi_key].track(
                    roi_mass_analysis_result, exact_mass=roi_exact_mass
                )

        # Monitor mass evolution over time
        for roi_key in rois:
            if analysis.has_gas:
                data = run_analysis[roi_key].data
                self.times[roi_key] = [t / 3600 for t in data.time]
                self.expected_mass[roi_key] = data.exact_mass_tot
                self.integrated_mass[roi_key] = data.mass_tot
                self.integrated_mass_g[roi_key] = data.mass_g
                self.integrated_mass_aq[roi_key] = data.mass_aq
            else:
                # For tracer the "mass" series hold volumes (m3); there is no gas.
                series = volume_series[roi_key]
                self.times[roi_key] = [t / 3600 for t in series["time"]]
                self.expected_mass[roi_key] = series["exact"]
                self.integrated_mass[roi_key] = series["detected"]
                self.integrated_mass_g[roi_key] = [0.0] * len(series["time"])
                self.integrated_mass_aq[roi_key] = series["detected"]

            # Errors
            relative_error = (
                np.array(self.integrated_mass[roi_key])
                - np.array(self.expected_mass[roi_key])
            ) / np.abs(self.expected_mass[roi_key]).clip(min=1e-8)

            logging.info(
                f"""Mass analysis for roi {roi_key} updated.\nNew relative """
                f"""error: {relative_error}"""
            )
        self.mass_analysis_done = True

    def max_expected_mass(self) -> float:
        """Largest injected amount over all rois and images (0 without data)."""
        return max(
            (
                max(self.expected_mass[k])
                for k in self.rois
                if len(self.expected_mass[k]) > 0
            ),
            default=0.0,
        )

    def max_time(self) -> float:
        """Latest time (hours) over all rois (0 without data)."""
        return max(
            (max(self.times[k]) for k in self.rois if len(self.times[k]) > 0),
            default=0.0,
        )

    # ! ---- PARAMETERS ----

    def signal_function(self, label: int | None = None):
        """Signal function of a label (default: the selected label)."""
        label = self.label_idx if label is None else label
        return self.analysis.signal_model.model[1][label]

    def check_values(self, values: list[float], label: int | None = None) -> str | None:
        """Message if the signal values of the label decrease, else None."""
        violation = monotonicity_violation(
            values, getattr(self.signal_function(label), "support_indices", None)
        )
        if violation is None:
            return None
        return monotonicity_message(values, violation)

    def apply_parameters(
        self,
        values: list[float],
        flash_values: list[float] | None = None,
        label: int | None = None,
    ) -> bool | None:
        """Set the signal values of a label and the flash parameters.

        Parameters
        ----------
        values : list[float]
            Signal values at the key colors of the label.
        flash_values : list[float], optional
            Min aq, max aq, min g, max g; ignored without a gas phase.
        label : int, optional
            Label; default is the selected one.

        Returns
        -------
        bool or None
            Whether any value changed, or None if the values are invalid, in which
            case nothing is applied.

        """
        if self.check_values(values, label) is not None:
            return None
        analysis = self.analysis
        function = self.signal_function(label)
        need_update = False
        need_update = need_update or any(
            [values[i] != function.values[i] for i in range(len(function.values))]
        )
        if analysis.has_gas:
            assert flash_values is not None
            need_update = need_update or any(
                [
                    flash_values[0] != analysis.flash.min_value_aq,
                    flash_values[1] != analysis.flash.max_value_aq,
                    flash_values[2] != analysis.flash.min_value_g,
                    flash_values[3] != analysis.flash.max_value_g,
                ]
            )

        # Update parameters
        function.update(values=list(values))
        if analysis.has_gas:
            analysis.flash.update(
                min_value_aq=flash_values[0],
                max_value_aq=flash_values[1],
                min_value_g=flash_values[2],
                max_value_g=flash_values[3],
            )
        return need_update
