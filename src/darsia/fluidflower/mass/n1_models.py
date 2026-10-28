"""Typed saved-optics and thermodynamics for the native N1 mass method.

Models are supplied as measured data. This module neither fits a model nor
imports a campaign, a controller, or a result validator.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from types import SimpleNamespace
import numpy as np
import darsia


@dataclass
class FrameData:
    """Registered scientific buffers and measured physical state of one frame."""
    run_id: str
    frame_index: int
    current_rgb: np.ndarray
    baseline_rgb: np.ndarray
    clear_water_rgb: np.ndarray
    clear_water_supported_mask: np.ndarray
    phase_labels: np.ndarray
    facies: np.ndarray
    domain: np.ndarray
    supported_mask: np.ndarray
    mass_eligible_mask: np.ndarray
    depth_m: np.ndarray
    porosity: np.ndarray
    pixel_area_m2: np.ndarray
    dimensions_m: tuple
    pressure_bar: float
    temperature_c: float
    elapsed_h: float
    nominal_injected_mass_g: float | None
    recipe: dict
    provenance: dict
    working_dtype: str = "float32"
    pressure_gradient_bar_per_m: float = 0.
    temperature_gradient_c_per_m: float = 0.
    excluded_masks: dict | None = None
    unknown_physical_mask: np.ndarray | None = None
    # Static expert knowledge per mode, as native boolean masks; the frame's
    # phase map is its dynamic expert knowledge (see phase_expert_knowledge).
    expert_static_masks: dict | None = None

    @cached_property
    def expert_knowledge(self):
        """This frame's static and dynamic (phase-map) expert knowledge."""
        from .phase_expert_knowledge import PhaseExpertKnowledge
        return PhaseExpertKnowledge(self.phase_labels, self.expert_static_masks)

    def __post_init__(self):
        """Require actual native categories, support and working precision."""
        shape = self.phase_labels.shape
        if self.working_dtype not in {"float32", "float64"}:
            raise ValueError("An explicit working pixel dtype is required")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("An exact nonnegative frame index is required")
        if (self.phase_labels.ndim != 2 or self.phase_labels.dtype != np.uint8
                or not np.isin(self.phase_labels, (0, 1, 2, 255)).all()):
            raise ValueError("Phase labels must retain uint8 categories 0/1/2/255")
        for name in ("current_rgb", "baseline_rgb", "clear_water_rgb"):
            value = getattr(self, name)
            if value.dtype != np.dtype(self.working_dtype) or value.shape != (*shape, 3):
                raise TypeError("Scientific RGB buffers differ from working precision/grid")
        for name in ("domain", "supported_mask", "clear_water_supported_mask", "mass_eligible_mask"):
            value = getattr(self, name)
            if value.shape != shape or value.dtype != bool:
                raise TypeError(name + " must be a native boolean mask")
        if (self.facies.shape != shape or not np.issubdtype(self.facies.dtype, np.integer)
                or np.any(self.facies < 0) or np.any(self.mass_eligible_mask & ~self.domain)):
            raise ValueError("Registered facies and mass support are required")
        if self.unknown_physical_mask is not None and (
                self.unknown_physical_mask.dtype != bool or self.unknown_physical_mask.shape != shape):
            raise TypeError("Unknown physical support must be explicit")
        physical = self.known_physical_domain
        if not physical.any():
            raise ValueError("No measured physical domain is available")
        for name in ("depth_m", "porosity", "pixel_area_m2"):
            value = np.broadcast_to(np.asarray(getattr(self, name)), shape)
            selected = self.domain if name == "pixel_area_m2" else physical
            if not np.isfinite(value[selected]).all() or np.any(value[selected] <= 0):
                raise ValueError("Positive measured " + name + " is required")
            if name == "porosity" and np.any(value > 1):
                raise ValueError("Porosity exceeds one")
        if (len(self.dimensions_m) != 2 or not np.isfinite(self.dimensions_m).all()
                or min(self.dimensions_m) <= 0 or self.pressure_bar <= 0
                or not np.isfinite([self.pressure_bar, self.temperature_c, self.elapsed_h]).all()):
            raise ValueError("Measured dimensions, state and elapsed time are required")
        observed = self.domain & self.supported_mask
        for value in (self.current_rgb, self.baseline_rgb):
            if not np.isfinite(value[observed]).all() or np.any(value[observed] < 0):
                raise ValueError("Scientific supported RAW has invalid values")
        if self.nominal_injected_mass_g is not None and (
                not np.isfinite(self.nominal_injected_mass_g) or self.nominal_injected_mass_g < 0):
            raise ValueError("Injected mass must be a measured nonnegative input")
        covered = np.zeros(shape, bool)
        for mask in (self.excluded_masks or {}).values():
            if (mask.dtype != bool or mask.shape != shape or np.any(mask & ~self.domain)
                    or np.any(mask & covered)):
                raise ValueError("Excluded masks must be disjoint on the native domain")
            covered |= mask
        if np.any(covered & self.mass_eligible_mask):
            raise ValueError("Excluded support entered the main mass")

    @property
    def known_physical_domain(self):
        """Independently measured positive physical support; no imputation."""
        depth = np.broadcast_to(np.asarray(self.depth_m), self.domain.shape)
        pore = np.broadcast_to(np.asarray(self.porosity), self.domain.shape)
        known = self.domain & np.isfinite(depth) & np.isfinite(pore) & (depth > 0) & (pore > 0)
        if self.unknown_physical_mask is not None:
            known &= ~self.unknown_physical_mask
        return known


class TypedColor:
    """Saved ColorPath interpretation using the chosen pixel provider."""
    def __init__(self, original, math, *, capture=None, label=None):
        """Retain the saved supports, values and ignore-spectrum semantics.

        ``capture`` (optional) is a dict shared by this model and the copies a
        HeterogeneousModel makes of it; each call stores the computed path
        parameter there under ``label``. Nothing is computed differently.
        """
        self.original, self.math = original, math
        self.capture, self.label = capture, label

    def __call__(self, colors):
        """Use the original fit/interpolation and ignore threshold/order."""
        math, original = self.math, self.original
        math.require_pixels(colors)
        path = original.color_path
        supports = path.colors if original.color_mode == darsia.ColorMode.ABSOLUTE else path.relative_colors
        if original.ignore_spectrum is None or colors.ndim == 1:
            parameter = math.fit_color_path(colors, supports, path.equidistant_distances)
        else:
            mask = np.linalg.norm(colors, axis=1) > 1e-1
            parameter = np.zeros(colors.shape[0], dtype=math.dtype)
            parameter[mask] = math.fit_color_path(colors[mask], supports, path.equidistant_distances)
            sub_colors = colors[mask]
            sub_parameter = math.fit_color_path(sub_colors, supports, path.equidistant_distances)
            parameter[mask] = sub_parameter
        if self.capture is not None:
            if self.label in self.capture:
                raise ValueError("The colour-path coordinate of a facies was captured twice")
            self.capture[self.label] = parameter
        return math.interpret_color_path(parameter, path.equidistant_distances, original.values)


class TypedSignal:
    """Saved piecewise signal transformation with unchanged endpoints."""
    def __init__(self, original, math):
        """Retain all frozen small-table values in their original dtype."""
        self.original, self.math = original, math
        self.supports, self.values = original.supports, original.values

    def __call__(self, values):
        """Apply the provider's original endpoint-error transformation."""
        return self.math.piecewise_linear(values, self.supports, self.values, endpoint="error")


class TypedFacies(darsia.HeterogeneousModel):
    """Per-facies scatter into an explicitly typed pixel output buffer."""
    def __init__(self, models, labels, ignored, math):
        """Use the native categorical partition and original model objects."""
        super().__init__(models, labels, ignore_labels=ignored)
        self.math, self.working_labels = math, labels.img

    def __call__(self, signal):
        """Keep the original per-label gather/model/scatter operation."""
        return self.math.per_facies(signal, self.working_labels, self.obj, ignore_labels=self.ignore_labels)


class TypedClip(darsia.ClipModel):
    """Native signal clipping with local bounds cast before pixel math."""
    def __init__(self, low, high, math):
        """Retain the intersection of the saved signal supports."""
        super().__init__(low, high)
        self.math = math

    def __call__(self, image):
        """Clip arrays/images without float64 pixel promotion."""
        low = None if self._min_value is None else self.math.dtype(self._min_value)
        high = None if self._max_value is None else self.math.dtype(self._max_value)
        if isinstance(image, np.ndarray):
            self.math.require_pixels(image)
            return np.clip(image, low, high)
        if isinstance(image, darsia.Image):
            self.math.require_pixels(image.img)
            result = image.copy()
            result.img = np.clip(result.img, low, high)
            return result
        raise ValueError("Invalid native signal type")


class TypedPhysics(darsia.CO2MassAnalysis):
    """Original frozen thermodynamic tables, chosen precision pixel fields."""
    def atmospheric_temperature_map(self):
        """Preserve the original pressure/temperature pixel operation order."""
        math = self.pixel_math
        height = math.pixels(self.height_map)
        return np.ones_like(height)*math.dtype(self.atmospheric_temperature) + (
            math.dtype(self.atmospheric_temperature_gradient)*height)

    def setup_density_gaseous_co2(self):
        """Keep float64 lookup tables and provider hydrostatic pixel kernels."""
        from scipy import interpolate
        self.co2_density_interpolator_20 = interpolate.interp1d(*self.data_NIST_20, kind="linear")
        self.co2_density_interpolator_23 = interpolate.interp1d(*self.data_NIST_23, kind="linear")
        self.co2_solubility_interpolator_20 = interpolate.interp1d(*self.solubility_co2_20, kind="linear")
        self.co2_solubility_interpolator_23 = interpolate.interp1d(*self.solubility_co2_23, kind="linear")
        result = self.pixel_math.hydrostatic_fields(temperature=self.atmospheric_temperature_map(),
            height=self.height_map, top_pressure=self.top_atmospheric_pressure(),
            water_density_20=self.water_density_20, water_density_23=self.water_density_23,
            density_20=self.data_NIST_20, density_23=self.data_NIST_23,
            solubility_20=self.solubility_co2_20, solubility_23=self.solubility_co2_23)
        self.density_gaseous_co2, self.solubility_co2 = result["density"], result["solubility"]


class NativeAnalysis:
    """Native saved optical pipeline with explicit data and a pixel provider."""
    def __init__(self, frame, baseline, models, math, *, domain=None, depth=None,
                 porosity=None, capture_colour_coordinate=False):
        """Build the same native ConcentrationAnalysis/Geometry objects.

        ``capture_colour_coordinate`` keeps each facies' colour-path parameter
        of the next interpretation for :meth:`colour_path_coordinate`.
        """
        self.math, self.models = math, models
        self.shape, self.dimensions = frame.domain.shape, frame.dimensions_m
        self.domain = frame.known_physical_domain if domain is None else domain
        self.labels = frame.facies
        depth = frame.depth_m if depth is None else depth
        porosity = frame.porosity if porosity is None else porosity
        depth = np.broadcast_to(np.asarray(depth, np.float64), self.shape).copy()
        pore = np.broadcast_to(np.asarray(porosity, np.float64), self.shape).copy()
        depth = np.where(self.domain, depth, 0.)
        pore = np.where(self.domain, pore, 0.)
        self.geometry = darsia.ExtrudedPorousGeometry(porosity=pore, depth=depth,
            space_dim=2, num_voxels=self.shape, dimensions=list(self.dimensions))
        nominal_volume = self.geometry.voxel_volume.copy()
        nominal_area = float(np.prod(self.dimensions))/np.prod(self.shape)
        self.geometry.voxel_volume = nominal_volume*(frame.pixel_area_m2/nominal_area)
        self.geometry.cached_voxel_volume = self.geometry.voxel_volume.copy()
        labels = darsia.ScalarImage(self.labels, dimensions=list(self.dimensions))
        unique = set(map(int, np.unique(self.labels)))
        if not unique.issubset(models["metadata"]["label_ids"]):
            raise ValueError("Per-frame facies are outside the saved native basis")
        ignored = list(map(int, models["metadata"]["ignore_labels"]))
        self.colour_capture = {} if capture_colour_coordinate else None
        self.colour_ignored = ignored
        color_models = {label:TypedColor(models["colors"][label], math,
                                         capture=self.colour_capture, label=label)
                        for label in unique}
        signal_models = {label:TypedSignal(models["signals"][label], math) for label in unique}
        options = {"diff option":"plain", "restoration -> model":False}
        color_model = darsia.CombinedModel([TypedFacies(color_models, labels, ignored, math)])
        self.color_analysis = darsia.ConcentrationAnalysis(base=self.rgb(baseline),
            restoration=None, model=color_model, labels=labels, **options)
        low = max(min(model.supports) for model in signal_models.values())
        high = min(max(model.supports) for model in signal_models.values())
        signal_model = darsia.CombinedModel([TypedClip(low, high, math),
            TypedFacies(signal_models, labels, ignored, math)])
        self.signal_analysis = darsia.ConcentrationAnalysis(base=None, restoration=None,
            model=signal_model, labels=labels, **options)

    def rgb(self, values):
        """Unclipped finite nonnegative native optical image."""
        self.math.require_pixels(values)
        if values.shape != (*self.shape, 3) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Finite unclipped scientific RGB is required")
        return darsia.OpticalImage(values.copy(), color_space="RGB", dimensions=list(self.dimensions))

    def scalar(self, values, elapsed_h=None):
        """Native ScalarImage with declared working precision and time."""
        image = darsia.ScalarImage(np.asarray(values, self.math.dtype), dimensions=list(self.dimensions))
        image.time = None if elapsed_h is None else float(elapsed_h)*3600
        return image

    def interpretation(self, rgb):
        """Apply native plain-relative colour then saved signal transforms."""
        return self.signal_analysis(self.color_analysis(self.rgb(rgb)))

    def colour_path_coordinate(self, valid):
        """The captured colour-path parameter of every pixel, float32, NaN elsewhere.

        Each facies' values are placed with the same label mask that the
        per-facies scatter used, so every pixel holds exactly the parameter its
        interpretation consumed. Pixels outside ``valid`` and ignored facies
        are NaN. The capture is released afterwards.
        """
        capture = self.colour_capture
        if capture is None:
            raise ValueError("This analysis was built without colour-coordinate capture")
        coordinate = np.full(self.shape, np.nan, np.float32)
        for label in np.unique(self.labels):
            if int(label) in self.colour_ignored:
                continue
            if int(label) not in capture:
                raise ValueError("A facies' colour-path coordinate was not captured")
            coordinate[self.labels == label] = capture[int(label)]
        coordinate[~np.asarray(valid, bool)] = np.nan
        capture.clear()
        return coordinate


def prepare_response(frame, models, math, swi_by_facies, *, solubility_domain=None,
                     colour_coordinate=False):
    """Interpret RGB with the original titration flash and explicit reference support.

    ``solubility_domain`` may bind the physical support used when the frozen
    calibration was evaluated. Changing the reporting domain must not silently
    change this median and therefore the colour-to-concentration transform.
    It is a reference for the flash only, never a mass or visibility mask.
    Omitting it retains the existing measured physical support exactly.

    ``colour_coordinate=True`` also returns ``colour_path_coordinate``: the
    colour-path parameter this interpretation computed per pixel (float32, NaN
    outside ``quantifiable``). It changes no computed value.
    """
    basis = frame.known_physical_domain if solubility_domain is None else np.asarray(solubility_domain)
    if basis.dtype != bool or basis.shape != frame.domain.shape or not basis.any():
        raise ValueError("Flash solubility support must be a nonempty boolean grid")
    clear = frame.clear_water_rgb
    support = frame.known_physical_domain & frame.supported_mask & frame.clear_water_supported_mask
    support &= np.isfinite(clear).all(axis=2) & (clear > 0).all(axis=2)
    current, baseline = math.reference_normalized(frame.current_rgb, frame.baseline_rgb, clear, support)
    if not np.isfinite(current[support]).all() or not np.isfinite(baseline[support]).all():
        raise ValueError("Nonfinite clear-water normalized signal")
    baseline = np.where(support[..., None], baseline, 0.).astype(math.dtype)
    current = np.where(support[..., None], current, baseline).astype(math.dtype)
    # The default call stays exactly the original one.
    analysis = (NativeAnalysis(frame, baseline, models, math, capture_colour_coordinate=True)
                if colour_coordinate else NativeAnalysis(frame, baseline, models, math))
    phase = np.where(frame.known_physical_domain & (frame.phase_labels != 255), frame.phase_labels, 0).astype(np.uint8)
    scalar = analysis.scalar(np.zeros(phase.shape, math.dtype), frame.elapsed_h)
    # The native constructor invokes setup: attach the provider before it runs.
    physics = TypedPhysics.__new__(TypedPhysics)
    physics.pixel_math = math
    TypedPhysics.__init__(physics, scalar, atmospheric_pressure=frame.pressure_bar,
        atmospheric_temperature=frame.temperature_c,
        atmospheric_pressure_gradient=frame.pressure_gradient_bar_per_m,
        atmospheric_temperature_gradient=frame.temperature_gradient_c_per_m)
    signal = analysis.interpretation(current)
    recipe = {name: frame.recipe.get(name) for name in ("alkalinity_M", "btb_M")}
    if any(value is None or not np.isfinite(value) or value < 0 for value in recipe.values()):
        raise ValueError("Measured titration recipe is required")
    if not np.isfinite(physics.solubility_co2[basis]).all() or np.any(physics.solubility_co2[basis] <= 0):
        raise ValueError("Flash solubility support requires positive finite measured physics")
    solubility_M = float(np.median(physics.solubility_co2[basis]))/44.0095
    flash = darsia.TitrationFlash(**models["metadata"]["flash_bounds"],
        **{name:float(value) for name,value in recipe.items()}, co2_sat_M=solubility_M)
    aqueous, gas = math.titration(signal.img, flash)
    prepared = SimpleNamespace(aqueous_relative=aqueous, gas_saturation=gas, phase_labels=phase)
    swi = np.full(frame.domain.shape, np.nan, math.dtype)
    for label, value in swi_by_facies.items():
        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Measured Swi must be in [0,1]")
        swi[frame.facies == int(label)] = math.dtype(value)
    quantifiable = support & ~np.isin(frame.facies, models["metadata"]["ignore_labels"])
    response = dict(frame=frame, analysis=analysis, physics=physics, flash=flash,
        prepared=prepared, native_signal=np.asarray(signal.img, math.dtype), quantifiable=quantifiable,
        metadata=models["metadata"], swi=swi, precision=math.precision)
    if colour_coordinate:
        response["colour_path_coordinate"] = analysis.colour_path_coordinate(quantifiable)
    return response


def own_colour_response(response, geometry, photo_support, models, math):
    """Original own-colour estimate using the same saved optics and flash."""
    frame = response["frame"]
    observed = (frame.domain & np.asarray(photo_support, bool) & frame.clear_water_supported_mask
        & np.isfinite(frame.clear_water_rgb).all(axis=2)
        & (frame.clear_water_rgb > 0).all(axis=2) & geometry.valid)
    available = observed & ~np.isin(frame.facies, models["metadata"]["ignore_labels"])
    reused = (np.array_equal(geometry.valid & frame.domain, frame.known_physical_domain)
              and np.array_equal(available, response["quantifiable"]))
    if reused:
        aqueous = response["prepared"].aqueous_relative
    else:
        current, baseline = np.zeros_like(frame.current_rgb), np.zeros_like(frame.baseline_rgb)
        np.divide(frame.current_rgb, frame.clear_water_rgb, out=current, where=observed[..., None])
        np.divide(frame.baseline_rgb, frame.clear_water_rgb, out=baseline, where=observed[..., None])
        analysis = NativeAnalysis(frame, baseline, models, math, domain=geometry.valid & frame.domain,
            depth=np.where(geometry.valid, geometry.depth_m, 0.),
            porosity=np.where(geometry.valid, geometry.porosity, 0.))
        aqueous, _ = math.titration(analysis.interpretation(current).img, response["flash"])
    return dict(aqueous_relative=np.asarray(aqueous, math.dtype), available=available,
        audit=dict(native_optics_used=True, original_flash_object_used=True,
            main_geometry_modified=False, colour_paths_modified=False,
            main_analysis_modified=False, measured_colour_pixels=int(available.sum()),
            photo_support_required=True, reused_original_native_interpretation=reused))
