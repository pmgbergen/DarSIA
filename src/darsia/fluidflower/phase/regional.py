"""Frozen regional optical support and gas confirmation on registered grids."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
import cv2
import numpy as np
from .heads import regional_probability

@dataclass(frozen=True)
class Settings:
    """Explicit physical scales and reconstruction thresholds for optical support."""
    working_spacing_m: float
    fine_sigma_m: float
    coarse_sigma_m: float
    support_threshold: float
    seed_threshold: float
    minimum_seed_density: float
    change_noise_multiplier: float
    minimum_chromatic_change: float

    def __post_init__(self):
        if not all((np.isfinite(v) for v in {name: getattr(self, name) for name in self.parameters['settings_fields']}.values())):
            raise ValueError('Region settings must be finite')
        if not (0 < self.working_spacing_m <= self.coarse_sigma_m <= self.parameters['bounds']['maximum_scale_m'] and 0 < self.fine_sigma_m <= self.coarse_sigma_m and (0 < self.support_threshold < self.seed_threshold < 1) and (0 < self.minimum_seed_density < 1) and (self.change_noise_multiplier >= 0) and (0 < self.minimum_chromatic_change < 1)):
            raise ValueError('Invalid optical region scales or thresholds')
    parameters: Mapping = field(compare=False, repr=False)

    @classmethod
    def from_mapping(cls, document, *, parameters):
        """Use original model values and explicit original defaults without casting."""
        values = dict(parameters['settings'])
        values.update(document)
        return cls(**values, parameters=parameters)

@dataclass(frozen=True)
class GasSettings:
    """Explicit categorical confirmation or physical component-area rule."""
    format: str
    threshold: float

    def __post_init__(self):
        if self.format != self.parameters['formats']['gas'] or not np.isfinite(self.threshold) or (not 0 < self.threshold < 1):
            raise ValueError('Invalid learned regional gas settings')
    parameters: Mapping = field(compare=False, repr=False)

    @classmethod
    def from_mapping(cls, document, *, parameters):
        """Retain original frozen guard values supplied as data."""
        values = dict(parameters['guard_defaults']['gas'])
        values.update(document)
        return cls(**values, parameters=parameters)

@dataclass(frozen=True)
class OutputSettings:
    """Explicit categorical confirmation or physical component-area rule."""
    format: str
    minimum_island_area_mm2: float

    def __post_init__(self):
        if self.format != self.parameters['formats']['output'] or not np.isfinite(self.minimum_island_area_mm2) or (not 0 < self.minimum_island_area_mm2 <= self.parameters['bounds']['maximum_guard_area_mm2']):
            raise ValueError('Invalid regional output guard settings')
    parameters: Mapping = field(compare=False, repr=False)

    @classmethod
    def from_mapping(cls, document, *, parameters):
        """Retain original frozen guard values supplied as data."""
        values = dict(parameters['guard_defaults']['output'])
        values.update(document)
        return cls(**values, parameters=parameters)

@dataclass(frozen=True)
class FreeGasSettings:
    """Explicit categorical confirmation or physical component-area rule."""
    format: str
    minimum_area_mm2: float

    def __post_init__(self):
        if self.format != self.parameters['formats']['free_gas'] or not np.isfinite(self.minimum_area_mm2) or (not 0 < self.minimum_area_mm2 <= self.parameters['bounds']['maximum_guard_area_mm2']):
            raise ValueError('Invalid free-gas minimum settings')
    parameters: Mapping = field(compare=False, repr=False)

    @classmethod
    def from_mapping(cls, document, *, parameters):
        """Retain original frozen guard values supplied as data."""
        values = dict(parameters['guard_defaults']['free_gas'])
        values.update(document)
        return cls(**values, parameters=parameters)

def prepare(current, baseline, presence, conditional_gas, domain, spacing_m, settings=None, *, parameters: Mapping):
    """Return a low-memory global working grid plus evidence for reconstruction."""
    if settings is None:
        settings = Settings.from_mapping({}, parameters=parameters)
    valid = np.asarray(domain, bool)
    p, g = (np.asarray(presence), np.asarray(conditional_gas))
    spacing = np.asarray(spacing_m, float)
    if valid.ndim != 2 or p.shape != valid.shape or g.shape != valid.shape:
        raise ValueError('Registered scores and domain must share a 2D grid')
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Positive native y/x spacing is required')
    if any((not np.isfinite(a).all() or (a < 0).any() or (a > 1).any() for a in (p, g))):
        raise ValueError('Optical inputs must be probabilities')
    shape = tuple((max(1, round(n * s / settings.working_spacing_m)) for n, s in zip(valid.shape, spacing)))
    actual_spacing = spacing * np.asarray(valid.shape) / shape
    size = shape[::-1]

    def reduce(a):
        'Validate and apply explicit frozen optical data without fitting or changing pixels.'
        return cv2.resize(np.asarray(a, np.float32), size, interpolation=cv2.INTER_AREA)
    coverage = reduce(valid)
    dom = coverage > parameters['pooling']['coverage_threshold']

    def pool(a):
        'Validate and apply explicit frozen optical data without fitting or changing pixels.'
        return np.divide(reduce(np.asarray(a, np.float32) * valid), coverage, out=np.zeros(shape, np.float32), where=coverage > parameters['pooling']['division_floor'])
    reduced_p = pool(p)
    reduced_gas = pool(p * g)
    density = pool(p >= parameters['pooling']['positive_probability'])
    strong = pool(p >= parameters['pooling']['strong_probability'])

    def rgb(a):
        'Validate and apply explicit frozen optical data without fitting or changing pixels.'
        a = np.asarray(a)
        if a.shape != (*valid.shape, 3):
            raise ValueError('Current and reference must be registered RGB images')
        if a.dtype != np.uint8 and (not np.isfinite(a).all() or (a < 0).any() or (a > 1).any()):
            raise ValueError('RGB must be uint8 or finite [0,1]')
        factor = 1 / parameters['pooling']['review_denominator'] if a.dtype == np.uint8 else 1
        return np.stack([pool(a[..., k]) * factor for k in range(3)], axis=-1)
    cur, ref = (rgb(current), rgb(baseline))
    fields, guide, noise_floor, seed_density = ([], None, None, None)
    for sigma in (settings.fine_sigma_m, settings.coarse_sigma_m):
        sy, sx = sigma / actual_spacing
        weight = cv2.GaussianBlur(coverage, (0, 0), float(sx), sigmaY=float(sy))

        def blur(a):
            'Validate and apply explicit frozen optical data without fitting or changing pixels.'
            return cv2.GaussianBlur(a * coverage, (0, 0), float(sx), sigmaY=float(sy)) / np.maximum(weight, parameters['pooling']['division_floor'])
        c, b = [np.stack([blur(a[..., k]) for k in range(3)], axis=-1) for a in (cur, ref)]
        delta = c / np.maximum(c.sum(-1, keepdims=True), parameters['pooling']['chromatic_floor']) - b / np.maximum(b.sum(-1, keepdims=True), parameters['pooling']['chromatic_floor'])
        quiet = dom & (reduced_p < parameters['pooling']['quiet_probability'])
        samples = delta[quiet]
        drift = np.median(samples, axis=0) if samples.shape[0] >= parameters['pooling']['quiet_min_count'] else np.zeros(3)
        delta = (delta - drift).astype(np.float32)
        magnitude = np.linalg.norm(delta, axis=-1)
        relative = np.log(np.maximum(c.mean(-1), parameters['pooling']['chromatic_floor']) / np.maximum(b.mean(-1), parameters['pooling']['chromatic_floor']))
        if quiet.sum() >= parameters['pooling']['quiet_min_count']:
            relative -= np.median(relative[quiet])
        den = blur(density)
        fields.extend((blur(reduced_p), blur(reduced_gas), den, blur(strong), *[delta[..., k] for k in range(3)], relative, magnitude, den * magnitude))
        if guide is None:
            guide = magnitude
            noise = float(np.median(magnitude[quiet])) if quiet.sum() >= parameters['pooling']['quiet_min_count'] else 0.0
            noise_floor = max(settings.minimum_chromatic_change, settings.change_noise_multiplier * noise)
            seed_density = den
    return dict(features=np.stack(fields, axis=-1), domain=dom, change=guide, change_floor=noise_floor, density=seed_density, native_shape=valid.shape, spacing_m=tuple(actual_spacing))

def reconstruct(p, prepared, settings=None, *, parameters: Mapping):
    """Fill connected optical support; never join across a rejected water path."""
    if settings is None:
        settings = Settings.from_mapping({}, parameters=parameters)
    domain = prepared['domain']
    if p.shape != domain.shape or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid regional prediction')
    changed = prepared['change'] >= prepared['change_floor']
    eligible = domain & changed & (p >= settings.support_threshold)
    seeds = eligible & (p >= settings.seed_threshold) & (prepared['density'] >= settings.minimum_seed_density)
    count, components = cv2.connectedComponents(eligible.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    keep[np.unique(components[seeds])] = True
    keep[0] = False
    return keep[components]

def predict(prepared, model, raw_labels, domain, *, parameters: Mapping):
    'Apply original packed optical scoring to prepared columns without changing the current grid.'
    if model['format'] != parameters['model_format'] or tuple(model['feature_names']) != tuple(parameters['feature_names']):
        raise ValueError('Incompatible optical region model')
    settings = Settings(**model['settings'], parameters=parameters)
    co2 = probability(prepared['features'], model['heads']['co2_presence'], parameters=parameters)
    connected = reconstruct(co2, prepared, settings, parameters=parameters)
    field = np.where(connected, co2, 0).astype(np.float32)
    h, w = raw_labels.shape
    support = cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR) >= settings.support_threshold
    support &= domain
    gas = raw_labels == 2
    labels = np.where(gas, 2, support).astype(np.uint8)
    labels[~domain] = raw_labels[~domain]
    return (labels, dict(added_pixels=int(((labels == 1) & (raw_labels == 0)).sum()), removed_pixels=int(((labels == 0) & (raw_labels == 1)).sum()), gas_changed_pixels=int(((labels == 2) != gas).sum()), change_floor=prepared['change_floor'], context_map_used=False))

def build_shared_regional_context(current, baseline, presence, domain, spacing_m, settings=None, *, parameters: Mapping) -> RegionalGasContext:
    """Run unchanged regional preparation once and cache its two blur weights."""
    if settings is None:
        settings = Settings.from_mapping({}, parameters=parameters)
    p = np.asarray(presence)
    valid = np.asarray(domain, bool)
    prepared = prepare(current, baseline, p, np.zeros_like(p), valid, spacing_m, settings, parameters=parameters)
    shape = prepared['domain'].shape
    coverage = cv2.resize(np.asarray(valid, np.float32), shape[::-1], interpolation=cv2.INTER_AREA)
    actual_spacing = np.asarray(prepared['spacing_m'], float)
    scales = []
    for sigma in (settings.fine_sigma_m, settings.coarse_sigma_m):
        sy, sx = sigma / actual_spacing
        weight = cv2.GaussianBlur(coverage, (0, 0), float(sx), sigmaY=float(sy))
        scales.append((float(sx), float(sy), np.maximum(weight, parameters['pooling']['division_floor'])))
    return RegionalGasContext(prepared, p, valid, coverage, tuple(scales))

def prepare_conditional_gas(context: RegionalGasContext, conditional_gas, *, parameters: Mapping) -> dict:
    """Recompute exactly the two gas-dependent fields on this frozen context."""
    gas = np.asarray(conditional_gas)
    if gas.shape != context.native_domain.shape or not np.isfinite(gas).all() or (gas < 0).any() or (gas > 1).any():
        raise ValueError('Conditional gas must match the fixed native probability grid')
    coverage = context.coverage
    shape = coverage.shape
    reduced_joint = np.divide(cv2.resize(np.asarray(context.presence * gas, np.float32) * context.native_domain, shape[::-1], interpolation=cv2.INTER_AREA), coverage, out=np.zeros(shape, np.float32), where=coverage > parameters['pooling']['division_floor'])
    weighted_joint = reduced_joint * coverage
    features = context.prepared['features'].copy()
    for column, (sigma_x, sigma_y, denominator) in zip(parameters['conditional_columns'], context.scales):
        features[..., column] = cv2.GaussianBlur(weighted_joint, (0, 0), sigma_x, sigmaY=sigma_y) / denominator
    return dict(context.prepared, features=features)

def confirm_gas(labels, regional_probability, domain, settings=None, *, parameters: Mapping):
    'Reclassify gas lacking the original interpolated regional confirmation as aqueous.'
    labels = np.asarray(labels)
    if settings is None:
        settings = GasSettings.from_mapping({}, parameters=parameters)
    values = np.asarray(regional_probability, np.float32)
    domain = np.asarray(domain, bool)
    if labels.ndim != 2 or labels.shape != domain.shape or (not np.isin(labels, (0, 1, 2)).all()) or (values.ndim != 2) or (not values.size) or (not np.isfinite(values).all()) or (values < 0).any() or (values > 1).any():
        raise ValueError('Invalid native labels, domain or regional gas probabilities')
    scores = cv2.resize(values, labels.shape[::-1], interpolation=cv2.INTER_LINEAR)
    rejected = domain & (labels == 2) & (scores < settings.threshold)
    result = labels.astype(np.uint8, copy=True)
    result[rejected] = 1
    return (result, rejected)

def guard_labels(labels, domain, spacing_m, settings=None, *, parameters: Mapping):
    'Remove physically small disconnected CO2 islands and single-row raster seams.'
    labels, domain = (np.asarray(labels), np.asarray(domain, bool))
    if settings is None:
        settings = OutputSettings.from_mapping({}, parameters=parameters)
    spacing = np.asarray(spacing_m, float)
    if labels.ndim != 2 or labels.shape != domain.shape or (not np.isin(labels, (0, 1, 2)).all()):
        raise ValueError('Guard requires same-grid categorical phase labels and domain')
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Guard requires positive registered y/x spacing')
    valid = (labels > 0) & domain
    count, components, stats, _ = cv2.connectedComponentsWithStats(valid.astype(np.uint8), connectivity=8)
    area_mm2 = float(np.prod(spacing) * 1000000.0)
    keep = stats[:, cv2.CC_STAT_AREA] * area_mm2 >= settings.minimum_island_area_mm2
    h, w = labels.shape
    border_seams = (stats[:, cv2.CC_STAT_WIDTH] == 1) & np.isin(stats[:, cv2.CC_STAT_LEFT], (0, w - 1)) | (stats[:, cv2.CC_STAT_HEIGHT] == 1) & np.isin(stats[:, cv2.CC_STAT_TOP], (0, h - 1))
    border_seams[0] = False
    keep[border_seams] = False
    keep[0] = False
    retained = keep[components]
    result = np.where(retained, labels, 0).astype(np.uint8)
    return (result, dict(format=parameters['formats']['output'], pixel_area_mm2=area_mm2, minimum_island_area_mm2=settings.minimum_island_area_mm2, removed_border_seams=int(border_seams.sum()), excluded_domain_pixels=int(((labels > 0) & ~domain).sum()), removed_islands=int((~keep[1:]).sum()), retained_islands=int(keep.sum()), removed_island_pixels=int((valid & ~retained).sum()), gas_removed_pixels=int(((labels == 2) & ~retained).sum()), aqueous_removed_pixels=int(((labels == 1) & ~retained).sum()), added_pixels=0, source_labels_modified=False))

def clean_labels(labels, spacing_m, settings=None, *, parameters: Mapping):
    'Withdraw four-connected free-gas components below the supplied physical minimum.'
    labels, spacing = (np.asarray(labels), np.asarray(spacing_m, float))
    if settings is None:
        settings = FreeGasSettings.from_mapping({}, parameters=parameters)
    if labels.ndim != 2 or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError('Free-gas cleanup requires categorical phase labels')
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Free-gas cleanup requires positive registered spacing')
    _, components, stats, _ = cv2.connectedComponentsWithStats((labels == 2).astype(np.uint8), connectivity=4)
    area = float(np.prod(spacing) * 1000000.0)
    small = stats[:, cv2.CC_STAT_AREA] * area < settings.minimum_area_mm2
    small[0] = False
    withdrawn = small[components]
    result = labels.astype(np.uint8, copy=True)
    result[withdrawn] = 1
    return (result, dict(format=parameters['formats']['free_gas'], minimum_area_mm2=settings.minimum_area_mm2, connectivity=4, pixel_area_mm2=area, withdrawn_gas_components=int(small.sum()), reclassified_gas_pixels=int(withdrawn.sum()), co2_support_unchanged=True))

def weighted_co2_output_guard(labels, valid, pixel_area_mm2, *, parameters: Mapping):
    """Preserve the inherited eight-connected island and raster-seam rules."""
    from .metrics import validate_pixel_area
    values = np.asarray(labels)
    valid = np.asarray(valid)
    if values.dtype != np.uint8 or valid.dtype != bool or values.shape != valid.shape:
        raise ValueError('CO2 output guard requires the exact native label/mask grid')
    area = validate_pixel_area(pixel_area_mm2, values.shape)
    co2 = np.isin(values, (1, 2)) & valid
    count, components, stats, _ = cv2.connectedComponentsWithStats(co2.astype(np.uint8), connectivity=8)
    if np.isscalar(area):
        physical = stats[:, cv2.CC_STAT_AREA].astype(np.float64) * area
    else:
        physical = np.bincount(components.ravel(), weights=np.asarray(area).ravel(), minlength=count)
    keep = physical >= parameters['weighted_guard']['minimum_area_mm2']
    h, w = values.shape
    seams = (stats[:, cv2.CC_STAT_WIDTH] == 1) & np.isin(stats[:, cv2.CC_STAT_LEFT], (0, w - 1)) | (stats[:, cv2.CC_STAT_HEIGHT] == 1) & np.isin(stats[:, cv2.CC_STAT_TOP], (0, h - 1))
    seams[0] = False
    keep[seams] = False
    keep[0] = False
    result = np.where(keep[components], values, 0).astype(np.uint8)
    return (result, dict(connectivity=8, minimum_island_area_mm2=parameters['weighted_guard']['minimum_area_mm2'], removed_border_seams=int(seams.sum()), removed_islands=int((~keep[1:]).sum()), removed_island_pixels=int(np.count_nonzero(co2 & ~keep[components])), area_weighted=True, added_pixels=0))

@dataclass(frozen=True)
class RegionalGasContext:
    prepared: dict[str, Any]
    presence: np.ndarray
    native_domain: np.ndarray
    coverage: np.ndarray
    scales: tuple[tuple[float, float, np.ndarray], ...]

def probability(features, fit, *, parameters):
    """The frozen regional head's float32 folded-weight scoring order."""
    return regional_probability(features, fit, parameters={'regional_feature_names': parameters['feature_names']})

def original_co2_labels(current, baseline, presence, generic_gas, domain, spacing_m, regional_model, profile, *, parameters: Mapping):
    """The original regional chain, with original generic gas as evidence."""
    settings = Settings(**regional_model['settings'], parameters=parameters)
    raw = np.where(presence >= parameters['partition']['presence_threshold'], np.where(generic_gas >= parameters['partition']['gas_threshold'], 2, 1), 0).astype(np.uint8)
    prepared = prepare(current, baseline, presence, generic_gas, domain, spacing_m, settings, parameters=parameters)
    labels, audit = predict(prepared, regional_model, raw, domain, parameters=parameters)
    if profile.get(parameters['profile_keys']['gas']) is not None:
        conditional = probability(prepared['features'], regional_model['heads']['gas_within_co2'], parameters=parameters)
        labels, rejected = confirm_gas(labels, conditional, domain, GasSettings(**profile[parameters['profile_keys']['gas']], parameters=parameters), parameters=parameters)
        audit['regional_gas_reclassified_pixels'] = int(rejected.sum())
    if profile.get(parameters['profile_keys']['output']) is not None:
        labels, audit['output_guards'] = guard_labels(labels, domain, spacing_m, OutputSettings(**profile[parameters['profile_keys']['output']], parameters=parameters), parameters=parameters)
    if profile.get(parameters['profile_keys']['free_gas']) is not None:
        labels, audit['free_gas_cleanup'] = clean_labels(labels, spacing_m, FreeGasSettings(**profile[parameters['profile_keys']['free_gas']], parameters=parameters), parameters=parameters)
    labels = labels.copy()
    labels[~domain] = 255
    audit.update(generic_gas_used_for_co2_context=True, g3_used_for_co2_context=False, original_profile_guards_only=True)
    return (labels, audit)
