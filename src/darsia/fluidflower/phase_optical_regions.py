"""Trainable regional optical evidence, followed by seeded reconstruction.

The frozen pixel model is an input, never a training label. All neighbourhood
statistics are computed on the complete registered frame in physical units.
Neither an old context map nor a plume envelope is an input to this model.
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.special import expit

VERSION = "baseline-relative-optical-regions-v1"


@dataclass(frozen=True)
class Settings:
    working_spacing_m: float = .001
    fine_sigma_m: float = .00075
    coarse_sigma_m: float = .002
    support_threshold: float = .5
    seed_threshold: float = .8
    minimum_seed_density: float = .05
    change_noise_multiplier: float = 3.
    minimum_chromatic_change: float = .008

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Region settings must be finite")
        if not (0 < self.working_spacing_m <= self.coarse_sigma_m <= .01
                and 0 < self.fine_sigma_m <= self.coarse_sigma_m
                and 0 < self.support_threshold < self.seed_threshold < 1
                and 0 < self.minimum_seed_density < 1
                and self.change_noise_multiplier >= 0
                and 0 < self.minimum_chromatic_change < 1):
            raise ValueError("Invalid optical region scales or thresholds")


FEATURE_NAMES = tuple(f"{scale}_{name}" for scale in ("fine", "coarse") for name in (
    "presence_mean", "gas_joint_mean", "positive_density", "strong_density",
    "red_fraction_change", "green_fraction_change", "blue_fraction_change",
    "relative_intensity_change", "chromatic_change_magnitude", "density_times_change"))


def prepare(current, baseline, presence, conditional_gas, domain, spacing_m, settings=Settings()):
    """Return a low-memory global working grid plus evidence for reconstruction."""
    valid = np.asarray(domain, bool)
    p, g = np.asarray(presence), np.asarray(conditional_gas)
    spacing = np.asarray(spacing_m, float)
    if valid.ndim != 2 or p.shape != valid.shape or g.shape != valid.shape:
        raise ValueError("Registered scores and domain must share a 2D grid")
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Positive native y/x spacing is required")
    if any(not np.isfinite(a).all() or (a < 0).any() or (a > 1).any() for a in (p, g)):
        raise ValueError("Optical inputs must be probabilities")
    shape = tuple(max(1, round(n*s/settings.working_spacing_m)) for n, s in zip(valid.shape, spacing))
    actual_spacing = spacing*np.asarray(valid.shape)/shape
    size = shape[::-1]
    def reduce(a):
        return cv2.resize(np.asarray(a, np.float32), size, interpolation=cv2.INTER_AREA)
    coverage = reduce(valid)
    dom = coverage > .999
    def pool(a):
        return np.divide(reduce(np.asarray(a, np.float32)*valid), coverage,
                         out=np.zeros(shape, np.float32), where=coverage > 1e-6)
    reduced_p = pool(p)
    reduced_gas = pool(p*g)
    density = pool(p >= .5)
    strong = pool(p >= .9)
    def rgb(a):
        a = np.asarray(a)
        if a.shape != (*valid.shape, 3):
            raise ValueError("Current and reference must be registered RGB images")
        if a.dtype != np.uint8 and (not np.isfinite(a).all() or (a < 0).any() or (a > 1).any()):
            raise ValueError("RGB must be uint8 or finite [0,1]")
        factor = 1/255 if a.dtype == np.uint8 else 1
        return np.stack([pool(a[..., k])*factor for k in range(3)], axis=-1)
    cur, ref = rgb(current), rgb(baseline)
    fields, guide, noise_floor, seed_density = [], None, None, None
    for sigma in (settings.fine_sigma_m, settings.coarse_sigma_m):
        sy, sx = sigma/actual_spacing
        weight = cv2.GaussianBlur(coverage, (0, 0), float(sx), sigmaY=float(sy))
        def blur(a):
            return cv2.GaussianBlur(a*coverage, (0, 0), float(sx), sigmaY=float(sy))/np.maximum(weight, 1e-6)
        c, b = [np.stack([blur(a[..., k]) for k in range(3)], axis=-1) for a in (cur, ref)]
        delta = c/np.maximum(c.sum(-1, keepdims=True), .01)-b/np.maximum(b.sum(-1, keepdims=True), .01)
        # Robust global drift is estimated from optically quiet current pixels.
        quiet = dom & (reduced_p < .01)
        samples = delta[quiet]
        drift = np.median(samples, axis=0) if samples.shape[0] >= 32 else np.zeros(3)
        delta = (delta-drift).astype(np.float32)
        magnitude = np.linalg.norm(delta, axis=-1)
        relative = np.log(np.maximum(c.mean(-1), .01)/np.maximum(b.mean(-1), .01))
        if quiet.sum() >= 32:
            relative -= np.median(relative[quiet])
        den = blur(density)
        fields.extend((blur(reduced_p), blur(reduced_gas), den, blur(strong),
                       *[delta[..., k] for k in range(3)], relative, magnitude, den*magnitude))
        if guide is None:
            guide = magnitude
            noise = float(np.median(magnitude[quiet])) if quiet.sum() >= 32 else 0.
            noise_floor = max(settings.minimum_chromatic_change, settings.change_noise_multiplier*noise)
            seed_density = den
    return dict(features=np.stack(fields, axis=-1), domain=dom,
                change=guide, change_floor=noise_floor, density=seed_density,
                native_shape=valid.shape, spacing_m=tuple(actual_spacing))


def probability(features, fit):
    values = np.asarray(features, np.float32)
    mean, scale, coef = (np.asarray(fit[k], np.float32) for k in ("mean", "scale", "coefficients"))
    if any(a.shape != (len(FEATURE_NAMES),) for a in (mean, scale, coef)) or (scale <= 0).any():
        raise ValueError("Region head differs from its feature contract")
    return expit(values @ (coef/scale) + float(fit["intercept"])-mean @ (coef/scale)).astype(np.float32)


def reconstruct(p, prepared, settings=Settings()):
    """Fill connected optical support; never join across a rejected water path."""
    domain = prepared["domain"]
    if p.shape != domain.shape or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid regional prediction")
    changed = prepared["change"] >= prepared["change_floor"]
    eligible = domain & changed & (p >= settings.support_threshold)
    seeds = eligible & (p >= settings.seed_threshold) & (prepared["density"] >= settings.minimum_seed_density)
    count, components = cv2.connectedComponents(eligible.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    keep[np.unique(components[seeds])] = True
    keep[0] = False
    return keep[components]


def predict(prepared, model, raw_labels, domain):
    if model["format"] != VERSION or tuple(model["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Incompatible optical region model")
    settings = Settings(**model["settings"])
    co2 = probability(prepared["features"], model["heads"]["co2_presence"])
    connected = reconstruct(co2, prepared, settings)
    # Upsample the continuous level set before thresholding, avoiding blocky
    # working-grid pixels. Rejected components cannot grow back on upsampling.
    field = np.where(connected, co2, 0).astype(np.float32)
    h, w = raw_labels.shape
    support = cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR) >= settings.support_threshold
    support &= domain
    # This experiment changes aqueous extent only. Gas classification remains
    # the frozen optical result; the coupled gas head supplies training evidence.
    gas = raw_labels == 2
    labels = np.where(gas, 2, support).astype(np.uint8)
    labels[~domain] = raw_labels[~domain]
    return labels, dict(added_pixels=int(((labels == 1) & (raw_labels == 0)).sum()),
        removed_pixels=int(((labels == 0) & (raw_labels == 1)).sum()),
        gas_changed_pixels=int(((labels == 2) != gas).sum()), change_floor=prepared["change_floor"],
        context_map_used=False)
