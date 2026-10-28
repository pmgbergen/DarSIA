"""Reference-only optical hypotheses for diagnostic CO2 concentration work.

Clear water is the sand/illumination background, NOT a known CO2-saturated
endpoint. These dimensionless signals still require an independent optical or
physical concentration calibration. Neither method uses current plume colors,
phase masks, injected mass, timestamps, or holdout images to fit its reference.

Under a multiplicative, linear-light image model, log(current / baseline)
cancels the static sand reflectance, and log(clear / baseline) measures the
initial indicator's optical response. The second ratio is regularized because
near-zero reference spans otherwise amplify noise into apparent concentration.
This is an optical hypothesis, not a claim that JPEG camera RGB obeys a
quantitative Beer-Lambert law.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

METHODS = ("joint_chromatic_od", "joint_vector_od")
_Q = np.array([.5, .5, -1.], dtype=np.float32)
RGB_ENCODINGS = ("srgb", "camera_linear")


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Decode normalized sRGB channel values into float32 linear-light RGB.

    The standard piecewise transfer preserves zero and unit endpoints; it
    does not apply camera correction, phase inference, or concentration fitting.
    """
    x = np.asarray(rgb, dtype=np.float32)
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def _image(rgb: np.ndarray, shape=None) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != 3 or (shape is not None and x.shape != shape):
        raise ValueError("RGB must have shape H x W x 3 on the reference grid")
    # Registered float images can overshoot by < 1e-3 during resampling. Accept
    # that roundoff only; uint8-range data must not be silently reinterpreted.
    if not np.isfinite(x).all() or np.min(x) < -.001 or np.max(x) > 1.001:
        raise ValueError("RGB must be finite normalized unit values in [0, 1]")
    return np.clip(x, 0, 1)


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(x, (0, 0), sigma) if sigma > 0 else x.copy()


def _chromatic(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=-1, keepdims=True)


@dataclass(frozen=True)
class JointReference:
    baseline_log: np.ndarray
    denominators: dict[str, np.ndarray]
    reliability: dict[str, np.ndarray]
    direction: np.ndarray
    diagnostics: dict[str, Any]
    sigma: float
    linear_floor: float
    usable: bool
    rgb_encoding: str = "srgb"


def _regularized_span(raw: np.ndarray, valid: np.ndarray, facies: np.ndarray,
                      sigma: float, min_span: float):
    """Shrink noisy reference-only local spans toward robust facies priors.

    Priors normalize indicator response, not plume concentration. A facies with
    no reliable positive reference gets the pooled prior and reliability zero.
    No method enforces equal means in current concentration/plume images.
    """
    positive = valid & (raw > min_span)
    pooled = float(np.median(raw[positive])) if positive.any() else 1.0
    # A relative floor bounds gain at 4 times the typical reference response.
    floor = max(min_span, .25 * pooled)
    local_weight = _blur(valid.astype(np.float32), sigma)
    mean = _blur(np.where(valid, raw, 0).astype(np.float32), sigma) / np.maximum(local_weight, .01)
    prior = np.empty(raw.shape, np.float32)
    reliability = np.zeros(raw.shape, np.float32)
    records = []
    for label in np.unique(facies):
        mask = facies == label
        supported = mask & positive
        count = int(supported.sum())
        median = float(np.median(raw[supported])) if count else pooled
        mad = float(np.median(np.abs(raw[supported] - median))) if count else 0.0
        prior[mask] = max(median, floor)
        if count:
            # Symmetric robust down-weighting of registration/texture outliers.
            spread = max(1.4826 * mad, .2 * median, min_span)
            agreement = 1 / (1 + ((mean[mask] - median) / (3 * spread)) ** 2)
            strength = np.clip((mean[mask] - min_span) / max(median - min_span, min_span), 0, 1)
            reliability[mask] = local_weight[mask] * agreement * strength
        records.append({"facies": int(label), "pixels": int(mask.sum()),
                        "positive_reference_fraction": count / int(mask.sum()),
                        "prior_span": median, "reference_mad": mad,
                        "pooled_fallback": count == 0})
    denominator = reliability * np.maximum(mean, floor) + (1 - reliability) * prior
    denominator = np.maximum(denominator, floor).astype(np.float32)
    return denominator, reliability, {"pooled_positive_span": pooled, "denominator_floor": floor,
                                     "positive_reference_fraction": float(positive.mean()),
                                     "facies": records}


def build_joint_reference(baseline: np.ndarray, clearwater: np.ndarray,
                          facies: np.ndarray, *, sigma: float,
                          linear_floor: float, min_span: float,
                          domain_mask: np.ndarray | None = None,
                          rgb_encoding: str = "srgb") -> JointReference:
    """Freeze two robust reference normalizers without seeing injection frames.

    ``reliability`` is confidence in a local optical denominator, not phase
    probability or concentration. Zero reliability falls back to a reference
    prior. If no positive blue-indicator reference exists anywhere, ``usable``
    is False and signal generation returns zero (the caller must flag failure).
    """
    if rgb_encoding not in RGB_ENCODINGS:
        raise ValueError("Joint optical RGB encoding must be srgb or camera_linear")
    baseline_clipped_fraction = float(np.mean((np.asarray(baseline) < 0) | (np.asarray(baseline) > 1)))
    clearwater_clipped_fraction = float(np.mean((np.asarray(clearwater) < 0) | (np.asarray(clearwater) > 1)))
    baseline = _image(baseline)
    clearwater = _image(clearwater, baseline.shape)
    facies = np.asarray(facies)
    if facies.shape != baseline.shape[:2] or not np.isfinite(facies).all():
        raise ValueError("Facies must be a finite H x W reference label array")
    if sigma < 0 or linear_floor <= 0 or min_span <= 0:
        raise ValueError("Invalid regularization parameters")
    domain = np.ones(facies.shape, bool) if domain_mask is None else np.asarray(domain_mask, bool)
    if domain.shape != facies.shape or not domain.any():
        raise ValueError("Domain must have reference H x W shape and contain pixels")
    bl, cl = ((srgb_to_linear(baseline), srgb_to_linear(clearwater))
              if rgb_encoding == "srgb" else (baseline, clearwater))
    baseline_log = np.log(np.maximum(bl, linear_floor)).astype(np.float32)
    clear_log = np.log(np.maximum(cl, linear_floor))
    photometric_valid = ((bl.min(axis=-1) > linear_floor) & (cl.min(axis=-1) > linear_floor)
                        & (baseline.max(axis=-1) < .995) & (clearwater.max(axis=-1) < .995))
    valid = photometric_valid & domain
    ref = _chromatic(clear_log - baseline_log)
    fixed_raw = ref @ _Q
    vector_support = valid & (fixed_raw > min_span)
    # One shared direction: robust component medians within each facies, then
    # equal-weight median across facies. Bright/large layers cannot dominate it.
    layer_vectors = [np.median(ref[(facies == label) & vector_support], axis=0)
                     for label in np.unique(facies)
                     if np.any((facies == label) & vector_support)]
    direction = np.median(layer_vectors, axis=0) if layer_vectors else _Q.copy()
    direction = _chromatic(direction.reshape(1, 1, 3)).reshape(3)
    direction = (direction / max(float(np.linalg.norm(direction)), 1e-12)).astype(np.float32)
    if direction @ _Q < 0:
        direction = -direction
    denominators, reliabilities, method_records = {}, {}, {}
    for method, raw in zip(METHODS, (fixed_raw, ref @ direction)):
        denom, rel, record = _regularized_span(raw, valid, facies, sigma, min_span)
        denominators[method], reliabilities[method], method_records[method] = denom, rel, record
    usable = bool(vector_support.any())
    return JointReference(baseline_log, denominators, reliabilities, direction,
                          {"clearwater_is_saturated_CO2_endpoint": False,
                           "fit_inputs": ["baseline", "clearwater", "facies"],
                           "rgb_encoding": rgb_encoding,
                           "usable": usable, "photometric_valid_fraction": float(photometric_valid[domain].mean()),
                           "domain_fraction": float(domain.mean()),
                           "baseline_resampling_clipped_fraction": baseline_clipped_fraction,
                           "clearwater_resampling_clipped_fraction": clearwater_clipped_fraction,
                           "shared_chromatic_direction": direction.tolist(),
                           "methods": method_records}, sigma, linear_floor, usable,
                          rgb_encoding)


def joint_optical_signals(current: np.ndarray, reference: JointReference) -> dict[str, np.ndarray]:
    """Return signed dimensionless signals, preserving baseline zero exactly.

    Do not interpret value 1 as physical CO2 saturation: it is a reference
    response scale. Do not multiply by reliability (that would darken less
    reliable sand layers systematically). Concentration clipping/calibration is
    the caller's responsibility. Negative optical changes remain available for
    diagnostics instead of being silently converted into acidification.
    """
    current = _image(current, reference.baseline_log.shape)
    if reference.rgb_encoding not in RGB_ENCODINGS:
        raise ValueError("Unknown joint optical RGB encoding")
    linear = (srgb_to_linear(current) if reference.rgb_encoding == "srgb" else current)
    delta = np.log(np.maximum(linear, reference.linear_floor)) - reference.baseline_log
    delta = _chromatic(delta)
    result = {}
    for method, direction in zip(METHODS, (_Q, reference.direction)):
        numerator = _blur(delta @ direction, reference.sigma)
        value = numerator / reference.denominators[method]
        result[method] = (value if reference.usable else np.zeros_like(value)).astype(np.float32)
    return result
