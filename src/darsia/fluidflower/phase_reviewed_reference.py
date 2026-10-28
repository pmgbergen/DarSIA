"""Review-calibrated acceptance of C2 aqueous proposals; frozen legacy runtimes.

This gate neither estimates concentration nor changes established C1 or gas
labels. Its score is a discriminant, not a calibrated physical probability.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from . import phase_reference_support as rs
from .project import sha256_file

VERSION = "review-calibrated-reference-gate-v1"
PROFILE_FILE = "phase_reviewed_reference.json"
FEATURE_VERSION = "registered-relative-chroma-1024-v1"
FEATURE_NAMES = ("delta_red_fraction", "delta_green_fraction", "delta_blue_fraction", "chroma_distance")


def change_fields(baseline, current):
    """Identical, bounded-resolution RGB averaging for fitting and prediction.

    No absolute hue or frame number enters the learned decision. The 1024-wide
    field reduces grain/sensor noise; native signed color evidence is still
    required by the inherited C2 proposal generator.
    """
    shape = np.asarray(baseline).shape[:2]
    b, c = (rs.rgb_float(v, shape) for v in (baseline, current))
    width = min(1024, shape[1])
    size = (width, max(1, round(shape[0]*width/shape[1])))
    b, c = (cv2.GaussianBlur(cv2.resize(v, size, interpolation=cv2.INTER_AREA),
                            (0, 0), sigmaX=1., sigmaY=1.) for v in (b, c))
    delta = c / np.maximum(c.sum(-1, keepdims=True), 1e-6) - b / np.maximum(b.sum(-1, keepdims=True), 1e-6)
    magnitude = np.linalg.norm(delta, axis=-1)
    features = np.concatenate((delta, magnitude[..., None]), axis=-1)
    ratio = c.mean(-1) / np.maximum(b.mean(-1), 1e-6)
    available = (b.mean(-1) >= .025) & (ratio >= .45) & (ratio <= 2.2)
    return features, available


def scores(features, calibration):
    x = np.asarray(features, np.float32)
    fit = calibration["fit"]
    z = (x - np.asarray(fit["mean"], np.float32)) / np.asarray(fit["scale"], np.float32)
    logits = z @ np.asarray(fit["coefficients"], np.float32) + float(fit["intercept"])
    return (1 / (1 + np.exp(-np.clip(logits, -30, 30)))).astype(np.float32)


def validate_profile(profile):
    if profile.get("format") != VERSION:
        raise ValueError("Unknown reviewed-reference profile")
    parent = rs.validate_profile(profile.get("reference_support", {}))
    if profile.get("runs") != parent["runs"] or profile.get("settings", {}) != parent.get("settings", {}):
        raise ValueError("Inherited contexts cannot change implicitly")
    calibration = profile["calibration"]
    if calibration.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Reference feature version differs from calibration")
    if {key.split('/')[0] for key in profile["runs"]} != {calibration.get("series")}:
        raise ValueError("Reference calibration belongs to another experimental series")
    if not isinstance(calibration.get("enabled"), bool):
        raise ValueError("Calibration must explicitly declare whether additions are enabled")
    if not 0 < calibration["threshold"] <= 1:
        raise ValueError("Invalid reviewed-reference threshold")
    if float(np.float32(calibration["threshold"])) != calibration["threshold"]:
        raise ValueError("Reference threshold must be exactly representable in float32")
    fit = calibration["fit"]
    for key in ("mean", "scale", "coefficients"):
        a = np.asarray(fit[key])
        if a.shape != (len(FEATURE_NAMES),) or not np.isfinite(a).all():
            raise ValueError("Invalid fitted reference coefficients")
    if min(fit["scale"]) <= 0 or not np.isfinite(fit["intercept"]):
        raise ValueError("Invalid fitted reference normalization")
    return profile


def load_profile(root):
    path = Path(root) / PROFILE_FILE
    return validate_profile(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else None


def runtime_identity(profile):
    validate_profile(profile)
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True).encode())
    digest.update(rs.runtime_identity(profile["reference_support"]).encode())
    digest.update(sha256_file(Path(__file__)).encode())
    return digest.hexdigest()


class ReviewedReferenceSequence(rs.ReferenceSequence):
    def __init__(self, context, profile):
        validate_profile(profile)
        super().__init__(context, profile["reference_support"])
        self.calibration = profile["calibration"]

    def bind_reference(self, baseline_rgb):
        super().bind_reference(baseline_rgb)
        self.calibration_baseline = np.asarray(baseline_rgb).copy()

    def apply_with_reference(self, predictions, elapsed_h, current_rgb):
        inherited, report = super().apply_with_reference(predictions, elapsed_h, current_rgb)
        features, available = change_fields(self.calibration_baseline, current_rgb)
        accepted = available & (features[..., 3] > 1e-7) & (scores(features, self.calibration) >= self.calibration["threshold"])
        if not self.calibration["enabled"]:
            accepted[:] = False
        native_accept = cv2.resize(accepted.astype(np.uint8), self.context.domain.shape[::-1],
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
        proposed = self.last_diagnostics["added_aqueous"].copy()
        rejected = proposed & ~native_accept
        adjusted = inherited["co2_presence_probability"].copy()
        # Restore the actual C1 score, not a made-up probability below 0.5.
        adjusted[rejected] = self.parent.last_diagnostics["context_presence_score"][rejected]
        conditional = inherited["gas_probability_conditional_on_co2"]
        free = adjusted*conditional
        result = {**inherited, "co2_presence_probability": adjusted,
                  "untouched_water_probability": 1-adjusted,
                  "aqueous_co2_probability": adjusted-free, "free_co2_gas_probability": free}
        self.last_diagnostics.update(added_aqueous=np.zeros_like(proposed), reference_accepted=proposed & native_accept,
            suppressed_presence=rejected, reference_rejected=rejected,
            unresolved=self.last_diagnostics["unresolved"] | rejected,
            spatial_only_presence=self.last_diagnostics["spatial_only_presence"] & ~rejected)
        for key in ("reference_frontier", "reference_residual"):
            self.last_diagnostics[key] &= native_accept
        report["reviewed_reference"] = dict(format=VERSION, calibration_enabled=self.calibration["enabled"],
            threshold=self.calibration["threshold"], rejected_c2_pixels=int(rejected.sum()),
            accepted_c2_pixels=int((proposed & native_accept).sum()),
            optical_weights_retrained=False, established_c1_labels_unchanged=True,
            gas_labels_unchanged=True, calibrated_physical_probability=False)
        return result, report
