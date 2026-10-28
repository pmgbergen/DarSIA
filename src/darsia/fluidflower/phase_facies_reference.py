"""Facies-specific acceptance of reference-supported aqueous proposals.

This optional final gate cannot create a proposal or change established C1
classes. Thresholds are development settings, not concentration or calibrated
physical probabilities. The C1, C2 and C3 implementations remain immutable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from . import phase_reference_support as rs, phase_reviewed_reference as rr
from .project import sha256_file

VERSION = "facies-reviewed-reference-gate-v1"
PROFILE_FILE = "phase_facies_reference.json"


def validate_profile(profile):
    if profile.get("format") != VERSION:
        raise ValueError("Unknown facies-reference profile")
    rr.validate_profile({**profile, "format": rr.VERSION})
    thresholds = profile["calibration"].get("facies_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("Explicit facies thresholds are required")
    for label, value in thresholds.items():
        if str(int(label)) != label or int(label) < 0:
            raise ValueError("Invalid facies identifier")
        if not np.isfinite(value) or not 0 < value <= 1 or float(np.float32(value)) != value:
            raise ValueError("Facies threshold must be finite and exactly float32")
    return profile


def load_profile(root):
    path = Path(root) / PROFILE_FILE
    return validate_profile(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else None


def runtime_identity(profile):
    validate_profile(profile)
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True).encode())
    digest.update(rs.runtime_identity(profile["reference_support"]).encode())
    digest.update(sha256_file(Path(rr.__file__)).encode())
    digest.update(sha256_file(Path(__file__)).encode())
    return digest.hexdigest()


def accepted_proposals(features, available, facies, proposed, calibration):
    """Resolve thresholds on the native facies grid, without blurring labels."""
    shape = proposed.shape
    if facies.shape != shape or available.shape != features.shape[:2]:
        raise ValueError("Facies/reference fields are not registered")
    coarse_score = rr.scores(features, calibration)
    score = cv2.resize(coarse_score, shape[::-1], interpolation=cv2.INTER_NEAREST)
    exposed = cv2.resize((available & (features[..., 3] > 1e-7)).astype(np.uint8),
                         shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    accepted = proposed & exposed & (score >= calibration["threshold"])
    for label, threshold in calibration["facies_thresholds"].items():
        selected = proposed & exposed & (facies == int(label))
        accepted[selected] = score[selected] >= threshold
    if not calibration["enabled"]:
        accepted[:] = False
    return accepted


class FaciesReferenceSequence(rs.ReferenceSequence):
    def __init__(self, context, profile):
        validate_profile(profile)
        super().__init__(context, profile["reference_support"])
        self.calibration = profile["calibration"]

    def bind_reference(self, baseline_rgb):
        super().bind_reference(baseline_rgb)
        self.calibration_baseline = np.asarray(baseline_rgb).copy()

    def apply_with_reference(self, predictions, elapsed_h, current_rgb):
        inherited, report = super().apply_with_reference(predictions, elapsed_h, current_rgb)
        features, available = rr.change_fields(self.calibration_baseline, current_rgb)
        proposed = self.last_diagnostics["added_aqueous"].copy()
        accepted = accepted_proposals(features, available, self.context.facies, proposed, self.calibration)
        rejected = proposed & ~accepted
        adjusted = inherited["co2_presence_probability"].copy()
        adjusted[rejected] = self.parent.last_diagnostics["context_presence_score"][rejected]
        conditional = inherited["gas_probability_conditional_on_co2"]
        free = adjusted * conditional
        result = {**inherited, "co2_presence_probability": adjusted,
                  "untouched_water_probability": 1-adjusted,
                  "aqueous_co2_probability": adjusted-free, "free_co2_gas_probability": free}
        self.last_diagnostics.update(added_aqueous=np.zeros_like(proposed),
            reference_accepted=accepted, suppressed_presence=rejected, reference_rejected=rejected,
            unresolved=self.last_diagnostics["unresolved"] | rejected,
            spatial_only_presence=self.last_diagnostics["spatial_only_presence"] & ~rejected)
        for key in ("reference_frontier", "reference_residual"):
            self.last_diagnostics[key] &= ~rejected
        report["facies_reference"] = dict(format=VERSION, calibration_enabled=self.calibration["enabled"],
            fallback_threshold=self.calibration["threshold"], facies_thresholds=self.calibration["facies_thresholds"],
            accepted_c2_pixels=int(accepted.sum()), rejected_c2_pixels=int(rejected.sum()),
            established_c1_labels_unchanged=True, gas_labels_unchanged=True,
            optical_weights_retrained=False, calibrated_physical_probability=False)
        return result, report
