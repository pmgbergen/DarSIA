"""Native continuous boundaries over an unchanged causal reference sequence.

The learned gate and reference signal are interpolated before classification.
Only a bounded band around existing C2 proposals may be refined. Native color,
domain, facies, physical distance and connectivity checks still apply; these
new decisions never teach a color direction or update the inherited history.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from . import phase_facies_reference as fr, phase_reference_support as rs, phase_reviewed_reference as rr
from .project import sha256_file

VERSION = "native-continuous-reference-gate-v1"
PROFILE_FILE = "phase_native_reference.json"


@dataclass(frozen=True)
class BoundarySettings:
    maximum_band_cells: float = 1.
    minimum_interpolation_weight: float = 1e-6

    def __post_init__(self):
        if not np.isfinite(self.maximum_band_cells) or not 0 <= self.maximum_band_cells <= 1:
            raise ValueError("Boundary refinement must be bounded to at most one inherited cell")
        if not np.isfinite(self.minimum_interpolation_weight) or not 0 < self.minimum_interpolation_weight <= 1:
            raise ValueError("Invalid minimum interpolation weight")


def validate_profile(profile):
    if profile.get("format") != VERSION:
        raise ValueError("Unknown native-reference profile")
    fr.validate_profile({**profile, "format": fr.VERSION})
    if not isinstance(profile.get("boundary_settings"), dict):
        raise ValueError("Explicit boundary_settings are required")
    BoundarySettings(**profile["boundary_settings"])
    return profile


def load_profile(root):
    path = Path(root) / PROFILE_FILE
    return validate_profile(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else None


def runtime_identity(profile):
    validate_profile(profile)
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True).encode())
    digest.update(rs.runtime_identity(profile["reference_support"]).encode())
    digest.update(sha256_file(Path(rr.__file__)).encode())
    digest.update(sha256_file(Path(fr.__file__)).encode())
    digest.update(sha256_file(Path(__file__)).encode())
    return digest.hexdigest()


def _facies_linear(values, available, facies, minimum_weight=1e-6):
    """Interpolate scalar evidence using only samples of the native facies.

    Facies identifiers themselves are selected with nearest-neighbor sampling,
    never averaged. A thin native facies with no coarse sample fails closed.
    NaNs mark destinations with no available same-facies evidence.
    """
    values = np.asarray(values, np.float32)
    available = np.asarray(available, bool)
    if values.ndim != 2 or available.shape != values.shape or facies.ndim != 2:
        raise ValueError("Continuous fields and facies must be registered two-dimensional arrays")
    native_size = facies.shape[::-1]
    coarse_facies = cv2.resize(np.asarray(facies, np.float32), values.shape[::-1],
                               interpolation=cv2.INTER_NEAREST)
    result = np.full(facies.shape, np.nan, np.float32)
    for label in np.unique(coarse_facies[available]):
        source = available & (coarse_facies == label) & np.isfinite(values)
        if not source.any():
            continue
        weight = cv2.resize(source.astype(np.float32), native_size, interpolation=cv2.INTER_LINEAR)
        numerator = cv2.resize(np.where(source, values, 0), native_size, interpolation=cv2.INTER_LINEAR)
        selected = (facies == label) & (weight >= minimum_weight)
        result[selected] = numerator[selected] / weight[selected]
    return result


def accepted_proposals(features, available, facies, proposed, calibration, *, minimum_interpolation_weight=1e-6):
    """Apply continuous learned evidence and exact native facies thresholds."""
    if facies.shape != proposed.shape or available.shape != features.shape[:2]:
        raise ValueError("Facies/reference fields are not registered")
    if not calibration["enabled"] or not proposed.any():
        return np.zeros_like(proposed, dtype=bool)
    eligible = available & (features[..., 3] > 1e-7)
    score = _facies_linear(rr.scores(features, calibration), eligible, facies, minimum_interpolation_weight)
    accepted = proposed & (score >= calibration["threshold"])
    for label, threshold in calibration["facies_thresholds"].items():
        selected = proposed & (facies == int(label))
        accepted[selected] = score[selected] >= threshold
    return accepted


def _anchored(candidates, anchors, domain):
    """Keep new native components connected without crossing excluded corners."""
    allowed = np.asarray(candidates & domain, np.uint8)
    count, labels = cv2.connectedComponents(allowed, connectivity=4)
    retained = np.zeros(count, bool)
    retained[np.unique(labels[anchors & domain])] = True
    retained[0] = False
    return retained[labels]


class NativeReferenceSequence(rs.ReferenceSequence):
    def __init__(self, context, profile):
        validate_profile(profile)
        super().__init__(context, profile["reference_support"])
        self.calibration = profile["calibration"]
        self.boundary_settings = BoundarySettings(**profile["boundary_settings"])

    def _color_fields(self, rgb):
        # Retain transient fields from the unmodified C2 computation. This does
        # not replace its history, signal thresholds, observations or outputs.
        fields = super()._color_fields(rgb)
        self._current_color_fields = fields
        return fields

    def bind_reference(self, baseline_rgb):
        super().bind_reference(baseline_rgb)
        self.calibration_baseline = np.asarray(baseline_rgb).copy()

    def _refine_proposals(self, proposed, current, elapsed_h, report):
        domain = self.context.domain
        s = self.settings
        b = self.boundary_settings
        if not proposed.any() or b.maximum_band_cells == 0 or report["pre_injection_negative_control"]:
            return proposed.copy()

        # The band follows the coarse proposal footprint, rather than every
        # native grain-sized hole left by the signed-color evidence check.
        # Majority occupancy avoids counting the fractional edge overlap from
        # resize(round(native*scale)) as a whole extra neighboring source cell.
        footprint = self.parent.full(self.parent.small(proposed) >= .5)
        rx = max(1, int(np.ceil(b.maximum_band_cells * domain.shape[1] / self.valid.shape[1])))
        ry = max(1, int(np.ceil(b.maximum_band_cells * domain.shape[0] / self.valid.shape[0])))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rx+1, 2*ry+1))
        outer = cv2.dilate(footprint.astype(np.uint8), kernel).astype(bool)
        inner = cv2.erode(footprint.astype(np.uint8), kernel).astype(bool)
        band = outer & ~inner & domain

        chroma, brightness = self._current_color_fields
        reference_chroma, reference_brightness = self.reference
        ratio = brightness / np.maximum(reference_brightness, 1e-6)
        exposed = (self.valid & (reference_brightness >= s.minimum_reference_brightness)
                   & (ratio >= s.minimum_brightness_ratio) & (ratio <= s.maximum_brightness_ratio))
        coarse_score = np.zeros(self.valid.shape, np.float32)
        coarse_norm = np.zeros(self.valid.shape, np.float32)
        available = np.zeros_like(self.valid)
        native_signal = np.zeros_like(domain)
        for controls in report["reference_support"]["facies_controls"]:
            if not controls.get("available"):
                continue
            label = controls["facies"]
            area = self.valid & (self.facies == label)
            direction = np.asarray(controls["aqueous_direction"], np.float32)
            shift = np.asarray(controls["common_shift"], np.float32)
            threshold = controls["chroma_threshold"]
            residual = chroma[area] - reference_chroma[area] - shift
            coarse_score[area] = (residual @ direction) / threshold
            coarse_norm[area] = np.linalg.norm(residual, axis=-1) / threshold
            available |= area
            selected = band & (self.context.facies == label)
            pixels = current[selected]
            native_chroma = pixels / np.maximum(pixels.sum(-1, keepdims=True), 1e-6)
            native_projection = (native_chroma - self.native_reference_chroma[selected] - shift) @ direction
            native_ratio = pixels.mean(-1) / np.maximum(self.native_reference_brightness[selected], 1e-6)
            native_signal[selected] = ((native_projection >= threshold*.25)
                & (self.native_reference_brightness[selected] >= s.minimum_reference_brightness)
                & (native_ratio >= s.minimum_brightness_ratio) & (native_ratio <= s.maximum_brightness_ratio))

        signal = (available & exposed & (coarse_score >= 1.)
                  & (coarse_score / np.maximum(coarse_norm, 1e-8) >= s.direction_cosine_minimum))
        # This reads the unchanged independent C2 observations after its update.
        # Today's observed-and-supported cells are included, exactly as in C2.
        trusted = self.valid & (elapsed_h-self.last_observed_h <= s.memory_hours)
        roots = signal & trusted
        if not roots.any():
            return proposed.copy()
        distance = distance_transform_edt(~roots, sampling=(self.parent.base.dy, self.parent.base.dx)).astype(np.float32)
        native_score = _facies_linear(coarse_score, available & exposed, self.context.facies,
                                      b.minimum_interpolation_weight)
        native_norm = _facies_linear(coarse_norm, available & exposed, self.context.facies,
                                     b.minimum_interpolation_weight)
        evidence = (band & native_signal & (native_score >= 1.)
                    & (native_score / np.maximum(native_norm, 1e-8) >= s.direction_cosine_minimum))
        native_distance = _facies_linear(distance, available, self.context.facies, b.minimum_interpolation_weight)
        independent = (self.parent.full(trusted) | ((native_score >= s.frontier_signal_multiplier)
                        & (native_distance <= s.maximum_front_distance_m)))
        candidates = (proposed & ~band) | (evidence & independent)
        established = self.parent.last_diagnostics["context_presence_score"] >= .5
        # A refinement cannot grow through a dark native hole, jump an excluded
        # corner, or form a new island without an existing C2 proposal anchor.
        connected = _anchored(candidates | established, proposed & candidates, domain)
        return candidates & connected & ~established

    def apply_with_reference(self, predictions, elapsed_h, current_rgb):
        inherited, report = super().apply_with_reference(predictions, elapsed_h, current_rgb)
        proposed = self.last_diagnostics["added_aqueous"].copy()
        current = rs.rgb_float(current_rgb, self.context.domain.shape)
        refined = self._refine_proposals(proposed, current, float(elapsed_h), report)
        features, available = rr.change_fields(self.calibration_baseline, current_rgb)
        c4_accepted = fr.accepted_proposals(features, available, self.context.facies, proposed, self.calibration)
        accepted = accepted_proposals(features, available, self.context.facies, refined, self.calibration,
            minimum_interpolation_weight=self.boundary_settings.minimum_interpolation_weight)
        established_score = self.parent.last_diagnostics["context_presence_score"]
        established = established_score >= .5
        accepted &= self.context.domain & ~established
        if (accepted & ~proposed).any():
            accepted &= _anchored(accepted | established, accepted & proposed, self.context.domain)
        if report["pre_injection_negative_control"]:
            accepted[:] = False

        adjusted = established_score.copy()
        adjusted[accepted] = .5
        conditional = inherited["gas_probability_conditional_on_co2"].copy()
        conditional[accepted] = np.minimum(conditional[accepted], .59)
        frozen_gas = established & (inherited["gas_probability_conditional_on_co2"] >= .6)
        if not np.array_equal((adjusted >= .5) & (conditional >= .6), frozen_gas):
            raise AssertionError("Native refinement changed inherited gas labels")
        free = adjusted*conditional
        result = {**inherited, "co2_presence_probability": adjusted,
            "gas_probability_conditional_on_co2": conditional,
            "untouched_water_probability": 1-adjusted,
            "aqueous_co2_probability": adjusted-free, "free_co2_gas_probability": free}

        rejected = proposed & ~accepted
        boundary_added = accepted & ~proposed
        boundary_removed = proposed & ~refined
        added_vs_c4 = accepted & ~c4_accepted
        removed_vs_c4 = c4_accepted & ~accepted
        trusted_native = self.parent.full(self.valid & (float(elapsed_h)-self.last_observed_h <= self.settings.memory_hours))
        self.last_diagnostics.update(added_aqueous=added_vs_c4,
            reference_accepted=accepted, reference_rejected=rejected, suppressed_presence=removed_vs_c4,
            boundary_added=boundary_added, boundary_removed=boundary_removed,
            unresolved=self.last_diagnostics["unresolved"] | rejected | (refined & ~accepted),
            spatial_only_presence=(self.last_diagnostics["spatial_only_presence"] & ~proposed) | accepted,
            reference_frontier=accepted & ~trusted_native, reference_residual=accepted & trusted_native)
        report["native_reference"] = dict(format=VERSION, calibration_enabled=self.calibration["enabled"],
            boundary_settings=asdict(self.boundary_settings),
            accepted_c2_pixels=int((accepted & proposed).sum()), rejected_c2_pixels=int(rejected.sum()),
            boundary_added_pixels=int(boundary_added.sum()), boundary_removed_pixels=int(boundary_removed.sum()),
            diagnostic_comparison=fr.VERSION,
            diagnostic_comparison_note="Frozen C4 gate on identical C2 proposals and calibration",
            added_vs_c4_pixels=int(added_vs_c4.sum()), removed_vs_c4_pixels=int(removed_vs_c4.sum()),
            total_accepted_aqueous_pixels=int(accepted.sum()), established_c1_labels_unchanged=True,
            gas_labels_unchanged=True, history_updates_from_refinement=False,
            optical_weights_retrained=False, calibrated_physical_probability=False)
        return result, report
