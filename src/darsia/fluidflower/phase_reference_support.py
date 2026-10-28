"""Opt-in baseline-relative aqueous support over an immutable C1 sequence.

This is a classification prior, not concentration, transport, or calibrated
probability. Only independent strong parent observations teach color directions
and seed history. Weak promotions never feed either mechanism.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from . import phase_presence_context as pc
from .project import sha256_file

VERSION = "baseline-relative-aqueous-support-v1"
PROFILE_FILE = "phase_reference_support.json"


@dataclass(frozen=True)
class ReferenceSettings:
    smoothing_m: float = .002
    minimum_chroma_change: float = .004
    noise_multiplier: float = 3.
    direction_cosine_minimum: float = .80
    minimum_direction_samples: int = 24
    minimum_reference_brightness: float = .025
    minimum_brightness_ratio: float = .45
    maximum_brightness_ratio: float = 2.2
    frontier_signal_multiplier: float = 1.5
    maximum_front_distance_m: float = .06
    reference_control_fraction: float = .05
    maximum_common_chroma_shift: float = .025
    memory_hours: float = 120.
    absence_hours: float = 4.

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not (0 < self.direction_cosine_minimum < 1):
            raise ValueError("Invalid color-direction cosine")
        if not (0 < self.reference_control_fraction < 1):
            raise ValueError("Invalid control fraction")
        if not (self.minimum_brightness_ratio < 1 < self.maximum_brightness_ratio):
            raise ValueError("Brightness-ratio limits must straddle one")
        if int(self.minimum_direction_samples) != self.minimum_direction_samples:
            raise ValueError("Direction sample count must be an integer")


def validate_profile(profile):
    if profile.get("format") != VERSION:
        raise ValueError("Unknown baseline support format")
    parent = pc.validate_profile(profile.get("presence_support", {}))
    if profile.get("runs") != parent["runs"] or profile.get("settings", {}) != parent.get("settings", {}):
        raise ValueError("Inherited physical contexts cannot change implicitly")
    ReferenceSettings(**profile.get("reference_settings", {}))
    return profile


def load_profile(root):
    path = Path(root) / PROFILE_FILE
    return validate_profile(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else None


def runtime_identity(profile):
    validate_profile(profile)
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True).encode())
    digest.update(pc.runtime_identity(profile["presence_support"]).encode())
    digest.update(sha256_file(Path(__file__)).encode())
    return digest.hexdigest()


def rgb_float(rgb, shape):
    values = np.asarray(rgb)
    if values.shape != (*shape, 3):
        raise ValueError("Reference and current RGB must match the registered physical grid")
    scale = 255. if values.dtype == np.uint8 else 1.
    values = np.asarray(values, np.float32) / scale
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("Expected finite RGB in [0, 1], or uint8 RGB")
    return values


class ReferenceSequence:
    def __init__(self, context, profile):
        validate_profile(profile)
        self.context = context
        self.settings = ReferenceSettings(**profile.get("reference_settings", {}))
        self.parent = pc.PresenceSequence(context, profile["presence_support"])
        self.target = self.parent.target
        self.valid = self.parent.valid
        self.facies = cv2.resize(context.facies.astype(np.float32), self.target,
                                 interpolation=cv2.INTER_NEAREST)
        self.reference = None
        self.directions = {}
        self.last_observed_h = np.full(self.valid.shape, -np.inf)
        self.absent_since_h = np.full(self.valid.shape, np.inf)
        self.last_diagnostics = {}
        self._facies_fields = []
        for label in np.unique(context.facies[context.domain]):
            mask = context.domain & (context.facies == label)
            weight = self._blur(self.parent.small(mask))
            self._facies_fields.append((label, mask, weight))

    def _blur(self, values):
        return cv2.GaussianBlur(values, (0, 0),
            sigmaX=self.settings.smoothing_m/self.parent.base.dx,
            sigmaY=self.settings.smoothing_m/self.parent.base.dy)

    def bind_reference(self, baseline_rgb):
        if self.reference is not None:
            raise ValueError("A sequence reference is immutable")
        baseline = rgb_float(baseline_rgb, self.context.domain.shape)
        self.reference = self._color_fields(baseline)
        self.native_reference_chroma = baseline / np.maximum(baseline.sum(axis=-1, keepdims=True), 1e-6)
        self.native_reference_brightness = baseline.mean(axis=-1)

    def _color_fields(self, rgb):
        # Blur RGB before taking fractions: darkness and sensor noise must not
        # receive the same weight as well-exposed grains in a physical region.
        smooth = np.zeros((*self.valid.shape, 3), np.float32)
        for label, mask, weight in self._facies_fields:
            selected = self.valid & (self.facies == label)
            for k in range(3):
                weighted = self.parent.small(np.where(mask, rgb[..., k], 0))
                values = self._blur(weighted) / np.maximum(weight, 1e-7)
                smooth[..., k][selected] = values[selected]
        brightness = smooth.mean(axis=-1)
        chroma = smooth / np.maximum(smooth.sum(axis=-1, keepdims=True), 1e-6)
        return chroma, brightness

    def apply(self, predictions, elapsed_h):
        raise ValueError("Baseline support requires apply_with_reference and corrected current RGB")

    def apply_with_reference(self, predictions, elapsed_h, current_rgb):
        if self.reference is None:
            raise ValueError("Bind the corrected pre-injection reference before prediction")
        current = rgb_float(current_rgb, self.context.domain.shape)
        inherited, report = self.parent.apply(predictions, elapsed_h)
        time = float(elapsed_h)
        s = self.settings
        domain = self.context.domain
        chroma, brightness = self._color_fields(current)
        reference_chroma, reference_brightness = self.reference
        delta = chroma - reference_chroma
        ratio = brightness / np.maximum(reference_brightness, 1e-6)
        exposed = (self.valid & (reference_brightness >= s.minimum_reference_brightness)
            & (ratio >= s.minimum_brightness_ratio) & (ratio <= s.maximum_brightness_ratio))
        raw = self.parent.small(predictions["co2_presence_probability"])
        gas = self.parent.small(inherited["gas_probability_conditional_on_co2"])
        parent_mask = inherited["co2_presence_probability"] >= .5
        observed = exposed & (self.parent.last_strong_h == time)
        trusted = self.valid & (time-self.last_observed_h <= s.memory_hours)
        # Control pixels must be outside both observed history and today's
        # accepted plume. This subtraction diagnoses common chromatic drift; it
        # does not modify source images or recalibrate the optical classifier.
        occupied = trusted | (self.parent.small(parent_mask) > .05)
        distance = distance_transform_edt(~occupied, sampling=(self.parent.base.dy, self.parent.base.dx))
        control_pool = exposed & (raw < .10) & (distance > s.maximum_front_distance_m)
        score = np.zeros(self.valid.shape, np.float32)
        color_supported = np.zeros_like(self.valid)
        available = np.zeros_like(self.valid)
        native_signal = np.zeros(domain.shape, bool)
        native_chroma = current / np.maximum(current.sum(axis=-1, keepdims=True), 1e-6)
        native_ratio = current.mean(axis=-1) / np.maximum(self.native_reference_brightness, 1e-6)
        controls_report = []
        pre = report["pre_injection_negative_control"]
        for label in np.unique(self.facies[self.valid]):
            area = self.valid & (self.facies == label)
            controls = control_pool & area
            required = max(s.minimum_direction_samples, int(area.sum()*s.reference_control_fraction))
            if controls.sum() < required:
                controls_report.append(dict(facies=float(label), available=False, reason="insufficient unchanged controls"))
                continue
            shift = np.median(delta[controls], axis=0)
            residual = delta - shift
            noise = np.linalg.norm(residual[controls], axis=1)
            median_noise = float(np.median(noise))
            threshold = max(s.minimum_chroma_change,
                median_noise + s.noise_multiplier*1.4826*float(np.median(np.abs(noise-median_noise))))
            if np.linalg.norm(shift) > s.maximum_common_chroma_shift:
                controls_report.append(dict(facies=float(label), available=False, reason="chromatic illumination change"))
                continue
            teaching = area & observed & (gas < .3) & (np.linalg.norm(residual, axis=-1) > threshold*3)
            if not pre and teaching.sum() >= s.minimum_direction_samples:
                direction = np.median(residual[teaching], axis=0)
                norm = np.linalg.norm(direction)
                if norm > threshold*3:
                    # Freeze the first reliable run/facies-specific aqueous
                    # direction. Repeated late frames cannot drift the template.
                    self.directions.setdefault(float(label), direction/norm)
            direction = self.directions.get(float(label))
            if direction is None:
                controls_report.append(dict(facies=float(label), available=False, reason="no independently supported aqueous color direction"))
                continue
            projection = residual @ direction
            cosine = projection / np.maximum(np.linalg.norm(residual, axis=-1), 1e-8)
            score[area] = np.maximum(projection[area], 0) / threshold
            color_supported[area] = exposed[area] & (cosine[area] >= s.direction_cosine_minimum)
            available |= area
            native_area = domain & (self.context.facies == label)
            native_delta = native_chroma[native_area] - self.native_reference_chroma[native_area] - shift
            native_projection = native_delta @ direction
            native_signal[native_area] = (
                (native_projection >= threshold*.25)
                & (self.native_reference_brightness[native_area] >= s.minimum_reference_brightness)
                & (native_ratio[native_area] >= s.minimum_brightness_ratio)
                & (native_ratio[native_area] <= s.maximum_brightness_ratio))
            controls_report.append(dict(facies=float(label), available=True, controls=int(controls.sum()),
                chroma_threshold=threshold, common_shift=shift.tolist(), aqueous_direction=direction.tolist()))
        signal = available & color_supported & (score >= 1.)
        absent = available & (score < .5)
        self.absent_since_h[~absent] = np.inf
        self.absent_since_h[absent & ~np.isfinite(self.absent_since_h)] = time
        self.last_observed_h[absent & (time-self.absent_since_h >= s.absence_hours)] = -np.inf
        trusted = self.valid & (time-self.last_observed_h <= s.memory_hours)
        if not pre:
            self.last_observed_h[observed & signal] = time
        roots = signal & (trusted | observed)
        # Unlike C1's cumulative transport hypothesis this extension is bounded
        # by distance to independent observations, never yesterday's promotions.
        root_distance = distance_transform_edt(~roots, sampling=(self.parent.base.dy, self.parent.base.dx)) if roots.any() else np.full(roots.shape, np.inf)
        frontier = signal & (score >= s.frontier_signal_multiplier) & (root_distance <= s.maximum_front_distance_m)
        corridor = signal & ((root_distance <= s.maximum_front_distance_m) | trusted)
        reached = np.isfinite(self.parent._reach(np.minimum(score/3, 1), corridor, roots,
            s.maximum_front_distance_m*(1+self.parent.settings.weak_path_cost))) & corridor
        support = reached & (frontier | trusted)
        if pre:
            support[:] = False
        # Only add aqueous labels where the native pixel has the same signed
        # chromatic evidence. A coarse supported cell never paints dark holes.
        added = self.parent.full(support) & domain & native_signal & ~parent_mask
        adjusted = inherited["co2_presence_probability"].copy()
        adjusted[added] = .5
        conditional = inherited["gas_probability_conditional_on_co2"].copy()
        conditional[added] = np.minimum(conditional[added], .59)
        frozen_gas = parent_mask & (inherited["gas_probability_conditional_on_co2"] >= .6)
        if not np.array_equal((adjusted >= .5) & (conditional >= .6), frozen_gas):
            raise AssertionError("Reference support changed inherited gas labels")
        free = adjusted*conditional
        result = {**inherited, "co2_presence_probability": adjusted,
            "gas_probability_conditional_on_co2": conditional,
            "untouched_water_probability": 1-adjusted,
            "aqueous_co2_probability": adjusted-free, "free_co2_gas_probability": free}
        self.last_diagnostics = {
            "added_aqueous": added, "suppressed_presence": np.zeros_like(domain),
            "unresolved": self.parent.last_diagnostics["unresolved"] | (self.parent.full(signal & ~support) & domain),
            "trusted_history": self.parent.full(trusted) & domain,
            "spatial_only_presence": parent_mask | (self.parent.full(signal & (score >= s.frontier_signal_multiplier)) & domain & native_signal),
            "reference_frontier": added & self.parent.full(~trusted),
            "reference_residual": added & self.parent.full(trusted),
            "reference_available": self.parent.full(available) & domain,
        }
        return result, {**report, "reference_support": {
            "format": VERSION, "calibrated_probability": False, "weights_trained": False,
            "gas_labels_unchanged": True, "memory_updates_from_promoted_pixels": False,
            "added_aqueous_pixels": int(added.sum()), "facies_controls": controls_report,
            "reference_grid_shape": list(self.valid.shape), "smoothing_m": s.smoothing_m,
            "reference_residual_pixels": int(self.last_diagnostics["reference_residual"].sum()),
            "reference_frontier_pixels": int(self.last_diagnostics["reference_frontier"].sum()),
        }}

    write_diagnostics = pc.PresenceSequence.write_diagnostics
