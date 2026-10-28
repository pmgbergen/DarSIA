"""Opt-in causal CO2 support; frozen optics and inherited gas decisions.

This first revision is an explicit, uncalibrated classification prior, not a
transport solver. Promoted weak pixels never become trusted propagation seeds.
Lengths/times below are reviewable hypotheses, not measured fluid velocities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.graph import MCP_Geometric

from . import phase_plume, phase_plume_envelope
from .project import sha256_file

VERSION = "causal-co2-presence-context-v1"
PROFILE_FILE = "phase_presence_context.json"


@dataclass(frozen=True)
class PresenceSettings:
    fine_sigma_m: float = .001
    regional_sigma_m: float = .004
    minimum_raw_score: float = .15
    corridor_score: float = .22
    retain_score: float = .32
    establish_score: float = .38
    strong_score: float = .65
    memory_hours: float = 120.
    absence_score: float = .10
    absence_hours: float = 4.
    path_base_m: float = .15
    path_allowance_m_per_h: float = .15
    maximum_interval_h: float = 12.
    weak_path_cost: float = 2.
    material_path_cost: float = .25
    maximum_unobserved_gap_m: float = .03
    unobserved_path_cost: float = 8.
    mode: str = "causal"

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == "mode":
                if value not in ("causal", "spatial"):
                    raise ValueError("Presence mode must be causal or spatial")
            elif not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not (0 < self.absence_score < self.minimum_raw_score < self.corridor_score
                <= self.retain_score < self.establish_score < .5 < self.strong_score < 1):
            raise ValueError("Invalid presence threshold ordering")
        if self.regional_sigma_m < self.fine_sigma_m:
            raise ValueError("Regional scale must not be smaller than fine scale")


def validate_profile(profile):
    if profile.get("format") != VERSION:
        raise ValueError("Unknown presence-context format")
    base = profile.get("gas_support", {})
    if base.get("format") not in (phase_plume.VERSION, phase_plume_envelope.VERSION):
        raise ValueError("An immutable supported gas profile is required")
    if not profile.get("runs") or profile["runs"] != base.get("runs"):
        raise ValueError("Presence and inherited gas contexts must be identical")
    if profile.get("settings", {}) != base.get("settings", {}):
        raise ValueError("Inherited gas settings cannot change implicitly")
    phase_plume.PlumeSettings(**base.get("settings", {}))
    PresenceSettings(**profile.get("presence_settings", {}))
    return profile


def load_profile(root):
    path = Path(root) / PROFILE_FILE
    return validate_profile(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else None


def runtime_identity(profile):
    validate_profile(profile)
    base = profile["gas_support"]
    identity = (phase_plume_envelope.runtime_identity(base)
                if base["format"] == phase_plume_envelope.VERSION else phase_plume.runtime_identity(base))
    h = hashlib.sha256(json.dumps(profile, sort_keys=True).encode())
    h.update(identity.encode())
    h.update(sha256_file(Path(__file__)).encode())
    return h.hexdigest()


def smooth_within_facies(values, domain, facies, sigma_x, sigma_y):
    """Normalized averaging does not import signal across a material boundary."""
    result = np.zeros(values.shape, np.float32)
    for label in np.unique(facies[domain]):
        mask = domain & (facies == label)
        weights = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
        weighted = cv2.GaussianBlur(np.where(mask, values, 0).astype(np.float32),
                                   (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
        result[mask] = weighted[mask] / np.maximum(weights[mask], 1e-7)
    return result


def inherited_gas_object_support(co2_mask, gas_mask):
    """Preserve existing CO2 objects containing frozen gas, without seeding history.

    This isolated aqueous revision cannot reject an entire object's source
    connection while simultaneously retaining its gas as an accepted phase.
    These objects remain unresolved for a future joint gas/presence review.
    """
    if not gas_mask.any():
        return np.zeros_like(co2_mask)
    count, labels = cv2.connectedComponents(co2_mask.astype(np.uint8), connectivity=8)
    protected = np.zeros(count, bool)
    protected[np.unique(labels[gas_mask])] = True
    protected[0] = False
    return protected[labels]


class PresenceSequence:
    def __init__(self, context, profile):
        validate_profile(profile)
        self.context = context
        self.settings = PresenceSettings(**profile.get("presence_settings", {}))
        gas_profile = profile["gas_support"]
        cls = (phase_plume_envelope.LocalFloorSequence if gas_profile["format"] == phase_plume_envelope.VERSION
               else phase_plume.PlumeSequence)
        self.base = cls(context, phase_plume.PlumeSettings(**gas_profile.get("settings", {})))
        self.target = self.base.target
        self.valid = self.small(context.domain.astype(np.float32)) >= .999
        self.last_strong_h = np.full(self.valid.shape, -np.inf)
        self.first_arrival_h = np.full(self.valid.shape, np.inf)
        self.absent_since_h = np.full(self.valid.shape, np.inf)
        self.previous_supported = np.zeros_like(self.valid)
        self.last_time = None
        self.last_diagnostics = {}
        # Material boundaries increase path cost but never become impermeable
        # aqueous barriers. Permeability is not a capillary-entry pressure.
        self.material_cost = np.zeros(self.valid.shape, np.float32)
        if context.permeability is not None:
            # Outside-domain permeability may be zero/undefined. It must not
            # enter logarithms or contaminate an adjacent valid path cost.
            safe_k = np.where(context.domain, context.permeability, 1.)
            logk = self.small(np.log(safe_k)) / np.maximum(self.small(context.domain), 1e-7)
            nearest = distance_transform_edt(~self.valid, return_distances=False, return_indices=True)
            logk = logk[tuple(nearest)]
            gy, gx = np.gradient(logk)
            self.material_cost = np.minimum(np.hypot(gx, gy), 4.)
        else:
            labels = cv2.resize(context.facies.astype(np.float32), self.target, interpolation=cv2.INTER_NEAREST)
            edges = np.zeros_like(self.valid)
            edges[1:] |= labels[1:] != labels[:-1]
            edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
            self.material_cost = edges.astype(np.float32)

    def small(self, values):
        return cv2.resize(np.asarray(values, np.float32), self.target, interpolation=cv2.INTER_AREA)

    def full(self, mask):
        return cv2.resize(np.asarray(mask, np.uint8), self.context.domain.shape[::-1],
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    def write_diagnostics(self, root, run, frame, canvas_shape, content_box):
        from PIL import Image
        output = Path(root) / run
        output.mkdir(parents=True, exist_ok=True)
        flags = {k: v for k, v in self.last_diagnostics.items() if v.dtype == bool}
        archive = output / f"frame_{frame:03d}.npz"
        with archive.open("xb") as stream:
            np.savez_compressed(stream, **flags)
        x, y, w, h = content_box
        for name, color in (("added_aqueous", (49, 217, 160, 255)),
                            ("suppressed_presence", (255, 70, 70, 255)),
                            ("unresolved", (30, 210, 255, 255))):
            canvas = np.zeros((*canvas_shape, 4), np.uint8)
            canvas[y:y+h, x:x+w][flags[name]] = color
            folder = output / name
            folder.mkdir(exist_ok=True)
            path = folder / f"frame_{frame:03d}.png"
            with path.open("xb") as stream:
                Image.fromarray(canvas).save(stream, format="PNG")

    def _reach(self, score, corridor, seeds, limit):
        if not seeds.any():
            return np.full(score.shape, np.inf)
        cost = 1 + self.settings.weak_path_cost * (1-score) + self.settings.material_path_cost*self.material_cost
        # Optical disconnection is not necessarily physical disconnection. A
        # short unresolved transit is allowed, but never labels the gap itself.
        gap_distance = distance_transform_edt(~corridor, sampling=(self.base.dy, self.base.dx))
        transit = self.valid & (gap_distance <= self.settings.maximum_unobserved_gap_m/2)
        cost = np.where(transit, cost + np.where(corridor, 0., self.settings.unobserved_path_cost), np.inf)
        # Four-neighbour paths cannot jump diagonally over an excluded corner.
        boundary = seeds & ~cv2.erode(seeds.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if not boundary.any():
            boundary = seeds
        distances, _ = MCP_Geometric(cost, fully_connected=False,
            sampling=(self.base.dy, self.base.dx)).find_costs(np.argwhere(boundary))
        distances[seeds] = 0
        distances[distances > limit] = np.inf
        return distances

    def apply(self, predictions, elapsed_h):
        time = float(elapsed_h)
        if not np.isfinite(time) or (self.last_time is not None and time <= self.last_time):
            raise ValueError("Presence history requires strictly increasing finite frame times")
        c = np.asarray(predictions["co2_presence_probability"], np.float32)
        inherited, gas_report = self.base.apply(predictions, time)
        s, domain = self.settings, self.context.domain
        previous_c = inherited["co2_presence_probability"]
        previous_g = inherited["gas_probability_conditional_on_co2"]
        old_mask = domain & (previous_c >= .5)
        frozen_gas = old_mask & (previous_g >= .6)
        inherited_objects = inherited_gas_object_support(old_mask, frozen_gas)
        dx, dy = self.context.x_m[1]-self.context.x_m[0], self.context.y_m[0]-self.context.y_m[1]
        fine = smooth_within_facies(c, domain, self.context.facies, s.fine_sigma_m/dx, s.fine_sigma_m/dy)
        regional = smooth_within_facies(c, domain, self.context.facies,
                                       s.regional_sigma_m/dx, s.regional_sigma_m/dy)
        optical = .75*fine + .25*regional
        score, raw = self.small(optical), self.small(c)
        strong = self.valid & (score >= s.strong_score) & (raw >= .5)
        corridor = self.valid & (score >= s.corridor_score) & (raw >= s.minimum_raw_score)
        absent = self.valid & (raw < s.absence_score)
        self.absent_since_h[~absent] = np.inf
        self.absent_since_h[absent & ~np.isfinite(self.absent_since_h)] = time
        contradicted = absent & (time-self.absent_since_h >= s.absence_hours)
        self.last_strong_h[contradicted] = -np.inf
        trusted = self.valid & (time-self.last_strong_h <= s.memory_hours)
        self.first_arrival_h[~trusted] = np.inf
        sources = [self.base.source(stage, time) for stage in self.context.injection_stages]
        source = np.logical_or.reduce(sources) & corridor & (raw >= .5)
        active = any(stage["start_h"] <= time < stage["end_h"] for stage in self.context.injection_stages)
        dt = 0. if self.last_time is None else time-self.last_time
        # Repeated weak frames must not reset the transport clock or create
        # roots. Use cumulative elapsed time from surviving observed history.
        history_age = max(0., time-float(np.min(self.first_arrival_h[trusted]))) if trusted.any() else 0.
        limit = s.path_base_m + s.path_allowance_m_per_h * min(history_age, s.maximum_interval_h)
        if s.mode == 'spatial':
            limit = float('inf')
        if active:
            # During forced injection do not invent a transport speed from rate
            # alone; require the observed corridor to the actual source instead.
            limit = float(np.hypot(np.ptp(self.context.x_m), np.ptp(self.context.y_m))) * (1+s.weak_path_cost)
        seeds = (source | (trusted & corridor))
        if s.mode == "spatial":
            seeds = source | strong
        distances = self._reach(score, corridor, seeds, limit)
        reached = np.isfinite(distances) & corridor
        pre = gas_report["pre_injection_negative_control"]
        history_available = bool(seeds.any()) and not pre
        retain = self.full(self.previous_supported) & (optical >= s.retain_score)
        evidence = domain & (c >= s.minimum_raw_score) & ((optical >= s.establish_score) | retain)
        supported = self.full(reached) & evidence
        # Strong but unconnected new objects remain reviewable. They are neither
        # seeds nor silently accepted because of their size/persistence alone.
        added = supported & ~old_mask if history_available else np.zeros_like(domain)
        removed = old_mask & ~self.full(reached) & ~inherited_objects if history_available else np.zeros_like(domain)
        adjusted = previous_c.copy()
        adjusted[added] = np.maximum(.5, optical[added])
        adjusted[removed] = np.minimum(adjusted[removed], .49)
        gas = previous_g.copy()
        gas[added] = np.minimum(gas[added], .59)
        unresolved = domain & (((c >= .5) & ~self.full(reached)) | (evidence & ~supported))
        if not history_available and not pre:
            unresolved |= old_mask
        if pre:
            added[:] = removed[:] = False
            unresolved = domain & (c >= .5)
        # Matched-input ablation: same optical field and frozen gas, but no
        # new source/history restriction. It is diagnostic, never a seed.
        spatial_reached = np.isfinite(self._reach(score, corridor, strong, float('inf'))) & corridor
        spatial_evidence = domain & (c >= s.minimum_raw_score) & (optical >= s.establish_score)
        spatial_added = self.full(spatial_reached) & spatial_evidence & ~old_mask
        spatial_removed = old_mask & ~self.full(spatial_reached) & ~inherited_objects
        spatial_mask = (old_mask | spatial_added) & ~spatial_removed if strong.any() and not pre else old_mask
        if not pre and s.mode == "causal":
            observed = strong & reached
            self.last_strong_h[observed] = time
            first = observed & ~np.isfinite(self.first_arrival_h)
            self.first_arrival_h[first] = time
        self.previous_supported = self.small(supported.astype(np.float32)) >= .5
        self.last_time = time
        self.last_diagnostics = {
            "added_aqueous": added, "suppressed_presence": removed,
            "unresolved": unresolved, "trusted_history": self.full(trusted) & domain,
            "spatial_only_presence": spatial_mask,
            "raw_presence_score": c, "context_presence_score": adjusted,
        }
        free = adjusted*gas
        result = {**inherited, "co2_presence_probability": adjusted,
                  "gas_probability_conditional_on_co2": gas,
                  "untouched_water_probability": 1-adjusted,
                  "aqueous_co2_probability": adjusted-free, "free_co2_gas_probability": free}
        if not np.array_equal((adjusted >= .5) & (gas >= .6), frozen_gas):
            raise AssertionError("Presence revision changed inherited gas classification")
        return result, {**gas_report, "presence_context": {
            "format": VERSION, "mode": s.mode, "weights_trained": False,
            "calibrated_probability": False, "elapsed_interval_h": dt,
            "path_cost_limit_m": limit if np.isfinite(limit) else None, "history_available": history_available,
            "trusted_history_age_h": history_age, "path_time_basis": "accumulated_observed_history",
            "trusted_cells": int(trusted.sum()), "new_strong_observations": int((strong & reached).sum()),
            "added_aqueous_pixels": int(added.sum()), "suppressed_presence_pixels": int(removed.sum()),
            "unresolved_pixels": int(unresolved.sum()), "gas_labels_unchanged": True,
            "inherited_gas_object_pixels_preserved": int((inherited_objects & ~self.full(reached)).sum()),
            "memory_updates_from_promoted_pixels": False,
        }}
