"""Versioned extensions to pre-review phase support; never mass constraints.

Models without an extension retain the exact legacy runtime identity and rules.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import maximum_filter1d

from . import phase_physics as legacy
from .project import sha256_file


CEILING_VERSION = "post-injection-column-ceiling-v1"


def _extension(profile):
    value = (profile or {}).get("post_injection_ceiling")
    if value is None:
        return None
    if value.get("format") != CEILING_VERSION or set(value) != {"format", "gas_score_cap"}:
        raise ValueError("Unknown post-injection phase-support extension")
    cap = float(value["gas_score_cap"])
    if not np.isfinite(cap) or not 0 <= cap < 0.5:
        raise ValueError("Post-injection gas-score cap must be below 0.5")
    return value


def physics_runtime_identity(profile):
    base = legacy.physics_runtime_identity(profile)
    if _extension(profile) is None:
        return base
    return hashlib.sha256(f"{base}:{sha256_file(Path(__file__))}".encode()).hexdigest()


def phase_physics_sequence(context, profile):
    settings = legacy.PhasePhysicsSettings(**profile.get("settings", {}))
    extension = _extension(profile)
    if extension is None:
        return legacy.PhasePhysicsSequence(context, settings)
    return PostInjectionCeilingSequence(context, settings, gas_score_cap=extension["gas_score_cap"])


class PostInjectionCeilingSequence(legacy.PhasePhysicsSequence):
    """Do not let optical confidence override an observed local depth limit.

Use the deepest source-associated gas observed during injection, not the latest
shrinking plume. Unobserved columns stay unknown. Ongoing, connected injection
can still drive gas downward. Multiple completed injection histories are united.
"""
    def __init__(self, context, settings, *, gas_score_cap=0.49):
        _extension({"post_injection_ceiling": {"format": CEILING_VERSION, "gas_score_cap": gas_score_cap}})
        super().__init__(context, settings)
        self.gas_score_cap = float(gas_score_cap)

    def apply(self, predictions, elapsed_h):
        result, report = super().apply(predictions, elapsed_h)
        history = np.zeros_like(self.valid)
        for stage, observed in zip(self.context.injection_stages, self.stage_history):
            if elapsed_h >= stage["end_h"]:
                history |= observed
        rows = np.arange(self.valid.shape[0])[:, None]
        deepest = np.max(np.where(history, rows, -1), axis=0)
        radius = max(0, round(self.settings.local_column_radius_m / self.dx))
        deepest = maximum_filter1d(deepest, size=2*radius+1, mode="nearest")
        known = deepest >= 0
        downward = known[None, :] & (rows > deepest[None, :] + self.settings.downward_tolerance_m/self.dy)

        # Preserve active-source-connected excursions, including a later injection.
        gas_support = self.valid & (self._small(predictions["co2_presence_probability"]) >= .5) & (
            self._small(predictions["gas_probability_conditional_on_co2"]) >= .6)
        _, labels = cv2.connectedComponents(gas_support.astype(np.uint8), connectivity=8)
        exempt = np.zeros_like(self.valid)
        for index, stage in enumerate(self.context.injection_stages):
            source = self._source(stage, elapsed_h)
            if source.any():
                ids = np.unique(labels[gas_support & (source | self._near(self.stage_history[index], self.settings.history_radius_m))])
                exempt |= source | np.isin(labels, ids[ids > 0])
        downward &= self.valid & ~exempt
        full = cv2.resize(downward.astype(np.uint8),
                          (self.context.domain.shape[1], self.context.domain.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        co2 = result["co2_presence_probability"]
        raw_gas = predictions["gas_probability_conditional_on_co2"]
        conflicts = full & self.context.domain & (co2 >= .5) & (raw_gas >= .6)
        gas = result["gas_probability_conditional_on_co2"].copy()
        gas[conflicts] = np.minimum(gas[conflicts], self.gas_score_cap)
        result = {**result, "gas_probability_conditional_on_co2": gas,
                  "free_co2_gas_probability": co2*gas,
                  "aqueous_co2_probability": co2*(1-gas)}
        self.previous_gas = self.previous & (self._small(gas) >= .6)
        report.update(
            post_injection_ceiling=CEILING_VERSION,
            post_injection_ceiling_pixels=int(conflicts.sum()),
            strong_gas_ceiling_pixels=int(np.count_nonzero(conflicts & (raw_gas >= self.settings.strong_gas_probability))),
            injection_history_columns=int(known.sum()),
            supported_gas_pixels=int(np.count_nonzero(self.context.domain & (co2 >= .5) & (gas >= .6))),
            co2_support_changed_by_ceiling=False,
            concentration_or_mass_constraints=False)
        return result, report
