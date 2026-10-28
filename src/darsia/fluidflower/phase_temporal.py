"""Opt-in, causal aqueous-front memory upstream of human phase review.

This is a phase-support hypothesis, not a transport solver or a concentration
model. Optical weights and the CO2-presence field are unchanged by this layer.
Legacy physics modules remain byte-identical so published bindings stay valid.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

from . import phase_support as previous
from .project import sha256_file


FRONT_VERSION = "dissolution-front-memory-v4"


@dataclass(frozen=True)
class AqueousFrontSettings:
    co2_seed_min: float = .65
    gas_seed_max: float = .4
    registration_tolerance_m: float = .003
    tangent_normal_cosine_max: float = .75
    minimum_log_interface_contrast: float = .05

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not .5 <= self.co2_seed_min <= 1 or not 0 < self.gas_seed_max < .5:
            raise ValueError("Aqueous seeds must have explicit CO2 and low gas scores")
        if self.tangent_normal_cosine_max > 1:
            raise ValueError("Tangent cosine must not exceed one")


def _settings(profile):
    value = (profile or {}).get("aqueous_front_memory")
    if value is None:
        return None
    if value.get("format") != FRONT_VERSION:
        raise ValueError("Unknown aqueous-front memory version")
    return AqueousFrontSettings(**{k: v for k, v in value.items() if k != "format"})


def physics_runtime_identity(profile):
    base = previous.physics_runtime_identity(profile)
    if _settings(profile) is None:
        return base
    return hashlib.sha256(f"{base}:{sha256_file(Path(__file__))}".encode()).hexdigest()


def phase_physics_sequence(context, profile):
    base = previous.phase_physics_sequence(context, profile)
    settings = _settings(profile)
    return AqueousFrontSequence(base, settings) if settings is not None else base


class LayerDirectedReach:
    """Directed connectivity in observed CO2, with curvature-following edges.

Downward/horizontal edges support aqueous transport; the opposite direction
supports possible gas supply. Reverse vertical steps require local alignment
with a measured permeability interface. No velocity is inferred from k.
"""
    def __init__(self, base, settings):
        self.shape, self.valid = base.valid.shape, base.valid
        self.dx, self.dy, self.settings = base.dx, base.dy, settings
        k = base._small(base.context.permeability, nearest=True)
        nearest = distance_transform_edt(~self.valid, return_distances=False, return_indices=True)
        log_k = np.log(k[tuple(nearest)]).astype(np.float32)
        smooth = cv2.GaussianBlur(log_k, (0, 0),
            sigmaX=max(.5, base.settings.interface_scale_m / self.dx),
            sigmaY=max(.5, base.settings.interface_scale_m / self.dy))
        gy, gx = np.gradient(smooth, self.dy, self.dx)
        magnitude = np.hypot(gx, gy)
        self.nx, self.ny = gx / np.maximum(magnitude, 1e-8), gy / np.maximum(magnitude, 1e-8)
        self.interface = self.valid & (magnitude * base.settings.interface_scale_m
                                      >= settings.minimum_log_interface_contrast)
        self.edges = {direction: self._edges(direction) for direction in (-1, 1)}

    def _edges(self, direction):
        height, width = self.shape
        ids = np.arange(height * width).reshape(self.shape)
        origins, targets = [], []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                a = (slice(max(0, -dr), min(height, height-dr)),
                     slice(max(0, -dc), min(width, width-dc)))
                b = (slice(max(0, dr), min(height, height+dr)),
                     slice(max(0, dc), min(width, width+dc)))
                allowed = self.valid[a] & self.valid[b]
                if dr * direction < 0:
                    length = np.hypot(dc*self.dx, dr*self.dy)
                    alignment = np.abs(self.nx[a]*dc*self.dx + self.ny[a]*dr*self.dy) / length
                    allowed &= self.interface[a] & self.interface[b] & (
                        alignment <= self.settings.tangent_normal_cosine_max)
                origins.append(ids[a][allowed])
                targets.append(ids[b][allowed])
        return np.concatenate(origins), np.concatenate(targets)

    def reach(self, support, seeds, *, direction, arrival=None):
        support = np.asarray(support, bool) & self.valid
        seeds = np.flatnonzero(support & seeds)
        if not len(seeds):
            return np.zeros(self.shape, bool)
        source, target = self.edges[direction]
        mask = support.ravel()[source] & support.ravel()[target]
        if arrival is not None:
            # A possible supply path cannot run backwards through the observed
            # arrival order of a different, descending front.
            arrival = np.asarray(arrival)
            if arrival.shape != self.shape:
                raise ValueError("Arrival history and connectivity grid differ")
            unknown = np.iinfo(np.int32).max
            mask &= ((arrival.ravel()[target] >= arrival.ravel()[source])
                     | (arrival.ravel()[source] == unknown) | (arrival.ravel()[target] == unknown))
        size = support.size
        rows = np.concatenate((source[mask], np.full(len(seeds), size)))
        cols = np.concatenate((target[mask], seeds))
        graph = csr_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(size+1, size+1))
        reached = breadth_first_order(graph, size, directed=True, return_predecessors=False)
        result = np.zeros(size+1, bool)
        result[reached] = True
        return result[:size].reshape(self.shape)


class AqueousFrontSequence:
    """Keep established dissolved fronts aqueous during post-injection advance.

No learning from this sequence's own gas rejections is used to seed a new
    front. Seeds need an earlier optical aqueous observation and an injection-era
    gas source. The CO2 footprint through the first frame at/after injection end
    is protected: a mistaken old aqueous score cannot erase pre-existing gas.
"""
    def __init__(self, base, settings):
        self.base, self.context, self.settings = base, base.context, settings
        self.paths = LayerDirectedReach(base, settings)
        self.ever_co2 = np.zeros_like(base.valid)
        self.optical_aqueous_memory = np.zeros_like(base.valid)
        self.front_memory = np.zeros_like(base.valid)
        self.last_front_mask = np.zeros_like(base.context.domain)
        self.reference_co2 = np.zeros_like(base.context.domain)
        self.reference_time = None
        self.arrival = np.full(base.valid.shape, np.iinfo(np.int32).max, np.int32)
        self.frame_number = 0

    def apply(self, predictions, elapsed_h):
        base, settings = self.base, self.settings
        # This also performs shape/range/chronology checks before state changes.
        result, report = base.apply(predictions, elapsed_h)
        co2 = base._small(result["co2_presence_probability"])
        gas = base._small(predictions["gas_probability_conditional_on_co2"])
        active = base.valid & (co2 >= .5)
        self.arrival[active & ~self.ever_co2] = self.frame_number
        history = np.logical_or.reduce(base.stage_history)
        injection_active = any(stage["start_h"] <= elapsed_h < stage["end_h"]
                               for stage in self.context.injection_stages)
        observed_end = all(elapsed_h >= stage["end_h"] for stage in self.context.injection_stages)
        capturing_reference = self.reference_time is None
        if capturing_reference:
            self.reference_co2 |= self.context.domain & (result["co2_presence_probability"] >= .5)
            if observed_end:
                self.reference_time = float(elapsed_h)
        # Only bridge registration-scale gaps; never create displayed CO2 there.
        radius = settings.registration_tolerance_m
        rx, ry = max(1, round(radius/base.dx)), max(1, round(radius/base.dy))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rx+1, 2*ry+1))
        corridor = cv2.morphologyEx(active.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool) & base.valid
        protected = base._near(history, radius)
        # Gas predictions can be speckled inside a genuine gas pocket. Use the
        # CO2 corridor for a possible path, but require a current gas seed and
        # the observed front arrival order. This is a veto, not a gas label.
        gas_supply = self.paths.reach(corridor, history & (gas >= .6), direction=-1,
                                      arrival=self.arrival)
        rows = np.arange(active.shape[0])[:, None]
        source_floor = np.full(active.shape, -1, np.int32)
        for observed in base.stage_history:
            if not observed.any():
                continue
            owned = self.paths.reach(corridor, observed, direction=1, arrival=self.arrival)
            deepest = int(np.nonzero(observed)[0].max())
            source_floor[owned] = np.maximum(source_floor[owned], deepest)
        descendants = source_floor >= 0
        below_source = descendants & (rows > source_floor + base.settings.downward_tolerance_m/base.dy)
        seed_memory = self.optical_aqueous_memory & ~protected & ~gas_supply
        reached = self.paths.reach(corridor & ~protected & ~gas_supply,
                                   seed_memory, direction=1)
        eligible = ~self.ever_co2 | self.optical_aqueous_memory | self.front_memory
        front = reached & below_source & eligible & active & ~protected & ~gas_supply
        if not observed_end or capturing_reference or injection_active or report["pre_injection_negative_control"]:
            front[:] = False
        full = cv2.resize(front.astype(np.uint8),
            (self.context.domain.shape[1], self.context.domain.shape[0]),
            interpolation=cv2.INTER_NEAREST).astype(bool)
        full &= self.context.domain & ~self.reference_co2 & (result["co2_presence_probability"] >= .5)
        corrected_gas = result["gas_probability_conditional_on_co2"].copy()
        old_gas = corrected_gas.copy()
        corrected_gas[full] = 0.
        c = result["co2_presence_probability"]
        result = {**result, "gas_probability_conditional_on_co2": corrected_gas,
            "free_co2_gas_probability": c*corrected_gas,
            "aqueous_co2_probability": c*(1-corrected_gas)}
        self.last_front_mask = full
        self.front_memory |= front
        # A documented gas-supply path can invalidate an old aqueous observation.
        self.optical_aqueous_memory[gas_supply | protected] = False
        self.front_memory[gas_supply | protected] = False
        self.optical_aqueous_memory |= descendants & ~protected & ~gas_supply & (
            co2 >= settings.co2_seed_min) & (gas <= settings.gas_seed_max)
        self.ever_co2 |= active
        self.frame_number += 1
        base.previous_gas = base.previous & (base._small(corrected_gas) >= .6)
        report.update(aqueous_front_memory=FRONT_VERSION,
            temporal_gas_removed_pixels=int(np.count_nonzero(full & (old_gas >= .6))),
            temporal_aqueous_support_pixels=int(full.sum()),
            temporal_optical_seed_pixels=int(seed_memory.sum()),
            temporal_unresolved_old_co2_pixels=int(np.count_nonzero(reached & ~eligible & active)),
            temporal_gas_supply_structure_pixels=int(gas_supply.sum()),
            temporal_history_available=bool(history.any()),
            temporal_paused_for_injection=not observed_end,
            temporal_reference_elapsed_h=self.reference_time,
            temporal_reference_kind="union through first observed frame at/after final injection end",
            temporal_reference_co2_pixels=int(self.reference_co2.sum()),
            temporal_interface_structure_pixels=int(self.paths.interface.sum()),
            temporal_below_frozen_source_structure_pixels=int(below_source.sum()),
            co2_support_changed_by_temporal_rule=False,
            concentrations_or_masses_changed=False,
            supported_gas_pixels=int(np.count_nonzero(self.context.domain & (c >= .5) & (corrected_gas >= .6))))
        return result, report
