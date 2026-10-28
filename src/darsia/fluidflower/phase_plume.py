"""Versioned, causal phase support with separate injection-plume histories.

This is an auditable classification prior, not a flow or mass solver. It never
creates CO2 or gas. Native isolated-component filtering is independent of the
optional facies/permeability geometry. Published legacy rules are not changed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import scipy
import skimage
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order
from skimage.graph import MCP_Geometric

from .project import sha256_file

VERSION = "per-injection-curved-gas-support-v1"


@dataclass(frozen=True)
class PlumeSettings:
    structure_scale: float = .2
    registration_tolerance_m: float = .003
    connectivity_gap_m: float = .003
    source_radius_m: float = .04
    minimum_component_area_m2: float = 2.5e-6
    anchor_component_area_m2: float = 25e-6
    noise_protection_radius_m: float = .005
    interface_scale_m: float = .006
    minimum_log_contrast: float = .05
    tangent_normal_cosine_max: float = .7

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.structure_scale > 1 or self.tangent_normal_cosine_max > 1:
            raise ValueError("Scale and direction cosine must not exceed one")
        if self.anchor_component_area_m2 < self.minimum_component_area_m2:
            raise ValueError("Noise anchors must be larger than discarded components")


@dataclass(frozen=True)
class PlumeContext:
    domain: np.ndarray
    facies: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    injection_stages: tuple
    permeability: np.ndarray | None = None
    # V5 keeps the legacy uniform axes solely for optical operators that need
    # a nominal spacing. These fields are the exact registered physical
    # coordinates and pixel area when a run has a projective camera geometry.
    x_m_native: np.ndarray | None = None
    y_m_native: np.ndarray | None = None
    pixel_area_m2: np.ndarray | None = None

    def __post_init__(self):
        shape = self.domain.shape
        if len(shape) != 2 or min(shape) < 2 or self.domain.dtype != bool or not self.domain.any():
            raise ValueError("A nonempty registered porous domain is required")
        if self.facies.shape != shape or not np.isfinite(self.facies[self.domain]).all():
            raise ValueError("Facies and domain must share a finite registered grid")
        for axis, size, sign in ((self.x_m, shape[1], 1), (self.y_m, shape[0], -1)):
            if axis.shape != (size,) or not np.isfinite(axis).all():
                raise ValueError("Invalid physical coordinates")
            step = np.diff(axis)
            if not (sign * step > 0).all() or not np.allclose(step, step[0], rtol=1e-4):
                raise ValueError("Uniform registered x-right/y-up coordinates required")
        exact = (self.x_m_native, self.y_m_native, self.pixel_area_m2)
        if any(value is not None for value in exact):
            if any(value is None for value in exact):
                raise ValueError("Exact native physical geometry must include x, y and pixel area")
            if any(value.shape != shape or not np.isfinite(value).all() for value in exact):
                raise ValueError("Exact native physical geometry differs from the domain grid")
            if not (self.pixel_area_m2[self.domain] > 0).all():
                raise ValueError("Exact native pixel area must be positive in the domain")
            if not ((np.diff(self.x_m_native, axis=1)[self.domain[:, 1:]] > 0).all()
                    and (np.diff(self.y_m_native, axis=0)[self.domain[1:, :]] < 0).all()):
                raise ValueError("Exact native physical coordinates have the wrong orientation")
        if self.permeability is not None:
            if self.permeability.shape != shape:
                raise ValueError("Permeability grid mismatch")
            k = self.permeability[self.domain]
            if not (np.isfinite(k) & (k > 0)).all():
                raise ValueError("Measured permeability must be finite and positive")
        for stage in self.injection_stages:
            if not all(np.isfinite(float(stage[k])) for k in ("x_m", "y_m", "start_h", "end_h")):
                raise ValueError("Invalid injection coordinates or times")
            if stage["end_h"] <= stage["start_h"]:
                raise ValueError("Injection must end after it starts")
            if not (self.x_m[0] <= stage["x_m"] <= self.x_m[-1]
                    and self.y_m[-1] <= stage["y_m"] <= self.y_m[0]):
                raise ValueError("Injection outside the registered grid")
            uncertainty = float(stage.get("uncertainty_m", 0))
            if not np.isfinite(uncertainty) or uncertainty < 0:
                raise ValueError("Invalid injection coordinate uncertainty")
        if not self.injection_stages:
            raise ValueError("Explicit injection stages are required")


def load_profile(root):
    path = Path(root) / "phase_plume_support.json"
    if not path.is_file():
        return None
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("format") != VERSION or not profile.get("runs"):
        raise ValueError("Invalid plume-support profile")
    PlumeSettings(**profile.get("settings", {}))
    return profile


def runtime_identity(profile):
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True).encode())
    digest.update(sha256_file(Path(__file__)).encode())
    digest.update(f"{np.__version__}:{scipy.__version__}:{cv2.__version__}:{skimage.__version__}".encode())
    return digest.hexdigest()


def load_context(root, profile, dataset, run, *, native_manifest_sha256):
    entry = profile["runs"][f"{dataset}/{run}".casefold()]
    path = Path(root) / entry["path"]
    if entry["native_manifest_sha256"] != native_manifest_sha256 or sha256_file(path) != entry["sha256"]:
        raise ValueError("Plume context or registered source revision changed")
    with np.load(path, allow_pickle=False) as data:
        exact_keys = ("x_m_native", "y_m_native", "pixel_area_m2")
        if any(key in data for key in exact_keys) and not all(key in data for key in exact_keys):
            raise ValueError("Incomplete exact native physical geometry")
        provenance = entry.get("provenance") or {}
        if not isinstance(provenance, dict):
            raise ValueError("Invalid plume-context provenance")
        v5_geometry = provenance.get("v5_run_geometry", entry.get("v5_run_geometry"))
        track1_photo = provenance.get("v5_track1_photo_geometry")
        if provenance.get("v5_raw_photometry") is not None and v5_geometry is None and track1_photo is None:
            raise ValueError("V5 raw photometry lacks run-native geometry or photo source binding")
        if track1_photo is not None:
            if (v5_geometry is not None or not isinstance(track1_photo, dict)
                    or track1_photo.get("contract") != "ffac-v5-track1-photo-geometry-cohort-v1"
                    or track1_photo.get("validation_only") is not True
                    or track1_photo.get("mass_eligible") is not False
                    or entry.get("permeability_available") is not False
                    or any(key in data for key in ("permeability", *exact_keys))):
                raise ValueError("Track1 optical-only context contains physical geometry")
        if v5_geometry is not None:
            if not isinstance(v5_geometry, dict):
                raise ValueError("Unknown V5 run geometry contract")
            contract = v5_geometry.get("contract")
            if contract == "ffac-v5-run-native-geometry-validation-only-v1":
                if (v5_geometry.get("validation_only") is not True
                        or v5_geometry.get("mass_eligible") is not False):
                    raise ValueError("V5 uncertain-facies geometry lost validation-only limits")
            elif contract != "ffac-v5-run-native-geometry-v1":
                raise ValueError("Unknown V5 run geometry contract")
        has_exact = all(key in data for key in exact_keys)
        if v5_geometry is not None and not has_exact:
            raise ValueError("V5 run geometry lacks exact native physical coordinates")
        context = PlumeContext(data["domain"], data["facies"], data["x_m"], data["y_m"],
            tuple(entry["injection_stages"]), data["permeability"] if "permeability" in data else None,
            *(tuple(data[key] for key in exact_keys) if has_exact else (None, None, None)))
    return context, dict(entry)


def freeze_context(root, dataset, run, context, *, native_manifest, provenance):
    if any(not s or not all(c.isalnum() or c in "_-" for c in s) for s in (dataset, run)):
        raise ValueError("Unsafe series or run identifier")
    relative = Path("plume_context") / dataset / f"{run}.npz"
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: getattr(context, k) for k in ("domain", "facies", "x_m", "y_m")}
    if context.permeability is not None:
        arrays["permeability"] = context.permeability
    if context.x_m_native is not None:
        arrays.update(x_m_native=context.x_m_native, y_m_native=context.y_m_native,
                      pixel_area_m2=context.pixel_area_m2)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {"path": relative.as_posix(), "sha256": sha256_file(path),
            "native_manifest_sha256": sha256_file(native_manifest),
            "injection_stages": list(context.injection_stages), "provenance": provenance,
            "permeability_available": context.permeability is not None}


class PlumeSequence:
    """First-arrival ownership cannot be reassigned by a later/deeper injection.

Gas history is frozen separately at each injection end. Directed paths can go
upward or along a locally resolved layer tangent, but cannot descend outside
that injection's observed curved gas region plus registration tolerance.
Unresolved ownership/history is reported instead of erasing large gas objects.
"""
    def __init__(self, context, settings=None):
        self.context, self.settings = context, settings or PlumeSettings()
        h, w = context.domain.shape
        self.target = max(2, round(w*self.settings.structure_scale)), max(2, round(h*self.settings.structure_scale))
        self.valid = self.small(context.domain, nearest=True)
        if not self.valid.any():
            raise ValueError("Structure grid loses the analysis domain")
        self.x = np.linspace(context.x_m[0], context.x_m[-1], self.target[0])
        self.y = np.linspace(context.y_m[0], context.y_m[-1], self.target[1])
        self.dx, self.dy = self.x[1]-self.x[0], self.y[0]-self.y[1]
        self.pixel_area = (context.x_m[1]-context.x_m[0])*(context.y_m[0]-context.y_m[1])
        self.owner = np.zeros(self.valid.shape, np.int16)
        self.history = [np.zeros_like(self.valid) for _ in context.injection_stages]
        self.co2_anchors = np.zeros_like(self.valid)
        self.gas_anchors = np.zeros_like(self.valid)
        self.last_time = -np.inf
        self._edges = self._make_edges()
        self.last_rejected = np.zeros_like(context.domain)

    def small(self, a, nearest=False):
        return cv2.resize(np.asarray(a, np.float32), self.target,
                          interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA) > (.5 if nearest else 0)

    def full(self, a):
        return cv2.resize(np.asarray(a, np.uint8), self.context.domain.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)

    def near(self, a, radius):
        if not a.any():
            return np.zeros_like(a)
        return distance_transform_edt(~a, sampling=(self.dy, self.dx)) <= radius

    def _make_edges(self):
        s = self.settings
        nearest = distance_transform_edt(~self.valid, return_distances=False, return_indices=True)
        k = self.context.permeability
        if k is not None:
            material = cv2.resize(k.astype(np.float32), self.target, interpolation=cv2.INTER_NEAREST)
            fields = [np.log(material[tuple(nearest)])]
        else:
            labels = cv2.resize(self.context.facies.astype(np.float32), self.target, interpolation=cv2.INTER_NEAREST)
            labels = labels[tuple(nearest)]
            # Facies numbers are categorical, not a fabricated permeability rank.
            fields = [(labels == label).astype(np.float32) for label in np.unique(labels[self.valid])]
        magnitude = np.zeros(self.valid.shape, np.float32)
        nx, ny = magnitude.copy(), magnitude.copy()
        for field in fields:
            smooth = cv2.GaussianBlur(field, (0, 0), sigmaX=max(.5, s.interface_scale_m/self.dx),
                                      sigmaY=max(.5, s.interface_scale_m/self.dy))
            gy, gx = np.gradient(smooth, self.dy, self.dx)
            mag = np.hypot(gx, gy)
            update = mag > magnitude
            nx[update], ny[update] = gx[update]/mag[update], gy[update]/mag[update]
            magnitude = np.maximum(magnitude, mag)
        interface = self.valid & (magnitude*s.interface_scale_m >= s.minimum_log_contrast)
        ids = np.arange(self.valid.size).reshape(self.valid.shape)
        origins, targets, down = [], [], []
        h, w = self.valid.shape
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                a = (slice(max(0, -dr), min(h, h-dr)), slice(max(0, -dc), min(w, w-dc)))
                b = (slice(max(0, dr), min(h, h+dr)), slice(max(0, dc), min(w, w+dc)))
                allowed = self.valid[a] & self.valid[b]
                if dr == 0:
                    aligned = np.abs(nx[a]) <= s.tangent_normal_cosine_max
                    # Non-interface horizontal edges can only be used INSIDE
                    # the frozen historical pocket (checked separately in reach).
                    restricted = ~(interface[a] & interface[b] & aligned)
                else:
                    restricted = np.full(allowed.shape, dr > 0)
                origins.append(ids[a][allowed])
                targets.append(ids[b][allowed])
                down.append(restricted[allowed])
        return np.concatenate(origins), np.concatenate(targets), np.concatenate(down)

    def source(self, stage, time):
        if not stage["start_h"] <= time < stage["end_h"]:
            return np.zeros_like(self.valid)
        radius = self.settings.source_radius_m + float(stage.get("uncertainty_m", 0))
        return self.valid & ((self.x[None, :]-stage["x_m"])**2
                             + (self.y[:, None]-stage["y_m"])**2 <= radius**2)

    def noise_filter(self, mask, previous_anchors, active_source):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA]*self.pixel_area
        anchors = areas >= self.settings.anchor_component_area_m2
        anchors[0] = False
        current_anchors = self.small(anchors[labels]) & self.valid
        protected = self.full(self.near(current_anchors | previous_anchors,
                                       self.settings.noise_protection_radius_m) | active_source)
        touches = np.bincount(labels[protected], minlength=count) > 0
        discard = (areas < self.settings.minimum_component_area_m2) & ~touches
        discard[0] = False
        return discard[labels], current_anchors, int(discard.sum())

    def assign_owners(self, corridor, sources):
        best = np.full(self.valid.shape, np.inf)
        proposed = np.zeros_like(self.owner)
        for index, source in enumerate(sources, 1):
            allowed = corridor & ((self.owner == 0) | (self.owner == index))
            seeds = allowed & ((self.owner == index) | source)
            # Only the edge of a large seed set is needed for geodesic distance.
            boundary = seeds & ~cv2.erode(seeds.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            if not seeds.any():
                continue
            if not boundary.any():
                boundary = seeds
            costs = np.where(allowed, 1., np.inf)
            distance, _ = MCP_Geometric(costs, fully_connected=True, sampling=(self.dy, self.dx)).find_costs(np.argwhere(boundary))
            distance[seeds] = 0
            take = (self.owner == 0) & (distance < best)
            best[take], proposed[take] = distance[take], index
        self.owner[(self.owner == 0) & corridor] = proposed[(self.owner == 0) & corridor]

    def reach(self, corridor, history):
        tolerance = self.near(history, self.settings.registration_tolerance_m) & self.valid
        seeds = np.flatnonzero(corridor & tolerance)
        if not len(seeds):
            return tolerance
        source, target, restricted = self._edges
        allowed = corridor.ravel()[source] & corridor.ravel()[target]
        allowed &= ~restricted | (tolerance.ravel()[source] & tolerance.ravel()[target])
        size = corridor.size
        rows = np.concatenate((source[allowed], np.full(len(seeds), size)))
        cols = np.concatenate((target[allowed], seeds))
        graph = csr_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(size+1, size+1))
        ids = breadth_first_order(graph, size, directed=True, return_predecessors=False)
        result = np.zeros(size+1, bool)
        result[ids] = True
        return result[:size].reshape(corridor.shape) | tolerance

    def apply(self, predictions, elapsed_h):
        time = float(elapsed_h)
        if not np.isfinite(time) or time < self.last_time:
            raise ValueError("Plume support requires chronological finite times")
        c = np.asarray(predictions["co2_presence_probability"], np.float32)
        g = np.asarray(predictions["gas_probability_conditional_on_co2"], np.float32)
        for a in (c, g):
            if a.shape != self.context.domain.shape or not (np.isfinite(a) & (a >= 0) & (a <= 1)).all():
                raise ValueError("Invalid optical score array")
        domain = self.context.domain
        co2 = np.where(domain, c, 0)
        gas = np.where(domain, g, 0)
        pre = time < min(s["start_h"] for s in self.context.injection_stages)
        raw_gas = domain & (c >= .5) & (g >= .6)
        sources = [self.source(s, time) for s in self.context.injection_stages]
        active_source = np.logical_or.reduce(sources)
        noise, anchors, nc = self.noise_filter(domain & (c >= .5), self.co2_anchors, active_source)
        co2[noise] = np.minimum(co2[noise], .49)
        # Fine gas fragmentation INSIDE an established CO2 body is not isolated
        # background noise. Its classification belongs to the directional rule.
        gas_noise, gas_anchors, ng = self.noise_filter((co2 >= .5) & (gas >= .6),
            self.gas_anchors | anchors, active_source)
        gas[gas_noise] = np.minimum(gas[gas_noise], .59)
        observed = self.small(co2 >= .5) & self.valid
        corridor = self.near(observed, self.settings.connectivity_gap_m) & self.valid
        if pre:
            co2[:] = gas[:] = 0
            observed[:] = corridor[:] = False
        self.assign_owners(corridor, sources)
        structural_gas = self.small((co2 >= .5) & (gas >= .6)) & observed
        rejected = np.zeros_like(self.valid)
        stage_reports = []
        for index, (stage, history) in enumerate(zip(self.context.injection_stages, self.history), 1):
            owned = self.owner == index
            active = stage["start_h"] <= time < stage["end_h"]
            if active:
                history |= structural_gas & owned
            excluded = np.zeros_like(self.valid)
            if time >= stage["end_h"] and history.any():
                eligible = self.reach(corridor & owned, history)
                excluded = owned & ~eligible
                rejected |= excluded
            stage_reports.append({"id": stage.get("id", f"injection-{index}"), "active": bool(active),
                "end_h": stage["end_h"], "history_pixels": int(history.sum()),
                "owned_pixels": int(owned.sum()), "rejected_structural_gas_pixels": int((excluded & structural_gas).sum()),
                "history_missing": not bool(history.any())})
        self.last_rejected = self.full(rejected) & (co2 >= .5) & (gas >= .6)
        gas[self.last_rejected] = np.minimum(gas[self.last_rejected], .59)
        if not pre:
            self.co2_anchors |= anchors
            self.gas_anchors |= gas_anchors & ~rejected
        self.last_time = time
        report = {"elapsed_h": time, "pre_injection_negative_control": pre,
            "raw_gas_pixels": int(raw_gas.sum()), "raw_co2_pixels_outside_domain": int((~domain & (c >= .5)).sum()),
            "supported_gas_pixels": int(((co2 >= .5) & (gas >= .6)).sum()),
            "isolated_co2_pixels_removed": int(noise.sum()), "isolated_co2_components_removed": nc,
            "isolated_gas_pixels_reclassified": int(gas_noise.sum()), "isolated_gas_components_removed": ng,
            "directional_gas_pixels_reclassified": int(self.last_rejected.sum()),
            "unowned_structural_gas_pixels": int((structural_gas & (self.owner == 0)).sum()),
            "stages": stage_reports, "permeability_available": self.context.permeability is not None,
            "native_pixel_area_m2": float(self.pixel_area), "structure_pixel_area_m2": float(self.dx*self.dy)}
        free = co2*gas
        return {**predictions, "co2_presence_probability": co2,
                "gas_probability_conditional_on_co2": gas, "untouched_water_probability": 1-co2,
                "aqueous_co2_probability": co2-free, "free_co2_gas_probability": free}, report
