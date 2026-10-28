"""Optional, auditable physical support for visible-2D phase interpretation.

This module is upstream of human approval, not part of mass analysis. It uses
the curved log-permeability interface and source/history cues from the older
causal-gas workflow. It deliberately omits saturation smoothing, mandatory
source connectivity, seal bans, and all gas-mass/time caps. Reduced gas support
is moved toward uncertainty (mixed), never automatically to pure aqueous CO2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import scipy
from scipy.ndimage import distance_transform_edt, maximum_filter1d

from darsia.fluidflower.project import sha256_file


PHASE_PHYSICS_VERSION = "visible-phase-physical-support-v1"


@dataclass(frozen=True)
class PhasePhysicsSettings:
    structure_scale: float = 0.2
    interface_scale_m: float = 0.006
    plume_interior_scale_m: float = 0.005
    history_radius_m: float = 0.015
    source_radius_x_m: float = 0.04
    source_radius_y_m: float = 0.03
    reference_rate_kg_s: float = 6.24e-8
    downward_tolerance_m: float = 0.005
    local_column_radius_m: float = 0.02
    minimum_support: float = 0.15
    strong_gas_probability: float = 0.9
    weak_noise_probability: float = 0.65
    minimum_component_area_m2: float = 1e-6

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.structure_scale <= 1:
            raise ValueError("structure_scale must be in (0, 1]")
        if not 0 < self.minimum_support < 1:
            raise ValueError("minimum_support must be in (0, 1)")
        if not 0.6 < self.strong_gas_probability <= 1:
            raise ValueError("strong_gas_probability must be in (0.6, 1]")
        if not 0.5 < self.weak_noise_probability < 1:
            raise ValueError("weak_noise_probability must be in (0.5, 1)")


@dataclass(frozen=True)
class PhasePhysicsContext:
    domain: np.ndarray
    facies: np.ndarray
    permeability: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    injection_stages: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        shape = self.domain.shape
        if len(shape) != 2 or min(shape) < 2 or self.domain.dtype != bool:
            raise ValueError("domain must be a 2D boolean mask")
        if not np.any(self.domain):
            raise ValueError("The porous-medium domain is empty")
        if self.facies.shape != shape or self.permeability.shape != shape:
            raise ValueError("Facies, permeability, and domain must share the registered grid")
        if self.x_m.shape != (shape[1],) or self.y_m.shape != (shape[0],):
            raise ValueError("Physical axes do not match the registered image")
        for axis, direction in ((self.x_m, 1), (self.y_m, -1)):
            if not np.all(np.isfinite(axis)) or not np.all(direction * np.diff(axis) > 0):
                raise ValueError("x must increase rightward and y upward (decreasing by row)")
            if not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=1e-4):
                raise ValueError("Phase physics requires a uniformly registered spatial grid")
        values = self.permeability[self.domain]
        if not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError("Every porous-medium pixel needs finite positive permeability")
        if not np.all(np.isfinite(self.facies[self.domain])):
            raise ValueError("Facies labels must be finite in the porous medium")
        if not self.injection_stages:
            raise ValueError("Explicit injection stages are required for source/time physics")
        for stage in self.injection_stages:
            for key in ("x_m", "y_m", "start_h", "end_h"):
                if not np.isfinite(float(stage[key])):
                    raise ValueError(f"Injection stage has invalid {key}")
            if float(stage["end_h"]) <= float(stage["start_h"]):
                raise ValueError("Injection must end after it starts")
            uncertainty = float(stage.get("uncertainty_m", 0.))
            if not np.isfinite(uncertainty) or uncertainty < 0:
                raise ValueError("Injection coordinate uncertainty must be finite and nonnegative")
            if not (self.x_m[0] <= stage["x_m"] <= self.x_m[-1]
                    and self.y_m[-1] <= stage["y_m"] <= self.y_m[0]):
                raise ValueError("Injection point lies outside the registered physical grid")
            segments = stage.get("rate_segments") or []
            if not segments:
                raise ValueError("Rate-scaled source support requires explicit rate segments")
            end = float(stage["start_h"])
            for segment in segments:
                a, b, rate = (float(segment[k]) for k in ("start_h", "end_h", "rate_kg_s"))
                if not all(np.isfinite([a, b, rate])) or b <= a or rate < 0:
                    raise ValueError("Invalid injection rate segment")
                if abs(a - end) > 1e-6:
                    raise ValueError("Rate segments must be contiguous and chronological")
                end = b
            if abs(end - float(stage["end_h"])) > 1e-6:
                raise ValueError("Rate segments must cover their injection stage")


def load_physics_profile(model_root: Path) -> dict[str, Any] | None:
    """Opt in per immutable model bundle; old models remain optical-only."""
    path = Path(model_root) / "phase_physics.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != PHASE_PHYSICS_VERSION:
        raise ValueError("Unknown phase-physics profile format")
    PhasePhysicsSettings(**payload.get("settings", {}))
    if not isinstance(payload.get("runs"), dict) or not payload["runs"]:
        raise ValueError("Phase-physics profile has no registered run contexts")
    for key, entry in payload["runs"].items():
        if not isinstance(entry, dict) or any(name not in entry for name in (
                "path", "sha256", "native_manifest_sha256", "injection_stages")):
            raise ValueError(f"Invalid registered phase-physics context: {key}")
    return payload


def physics_runtime_identity(profile: Mapping[str, Any] | None) -> str | None:
    if profile is None:
        return None
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True).encode())
    digest.update(sha256_file(Path(__file__)).encode())
    digest.update(f"{np.__version__}:{cv2.__version__}:{scipy.__version__}".encode())
    return digest.hexdigest()


def load_run_context(
    model_root: Path, profile: Mapping[str, Any], dataset: str, run: str,
    *, native_manifest_sha256: str,
) -> tuple[PhasePhysicsContext, dict[str, Any]]:
    key = f"{dataset.casefold()}/{run.casefold()}"
    entry = profile["runs"].get(key)
    if not entry:
        raise ValueError(f"No registered phase-physics context for {key}; cannot silently skip physics")
    if entry.get("native_manifest_sha256") != native_manifest_sha256:
        raise ValueError(f"{key}: physics context belongs to different registered source frames")
    path = Path(model_root) / entry["path"]
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"{key}: phase-physics context checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        context = PhasePhysicsContext(
            domain=archive["domain"], facies=archive["facies"],
            permeability=archive["permeability"], x_m=archive["x_m"], y_m=archive["y_m"],
            injection_stages=tuple(entry["injection_stages"]),
        )
    return context, dict(entry)


def freeze_run_context(
    model_root: Path, dataset: str, run: str, context: PhasePhysicsContext,
    *, native_manifest: Path, provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Snapshot registered arrays into a NEW bundle, without modifying sources."""
    if any(not token or not all(c.isalnum() or c in "_-" for c in token)
           for token in (dataset, run)):
        raise ValueError("Series and run IDs must be simple file-safe identifiers")
    relative = Path("physics_context") / dataset.casefold() / f"{run.casefold()}.npz"
    path = Path(model_root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, domain=context.domain, facies=context.facies,
                            permeability=context.permeability,
                            x_m=context.x_m, y_m=context.y_m)
    return {
        "path": relative.as_posix(), "sha256": sha256_file(path),
        "native_manifest_sha256": sha256_file(native_manifest),
        "injection_stages": list(context.injection_stages),
        "provenance": dict(provenance),
    }


def _interface_support(k: np.ndarray, valid: np.ndarray, dx: float, dy: float,
                       scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Relative permeability contrast and curvature, independent of facies IDs/units."""
    # Extend the nearest physical material outside the domain before smoothing:
    # a zero/NaN in the free-water region must not manufacture a sealing layer.
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    log_k = np.log(k[tuple(nearest)]).astype(np.float32)
    smooth = cv2.GaussianBlur(log_k, (0, 0), sigmaX=max(.5, scale / dx),
                              sigmaY=max(.5, scale / dy), borderType=cv2.BORDER_REPLICATE)
    gy, gx = np.gradient(smooth, -dy, dx)
    magnitude = np.hypot(gx, gy)
    nx = gx / np.maximum(magnitude, 1e-7)
    ny = gy / np.maximum(magnitude, 1e-7)
    curvature = np.tanh((np.gradient(nx, dx, axis=1)
                         + np.gradient(ny, -dy, axis=0)) * scale)
    # Use log contrast over a physical length, not normalization by the
    # strongest (possibly negligible) gradient anywhere in the image.
    contrast = 1 - np.exp(-magnitude * scale)
    support = contrast * np.clip(-ny, 0, 1) * (.75 + .25 * np.abs(curvature))
    return np.where(valid, support, 0).astype(np.float32), curvature


class PhasePhysicsSequence:
    """Chronological pre-review support. No future frames or mass values are used."""

    def __init__(self, context: PhasePhysicsContext, settings: PhasePhysicsSettings):
        self.context, self.settings = context, settings
        height, width = context.domain.shape
        self.target = (max(2, round(width * settings.structure_scale)),
                       max(2, round(height * settings.structure_scale)))
        self.valid = self._small(context.domain.astype(np.uint8), nearest=True).astype(bool)
        if not self.valid.any():
            raise ValueError("Physics structure grid loses the entire analysis domain")
        self.x = np.linspace(context.x_m[0], context.x_m[-1], self.target[0])
        self.y = np.linspace(context.y_m[0], context.y_m[-1], self.target[1])
        self.dx, self.dy = float(self.x[1] - self.x[0]), float(self.y[0] - self.y[1])
        k = self._small(context.permeability.astype(np.float32), nearest=True)
        self.barrier, self.curvature = _interface_support(
            k, self.valid, self.dx, self.dy, settings.interface_scale_m)
        self.previous = np.zeros(self.valid.shape, dtype=bool)
        self.previous_gas = np.zeros_like(self.previous)
        self.stage_history = [np.zeros_like(self.previous) for _ in context.injection_stages]
        self.last_time = -np.inf

    def _small(self, array: np.ndarray, *, nearest: bool = False) -> np.ndarray:
        return cv2.resize(array, self.target,
                          interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)

    def _near(self, mask: np.ndarray, radius: float) -> np.ndarray:
        if not mask.any():
            return np.zeros_like(mask)
        return distance_transform_edt(~mask, sampling=(self.dy, self.dx)) <= radius

    def _source(self, stage: Mapping[str, Any], time: float) -> np.ndarray:
        radius_scale = 0.0
        for part in stage["rate_segments"]:
            if part["start_h"] <= time < part["end_h"] and part["rate_kg_s"] > 0:
                radius_scale = np.clip(np.sqrt(part["rate_kg_s"] /
                                              self.settings.reference_rate_kg_s), .35, 2.25)
                break
        if not radius_scale:
            return np.zeros_like(self.valid)
        uncertainty = max(0., float(stage.get("uncertainty_m", 0.)))
        rx = self.settings.source_radius_x_m * radius_scale + uncertainty
        ry = self.settings.source_radius_y_m * radius_scale + uncertainty
        return self.valid & (((self.x[None, :] - stage["x_m"]) / rx) ** 2
                             + ((self.y[:, None] - stage["y_m"]) / ry) ** 2 <= 1)

    def apply(self, predictions: Mapping[str, np.ndarray], elapsed_h: float
              ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        time = float(elapsed_h)
        if not np.isfinite(time) or time <= self.last_time:
            raise ValueError("Phase physics requires strictly increasing, finite frame times")
        co2 = np.asarray(predictions["co2_presence_probability"], dtype=np.float32)
        gas = np.asarray(predictions["gas_probability_conditional_on_co2"], dtype=np.float32)
        domain = self.context.domain
        if co2.shape != domain.shape or gas.shape != domain.shape:
            raise ValueError("Prediction and registered physics grid differ; implicit resizing is forbidden")
        if any(not np.all(np.isfinite(a) & (a >= 0) & (a <= 1)) for a in (co2, gas)):
            raise ValueError("Optical probabilities must be finite and in [0, 1]")
        c, g = self._small(co2), self._small(gas)
        active = self.valid & (c >= .5)
        pre_injection = time < min(float(s["start_h"]) for s in self.context.injection_stages)
        if pre_injection:
            active[:] = False
        gas_support = active & (g >= .6)
        source_masks = [self._source(stage, time) for stage in self.context.injection_stages]
        active_source = np.logical_or.reduce(source_masks)
        history_near = self._near(self.previous_gas, self.settings.history_radius_m)
        co2_history_near = self._near(self.previous, self.settings.history_radius_m)
        # A connected, optically supported excursion from a currently injecting
        # source is exempt, even at the leading edge of a fast capillary jump.
        _, labels = cv2.connectedComponents(gas_support.astype(np.uint8), connectivity=8)
        source_connected = np.zeros_like(active)
        for index, (stage, source) in enumerate(zip(self.context.injection_stages, source_masks)):
            if stage["start_h"] <= time < stage["end_h"]:
                seeds = source | self._near(self.stage_history[index], self.settings.history_radius_m)
                ids = np.unique(labels[seeds & gas_support])
                connected = np.isin(labels, ids[ids > 0])
                self.stage_history[index] |= connected
                if source.any():
                    source_connected |= connected
        # Only observed local injection-era columns constrain later downward
        # expansion. Unsupported columns and missing early history stay unknown.
        downward = np.zeros_like(active)
        column_radius = max(0, round(self.settings.local_column_radius_m / self.dx))
        rows = np.arange(self.valid.shape[0])[:, None]
        for stage, history in zip(self.context.injection_stages, self.stage_history):
            if time < stage["end_h"] or not history.any():
                continue
            deepest = np.max(np.where(history, rows, -1), axis=0)
            deepest = maximum_filter1d(deepest, size=2 * column_radius + 1, mode="nearest")
            below = (deepest[None, :] >= 0) & (
                rows > deepest[None, :] + self.settings.downward_tolerance_m / self.dy)
            downward |= below
        downward &= ~(active_source | source_connected)
        padded = np.pad(active, 1, constant_values=False)
        interior = distance_transform_edt(padded, sampling=(self.dy, self.dx))[1:-1, 1:-1]
        interior = np.clip(interior / self.settings.plume_interior_scale_m, 0, 1)
        support = np.maximum.reduce([interior, self.barrier, .5 * history_near.astype(np.float32),
                                     active_source.astype(np.float32), source_connected.astype(np.float32)])
        support[downward] = np.minimum(support[downward], self.settings.minimum_support)
        support = self.settings.minimum_support + (1 - self.settings.minimum_support) * support
        full_support = cv2.resize(support.astype(np.float32), (gas.shape[1], gas.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
        adjusted_gas = gas.copy()
        weak_gas = domain & (co2 >= .5) & (gas >= .6) & (gas < self.settings.strong_gas_probability)
        adjusted_gas[weak_gas] = .5 + (gas[weak_gas] - .5) * full_support[weak_gas]
        # Remove only weak, small, temporally unsupported CO2 speckles. Large or
        # high-confidence objects are not filtered merely because disconnected.
        n, components, stats, _ = cv2.connectedComponentsWithStats(active.astype(np.uint8), 8)
        maxima = np.zeros(n, dtype=np.float32)
        np.maximum.at(maxima, components.ravel(), c.ravel())
        protected = np.bincount(components[co2_history_near | active_source], minlength=n) > 0
        small = stats[:, cv2.CC_STAT_AREA] * self.dx * self.dy < self.settings.minimum_component_area_m2
        noise_labels = small & ~protected & (maxima < self.settings.weak_noise_probability)
        noise_labels[0] = False
        noise_ids = np.flatnonzero(noise_labels)
        noise = cv2.resize(np.isin(components, noise_ids).astype(np.uint8),
                           (gas.shape[1], gas.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        # Retain strong native pixels lost in the structural downsample.
        noise &= domain & (co2 >= .5) & (co2 < self.settings.weak_noise_probability)
        adjusted_co2 = np.where(domain, co2, 0).astype(np.float32)
        adjusted_co2[noise] = np.minimum(adjusted_co2[noise], .49)
        adjusted_gas[~domain] = 0
        if pre_injection:
            adjusted_co2[:] = 0
            adjusted_gas[:] = 0
        report = {
            "elapsed_h": time, "excluded_domain_pixels": int((~domain).sum()),
            "pre_injection_negative_control": pre_injection,
            "gas_support_reduced_pixels": int(np.count_nonzero(adjusted_gas[domain] < gas[domain] - 1e-6)),
            "raw_gas_pixels": int(np.count_nonzero(domain & (co2 >= .5) & (gas >= .6))),
            "supported_gas_pixels": int(np.count_nonzero(domain & (adjusted_co2 >= .5) & (adjusted_gas >= .6))),
            "raw_co2_pixels_outside_domain": int(np.count_nonzero(~domain & (co2 >= .5))),
            "structure_pixel_area_m2": self.dx * self.dy,
            "weak_noise_pixels": int(noise.sum()),
            "downward_structural_pixels": int((downward & gas_support).sum()),
            "source_connected_structural_pixels": int(source_connected.sum()),
            "mean_interface_support": float(self.barrier[self.valid].mean()),
            "max_absolute_curvature": float(np.abs(self.curvature[self.valid]).max()),
            "injection_history_available": [bool(h.any()) for h in self.stage_history],
        }
        self.previous = self.valid & (self._small(adjusted_co2) >= .5)
        self.previous_gas = self.previous & (self._small(adjusted_gas) >= .6)
        self.last_time = time
        free = adjusted_co2 * adjusted_gas
        return {
            **predictions,
            "co2_presence_probability": adjusted_co2,
            "gas_probability_conditional_on_co2": adjusted_gas,
            "untouched_water_probability": 1 - adjusted_co2,
            "aqueous_co2_probability": adjusted_co2 - free,
            "free_co2_gas_probability": free,
        }, report
