"""Original pooled native colour paths with explicit cohort and input DATA.

This module provides the numerical sampler, equal-run fit and new package
writer. Its caller must construct FrameData through an actual checked native
source/Build adapter. Matching descriptor declarations here do not certify
that adapter, a Build result, a physical context or a production release.
Existing saved models and the generic embedding workflow are unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from .phase_expert_knowledge import expert_knowledge_of

FORMAT = "darsia-pooled-native-optical-recipe-v1"
METHOD = "equal-run-native-relative-anchors-v1"
SETUP_METHOD = "equal-run-corrected-facies-clearwater-relative-anchors-v1"
_FIELDS = {"path", "sha256", "size_bytes"}


def _json(value):
    """Compare immutable JSON DATA without conflating booleans and numbers."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _descriptor(value):
    """Validate an explicit file identity; never discover or read another path."""
    if (not isinstance(value, dict) or set(value) != _FIELDS
            or not isinstance(value["path"], str) or not Path(value["path"]).is_absolute()
            or ".." in Path(value["path"]).parts
            or not isinstance(value["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
            or type(value["size_bytes"]) is not int or value["size_bytes"] < 0):
        raise ValueError("Optical inputs require exact absolute path/SHA/size descriptors")


@dataclass(frozen=True)
class OpticalRecipe:
    """Immutable explicit recipe, including honest unresolved upstream slots."""
    encoded: str

    @property
    def document(self):
        """Return a detached copy rather than exposing mutable stored DATA."""
        return json.loads(self.encoded)

    @property
    def sha256(self):
        """Bind samples to every source, selection, role and method setting."""
        return hashlib.sha256(self.encoded.encode()).hexdigest()

    @property
    def missing_inputs(self):
        """Name missing implementation inputs without inventing QA or authority."""
        value = self.document
        setup = value["method"] == SETUP_METHOD
        missing = [key for key in ("adopted_revision", "build_context", "phase_configuration", "native_configuration")
                   if not (setup and key == "phase_configuration")
                   if value[key] is None]
        for run, row in value["runs"].items():
            missing.extend(run + "." + key for key in ("baseline", "clearwater", "profile") if row[key] is None)
            missing.extend(run + "." + frame["case_id"] + ".source" for frame in row["frames"] if frame["source"] is None)
        return tuple(missing)

    def require_inputs(self):
        """Refuse only genuinely absent upstream bindings before sampling/writing."""
        if self.missing_inputs:
            raise ValueError("Actual adopted/Build/native source inputs are not supplied: " + ", ".join(self.missing_inputs))


def recipe_from_document(value):
    """Prepare a typed recipe now; missing real Build descriptors stay explicit.

    The cohort is caller-supplied DATA. A campaign must pass its accepted fit
    set here, separately from held-out/excluded runs; no old cohort is inferred.
    """
    keys = {"format", "method", "fit_runs", "excluded_runs", "roles", "runs", "adopted_revision",
            "build_context", "phase_configuration", "native_configuration", "sampling", "flash_bounds", "path_name_prefix"}
    if not isinstance(value, dict) or set(value) != keys or value["format"] != FORMAT or value["method"] not in {METHOD, SETUP_METHOD}:
        raise ValueError("An explicit supported pooled optical recipe is required")
    fit, excluded = value["fit_runs"], value["excluded_runs"]
    for runs in (fit, excluded):
        if (not isinstance(runs, list)
                or any(not isinstance(run, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", run) is None for run in runs)
                or len(runs) != len(set(runs))):
            raise ValueError("Optical cohorts need ordered unique explicit run IDs")
    if len(fit) < 1 or set(fit) & set(excluded) or not isinstance(value["runs"], dict) or set(value["runs"]) != set(fit):
        raise ValueError("Every fit run, and no held-out/excluded run, needs its own input selection")
    roles = value["roles"]
    from ..layer_painting import ROLE_POROSITY
    if not isinstance(roles, list) or not roles:
        raise ValueError("The actual adopted table roles are required")
    ids = []
    for row in roles:
        if (not isinstance(row, dict) or set(row) != {"facies_id", "name", "analysis_role"}
                or type(row["facies_id"]) is not int or row["facies_id"] < 0
                or not isinstance(row["name"], str) or not row["name"].strip()
                or not isinstance(row["analysis_role"], str)
                or row["analysis_role"] not in {"porous_medium", "frame", *ROLE_POROSITY}):
            raise ValueError("Every optical facies needs its literal adopted ID/name/role")
        ids.append(row["facies_id"])
    if len(ids) != len(set(ids)) or not any(row["analysis_role"] == "porous_medium" for row in roles):
        raise ValueError("Optical roles need unique IDs and an actual porous medium")
    slots = [(key, value[key]) for key in ("adopted_revision", "build_context", "phase_configuration", "native_configuration")]
    for run, row in value["runs"].items():
        if not isinstance(row, dict) or set(row) != {"baseline", "clearwater", "profile", "frames"}:
            raise ValueError("Each optical run needs explicit baseline/clearwater/profile/frames")
        slots.extend((run + "." + key, row[key]) for key in ("baseline", "clearwater", "profile"))
        frames = row["frames"]
        if not isinstance(frames, list) or not frames:
            raise ValueError("Every optical run needs selected camera times")
        seen = set()
        for frame in frames:
            if (not isinstance(frame, dict) or set(frame) != {"case_id", "frame_index", "elapsed_h", "target_h", "source"}
                    or not isinstance(frame["case_id"], str) or not frame["case_id"] or frame["case_id"] in seen
                    or type(frame["frame_index"]) is not int or frame["frame_index"] < 0
                    or type(frame["elapsed_h"]) not in {int, float} or not np.isfinite(frame["elapsed_h"]) or frame["elapsed_h"] < 0
                    or frame["target_h"] != "last" and (type(frame["target_h"]) not in {int, float}
                        or not np.isfinite(frame["target_h"]) or frame["target_h"] < 0)):
                raise ValueError("Optical frames need literal unique identities and measured camera times")
            seen.add(frame["case_id"])
            slots.append((run + "." + frame["case_id"], frame["source"]))
        if len({frame["frame_index"] for frame in frames}) != len(frames):
            raise ValueError("An optical source frame cannot be fitted twice")
    for _, binding in slots:
        if binding is not None:
            _descriptor(binding)
    sampling = value["sampling"]
    if (not isinstance(sampling, dict) or set(sampling) != {"stride_px", "interior_radius_px", "min_samples", "min_runs"}
            or any(type(sampling[key]) is not int or sampling[key] < minimum for key, minimum in
                   (("stride_px", 1), ("interior_radius_px", 0), ("min_samples", 1), ("min_runs", 1 if len(fit) == 1 else 2)))
            or sampling["min_runs"] > len(fit)):
        raise ValueError("Original sampling controls must be explicit valid integers")
    bounds = value["flash_bounds"]
    if value["method"] == SETUP_METHOD and value["phase_configuration"] is not None:
        raise ValueError("Model-independent Setup paths cannot declare phase-model inputs")
    if (not isinstance(bounds, dict) or set(bounds) != {"min_value_aq", "max_value_aq", "min_value_g", "max_value_g"}
            or any(type(x) not in {int, float} or not np.isfinite(x) for x in bounds.values())
            or bounds["max_value_aq"] <= bounds["min_value_aq"] or bounds["max_value_g"] <= bounds["min_value_g"]):
        raise ValueError("Original explicit finite flash windows are required")
    if not isinstance(value["path_name_prefix"], str) or not value["path_name_prefix"].strip():
        raise ValueError("Optical path naming is explicit recipe DATA")
    return OpticalRecipe(_json(value))


@dataclass(frozen=True)
class FrameSamples:
    """Numerical samples tied to a recipe and declared verified-caller sources.

    This is not a file-reader or Build-qualification certificate. The caller's
    source adapter must actually verify and load FrameData before invoking it.
    """
    recipe_sha256: str
    run: str
    case_id: str
    samples: dict
    identities: dict
    audit: dict
    frame_provenance_json: str


@dataclass(frozen=True)
class PreparedFaciesFrame:
    """Corrected optical inputs only; no phase or mass eligibility is invented."""
    run_id: str
    frame_index: int
    elapsed_h: float
    current_rgb: np.ndarray
    baseline_rgb: np.ndarray
    clear_water_rgb: np.ndarray
    clear_water_supported_mask: np.ndarray
    facies: np.ndarray
    domain: np.ndarray
    supported_mask: np.ndarray
    provenance: dict

    @property
    def known_physical_domain(self):
        return self.domain


def _array_identity(value):
    """Detect changed sample bytes while retaining exact dtype/shape/order."""
    return dict(dtype=value.dtype.str, shape=list(value.shape), sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def _source_contract(value, run, selected):
    """Project the exact recipe's current native/source/Build binding declarations."""
    expected = {key: value[key] for key in ("adopted_revision", "build_context", "phase_configuration", "native_configuration")}
    expected.update({key: value["runs"][run][key] for key in ("baseline", "clearwater", "profile")})
    expected["source"] = selected["source"]
    return expected


def sample_frame_for_common_paths(frame, recipe, *, case_id):
    """Apply the original native-centre sampler to already verified FrameData."""
    import cv2
    recipe.require_inputs()
    value = recipe.document
    if frame.run_id not in value["fit_runs"]:
        raise ValueError("Held-out or excluded frames cannot enter pooled path sampling")
    frame_provenance_json = _json(frame.provenance)
    provenance = json.loads(frame_provenance_json)
    setup = value["method"] == SETUP_METHOD
    if setup and (not isinstance(frame, PreparedFaciesFrame) or provenance.get("optical_context_verified") is not True):
        raise ValueError("Setup sampling requires source-verified corrected facies inputs")
    if not setup and provenance.get("physical_context_verified") is not True:
        raise ValueError("Native facies, domain, depth and porosity must be independently source-verified")
    run = value["runs"][frame.run_id]
    selected = [row for row in run["frames"] if row["case_id"] == case_id]
    if (len(selected) != 1 or selected[0]["frame_index"] != frame.frame_index
            or _json(selected[0]["elapsed_h"]) != _json(frame.elapsed_h)):
        raise ValueError("The optical frame differs from its selected source/time identity")
    expected = _source_contract(value, frame.run_id, selected[0])
    if _json(provenance.get("optical_source_contract")) != _json(expected):
        raise ValueError("The verified-caller optical source contract differs from the recipe")
    rgb_fields = ("current_rgb", "baseline_rgb", "clear_water_rgb")
    if any(getattr(frame, key).dtype != np.float32 for key in rgb_fields):
        raise TypeError("Original pooled paths require float32 scientific RGB")
    declared = {row["facies_id"] for row in value["roles"]}
    observed_facies = frame.facies[frame.domain] if setup else frame.facies
    if not set(map(int, np.unique(observed_facies))).issubset(declared):
        raise ValueError("The actual native facies differ from declared adopted roles")
    current, baseline, support = _reference_normalized_rgb(frame)
    if not setup:
        support &= frame.mass_eligible_mask & expert_knowledge_of(frame).co2
    radius, stride = value["sampling"]["interior_radius_px"], value["sampling"]["stride_px"]
    yy, xx = np.mgrid[0:frame.domain.shape[0]:stride, 0:frame.domain.shape[1]:stride]
    samples, counts = {}, {}
    for label in sorted(row["facies_id"] for row in value["roles"] if row["analysis_role"] == "porous_medium"):
        mask = np.asarray(frame.facies == label, np.uint8)
        depth = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        selected = support[yy, xx] & (frame.facies[yy, xx] == label) & (depth[yy, xx] >= radius)
        values = (current[yy[selected], xx[selected]]-baseline[yy[selected], xx[selected]]).astype(np.float32)
        samples[label] = values
        counts[str(label)] = len(values)
    audit = dict(run=frame.run_id, frame_index=frame.frame_index, elapsed_h=frame.elapsed_h,
        sampling_stride_px=stride, facies_interior_radius_px=radius, sample_counts=counts,
        representation="corrected linear current/clearwater minus registered baseline/clearwater",
        source_binding=deepcopy(expected), frame_provenance=provenance,
        input_verification_scope=("source-verified corrected Build/photometry inputs; no phase classification"
            if setup else "already checked FrameData supplied by caller; no Build/source-reader certification here"))
    return FrameSamples(recipe.sha256, frame.run_id, case_id, samples,
                        {key: _array_identity(row) for key, row in samples.items()}, audit,
                        frame_provenance_json)


def _reference_normalized_rgb(frame):
    """Original float32 measured-clear-water normalization without clipping."""
    clear = frame.clear_water_rgb
    support = frame.known_physical_domain & frame.supported_mask & frame.clear_water_supported_mask
    support &= np.isfinite(clear).all(axis=2) & (clear > 0).all(axis=2)
    current = np.zeros(frame.current_rgb.shape, np.float32)
    baseline = np.zeros(frame.baseline_rgb.shape, np.float32)
    np.divide(frame.current_rgb, clear, out=current, where=support[..., None])
    np.divide(frame.baseline_rgb, clear, out=baseline, where=support[..., None])
    if not np.isfinite(current[support]).all() or not np.isfinite(baseline[support]).all():
        raise ValueError("Clear-water normalization produced nonfinite scientific signal")
    return current, baseline, support


def fit_pooled_color_paths(recipe, frames):
    """Run the original equal-run robust-anchor fit on exactly selected frames."""
    import darsia
    recipe.require_inputs()
    value = recipe.document
    expected = [(run, row["case_id"]) for run in value["fit_runs"] for row in value["runs"][run]["frames"]]
    seen, contributions, source_bindings = set(), {}, {}
    active = {row["facies_id"] for row in value["roles"] if row["analysis_role"] == "porous_medium"}
    for position, batch in enumerate(frames):
        if (not isinstance(batch, FrameSamples) or batch.recipe_sha256 != recipe.sha256
                or position >= len(expected) or (batch.run, batch.case_id) != expected[position]
                or (batch.run, batch.case_id) in seen
                or set(batch.samples) != active or set(batch.identities) != active):
            raise ValueError("Pooled samples differ from the exact recipe/selection/roles")
        seen.add((batch.run, batch.case_id))
        selected = next(row for row in value["runs"][batch.run]["frames"] if row["case_id"] == batch.case_id)
        if (not isinstance(batch.audit, dict) or batch.audit.get("run") != batch.run
                or batch.audit.get("frame_index") != selected["frame_index"]
                or _json(batch.audit.get("elapsed_h")) != _json(selected["elapsed_h"])
                or _json(batch.audit.get("source_binding")) != _json(_source_contract(value, batch.run, selected))
                or not isinstance(batch.frame_provenance_json, str)
                or _json(batch.audit.get("frame_provenance")) != batch.frame_provenance_json
                or batch.audit.get("sampling_stride_px") != value["sampling"]["stride_px"]
                or batch.audit.get("facies_interior_radius_px") != value["sampling"]["interior_radius_px"]):
            raise ValueError("Pooled source contract/audit changed")
        for label, samples in batch.samples.items():
            if (not isinstance(samples, np.ndarray) or samples.dtype != np.float32 or samples.ndim != 2
                    or samples.shape[1] != 3 or not np.isfinite(samples).all()
                    or _array_identity(samples) != batch.identities[label]
                    or batch.audit.get("sample_counts", {}).get(str(label)) != len(samples)):
                raise ValueError("Pooled sample bytes or identity changed")
            contributions.setdefault(label, {}).setdefault(batch.run, []).append(samples)
        source_bindings.setdefault(batch.run, []).append(deepcopy(batch.audit))
    if seen != set(expected):
        raise ValueError("The selected optical source collection is incomplete")
    pooled = {label: {run: np.concatenate(rows).astype(np.float32) for run, rows in runs.items()}
              for label, runs in contributions.items()}
    paths, audit = {}, {}
    for facies in sorted(pooled):
        runs = pooled[facies]
        anchors, counts = [], {}
        for run, raw in sorted(runs.items()):
            values = np.asarray(raw)
            counts[run] = len(values)
            if len(values) < value["sampling"]["min_samples"]:
                continue
            norms = np.linalg.norm(values, axis=1)
            if np.quantile(norms, .98) <= 1e-8:
                continue
            rows = []
            for fraction in (.25, .55, .85, .98):
                lower, upper = np.quantile(norms, (max(0., fraction-.075), min(1., fraction+.075)))
                rows.append(np.median(values[(norms >= lower) & (norms <= upper)], axis=0))
            anchors.append(np.asarray(rows, np.float32))
        if len(anchors) < value["sampling"]["min_runs"]:
            audit[str(facies)] = dict(status="unquantified", reason="insufficient independent calibration-run optical span", sample_counts=counts)
            continue
        knots = np.vstack((np.zeros(3, np.float32), np.median(np.stack(anchors), axis=0))).astype(np.float32)
        keep = np.r_[True, np.linalg.norm(np.diff(knots, axis=0), axis=1) > 1e-9]
        knots = knots[keep]
        if len(knots) < 2:
            audit[str(facies)] = dict(status="unquantified", reason="zero common color-path span", sample_counts=counts)
            continue
        path = darsia.ColorPath(colors=[row for row in knots], mode="rgb", name=value["path_name_prefix"] + str(facies))
        paths[int(facies)] = darsia.ColorPathInterpolation(color_path=path, color_mode=darsia.ColorMode.RELATIVE, values=np.asarray(path.relative_distances))
        audit[str(facies)] = dict(status="supported", run_count=len(anchors), sample_counts=counts, knots=knots.tolist(), equal_run_weight=True)
    if not paths:
        raise ValueError("No physically supported facies has a common optical response")
    return paths, audit, source_bindings


def save_common_path_package(folder, recipe, frames):
    """Emit NEW native-compatible files with the original ignored-model factory.

    This writes no old model/revision/configuration. It does not complete the
    missing source adapter or grant its own scientific/production release.
    """
    paths, audit, source_bindings = fit_pooled_color_paths(recipe, frames)
    return save_fitted_path_package(folder, recipe, paths, audit, source_bindings)


def save_fitted_path_package(folder, recipe, paths, audit, source_bindings, *, signal_axis=None):
    """Serialize an actual fit once; optional fixed units retain per-run shape.

    The legacy pooled entry point above keeps its original normalized values.
    A per-run Setup caller supplies one recorded facies scale shared by the
    complete included set, rather than stretching each run's endpoint to one.
    """
    import darsia
    paths, audit = deepcopy(paths), deepcopy(audit)
    if signal_axis is not None:
        scales = signal_axis["scales_by_facies"]
        for label, interpretation in paths.items():
            colours = np.asarray(interpretation.color_path.colors, dtype=np.float64)
            distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(colours, axis=0), axis=1))]
            scale = scales.get(str(label))
            if (type(scale) not in (float, int) or not np.isfinite(scale) or scale <= 0
                    or distances[-1] > scale * (1 + 1e-12)):
                raise ValueError("Per-run optical paths need the same positive recorded facies distance scale")
            interpretation.values = distances / scale
            audit[str(label)].update(optical_path_length=float(distances[-1]),
                shared_distance_scale=float(scale), coordinate_values=interpretation.values.tolist())
    root = Path(folder)
    if root.exists():
        raise FileExistsError("Common calibration artifacts are immutable; use a new folder")
    value = recipe.document
    bounds = value["flash_bounds"]
    ids = {row["facies_id"] for row in value["roles"]}
    ignored = tuple(sorted(ids - set(paths)))
    root.mkdir(parents=True)
    for label in sorted(ids):
        interpretation = paths.get(label)
        if interpretation is None:
            dummy = darsia.ColorPath(colors=[np.zeros(3), np.full(3, 1e-8)], mode="rgb")
            interpretation = darsia.ColorPathInterpolation(color_path=dummy, color_mode=darsia.ColorMode.RELATIVE, values=[0., 1.])
        interpretation.save(root / "color_path_interpretation" / f"color_path_interpretation_{label}")
        darsia.PWTransformation(supports=[0., 1.], values=[0., 1.]).save(root / "signal_model" / f"signal_model_{label}")
    darsia.SimpleFlash(**bounds).save(root / "flash" / "flash")
    metadata = dict(contract="darsia-pooled-native-optical-package-v1", basis="facies",
        label_ids=sorted(ids), ignore_labels=list(ignored), color_mode=darsia.ColorMode.RELATIVE.value,
        representation="corrected float32 camera-linear RAW divided by measured registered clear-water RAW; native current-minus-registered-baseline",
        shared_signal="identity native ColorPath coordinate for every supported facies", flash_bounds=bounds,
        calibration_runs=value["fit_runs"], heldout_or_excluded_runs=value["excluded_runs"], source_bindings=source_bindings,
        path_audit=audit, recipe=value, recipe_sha256=recipe.sha256,
        input_verification_scope=("source-verified corrected Build/photometry inputs; no phase classification"
            if value["method"] == SETUP_METHOD else
            "caller-supplied checked FrameData; source/Build adapter still independently required"),
        fitted_per_run_parameters=False, map_mass_rescaling=False, dilution="none")
    if value["method"] == SETUP_METHOD:
        metadata.update(mass_calibration_performed=False, phase_model_used=False,
            flash_status="initial or retained optical windows; Setup does not fit a mass calibration")
    if signal_axis is not None:
        metadata.update(signal_axis=deepcopy(signal_axis), path_scope="per-run",
            shared_signal="cumulative corrected RGB path distance / fixed shared facies scale",
            fitted_per_run_colour_paths=True)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    from .n1_pipeline import asset
    return dict(metadata=asset(root / "metadata.json"), flash=asset(root / "flash/flash.json"),
                color_paths={str(label): asset(root / "color_path_interpretation" / f"color_path_interpretation_{label}.json") for label in sorted(ids)},
                signals={str(label): asset(root / "signal_model" / f"signal_model_{label}.csv") for label in sorted(ids)})
