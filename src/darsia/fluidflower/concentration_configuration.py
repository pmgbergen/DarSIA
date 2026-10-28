"""Explicit, series-owned bindings for provisional native concentration analysis.

Saved artifacts are never interpreted as scientifically validated simply because
they exist. The profile declares their basis and geometry; loading verifies that
declaration against the selected Studio experiment and immutable artifact hashes.
"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tomllib
import zipfile

import numpy as np

from .series_paths import series_key, series_root


SCIENTIFIC_PROFILE_FIELDS = (
    "basis", "shape_rc", "dimensions_m", "origin_m", "baseline_source_name",
    "calibration_label_ids", "nonporous_label_ids", "ignored_label_ids",
    "calibration_folder", "artifact_record", "injection_protocol",
    "pressure_temperature_protocol", "allow_legacy_basis",
    "pressure_temperature_assumption",
)


def _identity(job):
    candidate = job.get("candidate", {})
    dataset = series_key(candidate.get("dataset") or job.get("dataset") or job.get("series_id"))
    run = str(candidate.get("run") or job.get("run") or "").lower()
    if not run:
        raise ValueError("Concentration request has no experiment identity")
    return dataset, run


def _path(value, root):
    from .storage_paths import resolve_storage_path
    root = Path(root)
    path = Path(value)
    return resolve_storage_path(path if path.is_absolute() else root / path, series_roots=(root,))


def _profile(job):
    dataset, run = _identity(job)
    root = series_root(dataset)
    path = _path(job.get("concentration_profile_path", "concentration.toml"), root)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Concentration profile overrides must belong to the selected series.")
    if not path.is_file():
        return dataset, run, root, path, {}, {}
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    if document.get("schema_version") != 1 or series_key(document["series_id"]) != dataset:
        raise ValueError(f"Wrong concentration profile identity: {path}")
    return dataset, run, root, path, document, document.get("runs", {}).get(run, {})


def describe_run(job, *, purpose="native"):
    """Return cheap JSON-safe readiness; no photograph or numerical array is read."""
    if purpose not in {"native", "joint_reference"}:
        raise ValueError("Unknown concentration configuration purpose.")
    dataset, run, root, path, document, profile = _profile(job)
    reasons = list(profile.get("reasons", []))
    if not profile:
        reasons.append("No explicit concentration calibration binding for this experiment.")
    elif not profile.get("enabled", False) and not reasons:
        reasons.append("The imported calibration has not passed its geometry/basis binding check.")
    if profile.get("enabled"):
        required = ["artifact_record", "injection_protocol", "pressure_temperature_protocol"]
        if purpose == "native":
            required.append("calibration_folder")
        for key in required:
            if key not in profile or not _path(profile[key], root).exists():
                reasons.append(f"Missing registered concentration input: {key}.")
    return dict(dataset=dataset, run=run, ready=bool(profile.get("enabled")) and not reasons,
                reasons=reasons, profile_path=str(path), provisional=True,
                status=profile.get("status", "not_configured"),
                chemistry_group=profile.get("chemistry_group", dataset),
                shared_fit_eligible=bool(profile.get("shared_fit_eligible", False)),
                basis=profile.get("basis"), calibration_folder=profile.get("calibration_folder"),
                warnings=list(profile.get("warnings", [])),
                artifact_record=profile.get("artifact_record"), purpose=purpose)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _array(path):
    # Native DarSIA NPZ metadata contains trusted NumPy/Python scalar metadata.
    # Only these explicitly registered NAS artifacts are opened with pickle.
    with np.load(path, allow_pickle=True) as archive:
        array = archive["array"]
        metadata = archive["metadata"].item()
    return array, metadata


def _array_geometry(path):
    """Read registered NPZ geometry without decoding an unused RGB baseline."""
    with zipfile.ZipFile(path) as archive:
        with archive.open("array.npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"Unsupported registered baseline NPY header: {version}")
        with archive.open("metadata.npy") as stream:
            metadata = np.lib.format.read_array(stream, allow_pickle=True).item()
    return shape, metadata


def _check_geometry(metadata, shape, profile, *, artifact):
    if tuple(shape[:2]) != tuple(profile["shape_rc"]):
        raise ValueError(f"{artifact}: registered concentration grid shape differs from Studio.")
    for key, expected in (("dimensions", profile["dimensions_m"]), ("origin", profile["origin_m"])):
        if key not in metadata or not np.allclose(metadata[key], expected, rtol=0, atol=1e-8):
            raise ValueError(f"{artifact}: {key} differs from the registered concentration grid.")
    if metadata.get("indexing") != "ij":
        raise ValueError(f"{artifact}: concentration requires explicitly registered ij indexing.")


def load_run_configuration(job, *, purpose="native"):
    """Load verified native calibration/geometry and registered atmospheric series.

    Returns baseline_rgb, labels, domain, depth_m, porosity, dimensions_m
    ({height,width}), calibration_folder, basis, allow_legacy_basis,
    pressure_temperature, and JSON-safe provenance. No resizing or label
    remapping is performed. Call pressure_temperature_at(config,t) per frame.
    """
    description = describe_run(job, purpose=purpose)
    if not description["ready"]:
        raise ValueError("Concentration analysis unavailable: " + " ".join(description["reasons"]))
    dataset, run, root, path, document, profile = _profile(job)
    from .concentration_inputs import inspect_run, load_baseline
    inspected = inspect_run(job)
    if inspected["dataset"] != dataset or inspected["run"] != run:
        raise ValueError("Selected phase maps belong to another concentration experiment.")
    _check_geometry(inspected["geometry_metadata"], inspected["shape_rc"], profile, artifact="Studio phase maps")
    reference = inspected.get("baseline_reference", {})
    source_name = reference.get("source_name") or Path(reference.get("source_path", "")).name
    if str(source_name).casefold() != profile["baseline_source_name"].casefold():
        raise ValueError("Studio baseline source differs from the registered calibration baseline.")
    record_path = _path(profile["artifact_record"], root)
    record = json.loads(record_path.read_text(encoding="utf-8-sig"))
    if record.get("dataset") != dataset or record.get("run") != run:
        raise ValueError("Concentration artifact receipt belongs to another experiment.")
    if record.get("basis") != profile["basis"]:
        raise ValueError("Calibration basis differs from its audited artifact receipt.")
    if purpose == "native" and profile.get("allow_legacy_basis") and not record.get("basis_evidence", {}).get("calibration_basis_verified"):
        raise ValueError("Legacy concentration calibration requires explicit verified calibration basis.")
    bound = {}
    for item in record["artifacts"]:
        artifact = _path(item["path"], root)
        if not artifact.is_file() or _sha(artifact) != item["sha256"]:
            raise ValueError(f"Registered concentration artifact changed: {artifact}")
        if item.get("role"):
            if item["role"] in bound:
                raise ValueError("Duplicate concentration artifact role")
            bound[item["role"]] = artifact
    for key in ("baseline", "labels", "depth", "porosity", "injection", "pressure_temperature"):
        if key not in bound:
            raise ValueError(f"Concentration artifact receipt has no {key}")
    if bound["injection"] != _path(profile["injection_protocol"], root) or bound["pressure_temperature"] != _path(profile["pressure_temperature_protocol"], root):
        raise ValueError("Concentration protocol paths differ from their artifact receipt.")
    baseline_shape, baseline_metadata = _array_geometry(bound["baseline"])
    _check_geometry(baseline_metadata, baseline_shape, profile, artifact="baseline")
    if tuple(baseline_shape) != (*profile["shape_rc"], 3):
        raise ValueError("Registered native baseline must be RGB on the concentration grid.")
    arrays = {}
    for key in ("labels", "depth", "porosity"):
        values, metadata = _array(bound[key])
        _check_geometry(metadata, values.shape, profile, artifact=key)
        arrays[key] = values
    labels = arrays["labels"]
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("Native calibration labels must be nonnegative integers.")
    observed = set(map(int, np.unique(labels)))
    expected = set(map(int, profile["calibration_label_ids"]))
    if observed != expected:
        raise ValueError(f"Native calibration label IDs do not match {profile['basis']}: {sorted(observed)} versus {sorted(expected)}")
    folder = _path(profile["calibration_folder"], root) if profile.get("calibration_folder") else None
    if purpose == "native":
        calibration_metadata = json.loads((folder/"metadata.json").read_text(encoding="utf-8-sig"))
        if sorted(map(int, calibration_metadata.get("ignore_labels", []))) != sorted(map(int, profile.get("ignored_label_ids", []))):
            raise ValueError("Profile ignored labels differ from saved native calibration metadata.")
        interpretations = {int(p.stem.rsplit("_", 1)[1]) for p in (folder/"color_path_interpretation").glob("color_path_interpretation_*.json")}
        signals = {int(p.stem.rsplit("_", 1)[1]) for p in (folder/"signal_model").glob("signal_model_*.csv")}
        if interpretations != expected or signals != expected:
            raise ValueError("Saved native interpretations/signals do not cover the exact registered label basis.")
        # The receipt must cover all calibration files, including newly added files.
        saved = {str(_path(x["path"], root).resolve()) for x in record["artifacts"] if x.get("kind") == "calibration"}
        current = {str(p.resolve()) for p in folder.rglob("*") if p.is_file() and p.suffix in {".json", ".csv"}}
        if current != saved:
            raise ValueError("Native calibration file inventory changed after registration.")
    domain = ~np.isin(labels, profile["nonporous_label_ids"])
    # Use the same corrected baseline contract as the selected phase photographs.
    # The native rig baseline remains hash-registered geometry/source evidence;
    # imported response curves are explicitly provisional under this radiometry.
    baseline, baseline_provenance = load_baseline(inspected)
    baseline = np.asarray(baseline, np.float32)
    depth = np.asarray(arrays["depth"], np.float32)
    porosity = np.asarray(arrays["porosity"], np.float32)
    if baseline.shape != (*labels.shape, 3) or not np.isfinite(baseline).all():
        raise ValueError("Registered baseline is not finite RGB on the native geometry.")
    if not domain.any() or not np.isfinite(depth[domain]).all() or np.any(depth[domain] <= 0):
        raise ValueError("Measured depth must be finite and positive inside the analysis domain.")
    if not np.isfinite(porosity[domain]).all() or np.any((porosity[domain] <= 0) | (porosity[domain] > 1)):
        raise ValueError("Porosity must lie in (0,1] inside the analysis domain.")
    origin = min(datetime.fromisoformat(row["start"]) for row in _read_rows(bound["injection"]))
    assumption = _pressure_temperature_assumption(profile, purpose)
    atmosphere = []
    for row in _read_rows(bound["pressure_temperature"]):
        if "PLACEHOLDER" in str(row).upper() and assumption.get("mode") != "constant":
            raise ValueError("Pressure/temperature protocol is a placeholder; supply observations before mass conversion.")
        if any(row.get(field) in (None, "") for field in ("pressure_gradient_bar", "temperature_gradient_celsius")):
            raise ValueError("Pressure/temperature gradients must be explicitly registered, including zero gradients.")
        atmosphere.append(dict(time_hours=(datetime.fromisoformat(row["datetime"])-origin).total_seconds()/3600,
            pressure_bar=float(row["pressure_bar"]), temperature_c=float(row["temperature_celsius"]),
            pressure_gradient_bar_per_m=float(row["pressure_gradient_bar"]),
            temperature_gradient_c_per_m=float(row["temperature_gradient_celsius"])))
    times = np.array([x["time_hours"] for x in atmosphere])
    if not len(times) or not np.isfinite([list(x.values()) for x in atmosphere]).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Registered pressure/temperature records must be finite and chronological.")
    if any(x["pressure_bar"] <= 0 for x in atmosphere):
        raise ValueError("Registered atmospheric pressure must be positive.")
    return dict(baseline_rgb=baseline, labels=labels, domain=domain, depth_m=np.where(domain, depth, 0),
        porosity=np.where(domain, porosity, 0), dimensions_m=dict(height=profile["dimensions_m"][0],width=profile["dimensions_m"][1]),
        calibration_folder=str(folder) if folder else None, basis=profile["basis"], allow_legacy_basis=bool(profile.get("allow_legacy_basis",False)),
        pressure_temperature=atmosphere, pressure_temperature_assumption=assumption, provisional=True,
        provenance={**description, "profile_sha256":_sha(path),
            "run_profile_sha256":hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest(),
            "scientific_profile_sha256":hashlib.sha256(json.dumps(
                {key:profile[key] for key in SCIENTIFIC_PROFILE_FIELDS if key in profile}, sort_keys=True).encode()).hexdigest(),
            "domain_policy":{"nonporous_label_ids":profile["nonporous_label_ids"],"ignored_label_ids":profile.get("ignored_label_ids",[])},
            "artifact_record_sha256":_sha(record_path),
            "pressure_temperature_assumption":assumption,
            "native_manifest_sha256":inspected["native_manifest_sha256"],"basis_evidence":record.get("basis_evidence"),
            "selected_phase_baseline":baseline_provenance,
            "coordinate_binding":"exact shape, dimensions, origin, ij indexing and baseline source name; no resampling", "artifacts":record["artifacts"]})


_PT_FIELDS = ("pressure_bar", "temperature_c", "pressure_gradient_bar_per_m", "temperature_gradient_c_per_m")


def _pressure_temperature_assumption(profile, purpose):
    """Require an explicit, bounded assumption; source measurements stay intact."""
    policy = dict(profile.get("pressure_temperature_assumption") or {})
    if not policy:
        return {}
    if purpose != "joint_reference":
        raise ValueError("Physical assumptions require the explicit joint_reference purpose.")
    if policy.get("mode") not in {"constant", "carry_forward"} or not str(policy.get("reason", "")).strip():
        raise ValueError("Pressure/temperature assumption needs a supported mode and reason.")
    coverage = [policy.get("coverage_start_h"), policy.get("coverage_end_h")]
    if any(not isinstance(x, (float, int)) for x in coverage) or not np.isfinite(coverage).all() or coverage[0] >= coverage[1]:
        raise ValueError("Pressure/temperature assumption needs finite increasing coverage.")
    if policy["mode"] == "constant":
        state = policy.get("state", {})
        if set(state) != set(_PT_FIELDS) or not np.isfinite(list(state.values())).all() or state["pressure_bar"] <= 0:
            raise ValueError("Constant pressure/temperature assumption needs one complete finite state.")
    elif "state" in policy:
        raise ValueError("Carry-forward must use the last observed state, not supplied values.")
    return policy


def pressure_temperature_provenance_at(configuration, time_hours):
    """Describe whether a frame uses observed interpolation or an explicit assumption."""
    rows = configuration["pressure_temperature"]
    times = np.asarray([r["time_hours"] for r in rows],float)
    value = float(time_hours)
    policy = configuration.get("pressure_temperature_assumption") or {}
    if not np.isfinite(value):
        raise ValueError("Selected frame lies outside observed pressure/temperature coverage.")
    if policy.get("mode") == "constant":
        if not policy["coverage_start_h"] <= value <= policy["coverage_end_h"]:
            raise ValueError("Selected frame lies outside declared pressure/temperature assumption coverage.")
        return dict(assumed=True, mode="constant", reason=policy["reason"], policy=policy)
    if times[0] <= value <= times[-1]:
        return dict(assumed=False, mode="observed_interpolation", observed_coverage_h=[float(times[0]), float(times[-1])])
    if (policy.get("mode") == "carry_forward" and value > times[-1]
            and policy["coverage_start_h"] <= value <= policy["coverage_end_h"]):
        return dict(assumed=True, mode="carry_forward", reason=policy["reason"],
                    last_observation_h=float(times[-1]), policy=policy)
    raise ValueError("Selected frame lies outside observed pressure/temperature coverage.")


def pressure_temperature_at(configuration, time_hours):
    """Interpolate observations, or apply an explicitly recorded bounded assumption."""
    provenance = pressure_temperature_provenance_at(configuration, time_hours)
    if provenance["mode"] == "constant":
        return {key:float(value) for key,value in provenance["policy"]["state"].items()}
    rows = configuration["pressure_temperature"]
    if provenance["mode"] == "carry_forward":
        return {key:float(value) for key,value in rows[-1].items() if key != "time_hours"}
    times = np.asarray([r["time_hours"] for r in rows],float)
    value = float(time_hours)
    if not np.isfinite(value) or value < times[0] or value > times[-1]:
        raise ValueError("Selected frame lies outside observed pressure/temperature coverage.")
    return {key:float(np.interp(value,times,[r[key] for r in rows])) for key in rows[0] if key != "time_hours"}
