"""Connect ordinary Studio review inputs to the existing experimental adapter.

Discovery reads the selected model's bound campaign and delivery receipts only.
It never prepares phase maps or starts phase inference, and never scans a result
archive. The small catalog also makes shared-calibration variants discoverable.
This module reads saved catalogs and registers already completed results.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from darsia.fluidflower.review_storage import resolve_review_path


CATALOG_FORMAT = "studio-mass-variant-delivery-v1"
IDENTITY_KEYS = ("dataset", "run", "candidate_id", "params_sha256", "optical_model_sha256")


def _read(path):
    return json.loads(resolve_review_path(path).read_text(encoding="utf-8-sig"))


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _identity(candidate):
    return {key: candidate.get(key) for key in IDENTITY_KEYS}


def _jsonable(value):
    return json.loads(json.dumps(value, default=str))


def _path(root, value):
    path = Path(value)
    return resolve_review_path(path if path.is_absolute() else root / path).resolve()


def _prepared_jobs(root, *, saved_comparison=False):
    from darsia.fluidflower.phase_evidence_audit import verified_review_binding
    from darsia.fluidflower.phase_saved_runtime_views import mass_candidate

    root = resolve_review_path(root).resolve()
    binding = verified_review_binding(root, saved_comparison=saved_comparison)
    combined = resolve_review_path(binding["combined_root"]).resolve()
    manifest = _read(combined / "postflight_manifest.json")
    model_id = binding.get("model_id") or binding.get("source_model_id")
    if not model_id:
        raise ValueError("The prepared phase campaign has no model identity")
    # Collections retain per-experiment physical/pipeline bindings. The ordinary
    # binding verifier has already checked their frozen source receipts.
    run_bindings = {}
    if manifest.get("format") == "fluidflower-phase-review-collection-v1":
        receipt = _read(manifest["collection_receipt"])
        for source in receipt["sources"]:
            source_binding = _read(source["binding"]["path"])
            for run in source_binding["selected_candidates"]:
                run_bindings[run.casefold()] = source_binding
    jobs = []
    for run, selected in binding["selected_candidates"].items():
        matches = [job for job in manifest["jobs"]
                   if job["run"].casefold() == run.casefold()
                   and job.get("candidate_id") == selected]
        if len(matches) != 1:
            raise ValueError(f"No unique bound phase candidate for {run}")
        source_job = matches[0]
        source_binding = run_bindings.get(run.casefold(), binding)
        expected_digest = source_binding.get("pipeline_sha256")
        if not expected_digest or source_job.get("params_sha256") != expected_digest:
            raise ValueError(f"Phase parameters differ from the prepared binding: {run}")
        if source_job.get("optical_model_sha256") != source_binding.get("model_sha256"):
            raise ValueError(f"Phase model differs from the prepared binding: {run}")
        if saved_comparison and source_job.get("physics_runtime") != source_binding.get("physics_runtime"):
            raise ValueError(f"Saved phase map has a different physical runtime: {run}")
        if not source_job.get("review_output_dir") or not source_job.get("phase_map_output_dir"):
            raise ValueError(f"Registered review photographs and phase maps are required: {run}")
        candidate = mass_candidate(combined, source_job)
        values = _jsonable(asdict(candidate))
        review_root = _path(combined, source_job["review_output_dir"])
        if not review_root.is_dir() or not Path(values["phase_map_output_dir"]).is_dir():
            raise ValueError(f"Prepared review inputs are missing: {run}")
        if _identity(values) != _identity(source_job):
            raise ValueError(f"Adapted mass candidate differs from the prepared phase candidate: {run}")
        jobs.append(dict(run=candidate.run, dataset=candidate.dataset, model_id=model_id,
            model_root=str(root), combined_root=str(combined), candidate=values,
            review_root=str(review_root)))
    datasets = {job["dataset"] for job in jobs}
    if len(datasets) != 1:
        raise ValueError("Concentration inputs require one experiment series")
    return model_id, next(iter(datasets)), jobs


def _matching_result(path, candidate):
    result = _read(path)
    if (result.get("complete") is not True or not result.get("frames")
            or _identity(result.get("selected_inputs", {}).get("candidate", {})) != _identity(candidate)):
        return None
    return result


def _ordinary_results(job):
    from darsia.fluidflower.phase_saved_runtime_views import selection, stamp

    candidate = job["candidate"]
    folder = Path(candidate["phase_map_output_dir"]) / "mass_results"
    found = []
    for path in sorted(folder.glob("*.json")):
        try:
            receipt = _read(path)
            selected = receipt["selection"]
            request = build_request(job, folder, selected["method"])
            if (receipt.get("format") != "studio-mass-delivery-v1"
                    or selected != selection(request, selected["method"])):
                continue
            # Listing may reject a changed delivery, but must never rehash its
            # photographs, rigs or numerical archives over the network.
            if any(stamp(name) != item["stamp"] for name, item in receipt.get("bindings", {}).items()):
                continue
            result_path = Path(receipt["result"]).resolve()
            result = _matching_result(result_path, candidate)
            if result is not None and all(result["method"]["settings"].get(k) == v
                                          for k, v in selected["method"].items()):
                found.append(str(result_path))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return found


def discover_model(root):
    """Return prepared jobs and only results bound to the selected phase model."""
    root = Path(root).resolve()
    model_id, dataset, jobs = _prepared_jobs(root)
    results_root = root.parent.parent / "results" / "concentration_mass"
    results = {path for job in jobs for path in _ordinary_results(job)}
    candidates = {_identity_key(job["candidate"]): job["candidate"] for job in jobs}
    for path in sorted((results_root / "catalog").glob("*.json")):
        try:
            receipt = _read(path)
            candidate = candidates.get(_identity_key(receipt["candidate"]))
            if receipt.get("format") != CATALOG_FORMAT or candidate is None:
                continue
            result_path = Path(receipt["result"]).resolve()
            # Only the result JSON is hashed. Dense arrays and frame archives
            # remain untouched during ordinary Studio discovery.
            if _digest(result_path) != receipt["result_sha256"]:
                continue
            result = _matching_result(result_path, candidate)
            if result is None:
                continue
            source_bindings = result.get("source_bindings", {})
            if source_bindings != receipt.get("source_bindings", {}):
                continue
            results.add(str(result_path))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return dict(model_id=model_id, dataset=dataset, jobs=jobs,
                results=sorted(results), results_root=str(results_root))


def _identity_key(candidate):
    return tuple(candidate.get(key) for key in IDENTITY_KEYS)


def build_request(job, results_root, method=None):
    """Use exactly the review photographs and categorical maps already selected."""
    if method is None:
        method = dict(optical="reference_soft_log", restoration="darsia_tvd", chemistry="empirical_power")
    request = dict(candidate=_jsonable(job["candidate"]), review_root=str(job["review_root"]),
        frame_index=0, reference_mode=False, local_baseline_search=False,
        analysis_width=1024, results_root=str(results_root), method=dict(method))
    for key in ("model_root", "model_id", "combined_root"):
        if key in job:
            request[key] = str(job[key])
    return request


def register_variant(result_path, results_root):
    """Publish a small immutable discovery receipt for one completed result."""
    result_path = Path(result_path).resolve()
    before = _digest(result_path)
    result = _read(result_path)
    candidate = result.get("selected_inputs", {}).get("candidate", {})
    if (result.get("complete") is not True or not result.get("frames")
            or any(not candidate.get(key) for key in IDENTITY_KEYS[:4])):
        raise ValueError("A complete result with a bound phase candidate is required")
    if _digest(result_path) != before:
        raise ValueError("Concentration result changed during registration")
    key = hashlib.sha256(str(result_path).encode()).hexdigest()
    target = Path(results_root) / "catalog" / (key + ".json")
    receipt = dict(format=CATALOG_FORMAT, result=str(result_path), result_sha256=before,
        candidate=_identity(candidate), source_bindings=result.get("source_bindings", {}),
        created_utc=result.get("created_utc", ""))
    if target.is_file():
        if _read(target) != receipt:
            raise ValueError("Registered concentration result changed; use a new result path")
        _index_saved_calibration(result_path, results_root)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves previous publications, including races.
    try:
        with target.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
    except FileExistsError:
        if _read(target) != receipt:
            raise ValueError("Registered concentration result changed; use a new result path")
    _index_saved_calibration(result_path, results_root)
    return target


def _index_saved_calibration(result_path, results_root):
    """Publish a library reference after saving; never alter numerical results."""
    import warnings
    from darsia.fluidflower.mass_calibration_library import index_saved_result

    try:
        index_saved_result(result_path, results_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        warnings.warn("Completed concentration result is preserved, but its mass-library reference could not be indexed: " + str(error), RuntimeWarning)
