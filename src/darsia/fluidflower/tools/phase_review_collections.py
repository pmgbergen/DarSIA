"""Ordinary validation collections backed by existing, individually bound maps.

Collections do not change a model or run inference. Each experiment retains the
evaluation and physical context that produced its maps. New normal sessions use
their own reviewer/store; old transfer sessions retain their original policy.
A model bundle that carries no base validation of its own is collected from its
completed evaluations alone.
"""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


from darsia.fluidflower.phase_runtime import load_physics_profile, physics_runtime_identity
from darsia.fluidflower.phase_validation import _model_identity, _atomic_json
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.tools._paths import canonical_metadata_paths

FORMAT = "fluidflower-phase-review-collection-v1"


class InvalidCollectionError(ValueError):
    """A registered collection no longer matches its source or prepared maps."""


def collection_root(dataset=None):
    explicit = os.environ.get("FLUIDFLOWER_PHASE_COLLECTION_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if dataset is not None:
        return series_path(dataset, "review_collections")
    # Read-only fallback for collections registered before per-series storage.
    return local_cache_root() / "phase_review_collections"


def _model_series(model_root):
    summary = _read(Path(model_root) / "summary.json")
    datasets = summary.get("inventory", {}).get("datasets", [])
    from darsia.fluidflower.series_paths import series_key
    keys = {series_key(value) for value in datasets}
    if len(keys) != 1:
        raise InvalidCollectionError("A review collection must belong to one experimental series")
    return keys.pop()


def _model_collection_root(model_root):
    return collection_root() if os.environ.get("FLUIDFLOWER_PHASE_COLLECTION_ROOT") else collection_root(_model_series(model_root))


def _read(path):
    return json.loads(resolve_review_path(path).read_text(encoding="utf-8-sig"))


def _file(path):
    path = resolve_review_path(path).resolve()
    return dict(path=str(path), sha256=sha256_file(path))


def _check(record):
    path = resolve_review_path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise InvalidCollectionError(f"Review collection source changed: {path}")
    return path


def _identity(model):
    model = resolve_review_path(model)
    model_id, digest = _model_identity(Path(model))
    return dict(source_model_id=model_id, source_model_sha256=digest,
        source_physics_runtime=physics_runtime_identity(load_physics_profile(Path(model))))


def _verify_generated(evaluation_root):
    root = resolve_review_path(Path(evaluation_root) / "validation").resolve()
    materialization = _read(root / "transfer_materialization.json")
    for name, expected in materialization["generated_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != expected:
            raise InvalidCollectionError(f"Prepared result changed: {path}")


def _bound_jobs(manifest_path, binding):
    from darsia.fluidflower.tools.phase_reference_readiness import require_reference_readiness
    require_reference_readiness(Path(manifest_path).parent, allow_pending=True)
    manifest = _read(manifest_path)
    jobs = []
    for run, candidate in binding["selected_candidates"].items():
        matches = [j for j in manifest["jobs"] if str(j["run"]).lower() == run
                   and j.get("candidate_id") == candidate]
        if len(matches) != 1:
            raise InvalidCollectionError(f"No unique prepared candidate for {run}")
        job = matches[0]
        for key in ("candidate_folder", "review_output_dir", "phase_map_output_dir"):
            if not Path(job[key]).is_absolute():
                raise InvalidCollectionError(f"Collection needs an absolute existing {key}: {run}")
        if (job.get("optical_model_sha256"), job.get("physics_runtime")) != (
                binding["model_sha256"], binding["physics_runtime"]):
            raise InvalidCollectionError(f"Candidate differs from its evaluation: {run}")
        jobs.append(job)
    if (manifest.get("model_id"), manifest.get("model_sha256"), manifest.get("physics_runtime")) != (
            binding["model_id"], binding["model_sha256"], binding["physics_runtime"]):
        raise InvalidCollectionError("Source manifest differs from its evaluation binding")
    return jobs


def build_collection(model_root, evaluation_roots, output_root=None, *, base_root=None):
    """Publish only collection metadata, using completed full-model results."""
    from darsia.fluidflower.phase_saved_promotions import _selection, _validate_science
    from darsia.fluidflower import phase_saved_transfer_review as transfer
    model = resolve_review_path(model_root).resolve()
    identity = _identity(model)
    # The bundle holds a base validation only when phase maps were prepared right after
    # training; without one the collection is made of completed evaluations alone.
    explicit_base = base_root is not None
    base_root = resolve_review_path(base_root if explicit_base else model / "validation").resolve()
    base_binding = None
    sources: list[dict] = []
    runs: set[str] = set()
    if (base_root / "validation_binding.json").is_file():
        base_binding = _read(base_root / "validation_binding.json")
        if (base_binding["model_id"], base_binding["model_sha256"], base_binding["physics_runtime"]) != (
                identity["source_model_id"], identity["source_model_sha256"], identity["source_physics_runtime"]):
            raise InvalidCollectionError("Base validation does not match selected model")
        base_jobs = _bound_jobs(base_root / "postflight_manifest.json", base_binding)
        sources.append(dict(kind="base", manifest=_file(base_root / "postflight_manifest.json"),
            binding=_file(base_root / "validation_binding.json"), jobs=base_jobs))
        runs = {str(job["run"]).lower() for job in base_jobs}
    for evaluation in evaluation_roots:
        record, plan, target = _selection(evaluation, model)
        if target["run"] in runs:
            raise InvalidCollectionError(f"Duplicate experiment in collection: {target['run']}")
        _validate_science(record, plan, target, model)
        root = resolve_review_path(evaluation).resolve()
        validation = resolve_review_path(root / "validation")
        _verify_generated(root)
        binding = _read(record["binding_path"])
        jobs = _bound_jobs(validation / "postflight_manifest.json", binding)
        sources.append(dict(kind="extra", evaluation_root=str(root),
            manifest=_file(validation / "postflight_manifest.json"),
            binding=_file(record["binding_path"]), plan=_file(record["plan_path"]),
            result=_file(root / "transfer_result.json"),
            materialization=_file(validation / "transfer_materialization.json"),
            native=_file(target["native_manifest"]), source_manifest=_file(target["source_manifest"]),
            target=target, jobs=jobs))
        runs.add(target["run"])
    if not any(source["kind"] == "extra" for source in sources) and not explicit_base:
        raise InvalidCollectionError("Select at least one completed additional experiment")
    # Display policy and the model file list come from the base when the bundle has one, and
    # otherwise from the first completed evaluation; both record the same immutable model.
    display_manifest = Path(sources[0]["manifest"]["path"])
    display_binding = base_binding if base_binding is not None else _read(
        sources[0]["binding"]["path"])
    receipt = dict(format=FORMAT, **identity, source_model_root=str(model), sources=sources)
    digest = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    parent = Path(output_root).resolve() if output_root else _model_collection_root(model)
    destination = parent / (identity["source_model_id"] + "-" + digest[:16])
    if destination == model or destination.is_relative_to(model):
        raise InvalidCollectionError("Collections must remain outside immutable model bundles")
    receipt_path = destination / "collection_receipt.json"
    if not receipt_path.is_file():
        _atomic_json(receipt_path, receipt)
    elif _read(receipt_path) != receipt:
        raise InvalidCollectionError("Existing collection receipt differs")
    jobs = [j for source in sources for j in source["jobs"]]
    selected = {str(j["run"]).lower(): j["candidate_id"] for j in jobs}
    common = dict(format=FORMAT, **identity, campaign_id=destination.name,
        collection_receipt=str(receipt_path), collection_receipt_sha256=sha256_file(receipt_path))
    manifest = dict(common, jobs=jobs, evidence_role="normal_model_validation",
        include_in_training=True, requires_explicit_training_selection=False,
        models=_read(display_manifest).get("models", {}),
        physics_scope="Per-experiment evaluation bindings in collection_receipt; no shared runtime asserted")
    binding = dict(common, **{key:str(destination) for key in ("ffac_root","ffum_root","combined_root")},
        selected_candidates=selected, roles=sorted({j["role"] for j in jobs}),
        run_count=len(selected), phase_display_policy=display_binding.get("phase_display_policy"))
    for name, value in (("postflight_manifest.json", manifest), ("validation_binding.json", binding)):
        path = destination / name
        if path.exists() and _read(path) != value:
            raise InvalidCollectionError(f"Existing collection metadata differs: {path}")
        if not path.exists():
            _atomic_json(path, value)
    # Full original-source and generated-map verification was performed above.
    validated_collection(destination / "postflight_manifest.json", model, full=False)
    index_path = parent / "index.json"
    from darsia.fluidflower.tools.calibration_pairwise_review import _exclusive_path_lock
    with _exclusive_path_lock(parent / "index.lock", timeout_s=30, stale_after_s=600):
        index = _read(index_path) if index_path.is_file() else dict(format=FORMAT, collections=[])
        entries = [entry for entry in index["collections"] if entry.get("source_model_id") != identity["source_model_id"]]
        entries.append(dict(identity, manifest=_file(destination / "postflight_manifest.json"),
            updated_utc=datetime.now(timezone.utc).isoformat()))
        _atomic_json(index_path, dict(format=FORMAT, collections=entries))
    return binding


def refresh_prepared_collection(model_root, prepared_root):
    """Advance discovery while preserving the registered collection's old sources.

    A registered base may point at model/validation. Once registered, that path
    must remain frozen; subsequent preparations use immutable prepared sets.
    """
    model = Path(model_root).resolve()
    previous = discover_collection(model)
    if previous is None:
        return None
    receipt = _read(Path(previous["collection_receipt"]))
    selected = _read(Path(prepared_root) / "validation_binding.json")["selected_candidates"]
    extras = [source["evaluation_root"] for source in receipt["sources"]
              if source["kind"] == "extra" and source["target"]["run"] not in selected]
    refreshed = build_collection(model, extras, output_root=Path(previous["combined_root"]).parent,
                                 base_root=prepared_root)
    _atomic_json(resolve_review_path(model / "validation") / "registered_base_frozen.json", {
        "format": "registered-review-base-preserved-v1", "latest_collection": refreshed["combined_root"]})
    return refreshed


def _saved_collection_identity(model_root, manifest):
    """Retain a recorded runtime only with an unchanged source-model profile.

    This is exclusively a saved-map viewing guard. It cannot authorize new
    inference or continuation training and never rewrites historical receipts.
    """
    identity = _identity(model_root)
    for key in ("source_model_id", "source_model_sha256"):
        if manifest.get(key) != identity[key]:
            raise InvalidCollectionError("Review collection uses another source model")
    recorded = manifest.get("source_physics_runtime")
    if recorded == identity["source_physics_runtime"]:
        return identity
    receipt = _read(_check(dict(path=manifest["collection_receipt"],
                               sha256=manifest["collection_receipt_sha256"])))
    profile = load_physics_profile(resolve_review_path(model_root))
    witnessed = False
    for source in receipt["sources"]:
        if source["kind"] == "base":
            from darsia.fluidflower.phase_evidence_audit import verified_review_binding
            binding_path = _check(source["binding"])
            binding = verified_review_binding(
                model_root, binding_path.parent, saved_comparison=True
            )
            if binding.get("physics_runtime") != recorded:
                raise InvalidCollectionError("Base maps use another recorded source runtime")
            witnessed = True
        elif source["kind"] == "extra":
            # Collections without base maps still retain the source profile's
            # full file digest in their immutable transfer plans.
            from darsia.fluidflower import phase_saved_transfer_review as transfer
            plan = _read(_check(source["plan"]))
            original = resolve_review_path(plan.get("source_model") or model_root)
            current = resolve_review_path(model_root)
            profiles = [current / name for name in transfer.PHYSICS_FILES
                        if (current / name).is_file()]
            if len(profiles) == 1 and profile is not None:
                path = profiles[0]
                expected = (plan.get("protected_sha256") or {}).get(
                    str((original / path.name).resolve()))
                if expected and sha256_file(path) == expected:
                    witnessed = True
    if not witnessed:
        raise InvalidCollectionError("Saved collection lacks an unchanged source physical profile")
    return dict(identity, source_physics_runtime=recorded)


def validated_collection(manifest_path, model_root, *, full=True, saved_comparison=False):
    """Check composite identity without substituting one run's physical runtime."""
    from darsia.fluidflower import phase_saved_transfer_review as transfer
    from darsia.fluidflower.phase_saved_promotions import _validate_science
    manifest_path = resolve_review_path(manifest_path)
    manifest = _read(manifest_path)
    binding = _read(manifest_path.parent / "validation_binding.json")
    if manifest.get("format") != FORMAT or binding.get("format") != FORMAT:
        raise InvalidCollectionError("Unsupported review collection format")
    identity = (_saved_collection_identity(model_root, manifest)
                if saved_comparison else _identity(model_root))
    for value in (manifest, binding):
        if any(value.get(key) != expected for key, expected in identity.items()):
            raise InvalidCollectionError("Review collection uses another model or physical rules")
    receipt_path = resolve_review_path(manifest["collection_receipt"])
    receipt_sha = sha256_file(receipt_path)
    if manifest.get("collection_receipt_sha256") != receipt_sha or binding.get("collection_receipt_sha256") != receipt_sha:
        raise InvalidCollectionError("Review collection receipt changed")
    receipt = _read(receipt_path)
    if any(receipt.get(key) != value for key, value in identity.items()):
        raise InvalidCollectionError("Collection receipt uses another source model")
    expected_jobs, selections = [], []
    for source in receipt["sources"]:
        original_manifest = _check(source["manifest"])
        original_binding = _read(_check(source["binding"]))
        jobs = _bound_jobs(original_manifest, original_binding)
        if jobs != source["jobs"]:
            raise InvalidCollectionError("A collection job differs from its original evaluation")
        expected_jobs.extend(jobs)
        if source["kind"] == "base":
            if (original_binding["model_id"], original_binding["model_sha256"], original_binding["physics_runtime"]) != (
                    identity["source_model_id"],identity["source_model_sha256"],identity["source_physics_runtime"]):
                raise InvalidCollectionError("Base collection maps use another model")
            continue
        plan = _read(_check(source["plan"]))
        for key in ("result", "materialization", "native", "source_manifest"):
            _check(source[key])
        if any(plan.get(key) != value for key, value in identity.items()):
            raise InvalidCollectionError("Extra maps were prepared with another source model")
        if plan.get("review_mode") != "visible" or plan.get("physics_mode") not in {"prepared-target", "registered-target"}:
            raise InvalidCollectionError("Collection needs visible full-model evaluations")
        targets = [row for row in transfer.target_sources(source["source_manifest"]["path"],plan["dataset"])
                   if row["run"] == plan["run"] and row["native_manifest_sha256"] == source["native"]["sha256"]]
        if len(targets) != 1 or canonical_metadata_paths(targets[0]) != canonical_metadata_paths(source["target"]):
            raise InvalidCollectionError("Extra experiment's native source or physical context changed")
        if (original_binding["model_id"],original_binding["model_sha256"],original_binding["physics_runtime"]) != (
                plan["evaluation_model_id"],plan["evaluation_model_sha256"],plan["physics_runtime"]):
            raise InvalidCollectionError("Extra maps differ from their original physical evaluation")
        if full:
            _validate_science(dict(binding_path=source["binding"]["path"]), plan, targets[0], model_root)
            _verify_generated(source["evaluation_root"])
        selections.append(dict(receipt_path=str(receipt_path), receipt_sha256=receipt_sha,
            source_model_id=identity["source_model_id"],source_model_sha256=identity["source_model_sha256"],
            evaluation_root=source["evaluation_root"],target=targets[0],selection_kind="normal_visible_review"))
    if manifest.get("jobs") != expected_jobs:
        raise InvalidCollectionError("Collection jobs differ from their bound sources")
    selected = {str(j["run"]).lower(): j["candidate_id"] for j in expected_jobs}
    if len(selected) != len(expected_jobs) or binding.get("selected_candidates") != selected or binding.get("run_count") != len(selected):
        raise InvalidCollectionError("Collection candidate selection changed")
    if any(resolve_review_path(binding[key]) != manifest_path.parent for key in ("ffac_root","ffum_root","combined_root")):
        raise InvalidCollectionError("Collection review roots changed")
    resolved_binding = dict(binding)
    for key in ("ffac_root", "ffum_root", "combined_root", "collection_receipt"):
        if resolved_binding.get(key):
            resolved_binding[key] = str(resolve_review_path(resolved_binding[key]))
    return dict(binding=resolved_binding, context_selections=selections)


def discover_collection(model_root, *, saved_comparison=False):
    """Find the registered ordinary collection using only bounded metadata reads."""
    try:
        identity = _identity(model_root)
        matching = []
        for root in (_model_collection_root(model_root), collection_root()):
            path = root / "index.json"
            if path.is_file():
                matching = [row for row in _read(path)["collections"]
                            if row.get("source_model_id") == identity["source_model_id"]]
                if matching:
                    break
        if not matching:
            return None
        row = matching[-1]
        compare = ("source_model_id", "source_model_sha256") if saved_comparison else identity
        if any(row.get(key) != identity[key] for key in compare):
            raise InvalidCollectionError("Registered review collection uses an older model revision")
        manifest_path = _check(row["manifest"])
        if row.get("source_physics_runtime") != _read(manifest_path).get("source_physics_runtime"):
            raise InvalidCollectionError("Registered collection runtime differs from its manifest")
        return validated_collection(manifest_path, model_root, full=False,
                                    saved_comparison=saved_comparison)["binding"]
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise InvalidCollectionError(str(error)) from error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    result = build_collection(args.model, args.evaluation, args.output_root)
    print(json.dumps(result, indent=2))
