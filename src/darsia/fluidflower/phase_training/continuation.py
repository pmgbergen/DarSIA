"""Explicit series-scoped continuation candidates using the existing CPU MIL optimizer.

The parent's feature contract, scaling, photometry and historical evidence
remain bound. Optimization starts at its saved coefficients. New versions are
review candidates and require independent validation before deployment.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..phase_features import OpticalFeatureContract
from ..phase_saved_assets import VerifiedAssetReader, asset
from . import logistic
from .model import DualPhaseSignalModel, TRANSFERABLE_MODEL_SCHEMA
from .scope import series_id

CONTRACT = "darsia-phase-continuation-v1"
HEADS = ("co2_presence", "gas_within_co2")
RUNTIME_SETTINGS = frozenset({"stage_workers", "extraction_workers", "opencv_threads", "mil_max_iters"})


def _signature(region):
    """Identify reviewed evidence by content while allowing bound assets to relocate."""
    def content_identity(value):
        if isinstance(value, Mapping):
            # Freezing a dependency may copy its unchanged bytes to a new
            # version folder. Its bound content, array key and other metadata
            # define evidence identity independently of that copied path.
            bound = isinstance(value.get("sha256"), str) and len(value["sha256"]) == 64
            return {key: content_identity(item) for key, item in value.items()
                    if not (bound and key == "path")}
        if isinstance(value, (list, tuple)):
            return [content_identity(item) for item in value]
        return value
    return json.dumps(content_identity(asdict(region)), sort_keys=True, separators=(",", ":"), allow_nan=False)


def require_retained_evidence(previous, current):
    """Refuse removal, replacement or reclassification of historical evidence."""
    old = {row.region_id: _signature(row) for row in previous}
    new = {row.region_id: _signature(row) for row in current}
    if len(old) != len(previous) or len(new) != len(current):
        raise ValueError("Continuation evidence must have unique region identities")
    if not old or any(new.get(key) != value for key, value in old.items()):
        raise ValueError("Historical evidence changed; create a fresh training version")
    if not set(new) - set(old):
        raise ValueError("Save new reviewed evidence before continuing this model")


def fit_region_ids(previous, current, settings, parents, *, reader=None):
    """Extend fit evidence only within the parent's declared training runs.

    Original evaluation-only partition rows and excluded chemistry runs stay
    excluded. Additional IDs are recorded separately from the frozen partition.
    """
    reader = reader or VerifiedAssetReader()
    excluded = {str(run).casefold() for run in
                (settings.get("v5_chemistry_fit_scope") or {}).get("excluded_runs", ())}
    training_runs = []
    for model in parents.values():
        runs = {str(run).casefold() for run in model.training_provenance.get("head_training_runs") or ()}
        if not runs or runs & excluded:
            raise ValueError("The parent lacks an unambiguous training-run scope")
        training_runs.append(runs)
    allowed_runs = set.union(*training_runs)
    previous_ids = {row.region_id for row in previous}
    partition = settings.get("v5_review_fit_partition")
    if partition:
        rows = reader.json(partition).get("rows") or []
        by_id = {row.get("region_id"): row for row in rows}
        if (len(by_id) != len(rows) or not previous_ids <= set(by_id)
                or any(row.get("role") not in {"fit", "evaluation-only"} for row in rows)):
            raise ValueError("The parent review partition does not bind its exact historical evidence")
        for region in previous:
            if str(by_id[region.region_id].get("run") or "").casefold() != region.run.casefold():
                raise ValueError("The parent review partition names another source run")
        old_fit = {key for key in previous_ids if by_id[key]["role"] == "fit"}
    else:
        old_fit = previous_ids
    result = {row.region_id for row in current if row.run.casefold() in allowed_runs
              and row.run.casefold() not in excluded
              and (row.region_id not in previous_ids or row.region_id in old_fit)}
    if not result - previous_ids:
        raise ValueError("No new reviewed evidence is within the parent's training scope; evaluation runs remain held out")
    return result


def continue_head(parent: DualPhaseSignalModel, features, owners, regions, *, head,
                  feature_contract: OpticalFeatureContract, regularization: float,
                  max_iter=160, learning_rate=0.035, tolerance=2e-5,
                  weak_strength=0.35, positive_top_fraction=0.10):
    """Optimize from the saved weights and scalers; never initialize a fresh fit."""
    if (parent.schema != TRANSFERABLE_MODEL_SCHEMA
            or parent.feature_contract != feature_contract.to_dict()
            or tuple(parent.feature_names) != feature_contract.feature_names):
        raise ValueError("Continuation requires the parent's exact optical feature contract")
    provenance = parent.training_provenance
    declared_series = provenance.get("series_scope")
    covered = ([declared_series] if declared_series else provenance.get("datasets") or ())
    covered = {series_id(value) for value in covered}
    if (head not in HEADS
            or provenance.get("target_kind") != f"visible_{head}_phase_probability"
            or len(covered) != 1):
        raise ValueError("Choose a compatible parent head with a declared series scope")
    selected_series = next(iter(covered))
    if not regions or any(series_id(row.dataset) != selected_series for row in regions):
        raise ValueError("Continuation cannot use another series' evidence")
    if (not np.isfinite(regularization) or regularization <= 0
            or not np.isfinite(weak_strength) or weak_strength < 0
            or not 0 < positive_top_fraction <= 1
            or not np.isfinite(learning_rate) or learning_rate <= 0
            or not np.isfinite(tolerance) or tolerance <= 0 or int(max_iter) < 1):
        raise ValueError("Continuation optimizer settings must be finite and positive")
    matrix, owners = np.asarray(features, np.float32), np.asarray(owners)
    if (matrix.ndim != 2 or matrix.shape[1] != len(parent.feature_names)
            or owners.shape != (len(matrix),) or owners.dtype.kind not in "iu"
            or not len(matrix) or not np.isfinite(matrix).all()
            or owners.min() < 0 or owners.max() >= len(regions)):
        raise ValueError("Continuation samples differ from their feature and evidence contracts")
    logistic.assert_safe_training_regions(regions)
    runs = {str(run).casefold() for run in provenance.get("head_training_runs") or ()}
    if not runs:
        raise ValueError("The parent lacks its declared head training runs")
    pure, labels, weights, _ = logistic.balanced_pixel_view(matrix, owners, regions, head=head, runs=runs)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Continuation needs historical pure evidence for both target classes")
    bags = logistic._weak_region_bags(owners, regions, head=head, runs=runs)
    selection = logistic._dominance_bag_selection(regions, bags, head=head)
    initial = logistic.LogisticFit(
        mean=np.asarray(parent.feature_mean, np.float64), scale=np.asarray(parent.feature_scale, np.float64),
        coefficients=np.asarray(parent.coefficients, np.float64), intercept=parent.intercept,
        backend="cpu", precision="float64", iterations=0, converged=False,
        objective=float("nan"), regularization=float(regularization))
    fit, diagnostics = logistic._fit_weighted_logistic_mil_cpu(
        matrix, pure, labels, weights, initial, bags, regularization=float(regularization),
        weak_strength=float(weak_strength), positive_top_fraction=float(positive_top_fraction),
        max_iter=int(max_iter), learning_rate=float(learning_rate), tolerance=float(tolerance),
        bag_selection=selection)
    diagnostics.update(initialization="saved-parent-coefficients-and-scalers",
        parent_model_sha256=parent.to_dict()["model_sha256"], fresh_fit_performed=False,
        normalization="unchanged-parent-scaling", optimizer_state="new-Adam-moments")
    return fit, diagnostics


def _bundles(document, reader):
    """Verify and resolve frozen decision and manifest pairs from a request."""
    from .trainer import ReviewBundle
    result = []
    for row in document["bundles"]:
        paths = []
        for role in ("decisions", "combined_manifest"):
            descriptor = dict(row[role], path=row[role]["snapshot_path"])
            paths.append(reader.path(descriptor))
        result.append(ReviewBundle(*paths))
    if not result:
        raise ValueError("The saved parent has no retained training evidence")
    return result


def inspect_parent(parent_entry: Mapping):
    """Read the indexed parent and exact retained evidence without creating files."""
    reader = VerifiedAssetReader()
    identity = parent_entry.get("library_source_identity") or {}
    expected = (identity.get("metadata_sha256") or {}).get("training_request.json")
    if not expected or identity.get("model_id") != parent_entry.get("model_id"):
        raise ValueError("Choose an indexed parent with its recorded training-request SHA")
    root = Path(parent_entry["root"])
    parent_descriptor = dict(path=str(root / "training_request.json"), sha256=expected)
    parent = reader.json(parent_descriptor)
    if parent.get("model_id") != parent_entry["model_id"]:
        raise ValueError("The parent training request belongs to another model")
    settings = dict(parent["settings"])
    selected = {series_id(value) for value in settings.get("series") or ()}
    if len(selected) != 1:
        raise ValueError("Continuation requires a single declared parent series")
    previous = _bundles(parent, reader)
    from ..native_training_photometry import require_saved_sidecars
    require_saved_sidecars([(row.decisions_path, row.combined_manifest_path) for row in previous],
                           SimpleNamespace(settings=settings))
    parents = {}
    for head in HEADS:
        relative = f"models/generic/{head}_signal.json"
        digest = (identity.get("metadata_sha256") or {}).get(relative)
        if not digest:
            raise ValueError("The indexed parent lacks its saved phase-head SHA")
        bound = dict(path=str(root / relative), sha256=digest)
        model = DualPhaseSignalModel.from_dict(reader.json(bound))
        if model.training_provenance.get("campaign_id") != parent["model_id"]:
            raise ValueError("The saved phase head belongs to another parent request")
        covered = model.training_provenance.get("series_scope")
        covered = [covered] if covered else model.training_provenance.get("datasets") or ()
        if ({series_id(value) for value in covered} != selected
                or not model.training_provenance.get("head_training_runs")
                or model.training_provenance.get("target_kind") != f"visible_{head}_phase_probability"
                or model.schema != TRANSFERABLE_MODEL_SCHEMA):
            raise ValueError("The parent head lacks a compatible series, target or training-run contract")
        parents[head] = bound
    return parent_descriptor, parent, previous, parents


def prepare_request(*, parent_entry: Mapping, output_parent: Path, cache_parent: Path,
                    bundles: Sequence[tuple[Path, Path]], actor: str,
                    runtime_settings: Mapping | None = None):
    """Freeze earlier evidence and saved corrections into a new continuation request."""
    from ..phase_workflow import prepare_phase_training_request
    parent_descriptor, parent, previous, parents = inspect_parent(parent_entry)
    settings = dict(parent["settings"])
    overrides = dict(runtime_settings or {})
    if set(overrides) - RUNTIME_SETTINGS:
        raise ValueError("Continuation keeps the parent's scientific settings unchanged")
    settings.update(overrides, backend="cpu")
    pairs = [(row.decisions_path, row.combined_manifest_path) for row in previous]
    pairs += [(Path(decisions), Path(manifest)) for decisions, manifest in bundles]
    pairs = list(dict.fromkeys(pairs))
    from . import trainer as t
    from .. import phase_review_retractions as withdrawals
    from .scope import requested_reviewers
    from ..native_training_photometry import require_saved_sidecars
    require_saved_sidecars(pairs, SimpleNamespace(settings=settings))
    policy = withdrawals.snapshot_for_bundles(pairs, settings)
    old_policy = settings.get(withdrawals.DOCUMENT_KEY) or {}
    if policy.get("targets") and policy.get("digest") != old_policy.get("digest"):
        raise ValueError("Withdrawn evidence changed; create a fresh training version")
    _, earlier, _ = t._load_sources_and_regions(previous, preserve_native_revisions=True,
                                               withdrawal_snapshot=old_policy or None)
    _, current, _ = t._load_sources_and_regions([t.ReviewBundle(*pair) for pair in pairs],
        preserve_native_revisions=True, withdrawal_snapshot=policy, reviewers=requested_reviewers(settings))
    require_retained_evidence(earlier, current)
    reader = VerifiedAssetReader()
    models = {head: DualPhaseSignalModel.from_dict(reader.json(bound)) for head, bound in parents.items()}
    fit_region_ids(earlier, current, settings, models, reader=reader)
    request = prepare_phase_training_request(output_parent=output_parent, cache_parent=cache_parent,
        bundles=pairs, actor=actor, settings=settings)
    request["continuation"] = dict(contract=CONTRACT, parent_model_id=parent["model_id"],
        parent_request=parent_descriptor, parent_heads=parents, backend="cpu",
        initialization="saved-parent-coefficients-and-scalers", deployment_allowed=False)
    path = Path(request["request_path"])
    persisted = {key: value for key, value in request.items() if key not in {"request_path", "frozen_pairs"}}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(persisted, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return request


def build_command(request):
    """Return the interpreter command for the prepared continuation producer."""
    from ..phase_workflow import phase_python_executable
    return [phase_python_executable(), "-B", "-m", __name__, "--request", request["request_path"]]


def train(request_path: Path):
    """Extract the frozen union and publish new heads as an unvalidated candidate."""
    from . import trainer as t
    reader = VerifiedAssetReader()
    request = reader.json(asset(request_path))
    continuation = request.get("continuation") or {}
    if continuation.get("contract") != CONTRACT or continuation.get("backend") != "cpu":
        raise ValueError("An explicit CPU phase-continuation request is required")
    output, settings = Path(request["output_root"]), request["settings"]
    if (output / "summary.json").exists():
        raise FileExistsError("Existing model versions are immutable")
    parent = reader.json(continuation["parent_request"])
    if parent.get("model_id") != continuation["parent_model_id"]:
        raise ValueError("Continuation parent request identity changed")
    from .. import phase_review_retractions as withdrawals
    old_policy = parent["settings"].get(withdrawals.DOCUMENT_KEY) or {}
    policy = settings.get(withdrawals.DOCUMENT_KEY) or {}
    if ((policy.get("targets") or old_policy.get("targets"))
            and policy.get("digest") != old_policy.get("digest")):
        raise ValueError("Withdrawn evidence changed; create a fresh training version")
    stable = set(parent["settings"]) | set(settings)
    stable -= RUNTIME_SETTINGS | {"backend", withdrawals.DOCUMENT_KEY}
    if any(parent["settings"].get(key) != settings.get(key) for key in stable):
        raise ValueError("Continuation changed the parent's scientific settings")
    # Continuation retains each historical annotation's own acquisition canvas.
    # A later review root must never replace its original image implicitly.
    preserve_revisions = True
    origins = {}
    parent_bundles, current_bundles = _bundles(parent, reader), _bundles(request, reader)
    from ..native_training_photometry import require_saved_sidecars
    require_saved_sidecars([(row.decisions_path, row.combined_manifest_path) for row in
                           (*parent_bundles, *current_bundles)], SimpleNamespace(settings=settings))
    old_sources, previous, _ = t._load_sources_and_regions(parent_bundles,
        preserve_native_revisions=preserve_revisions)
    sources, regions, conflicts = t._load_sources_and_regions(current_bundles,
        preserve_native_revisions=preserve_revisions, region_sources=origins)
    require_retained_evidence(previous, regions)
    all_sources = [*old_sources, *sources, *(origins.values() if origins is not None else ())]
    from .log_linear_input import SOURCE_FORMAT
    for row in all_sources:
        native = reader.json(asset(row.asset_manifest_path))
        if (not t._raw_review_manifest(native)
                and (native.get("v5_photometry") or {}).get("format") != SOURCE_FORMAT):
            raise ValueError("Continuation requires saved camera-linear inputs; display photographs cannot be training inputs")
    parents = {head: DualPhaseSignalModel.from_dict(reader.json(continuation["parent_heads"][head]))
               for head in HEADS}
    first = parents[HEADS[0]]
    contract = OpticalFeatureContract.from_dict(first.feature_contract or {})
    if any(model.feature_contract != first.feature_contract or model.texture_sigma != first.texture_sigma
           or model.training_provenance.get("campaign_id") != parent["model_id"] for model in parents.values()):
        raise ValueError("The parent heads have incompatible training identities")
    allowed_ids = fit_region_ids(previous, regions, settings, parents, reader=reader)
    allowed_runs = {row.run for row in regions if row.region_id in allowed_ids}
    excluded_runs = {row.run for row in regions} - allowed_runs
    cache = Path(request["cache_root"])
    staging = cache / "continuation_staging"
    staging.mkdir(parents=True, exist_ok=False)
    matrix, owners, ordered, extraction = t._load_feature_caches(sources, regions,
        cache_root=cache, stage_workers=int(settings.get("stage_workers", 1)),
        extraction_workers=int(settings.get("extraction_workers", 1)),
        opencv_threads=int(settings.get("opencv_threads", 2)),
        pixels_per_region=int(settings["pixels_per_region"]), texture_sigma=first.texture_sigma,
        radiometry_profile=None, feature_contract=contract, fit_allowed_review_ids=allowed_ids,
        fit_excluded_runs=sorted(excluded_runs), origin_sources=origins)
    reports = {}
    for head, model in parents.items():
        regularization = (model.training_provenance.get("cross_validation") or {}).get(
            "selected_regularization", settings.get("regularization"))
        if regularization is None:
            raise ValueError("The parent lacks its fitted head regularization")
        fit, diagnostics = continue_head(model, matrix, owners, ordered, head=head,
            feature_contract=contract, regularization=float(regularization),
            max_iter=int(settings.get("mil_max_iters", 160)))
        head_ids = {row.region_id for row in ordered if row.region_id in allowed_ids
                    and row.run in set(model.training_provenance["head_training_runs"])}
        head_owners = [index for index, row in enumerate(ordered) if row.region_id in head_ids]
        provenance = dict(model.training_provenance, campaign_id=request["model_id"],
            deployment_allowed=False, review_candidate_only=True,
            continuation=dict(continuation, parent_head=continuation["parent_heads"][head],
                parent_numeric_sha256=model.to_dict()["model_sha256"],
                request=asset(request_path), historical_evidence_retained=True),
            training_region_ids=sorted(head_ids),
            training_inventory=dict(regions=len(head_ids), pixel_rows=int(np.isin(owners, head_owners).sum()),
                runs=sorted(model.training_provenance["head_training_runs"]),
                datasets=sorted({row.dataset for row in ordered if row.region_id in head_ids})),
            cross_validation=dict(evaluation_mode="unvalidated-continuation", generalization_validated=False,
                deployment_allowed=False, selected_regularization=float(regularization), final_fit=diagnostics))
        updated = replace(model, coefficients=tuple(map(float, fit.coefficients)), intercept=fit.intercept,
                          training_provenance=t._json_safe(provenance))
        path = updated.save(staging / "models/generic" / f"{head}_signal.json")
        reports[head] = dict(head=head, model_path=str(output / path.relative_to(staging)),
            model_sha256=updated.to_dict()["model_sha256"],
            final_fit=t._json_safe(diagnostics), deployment_allowed=False,
            generalization_validated=False, evaluation_mode="continuation-candidate")
    t._write_region_inventory(staging / "region_inventory.csv", ordered, reviewer_ids=True)
    np.savez_compressed(staging / "training_samples.npz", features=matrix, region_indices=owners,
                        feature_names=np.asarray(contract.feature_names))
    summary = dict(schema=1, model_id=request["model_id"], campaign_id=request["model_id"],
        artifact_kind="phase-continuation-candidate", created_utc=request["created_utc"],
        continuation=continuation, evaluation=dict(kind="unvalidated-continuation", generalization_validated=False,
            deployment_allowed=False, independent_experiment_required=True),
        inventory=dict(regions=len(ordered), pixel_rows=len(matrix), datasets=sorted({row.dataset for row in ordered}),
            runs=sorted({row.run for row in ordered}), deduplication_conflicts=conflicts),
        models=dict(generic=dict(feature_contract=contract.to_dict(), heads=reports)),
        source_snapshot=request["bundles"], feature_extraction=extraction)
    summary["continuation"] = dict(continuation, fit_region_ids=sorted(allowed_ids),
        additional_fit_region_ids=sorted(allowed_ids - {row.region_id for row in previous}),
        native_revision_policy="per-annotation-original-canvas-v1")
    summary["runs"] = sorted({row.run for row in ordered})
    summary["nonconverged_heads"] = [head for head, report in reports.items()
                                   if not report["final_fit"].get("converged")]
    t._write_json(staging / "summary.json", t._json_safe(summary))
    t._sync_result(staging, output)
    for path in staging.rglob("*"):
        if path.is_file() and t._sha256(path) != t._sha256(output / path.relative_to(staging)):
            raise OSError("Published continuation file hash mismatch: " + str(path))
    return summary


def main():
    """Run the continuation producer for the explicitly selected request file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    train(args.request)


if __name__ == "__main__":
    main()
