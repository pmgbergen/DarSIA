"""Fit an explicitly unvalidated, chemistry-specific review candidate.

This opt-in path does not relax the cross-series trainer's complete-run
validation requirement or describe training scores as holdout accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from . import trainer as t
from ..raw_photographic_review import timeline_rows


def require_single_run(sources):
    if len(sources) != 1:
        raise ValueError("This exploratory path requires exactly one experiment")


def train(request_path: Path, *, heads=("co2_presence", "gas_within_co2")):
    heads = tuple(heads)
    if not heads or len(set(heads)) != len(heads) or not set(heads) <= {"co2_presence", "gas_within_co2"}:
        raise ValueError("Select one or both distinct supported phase heads")
    started = time.perf_counter()
    request = t._read_json(request_path)
    settings = request["settings"]
    variants = t.model_variants(generic_version=settings.get("feature_version", t.GENERIC_FEATURE_VERSION))
    maximal_contract = variants["btb-ratiometric-relative-v2"]
    if settings.get("evaluation_mode") != "fit-only-single-experiment":
        raise ValueError("Training-only evaluation must be explicitly requested")
    output = Path(request["output_root"])
    if (output / "summary.json").exists():
        raise FileExistsError("Create a new model version; existing fits are immutable")
    bundles = [t.ReviewBundle(Path(b["decisions"]["snapshot_path"]),
                             Path(b["combined_manifest"]["snapshot_path"])) for b in request["bundles"]]
    from .. import phase_review_retractions as withdrawals
    from .scope import partition_review_events, requested_reviewers
    reviewers = requested_reviewers(settings)
    withdrawal_snapshot = withdrawals.model_retractions(output)
    if withdrawal_snapshot is None:
        withdrawal_snapshot = withdrawals.snapshot_for_bundles(
            [(b.decisions_path, b.combined_manifest_path) for b in bundles], settings)
    withdrawal_snapshot = partition_review_events(withdrawal_snapshot, reviewers)
    withdrawals.require_fresh_fit(withdrawal_snapshot, settings)
    withdrawal_record = withdrawals.write_model_snapshot(output, withdrawal_snapshot)
    sources, regions, conflicts = t._load_sources_and_regions(bundles, withdrawal_snapshot=withdrawal_snapshot,
                                                              reviewers=reviewers)
    from .scope import requested_series, validate_training_series
    selected_series = requested_series(settings)
    validate_training_series(sources, regions, selected_series)
    require_single_run(sources)
    for source in sources:
        frames = timeline_rows(t._read_json(source.asset_manifest_path))
        for region in regions:
            if not np.isclose(region.elapsed_h, frames[region.frame_index]["elapsed_h"], atol=1e-5, rtol=0):
                raise ValueError("Annotation timing differs from its registered source image")
    cache = Path(request["cache_root"])
    from .radiometry import resolve_profile
    radiometry_profile = resolve_profile(settings.get("radiometry_profile", "legacy"))
    radiometry_runtime = t._radiometry_runtime(radiometry_profile)
    staging = cache / "result_staging"
    staging.mkdir(parents=True, exist_ok=False)
    t._status(f"Extracting {len(regions)} single-experiment regions; no holdout claim")
    features, owners, ordered, extraction = t._load_feature_caches(
        sources, regions, cache_root=Path(settings.get("feature_cache_root") or cache), stage_workers=settings["stage_workers"],
        extraction_workers=1, opencv_threads=settings["opencv_threads"],
        pixels_per_region=settings["pixels_per_region"], texture_sigma=2.0,
        radiometry_profile=radiometry_profile, feature_contract=maximal_contract)
    if radiometry_profile is not None:
        t._write_json(staging / "radiometry_profile.json", radiometry_profile)
    inventory = {
        "regions": len(ordered), "pixel_rows": len(features),
        "regions_by_series_run": dict(Counter(f"{r.dataset}/{r.run}" for r in ordered)),
        "runs": [sources[0].run], "datasets": [sources[0].dataset],
        "series_scope": selected_series,
        "phase_classes": dict(Counter(r.phase_class for r in ordered)),
        "reviewers": dict(Counter(r.reviewer for r in ordered)),
        **t._reviewer_inventory(ordered, reviewers),
        "deduplication_conflicts": conflicts,
    }
    from .gas_negative_sampling import sampling_inventory
    inventory.update(sampling_inventory(ordered))
    t._write_region_inventory(staging / "region_inventory.csv", ordered, reviewer_ids=reviewers is not None)
    t._write_json(staging / "training_inventory.json", inventory)
    np.savez_compressed(staging / "training_samples.npz", features=features,
                        region_indices=owners, feature_names=np.asarray(maximal_contract.feature_names))
    contract = variants["generic"]
    matrix = t._variant_matrix(features, contract, maximal_contract)
    runs = {sources[0].run}
    reports, policy = {}, {}
    for head in heads:
        t._status(f"Fitting {sources[0].dataset}/{head} on {settings['backend']}")
        fit, diagnostics = t.fit_weighted_logistic_mil(
            matrix, owners, ordered, head=head, runs=runs,
            regularization=float(settings["regularization"]), backend=settings["backend"],
            weak_strength=0.35, positive_top_fraction=0.10, max_iter=settings["mil_max_iters"])
        full, labels, _, _ = t.balanced_pixel_view(matrix, owners, ordered, head=head, runs=runs)
        parity = (t._cpu_reference_check(fit, full, limit=120_000)
                  if settings["backend"] == "cpu" else
                  t._inference_parity_check(fit, full, limit=120_000))
        if not parity.get("passed"):
            raise RuntimeError(f"Numerical validation failed for {head}: {parity}")
        scores = t.region_probability_scores(t.predict_logistic(fit, matrix, backend=settings["backend"]), owners, len(ordered))
        values, targets, weights, _ = t.regional_evaluation_view(scores, ordered, head=head, runs=runs)
        metrics = t.regional_metrics(values, targets, weights, threshold=0.5)
        report = {
            "head": head, "eligible_runs": sorted(runs), "fit_mode": "mil",
            "evaluation_mode": "training-only", "generalization_validated": False,
            "deployment_allowed": False, "selected_regularization": settings["regularization"],
            "selection_policy": "fixed regularization; no selection on training accuracy",
            "training_metrics": metrics, "numerical_validation": parity,
            "cpu64_cuda32_parity": t._cuda_parity_record(parity),
            "final_fit": {"backend": fit.backend, "precision": fit.precision,
                "iterations": fit.iterations, "converged": fit.converged,
                "objective": fit.objective, "pixel_rows": len(full), "diagnostics": diagnostics},
        }
        provenance = {
            "campaign_id": request["model_id"], "target_kind": f"visible_{head}_phase_probability",
            "fit_mode": "mil", "deployment_allowed": False, "review_candidate_only": True,
            "chemistry": settings["chemistry"], "datasets": inventory["datasets"],
            "series_scope": selected_series,
            "runs": sorted(runs), "head_training_runs": sorted(runs),
            "training_inventory": inventory, "cross_validation": report,
            "backend_contract": {"production_backend": fit.backend,
                "production_precision": fit.precision, "authoritative_reference": "cpu-float64",
                "validated_backends": ["cpu"] if settings["backend"] == "cpu" else ["cpu", "cuda"],
                "parity": t._json_safe(parity)},
            "reviewers": sorted(inventory["reviewers"]), **t._reviewer_provenance(ordered, reviewers),
            "radiometry_profile": radiometry_profile,
            "radiometry_runtime": radiometry_runtime,
            "source_sha256": {"trainer": t._sha256(Path(__file__)),
                              "training_primitives": t._sha256(t.PACKAGE_ROOT / "logistic.py")},
        }
        model = t._fit_model_object(fit, feature_contract=contract, provenance=t._json_safe(provenance), texture_sigma=2.0)
        path = model.save(staging / "models/generic" / f"{head}_signal.json")
        report.update(model_path=str(path), model_sha256=model.to_dict()["model_sha256"])
        reports[head] = report
        policy[head] = {"presence_threshold": 0.5, "threshold_basis": "fixed; not validated",
                        "model_sha256": report["model_sha256"]}
        t._status(f"Finished {head}: training balanced accuracy {metrics['balanced_accuracy']:.3f}; independent validation required")
    summary = {
        "schema": 1, "model_id": request["model_id"], "campaign_id": request["model_id"],
        "artifact_kind": "single-run-training-candidate", "created_utc": request["created_utc"],
        "fit_heads": list(heads),
        "evaluation": {"kind": "training-only", "generalization_validated": False,
                       "independent_experiment_required": True},
        "radiometry_profile": radiometry_profile, "radiometry_runtime": radiometry_runtime,
        "inventory": inventory, "runs": sorted(runs), "feature_extraction": extraction,
        "models": {"generic": {"feature_contract": contract.to_dict(), "heads": reports}},
        "source_snapshot": request["bundles"], "seconds": time.perf_counter()-started,
        "review_retractions": withdrawal_record,
    }
    t._write_json(staging / "summary.json", t._json_safe(summary))
    t._write_json(staging / "phase_decision_policy.json", {"schema": 1, "models": {"generic": policy}})
    t._sync_result(staging, output)
    for path in staging.rglob("*"):
        if path.is_file() and t._sha256(path) != t._sha256(output / path.relative_to(staging)):
            raise OSError(f"Published file hash mismatch: {path}")
    t._status(f"Published single-run review candidate: {output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--heads", nargs="+", choices=("co2_presence", "gas_within_co2"),
                        default=("co2_presence", "gas_within_co2"))
    args = parser.parse_args()
    train(args.request, heads=args.heads)


if __name__ == "__main__":
    main()
