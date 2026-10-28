"""Fit fresh RAW review candidates with the existing optical extraction and MIL.

The preparation remains immutable and non-runnable by the frozen trainer.
This explicit producer binds its own execution request, context views and
unvalidated models. Physical context admits inputs; optical sampling and
feature arithmetic remain those of the existing trainer.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid
from concurrent.futures import ThreadPoolExecutor

HEADS = ("co2_presence", "gas_within_co2")
OPTIONS = ("feature_version", "texture_sigma", "pixels_per_region", "stage_workers",
           "extraction_workers", "opencv_threads", "backend", "regularization_grid",
           "weak_strength", "positive_top_fraction", "mil_max_iters",
           "gas_selection_policy", "change_evidence_ablation")


def execution_limits(options, shapes, *, resources=None):
    """Cap independent run workers using current CPU affinity and RAM estimates."""
    if resources is None:
        import psutil
        try:
            affinity = len(psutil.Process().cpu_affinity())
        except (AttributeError, psutil.Error):
            affinity = psutil.cpu_count() or 1
        resources = dict(cores=min(affinity, psutil.cpu_count(logical=False) or affinity),
                         available_bytes=psutil.virtual_memory().available)
    pixels = max(int(shape[0]) * int(shape[1]) for shape in shapes)
    # Baseline/current/previous RGB, feature/texture workspaces, masks and imports.
    per_worker = 1024**3 + pixels * 512
    memory_limit = int(resources["available_bytes"] * .8) // per_worker
    if memory_limit < 1:
        raise ValueError("Available RAM is below the fresh RAW extraction workspace estimate")
    threads = max(1, int(options["opencv_threads"]))
    cpu_limit = max(1, int(resources["cores"]) // threads)
    limit = max(1, min(memory_limit, cpu_limit, len(shapes)))
    return dict(stage_workers=min(max(1, int(options["stage_workers"])), limit),
                extraction_workers=min(max(1, int(options["extraction_workers"])), limit),
                opencv_threads=threads, blas_threads=1,
                per_worker_estimated_bytes=per_worker, available_bytes=resources["available_bytes"],
                memory_worker_limit=memory_limit, cpu_worker_limit=cpu_limit)


def load_features_with_retries(trainer, *args, extraction_workers, attempts=3, used=None, **kwargs):
    """Extract features again with half the workers when a worker process stops abruptly.

    Finished runs stay in the feature cache, whose bytes are SHA-checked when
    reused, so a new attempt extracts only the unfinished runs and the features
    are those of one uninterrupted extraction. ``used`` collects each attempt's
    worker count.
    """
    from concurrent.futures.process import BrokenProcessPool
    workers = max(1, int(extraction_workers))
    for attempt in range(1, attempts + 1):
        if used is not None:
            used.append(workers)
        try:
            return trainer._load_feature_caches(*args, extraction_workers=workers, **kwargs)
        except BrokenProcessPool:
            if attempt == attempts:
                raise
            workers = max(1, workers // 2)
            trainer._status("A feature extraction process stopped unexpectedly; "
                            f"extracting the unfinished runs again with {workers} workers")


def _bound_json(descriptor):
    """Read a supplied content-bound document before importing numerical code."""
    from ..review_storage import resolve_review_path
    path = resolve_review_path(descriptor["path"])
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != descriptor["sha256"]:
        raise ValueError("Supplied preparation or configuration SHA changed")
    document = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(document, Mapping):
        raise ValueError("A bound JSON object is required")
    return path, document


def _effective_settings(settings, trainer):
    """Resolve only existing numerical options, without historical input defaults."""
    parser = trainer.build_parser()
    effective = {key: deepcopy(settings.get(key, parser.get_default(key))) for key in OPTIONS}
    effective["backend_requested"] = effective["backend"]
    if effective["backend"] == "auto":
        effective["backend"] = "cuda" if trainer.cuda_available() else "cpu"
    if settings.get("fit_mode", "mil") != "mil":
        raise ValueError("This fresh candidate producer uses the existing MIL fit")
    if settings.get("evaluation_mode", "whole-run") != "whole-run":
        raise ValueError("This producer uses whole-experiment cross-validation")
    if settings.get("adapters") not in (None, ["generic"], ("generic",)):
        raise ValueError("This producer fits the generic optical heads")
    effective.update(fit_mode="mil", evaluation_mode="whole-run", adapters=["generic"])
    return effective


def _fit_view(features, owners, regions):
    """Remove ownerless inventory rows and compact indices without changing pixels."""
    import numpy as np
    matrix, indices = np.asarray(features), np.asarray(owners)
    if (matrix.ndim != 2 or indices.shape != (len(matrix),) or indices.dtype.kind not in "iu"
            or not len(indices) or indices.min() < 0 or indices.max() >= len(regions)):
        raise ValueError("Sample owners differ from the returned region inventory")
    selected = np.unique(indices)
    remap = np.full(len(regions), -1, dtype=np.int32)
    remap[selected] = np.arange(len(selected), dtype=np.int32)
    return matrix, remap[indices], [regions[int(index)] for index in selected]


def _request_bytes(document, fragment):
    """Serialize the execution request while retaining the exact input-photometry bytes."""
    from ..native_training_photometry import _fragment
    from .raw_fresh_request import PHOTOMETRY_KEY
    value = deepcopy(document)
    marker = uuid.uuid4().hex
    value["settings"][PHOTOMETRY_KEY] = marker
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    encoded = encoded.replace(json.dumps(marker), fragment.decode("utf-8"), 1).encode("utf-8")
    if _fragment(_fragment(encoded, "settings"), PHOTOMETRY_KEY) != fragment:
        raise ValueError("Execution request changed saved input photometry")
    return encoded + b"\n"


def _context_views(sources, origins, regions, context, root, reader, bound_sources):
    """Make explicit context-only views of every original reviewed source revision."""
    from ..phase_saved_assets import asset
    from .frame_context import PHOTO_FIELDS, V5FrameGeometryCohort
    from .raw_frame_context import RawFrameGeometryCohort
    from .raw_fresh_request import PHYSICAL_BINDING_KEYS
    from . import trainer
    unique = {(str(row.asset_manifest_path.resolve()), str(row.review_root.resolve())): row
              for row in (*sources, *origins.values())}
    frames = {}
    for source, resolved in trainer._native_revision_task_groups(regions, origins):
        identity = (str(source.asset_manifest_path.resolve()), str(source.review_root.resolve()))
        frames[identity] = sorted({row.frame_index for row in resolved})
    views, records = {}, []
    for index, (identity, source) in enumerate(sorted(unique.items()), start=1):
        original = dict(bound_sources[str(source.asset_manifest_path.resolve())])
        manifest = reader.json(original)
        path = source.asset_manifest_path
        if context is not None:
            view = deepcopy(manifest)
            photo = {key: value for key, value in (manifest.get("training_context") or {}).items()
                     if key not in PHYSICAL_BINDING_KEYS}
            view["training_context"] = dict(photo, **context["binding"])
            path = root / f"{index:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                json.dump(view, stream, indent=2, ensure_ascii=False, allow_nan=False)
        cohort = RawFrameGeometryCohort.from_manifest(path)
        derived = []
        for frame in frames.get(identity, ()):
            if context is None:
                continue
            arrays, binding = V5FrameGeometryCohort.context(cohort, frame,
                fields=("observable_phase_domain", *PHOTO_FIELDS))
            derived.append(dict(frame_index=frame, receipt_sha256=binding["receipt_sha256"],
                fields={key: dict(shape=list(value.shape), dtype=str(value.dtype),
                                  sha256=sha256(value.tobytes(order="C")).hexdigest())
                        for key, value in arrays.items()}))
            del arrays
        views[identity] = replace(source, asset_manifest_path=path)
        records.append(dict(original_manifest=original, extraction_manifest=asset(path),
            review_root=str(source.review_root), run=source.run, dataset=source.dataset,
            canvas_wh=manifest["canvas_wh"], content_box_xywh=manifest["content_box_xywh"],
            independent_reference_frames=list(source.independent_reference_frames),
            full_context_derivations=derived))
    def view_for(source):
        return views[(str(source.asset_manifest_path.resolve()), str(source.review_root.resolve()))]
    return [view_for(row) for row in sources], {key: view_for(row) for key, row in origins.items()}, records


def train(preparation: Mapping, *, output_root: Path, cache_root: Path):
    """Verify selected evidence and publish a fresh, independently unvalidated candidate."""
    _path, preliminary = _bound_json(preparation)
    configuration_path, _document = _bound_json(preliminary["native_configuration"])
    from ..native.config import configure_parameters, load_parameters, reset_parameters
    values = dict(DARSIA_NATIVE_CONFIGURATION=str(configuration_path.resolve()),
                  DARSIA_NATIVE_CONFIGURATION_SHA256=preliminary["native_configuration"]["sha256"],
                  OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    previous = {key: os.environ.get(key) for key in values}
    token = configure_parameters(load_parameters(configuration_path, values["DARSIA_NATIVE_CONFIGURATION_SHA256"]))
    os.environ.update(values)
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            return _train(preparation, output_root=output_root, cache_root=cache_root)
    finally:
        reset_parameters(token)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _train(preparation, *, output_root, cache_root):
    """Execute under the explicitly bound configuration, including spawned extraction."""
    preparation_path, _preliminary = _bound_json(preparation)
    import numpy as np
    from ..native_training_photometry import _fragment, _read, require_saved_sidecars
    from ..phase_saved_assets import VerifiedAssetReader, asset
    from ..phase_workflow import PHASE_TRAINING_REQUEST_FORMAT
    from .. import phase_review_retractions as withdrawals
    from . import raw_fresh_request as fresh, trainer as t
    from .scope import requested_reviewers, require_reviewer_scope_allowed, validate_training_series
    from .review_sampling import sampling_receipt

    reader = VerifiedAssetReader()
    document = reader.json(preparation)
    if document.get("format") != fresh.FORMAT:
        raise ValueError("An explicit saved RAW training preparation is required")
    configuration_path = reader.path(document["native_configuration"])
    settings = deepcopy(document["settings"])
    require_reviewer_scope_allowed(settings)
    if any(key in settings for key in fresh.FROZEN_ANNOTATION_SET_SETTINGS) or any(settings.get(key)
            for key in ("resume_completed_heads", "completed_fit_source", "imported_completed_heads", "inherited_heads")):
        raise ValueError("A fresh cohort cannot inherit frozen partitions or completed heads")
    if settings.get("radiometry_profile") not in (None, "", "legacy"):
        raise ValueError("RAW camera-linear input cannot use legacy radiometry")
    raw, _ = _read(preparation_path, preparation["sha256"])
    parent_raw, parent = _read(**dict(path=document["provenance"]["parent_request"]["path"],
                                    expected=document["provenance"]["parent_request"]["sha256"]))
    fragment = _fragment(_fragment(raw, "settings"), fresh.PHOTOMETRY_KEY)
    if (fragment != _fragment(_fragment(parent_raw, "settings"), fresh.PHOTOMETRY_KEY)
            or fragment != reader.path(document["provenance"]["photometry_fragment"]).read_bytes()
            or settings[fresh.PHOTOMETRY_KEY] != parent["settings"][fresh.PHOTOMETRY_KEY]):
        raise ValueError("Prepared input photometry differs from its bound parent")
    fresh._verify_photometry_inputs(settings[fresh.PHOTOMETRY_KEY], reader)
    context = None
    if document["context_mode"] == "full":
        context = fresh.inspect_physical_context(document["physical_context"]["source"],
                                                document["native_configuration"], reader=reader)
        if context != document["physical_context"]:
            raise ValueError("Prepared physical-context binding changed")
    elif document["context_mode"] != "photo-only":
        raise ValueError("Choose explicit full or photo-only context")
    bundles, records, bound_sources, checked_jobs, native_paths = [], [], {}, {}, {}
    for record in document["bundles"]:
        paths = [reader.path(record[role]) for role in ("decisions", "combined_manifest")]
        actual = fresh._verify_raw_jobs(reader.json(record["combined_manifest"]), reader=reader, checked=checked_jobs)
        if actual != record["source_jobs"]:
            raise ValueError("Prepared RAW source declarations changed")
        for row in actual:
            pin = row["native_manifest"]
            key = (pin["path"], pin["sha256"])
            if key not in native_paths:
                native_paths[key] = str(reader.path(pin).resolve())
            bound_sources[native_paths[key]] = pin
        bundles.append(t.ReviewBundle(*paths))
        records.append(dict(record))
    require_saved_sidecars([(row.decisions_path, row.combined_manifest_path) for row in bundles],
                           SimpleNamespace(settings=settings))
    reviewers = requested_reviewers(settings)
    policy = withdrawals.snapshot_for_bundles(
        [(row.decisions_path, row.combined_manifest_path) for row in bundles], settings)
    from .scope import partition_review_events
    policy = partition_review_events(policy, reviewers)
    withdrawals.require_fresh_fit(policy, settings)
    settings[withdrawals.DOCUMENT_KEY] = policy
    options = _effective_settings(settings, t)
    limits = execution_limits(options, [row["native_shape_hw"]
                                       for record in records for row in record["source_jobs"]])
    options.update({key: limits[key] for key in ("stage_workers", "extraction_workers", "opencv_threads")})
    source_sha256 = dict(trainer=t._sha256(Path(t.__file__)),
        training_primitives=t._sha256(t.PACKAGE_ROOT / "logistic.py"),
        optical_features=t._sha256(t.FLUIDFLOWER_ROOT / "phase_features.py"),
        fresh_runner=t._sha256(Path(__file__)),
        reviewed_raw_basis=t._sha256(Path(__file__).with_name("raw_review_training.py")))
    output, cache = Path(output_root).resolve(), Path(cache_root).resolve()
    studio_owned = document.get("prepared_model_root") is not None
    if studio_owned:
        from .raw_review_training import FAMILY
        if (Path(document["prepared_model_root"]).resolve() != output
                or Path(document["provenance"]["parent_request"]["path"]).resolve() != output / "training_request.json"
                or document.get("execution_family") != FAMILY
                or parent.get("settings", {}).get("training_family") != FAMILY):
            raise ValueError("The fresh training output does not belong to its original Studio request")
        allowed = {"training_request.json", "raw_training_preparation.json", "input_snapshot",
                   "model_ui_name.json", Path(parent["review_retractions"]["path"]).name}
        if {path.name for path in output.iterdir()} - allowed:
            raise FileExistsError("This fresh training job already contains work; retain it and prepare a new version")
        name_path = output / "model_ui_name.json"
        if name_path.exists():
            name = json.loads(name_path.read_text(encoding="utf-8-sig"))
            if (name.get("format") != "phase-model-ui-name-v1" or name.get("model_id") != parent["model_id"]
                    or name.get("training_request_sha256") != document["provenance"]["parent_request"]["sha256"]):
                raise ValueError("The saved display name belongs to another training request")
    protected = [preparation_path.resolve(), configuration_path.resolve(), *[path.resolve()
        for row in bundles for path in (row.decisions_path, row.combined_manifest_path)],
        *[Path(path) for path in bound_sources]]
    if (output == cache or output.is_relative_to(cache) or cache.is_relative_to(output)
            or any(path.is_relative_to(cache) or (path.is_relative_to(output) and not studio_owned) for path in protected)):
        raise ValueError("Choose separate output and cache locations outside the bound inputs")
    output.mkdir(parents=True, exist_ok=studio_owned)
    if studio_owned:
        with (output / "training_started.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(preparation=dict(preparation), created_utc=datetime.now(timezone.utc).isoformat()), stream)
    snapshot = output / "input_snapshot"
    snapshot.mkdir(exist_ok=studio_owned)
    withdrawal_record = withdrawals.write_model_snapshot(output, policy)
    for event, decisions, manifest, _ in withdrawals.materialize_replacements(policy, snapshot / "evidence_actions"):
        bundles.append(t.ReviewBundle(decisions, manifest))
        records.append(dict(decisions=asset(decisions), combined_manifest=asset(manifest)))
        bound = event["replacement"]["native_manifest"]
        bound_sources[str(reader.path(bound).resolve())] = bound
        fresh._verify_raw_jobs(t._read_json(manifest), reader=reader)
    origins, audit = {}, {}
    sources, regions, conflicts = t._load_sources_and_regions(bundles, withdrawal_snapshot=policy,
        preserve_native_revisions=True, region_sources=origins, label_policy_audit=audit, reviewers=reviewers)
    validate_training_series(sources, regions, document["series_scope"])
    for source in (*sources, *origins.values()):
        bound = bound_sources.get(str(source.asset_manifest_path.resolve()))
        if bound is None:
            raise ValueError("Reviewed native source is outside the bound RAW inputs")
        reader.path(bound)
    excluded = {str(run).casefold() for run in settings.get("fit_excluded_runs", ())}
    allowed = {row.region_id for row in regions if row.run.casefold() not in excluded
        and sampling_receipt(row.phase_class, row.payload)["role"] in
            {"whole_region_uniform", "regional_mil_uniform_candidates"}}
    sources, origins, views = _context_views(sources, origins,
        [row for row in regions if row.region_id in allowed], context,
        snapshot / "native_context_views", reader, bound_sources)
    settings.update(options)
    configuration = document["native_configuration"]
    feature_cache = cache / ("native-" + configuration["sha256"])
    request = dict(schema=1, format=PHASE_TRAINING_REQUEST_FORMAT, model_id=output.name,
        created_utc=datetime.now(timezone.utc).isoformat(), created_by=document.get("created_by", ""),
        output_root=str(output), cache_root=str(cache), settings=settings,
        bundles=[dict(bundle_index=index, **{role: dict(value[role], snapshot_path=value[role]["path"])
                 for role in ("decisions", "combined_manifest")}) for index, value in enumerate(records, 1)],
        provenance=dict(preparation=dict(preparation), original_to_context_views=views,
            native_configuration=configuration, feature_cache_root=str(feature_cache),
            source_sha256=source_sha256,
            parent_request=document["provenance"]["parent_request"],
            excluded_inherited_settings=document["provenance"]["excluded_inherited_settings"]),
        review_retractions=withdrawal_record, execution_resources=limits)
    request_path = output / ("training_execution.json" if studio_owned else "training_request.json")
    with request_path.open("xb") as stream:
        stream.write(_request_bytes(request, fragment))
    staging = cache / "fresh_result_staging"
    staging.mkdir(parents=True, exist_ok=False)
    variants = t.model_variants(adapters=("generic",), generic_version=options["feature_version"],
                                include_change_ablation=options["change_evidence_ablation"])
    maximal = replace(t.MAXIMAL_CONTRACT, generic_version=options["feature_version"])
    extraction_attempts = limits["extraction_workers_by_attempt"] = []
    features, owners, ordered, extraction = load_features_with_retries(t, sources, regions,
        cache_root=feature_cache, stage_workers=options["stage_workers"],
        extraction_workers=options["extraction_workers"], opencv_threads=options["opencv_threads"],
        pixels_per_region=options["pixels_per_region"], texture_sigma=options["texture_sigma"],
        radiometry_profile=None, feature_contract=maximal, fit_allowed_review_ids=allowed,
        fit_excluded_runs=sorted(excluded), origin_sources=origins, used=extraction_attempts)
    matrix, fit_owners, fit_regions = _fit_view(features, owners, ordered)
    inventory = dict(regions=len(ordered), pixel_rows=len(features), runs=sorted({row.run for row in ordered}),
        datasets=sorted({row.dataset for row in ordered}), training_label_policy=audit,
        fit_region_ids=[row.region_id for row in fit_regions],
        evaluation_only_region_ids=[row.region_id for row in ordered if row.region_id not in allowed],
        deduplication_conflicts=conflicts, **t._reviewer_inventory(ordered, reviewers))
    from .raw_review_training import reviewed_input_basis, canonical_digest
    image_basis = reviewed_input_basis([(row.decisions_path, row.combined_manifest_path) for row in bundles], reader=reader)
    image_basis["input_photometry"] = deepcopy(settings[fresh.PHOTOMETRY_KEY])
    image_basis["input_photometry_sha256"] = canonical_digest(image_basis["input_photometry"])
    basis_path = output / "image_basis.json"
    with basis_path.open("x", encoding="utf-8") as stream:
        json.dump(image_basis, stream, indent=2, ensure_ascii=False, allow_nan=False)
    basis_binding = asset(basis_path)
    reports, decisions = {}, {}
    for variant, contract in variants.items():
        selected = t._variant_matrix(matrix, contract, maximal)
        heads, thresholds = {}, {}
        def fit_head(head):
            return t._cross_validate_head(selected, fit_owners, fit_regions,
                head=head, regularization_grid=options["regularization_grid"], backend=options["backend"],
                fit_mode="mil", weak_strength=options["weak_strength"],
                positive_top_fraction=options["positive_top_fraction"], mil_max_iters=options["mil_max_iters"],
                selection_policy=options["gas_selection_policy"] if head == "gas_within_co2" else "regional-oof-v1")
        # Head fits are independent on CPU. A CUDA backend owns one device and
        # stays serial; parallel run staging/extraction still uses the CPU.
        fit_workers = min(2, options["extraction_workers"]) if options["backend"] == "cpu" else 1
        import psutil
        per_fit = max(128 * 1024**2, selected.nbytes * 8)
        fit_workers = min(fit_workers, max(1, int(psutil.virtual_memory().available * .8) // per_fit))
        limits["fit_workers"] = fit_workers
        if fit_workers > 1:
            with ThreadPoolExecutor(max_workers=fit_workers) as executor:
                futures = {head: executor.submit(fit_head, head) for head in HEADS}
                fitted = {head: futures[head].result() for head in HEADS}
        else:
            fitted = {head: fit_head(head) for head in HEADS}
        for head in HEADS:
            report, fit, threshold, parity = fitted[head]
            report.update(deployment_allowed=False, generalization_validated=False)
            provenance = dict(campaign_id=output.name, target_kind=f"visible_{head}_phase_probability",
                fit_mode="mil", deployment_allowed=False, review_candidate_only=True,
                series_scope=document["series_scope"], datasets=inventory["datasets"],
                head_training_runs=report["eligible_runs"], training_inventory=inventory,
                cross_validation=t._json_safe(report), fresh_preparation=dict(preparation),
                backend_contract=dict(production_backend=fit.backend,
                    production_precision=fit.precision, authoritative_reference="cpu-float64",
                    validated_backends=["cpu"] if options["backend"] == "cpu" else ["cpu", "cuda"],
                    parity=t._json_safe(parity)),
                original_to_context_views=views, native_configuration=configuration,
                image_basis=basis_binding,
                template_model=deepcopy(settings.get("template_model")),
                source_sha256=source_sha256,
                input_photometry=settings[fresh.PHOTOMETRY_KEY],
                **t._reviewer_provenance(fit_regions, reviewers))
            model = t._fit_model_object(fit, feature_contract=contract, provenance=t._json_safe(provenance),
                                       texture_sigma=options["texture_sigma"])
            path = model.save(staging / "models" / variant / f"{head}_signal.json")
            report.update(model_path=str(output / path.relative_to(staging)),
                          model_sha256=model.to_dict()["model_sha256"], numerical_validation=parity)
            heads[head] = t._json_safe(report)
            thresholds[head] = dict(presence_threshold=threshold, model_sha256=report["model_sha256"])
        reports[variant] = dict(feature_contract=contract.to_dict(), heads=heads)
        decisions[variant] = thresholds
    t._write_region_inventory(staging / "region_inventory.csv", ordered, reviewer_ids=True)
    np.savez_compressed(staging / "training_samples.npz", features=features, region_indices=owners,
                        feature_names=np.asarray(maximal.feature_names))
    summary = dict(schema=1, model_id=output.name, campaign_id=output.name,
        artifact_kind="fresh-raw-training-candidate", created_utc=request["created_utc"],
        deployment_allowed=False, generalization_validated=False,
        evaluation=dict(kind="whole-experiment-candidate", independent_experiment_required=True,
                        deployment_allowed=False, generalization_validated=False),
        inventory=inventory, runs=inventory["runs"], models=reports, feature_extraction=extraction,
        source_snapshot=request["bundles"], provenance=request["provenance"], review_retractions=withdrawal_record,
        execution_resources=limits)
    summary["image_basis"] = basis_binding
    for name, value in (("summary.json", summary), ("training_inventory.json", inventory),
                        ("cross_validation.json", reports), ("phase_decision_policy.json", dict(schema=1, models=decisions))):
        t._write_json(staging / name, t._json_safe(value))
    t._sync_result(staging, output)
    for path in staging.rglob("*"):
        if path.is_file() and t._sha256(path) != t._sha256(output / path.relative_to(staging)):
            raise OSError("Published fresh candidate hash mismatch: " + str(path))
    return summary


def build_command(preparation: Mapping, *, output_root: Path, cache_root: Path):
    """Build this explicit producer's command without routing through the frozen trainer."""
    from ..phase_workflow import phase_python_executable
    return [phase_python_executable(), "-B", "-m", __name__, "--preparation", str(preparation["path"]),
            "--preparation-sha256", preparation["sha256"], "--output-root", str(output_root),
            "--cache-root", str(cache_root)]


def main(argv=None):
    """Run one explicitly supplied preparation with bound configuration and fresh outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--preparation-sha256")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--request-sha256")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args(argv)
    if args.request:
        if not args.request_sha256 or args.preparation:
            parser.error("Choose one SHA-bound Studio request or one SHA-bound preparation")
        from .raw_review_training import execution_preparation
        request = dict(path=str(args.request), sha256=args.request_sha256)
        _path, document = _bound_json(request)
        prepared = execution_preparation(request)
        result = train(prepared, output_root=Path(document["output_root"]), cache_root=Path(document["cache_root"]))
    else:
        if not all((args.preparation, args.preparation_sha256, args.output_root, args.cache_root)):
            parser.error("A preparation needs its SHA, output root and cache root")
        result = train(dict(path=str(args.preparation), sha256=args.preparation_sha256),
                       output_root=args.output_root, cache_root=args.cache_root)
    print(json.dumps(dict(model_id=result["model_id"], deployment_allowed=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
