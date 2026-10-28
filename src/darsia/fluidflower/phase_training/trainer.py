"""Train transferable gas/aqueous optical heads from reviewed FF-AC and FF-UM.

The command consumes immutable review snapshots, stages native corrected
review images locally, extracts the shared optical feature contract in
parallel, and performs leave-one-run-out validation.  Repeated logistic fits
run on CUDA float32 when available; a deterministic CPU float64 fit remains
the numerical parity reference.
"""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path
from darsia.fluidflower.series_paths import local_cache_root

import argparse
import csv
import gc
import hashlib
import json
import multiprocessing
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from darsia.fluidflower.phase_features import (
    DARSIA_CHANGE_DETECTOR,
    GENERIC_FEATURE_VERSION,
    RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION,
    NO_CHANGE_DETECTOR,
    OpticalFeatureContract,
)
from darsia.fluidflower.phase_training.logistic import (
    CANDIDATE_ONLY_UNSUPPORTED_PHASE_ISSUES,
    LEGACY_CANDIDATE_AQUEOUS_ISSUE,
    LEGACY_CANDIDATE_GAS_ISSUE,
    MANUAL_TRAINING_LABEL_POLICY_VERSION,
    MIL_NORMALIZATION_POLICY,
    MIL_SUPERVISION_POLICY,
    pixel_supervision_metadata,
    LogisticFit,
    ReviewedRegion,
    balanced_pixel_view,
    canonical_phase_class,
    cuda_available,
    fit_weighted_logistic,
    fit_weighted_logistic_mil,
    predict_logistic,
    _try_cuda_prediction_feature_buffer,
    region_probability_scores,
    regional_evaluation_view,
    regional_metrics,
    regional_presence_target,
    tune_presence_threshold,
)
from darsia.fluidflower.phase_training.model import (
    TRANSFERABLE_MODEL_ALGORITHM,
    TRANSFERABLE_MODEL_SCHEMA,
    DualPhaseSignalModel,
    feature_samples,
)
from darsia.fluidflower.phase_training.review_regions import (
    _annotation_mask,
    _load_rgb,
    _sample_mask,
)
from darsia.fluidflower.raw_photographic_review import review_content_shape, timeline_rows

PACKAGE_ROOT = Path(__file__).resolve().parent
FLUIDFLOWER_ROOT = PACKAGE_ROOT.parent


CAMPAIGN_ID = "crossseries_phase_model_v3_20260830"
DEFAULT_OUTPUT = configured_tool_path('ffac', 'calibration_logs') / f'{CAMPAIGN_ID}'
DEFAULT_CACHE = local_cache_root() / CAMPAIGN_ID
DEFAULT_SNAPSHOT = DEFAULT_OUTPUT / "input_snapshot"
DEFAULT_BUNDLES = (
    (
        DEFAULT_SNAPSHOT / "training_decisions_ac18_run15.json",
        DEFAULT_SNAPSHOT / "training_combined_manifest.json",
    ),
    (
        DEFAULT_SNAPSHOT / "completed_validation_ac52_run16.json",
        DEFAULT_SNAPSHOT / "validation_combined_manifest.json",
    ),
)

MODEL_VARIANTS: Mapping[str, OpticalFeatureContract] = {
    "generic": OpticalFeatureContract(
        chemistry_adapter="none",
        reference_adapter="none",
        include_dynamic=True,
        generic_version=GENERIC_FEATURE_VERSION,
        change_detector=DARSIA_CHANGE_DETECTOR,
    ),
    "btb-ratiometric-relative-v2": OpticalFeatureContract(
        chemistry_adapter="btb-ratiometric-relative-v2",
        reference_adapter="none",
        include_dynamic=True,
        generic_version=GENERIC_FEATURE_VERSION,
        change_detector=DARSIA_CHANGE_DETECTOR,
    ),
}
CHANGE_ABLATION_VARIANT = "generic-no-darsia-change"
CHANGE_ABLATION_CONTRACT = OpticalFeatureContract(
    chemistry_adapter="none",
    reference_adapter="none",
    include_dynamic=True,
    generic_version=GENERIC_FEATURE_VERSION,
    change_detector=NO_CHANGE_DETECTOR,
)
MAXIMAL_CONTRACT = MODEL_VARIANTS["btb-ratiometric-relative-v2"]


def model_variants(
    *, include_change_ablation: bool = False, generic_version: str = GENERIC_FEATURE_VERSION,
    adapters: Sequence[str] | None = None,
) -> dict[str, OpticalFeatureContract]:
    """Return the explicitly selected production variants and optional ablation."""

    if generic_version not in {GENERIC_FEATURE_VERSION, RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION}:
        raise ValueError(f"Unsupported training feature version {generic_version!r}")
    selected = tuple(MODEL_VARIANTS if adapters is None else adapters)
    if not selected or len(set(selected)) != len(selected) or any(
        name not in MODEL_VARIANTS for name in selected
    ):
        raise ValueError(f"Select distinct supported optical adapters: {selected!r}")
    variants = {name: replace(MODEL_VARIANTS[name], generic_version=generic_version)
                for name in selected}
    if include_change_ablation:
        variants[CHANGE_ABLATION_VARIANT] = replace(CHANGE_ABLATION_CONTRACT, generic_version=generic_version)
    return variants


@dataclass(frozen=True)
class ReviewBundle:
    decisions_path: Path
    combined_manifest_path: Path


@dataclass(frozen=True)
class RunSource:
    run: str
    dataset: str
    review_root: Path
    asset_manifest_path: Path
    decisions_path: Path
    combined_manifest_path: Path
    independent_reference_frames: tuple[int, ...] = ()


@dataclass(frozen=True)
class StagedRun:
    run: str
    dataset: str
    root: Path
    asset_manifest_path: Path
    baseline_path: Path
    frame_paths: tuple[Path, ...]
    asset_sha256: str
    independent_reference_frames: tuple[int, ...] = ()
    baseline_extended_path: Path | None = None
    frame_extended_paths: tuple[Path, ...] = ()
    baseline_valid_mask_path: Path | None = None
    frame_valid_mask_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ExtractionTask:
    staged: StagedRun
    regions: tuple[ReviewedRegion, ...]
    output_path: Path
    metadata_path: Path
    cache_key: str
    pixels_per_region: int
    texture_sigma: float
    opencv_threads: int
    radiometry_profile: dict[str, Any] | None = None
    feature_contract: OpticalFeatureContract = MAXIMAL_CONTRACT


def _sha256(path: Path) -> str:
    from ..review_storage import resolve_review_path
    digest = hashlib.sha256()
    with resolve_review_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    from ..review_storage import resolve_review_path
    return json.loads(resolve_review_path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(destination)


def _status(message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _source_priority(region: ReviewedRegion) -> tuple[int, int, str, str]:
    source = 1 if region.source == "manual" else 0
    confidence = {"high": 2, "medium": 1, "low": 0}.get(region.confidence, 0)
    timestamp = str(
        region.payload.get("updated_utc") or region.payload.get("answered_utc") or ""
    )
    return source, confidence, timestamp, region.region_id


def _geometry_key(run: str, item: Mapping[str, Any]) -> str:
    points = item.get("points") or []
    if points:
        geometry: Any = [
            [round(float(point[0]), 7), round(float(point[1]), 7)] for point in points
        ]
    else:
        geometry = [
            round(float(item.get(name) or 0.0), 7)
            for name in ("x", "y", "width", "height")
        ]
    return _canonical_hash(
        {
            "run": run,
            "frame_index": int(item.get("frame_index") or 0),
            "geometry": geometry,
        }
    )


def _load_sources_and_regions(
    bundles: Sequence[ReviewBundle],
    *, withdrawal_snapshot: Mapping[str, Any] | None = None,
    preserve_native_revisions: bool = False,
    label_policy_audit: dict[str, Any] | None = None,
    region_sources: dict[str, RunSource] | None = None,
    reviewers: Sequence[str] | None = None,
) -> tuple[list[RunSource], list[ReviewedRegion], list[dict[str, Any]]]:
    """Reviewed regions and their run sources; ``reviewers`` keeps only their evidence."""
    from .scope import evidence_reviewer
    sources: dict[str, RunSource] = {}
    selected_by_geometry: dict[str, ReviewedRegion] = {}
    selected_source_by_geometry: dict[str, RunSource] = {}
    conflicts: list[dict[str, Any]] = []
    pending: list[tuple[RunSource, ReviewedRegion]] = []
    evidence_records: list[tuple[str, dict[str, Any]]] = []
    excluded_candidate_phase_issues: list[dict[str, Any]] = []

    from .. import phase_review_retractions as withdrawals
    inputs = [(bundle, _read_json(bundle.decisions_path), _read_json(bundle.combined_manifest_path)) for bundle in bundles]
    if reviewers is not None:
        from .scope import partition_review_events, partition_reviewers
        if withdrawal_snapshot is None:
            raise ValueError("A reviewer scope needs the frozen evidence-action policy")
        withdrawal_snapshot = partition_review_events(withdrawal_snapshot, reviewers)
        inputs = [(bundle, partition_reviewers(doc, reviewers), manifest) for bundle, doc, manifest in inputs]
    policy = withdrawal_snapshot
    if policy is None:
        embedded = [doc[withdrawals.DOCUMENT_KEY] for _, doc, _ in inputs if doc.get(withdrawals.DOCUMENT_KEY)]
        if embedded:
            policy = withdrawals.validate_snapshot(embedded[0])
            if any(withdrawals.validate_snapshot(row) != policy for row in embedded):
                raise ValueError("Training bundles contain different frozen evidence-action policies")
    if policy is not None:
        prepared_inputs = []
        for bundle, doc, manifest in inputs:
            filtered, _ = withdrawals.filter_document(doc, manifest, policy)
            prepared_inputs.append((bundle, filtered, manifest))
        inputs = prepared_inputs
        represented = {str(item.get("id")) for _, doc, _ in inputs
                       for decision in (doc.get("decisions") or {}).values()
                       for item in decision.get("annotations") or []}
        for event, doc, manifest in withdrawals.replacement_documents(policy):
            annotation = event["replacement"]["annotation"]
            if str(annotation["id"]) in represented:
                continue
            # Direct trainer callers also receive replacements absent from their
            # selected old bundles. Prepared jobs normally materialize these files.
            origin = Path(event["replacement"]["native_manifest"]["path"]).parent
            bundle = ReviewBundle(origin / ("evidence-action-" + event["event_id"] + ".json"),
                                  Path(event["replacement"]["combined_manifest"]["path"]))
            inputs.append((bundle, doc, manifest))
            represented.add(str(annotation["id"]))
    for bundle, decisions_payload, combined in inputs:
        snapshot = withdrawal_snapshot or decisions_payload.get(withdrawals.DOCUMENT_KEY)
        if snapshot is not None:
            decisions_payload, _ = withdrawals.filter_document(decisions_payload, combined, snapshot)
        jobs_by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for job in combined.get("jobs") or []:
            jobs_by_run[str(job.get("run") or "").lower()].append(job)

        from .gas_negative_sampling import resolve_document_sources
        decisions_payload = resolve_document_sources(decisions_payload, Path(bundle.decisions_path).parent)
        for run, decision in sorted((decisions_payload.get("decisions") or {}).items()):
            normalized_run = str(run).lower()
            jobs = jobs_by_run.get(normalized_run) or []
            if not jobs:
                raise RuntimeError(
                    f"No review job maps {normalized_run} in {bundle.combined_manifest_path}"
                )
            dataset = str(jobs[0].get("dataset") or "unknown").lower()
            from ..review_storage import resolve_review_path
            review_root = resolve_review_path(str(jobs[0].get("analysis_source_review_root") or jobs[0]["review_output_dir"]))
            manifest_path = (
                review_root / "native_frames" / normalized_run / "manifest.json"
            )
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            from .radiometry import analysis_review_root
            review_root = analysis_review_root(review_root, normalized_run)
            manifest_path = review_root / "native_frames" / normalized_run / "manifest.json"
            source = RunSource(
                run=normalized_run,
                dataset=dataset,
                review_root=review_root,
                asset_manifest_path=manifest_path,
                decisions_path=Path(bundle.decisions_path),
                combined_manifest_path=Path(bundle.combined_manifest_path),
                independent_reference_frames=tuple(int(index) for index in
                                                   (jobs[0].get("independent_reference_frames") or ())),
            )
            existing_source = sources.get(normalized_run)
            if existing_source is not None and existing_source.dataset != dataset:
                raise RuntimeError(
                    f"Conflicting dataset assignment for {normalized_run}"
                )
            sources[normalized_run] = source

            reviewer = str(decision.get("reviewer") or "unknown")
            records = (
                ("manual", decision.get("annotations") or []),
                ("quiz", decision.get("phase_quiz_answers") or []),
            )
            for review_source, items in records:
                for item in items:
                    identifier = str(
                        item.get("id")
                        or item.get("proposal_id")
                        or _geometry_key(normalized_run, item)[:16]
                    )
                    evidence_records.append((f"{normalized_run}:{review_source}:{identifier}", dict(item)))
                    if (review_source == "manual" and
                            str(item.get("category") or "").strip().lower()
                            in CANDIDATE_ONLY_UNSUPPORTED_PHASE_ISSUES):
                        excluded_candidate_phase_issues.append(dict(
                            region_id=f"{normalized_run}:manual:{identifier}",
                            category=str(item["category"]).strip().lower(),
                            dataset=dataset, run=normalized_run,
                            frame_index=item.get("frame_index"),
                            candidate=item.get("candidate"),
                            candidate_id=item.get("candidate_id"),
                            params_sha256=item.get("params_sha256"),
                            candidate_identity_complete=bool(item.get("candidate_id")
                                                             and item.get("params_sha256")),
                            evaluation_scope="trigger_candidate_only"))
                    phase_class = canonical_phase_class(item, review_source)
                    if phase_class is None or phase_class == "ignored":
                        continue
                    confidence = str(item.get("confidence") or "low").lower()
                    region = ReviewedRegion(
                        region_id=f"{normalized_run}:{review_source}:{identifier}",
                        run=normalized_run,
                        dataset=dataset,
                        frame_index=int(item.get("frame_index") or 0),
                        elapsed_h=float(item.get("elapsed_h") or 0.0),
                        phase_class=phase_class,
                        confidence=confidence,
                        source=review_source,
                        reviewer=str(item.get("reviewer") or reviewer),
                        payload=dict(item),
                        reviewer_id=evidence_reviewer(item, decision),
                    )
                    pending.append((source, region))

    from ..phase_evidence_audit import superseded_evidence_ids
    superseded = superseded_evidence_ids(evidence_records)
    if label_policy_audit is not None:
        active_by_id = {row["region_id"]: row for row in excluded_candidate_phase_issues
                        if row["region_id"] not in superseded}
        active_issues = [active_by_id[key] for key in sorted(active_by_id)]
        gas_issues = [row for row in active_issues
                      if row["category"] == LEGACY_CANDIDATE_GAS_ISSUE]
        aqueous_issues = [row for row in active_issues
                          if row["category"] == LEGACY_CANDIDATE_AQUEOUS_ISSUE]
        label_policy_audit.update(
            format="visible-phase-training-label-audit-v2",
            policy_version=MANUAL_TRAINING_LABEL_POLICY_VERSION,
            excluded_candidate_issue_count=len(active_issues),
            excluded_candidate_issues_by_category=dict(sorted(Counter(
                row["category"] for row in active_issues).items())),
            excluded_legacy_candidate_gas_issue_count=len(gas_issues),
            excluded_legacy_candidate_gas_issue_ids=[row["region_id"] for row in gas_issues],
            excluded_legacy_candidate_aqueous_issue_count=len(aqueous_issues),
            excluded_legacy_candidate_aqueous_issue_ids=[row["region_id"] for row in aqueous_issues],
            candidate_issue_evaluation_records=active_issues,
            missing_candidate_identity_count=sum(
                not row["candidate_identity_complete"] for row in active_issues),
            missing_candidate_identity_ids=[row["region_id"] for row in active_issues
                                            if not row["candidate_identity_complete"]],
            excluded_legacy_candidate_gas_issues_by_series=dict(sorted(Counter(
                row["dataset"] for row in gas_issues).items())),
            excluded_legacy_candidate_gas_issues_by_run=dict(sorted(Counter(
                row["run"] for row in gas_issues).items())),
            excluded_legacy_candidate_aqueous_issues_by_series=dict(sorted(Counter(
                row["dataset"] for row in aqueous_issues).items())),
            excluded_legacy_candidate_aqueous_issues_by_run=dict(sorted(Counter(
                row["run"] for row in aqueous_issues).items())),
            source_review_records_preserved=True,
        )
    manifests: dict[Path, Mapping[str, Any]] = {}
    # An active replacement may also occur in several inherited frozen bundles.
    # Its stable human annotation ID represents one label, not repeated votes.
    replacement_ids = {f'{e["target"]["run"]}:manual:{e["replacement"]["annotation"]["id"]}'
                       for e in withdrawals.active_replacements(policy)} if policy else set()
    last_replacement = {region.region_id: index for index, (_, region) in enumerate(pending)
                        if region.region_id in replacement_ids}
    for pending_index, (source, region) in enumerate(pending):
        if region.region_id in last_replacement and last_replacement[region.region_id] != pending_index:
            continue
        if region.region_id in superseded:
            continue
        target = sources[region.run]
        if not preserve_native_revisions and source.asset_manifest_path != target.asset_manifest_path:
            for path in (source.asset_manifest_path, target.asset_manifest_path):
                if path not in manifests:
                    manifests[path] = _read_json(path)
            region = _align_region_timeline(region, manifests[source.asset_manifest_path],
                                             manifests[target.asset_manifest_path])
        key = _geometry_key(region.run, region.payload)
        if preserve_native_revisions:
            # The caller must resolve images through the annotation's original
            # bundle, not the returned last-per-run sources. Keep independent
            # IDs, including conflicting labels, for spatial overlap auditing.
            key = str(source.asset_manifest_path) + ":" + region.region_id
        incumbent = selected_by_geometry.get(key)
        if incumbent is not None and incumbent.phase_class != region.phase_class:
            conflicts.append(dict(geometry_key=key, run=region.run,
                kept=incumbent.region_id, kept_class=incumbent.phase_class,
                candidate=region.region_id, candidate_class=region.phase_class))
        if incumbent is None or _source_priority(region) > _source_priority(incumbent):
            selected_by_geometry[key] = region
            selected_source_by_geometry[key] = source

    regions = sorted(
        selected_by_geometry.values(),
        key=lambda region: (
            region.dataset,
            region.run,
            region.frame_index,
            region.region_id,
        ),
    )
    if region_sources is not None:
        region_sources.clear()
        for key, region in selected_by_geometry.items():
            source = selected_source_by_geometry[key]
            existing = region_sources.setdefault(region.region_id, source)
            if (existing.asset_manifest_path != source.asset_manifest_path
                    or existing.review_root != source.review_root):
                raise ValueError(f"One annotation ID belongs to two native revisions: {region.region_id}")
        if set(region_sources) != {region.region_id for region in regions}:
            raise ValueError("Original native source binding is incomplete")
    return sorted(sources.values(), key=lambda value: value.run), regions, conflicts


def _align_region_timeline(region, original, selected):
    """Keep a reviewed image fixed when a newer timeline prepends a baseline."""
    old_rows, new_rows = original.get("frame_sources"), selected.get("frame_sources")
    if old_rows == new_rows and original.get("frame_count") == selected.get("frame_count"):
        return region
    if not old_rows or not new_rows or not 0 <= region.frame_index < len(old_rows):
        raise ValueError(f"{region.region_id}: cannot verify changed review timeline")
    old = old_rows[region.frame_index]
    elapsed = float(old["elapsed_h"])
    if "elapsed_h" in region.payload and not np.isclose(region.elapsed_h, elapsed, atol=1e-6, rtol=0):
        raise ValueError(f"{region.region_id}: review time differs from its source image")
    matches = [row for row in new_rows
               if np.isclose(float(row["elapsed_h"]), elapsed, atol=1e-6, rtol=0)]
    if len(matches) != 1:
        raise ValueError(f"{region.region_id}: missing or ambiguous image in selected timeline")
    target = matches[0]
    if (old.get("source_name") and target.get("source_name")
            and old["source_name"] != target["source_name"]):
        raise ValueError(f"{region.region_id}: image identity differs between timelines")
    index = int(target["frame_index"])
    payload = dict(region.payload, frame_index=index, elapsed_h=elapsed)
    payload.setdefault("original_frame_index", region.frame_index)
    return replace(region, frame_index=index, elapsed_h=elapsed, payload=payload)


def _copy_file(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_stat = source.stat()
    if not destination.is_file() or destination.stat().st_size != source_stat.st_size:
        temporary = destination.with_suffix(destination.suffix + ".part")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    return {
        "name": destination.name,
        "bytes": int(destination.stat().st_size),
        "sha256": _sha256(destination),
    }


def _raw_review_manifest(manifest: Mapping[str, Any]) -> bool:
    """Whether a native manifest is a RAW photographic review (``raw_photographic_review``)."""
    from darsia.fluidflower.raw_photographic_review import RAW_REVIEW_FORMAT
    return manifest.get("format") == RAW_REVIEW_FORMAT


def _stage_raw_review_run(source: RunSource, cache_root: Path, *,
                          native_revision: bool = False) -> StagedRun:
    """Stage a RAW review run's float inputs, each checked against its SHA-bound record.

    Baseline B and every frame contribute their native log-linear camera RGB
    and exact valid mask. Local copies take the trainer's float-input names,
    so the unchanged decoder reads them. A complete earlier staging is reused
    while the review manifest is unchanged.
    """
    from .raw_frame_context import RawFrameGeometryCohort

    manifest_sha = _sha256(source.asset_manifest_path)
    suffix = ("_" + _canonical_hash(str(source.review_root.resolve()))[:16]
              if native_revision else "")
    local_root = (Path(cache_root) / "assets"
                  / f"{source.dataset}_{source.run}_{manifest_sha[:16]}{suffix}")
    cohort = RawFrameGeometryCohort.from_manifest(source.asset_manifest_path)
    if cohort.run != str(source.run).casefold():
        raise ValueError(f"RAW review belongs to another run: {source.asset_manifest_path}")
    names = ["pre_injection", *(f"frame_{index:03d}" for index in range(len(cohort.rows)))]
    centres = [cohort.native_centre(None), *(cohort.native_centre(index) for index in range(len(cohort.rows)))]
    originals = [dict(path=str(source.asset_manifest_path), sha256=manifest_sha)]
    staged = [local_root / "manifest.json"]
    for name, centre in zip(names, centres, strict=True):
        originals += [centre["log_linear_rgb"], centre["valid_mask"]]
        staged += [local_root / f"{name}.log_linear_rgb.npy", local_root / f"{name}.log_linear_valid_mask.npy"]

    def staged_run(asset_sha: str) -> StagedRun:
        return StagedRun(run=source.run, dataset=source.dataset, root=local_root,
                         asset_manifest_path=staged[0], baseline_path=staged[1],
                         frame_paths=tuple(staged[3::2]), asset_sha256=asset_sha,
                         independent_reference_frames=source.independent_reference_frames,
                         baseline_valid_mask_path=staged[2], frame_valid_mask_paths=tuple(staged[4::2]))

    complete_path = local_root / "staged_assets.json"
    if complete_path.is_file():
        complete = _read_json(complete_path)
        if str(complete.get("source_manifest_sha256")) == manifest_sha and all(path.is_file() for path in staged):
            files = complete.get("files") or []
            if (len(files) != len(originals)
                    or any(row.get("sha256") != original["sha256"] for row, original in zip(files, originals))
                    or (native_revision and any(_sha256(path) != row["sha256"]
                                                for path, row in zip(staged, files)))):
                raise ValueError(f"Staged RAW review inputs differ from their records: {source.asset_manifest_path}")
            return staged_run(str(complete["asset_sha256"]))
    local_root.mkdir(parents=True, exist_ok=True)
    files = []
    for original, local in zip(originals, staged, strict=True):
        row = _copy_file(Path(original["path"]), local)
        if row["sha256"] != original["sha256"]:
            raise ValueError(f"RAW review input differs from its record: {original['path']}")
        files.append(row)
    asset_sha = _canonical_hash(files)
    _write_json(complete_path, {
        "schema": 1, "run": source.run, "dataset": source.dataset, "source_kind": "raw",
        "source_review_root": str(source.review_root), "source_manifest_sha256": manifest_sha,
        "asset_sha256": asset_sha, "files": files,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds")})
    return staged_run(asset_sha)


def _stage_run(source: RunSource, cache_root: Path, *, native_revision: bool = False) -> StagedRun:
    if _raw_review_manifest(_read_json(source.asset_manifest_path)):
        return _stage_raw_review_run(source, cache_root, native_revision=native_revision)
    manifest_sha = _sha256(source.asset_manifest_path)
    source_suffix = (
        "_" + _canonical_hash(str(source.review_root.resolve()))[:16]
        if native_revision else ""
    )
    local_root = (
        Path(cache_root)
        / "assets"
        / f"{source.dataset}_{source.run}_{manifest_sha[:16]}{source_suffix}"
    )
    local_manifest = local_root / "manifest.json"
    local_baseline = local_root / "pre_injection.jpg"
    source_baseline = (
        source.review_root / "native_references" / source.run / "pre_injection.jpg"
    )
    manifest = _read_json(source.asset_manifest_path)
    frame_count = int(manifest.get("frame_count") or 0)
    source_frame_root = source.review_root / "native_frames" / source.run
    local_frames = tuple(
        local_root / f"frame_{index:03d}.jpg" for index in range(frame_count)
    )
    source_frames = tuple(
        source_frame_root / f"frame_{index:03d}.jpg" for index in range(frame_count)
    )
    v5 = (manifest.get("v5_photometry") or {}).get("format") == (
        "ffac-v5-native-source-materialization-v1")
    if v5:
        reference = manifest.get("reference") or {}
        source_linear_baseline = Path(reference["v5_log_linear_rgb_path"])
        source_linear_frames = tuple(Path(row["v5_log_linear_rgb_path"])
                                     for row in manifest["frame_sources"])
        if (len(source_linear_frames) != frame_count
                or source_linear_baseline.resolve()
                    != source_baseline.with_suffix(".log_linear_rgb.npy").resolve()
                or any(path.resolve() != jpg.with_suffix(".log_linear_rgb.npy").resolve()
                       for path, jpg in zip(source_linear_frames, source_frames, strict=True))
                or _sha256(source_linear_baseline) != reference["v5_log_linear_rgb_sha256"]
                or any(_sha256(path) != row["v5_log_linear_rgb_sha256"]
                       for path, row in zip(source_linear_frames,
                                            manifest["frame_sources"], strict=True))):
            raise ValueError("V5 camera-linear sidecars differ from the native manifest")
        local_linear_baseline = local_baseline.with_suffix(".log_linear_rgb.npy")
        local_linear_frames = tuple(path.with_suffix(".log_linear_rgb.npy")
                                    for path in local_frames)
        from darsia.fluidflower.phase_training.log_linear_input import EXACT_VALID_MASK_POLICY
        invalid_policy = manifest.get("v5_float_support_policy")
        if invalid_policy not in (None, EXACT_VALID_MASK_POLICY):
            raise ValueError("Unknown V5 scientific valid-mask policy")
        if invalid_policy == EXACT_VALID_MASK_POLICY:
            source_valid_baseline = Path(reference["v5_log_linear_valid_mask_path"])
            source_valid_frames = tuple(Path(row["v5_log_linear_valid_mask_path"])
                                        for row in manifest["frame_sources"])
            if (source_valid_baseline.resolve() != source_baseline.with_suffix(
                    ".log_linear_valid_mask.npy").resolve()
                    or len(source_valid_frames) != frame_count
                    or any(path.resolve() != jpg.with_suffix(
                        ".log_linear_valid_mask.npy").resolve()
                        for path, jpg in zip(source_valid_frames, source_frames, strict=True))
                    or _sha256(source_valid_baseline) != reference[
                        "v5_log_linear_valid_mask_sha256"]
                    or any(_sha256(path) != row["v5_log_linear_valid_mask_sha256"]
                        for path, row in zip(source_valid_frames,
                                             manifest["frame_sources"], strict=True))):
                raise ValueError("V5 finite RAW masks differ from the native manifest")
            local_valid_baseline = local_baseline.with_suffix(
                ".log_linear_valid_mask.npy")
            local_valid_frames = tuple(path.with_suffix(
                ".log_linear_valid_mask.npy") for path in local_frames)
        else:
            if (reference.get("v5_log_linear_valid_mask_path") is not None
                    or any(row.get("v5_log_linear_valid_mask_path") is not None
                           for row in manifest["frame_sources"])):
                raise ValueError("V5 finite RAW masks require an explicit policy")
            source_valid_baseline = None
            source_valid_frames = ()
            local_valid_baseline = None
            local_valid_frames = ()
        temporal_mode = (manifest.get("v5_temporal_reference") or {}).get("contract")
        extended = temporal_mode == "ffac-v5-source-backed-extended-ring-v1"
        if temporal_mode not in (None, "ffac-v5-native-crop-source-supported-v1",
                                 "ffac-v5-source-backed-extended-ring-v1"):
            raise ValueError("Unrecognized V5 temporal reference mode")
        if extended:
            source_extended_baseline = Path(reference["v5_extended_ring_path"])
            source_extended_frames = tuple(Path(row["v5_extended_ring_path"])
                                           for row in manifest["frame_sources"])
            if (source_extended_baseline.resolve() !=
                    source_baseline.with_suffix(".extended_ring.npz").resolve()
                    or len(source_extended_frames) != frame_count
                    or any(path.resolve() != jpg.with_suffix(
                        ".extended_ring.npz").resolve()
                        for path, jpg in zip(source_extended_frames,
                                             source_frames, strict=True))
                    or _sha256(source_extended_baseline) !=
                    reference["v5_extended_ring_sha256"]
                    or any(_sha256(path) != row["v5_extended_ring_sha256"]
                           for path, row in zip(source_extended_frames,
                                                manifest["frame_sources"], strict=True))):
                raise ValueError("V5 extended RAW rings differ from native manifest")
            local_extended_baseline = local_baseline.with_suffix(".extended_ring.npz")
            local_extended_frames = tuple(path.with_suffix(".extended_ring.npz")
                                          for path in local_frames)
        else:
            source_extended_baseline = None
            source_extended_frames = ()
            local_extended_baseline = None
            local_extended_frames = ()
    else:
        source_linear_baseline = None
        source_linear_frames = ()
        local_linear_baseline = None
        local_linear_frames = ()
        source_extended_baseline = None
        source_extended_frames = ()
        local_extended_baseline = None
        local_extended_frames = ()
        source_valid_baseline = None
        source_valid_frames = ()
        local_valid_baseline = None
        local_valid_frames = ()
    originals = (source.asset_manifest_path, source_baseline, *source_frames,
                 *((source_linear_baseline, *source_linear_frames) if v5 else ()),
                  *((source_valid_baseline, *source_valid_frames)
                    if source_valid_baseline is not None else ()),
                 *((source_extended_baseline, *source_extended_frames)
                   if source_extended_baseline is not None else ()))
    staged = (local_manifest, local_baseline, *local_frames,
              *((local_linear_baseline, *local_linear_frames) if v5 else ()),
               *((local_valid_baseline, *local_valid_frames)
                 if local_valid_baseline is not None else ()),
              *((local_extended_baseline, *local_extended_frames)
                if local_extended_baseline is not None else ()))
    complete_path = local_root / "staged_assets.json"
    if complete_path.is_file():
        complete = _read_json(complete_path)
        required = staged
        if str(complete.get("source_manifest_sha256")) == manifest_sha and all(
            path.is_file() for path in required
        ):
            if native_revision:
                files = complete.get("files") or []
                if len(originals) != len(files) or any(
                    _sha256(path) != row.get("sha256") or
                    _sha256(local) != row.get("sha256")
                    for path, local, row in zip(originals, required, files, strict=True)
                ):
                    raise ValueError(f"Original or staged native revision changed: {source.asset_manifest_path}")
            return StagedRun(
                run=source.run,
                dataset=source.dataset,
                root=local_root,
                asset_manifest_path=local_manifest,
                baseline_path=local_linear_baseline if v5 else local_baseline,
                frame_paths=local_linear_frames if v5 else local_frames,
                asset_sha256=str(complete["asset_sha256"]),
                independent_reference_frames=source.independent_reference_frames,
                baseline_extended_path=local_extended_baseline,
                frame_extended_paths=local_extended_frames,
                baseline_valid_mask_path=local_valid_baseline,
                frame_valid_mask_paths=local_valid_frames,
            )

    local_root.mkdir(parents=True, exist_ok=True)
    files = [
        _copy_file(source.asset_manifest_path, local_manifest),
        _copy_file(source_baseline, local_baseline),
    ]
    for source_frame, local_frame in zip(source_frames, local_frames, strict=True):
        files.append(_copy_file(source_frame, local_frame))
    if v5:
        files.append(_copy_file(source_linear_baseline, local_linear_baseline))
        for source_frame, local_frame in zip(source_linear_frames,
                                             local_linear_frames, strict=True):
            files.append(_copy_file(source_frame, local_frame))
        if source_valid_baseline is not None:
            files.append(_copy_file(source_valid_baseline, local_valid_baseline))
            for source_frame, local_frame in zip(source_valid_frames,
                                                 local_valid_frames, strict=True):
                files.append(_copy_file(source_frame, local_frame))
        if source_extended_baseline is not None:
            files.append(_copy_file(source_extended_baseline,
                                    local_extended_baseline))
            for source_frame, local_frame in zip(source_extended_frames,
                                                 local_extended_frames, strict=True):
                files.append(_copy_file(source_frame, local_frame))
    if native_revision and any(
        _sha256(path) != row["sha256"]
        for path, row in zip(originals, files, strict=True)
    ):
        raise ValueError(f"Native source changed while staging: {source.asset_manifest_path}")
    asset_sha = _canonical_hash(files)
    _write_json(
        complete_path,
        {
            "schema": 1,
            "run": source.run,
            "dataset": source.dataset,
            "source_review_root": str(source.review_root),
            "source_manifest_sha256": manifest_sha,
            "asset_sha256": asset_sha,
            "files": files,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    return StagedRun(
        run=source.run,
        dataset=source.dataset,
        root=local_root,
        asset_manifest_path=local_manifest,
        baseline_path=local_linear_baseline if v5 else local_baseline,
        frame_paths=local_linear_frames if v5 else local_frames,
        asset_sha256=asset_sha,
        independent_reference_frames=source.independent_reference_frames,
        baseline_extended_path=local_extended_baseline,
        frame_extended_paths=local_extended_frames,
        baseline_valid_mask_path=local_valid_baseline,
        frame_valid_mask_paths=local_valid_frames,
    )


def _initialize_extraction_worker(native_threads: int) -> None:
    """Bound each spawned worker's OpenCV and BLAS/OpenMP thread teams."""
    from darsia._opencv_runtime import scientific_thread
    scientific_thread("training-extraction-worker:start")
    from threadpoolctl import threadpool_limits
    cv2.setNumThreads(max(1, int(native_threads)))
    threadpool_limits(limits=max(1, int(native_threads)))


def _prior_frame_index(frame_index: int,
                       independent_reference_frames: Sequence[int]) -> int | None:
    """Match map inference's prior-frame policy for independent controls."""
    excluded = set(int(value) for value in independent_reference_frames)
    prior = (index for index in range(int(frame_index) - 1, -1, -1)
             if index not in excluded)
    return next(prior, None)


def _frame_dependencies(frame_index: int,
                        independent_reference_frames: Sequence[int]) -> tuple[int, ...]:
    prior = _prior_frame_index(frame_index, independent_reference_frames)
    return ((int(frame_index),) if prior is None else (int(frame_index), prior))


def _require_v5_float_rgb(value: np.ndarray, *, role: str) -> None:
    """Never let a display JPEG or uint8 array become V5 model input."""
    if (not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32")
            or value.ndim != 3 or value.shape[2] != 3):
        raise ValueError(f"V5 {role} model input must be native float32 RGB")


def _extract_run_features(task: ExtractionTask, *, feature_augmenter=None) -> dict[str, Any]:
    from darsia._opencv_runtime import POLICY, scientific_thread, completed_frame, require_ipp_off
    ipp_start = scientific_thread("training-extraction:start")
    cv2.setNumThreads(max(1, int(task.opencv_threads)))
    manifest = _read_json(task.staged.asset_manifest_path)
    v5 = (manifest.get("v5_photometry") or {}).get("format") == (
        "ffac-v5-native-source-materialization-v1")
    frame_geometry = None
    track1_photo = bool(manifest.get("v5_track1_photo_geometry"))
    temporal_reference = None
    masked_temporal = False
    raw = _raw_review_manifest(manifest)
    prefix = "raw" if raw else "v5"
    if raw:
        # A RAW photographic review: SHA-bound native records and the RAW geometry cohort.
        if task.radiometry_profile is not None:
            raise ValueError("RAW camera-linear optical input cannot use legacy radiometry")
        from darsia.fluidflower.phase_training.raw_frame_context import (
            TEMPORAL_CONTRACT, RawFrameGeometryCohort)
        frame_geometry = RawFrameGeometryCohort.from_manifest(task.staged.asset_manifest_path)
        temporal_reference, masked_temporal, extended_temporal = TEMPORAL_CONTRACT, True, False
        for index in sorted({i for region in task.regions
                             for i in _frame_dependencies(
                                 int(region.frame_index),
                                 task.staged.independent_reference_frames)}):
            frame_geometry.frame(index)
    if v5:
        if task.radiometry_profile is not None:
            raise ValueError("V5 camera-linear optical input cannot use legacy radiometry")
        if track1_photo:
            from darsia.fluidflower.phase_training.photo_context import (
                Track1FramePhotoCohort,
            )
            frame_geometry = Track1FramePhotoCohort(task.staged.asset_manifest_path)
        else:
            from darsia.fluidflower.phase_training.frame_context import (
                V5FrameGeometryCohort,
            )
            frame_geometry = V5FrameGeometryCohort(task.staged.asset_manifest_path)
        temporal_reference = (manifest.get("v5_temporal_reference") or {}).get("contract")
        masked_temporal = temporal_reference in (
            "ffac-v5-native-crop-source-supported-v1",
            "ffac-v5-source-backed-extended-ring-v1")
        extended_temporal = temporal_reference == "ffac-v5-source-backed-extended-ring-v1"
        if temporal_reference not in (None, "ffac-v5-native-crop-source-supported-v1",
                                  "ffac-v5-source-backed-extended-ring-v1"):
            raise ValueError("Unknown V5 temporal reference contract")
        # Cache hits must not bypass the external, SHA-bound frame receipts.
        for index in sorted({i for region in task.regions
                             for i in _frame_dependencies(
                                 int(region.frame_index),
                                 task.staged.independent_reference_frames)}):
            frame_geometry.frame(index)
    invalid_policy = manifest.get("v5_float_support_policy") if v5 else None

    def staged_rgb(path: Path, record: Mapping[str, Any],
                   valid_mask_path: Path | None) -> np.ndarray:
        rgb = _load_rgb(
            path, content_box,
            expected_sha256=record["v5_log_linear_rgb_sha256"],
            invalid_policy=invalid_policy,
            valid_mask_path=valid_mask_path,
            valid_mask_sha256=record.get("v5_log_linear_valid_mask_sha256"),
        )
        _require_v5_float_rgb(rgb, role="decoded")
        return rgb

    def raw_rgb(index: int | None, path: Path, valid_mask_path: Path) -> np.ndarray:
        # The staged copy must still equal its record's array and exact valid mask.
        from darsia.fluidflower.phase_training.log_linear_input import EXACT_VALID_MASK_POLICY
        centre = frame_geometry.native_centre(index)
        rgb = _load_rgb(
            path, content_box,
            expected_sha256=centre["log_linear_rgb"]["sha256"],
            invalid_policy=EXACT_VALID_MASK_POLICY,
            valid_mask_path=valid_mask_path,
            valid_mask_sha256=centre["valid_mask"]["sha256"],
        )
        _require_v5_float_rgb(rgb, role="decoded")
        return rgb
    if task.output_path.is_file() and task.metadata_path.is_file():
        metadata = _read_json(task.metadata_path)
        if str(metadata.get("cache_key")) == task.cache_key:
            if metadata.get("opencv_ipp_policy") != POLICY:
                raise ValueError("Feature cache predates the IPP-off policy; preserve it and select a new cache key")
            if not (v5 or raw) and any("training_sampling" in region.payload for region in task.regions):
                from .gas_negative_sampling import sample_region
                native = _read_json(task.staged.asset_manifest_path)
                checked = {}
                for region in task.regions:
                    if "training_sampling" in region.payload:
                        mask = _annotation_mask(region.payload, canvas_wh=native["canvas_wh"],
                                                content_box=native["content_box_xywh"])
                        sample_region(region, mask, task.staged.asset_manifest_path,
                                      limit=max(1, int(task.pixels_per_region)), cache=checked)
            if _sha256(task.output_path) != metadata.get("features_sha256"):
                raise ValueError(f"Corrupt feature cache for {task.staged.dataset}/{task.staged.run}")
            if task.radiometry_profile is not None and (metadata.get("radiometry") or {}).get("profile") != task.radiometry_profile:
                raise ValueError("Feature cache belongs to a different radiometric profile")
            return {**metadata, "cache_hit": True, "seconds_current": 0.0,
                    "current_cache_read_opencv_ipp": require_ipp_off("training-extraction:cache-read")}

    started = time.perf_counter()
    canvas_wh = tuple(int(value) for value in manifest["canvas_wh"])
    content_box = tuple(int(value) for value in manifest["content_box_xywh"])
    radiometry = None
    if task.radiometry_profile is not None:
        from .radiometry import RunRadiometry
        radiometry = RunRadiometry(
            task.radiometry_profile, series_id=task.staged.dataset,
            run_id=task.staged.run, baseline_path=task.staged.baseline_path,
            content_box=content_box, cache_root=task.staged.root / "uniform24",
            source_manifest=task.staged.asset_manifest_path,
        )
        baseline = radiometry.load(task.staged.baseline_path, "pre_injection")
    elif raw:
        baseline = raw_rgb(None, task.staged.baseline_path, task.staged.baseline_valid_mask_path)
    else:
        baseline = (None if v5 and extended_temporal else
                    staged_rgb(task.staged.baseline_path, manifest["reference"],
                               task.staged.baseline_valid_mask_path)
                    if v5 else _load_rgb(task.staged.baseline_path, content_box))
    baseline_source_valid = None
    baseline_origin = (0, 0)
    if v5 and extended_temporal:
        from darsia.fluidflower.phase_training.log_linear_input import decode_extended_log_linear
        reference = manifest["reference"]
        if task.staged.baseline_extended_path is None:
            raise ValueError("V5 source-backed temporal baseline ring was not staged")
        baseline, baseline_origin, baseline_source_valid = decode_extended_log_linear(
            task.staged.baseline_path, task.staged.baseline_extended_path,
            core_sha256=reference["v5_log_linear_rgb_sha256"],
            ring_sha256=reference["v5_extended_ring_sha256"],
            content_shape_hw=(content_box[3], content_box[2]),
            pad_px=int(reference["v5_extended_pad_px"]),
            invalid_policy=invalid_policy,
            valid_mask_path=task.staged.baseline_valid_mask_path,
            valid_mask_sha256=reference.get("v5_log_linear_valid_mask_sha256"))
    if (v5 or raw) and masked_temporal:
        if not extended_temporal:
            if task.staged.baseline_valid_mask_path is not None:
                baseline_source_valid = np.load(
                    task.staged.baseline_valid_mask_path, allow_pickle=False,
                    mmap_mode="r")
            else:
                baseline_log = np.load(task.staged.baseline_path, allow_pickle=False,
                                       mmap_mode="r")
                baseline_source_valid = np.isfinite(baseline_log).all(axis=0)
            if baseline_source_valid.shape != baseline.shape[:2]:
                raise ValueError("V5 baseline validity mask differs from native image")
    by_frame: dict[int, list[tuple[int, ReviewedRegion]]] = defaultdict(list)
    for local_index, region in enumerate(task.regions):
        by_frame[int(region.frame_index)].append((local_index, region))

    feature_chunks: list[np.ndarray] = []
    owner_chunks: list[np.ndarray] = []
    frame_rows: list[dict[str, Any]] = []
    sampling_sources = {}
    for frame_index in sorted(by_frame):
        ipp_before = scientific_thread("training-extraction:frame-before")
        current = (
            raw_rgb(frame_index, task.staged.frame_paths[frame_index],
                    task.staged.frame_valid_mask_paths[frame_index]) if raw else
            staged_rgb(task.staged.frame_paths[frame_index],
                       manifest["frame_sources"][frame_index],
                       (task.staged.frame_valid_mask_paths[frame_index]
                        if task.staged.frame_valid_mask_paths else None))
            if v5 else
            (_load_rgb(task.staged.frame_paths[frame_index], content_box)
             if radiometry is None else radiometry.load(
                 task.staged.frame_paths[frame_index], f"frame_{frame_index:03d}"))
        )
        previous = None
        previous_origin = (0, 0)
        previous_source_valid = None
        previous_index = (_prior_frame_index(
            frame_index, task.staged.independent_reference_frames)
            if (v5 or raw) else frame_index - 1 if frame_index > 0 else None)
        if previous_index is not None:
            previous = (raw_rgb(previous_index, task.staged.frame_paths[previous_index],
                                task.staged.frame_valid_mask_paths[previous_index]) if raw else
                        None if v5 and extended_temporal else
                staged_rgb(task.staged.frame_paths[previous_index],
                           manifest["frame_sources"][previous_index],
                           (task.staged.frame_valid_mask_paths[previous_index]
                            if task.staged.frame_valid_mask_paths else None))
                if v5 else
                (_load_rgb(task.staged.frame_paths[previous_index], content_box)
                 if radiometry is None else radiometry.load(
                     task.staged.frame_paths[previous_index],
                     f"frame_{previous_index:03d}"))
            )
            if v5 and extended_temporal:
                from darsia.fluidflower.phase_training.log_linear_input import decode_extended_log_linear
                if len(task.staged.frame_extended_paths) != len(task.staged.frame_paths):
                    raise ValueError("V5 source-backed temporal frame rings were not staged")
                previous_record = manifest["frame_sources"][previous_index]
                previous, previous_origin, previous_source_valid = decode_extended_log_linear(
                    task.staged.frame_paths[previous_index],
                    task.staged.frame_extended_paths[previous_index],
                    core_sha256=previous_record["v5_log_linear_rgb_sha256"],
                    ring_sha256=previous_record["v5_extended_ring_sha256"],
                    content_shape_hw=(content_box[3], content_box[2]),
                    pad_px=int(previous_record["v5_extended_pad_px"]),
                    invalid_policy=invalid_policy,
                    valid_mask_path=(task.staged.frame_valid_mask_paths[previous_index]
                                     if task.staged.frame_valid_mask_paths else None),
                    valid_mask_sha256=previous_record.get(
                        "v5_log_linear_valid_mask_sha256"))
        current_source_valid = None
        if (v5 or raw) and masked_temporal:
            if task.staged.frame_valid_mask_paths:
                current_source_valid = np.load(
                    task.staged.frame_valid_mask_paths[frame_index],
                    allow_pickle=False, mmap_mode="r")
            else:
                current_log = np.load(task.staged.frame_paths[frame_index],
                                      allow_pickle=False, mmap_mode="r")
                current_source_valid = np.isfinite(current_log).all(axis=0)
            if current_source_valid.shape != current.shape[:2]:
                raise ValueError("V5 current validity mask differs from native image")
            if previous_index is not None and not extended_temporal:
                if task.staged.frame_valid_mask_paths:
                    previous_source_valid = np.load(
                        task.staged.frame_valid_mask_paths[previous_index],
                        allow_pickle=False, mmap_mode="r")
                else:
                    previous_log = np.load(task.staged.frame_paths[previous_index],
                                           allow_pickle=False, mmap_mode="r")
                    previous_source_valid = np.isfinite(previous_log).all(axis=0)
                if previous_source_valid.shape != current.shape[:2]:
                    raise ValueError("V5 previous validity mask differs from native image")

        geometry_binding = None
        current_domain = None
        certified_sand = None
        uncertain_photo_boundary = None
        if frame_geometry is not None:
            from darsia.fluidflower.phase_training.temporal_alignment import align_temporal_inputs
            from darsia.fluidflower.phase_training.frame_context import compute_feature_support
            if track1_photo:
                certified_sand, uncertain_photo_boundary, geometry_binding = (
                    frame_geometry.photo(frame_index))
                current_domain = np.ones(certified_sand.shape, dtype=bool)
                if not masked_temporal:
                    raise ValueError("Track1 photo-only optical fit needs source support")
            else:
                context_fields = ("observable_phase_domain",)
                if masked_temporal:
                    context_fields += ("photo_certified_sand_domain",
                                       "photo_boundary_uncertain_domain")
                context, geometry_binding = frame_geometry.context(
                    frame_index, fields=context_fields)
                current_domain = context["observable_phase_domain"]
                if masked_temporal:
                    certified_sand = context["photo_certified_sand_domain"]
                    uncertain_photo_boundary = context["photo_boundary_uncertain_domain"]
            baseline_for_frame, previous_for_frame, alignment = align_temporal_inputs(
                baseline=baseline, current=current, previous=previous,
                h_baseline_to_current=frame_geometry.H_v5_B_to_frame(frame_index),
                h_baseline_to_previous=(None if previous_index is None else
                                        frame_geometry.H_v5_B_to_frame(previous_index)),
                required_valid_mask=(None if masked_temporal else current_domain),
                baseline_source_valid_mask=baseline_source_valid,
                previous_source_valid_mask=previous_source_valid,
                baseline_origin_xy=baseline_origin,
                previous_origin_xy=previous_origin,
                return_support_mask=masked_temporal,
            )
            if masked_temporal:
                temporal_valid = alignment.pop("joint_valid_mask")
                if abs(float(task.texture_sigma) - 2.0) > 1e-9:
                    raise ValueError("V5 masked optical feature contract requires sigma=2")
                feature_valid, source_valid = compute_feature_support(
                    current_valid=current_domain & current_source_valid,
                    baseline_valid=baseline_source_valid,
                    previous_valid=previous_source_valid,
                    h_baseline_to_current=frame_geometry.H_v5_B_to_frame(
                        frame_index),
                    h_baseline_to_previous=(None if previous_index is None else
                        frame_geometry.H_v5_B_to_frame(previous_index)),
                    baseline_origin_xy=baseline_origin,
                    previous_origin_xy=previous_origin)
                if not np.array_equal(source_valid, current_domain &
                                      current_source_valid & temporal_valid):
                    raise ValueError("V5 trainer and map source-support footprints differ")
                alignment["raw_supported_observation_pixels"] = int(
                    np.count_nonzero(source_valid))
                alignment["supported_observation_pixels"] = int(
                    np.count_nonzero(feature_valid))
                alignment["supported_observation_fraction"] = float(
                    np.mean(feature_valid))
                alignment["support_policy"] = temporal_reference
                alignment["feature_support_erosion_radius_px"] = 8
            else:
                feature_valid = current_domain
        else:
            baseline_for_frame, previous_for_frame = baseline, previous
            alignment = None
            feature_valid = None
        if v5 or raw:
            _require_v5_float_rgb(current, role="current")
            _require_v5_float_rgb(baseline_for_frame, role="aligned baseline")
            if previous_for_frame is not None:
                _require_v5_float_rgb(previous_for_frame,
                                      role="aligned previous")

        row_chunks: list[np.ndarray] = []
        column_chunks: list[np.ndarray] = []
        owners: list[np.ndarray] = []
        support_rows: list[dict[str, Any]] = []
        for local_index, region in by_frame[frame_index]:
            if v5 or raw:
                from .review_sampling import sampling_receipt
                review_sampling = sampling_receipt(region.phase_class, region.payload)
                if review_sampling["role"] in (
                        "candidate_issue_evaluation_only", "evaluation_only"):
                    raise ValueError(
                        f"{region.region_id}: evaluation-only review reached V5 fitting")
            else:
                review_sampling = None
            mask = _annotation_mask(
                region.payload,
                canvas_wh=canvas_wh,
                content_box=content_box,
            )
            if current_domain is not None and np.any(mask & ~current_domain):
                raise ValueError(
                    f"{region.region_id}: V5 reviewed annotation leaves this frame's "
                    "source-bound observable phase domain")
            original_pixels = int(np.count_nonzero(mask))
            if masked_temporal:
                sand_pixels = int(np.count_nonzero(mask & certified_sand))
                photo_uncertain_pixels = int(np.count_nonzero(
                    mask & uncertain_photo_boundary))
                retained_pixels = int(np.count_nonzero(
                    mask & certified_sand & feature_valid))
                raw_retained_pixels = int(np.count_nonzero(
                    mask & certified_sand & source_valid))
                if sand_pixels == 0:
                    raise ValueError(
                        f"{region.region_id}: no independently certified sand pixels; "
                        "preserve annotation for review, but do not invent training samples")
                if retained_pixels < .90 * sand_pixels:
                    raise ValueError(
                        f"{region.region_id}: V5 source-supported optical evidence "
                        f"retains {retained_pixels}/{sand_pixels} certified sand pixels "
                        f"after 8-pixel texture support guard (<90%); "
                        f"whole annotation={original_pixels}, raw={raw_retained_pixels}")
            if not (v5 or raw) and "training_sampling" in region.payload:
                from .gas_negative_sampling import sample_region
                rows, columns = sample_region(region, mask, task.staged.asset_manifest_path,
                    limit=max(1, int(task.pixels_per_region)), cache=sampling_sources)
                if masked_temporal:
                    selected = feature_valid[rows, columns]
                    rows, columns = rows[selected], columns[selected]
            else:
                sampling_mask = (mask & certified_sand & feature_valid
                                 if masked_temporal else mask)
                rows, columns = _sample_mask(sampling_mask,
                                             limit=max(1, int(task.pixels_per_region)),
                                             annotation_id=region.region_id)
            if not rows.size:
                raise ValueError(f"{region.region_id}: no source-supported optical samples")
            if masked_temporal:
                support_rows.append(dict(
                    region_id=region.region_id,
                    phase_class=region.phase_class,
                    review_sampling=review_sampling,
                    original_pixels=original_pixels,
                    photo_certified_sand_pixels=sand_pixels,
                    photo_boundary_uncertain_pixels=photo_uncertain_pixels,
                    off_certified_sand_pixels=original_pixels-sand_pixels,
                    source_supported_pixels=raw_retained_pixels,
                    feature_supported_pixels=retained_pixels,
                    retained_fraction_of_certified_sand=float(
                        retained_pixels / sand_pixels),
                    retained_fraction_of_whole_annotation=float(
                        retained_pixels / original_pixels),
                    sampled_pixels=int(rows.size),
                    feature_support_erosion_radius_px=8,
                ))
            row_chunks.append(rows)
            column_chunks.append(columns)
            owners.append(np.full(rows.size, local_index, dtype=np.int32))
        all_rows = np.concatenate(row_chunks)
        all_columns = np.concatenate(column_chunks)
        extracted = feature_samples(
            current,
            baseline_for_frame,
            all_rows,
            all_columns,
            texture_sigma=float(task.texture_sigma),
            previous_rgb=previous_for_frame,
            feature_contract=task.feature_contract,
        )
        augmentation = None
        if feature_augmenter is not None:
            extra, usable, augmentation = feature_augmenter(
                frame_index=frame_index, rows=all_rows, columns=all_columns,
                current_source_valid=current_source_valid,
                baseline_source_valid=baseline_source_valid,
                frame_geometry=frame_geometry,
            )
            usable = np.asarray(usable)
            if (usable.dtype != np.bool_ or usable.shape != (all_rows.size,)
                    or np.asarray(extra).dtype != np.float32
                    or np.asarray(extra).shape != (
                        all_rows.size, len(feature_augmenter.feature_names))):
                raise ValueError("Optical feature augmentation has a malformed sample contract")
            # All gas variants use the identical supported rows. Never turn an
            # unavailable flat-field channel into an invented negative example.
            extracted = np.column_stack((extracted[usable], extra[usable]))
            combined_owners = np.concatenate(owners)[usable]
            if not extracted.shape[0] or not np.isfinite(extracted).all():
                raise ValueError("Augmented optical samples have no finite supported evidence")
            if set(combined_owners.tolist()) != set(np.concatenate(owners).tolist()):
                raise ValueError("Feature augmentation removed every pixel of a reviewed region")
        else:
            combined_owners = np.concatenate(owners)
        if (v5 or raw) and (not isinstance(extracted, np.ndarray)
                            or extracted.dtype != np.dtype("float32")):
            raise ValueError("V5 optical model features must be float32")
        feature_chunks.append(np.asarray(extracted, dtype=np.float32))
        owner_chunks.append(combined_owners)
        frame_rows.append(
            {
                "frame_index": int(frame_index),
                "regions": len(by_frame[frame_index]),
                "pixels": int(all_rows.size),
                "opencv_ipp": completed_frame(ipp_before, "training-extraction:frame-after"),
                **({"feature_augmentation": augmentation} if augmentation is not None else {}),
                **({} if geometry_binding is None else {
                    f"{prefix}_frame_geometry_receipt_sha256": geometry_binding["receipt_sha256"],
                    f"{prefix}_previous_frame_index": previous_index,
                    f"{prefix}_previous_geometry_receipt_sha256": (
                        None if previous_index is None else
                        frame_geometry.frame(previous_index)["receipt_sha256"]),
                    f"{prefix}_temporal_alignment": alignment,
                    f"{prefix}_temporal_reference_contract": temporal_reference,
                    f"{prefix}_region_source_support": support_rows,
                }),
            }
        )
        del current, previous, baseline_for_frame, previous_for_frame
        del extracted, all_rows, all_columns
        gc.collect()

    matrix = np.vstack(feature_chunks)
    region_indices = np.concatenate(owner_chunks)
    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = task.output_path.with_suffix(task.output_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            features=np.asarray(matrix, dtype=np.float32),
            region_indices=np.asarray(region_indices, dtype=np.int32),
            feature_names=np.asarray((*task.feature_contract.feature_names,
                *(feature_augmenter.feature_names if feature_augmenter is not None else ()))),
        )
    require_ipp_off("training-extraction:before-cache-publication")
    temporary.replace(task.output_path)
    metadata = {
        "schema": 1,
        "run": task.staged.run,
        "dataset": task.staged.dataset,
        "cache_key": task.cache_key,
        "radiometry": None if radiometry is None else radiometry.report(),
        "asset_sha256": task.staged.asset_sha256,
        "features_path": str(task.output_path),
        "features_sha256": _sha256(task.output_path),
        "region_count": len(task.regions),
        "pixel_rows": int(matrix.shape[0]),
        "feature_columns": int(matrix.shape[1]),
        "pixels_per_region": int(task.pixels_per_region),
        "texture_sigma": float(task.texture_sigma),
        "opencv_threads": int(task.opencv_threads),
        "opencv_ipp_policy": POLICY,
        "opencv_ipp": dict(start=ipp_start, completed=require_ipp_off("training-extraction:completed")),
        "frames": frame_rows,
        "seconds": float(time.perf_counter() - started),
        "cache_hit": False,
        "seconds_current": float(time.perf_counter() - started),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_json(task.metadata_path, metadata)
    return metadata


def _feature_cache_key(
    staged: StagedRun,
    regions: Sequence[ReviewedRegion],
    *,
    pixels_per_region: int,
    texture_sigma: float,
    radiometry_profile: dict[str, Any] | None = None,
    feature_contract: OpticalFeatureContract = MAXIMAL_CONTRACT,
) -> str:
    from darsia._opencv_runtime import POLICY
    native = (_read_json(staged.asset_manifest_path)
              if getattr(staged, "asset_manifest_path", None) is not None else {})
    v5 = (native.get("v5_photometry") or {}).get("format") == (
        "ffac-v5-native-source-materialization-v1")
    raw = _raw_review_manifest(native)
    v5_geometry = None
    if v5:
        from darsia.fluidflower.phase_training import frame_context as ffac_v5_frame_context, temporal_alignment as ffac_v5_temporal_alignment
        track1_photo = bool(native.get("v5_track1_photo_geometry"))
        if track1_photo:
            from darsia.fluidflower.phase_training import photo_context as ffac_v5_track1_photo_context
            from . import log_linear_input as v5_linear_input
            cohort = ffac_v5_track1_photo_context.Track1FramePhotoCohort(
                staged.asset_manifest_path)
        else:
            cohort = ffac_v5_frame_context.V5FrameGeometryCohort(
                staged.asset_manifest_path)
        indices = sorted({i for region in regions
                          for i in _frame_dependencies(
                              int(region.frame_index),
                              staged.independent_reference_frames)})
        v5_geometry = {
            "cohort_manifest_sha256": cohort.native_manifest_sha256,
            "source_manifest_sha256": cohort.source_manifest_sha256,
            "frame_receipts": [
                {"index": index, "sha256": cohort.frame(index)["receipt_sha256"]}
                for index in indices
            ],
            "frame_context_code_sha256": _sha256(Path(ffac_v5_frame_context.__file__)),
            "temporal_alignment_code_sha256": _sha256(
                Path(ffac_v5_temporal_alignment.__file__)),
        }
        if track1_photo:
            v5_geometry.update(
                cohort_contract=ffac_v5_track1_photo_context.COHORT_CONTRACT,
                track1_photo_context_code_sha256=_sha256(
                    Path(ffac_v5_track1_photo_context.__file__)),
                float_input_code_sha256=_sha256(Path(v5_linear_input.__file__)),
            )
    region_contract = [
        {
            "region_id": region.region_id,
            "resolved_native_frame_index": region.frame_index,
            "phase_class": region.phase_class,
            "confidence": region.confidence,
            "source": region.source,
            "geometry": _geometry_key(region.run, region.payload),
        }
        for region in regions
    ]
    for item, region in zip(region_contract, regions):
        if v5 or raw:
            from .review_sampling import sampling_receipt
            item["sampling"] = sampling_receipt(region.phase_class, region.payload)
        elif "training_sampling" in region.payload:
            from . import gas_negative_sampling
            from ..tools.phase_optical_cache_compatibility import canonical_implementation
            item["sampling"] = gas_negative_sampling.contract(region.payload["training_sampling"])
            item["sampling_runtime_sha256"] = canonical_implementation(
                gas_negative_sampling.__name__, _sha256(Path(gas_negative_sampling.__file__)))
    payload = (
        {
            "asset_sha256": staged.asset_sha256,
            "v5_independent_reference_frames": (
                getattr(staged, "independent_reference_frames", ()) if v5 else None),
            "regions": region_contract,
            "pixels_per_region": int(pixels_per_region),
            "texture_sigma": float(texture_sigma),
            "feature_contract": feature_contract.to_dict(),
            "v5_review_sampling_code_sha256": (
                _sha256(PACKAGE_ROOT / "review_sampling.py") if v5 else None),
            "optical_phase_features_sha256": _sha256(
                FLUIDFLOWER_ROOT / "phase_features.py"
            ),
            "dual_phase_signal_sha256": _sha256(PACKAGE_ROOT / "model.py"),
            "annotation_geometry_sha256": _sha256(PACKAGE_ROOT / "review_regions.py"),
            # Model fitting, validation, reporting, and backend changes must
            # not invalidate expensive optical feature extraction.
            "feature_extraction_contract_version": 1,
            "opencv_ipp_policy": POLICY,
            "radiometry_profile": radiometry_profile,
            "radiometry_runtime": _radiometry_runtime(radiometry_profile),
            "v5_frame_geometry": v5_geometry,
        }
    )
    if raw:
        payload.update(raw_frame_geometry=_raw_geometry_identity(staged, regions),
                       raw_independent_reference_frames=list(staged.independent_reference_frames),
                       raw_review_sampling_code_sha256=_sha256(PACKAGE_ROOT / "review_sampling.py"))
    return _canonical_hash(payload)


def _raw_geometry_identity(staged: StagedRun, regions: Sequence[ReviewedRegion]) -> dict[str, Any]:
    """The RAW frame-geometry identity of a feature cache: receipts, training context and code."""
    from darsia.fluidflower.phase_training import frame_context, raw_frame_context, temporal_alignment

    cohort = raw_frame_context.RawFrameGeometryCohort.from_manifest(staged.asset_manifest_path)
    indices = sorted({i for region in regions
                      for i in _frame_dependencies(int(region.frame_index),
                                                   staged.independent_reference_frames)})
    return {
        "cohort_manifest_sha256": cohort.native_manifest_sha256,
        "frame_receipts": [{"index": index, "sha256": cohort.frame(index)["receipt_sha256"]}
                           for index in indices],
        "training_context": cohort.context_binding,
        "temporal_reference_contract": raw_frame_context.TEMPORAL_CONTRACT,
        "raw_frame_context_code_sha256": _sha256(Path(raw_frame_context.__file__)),
        "frame_context_code_sha256": _sha256(Path(frame_context.__file__)),
        "temporal_alignment_code_sha256": _sha256(Path(temporal_alignment.__file__)),
    }


def _native_revision_task_groups(
    regions: Sequence[ReviewedRegion],
    origin_sources: Mapping[str, RunSource],
) -> list[tuple[RunSource, tuple[ReviewedRegion, ...]]]:
    """Bind every region to its reviewed native canvas, without moving labels.

    The run on a ReviewedRegion remains the physical experiment for whole-run
    cross-validation. The source identity is only used for image extraction.
    """
    fit_ids = [region.region_id for region in regions]
    if len(fit_ids) != len(set(fit_ids)) or not set(fit_ids).issubset(origin_sources):
        raise ValueError("Each reviewed region needs exactly one original native source")
    grouped: dict[tuple[str, str, str], tuple[RunSource, list[ReviewedRegion]]] = {}
    manifests: dict[Path, dict[str, Any]] = {}
    for region in regions:
        source = origin_sources[region.region_id]
        if (source.run, source.dataset) != (region.run, region.dataset):
            raise ValueError(f"Wrong native source for {region.region_id}")
        manifest_path = source.asset_manifest_path.resolve()
        if manifest_path not in manifests:
            manifest = _read_json(manifest_path)
            width, height = (int(value) for value in manifest["canvas_wh"])
            x, y, crop_width, crop_height = (int(value) for value in manifest["content_box_xywh"])
            if (width <= 0 or height <= 0 or x < 0 or y < 0 or crop_width <= 0
                    or crop_height <= 0 or x + crop_width > width or y + crop_height > height
                    or review_content_shape(manifest) != (crop_height, crop_width)
                    or len(timeline_rows(manifest)) != int(manifest["frame_count"])):
                raise ValueError(f"Invalid reviewed native geometry or timeline: {manifest_path}")
            manifests[manifest_path] = manifest
        manifest = manifests[manifest_path]
        frame = int(region.frame_index)
        rows = timeline_rows(manifest)
        if not 0 <= frame < len(rows):
            raise ValueError(f"Review frame is outside its original source: {region.region_id}")
        if "elapsed_h" in region.payload and not np.isclose(
            float(rows[frame]["elapsed_h"]), float(region.elapsed_h), atol=1e-5, rtol=0
        ):
            matches = [index for index, row in enumerate(rows) if np.isclose(
                float(row["elapsed_h"]), float(region.elapsed_h), atol=1e-5, rtol=0)]
            if len(matches) != 1:
                raise ValueError(f"No unique original reviewed image for {region.region_id}")
            # A reference-frame insertion can change only the array index.
            # Geometry, source review root and label payload remain untouched.
            region = replace(region, frame_index=matches[0])
        key = (region.dataset, region.run, str(manifest_path))
        if key not in grouped:
            grouped[key] = (source, [])
        elif grouped[key][0].review_root.resolve() != source.review_root.resolve():
            raise ValueError(f"Same native manifest maps to two image roots: {manifest_path}")
        elif (grouped[key][0].independent_reference_frames !=
              source.independent_reference_frames):
            raise ValueError(f"Same native manifest has conflicting independent controls: {manifest_path}")
        grouped[key][1].append(region)
    return [(source, tuple(sorted(rows, key=lambda r: (r.frame_index, r.region_id))))
            for _, (source, rows) in sorted(grouped.items())]


def _load_feature_caches(
    sources: Sequence[RunSource],
    regions: Sequence[ReviewedRegion],
    *,
    cache_root: Path,
    stage_workers: int,
    extraction_workers: int,
    opencv_threads: int,
    pixels_per_region: int,
    texture_sigma: float,
    radiometry_profile: dict[str, Any] | None = None,
    feature_contract: OpticalFeatureContract = MAXIMAL_CONTRACT,
    origin_sources: Mapping[str, RunSource] | None = None,
    fit_excluded_runs: Sequence[str] = (),
    fit_allowed_review_ids: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[ReviewedRegion], list[dict[str, Any]]]:
    excluded = {str(run).casefold() for run in fit_excluded_runs}
    if (fit_allowed_review_ids is not None and
            any(region.run.casefold() in excluded and
                region.region_id in fit_allowed_review_ids for region in regions)):
        raise ValueError("Excluded chemistry source is in the Track1 fit partition")
    fit_regions = [region for region in regions if region.run.casefold() not in excluded
                   and (fit_allowed_review_ids is None or
                        region.region_id in fit_allowed_review_ids)]
    evaluation_only_regions = [region for region in regions
                               if region.run.casefold() in excluded or
                               (fit_allowed_review_ids is not None and
                                region.region_id not in fit_allowed_review_ids)]
    if not fit_regions:
        raise ValueError("V5 fit exclusion leaves no optical training regions")
    revision_groups = (_native_revision_task_groups(fit_regions, origin_sources)
                       if origin_sources is not None else None)
    stage_sources = ([source for source, _ in revision_groups]
                     if revision_groups is not None else
                     [source for source in sources if source.run.casefold() not in excluded])
    with ThreadPoolExecutor(max_workers=max(1, int(stage_workers))) as executor:
        future_to_source = {
            executor.submit(_stage_run, source, cache_root,
                            native_revision=revision_groups is not None): source
            for source in stage_sources
        }
        staged_by_source: dict[tuple[str, str, str], StagedRun] = {}
        for future in as_completed(future_to_source):
            result = future.result()
            source = future_to_source[future]
            staged_by_source[(source.dataset, source.run, str(source.asset_manifest_path.resolve()))] = result
            _status(f"Staged native corrected assets for {result.run}")

    tasks: list[ExtractionTask] = []
    task_groups = (revision_groups if revision_groups is not None else
                   [(source, tuple(region for region in fit_regions if region.run == source.run))
                    for source in stage_sources])
    for source, run_regions in task_groups:
        if not run_regions:
            continue
        staged_run = staged_by_source[(source.dataset, source.run,
                                       str(source.asset_manifest_path.resolve()))]
        cache_key = _feature_cache_key(
            staged_run,
            run_regions,
            pixels_per_region=pixels_per_region,
            texture_sigma=texture_sigma,
            radiometry_profile=radiometry_profile,
            feature_contract=feature_contract,
        )
        run_cache = (Path(cache_root) / "features" / staged_run.run /
                     (staged_run.asset_sha256[:16] if revision_groups is not None else "") /
                     cache_key[:20])
        tasks.append(
            ExtractionTask(
                staged=staged_run,
                regions=run_regions,
                output_path=run_cache / "samples.npz",
                metadata_path=run_cache / "metadata.json",
                cache_key=cache_key,
                pixels_per_region=int(pixels_per_region),
                texture_sigma=float(texture_sigma),
                opencv_threads=int(opencv_threads),
                radiometry_profile=radiometry_profile,
                feature_contract=feature_contract,
            )
        )

    context = multiprocessing.get_context("spawn")
    reports: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=max(1, min(int(extraction_workers), len(tasks))),
        mp_context=context,
        initializer=_initialize_extraction_worker,
        initargs=(max(1, int(opencv_threads)),),
    ) as executor:
        future_to_task = {
            executor.submit(_extract_run_features, task): task for task in tasks
        }
        for future in as_completed(future_to_task):
            report = future.result()
            reports.append(report)
            _status(
                f"Feature extraction ready for {report['run']}: "
                f"{report['pixel_rows']} rows"
                + (
                    " (cache hit)"
                    if report.get("cache_hit")
                    else f" in {report['seconds_current']:.1f} s"
                )
            )
    reports.sort(key=lambda row: str(row["run"]))

    combined_features: list[np.ndarray] = []
    combined_owners: list[np.ndarray] = []
    combined_regions: list[ReviewedRegion] = []
    for task in tasks:
        with np.load(task.output_path, allow_pickle=False) as archive:
            raw_features = archive["features"]
            task_manifest = _read_json(task.staged.asset_manifest_path)
            v5_task = ((task_manifest.get("v5_photometry") or {}).get("format") == (
                "ffac-v5-native-source-materialization-v1") or _raw_review_manifest(task_manifest))
            if v5_task and raw_features.dtype != np.dtype("float32"):
                raise ValueError("V5 cached optical model features must be float32")
            features = np.asarray(raw_features, dtype=np.float32)
            local_owners = np.asarray(archive["region_indices"], dtype=np.int32)
        offset = len(combined_regions)
        combined_features.append(features)
        combined_owners.append(local_owners + offset)
        combined_regions.extend(task.regions)
    # Preserve every frozen review ID and its original meaning in inventory.
    # OOD chemistry runs have no sampled rows and cannot enter an optimizer.
    combined_regions.extend(sorted(evaluation_only_regions,
                                   key=lambda row: (row.run, row.frame_index,
                                                    row.region_id)))
    return (
        np.vstack(combined_features),
        np.concatenate(combined_owners),
        combined_regions,
        reports,
    )


def _variant_matrix(
    maximal_features: np.ndarray,
    contract: OpticalFeatureContract,
    maximal_contract: OpticalFeatureContract = MAXIMAL_CONTRACT,
) -> np.ndarray:
    maximal_names = list(maximal_contract.feature_names)
    if maximal_features.shape[1] != len(maximal_names):
        raise ValueError("Feature matrix differs from its extraction contract")
    columns = [maximal_names.index(name) for name in contract.feature_names]
    return np.asarray(maximal_features[:, columns], dtype=np.float32)


def _fit_model_object(
    fit: LogisticFit,
    *,
    feature_contract: OpticalFeatureContract,
    provenance: Mapping[str, Any],
    texture_sigma: float,
) -> DualPhaseSignalModel:
    return DualPhaseSignalModel(
        feature_mean=tuple(float(value) for value in fit.mean),
        feature_scale=tuple(float(value) for value in fit.scale),
        coefficients=tuple(float(value) for value in fit.coefficients),
        intercept=float(fit.intercept),
        texture_sigma=float(texture_sigma),
        training_provenance=dict(provenance),
        feature_names=feature_contract.feature_names,
        feature_contract=feature_contract.to_dict(),
        schema=TRANSFERABLE_MODEL_SCHEMA,
        algorithm=TRANSFERABLE_MODEL_ALGORITHM,
    )


def _parity_check(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    regularization: float,
    limit: int,
) -> dict[str, Any]:
    if not cuda_available():
        return {
            "status": "skipped",
            "reason": "CUDA/CuPy unavailable",
            "passed": False,
        }
    rng = np.random.default_rng(20260830)
    selected_chunks: list[np.ndarray] = []
    per_class = max(1000, int(limit) // 2)
    for value in (0, 1):
        candidates = np.flatnonzero(labels == value)
        if candidates.size > per_class:
            candidates = np.sort(rng.choice(candidates, per_class, replace=False))
        selected_chunks.append(candidates)
    selected = np.concatenate(selected_chunks)
    subset = matrix[selected]
    subset_labels = labels[selected]
    subset_weights = weights[selected]
    cpu = fit_weighted_logistic(
        subset,
        subset_labels,
        subset_weights,
        regularization=regularization,
        backend="cpu",
    )
    cuda = fit_weighted_logistic(
        subset,
        subset_labels,
        subset_weights,
        regularization=regularization,
        backend="cuda",
        normalization=(cpu.mean, cpu.scale),
    )
    cpu_probability = cpu.predict(subset)
    cuda_probability = cuda.predict(subset)
    difference = np.abs(cpu_probability - cuda_probability)
    agreement = float(np.mean((cpu_probability >= 0.5) == (cuda_probability >= 0.5)))
    report = {
        "status": "ok",
        "rows": int(subset.shape[0]),
        "cpu_precision": cpu.precision,
        "cuda_precision": cuda.precision,
        "median_probability_abs_difference": float(np.median(difference)),
        "p99_probability_abs_difference": float(np.quantile(difference, 0.99)),
        "max_probability_abs_difference": float(np.max(difference)),
        "classification_agreement_at_0p5": agreement,
    }
    report["passed"] = bool(
        report["p99_probability_abs_difference"] <= 0.005
        and report["max_probability_abs_difference"] <= 0.03
        and agreement >= 0.995
    )
    return report


def _inference_parity_check(
    fit: LogisticFit,
    matrix: np.ndarray,
    *,
    limit: int,
) -> dict[str, Any]:
    if not cuda_available():
        return {
            "status": "skipped",
            "reason": "CUDA/CuPy unavailable",
            "passed": False,
        }
    rng = np.random.default_rng(20260830)
    if matrix.shape[0] > int(limit):
        selected = np.sort(rng.choice(matrix.shape[0], int(limit), replace=False))
        subset = matrix[selected]
    else:
        subset = matrix
    cpu_probability = fit.predict(subset)
    cuda_probability = predict_logistic(fit, subset, backend="cuda")
    difference = np.abs(cpu_probability - cuda_probability)
    agreement = float(np.mean((cpu_probability >= 0.5) == (cuda_probability >= 0.5)))
    report = {
        "status": "ok",
        "kind": "serialized-model-cpu64-cuda32-inference-parity",
        "rows": int(subset.shape[0]),
        "median_probability_abs_difference": float(np.median(difference)),
        "p99_probability_abs_difference": float(np.quantile(difference, 0.99)),
        "max_probability_abs_difference": float(np.max(difference)),
        "classification_agreement_at_0p5": agreement,
    }
    report["passed"] = bool(
        report["p99_probability_abs_difference"] <= 0.005
        and report["max_probability_abs_difference"] <= 0.03
        and agreement >= 0.995
    )
    return report


def _cpu_reference_check(fit: LogisticFit, matrix: np.ndarray, *, limit: int) -> dict[str, Any]:
    """Check the CPU prediction path against stable float64 without touching CUDA."""
    if fit.backend != "cpu":
        raise ValueError("CPU reference validation requires a CPU fit")
    rng = np.random.default_rng(20260830)
    selected = (np.sort(rng.choice(len(matrix), int(limit), replace=False))
                if len(matrix) > int(limit) else slice(None))
    subset = np.asarray(matrix[selected], dtype=np.float64)
    logits = ((subset - fit.mean) / fit.scale) @ fit.coefficients + fit.intercept
    reference = np.exp(-np.logaddexp(0.0, -logits))
    probability = fit.predict(subset)
    difference = np.abs(reference - probability)
    agreement = float(np.mean((reference >= .5) == (probability >= .5)))
    report = {"status": "ok", "kind": "cpu-prediction-vs-stable-float64",
              "cuda_checked": False, "rows": len(subset),
              "p99_probability_abs_difference": float(np.quantile(difference, .99)),
              "max_probability_abs_difference": float(np.max(difference)),
              "classification_agreement_at_0p5": agreement}
    report["passed"] = bool(np.all(np.isfinite(probability))
                            and report["p99_probability_abs_difference"] <= .005
                            and report["max_probability_abs_difference"] <= .03
                            and agreement >= .995)
    return report


def _cuda_parity_record(validation: Mapping[str, Any]) -> dict[str, Any]:
    if validation.get("cuda_checked") is False:
        return {"status": "not_run", "passed": False,
                "reason": "Explicit CPU-only training; CUDA has not been validated"}
    return dict(validation)


def _eligible_runs_for_head(
    regions: Sequence[ReviewedRegion],
    head: str,
) -> list[str]:
    return sorted(
        {
            region.run
            for region in regions
            if regional_presence_target(region.phase_class, head) is not None
        }
    )


def _cross_validate_head(
    matrix: np.ndarray,
    region_indices: np.ndarray,
    regions: Sequence[ReviewedRegion],
    *,
    head: str,
    regularization_grid: Sequence[float],
    backend: str,
    fit_mode: str = "pure",
    weak_strength: float = 0.35,
    positive_top_fraction: float = 0.10,
    mil_max_iters: int = 160,
    selection_policy: str = "regional-oof-v1",
    fit_excluded_runs: Sequence[str] = (),
) -> tuple[dict[str, Any], LogisticFit, float, dict[str, Any]]:
    from .selection import ANCHOR_POLICY, LEGACY_POLICY, display_anchors, summarize_anchors, select_anchor_record
    if selection_policy not in (ANCHOR_POLICY, LEGACY_POLICY):
        raise ValueError("Unknown optical model selection policy")
    excluded_runs = {str(run).casefold() for run in fit_excluded_runs}
    runs = [run for run in _eligible_runs_for_head(regions, head)
            if run.casefold() not in excluded_runs]
    if len(runs) < 2:
        raise ValueError(
            f"{head} requires regional evaluation labels from at least two runs; "
            f"found {runs}"
        )
    grid_records: list[dict[str, Any]] = []
    score_cache: dict[tuple[float, str], np.ndarray] = {}
    prediction_buffer = None
    prediction_buffer_attempted = False
    for regularization in regularization_grid:
        fold_rows: list[dict[str, Any]] = []
        anchor_folds = []
        for held_out in runs:
            training_runs = set(runs) - {held_out}
            if fit_mode == "mil":
                fit, fit_diagnostics = fit_weighted_logistic_mil(
                    matrix,
                    region_indices,
                    regions,
                    head=head,
                    runs=training_runs,
                    regularization=float(regularization),
                    backend=backend,
                    weak_strength=float(weak_strength),
                    positive_top_fraction=float(positive_top_fraction),
                    max_iter=int(mil_max_iters),
                )
            elif fit_mode == "pure":
                train_matrix, train_labels, train_weights, _owners = balanced_pixel_view(
                    matrix,
                    region_indices,
                    regions,
                    head=head,
                    runs=training_runs,
                )
                fit = fit_weighted_logistic(
                    train_matrix,
                    train_labels,
                    train_weights,
                    regularization=float(regularization),
                    backend=backend,
                )
                fit_diagnostics = {"mode": "pure-region-irls", "weak_bags": 0,
                    "regional_supervision": {"policy": MIL_SUPERVISION_POLICY,
                        "pixel_supervision": pixel_supervision_metadata(
                            region_indices, regions, head=head, runs=training_runs)}}
            else:
                raise ValueError(f"Unsupported fit mode {fit_mode!r}")
            if not prediction_buffer_attempted:
                prediction_buffer = _try_cuda_prediction_feature_buffer(
                    matrix, backend=fit.backend)
                prediction_buffer_attempted = True
            probabilities = predict_logistic(
                fit, matrix, backend=backend, feature_buffer=prediction_buffer)
            if selection_policy == ANCHOR_POLICY:
                anchor_folds.append(display_anchors(probabilities, region_indices, regions, head=head, run=held_out))
            scores = region_probability_scores(
                probabilities,
                region_indices,
                len(regions),
            )
            score_cache[(float(regularization), held_out)] = scores
            train_scores, train_targets, train_region_weights, _ = (
                regional_evaluation_view(
                    scores,
                    regions,
                    head=head,
                    runs=training_runs,
                )
            )
            threshold, _ = tune_presence_threshold(
                train_scores,
                train_targets,
                train_region_weights,
            )
            held_scores, held_targets, held_weights, held_indices = (
                regional_evaluation_view(
                    scores,
                    regions,
                    head=head,
                    runs={held_out},
                )
            )
            metrics = regional_metrics(
                held_scores,
                held_targets,
                held_weights,
                threshold=threshold,
            )
            fold_rows.append(
                {
                    "held_out_run": held_out,
                    "held_out_dataset": regions[int(held_indices[0])].dataset,
                    "fit_backend": fit.backend,
                    "fit_precision": fit.precision,
                    "fit_iterations": fit.iterations,
                    "fit_converged": fit.converged,
                    "fit_diagnostics": fit_diagnostics,
                    **metrics,
                }
            )
        balanced = [float(row["balanced_accuracy"]) for row in fold_rows]
        auc = [float(row["roc_auc"]) for row in fold_rows]
        alpha_oof_scores = np.full(len(regions), np.nan, dtype=np.float64)
        for held_out in runs:
            held_scores = score_cache[(float(regularization), held_out)]
            for index, region in enumerate(regions):
                if region.run == held_out:
                    alpha_oof_scores[index] = held_scores[index]
        all_scores, all_targets, all_weights, _ = regional_evaluation_view(
            alpha_oof_scores,
            regions,
            head=head,
            runs=set(runs),
        )
        global_threshold, _ = tune_presence_threshold(
            all_scores,
            all_targets,
            all_weights,
        )
        global_metrics = regional_metrics(
            all_scores,
            all_targets,
            all_weights,
            threshold=global_threshold,
        )
        global_run_metrics: list[dict[str, Any]] = []
        for held_out in runs:
            run_scores, run_targets, run_weights, _ = regional_evaluation_view(
                alpha_oof_scores,
                regions,
                head=head,
                runs={held_out},
            )
            global_run_metrics.append(
                {
                    "held_out_run": held_out,
                    **regional_metrics(
                        run_scores,
                        run_targets,
                        run_weights,
                        threshold=global_threshold,
                    ),
                }
            )
        global_balanced = [
            float(row["balanced_accuracy"]) for row in global_run_metrics
        ]
        grid_records.append(
            {
                "regularization": float(regularization),
                "mean_balanced_accuracy": float(np.mean(balanced)),
                "minimum_balanced_accuracy": float(np.min(balanced)),
                "mean_roc_auc": float(np.nanmean(auc)),
                "global_oof_threshold": float(global_threshold),
                "global_oof_metrics": global_metrics,
                "global_threshold_run_metrics": global_run_metrics,
                "minimum_global_threshold_balanced_accuracy": float(
                    np.min(global_balanced)
                ),
                "mean_global_threshold_balanced_accuracy": float(
                    np.mean(global_balanced)
                ),
                "fold_threshold_range": float(
                    np.max([float(row["threshold"]) for row in fold_rows])
                    - np.min([float(row["threshold"]) for row in fold_rows])
                ),
                "folds": fold_rows,
            }
        )
        _status(
            f"{head}: regularization={regularization:g}, "
            f"global-threshold mean={np.mean(global_balanced):.3f}, "
            f"worst={np.min(global_balanced):.3f}"
        )
        if selection_policy == ANCHOR_POLICY:
            grid_records[-1]["display_anchors"] = summarize_anchors(anchor_folds)
            _status(f"{head}: fixed-display anchors={grid_records[-1]['display_anchors']}")

    # Stop retaining the full matrix before the final full-data fit.
    prediction_buffer = None
    selected_record = max(
        grid_records,
        key=lambda row: (
            float(row["minimum_global_threshold_balanced_accuracy"]),
            float(row["mean_global_threshold_balanced_accuracy"]),
            -float(row["fold_threshold_range"]),
            float(row["mean_roc_auc"]),
            -abs(np.log10(float(row["regularization"])) + 3.0),
        ),
    )
    if selection_policy == ANCHOR_POLICY:
        selected_record = select_anchor_record(grid_records)
    selected_regularization = float(selected_record["regularization"])

    oof_scores = np.full(len(regions), np.nan, dtype=np.float64)
    for held_out in runs:
        fold_scores = score_cache[(selected_regularization, held_out)]
        for index, region in enumerate(regions):
            if region.run == held_out:
                oof_scores[index] = fold_scores[index]
    eval_scores, eval_targets, eval_weights, _ = regional_evaluation_view(
        oof_scores,
        regions,
        head=head,
        runs=set(runs),
    )
    final_threshold, threshold_accuracy = tune_presence_threshold(
        eval_scores,
        eval_targets,
        eval_weights,
    )
    oof_metrics = regional_metrics(
        eval_scores,
        eval_targets,
        eval_weights,
        threshold=final_threshold,
    )

    full_matrix, full_labels, full_weights, _ = balanced_pixel_view(
        matrix,
        region_indices,
        regions,
        head=head,
        runs=set(runs),
    )
    if fit_mode == "mil":
        final_fit, final_fit_diagnostics = fit_weighted_logistic_mil(
            matrix,
            region_indices,
            regions,
            head=head,
            runs=set(runs),
            regularization=selected_regularization,
            backend=backend,
            weak_strength=float(weak_strength),
            positive_top_fraction=float(positive_top_fraction),
            max_iter=int(mil_max_iters),
        )
        parity = (_cpu_reference_check(final_fit, full_matrix, limit=120_000)
                  if backend == "cpu" else
                  _inference_parity_check(final_fit, full_matrix, limit=120_000))
    else:
        parity = None if backend == "cpu" else _parity_check(
            full_matrix,
            full_labels,
            full_weights,
            regularization=selected_regularization,
            limit=120_000,
        )
        final_fit = fit_weighted_logistic(
            full_matrix,
            full_labels,
            full_weights,
            regularization=selected_regularization,
            backend=backend,
        )
        final_fit_diagnostics = {"mode": "pure-region-irls", "weak_bags": 0,
            "regional_supervision": {"policy": MIL_SUPERVISION_POLICY,
                "pixel_supervision": pixel_supervision_metadata(
                    region_indices, regions, head=head, runs=set(runs))}}
        if backend == "cpu":
            parity = _cpu_reference_check(final_fit, full_matrix, limit=120_000)
    report = {
        "head": head,
        "eligible_runs": runs,
        "fit_excluded_runs": sorted(excluded_runs),
        "fit_mode": fit_mode,
        "selection_policy": (
            "maximize the worst held-out balanced accuracy with one shared "
            "global OOF threshold, then mean accuracy, threshold stability, "
            "and ROC AUC"
        ),
        "regularization_grid": grid_records,
        "selected_regularization": selected_regularization,
        "oof_threshold": float(final_threshold),
        "oof_threshold_balanced_accuracy": float(threshold_accuracy),
        "oof_metrics": oof_metrics,
        "numerical_validation": parity,
        "cpu64_cuda32_parity": _cuda_parity_record(parity),
        "final_fit": {
            "backend": final_fit.backend,
            "precision": final_fit.precision,
            "iterations": final_fit.iterations,
            "converged": final_fit.converged,
            "objective": final_fit.objective,
            "pixel_rows": int(full_matrix.shape[0]),
            "positive_pixel_fraction": float(np.mean(full_labels)),
            "diagnostics": final_fit_diagnostics,
        },
    }
    if selection_policy == ANCHOR_POLICY:
        report.update(selection_policy=ANCHOR_POLICY,
                      selected_display_anchors=selected_record["display_anchors"],
                      positive_recall_tolerance=.03,
                      anchor_scope="held-out pure human regions; not full-image pixel ground truth")
    return report, final_fit, final_threshold, parity


def _write_region_inventory(path: Path, regions: Sequence[ReviewedRegion], *,
                            reviewer_ids: bool = False) -> None:
    """Write region_inventory.csv; ``reviewer_ids`` adds a reviewer_id column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "region_index",
                "region_id",
                "dataset",
                "run",
                "frame_index",
                "elapsed_h",
                "phase_class",
                "confidence",
                "source",
                "reviewer",
                *(("reviewer_id",) if reviewer_ids else ()),
                "supervision_weight",
            ),
        )
        writer.writeheader()
        for index, region in enumerate(regions):
            writer.writerow(
                {
                    "region_index": index,
                    "region_id": region.region_id,
                    "dataset": region.dataset,
                    "run": region.run,
                    "frame_index": region.frame_index,
                    "elapsed_h": region.elapsed_h,
                    "phase_class": region.phase_class,
                    "confidence": region.confidence,
                    "source": region.source,
                    "reviewer": region.reviewer,
                    **({"reviewer_id": region.reviewer_id} if reviewer_ids else {}),
                    "supervision_weight": region.supervision_weight,
                }
            )


def _reviewer_inventory(regions: Sequence[ReviewedRegion], reviewers: Sequence[str] | None) -> dict[str, Any]:
    """Reviewer-scope fields of the training inventory; none without a scope."""
    if reviewers is None:
        return {}
    return {"reviewer_scope": list(reviewers),
            "reviewer_ids": dict(Counter(region.reviewer_id for region in regions))}


def _reviewer_provenance(regions: Sequence[ReviewedRegion], reviewers: Sequence[str] | None) -> dict[str, Any]:
    """Reviewer-scope fields of a model head's provenance; none without a scope."""
    if reviewers is None:
        return {}
    return {"reviewer_scope": list(reviewers),
            "reviewer_ids": sorted({region.reviewer_id for region in regions})}


def _sync_result(local_result: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for source in local_result.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(local_result)
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)


def _mil_checkpoint_normalization_compatible(report: Mapping[str, Any]) -> bool:
    """Never resume fits or CV selection with stale supervision semantics.

    This guard belongs to retraining checkpoints, not serialized-model loading:
    frozen prior models still infer with their own saved means and scales.
    Pure-only/disabled-MIL fits retain their normalization, but explicit gas
    absence changes their direct targets too and must invalidate old fits.
    """
    def compatible(diagnostics: Any) -> bool:
        if not isinstance(diagnostics, Mapping):
            return False
        supervision = diagnostics.get("regional_supervision")
        if not isinstance(supervision, Mapping) or supervision.get("policy") != MIL_SUPERVISION_POLICY:
            return False
        normalization = diagnostics.get("normalization", {})
        if not isinstance(normalization, Mapping):
            return False
        mode = diagnostics.get("mode")
        policy = normalization.get("policy")
        if mode == "pure-region-irls":
            return policy in (None, "pure-weighted-standardization-v1")
        return (
            mode == "pure-irls-plus-mixed-region-mil"
            and policy == MIL_NORMALIZATION_POLICY
        )

    final = report.get("final_fit")
    if not isinstance(final, Mapping) or not compatible(final.get("diagnostics")):
        return False
    grid = report.get("regularization_grid")
    if not isinstance(grid, (list, tuple)) or not grid:
        return False
    for record in grid:
        if not isinstance(record, Mapping):
            return False
        folds = record.get("folds")
        if not isinstance(folds, (list, tuple)) or not folds:
            return False
        if any(
            not isinstance(fold, Mapping) or not compatible(fold.get("fit_diagnostics"))
            for fold in folds
        ):
            return False
    return True


def _fit_checkpoint_contract(
    args: argparse.Namespace,
    features: np.ndarray,
    region_indices: np.ndarray,
    regions: Sequence[ReviewedRegion],
) -> dict[str, Any]:
    """Bind a resumable fit to its samples, labels, knobs, and implementation."""

    digest = hashlib.sha256()
    for array in (features, region_indices):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(memoryview(contiguous).cast("B"))
    return {
        "schema": 1,
        "sample_matrix_sha256": digest.hexdigest(),
        "ordered_regions_sha256": _canonical_hash([
            {
                "region_id": row.region_id, "run": row.run, "dataset": row.dataset,
                "frame_index": row.frame_index, "elapsed_h": row.elapsed_h,
                "phase_class": row.phase_class, "confidence": row.confidence,
                "source": row.source, "reviewer": row.reviewer, "payload": row.payload,
            }
            for row in regions
        ]),
        "regularization_grid": [float(value) for value in args.regularization_grid],
        "backend": str(args.backend),
        "fit_mode": str(args.fit_mode),
        "weak_strength": float(args.weak_strength),
        "positive_top_fraction": float(args.positive_top_fraction),
        "mil_max_iters": int(args.mil_max_iters),
        "gas_selection_policy": str(args.gas_selection_policy),
        "fit_excluded_runs": list(getattr(args, "fit_excluded_runs", ()) or ()),
        "minimum_balanced_accuracy": float(args.minimum_balanced_accuracy),
        "implementation_sha256": {
            "trainer": _sha256(Path(__file__)),
            "training_primitives": _sha256(PACKAGE_ROOT / "logistic.py"),
            "optical_features": _sha256(FLUIDFLOWER_ROOT / "phase_features.py"),
        },
    }


def _load_reusable_head(
    path: Path,
    *,
    head: str,
    fit_mode: str,
    inventory: Mapping[str, Any],
    fit_checkpoint_contract: Mapping[str, Any] | None = None,
) -> tuple[DualPhaseSignalModel, dict[str, Any], float, dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        model = DualPhaseSignalModel.load(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    provenance = model.training_provenance
    if (
        provenance.get("campaign_id") != CAMPAIGN_ID
        or provenance.get("target_kind") != f"visible_{head}_phase_probability"
        or provenance.get("fit_mode") != fit_mode
        or provenance.get("training_inventory") != _json_safe(inventory)
        or (fit_checkpoint_contract is not None and
            provenance.get("fit_checkpoint_contract") != dict(fit_checkpoint_contract))
    ):
        return None
    report = provenance.get("cross_validation")
    backend_contract = provenance.get("backend_contract")
    if not isinstance(report, Mapping) or not isinstance(backend_contract, Mapping):
        return None
    if not _mil_checkpoint_normalization_compatible(report):
        return None
    parity = backend_contract.get("parity")
    if not isinstance(parity, Mapping):
        return None
    threshold = report.get("oof_threshold")
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
        return None
    restored_report = dict(report)
    restored_report["deployment_allowed"] = bool(provenance.get("deployment_allowed"))
    restored_report["model_path"] = str(path)
    restored_report["model_sha256"] = model.to_dict()["model_sha256"]
    restored_report["reused_from_verified_staging_checkpoint"] = True
    return model, restored_report, float(threshold), dict(parity)


def _selected_grid_record(report: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = float(report.get("selected_regularization", float("nan")))
    records = list(report.get("regularization_grid") or [])
    for record in records:
        if np.isclose(float(record.get("regularization", float("nan"))), selected):
            return record
    return {}


def _selected_cv_folds_converged(report: Mapping[str, Any]) -> bool:
    folds = (_selected_grid_record(report).get("folds") or [])
    return bool(folds) and all(
        bool(fold.get("fit_converged")) for fold in folds
    )


def _numerically_converged_for_deployment(report: Mapping[str, Any]) -> bool:
    """Do not advertise an uncertified selected CV fit as deployable."""

    final_fit = report.get("final_fit") or {}
    return bool(final_fit.get("converged")) and _selected_cv_folds_converged(report)


def change_evidence_ablation_report(
    model_reports: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare full-run holdout metrics with and without DarSIA change evidence."""

    reference = dict(model_reports.get("generic") or {})
    ablated = dict(model_reports.get(CHANGE_ABLATION_VARIANT) or {})
    if not reference or not ablated:
        return {
            "status": "not_requested",
            "reference_variant": "generic",
            "ablated_variant": CHANGE_ABLATION_VARIANT,
        }
    heads: dict[str, Any] = {}
    balanced_deltas: list[float] = []
    reference_heads = dict(reference.get("heads") or {})
    ablated_heads = dict(ablated.get("heads") or {})
    for head in sorted(set(reference_heads).intersection(ablated_heads)):
        full_report = dict(reference_heads[head])
        ablated_report = dict(ablated_heads[head])
        full_metrics = dict(full_report.get("oof_metrics") or {})
        ablated_metrics = dict(ablated_report.get("oof_metrics") or {})
        metric_delta: dict[str, float | None] = {}
        for metric in ("balanced_accuracy", "roc_auc", "average_precision"):
            left = full_metrics.get(metric)
            right = ablated_metrics.get(metric)
            metric_delta[metric] = (
                None
                if not isinstance(left, (int, float))
                or not isinstance(right, (int, float))
                or not np.isfinite(left)
                or not np.isfinite(right)
                else float(left) - float(right)
            )
        if metric_delta["balanced_accuracy"] is not None:
            balanced_deltas.append(float(metric_delta["balanced_accuracy"]))
        full_runs = {
            str(row["held_out_run"]): row
            for row in _selected_grid_record(full_report).get(
                "global_threshold_run_metrics", []
            )
        }
        ablated_runs = {
            str(row["held_out_run"]): row
            for row in _selected_grid_record(ablated_report).get(
                "global_threshold_run_metrics", []
            )
        }
        holdouts = []
        for run in sorted(set(full_runs).intersection(ablated_runs)):
            full_score = float(full_runs[run]["balanced_accuracy"])
            ablated_score = float(ablated_runs[run]["balanced_accuracy"])
            holdouts.append(
                {
                    "run": run,
                    "with_darsia_change": full_score,
                    "without_darsia_change": ablated_score,
                    "delta": full_score - ablated_score,
                }
            )
        heads[head] = {
            "with_darsia_change": full_metrics,
            "without_darsia_change": ablated_metrics,
            "delta_with_minus_without": metric_delta,
            "leave_one_run_out": holdouts,
        }
    mean_delta = float(np.mean(balanced_deltas)) if balanced_deltas else None
    return {
        "status": "ok",
        "reference_variant": "generic",
        "ablated_variant": CHANGE_ABLATION_VARIANT,
        "scope": "registered visible 2D optical classification only",
        "validation": "leave-one-complete-run-out",
        "heads": heads,
        "mean_balanced_accuracy_delta": mean_delta,
        "interpretation": (
            "Positive values support retaining DarSIA change evidence; this report "
            "does not evaluate concentration, depth, saturation, or mass."
        ),
    }


def _radiometry_runtime(profile):
    if profile is None:
        return None
    from .radiometry import runtime_identity
    return runtime_identity()


def train_campaign(args: argparse.Namespace) -> dict[str, Any]:
    from darsia._opencv_runtime import scientific_thread, require_ipp_off
    ipp_start = scientific_thread("train_campaign:start")
    from ..review_storage import resolve_review_path
    started = time.perf_counter()
    regularization_grid = tuple(float(value) for value in args.regularization_grid)
    if (not regularization_grid or any(not np.isfinite(value) or value <= 0
                                      for value in regularization_grid)
            or len(set(regularization_grid)) != len(regularization_grid)):
        raise ValueError("Select distinct, finite, positive regularization values")
    selected_variants = model_variants(
        include_change_ablation=bool(getattr(args, "change_evidence_ablation", False)),
        generic_version=getattr(args, "feature_version", GENERIC_FEATURE_VERSION),
        adapters=getattr(args, "adapters", None))
    # Keep the maximal optical extraction and its cache identity even when the
    # fit selects only generic heads. Adapter selection changes fit time only.
    maximal_contract = replace(MAXIMAL_CONTRACT,
        generic_version=getattr(args, "feature_version", GENERIC_FEATURE_VERSION))
    bundles = [
        ReviewBundle(resolve_review_path(decisions), resolve_review_path(manifest))
        for decisions, manifest in args.bundle
    ]
    from .. import phase_review_retractions as withdrawals
    from .scope import (FROZEN_ANNOTATION_SET_REFUSAL, partition_review_events, requested_reviewers,
                        require_reviewer_scope_allowed)
    reviewers = requested_reviewers({"reviewers": getattr(args, "reviewers", None)})
    frozen_request = Path(args.output_root) / "training_request.json"
    if reviewers is not None and frozen_request.is_file():
        require_reviewer_scope_allowed({**(_read_json(frozen_request).get("settings") or {}),
                                        "reviewers": list(reviewers)})
    withdrawal_snapshot = withdrawals.model_retractions(Path(args.output_root))
    if withdrawal_snapshot is None:
        withdrawal_snapshot = withdrawals.snapshot_for_bundles(
            [(b.decisions_path, b.combined_manifest_path) for b in bundles])
    withdrawal_snapshot = partition_review_events(withdrawal_snapshot, reviewers)
    withdrawals.require_fresh_fit(withdrawal_snapshot, vars(args))
    if withdrawal_snapshot["targets"] and (Path(args.output_root) / "summary.json").exists():
        raise FileExistsError("Withdrawn evidence requires a new model version; completed models are immutable")
    withdrawal_record = withdrawals.write_model_snapshot(Path(args.output_root), withdrawal_snapshot)
    label_policy_audit: dict[str, Any] = {}
    preserve_revisions = bool(getattr(args, "preserve_native_revisions", False))
    origin_sources: dict[str, RunSource] | None = {} if preserve_revisions else None
    sources, regions, conflicts = _load_sources_and_regions(
        bundles, withdrawal_snapshot=withdrawal_snapshot,
        preserve_native_revisions=preserve_revisions,
        label_policy_audit=label_policy_audit, region_sources=origin_sources,
        reviewers=reviewers)
    series_filter = {str(value).casefold().replace("_", "") for value in getattr(args, "series", []) or []}
    if series_filter:
        sources = [s for s in sources if s.dataset.casefold().replace("_", "") in series_filter]
        regions = [r for r in regions if r.dataset.casefold().replace("_", "") in series_filter]
        if origin_sources is not None:
            selected_ids = {region.region_id for region in regions}
            origin_sources = {identifier: source for identifier, source in origin_sources.items()
                              if identifier in selected_ids}
    from .scope import requested_series, validate_training_series
    selected_series = requested_series({"model_scope": getattr(args, "model_scope", "shared"),
                                       "series": getattr(args, "series", [])})
    validate_training_series(sources, regions, selected_series)
    if len({s.run for s in sources}) != len(sources):
        raise ValueError("Run IDs collide across series; use distinct run IDs before training")
    from .radiometry import resolve_profile
    radiometry_profile = resolve_profile(getattr(args, "radiometry_profile", None))
    if len(sources) < 4:
        raise RuntimeError(
            "Cross-series v3 training requires at least four reviewed runs"
        )
    inventory_sources = (list(origin_sources.values()) if origin_sources is not None
                         else list(sources))
    v5_source_flags = [
        (_read_json(source.asset_manifest_path).get("v5_photometry") or {}).get(
            "format") == "ffac-v5-native-source-materialization-v1"
        for source in inventory_sources
    ]
    if any(v5_source_flags) and not all(v5_source_flags):
        raise ValueError("Cannot combine V5 RAW and historical JPEG review inputs")
    raw_source_flags = [_raw_review_manifest(_read_json(source.asset_manifest_path))
                        for source in inventory_sources]
    if any(raw_source_flags) and not all(raw_source_flags):
        raise ValueError("Cannot combine RAW photographic reviews with other review inputs")
    # RAW reviews are float camera-linear inputs and follow the V5 fit rules.
    float_source_flags = [v5 or raw for v5, raw in zip(v5_source_flags, raw_source_flags)]
    if reviewers is not None and float_source_flags and all(float_source_flags):
        # Its fit scope pins exact annotation counts and IDs (see chemistry_fit_scope).
        raise ValueError(FROZEN_ANNOTATION_SET_REFUSAL)
    fit_excluded_runs = tuple(getattr(args, "fit_excluded_runs", ()) or ())
    chemistry_scope = None
    fit_allowed_review_ids = None
    partition_binding = None
    track1_flags = [bool(_read_json(source.asset_manifest_path).get(
        "v5_track1_photo_geometry")) for source in inventory_sources]
    if any(track1_flags) and not all(track1_flags):
        raise ValueError("Cannot mix Track1 photo-only and physical V5 sources")
    if float_source_flags and all(float_source_flags):
        from darsia.fluidflower.phase_training.chemistry_fit_scope import EXCLUDED_RUNS, verify_scope
        request_path = Path(args.output_root) / "training_request.json"
        if not request_path.is_file():
            raise ValueError("V5 fit requires its frozen training request")
        request = _read_json(request_path)
        chemistry_scope = (request.get("settings") or {}).get("v5_chemistry_fit_scope")
        if not chemistry_scope or fit_excluded_runs != EXCLUDED_RUNS:
            raise ValueError("V5 fit omitted its six-run chemistry exclusion")
        verify_scope(chemistry_scope, regions)
        if track1_flags and all(track1_flags):
            from .review_fit_partition import verify_fit_partition
            partition_binding = (request.get("settings") or {}).get(
                "v5_review_fit_partition")
            if not partition_binding or origin_sources is None:
                raise ValueError("Track1 fit omitted source-bound 4048-ID partition")
            fit_allowed_review_ids = verify_fit_partition(
                partition_binding, regions, origin_sources)
    elif fit_excluded_runs:
        raise ValueError("Chemistry fit exclusion is only valid for V5 RAW sources")
    run_names = sorted(source.run for source in sources)
    _status(
        f"Loaded {len(regions)} deduplicated reviewed regions from "
        f"{', '.join(run_names)}"
    )
    local_result = Path(args.cache_root) / "result_staging"
    if local_result.exists() and not args.resume_completed_heads:
        shutil.rmtree(local_result)
    local_result.mkdir(parents=True, exist_ok=True)
    if radiometry_profile is not None:
        _write_json(local_result / "radiometry_profile.json", radiometry_profile)

    maximal_features, region_indices, ordered_regions, extraction_reports = (
        _load_feature_caches(
            sources,
            regions,
            cache_root=Path(getattr(args, "feature_cache_root", None) or args.cache_root),
            stage_workers=args.stage_workers,
            extraction_workers=args.extraction_workers,
            opencv_threads=args.opencv_threads,
            pixels_per_region=args.pixels_per_region,
            texture_sigma=args.texture_sigma,
            radiometry_profile=radiometry_profile,
            feature_contract=maximal_contract,
            origin_sources=origin_sources,
            fit_excluded_runs=fit_excluded_runs,
            fit_allowed_review_ids=fit_allowed_review_ids,
        )
    )
    _status(
        f"Combined feature matrix: {maximal_features.shape[0]} x "
        f"{maximal_features.shape[1]}"
    )
    _write_region_inventory(local_result / "region_inventory.csv", ordered_regions,
                            reviewer_ids=reviewers is not None)
    with (local_result / "training_samples.npz").open("wb") as stream:
        np.savez_compressed(
            stream,
            features=np.asarray(maximal_features, dtype=np.float32),
            region_indices=np.asarray(region_indices, dtype=np.int32),
            feature_names=np.asarray(maximal_contract.feature_names),
        )

    inventory = {
        "training_label_policy": label_policy_audit,
        "native_revision_policy": ("per-annotation-original-canvas-v1" if preserve_revisions
                                   else "one-selected-source-per-run"),
        "native_source_revisions": (len({(str(source.asset_manifest_path.resolve()),
                                          str(source.review_root.resolve()))
                                         for source in origin_sources.values()})
                                    if origin_sources is not None else len(sources)),
        "radiometry_profile": radiometry_profile,
        "series_filter": sorted(series_filter),
        "regions_by_series_run": dict(Counter(f"{r.dataset}/{r.run}" for r in ordered_regions)),
        "regions": len(ordered_regions),
        "pixel_rows": int(maximal_features.shape[0]),
        "runs": run_names,
        "datasets": sorted({region.dataset for region in ordered_regions}),
        "series_scope": selected_series,
        "phase_classes": dict(
            Counter(region.phase_class for region in ordered_regions)
        ),
        "confidence": dict(Counter(region.confidence for region in ordered_regions)),
        "sources": dict(Counter(region.source for region in ordered_regions)),
        "reviewers": dict(Counter(region.reviewer for region in ordered_regions)),
        **_reviewer_inventory(ordered_regions, reviewers),
        "deduplication_conflicts": conflicts,
    }
    if float_source_flags and all(float_source_flags):
        verify_scope(chemistry_scope, ordered_regions)
        inventory["v5_chemistry_fit_scope"] = chemistry_scope
        if partition_binding is not None:
            inventory["v5_review_fit_partition"] = partition_binding
        from .review_sampling import inventory as v5_sampling_inventory
        inventory.update(v5_sampling_inventory(ordered_regions))
    else:
        from .gas_negative_sampling import sampling_inventory
        inventory.update(sampling_inventory(ordered_regions))
    _write_json(local_result / "training_inventory.json", inventory)
    fit_checkpoint_contract = _fit_checkpoint_contract(
        args, maximal_features, region_indices, ordered_regions)

    model_reports: dict[str, Any] = {}
    reused_heads: list[str] = []
    decision_policy: dict[str, Any] = {
        "schema": 1,
        "region_score": "90th percentile of pixel probability",
        "models": {},
    }
    for variant, feature_contract in selected_variants.items():
        matrix = _variant_matrix(maximal_features, feature_contract, maximal_contract)
        variant_root = local_result / "models" / variant
        variant_report: dict[str, Any] = {
            "feature_contract": feature_contract.to_dict(),
            "heads": {},
        }
        decision_policy["models"][variant] = {}
        for head in args.heads:
            model_path = variant_root / f"{head}_signal.json"
            reusable = (
                _load_reusable_head(
                    model_path,
                    head=head,
                    fit_mode=args.fit_mode,
                    inventory=inventory,
                    fit_checkpoint_contract=fit_checkpoint_contract,
                )
                if args.resume_completed_heads
                else None
            )
            if reusable is not None and reusable[0].feature_contract != feature_contract.to_dict():
                reusable = None
            if reusable is not None:
                model, report, threshold, _parity = reusable
                variant_report["heads"][head] = report
                decision_policy["models"][variant][head] = {
                    "presence_threshold": float(threshold),
                    "model_sha256": model.to_dict()["model_sha256"],
                }
                reused_heads.append(f"{variant}/{head}")
                _status(f"Reused verified staging checkpoint for {variant}/{head}")
                continue
            _status(f"Cross-validating {variant}/{head} on {args.backend}")
            report, fit, threshold, parity = _cross_validate_head(
                matrix,
                region_indices,
                ordered_regions,
                head=head,
                regularization_grid=args.regularization_grid,
                backend=args.backend,
                fit_mode=args.fit_mode,
                weak_strength=args.weak_strength,
                positive_top_fraction=args.positive_top_fraction,
                mil_max_iters=args.mil_max_iters,
                selection_policy=(getattr(args, "gas_selection_policy", "regional-oof-v1")
                                  if head == "gas_within_co2" else "regional-oof-v1"),
                fit_excluded_runs=fit_excluded_runs,
            )
            numerically_converged = _numerically_converged_for_deployment(report)
            report["selected_cv_folds_converged"] = _selected_cv_folds_converged(report)
            report["numerically_converged_for_deployment"] = numerically_converged
            deployment_allowed = bool(
                feature_contract.change_detector != NO_CHANGE_DETECTOR
                and numerically_converged
                and parity.get("passed")
                and float(report["oof_metrics"]["balanced_accuracy"])
                >= float(args.minimum_balanced_accuracy)
            )
            provenance = {
                "radiometry_profile": radiometry_profile,
                "radiometry_runtime": _radiometry_runtime(radiometry_profile),
                "campaign_id": CAMPAIGN_ID,
                "target_kind": f"visible_{head}_phase_probability",
                "fit_mode": args.fit_mode,
                "deployment_allowed": deployment_allowed,
                "deployment_scope": (
                    "Radiometrically corrected RGB frames with this exact "
                    "feature contract; output is optical probability, not mass "
                    "or saturation."
                ),
                "runs": run_names,
                "head_training_runs": list(report["eligible_runs"]),
                "datasets": sorted({source.dataset for source in sources}),
                "series_scope": selected_series,
                "reviewers": sorted({region.reviewer for region in ordered_regions}),
                **_reviewer_provenance(ordered_regions, reviewers),
                "training_inventory": inventory,
                "fit_checkpoint_contract": fit_checkpoint_contract,
                "cross_validation": _json_safe(report),
                "backend_contract": {
                    "production_backend": fit.backend,
                    "production_precision": fit.precision,
                    "authoritative_reference": "cpu-float64",
                    "validated_backends": ["cpu"] if args.backend == "cpu" else ["cpu", "cuda"],
                    "parity": _json_safe(parity),
                },
                "source_sha256": {
                    "trainer": _sha256(Path(__file__)),
                    "training_primitives": _sha256(PACKAGE_ROOT / "logistic.py"),
                    "optical_features": _sha256(FLUIDFLOWER_ROOT / "phase_features.py"),
                },
                "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            model = _fit_model_object(
                fit,
                feature_contract=feature_contract,
                provenance=provenance,
                texture_sigma=args.texture_sigma,
            )
            model_path = model.save(model_path)
            report["deployment_allowed"] = deployment_allowed
            report["model_path"] = str(model_path)
            report["model_sha256"] = model.to_dict()["model_sha256"]
            variant_report["heads"][head] = report
            decision_policy["models"][variant][head] = {
                "presence_threshold": float(threshold),
                "model_sha256": model.to_dict()["model_sha256"],
            }
            _status(
                f"Finished {variant}/{head}: OOF balanced accuracy "
                f"{report['oof_metrics']['balanced_accuracy']:.3f}, "
                f"deployment_allowed={deployment_allowed}"
            )
        model_reports[variant] = variant_report

    ablation = change_evidence_ablation_report(model_reports)
    _write_json(local_result / "cross_validation.json", _json_safe(model_reports))
    _write_json(local_result / "change_evidence_ablation.json", _json_safe(ablation))
    _write_json(local_result / "phase_decision_policy.json", decision_policy)
    source_snapshot = [
        {
            "decisions": str(bundle.decisions_path),
            "decisions_sha256": _sha256(bundle.decisions_path),
            "combined_manifest": str(bundle.combined_manifest_path),
            "combined_manifest_sha256": _sha256(bundle.combined_manifest_path),
        }
        for bundle in bundles
    ]
    summary = {
        "opencv_ipp": dict(start=ipp_start, completed=require_ipp_off("train_campaign:completed")),
        "radiometry_profile": radiometry_profile,
        "radiometry_runtime": _radiometry_runtime(radiometry_profile),
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "backend_requested": args.backend,
        "cuda_available": None if args.backend == "cpu" else cuda_available(),
        "cuda_probed": args.backend != "cpu",
        "runs": run_names,
        "inventory": inventory,
        "fit_checkpoint_contract": fit_checkpoint_contract,
        "source_snapshot": source_snapshot,
        "review_retractions": withdrawal_record,
        "feature_extraction": extraction_reports,
        "models": model_reports,
        "change_evidence_ablation": ablation,
        "reused_heads": reused_heads,
        "seconds": float(time.perf_counter() - started),
    }
    _write_json(local_result / "summary.json", _json_safe(summary))
    _write_json(local_result / "radiometry_inventory.json", {
        "profile": radiometry_profile,
        "series_filter": sorted(series_filter),
        "runs": [{"series_id": r["dataset"], "run_id": r["run"],
                  "source_asset_sha256": r["asset_sha256"],
                  "radiometry": r.get("radiometry")} for r in extraction_reports],
    })
    _status(f"Syncing final result to {args.output_root}")
    _sync_result(local_result, Path(args.output_root))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-id",
        default=CAMPAIGN_ID,
        help="Immutable model-version identifier recorded in every model head.",
    )
    parser.add_argument(
        "--bundle",
        nargs=2,
        action="append",
        metavar=("DECISIONS_JSON", "COMBINED_MANIFEST_JSON"),
        help="Repeat for each frozen review bundle.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--radiometry-profile", default="legacy",
                        help="image:PATH, uniform24-after2014, a frozen profile JSON, or explicit legacy input")
    parser.add_argument("--series", nargs="+", help="Optional series subset, e.g. FF_AC FF_UM")
    parser.add_argument("--reviewer", action="append", dest="reviewers", metavar="REVIEWER_ID",
                        help="Train only on this reviewer's annotations, answers and evidence actions; "
                             "repeat for several people (default: everyone)")
    parser.add_argument("--model-scope", choices=("shared", "single-series"), default="shared")
    parser.add_argument("--feature-cache-root", type=Path,
                        help="Reusable image-feature cache, separate from this model's fit checkpoints")
    parser.add_argument("--preserve-native-revisions", action="store_true",
                        help="Extract every label from its original reviewed native image revision")
    parser.add_argument("--pixels-per-region", type=int, default=1024)
    parser.add_argument("--feature-version", default=GENERIC_FEATURE_VERSION,
                        choices=(GENERIC_FEATURE_VERSION, RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION))
    parser.add_argument("--adapters", nargs="+", choices=tuple(MODEL_VARIANTS),
                        default=tuple(MODEL_VARIANTS),
                        help="Production optical adapters to fit; maximal feature extraction is unchanged")
    parser.add_argument("--texture-sigma", type=float, default=2.0)
    parser.add_argument("--stage-workers", type=int, default=4)
    parser.add_argument("--extraction-workers", type=int, default=4)
    parser.add_argument("--opencv-threads", type=int, default=6)
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--regularization-grid",
        nargs="+",
        type=float,
        default=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    )
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.65)
    parser.add_argument(
        "--change-evidence-ablation",
        action="store_true",
        help=(
            "Also fit a non-deployable generic control without DarSIA structured "
            "change-evidence channels and compare complete-run holdouts."
        ),
    )
    parser.add_argument(
        "--fit-mode",
        choices=("pure", "mil"),
        default="pure",
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        choices=("gas", "aqueous", "co2_presence", "gas_within_co2"),
        default=("gas", "aqueous", "co2_presence", "gas_within_co2"),
    )
    parser.add_argument("--weak-strength", type=float, default=0.35)
    parser.add_argument("--positive-top-fraction", type=float, default=0.10)
    parser.add_argument("--mil-max-iters", type=int, default=160)
    parser.add_argument("--gas-selection-policy", choices=("regional-oof-v1", "fixed-display-phase-anchors-v1"), default="regional-oof-v1")
    parser.add_argument("--fit-excluded-runs", nargs="+", default=(),
                        help="Frozen V5 chemistry OOD runs excluded from all fit and CV folds")
    parser.add_argument(
        "--resume-completed-heads",
        action="store_true",
        help=(
            "Reuse a valid per-head model in local result_staging when its "
            "campaign, fit mode, target, and training inventory match exactly."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    global CAMPAIGN_ID
    parser = build_parser()
    args = parser.parse_args(argv)
    CAMPAIGN_ID = str(args.campaign_id).strip() or CAMPAIGN_ID
    if not args.bundle:
        args.bundle = [
            (str(decisions), str(manifest)) for decisions, manifest in DEFAULT_BUNDLES
        ]
    summary = train_campaign(args)
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "output_root": str(Path(args.output_root).resolve()),
                "runs": summary["runs"],
                "regions": summary["inventory"]["regions"],
                "pixel_rows": summary["inventory"]["pixel_rows"],
                "seconds": summary["seconds"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
