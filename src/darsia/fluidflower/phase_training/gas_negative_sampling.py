"""Frozen, bounded hard-negative sampling inside explicit gas-absence reviews.

Saved predictions select difficult examples; only the human regional exclusion
supplies supervision. Ordinary uniform rows and each region's total loss weight
are preserved. The sampler invents no labels: the annotation supplies gas-only
negative targets without choosing aqueous CO2 versus water. No new phase map
is inferred here.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .logistic import PHASE_GAS_ABSENT_REGION, PHASE_AQUEOUS_PRESENT_GAS_ABSENT_REGION
from .review_regions import annotation_mask, sample_mask
from ..project import sha256_file
from ..review_storage import resolve_review_path
from ..tools.phase_optical_cache_compatibility import canonical_implementation

POLICY = "certified-gas-absence-hard-negative-mixture-v1"
FIELD = "training_sampling"
EXPLICIT_ABSENCE = {PHASE_GAS_ABSENT_REGION, PHASE_AQUEOUS_PRESENT_GAS_ABSENT_REGION}
MAX_EXTRA = 1024
MAX_UNIFORM_FRACTION = .001


def read(path):
    return json.loads(resolve_review_path(path).read_text(encoding="utf-8-sig"))


def contract(record):
    value = copy.deepcopy(record)
    value.pop("contract_sha256", None)
    for source in value.get("source_files", {}).values():
        source.pop("path", None)  # snapshots may relocate; content identity may not
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sampling_inventory(regions):
    selected = [{"region_id": row.region_id, "contract": contract(row.payload[FIELD])}
                for row in regions if FIELD in row.payload]
    if not selected:
        return {}
    return {"sampling_policy": POLICY, "sampling_sha256": digest(selected),
            "sampling_regions": len(selected),
            "sampling_extra_rows": sum(len(row["contract"].get("rows", [])) for row in selected),
            "sampling_evaluation_scope": "Model-guided training sampling; saved selector may have seen held-out runs. Whole-run CV is not independent selector validation."}


def select_extra(mask, domain, gas, *, limit, annotation_id, model_sha256):
    """Return ordinary coordinates plus deterministic, distinct gas negatives."""
    if mask.shape != domain.shape or mask.shape != gas.shape:
        raise ValueError("Sampling masks must share the exact native geometry")
    rows, columns = sample_mask(mask, limit=limit, annotation_id=annotation_id)
    pool = mask & domain & gas
    pool[rows, columns] = False
    if len(rows)/int(mask.sum()) >= MAX_UNIFORM_FRACTION or not pool.any():
        return rows, columns, np.empty(0, np.int64), np.empty(0, np.int64)
    extra_rows, extra_columns = sample_mask(pool, limit=min(MAX_EXTRA, int(limit)),
        annotation_id=f"{POLICY}|{annotation_id}|{model_sha256}")
    return rows, columns, extra_rows, extra_columns


def file_record(path):
    path = resolve_review_path(path)
    return dict(path=str(path), original_path=str(path), sha256=sha256_file(path))


def checked_sources(record, cache=None):
    """Check all frozen content and the model/native/map/context relationship."""
    key = digest({name: record.get(name) for name in ("source_model_id", "source_model_sha256",
        "physics_runtime", "dataset", "run", "frame_index", "canvas_wh", "content_box_xywh", "native_elapsed_h")}
        | {"files": contract(record).get("source_files")})
    if cache is not None and key in cache:
        return cache[key]
    sources = record["source_files"]
    for source in sources.values():
        if sha256_file(resolve_review_path(source["path"])) != source["sha256"]:
            raise ValueError("Frozen gas-negative sampling source changed")
    required = {"summary", "co2_head", "gas_head", "profile", "binding", "manifest", "native", "context", "gas_map", "sampler_code"}
    if not required <= set(sources):
        raise ValueError("Incomplete gas-negative sampling provenance")
    if canonical_implementation(__name__, sources["sampler_code"]["sha256"]) != canonical_implementation(__name__, sha256_file(Path(__file__))):
        raise ValueError("Frozen sampling implementation differs from current extraction")
    identity = hashlib.sha256()
    for role, name in (("summary", "summary.json"), ("co2_head", "co2_presence_signal.json"), ("gas_head", "gas_within_co2_signal.json")):
        identity.update(name.encode())
        identity.update(sources[role]["sha256"].encode())
    if identity.hexdigest() != record["source_model_sha256"]:
        raise ValueError("Saved prediction model identity differs")
    summary = read(sources["summary"]["path"])
    if str(summary.get("model_id") or Path(record["source_model_root"]).name) != record["source_model_id"]:
        raise ValueError("Saved prediction model ID differs")
    binding, manifest = (read(sources[role]["path"]) for role in ("binding", "manifest"))
    for value in (binding, manifest):
        if (value.get("model_id"), value.get("model_sha256"), value.get("physics_runtime")) != (
                record["source_model_id"], record["source_model_sha256"], record["physics_runtime"]):
            raise ValueError("Saved prediction binding differs")
    if not binding.get("completed_utc"):
        raise ValueError("Saved prediction campaign is incomplete")
    jobs = [job for job in manifest["jobs"] if (job["dataset"], job["run"]) == (record["dataset"], record["run"])]
    if len(jobs) != 1:
        raise ValueError("Sampling target experiment is ambiguous")
    job = jobs[0]
    expected_map = (Path(job["phase_map_output_dir"]) / "native_phase_maps" / record["run"] /
                    ("A_" + job["params_sha256"][:16]) / f"frame_{record['frame_index']:03d}.png")
    expected_native = Path(job["review_output_dir"]) / "native_frames" / record["run"] / "manifest.json"
    if resolve_review_path(expected_map) != resolve_review_path(sources["gas_map"]["original_path"]) or resolve_review_path(expected_native) != resolve_review_path(sources["native"]["original_path"]):
        raise ValueError("Sampling map/native paths differ from the saved campaign")
    profile, native = (read(sources[role]["path"]) for role in ("profile", "native"))
    entry = profile["runs"][f"{record['dataset']}/{record['run']}"]
    if entry["sha256"] != sources["context"]["sha256"] or entry["native_manifest_sha256"] != sources["native"]["sha256"]:
        raise ValueError("Physical context and source timeline differ")
    if list(native["canvas_wh"]) != record["canvas_wh"] or list(native["content_box_xywh"]) != record["content_box_xywh"]:
        raise ValueError("Saved prediction native geometry differs")
    frame = native["frame_sources"][record["frame_index"]]
    if int(frame["frame_index"]) != record["frame_index"] or float(frame["elapsed_h"]) != record["native_elapsed_h"]:
        raise ValueError("Saved prediction frame identity differs")
    with np.load(resolve_review_path(sources["context"]["path"]), allow_pickle=False) as archive:
        domain = np.asarray(archive["domain"], bool)
    with Image.open(resolve_review_path(sources["gas_map"]["path"])) as image:
        if image.size != tuple(native["canvas_wh"]):
            raise ValueError("Saved prediction canvas geometry differs")
        x, y, width, height = native["content_box_xywh"]
        rgba = np.array(image.convert("RGBA"))[y:y+height, x:x+width]
    if domain.shape != (height, width) or rgba.shape != (height, width, 4):
        raise ValueError("Saved prediction and domain native grids differ")
    gas = (rgba[..., 3] > 0) & np.all(rgba[..., :3] == [242, 76, 157], axis=-1)
    if gas[~domain].any():
        raise ValueError("Saved gas map contains pixels outside its domain")
    result = domain, gas
    if cache is not None:
        cache[key] = result
    return result


def sample_region(region, mask, native_manifest, *, limit, cache=None):
    """Validate the frozen contract, then append extras to the ordinary sample."""
    ordinary = sample_mask(mask, limit=limit, annotation_id=region.region_id)
    record = region.payload.get(FIELD)
    if record is None:
        return ordinary
    if record.get("policy") != POLICY or record.get("contract_sha256") != digest(contract(record)):
        raise ValueError("Invalid frozen gas-negative sampling contract")
    if region.phase_class not in EXPLICIT_ABSENCE or region.supervision_weight <= 0:
        raise ValueError("Targeted sampling requires explicit reviewed gas absence")
    if record["uniform_limit"] != limit or record["region_id"] != region.region_id or record["frame_index"] != region.frame_index:
        raise ValueError("Sampling owner/frame/limit differs from its frozen contract")
    if record["status"] == "uniform_fallback":
        if record.get("rows") or record.get("columns"):
            raise ValueError("Uniform fallback cannot contain targeted coordinates")
        return ordinary
    if record["status"] != "enriched" or sha256_file(resolve_review_path(native_manifest)) != record["source_files"]["native"]["sha256"]:
        raise ValueError("Sampling source differs from current registered image")
    domain, gas = checked_sources(record, cache)
    rows, columns, extra_rows, extra_columns = select_extra(mask, domain, gas, limit=limit,
        annotation_id=region.region_id, model_sha256=record["source_model_sha256"])
    if record["rows"] != extra_rows.tolist() or record["columns"] != extra_columns.tolist():
        raise ValueError("Targeted sampling coordinates differ from deterministic certified selection")
    return np.r_[rows, extra_rows], np.r_[columns, extra_columns]


def resolve_document_sources(document, source_root):
    """Resolve portable dependency paths in memory without rewriting evidence."""
    value = copy.deepcopy(document)
    root = resolve_review_path(source_root)
    for decision in value.get("decisions", {}).values():
        for field in ("annotations", "phase_quiz_answers"):
            for item in decision.get(field, []):
                for source in item.get(FIELD, {}).get("source_files", {}).values():
                    path = Path(source["path"])
                    if not path.is_absolute():
                        resolved = (root / path).resolve()
                        if not resolved.is_relative_to(root / "sampling_sources"):
                            raise ValueError("Relative sampling dependency escaped its input snapshot")
                        source["path"] = str(resolved)
                    else:
                        source["path"] = str(resolve_review_path(path))
    return value


def freeze_document_sources(document, snapshot_root, source_root=None):
    """Make dependency snapshots self-contained; preserve their original bytes."""
    value = (copy.deepcopy(document) if source_root is None else
             resolve_document_sources(document, source_root))
    for decision in value.get("decisions", {}).values():
        for field in ("annotations", "phase_quiz_answers"):
            for item in decision.get(field, []):
                record = item.get(FIELD)
                if record is None:
                    continue
                if record.get("contract_sha256") != digest(contract(record)):
                    raise ValueError("Sampling contract changed before input freezing")
                for source in record.get("source_files", {}).values():
                    path = resolve_review_path(source["path"])
                    checksum = sha256_file(path)
                    if checksum != source["sha256"] or path.suffix.lower() not in {".json", ".png", ".npz", ".py"}:
                        raise ValueError("Invalid sampling snapshot source")
                    target = Path(snapshot_root) / "sampling_sources" / (checksum + path.suffix.lower())
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        shutil.copy2(path, target)
                    if sha256_file(target) != checksum:
                        raise ValueError("Sampling snapshot changed during copying")
                    source["path"] = target.relative_to(Path(snapshot_root)).as_posix()
    return value


def enrich_training_documents(documents, regions, sources, parent, *, uniform_limit=1024):
    """Derive sampling evidence; unavailable saved maps are recorded fallbacks."""
    from ..phase_runtime import load_physics_profile, physics_runtime_identity
    from ..phase_validation import _model_identity
    parent = resolve_review_path(parent)
    values = copy.deepcopy(documents)
    records, audit, cached = {}, [], {}
    source_by_run = {source.run: source for source in sources}
    initialization_error = None
    try:
        identity = _model_identity(parent)
        profile = load_physics_profile(parent)
        runtime = physics_runtime_identity(profile)
        PROFILE_FILES = ("phase_native_reference.json", "phase_facies_reference.json",
            "phase_reviewed_reference.json", "phase_reference_support.json", "phase_presence_context.json",
            "phase_physics.json", "phase_plume_support.json")
        profile_file = next(parent / name for name in PROFILE_FILES if (parent / name).is_file())
        manifest_file = parent / "validation/postflight_manifest.json"
        manifest = read(manifest_file)
        jobs = {job["run"]: job for job in manifest["jobs"]}
    except (OSError, ValueError, KeyError, StopIteration) as error:
        initialization_error = f"{type(error).__name__}: {error}"
    for region in regions:
        if region.phase_class not in EXPLICIT_ABSENCE or region.supervision_weight <= 0:
            continue
        native_path = resolve_review_path(source_by_run[region.run].asset_manifest_path)
        native = read(native_path)
        mask = annotation_mask(region.payload, canvas_wh=native["canvas_wh"], content_box=native["content_box_xywh"])
        uniform_rows = min(int(mask.sum()), int(uniform_limit))
        if not mask.any() or uniform_rows/int(mask.sum()) >= MAX_UNIFORM_FRACTION:
            continue
        record = dict(policy=POLICY, status="uniform_fallback", region_id=region.region_id,
            frame_index=region.frame_index, uniform_limit=int(uniform_limit),
            maximum_extra_rows=min(MAX_EXTRA, int(uniform_limit)), maximum_uniform_fraction=MAX_UNIFORM_FRACTION,
            uniform_rows_preserved=True, region_owner_and_total_weight_preserved=True, rows=[], columns=[])
        try:
            if initialization_error:
                raise ValueError(initialization_error)
            job = jobs[region.run]
            entry = profile["runs"][f"{region.dataset}/{region.run}"]
            saved_native = resolve_review_path(Path(job["review_output_dir"]) / "native_frames" / region.run / "manifest.json")
            if sha256_file(saved_native) != sha256_file(native_path):
                raise ValueError("Current annotation timeline differs from saved prediction")
            map_path = (Path(job["phase_map_output_dir"]) / "native_phase_maps" / region.run /
                        ("A_"+job["params_sha256"][:16]) / f"frame_{region.frame_index:03d}.png")
            paths = dict(summary=parent / "summary.json", co2_head=parent / "models/generic/co2_presence_signal.json",
                gas_head=parent / "models/generic/gas_within_co2_signal.json", profile=profile_file,
                binding=parent / "validation/validation_binding.json", manifest=manifest_file,
                native=saved_native, context=parent / entry["path"], gas_map=map_path, sampler_code=Path(__file__))
            record.update(source_files={role: file_record(path) for role, path in paths.items()},
                source_model_root=str(parent), source_model_id=identity[0], source_model_sha256=identity[1],
                physics_runtime=runtime, dataset=region.dataset, run=region.run,
                canvas_wh=list(native["canvas_wh"]), content_box_xywh=list(native["content_box_xywh"]),
                native_elapsed_h=float(native["frame_sources"][region.frame_index]["elapsed_h"]))
            domain, gas = checked_sources(record, cached)
            _, _, rr, cc = select_extra(mask, domain, gas, limit=int(uniform_limit),
                annotation_id=region.region_id, model_sha256=identity[1])
            if not len(rr):
                raise ValueError("No additional certified predicted-gas pixels")
            record.update(status="enriched", rows=rr.tolist(), columns=cc.tolist())
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            record = {key: value for key, value in record.items() if key not in {
                "source_files", "source_model_root", "source_model_id", "source_model_sha256", "physics_runtime",
                "dataset", "run", "canvas_wh", "content_box_xywh", "native_elapsed_h"}}
            record["fallback_reason"] = f"{type(error).__name__}: {error}"
        record["contract_sha256"] = digest(contract(record))
        records[region.region_id] = record
        audit.append(dict(region_id=region.region_id, status=record["status"], extra_rows=len(record["rows"]),
                          fallback_reason=record.get("fallback_reason")))
    # Replace stale inherited sampling only in this derived view, never in live
    # evidence or preserved originals. ID formatting matches the shared loader.
    from .trainer import _geometry_key
    attached = set()
    for document in values:
        for run, decision in document.get("decisions", {}).items():
            for field, kind in (("annotations", "manual"), ("phase_quiz_answers", "quiz")):
                for item in decision.get(field, []):
                    item.pop(FIELD, None)
                    identifier = item.get("id") or item.get("proposal_id") or _geometry_key(run.lower(), item)[:16]
                    key = f"{run.lower()}:{kind}:{identifier}"
                    if key in records:
                        item[FIELD] = records[key]
                        attached.add(key)
    if set(records) != attached:
        raise ValueError("Sampling records could not be bound to their exact source annotations")
    return values, dict(policy=POLICY, regions=audit, enriched_regions=sum(row["status"] == "enriched" for row in audit),
        extra_rows=sum(row["extra_rows"] for row in audit), uniform_sampling_preserved=True,
        selector_independent_validation=False)
