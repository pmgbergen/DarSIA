"""Explicit, metadata-only conversion of an existing DarSIA series.

No model is fitted, image read, geometry edited or GUI job launched. Scientific
admission stays with the existing application. This module does not add native
parameters or select a model by age, name, ranking or filesystem time.
"""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import difflib
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tomllib
import traceback
import uuid

from darsia.presets.workflows.config import utils

MAX_MANIFEST = 1024 * 1024
MAX_METADATA = 16 * 1024 * 1024
MAX_ENTRIES = 128
REGISTRY_ARRAYS = {"data_interval", "data_window", "data_time", "data_path", "roi", "format", "color_path", "color_range", "color_channel"}


def native_merged(paths: list[Path]) -> dict:
    """Match scalar deep merging plus native named registry accumulation."""
    merged = {}
    arrays = {}
    for path in paths:
        part = tomllib.loads(path.read_text(encoding="utf-8-sig"))
        for key in REGISTRY_ARRAYS:
            if key in part:
                values = part.pop(key)
                assert isinstance(values, list), (path, key)
                arrays.setdefault(key, []).extend(values)
        utils._deep_merge(merged, part)
    names = set()
    for key, values in arrays.items():
        local = [value["name"] for value in values]
        assert len(local) == len(set(local)), (paths, key, local)
        if key.startswith("data_"):
            assert not names.intersection(local), (paths, key)
            names.update(local)
        merged[key] = values
    return merged


def _bytes(path, limit=MAX_METADATA):
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Metadata exceeds its explicit byte limit: " + str(path))
    return raw


def _descriptor(path, raw=None):
    path = Path(path)
    raw = _bytes(path) if raw is None else raw
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _bound_source(bound):
    raw = _bytes(bound["path"])
    if _descriptor(bound["path"], raw) != bound:
        raise ValueError("Conversion input changed: " + str(bound["path"]))
    return raw


def _relative(root, value):
    """Metadata targets must remain within this exact series, including links."""
    text = str(value).replace("\\", "/")
    relative = PurePosixPath(text)
    if (relative.is_absolute() or ":" in text
            or any(part in {"", ".", ".."} for part in text.split("/"))):
        raise ValueError("A series-relative conversion target is required")
    path = Path(root).joinpath(*relative.parts)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Conversion target escapes its series")
    return path


def _input(root, value):
    path = Path(value)
    return path if path.is_absolute() else _relative(root, value)


def _same(left, right):
    from .series_method import _same_values
    return _same_values(left, right)


def _list(folder):
    """Inspect only one explicit metadata folder; no recursive source scan."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    entries = []
    with os.scandir(folder) as stream:
        for entry in stream:
            if len(entries) == MAX_ENTRIES:
                raise ValueError("Explicit metadata folder exceeds 128 entries: " + str(folder))
            entries.append(Path(entry.path))
    return sorted(entries, key=lambda p: p.name)


def _toml(root, value, sources):
    path = _input(root, value)
    raw = _bytes(path, MAX_MANIFEST)
    sources[str(path)] = _descriptor(path, raw)
    return path, tomllib.loads(raw.decode("utf-8-sig"))


def inspect_protocol_extras(folder):
    """Report extra metadata without making it a Protocols completion criterion.

    A path blacklist excludes photographs. It is never evidence that the entire
    experimental run belongs in runs.excluded.
    """
    result = []
    for path in _list(folder):
        if not path.is_file():
            continue
        if path.name.startswith("blacklist_protocol") and path.suffix.lower() == ".csv":
            raw = _bytes(path, MAX_MANIFEST)
            reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
            rows = list(reader)
            result.append(dict(file=_descriptor(path, raw), role="image_blacklist",
                columns=reader.fieldnames or [], row_count=len(rows),
                affects_run_exclusion=False, ignored_for_protocol_completion=True))
        elif not (path.name.startswith("imaging_protocol")
                  or path.name in {"injection_protocol.csv", "pressure_temperature_protocol.csv"}):
            result.append(dict(path=str(path), role="other_protocol_metadata",
                size_bytes=path.stat().st_size, ignored_for_protocol_completion=True))
    return result


def inspect_models(root, model_roots, series_id):
    """Reuse the real library source identity; do not run model admission."""
    from .phase_model_library import source_identity, library_root, resolve_reference
    from .series_paths import series_key
    approved, invalid = [], []
    seen_ids = set()
    for value in model_roots:
        path = _input(root, value)
        try:
            summary = _bytes(path / "summary.json")
            document = json.loads(summary.decode("utf-8-sig"))
            if not isinstance(document, dict):
                raise ValueError("Saved summary must be a JSON object")
            declared_id = document.get("model_id") or document.get("campaign_id")
            if not isinstance(declared_id, str) or not declared_id.strip():
                raise ValueError("Saved summary has no explicit immutable model ID")
            datasets = (document.get("inventory") or {}).get("datasets") or []
            if datasets and {series_key(v) for v in datasets} != {series_key(series_id)}:
                raise ValueError("Saved model belongs to another series")
            identity = source_identity(path)
            if identity.get("complete_heads") is not True:
                raise ValueError("Saved model has an incomplete original head set")
            if identity["model_id"] in seen_ids:
                raise ValueError("Two explicit sources have the same model ID; choose the canonical source explicitly")
            seen_ids.add(identity["model_id"])
            # Bind exactly the metadata files returned by the original indexer.
            bound = [_descriptor(path / name) for name in identity["metadata_sha256"]]
            if any(item["sha256"] != identity["metadata_sha256"][
                Path(item["path"]).relative_to(path).as_posix()] for item in bound):
                raise ValueError("Saved model metadata changed after original indexing inspection")
            filename = re.sub(r"[^A-Za-z0-9_.-]", "_", identity["model_id"]) + ".json"
            target = library_root(root) / filename
            state = "new_reference"
            reference = None
            if target.exists():
                raw = _bytes(target, MAX_MANIFEST)
                entry = json.loads(raw.decode("utf-8-sig"))
                if entry.get("source_identity") != identity or resolve_reference(target, series_root=root).resolve() != path.resolve():
                    raise ValueError("Existing library pointer differs; preserve it for explicit review")
                state, reference = "existing_reference", _descriptor(target, raw)
            approved.append(dict(source=str(path), source_identity=identity,
                metadata_sources=bound, target_relative=str(target.relative_to(root)).replace("\\", "/"),
                state=state, existing_reference=reference))
        except (OSError, ValueError, KeyError, TypeError) as error:
            invalid.append(dict(source=str(path), reason=str(error), indexed=False))
    return approved, invalid


def inspect_review_registration(root):
    """Existing collection registrations stay literal; do not repair by name."""
    from .series_paths import phase_model_root, series_folder_path
    folder = (series_folder_path(root, "review_collections") if phase_model_root(root) is not None
              else Path(root) / "phase_review_collections")
    path = folder / "index.json"
    if not path.is_file():
        return dict(state="not_registered", entries=[], pending=[])
    raw = _bytes(path, MAX_MANIFEST)
    value = json.loads(raw.decode("utf-8-sig"))
    if value.get("format") != "fluidflower-phase-review-collection-v1":
        raise ValueError("Unknown review collection index format")
    rows, pending = [], []
    for row in value.get("collections", []):
        manifest = row.get("manifest") or {}
        rows.append(deepcopy(row))
        # Only report this exact original pointer. No NAS same-name fallback.
        target = Path(manifest.get("path", ""))
        reason = "Original review pointer unavailable; relocation requires exact model/SHA identity"
        if manifest.get("path") and target.is_file():
            bound = _bytes(target, MAX_MANIFEST)
            document = json.loads(bound.decode("utf-8-sig"))
            keys = ("source_model_id", "source_model_sha256", "source_physics_runtime")
            if (hashlib.sha256(bound).hexdigest() != manifest.get("sha256")
                    or any(document.get(key) != row.get(key) for key in keys)):
                reason = "Original review manifest metadata differs from its registered identity"
            else:
                reason = "Original review metadata matches; original current-model/runtime admission remains pending"
        pending.append(dict(source_model_id=row.get("source_model_id"),
            original_manifest=deepcopy(manifest), reason=reason))
    return dict(state="existing_registration", index=_descriptor(path, raw),
                entries=rows, pending=pending, changed=False)


def _conversion_roots(root, document, choices):
    """Keep explicit roots and name only literal, existing legacy declarations.

    This is metadata conversion, not job admission. An undeclared result root
    is not invented: the original job resolver still refuses consumed inputs
    outside the declared roots. Explicit root choices remain SHA-bound in the
    conversion request and receipt.
    """
    from .store_map import manifest_roots
    existing = document.get("roots")
    if existing is not None and not isinstance(existing, dict):
        raise ValueError("Existing [roots] must be a table")
    declared = deepcopy(existing or {})
    origins = {alias: "series.toml.roots." + alias for alias in declared}
    legacy = document.get("legacy") or {}
    if not isinstance(legacy, dict):
        raise ValueError("Existing [legacy] must be a table")
    for alias, field in (("raw", "raw_inputs"), ("legacy_results", "previous_results"),
                         ("runtime_inputs", "runtime_inputs"), ("article_inputs", "article_inputs")):
        value = legacy.get(field)
        if alias not in declared and isinstance(value, str) and value.strip():
            declared[alias] = value
            origins[alias] = "series.toml.legacy." + field
    selected = choices.get("roots", {})
    if not isinstance(selected, dict):
        raise ValueError("Explicit roots choices must be a table")
    for alias, value in selected.items():
        declared[alias] = deepcopy(value)
        origins[alias] = "choices.roots." + str(alias)
    if "series" not in declared:
        declared["series"] = "."
        origins["series"] = "current series manifest directory"
    if declared["series"] != ".":
        raise ValueError("Conversion [roots].series must be '.'")
    check = {"schema_version": document.get("schema_version"),
             "series_id": document.get("series_id"), "roots": declared}
    resolved = manifest_roots(root, check)
    aliases = list(resolved)
    for index, alias in enumerate(aliases):
        if any(resolved[alias] == resolved[other] for other in aliases[:index]):
            raise ValueError("Ambiguous conversion roots resolve to the same folder: " + alias)
    return declared, dict(declared=deepcopy(declared),
        resolved={alias: str(path) for alias, path in resolved.items()}, sources=origins,
        not_declared=[alias for alias in ("raw", "legacy_results") if alias not in declared],
        admission="Metadata only; original job input resolver must admit every consumed input",
        historical_legacy_fields_preserved=True, payloads_opened=False)


def plan_series_conversion(root, choices):
    """Prepare data-only changes from explicit per-series inputs; never write."""
    import toml
    root = Path(root).resolve()
    if choices.get("schema_version") != 1:
        raise ValueError("Conversion choices require schema_version=1")
    path = root / "series.toml"
    raw = _bytes(path, MAX_MANIFEST)
    before = tomllib.loads(raw.decode("utf-8-sig"))
    if before.get("series_id") != choices.get("series_id"):
        raise ValueError("Conversion choices belong to another series")
    expected = choices.get("expected_series_sha256")
    if expected != hashlib.sha256(raw).hexdigest():
        # An already applied idempotent plan is permitted only with its exact
        # existing options identity, never because a series ID happens to match.
        mark = (before.get("provenance") or {}).get("conversion_" + choices.get("operation", ""))
        request_sha = hashlib.sha256(json.dumps(choices, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if not isinstance(mark, dict) or mark.get("choices_sha256") != request_sha:
            raise ValueError("Starting series manifest differs from its explicit expected SHA")
    operation = choices.get("operation", "")
    if re.fullmatch(r"[A-Za-z0-9_]+", operation) is None:
        raise ValueError("One explicit conversion operation name is required")
    after = deepcopy(before)
    after["roots"], root_report = _conversion_roots(root, before, choices)
    sources = {str(path): _descriptor(path, raw)}
    paths, runs = before.get("paths") or {}, before.get("runs") or {}
    ids = runs.get("ids") or []
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise ValueError("The series needs explicit distinct run IDs")
    active = choices.get("active_method")
    if active is not None:
        if active not in {"phase_model", "colour_path"}:
            raise ValueError("Unknown explicit method")
        method = after.setdefault("method", {})
        if method.get("active") not in (None, "", active):
            raise ValueError("Existing method conflicts with explicit conversion input")
        method["active"] = active
    excluded = choices.get("excluded_runs") or {}
    if excluded:
        evidence = choices.get("run_exclusion_sources") or []
        if not evidence:
            raise ValueError("Run exclusions require explicit documented source bindings")
        for bound in evidence:
            raw_evidence = _bound_source(bound)
            evidence_path = Path(bound["path"])
            if evidence_path.name.startswith("blacklist_protocol"):
                raise ValueError("Image blacklist is not documented run-exclusion evidence")
            if evidence_path.suffix.lower() == ".csv":
                columns = csv.DictReader(io.StringIO(raw_evidence.decode("utf-8-sig"))).fieldnames
                if columns == ["path"]:
                    raise ValueError("Photograph path blacklist cannot exclude an entire run")
            sources[bound["path"]] = deepcopy(bound)
        if (set(excluded)-set(ids) or any(not isinstance(v, str) or not v.strip() for v in excluded.values())):
            raise ValueError("Run exclusions must name registered runs and literal reasons")
        current = after.setdefault("runs", {}).setdefault("excluded", {})
        for key, reason in excluded.items():
            if key in current and current[key] != reason:
                raise ValueError("Existing run exclusion reason differs")
            current[key] = reason
    selection = choices.get("model_choice")
    pending = []
    if selection is not None:
        if not isinstance(selection, dict) or not selection.get("authority"):
            raise ValueError("A selected model needs explicit project-owner authority")
        model = _input(root, selection["path"])
        selected, errors = inspect_models(root, [selection["path"]], before["series_id"])
        if len(selected) != 1 or errors:
            raise ValueError("Named model has no complete matching original library identity")
        identity = selected[0]["source_identity"]
        if identity["model_id"] != selection["model_id"]:
            raise ValueError("Named model differs from its original immutable ID")
        method = after.setdefault("method", {}).setdefault("phase_model", {})
        relative = str(model.relative_to(root)).replace("\\", "/")
        if method.get("model_version") not in (None, "", relative):
            raise ValueError("Existing selected model differs")
        method["model_version"] = relative
    elif active == "phase_model" and not (after.get("method", {}).get("phase_model") or {}).get("model_version"):
        pending.append("needs_decision: no model selected; library references do not select a model")
    reference = choices.get("results_reference")
    if reference is not None:
        if not isinstance(reference, dict) or not reference.get("authority"):
            raise ValueError("Frozen result reference requires explicit authority")
        value = deepcopy(reference["value"])
        existing = (after.get("results") or {}).get("reference")
        if existing not in (None, value):
            raise ValueError("Existing frozen result reference differs")
        after.setdefault("results", {})["reference"] = value
    writes, merge_status, protocol_status, field_status = [], [], [], []
    common_path, _common = _toml(root, paths.get("common_config", "config/common.toml"), sources)
    for run in ids:
        partial_name = (runs.get("config_files") or {}).get(run, str(paths.get("run_configs", "config/runs")) + "/" + run + ".toml")
        partial_path, _partial = _toml(root, partial_name, sources)
        # This is the original 20260921 native function, not a new merge policy.
        merged = native_merged([common_path, partial_path])
        target_name = (runs.get("merged_files") or {}).get(run, str(paths.get("merged_configs", "config/merged")) + "/" + run + ".toml")
        target = _relative(root, target_name)
        encoded = toml.dumps(merged).encode("utf-8")
        if len(encoded) > MAX_MANIFEST:
            raise ValueError("Merged configuration exceeds metadata budget")
        existing_target = target.exists()
        if existing_target:
            actual = _bytes(target, MAX_MANIFEST)
            sources[str(target)] = _descriptor(target, actual)
            semantic = _same(tomllib.loads(actual.decode("utf-8-sig")), merged)
            difference = "".join(difflib.unified_diff(actual.decode("utf-8-sig").splitlines(True),
                encoded.decode().splitlines(True), fromfile="existing/" + target.name, tofile="regenerated/" + target.name))
            merge_status.append(dict(run=run, state="existing_preserved", semantic_exact=semantic,
                byte_exact=actual == encoded, exact_unified_diff=difference, written=False))
            if not semantic:
                pending.append("Existing merged configuration differs: " + run + "; no overwrite")
        else:
            writes.append(dict(step="merged_configurations", relative_path=str(target.relative_to(root)).replace("\\", "/"),
                               before=None, after_utf8=encoded.decode()))
            after.setdefault("runs", {}).setdefault("merged_files", {})[run] = str(target.relative_to(root)).replace("\\", "/")
            merge_status.append(dict(run=run, state="missing_planned_original_merge", written=False))
        observed = tomllib.loads(actual.decode("utf-8-sig")) if existing_target else merged
        depth_path = (observed.get("depth") or {}).get("depth_map")
        rig_path = (observed.get("rig") or {}).get("path")
        field_status.append(dict(run=run, configuration_exists=existing_target,
            depth_map=dict(path=depth_path, exists=bool(depth_path) and _input(root, depth_path).is_file(),
                           payload_opened=False, SHA_verified=False),
            rig_model=dict(path=rig_path, exists=bool(rig_path) and _input(root, rig_path).is_dir(),
                           payload_opened=False, SHA_verified=False)))
        protocol_status.append(dict(run=run, ignored_extras=inspect_protocol_extras(
            root / str(paths.get("protocols", "protocols")) / run)))
    model_roots = choices.get("model_sources")
    if model_roots is None:
        model_roots = [str(p.relative_to(root)) for p in _list(root / str(paths.get("models", "models")))
                       if p.is_dir() and p.name != "phase"]
    approved, invalid = inspect_models(root, model_roots, before["series_id"])
    for row in approved:
        for bound in row["metadata_sources"]:
            sources[bound["path"]] = bound
    reviews = inspect_review_registration(root)
    if "index" in reviews:
        sources[reviews["index"]["path"]] = reviews["index"]
    for row in reviews["pending"]:
        pending.append(row["reason"])
    workspace_path = root / "geometry_facies/overrides/geometry_facies/current.json"
    workspace = dict(path=str(workspace_path), exists=workspace_path.is_file(),
                     validated=False, computation_started=False)
    if workspace["exists"]:
        workspace_raw = _bytes(workspace_path, MAX_MANIFEST)
        workspace_value = json.loads(workspace_raw.decode("utf-8-sig"))
        workspace.update(file=_descriptor(workspace_path, workspace_raw),
                         format=workspace_value.get("format"),
                         revision_id=workspace_value.get("revision_id"))
        sources[str(workspace_path)] = workspace["file"]
    # Geometry/depth/rig production is deliberately not part of data conversion.
    pending.extend([
        "Geometry adoption/registered revision: original Store/workspace and the series' recorded decision required",
        "Build run model: where absent, explicit existing GUI Setup action only after valid geometry/depth",
        "Native parameters: not configured where absent; no native table is created",
        "Fresh-clone Setup/Calibration/Analysis acceptance remains required",
    ])
    missing_depth = [row["run"] for row in field_status if not row["depth_map"]["exists"]]
    if missing_depth:
        pending.append("Depth: no finished map for " + ", ".join(missing_depth)
                       + "; measurement CSV is not a computed map")
    request_sha = hashlib.sha256(json.dumps(choices, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    name = "conversion_" + operation
    provenance = after.setdefault("provenance", {})
    marker = provenance.get(name)
    desired = dict(choices_sha256=request_sha, specification=choices.get("specification", ""),
                   original_series_sha256=expected, metadata_only=True)
    if marker is not None and marker != desired:
        raise ValueError("Existing conversion provenance conflicts; no replacement")
    provenance[name] = desired
    if "native" not in before and "native" in after:
        raise ValueError("Conversion must not invent a native parameter document")
    changed = not _same(before, after)
    if changed:
        text = toml.dumps(after)
        if b"\r\n" in raw:
            text = text.replace("\n", "\r\n")
        encoded = (b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b"") + text.encode()
        if len(encoded) > MAX_MANIFEST or not _same(tomllib.loads(encoded.decode("utf-8-sig")), after):
            raise ValueError("Manifest serialization changed values or exceeds budget")
        writes.append(dict(step="manifest_completeness", relative_path="series.toml",
                           before=_descriptor(path, raw), after_utf8=encoded.decode("utf-8")))
    return dict(format="darsia-series-conversion-plan-v1", series_root=str(root),
        series_id=before["series_id"], operation=operation, choices=deepcopy(choices),
        inputs=list(sources.values()), writes=writes, roots=root_report, model_references=approved,
        invalid_or_incomplete_models=invalid, merged_configurations=merge_status,
        protocols=protocol_status, configured_fields_stat_only=field_status, review_collections=reviews, workspace=workspace, pending=pending,
        complete=False, scientific_computation=False, adopted=False)


def _new(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def apply_metadata_conversion(plan):
    """Explicit reviewed data migration; nothing runs automatically on GUI load.

    Original sources and pre-existing library pointers are preserved. This is
    not scientific acceptance and does not launch dependent Setup jobs.
    """
    from .phase_model_library import index_model, source_identity
    root = Path(plan["series_root"]).resolve()
    if plan.get("format") != "darsia-series-conversion-plan-v1":
        raise ValueError("Unknown conversion plan")
    for bound in plan["inputs"]:
        _bound_source(bound)
    for write in plan["writes"]:
        target = _relative(root, write["relative_path"])
        if write["before"] is None:
            if target.exists():
                raise FileExistsError("A planned new file appeared: " + str(target))
        elif _descriptor(target) != write["before"]:
            raise ValueError("A conversion target changed before explicit apply")
    # Reject missing/invalid declared roots before backups, indexing or writes.
    manifest_writes = [row for row in plan["writes"] if row["relative_path"] == "series.toml"]
    if len(manifest_writes) > 1:
        raise ValueError("A conversion plan contains duplicate manifest writes")
    manifest_raw = (manifest_writes[0]["after_utf8"].encode("utf-8") if manifest_writes
                    else _bytes(root / "series.toml", MAX_MANIFEST))
    manifest = tomllib.loads(manifest_raw.decode("utf-8-sig"))
    if not isinstance(manifest.get("roots"), dict):
        raise ValueError("A conversion plan must declare [roots] before migration")
    _conversion_roots(root, manifest, {})
    for model in plan["model_references"]:
        if source_identity(model["source"]) != model["source_identity"]:
            raise ValueError("Saved model metadata changed before indexing")
        target = _relative(root, model["target_relative"])
        if model["state"] == "new_reference" and target.exists():
            raise FileExistsError("A library pointer appeared; preserve it for review")
    operation = plan["operation"] + "_" + uuid.uuid4().hex
    backup = root / "config" / "_previous" / operation
    receipt = dict(format="darsia-series-conversion-receipt-v1", series_id=plan["series_id"],
        operation=operation, steps=[], pending=deepcopy(plan["pending"]), metadata_only=True,
        scientific_computation=False, complete=False, started_utc=datetime.now(timezone.utc).isoformat())
    try:
        # Backups precede any replacement, including the manifest.
        for write in plan["writes"]:
            if write["before"] is not None:
                raw = _bound_source(write["before"])
                _new(backup / write["relative_path"], raw)
        for write in plan["writes"]:
            target = _relative(root, write["relative_path"])
            raw = write["after_utf8"].encode("utf-8")
            if write["before"] is None:
                _new(target, raw)
            else:
                temporary = target.with_name("." + target.name + "." + uuid.uuid4().hex + ".tmp")
                # Successful replace consumes the temporary file. Preserve it
                # on failure so the exact interrupted migration stays reviewable.
                _bound_source(write["before"])
                _new(temporary, raw)
                _bound_source(write["before"])
                os.replace(temporary, target)
            step = dict(step=write["step"], file=_descriptor(target, raw), applied=True)
            receipt["steps"].append(step)
            _new(backup / ("step_" + str(len(receipt["steps"])) + ".json"),
                 (json.dumps(step, indent=2) + "\n").encode())
        for model in plan["model_references"]:
            if model["state"] == "existing_reference":
                _bound_source(model["existing_reference"])
                continue
            target = _relative(root, model["target_relative"])
            if target.exists():
                raise FileExistsError("A library pointer appeared before indexing")
            # Exact original indexer. Do not call index_series: it consolidates by
            # deleting superseded pointers, which is outside this conversion.
            result = index_model(root, model["source"])
            if Path(result).resolve() != target.resolve():
                raise ValueError("Original indexer selected an unexpected pointer; preserve and review")
            step = dict(step="model_library_reference", file=_descriptor(result), applied=True,
                        original_indexer="phase_model_library.index_model")
            receipt["steps"].append(step)
            _new(backup / ("step_" + str(len(receipt["steps"])) + ".json"),
                 (json.dumps(step, indent=2) + "\n").encode())
        receipt.update(metadata_steps_applied=True, finished_utc=datetime.now(timezone.utc).isoformat())
        _new(backup / "conversion_receipt.json", (json.dumps(receipt, indent=2) + "\n").encode())
    except Exception:
        receipt.update(failed=True, traceback=traceback.format_exc(),
                       failed_utc=datetime.now(timezone.utc).isoformat())
        _new(backup / "conversion_failed.json", (json.dumps(receipt, indent=2) + "\n").encode())
        raise
    return receipt


def main(argv=None):
    """Explicit conversion CLI; default writes only the chosen preparation output."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series_root", type=Path)
    parser.add_argument("--choices", type=Path, required=True)
    parser.add_argument("--choices-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("Use a fresh conversion output; preserve previous receipts")
    raw = _bytes(args.choices, MAX_MANIFEST)
    if hashlib.sha256(raw).hexdigest() != args.choices_sha256:
        raise ValueError("Explicit conversion choices SHA differs")
    choices = json.loads(raw.decode("utf-8-sig"))
    try:
        plan = plan_series_conversion(args.series_root, choices)
        _new(args.out / "plan.json", (json.dumps(plan, indent=2, allow_nan=False) + "\n").encode())
        result = apply_metadata_conversion(plan) if args.apply else dict(
            preparation_only=True, applied=False, complete=False, pending=plan["pending"])
        _new(args.out / "completed.json", (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    except Exception:
        print("Stopped at " + datetime.now(timezone.utc).isoformat() + ":", flush=True)
        traceback.print_exc()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
