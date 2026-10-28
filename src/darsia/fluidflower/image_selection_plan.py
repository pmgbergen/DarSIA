"""Metadata-only image selection for FluidFlower review, calibration and production.

The same :func:`resolve_selection` API can be called by a UI. The CLI reads a
versioned TOML plan and a JSON frame manifest, then writes one explicit JSON
receipt. It never opens or decodes an image.

Plan example::

    schema_version = 1

    [purposes.review]
    mode = "target_hours"
    target_hours = [-0.05, 0.25, 1.0, 24.0, 96.0]
    tolerance_seconds = 300
    required_end_hour = 96.0

    [purposes.calibration]
    mode = "target_hours"
    target_hours = [0.5, 2.5, 5.0, 24.0, 48.0]
    tolerance_seconds = 300

    [purposes.production]
    mode = "all"
    start_hour = 0.0
    end_hour = 96.0
    required_end_hour = 96.0
    tolerance_seconds = 300

The input may be an existing single-run native manifest with ``frame_sources``
and ``elapsed_h``, or a ``{"runs": [{"run": ..., "frames": [...]}]}`` wrapper.
For target-hour selection, each requested time is matched to the nearest
eligible frame within the stated tolerance. A missing late target is an error
even if ``on_unmatched_target = "skip"`` was chosen for earlier targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tomllib
from typing import Any, Mapping


RECEIPT_FORMAT = "fluidflower-image-selection-receipt-v1"
PURPOSES = ("review", "calibration", "production")


def resolve_quality_time_grid(
    frames: list[Mapping[str, Any]],
    target_hours: list[float],
    *,
    quality: Any,
    early_end_h: float = 3.0,
    early_tolerance_seconds: float = 15.0,
    late_tolerance_seconds: float = 150.0,
    early_replacement_seconds: float = 60.0,
    late_replacement_seconds: float = 600.0,
) -> dict[str, Any]:
    """Select a common camera grid with a declared quality replacement rule.

    ``quality(frame)`` returns a source-bound measurement with ``class`` equal
    to ok/minor/reject. It may lazily measure RAW chart/pillar patches. The
    closest acquisition is considered first; only a rejected acquisition can
    be replaced, and its replacement must be ok. All bounds are relative to
    the requested time, inclusive. An unavailable time remains explicit.
    This is an opt-in selector; legacy selection/late-coverage semantics are
    unchanged. No phase labels or mass values enter the decision.
    """
    targets = [_number(value, "target_hours entry") for value in target_hours]
    if not targets or any(a >= b for a, b in zip(targets, targets[1:])):
        raise ValueError("target_hours must be nonempty and strictly increasing")
    bounds = [early_tolerance_seconds, late_tolerance_seconds,
              early_replacement_seconds, late_replacement_seconds]
    if any(_number(value, "time tolerance") < 0 for value in bounds):
        raise ValueError("Time tolerances must be nonnegative")
    eligible, omitted = _eligible_frames(frames, check_files=False, source_root=None)
    by_index = {i: dict(frame) for i, frame in enumerate(frames)}
    cache: dict[int, dict[str, Any]] = {}

    def assessed(row):
        i = row["input_index"]
        if i not in cache:
            value = dict(quality(by_index[i]))
            if value.get("class") not in {"ok", "minor", "reject"}:
                raise ValueError("Quality must be assessed as ok/minor/reject")
            cache[i] = value
        return cache[i]

    matches, selected = [], {}
    for target in targets:
        early = target <= early_end_h
        tolerance = early_tolerance_seconds if early else late_tolerance_seconds
        replacement_bound = early_replacement_seconds if early else late_replacement_seconds
        ordered = sorted(eligible, key=lambda row: (
            abs(row["elapsed_h"] - target), row["elapsed_h"], row["input_index"]))
        nearest = ordered[0] if ordered else None
        record = dict(target_hour=target, tolerance_seconds=tolerance,
                      replacement_tolerance_seconds=replacement_bound,
                      selected=False, replacement=None)
        if nearest is None or abs(nearest["elapsed_h"] - target)*3600 > tolerance+1e-8:
            record.update(reason="no_camera_image_within_tolerance",
                nearest_available_hour=None if nearest is None else nearest["elapsed_h"])
            matches.append(record)
            continue
        original_quality = assessed(nearest)
        chosen, q = nearest, original_quality
        if q["class"] == "reject":
            chosen = None
            for candidate in ordered:
                if abs(candidate["elapsed_h"] - target)*3600 > replacement_bound+1e-8:
                    break
                cq = assessed(candidate)
                if cq["class"] == "ok":
                    chosen, q = candidate, cq
                    break
            original = {**by_index[nearest["input_index"]],
                        "quality": original_quality,
                        "offset_seconds": (nearest["elapsed_h"]-target)*3600}
            if chosen is None:
                record.update(reason="reject_without_bounded_ok_replacement",
                              rejected_image=original)
                matches.append(record)
                continue
            record["replacement"] = original
        frame = {**by_index[chosen["input_index"]],
                 "input_index": chosen["input_index"],
                 "elapsed_h": chosen["elapsed_h"], "quality": q}
        selected[chosen["input_index"]] = frame
        record.update(selected=True, input_index=chosen["input_index"],
            source_path=frame["source_path"], elapsed_h=frame["elapsed_h"],
            offset_seconds=(frame["elapsed_h"]-target)*3600,
            frame_quality_class=q["class"], quality=q)
        matches.append(record)
    ordered_selected = sorted(selected.values(), key=lambda row:(row["elapsed_h"],row["input_index"]))
    indices = {row["input_index"]: i for i, row in enumerate(ordered_selected)}
    for i, row in enumerate(ordered_selected):
        row["selection_index"] = i
    for row in matches:
        row["selection_index"] = indices.get(row.get("input_index"))
    return dict(format="fluidflower-quality-camera-grid-v1", status="ready",
                target_matches=matches, selected=ordered_selected, omitted=omitted,
                selected_frame_count=len(ordered_selected),
                missing_target_count=sum(not row["selected"] for row in matches),
                quality_assessment_count=len(cache), optional_late_coverage=True)


def _canonical_sha256(document: Mapping[str, Any]) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _purpose_settings(plan: Mapping[str, Any], purpose: str) -> dict[str, Any]:
    if type(plan.get("schema_version")) is not int or plan["schema_version"] != 1:
        raise ValueError("Image-selection plan requires schema_version = 1")
    if purpose not in PURPOSES:
        raise ValueError(f"Purpose must be one of {', '.join(PURPOSES)}")
    purposes = plan.get("purposes")
    raw = purposes.get(purpose) if isinstance(purposes, Mapping) else None
    if not isinstance(raw, Mapping):
        raise ValueError(f"Plan has no [purposes.{purpose}] section")
    settings = dict(raw)
    mode = settings.get("mode")
    if mode not in {"all", "target_hours"}:
        raise ValueError(f"{purpose}: mode must be all or target_hours")
    allowed = {"mode", "tolerance_seconds", "required_end_hour"}
    allowed |= (
        {"target_hours", "on_unmatched_target"}
        if mode == "target_hours"
        else {"start_hour", "end_hour"}
    )
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise ValueError(f"{purpose}: unknown setting(s): {', '.join(unknown)}")
    tolerance = _number(settings.get("tolerance_seconds", 0), "tolerance_seconds")
    if tolerance < 0:
        raise ValueError("tolerance_seconds must be nonnegative")
    settings["tolerance_seconds"] = tolerance
    if mode == "target_hours":
        raw_targets = settings.get("target_hours")
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ValueError(f"{purpose}: target_hours must be a nonempty list")
        targets = [_number(value, "target_hours entry") for value in raw_targets]
        if any(left >= right for left, right in zip(targets, targets[1:])):
            raise ValueError("target_hours must be strictly increasing")
        settings["target_hours"] = targets
        if "start_hour" in settings or "end_hour" in settings:
            raise ValueError("start_hour/end_hour are only supported in all mode")
        action = settings.get("on_unmatched_target", "error")
        if action not in {"error", "skip"}:
            raise ValueError("on_unmatched_target must be error or skip")
        settings["on_unmatched_target"] = action
    else:
        if "target_hours" in settings or "on_unmatched_target" in settings:
            raise ValueError(
                "target_hours/on_unmatched_target require target_hours mode"
            )
        for key in ("start_hour", "end_hour"):
            if key in settings:
                settings[key] = _number(settings[key], key)
        if (
            "start_hour" in settings
            and "end_hour" in settings
            and settings["start_hour"] > settings["end_hour"]
        ):
            raise ValueError("start_hour must not exceed end_hour")
        if "required_end_hour" not in settings:
            raise ValueError("all mode requires required_end_hour for late coverage")
    if "required_end_hour" in settings:
        settings["required_end_hour"] = _number(
            settings["required_end_hour"], "required_end_hour"
        )
    return settings


def _manifest_runs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_runs = manifest.get("runs")
    if raw_runs is None:
        raw_runs = [manifest]
    elif isinstance(raw_runs, Mapping):
        if any(not isinstance(value, Mapping) for value in raw_runs.values()):
            raise ValueError("Every run in manifest.runs must be a mapping")
        raw_runs = [dict(value, run=key) for key, value in raw_runs.items()]
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("Manifest must contain one run or a nonempty runs list")
    runs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_runs):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Run {index} must be a mapping")
        run_id = str(raw.get("run_id") or raw.get("run") or "").strip()
        if not run_id or run_id.casefold() in seen:
            raise ValueError("Every manifest run needs a unique run or run_id")
        seen.add(run_id.casefold())
        frames = raw.get("frames", raw.get("frame_sources"))
        if not isinstance(frames, list):
            raise ValueError(f"{run_id}: expected frames or frame_sources list")
        if any(not isinstance(frame, Mapping) for frame in frames):
            raise ValueError(f"{run_id}: every frame must be a mapping")
        runs.append(
            {
                "run_id": run_id,
                "dataset": str(raw.get("dataset") or manifest.get("dataset") or ""),
                "frames": frames,
            }
        )
    return runs


def _frame_identity(row: Mapping[str, Any], input_index: int) -> dict[str, Any]:
    path = str(
        row.get("source_path") or row.get("path") or row.get("image_path") or ""
    ).strip()
    frame_index = row.get("frame_index", input_index)
    image_id = str(
        row.get("image_id")
        or row.get("frame_id")
        or row.get("source_name")
        or Path(path).name
        or f"frame_{input_index:03d}"
    )
    return {
        "input_index": input_index,
        "frame_index": frame_index,
        "image_id": image_id,
        "source_path": path,
    }


def _eligible_frames(
    frames: list[Mapping[str, Any]], *, check_files: bool, source_root: Path | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    for index, row in enumerate(frames):
        item = _frame_identity(row, index)
        path = item["source_path"]
        reason = None
        if (
            row.get("blacklisted") is True
            or row.get("is_blacklisted") is True
            or row.get("excluded") is True
        ):
            reason = "blacklisted"
        elif row.get("missing") is True or row.get("available") is False:
            reason = "missing_or_unavailable_in_manifest"
        elif not path:
            reason = "missing_path"
        else:
            raw_time = next(
                (
                    row[key]
                    for key in ("elapsed_h", "time_hours", "time_h")
                    if key in row
                ),
                None,
            )
            try:
                item["elapsed_h"] = _number(raw_time, "elapsed_h")
            except ValueError:
                reason = "missing_or_invalid_time"
        if reason is None and check_files:
            resolved = Path(path)
            if not resolved.is_absolute() and source_root is not None:
                resolved = source_root / resolved
            if not resolved.is_file():
                reason = "source_file_missing"
        if reason:
            omitted.append({**item, "reason": reason})
        else:
            eligible.append(item)
    eligible.sort(key=lambda item: (item["elapsed_h"], item["input_index"]))
    unique: list[dict[str, Any]] = []
    seen_paths: dict[str, float] = {}
    for item in eligible:
        key = item["source_path"].replace("\\", "/").casefold()
        if key in seen_paths:
            reason = (
                "duplicate_source_path"
                if item["elapsed_h"] == seen_paths[key]
                else "duplicate_source_path_conflicting_time"
            )
            omitted.append({**item, "reason": reason})
            continue
        seen_paths[key] = item["elapsed_h"]
        unique.append(item)
    return unique, omitted


def _resolve_run(
    run: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    check_files: bool,
    source_root: Path | None,
) -> dict[str, Any]:
    eligible, omitted = _eligible_frames(
        run["frames"], check_files=check_files, source_root=source_root
    )
    selected: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for row in omitted:
        if row["reason"] == "duplicate_source_path_conflicting_time":
            issues.append(
                {
                    "code": "duplicate_source_path_conflicting_time",
                    "source_path": row["source_path"],
                    "message": "The same source path has conflicting acquisition times",
                }
            )
    mode = settings["mode"]
    tolerance = settings["tolerance_seconds"]
    if mode == "all":
        start = settings.get("start_hour", -math.inf)
        end = settings.get("end_hour", math.inf)
        for item in eligible:
            if start <= item["elapsed_h"] <= end:
                selected.append(item)
            else:
                omitted.append({**item, "reason": "outside_time_window"})
    else:
        selected_indices: set[int] = set()
        for target in settings["target_hours"]:
            nearest = (
                min(
                    eligible,
                    key=lambda item: (
                        abs(item["elapsed_h"] - target),
                        item["elapsed_h"],
                        item["input_index"],
                    ),
                )
                if eligible
                else None
            )
            deviation = (
                abs(nearest["elapsed_h"] - target) * 3600
                if nearest is not None
                else None
            )
            if nearest is None or deviation > tolerance + 1e-8:
                unmatched.append(
                    {
                        "target_hour": target,
                        "nearest_available_hour": (
                            None if nearest is None else nearest["elapsed_h"]
                        ),
                        "nearest_distance_seconds": deviation,
                    }
                )
                if settings["on_unmatched_target"] == "error":
                    issues.append(
                        {
                            "code": "target_unmatched",
                            "target_hour": target,
                            "message": (
                                f"No eligible image is within {tolerance:g} s "
                                f"of {target:g} h"
                            ),
                        }
                    )
                continue
            matches.append(
                {
                    "target_hour": target,
                    "input_index": nearest["input_index"],
                    "image_id": nearest["image_id"],
                    "source_path": nearest["source_path"],
                    "elapsed_h": nearest["elapsed_h"],
                    "distance_seconds": deviation,
                }
            )
            selected_indices.add(nearest["input_index"])
        selected = [
            item for item in eligible if item["input_index"] in selected_indices
        ]
        omitted.extend(
            {**item, "reason": "not_selected_by_targets"}
            for item in eligible
            if item["input_index"] not in selected_indices
        )
    last_available = eligible[-1]["elapsed_h"] if eligible else None
    last_selected = selected[-1]["elapsed_h"] if selected else None
    required_end = settings.get("required_end_hour")
    if mode == "target_hours":
        required_end = max(
            settings["target_hours"][-1],
            required_end if required_end is not None else -math.inf,
        )
    if not selected:
        issues.append(
            {
                "code": "no_selected_images",
                "message": "No eligible images were selected",
            }
        )
    for label, actual in (("available", last_available), ("selected", last_selected)):
        if actual is None or actual + tolerance / 3600 < required_end - 1e-10:
            issues.append(
                {
                    "code": f"late_{label}_coverage_unmet",
                    "required_end_hour": required_end,
                    "last_hour": actual,
                    "message": (
                        f"Last {label} image does not reach {required_end:g} h "
                        f"within {tolerance:g} s"
                    ),
                }
            )
    return {
        "run_id": run["run_id"],
        "dataset": run["dataset"],
        "status": "failed" if issues else "ready",
        "input_frame_count": len(run["frames"]),
        "eligible_frame_count": len(eligible),
        "selected_frame_count": len(selected),
        "last_available_hour": last_available,
        "last_selected_hour": last_selected,
        "required_end_hour": required_end,
        "selected": [
            {"selection_index": index, **item} for index, item in enumerate(selected)
        ],
        "target_matches": matches,
        "unmatched_targets": unmatched,
        "omitted": omitted,
        "issues": issues,
    }


def resolve_selection(
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    purpose: str,
    *,
    check_files: bool = False,
    source_root: str | Path | None = None,
    manifest_sha256: str | None = None,
    plan_sha256: str | None = None,
    manifest_path: str | Path | None = None,
    plan_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return an auditable receipt without writing files or reading image pixels.

    ``check_files`` only calls ``Path.is_file``; relative paths resolve against
    ``source_root``. Omitted frames and unmatched targets remain visible in the
    receipt. A failed receipt must not be used for downstream analysis.
    """
    settings = _purpose_settings(plan, purpose)
    runs = _manifest_runs(manifest)
    root = Path(source_root) if source_root is not None else None
    results = [
        _resolve_run(run, settings, check_files=check_files, source_root=root)
        for run in runs
    ]
    return {
        "format": RECEIPT_FORMAT,
        "schema_version": 1,
        "purpose": purpose,
        "status": (
            "failed" if any(run["status"] == "failed" for run in results) else "ready"
        ),
        "selection_plan": settings,
        "manifest": {
            "path": None if manifest_path is None else str(manifest_path),
            "sha256": manifest_sha256 or _canonical_sha256(manifest),
            "hash_basis": "file" if manifest_sha256 else "canonical_json",
        },
        "plan": {
            "path": None if plan_path is None else str(plan_path),
            "sha256": plan_sha256 or _canonical_sha256(plan),
            "hash_basis": "file" if plan_sha256 else "canonical_json",
        },
        "check_files": check_files,
        "runs": results,
    }


def resolve_selection_files(
    manifest_path: str | Path,
    plan_path: str | Path,
    purpose: str,
    *,
    check_files: bool = False,
) -> dict[str, Any]:
    """Load an existing JSON manifest and TOML plan, hashing their exact bytes."""
    # Resolve before recording the binding: the worker may be launched from a
    # different working directory than the plan creator.
    source = Path(manifest_path).expanduser().resolve()
    policy = Path(plan_path).expanduser().resolve()
    manifest_bytes = source.read_bytes()
    plan_bytes = policy.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    plan = tomllib.loads(plan_bytes.decode("utf-8-sig"))
    if not isinstance(manifest, Mapping) or not isinstance(plan, Mapping):
        raise ValueError("Manifest and plan must be mappings")
    return resolve_selection(
        manifest,
        plan,
        purpose,
        check_files=check_files,
        source_root=source.parent,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        manifest_path=source,
        plan_path=policy,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON native-frame manifest or multi-run wrapper",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Versioned TOML plan in the selected project",
    )
    parser.add_argument("--purpose", choices=PURPOSES, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Explicit new JSON receipt path (never overwritten)",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Check source file existence without decoding images",
    )
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        if output in {args.manifest.resolve(), args.plan.resolve()}:
            raise ValueError("Output path must differ from manifest and plan")
        receipt = resolve_selection_files(
            args.manifest, args.plan, args.purpose, check_files=args.check_files
        )
        with output.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
    ) as error:
        print(f"Image selection failed: {error}", file=sys.stderr)
        return 2
    print(f"{receipt['status']}: {len(receipt['runs'])} run(s); receipt: {output}")
    for run in receipt["runs"]:
        print(
            f"  {run['run_id']}: {run['selected_frame_count']} selected; "
            f"last available={run['last_available_hour']} h; "
            f"last selected={run['last_selected_hour']} h; "
            f"required={run['required_end_hour']} h"
        )
        for issue in run["issues"]:
            print(f"    {issue['code']}: {issue['message']}", file=sys.stderr)
    return 0 if receipt["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
