"""Read-only views and explicit metadata creation for the existing Git/UNC queue.

No queue constructor is used for browsing; it would create directories. No
scientific payload is read and no controller/worker/SSH process is started.
"""

from __future__ import annotations
from darsia_cache.transient import retry_transient

from datetime import datetime, timezone
import hashlib
import json
import ntpath
from pathlib import Path
import re
import socket
import time
import tomllib
import traceback

from darsia.fluidflower.pipeline.job_queue import (
    descriptor,
    name,
    read_json,
    require_unc,
    retired_lease_directory,
    write_immutable,
)
from .job_launch_state import read_launch_requests
from .jobs_progress import retain_execution, retain_event, retain_log_reports


def _key(path):
    return ntpath.normcase(ntpath.normpath(str(path)))


def _inside(path, root):
    try:
        return ntpath.commonpath([_key(path), _key(root)]) == _key(root)
    except ValueError:
        return False


def _epoch(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def _bound(bound):
    if (
        not isinstance(bound, dict)
        or not isinstance(bound.get("path"), str)
        or re.fullmatch(r"[a-f0-9]{64}", bound.get("sha256", "")) is None
        or type(bound.get("size_bytes")) is not int
        or bound["size_bytes"] < 0
    ):
        raise ValueError("Full metadata path/SHA/size binding required")


def bind_local_selection_roles(resolved, config, reader):
    """Bind explicit local baseline roles without creating frame outputs.

    Ordinary execution plans retain their original one-row/one-frame contract.
    Only the current-selection assembler's byte-bound record can separate its
    baseline from selected cases. This is metadata admission, not runtime approval.
    """
    selected = resolved.get("selected") or []
    local = (config.get("provenance") or {}).get("local_current_selection")
    if local is None:
        return selected, None
    raw_phase = config.get("format") == "darsia-build-raw-phase-inputs-v1"
    if (config.get("format") not in {
            "darsia-build-raw-phase-inputs-v1", "darsia-native-phase-statistics-run-v1",
            "darsia-native-phase-n1-run-v1"}
            or not isinstance(local, dict)):
        raise ValueError("Local baseline roles require explicit current RAW-phase DATA")
    bound = local.get("selection_record")
    _bound(bound)
    record = reader.read(Path(bound["path"]), bound)
    decision = record.get("decision") or {}
    # The record is pinned by its SHA above; who decided is read from it.
    if (record.get("format") != "ffac-local-p1-selection-data-v1"
            or not isinstance(decision.get("actor"), str) or not decision["actor"].strip()
            or not decision.get("reason")
            or record.get("input_authority", {}).get("current_series") != resolved.get("manifest")
            or local.get("current_series_manifest") != resolved.get("manifest")
            or config.get("history_policy") != record.get("history_policy")):
        raise ValueError("Local selection record differs from the current series/history binding")
    runs = record.get("runs")
    if (not isinstance(runs, list) or not runs
            or any(not isinstance(item, dict) or not isinstance(item.get("run"), str) for item in runs)
            or len({item["run"] for item in runs}) != len(runs)):
        raise ValueError("Local selection needs distinct explicit run records")
    matches = [item for item in runs if item["run"] == config.get("run") == resolved.get("run_id")]
    if len(matches) != 1:
        raise ValueError("Local selection record does not bind this exact run")
    run = matches[0]
    baseline, frames = run.get("baseline"), run.get("selected_frames")
    if (not isinstance(baseline, dict) or baseline.get("baseline") is not True
            or not isinstance(frames, list) or not frames
            or resolved.get("selected_frame_count") != len(selected)
            or len(selected) != len(frames) + 1 or len(config.get("frames", ())) != len(frames)):
        raise ValueError("Explicit baseline and selected cases do not cover the current selection")
    for position, (saved, declared) in enumerate(zip(selected, [baseline, *frames])):
        if (declared.get("current_selection_position") != position
                or _key(saved["source_path"]) != _key(declared.get("raw_path", ""))
                or saved["elapsed_h"] != declared.get("elapsed_h")):
            raise ValueError("Local selection RAW path/time/order differs from the saved selection")

    def bind_raw(raw, declared):
        _bound(raw)
        if (raw["path"] != declared.get("raw_path")
                or raw["sha256"] != declared.get("raw_sha256")
                or (declared.get("recorded_raw_size_bytes") is not None
                    and raw["size_bytes"] != declared["recorded_raw_size_bytes"])):
            raise ValueError("Explicit RAW identity differs from its bound acquisition")
        return raw

    def bind_source(source, raw, declared, is_selected):
        if (source.get("source_sha256") != raw["sha256"]
                or source.get("source_binding", {}).get("raw_acquisition") != raw
                or source.get("run") != config["run"] or source.get("selected") is not is_selected
                or source.get("previous_source_sha256") != declared.get("previous_raw_sha256")
                or source.get("elapsed_h") != declared.get("elapsed_h")
                or source.get(config["selection_index_field"]) != declared.get("current_frame_index")
                or declared.get("baseline") is not (not is_selected)):
            raise ValueError("Explicit RAW-phase role differs from its bound acquisition/time/history")

    def bind(spec, declared, is_selected):
        source = spec.get("source") or {}
        raw = bind_raw(spec.get("raw_acquisition") if raw_phase else
                       source.get("source_binding", {}).get("raw_acquisition"), declared)
        bind_source(source, raw, declared, is_selected)
        if (spec.get("elapsed_h") != declared.get("elapsed_h")
                or spec.get("frame_index") != declared.get("current_frame_index")):
            raise ValueError("Explicit RAW-phase frame differs from its acquisition time/index")
        return raw

    if raw_phase:
        reference, previous = config.get("reference_spec") or {}, config.get("initial_previous_spec") or {}
        raw = bind(reference, baseline, False)
        bind(previous, baseline, False)
        if reference != previous:
            raise ValueError("Reference and initial history must retain the exact bound baseline")
        roles = dict(reference_case_id=reference["case_id"], initial_previous_case_id=previous["case_id"])
        cases = {reference["case_id"]}
    else:
        phase = config.get("phase") or {}
        reference, previous = phase.get("reference") or {}, phase.get("initial_previous") or {}
        raws = []
        for role in (reference, previous):
            native_row = role.get("native_row")
            _bound(native_row)
            saved_row = reader.read(Path(native_row["path"]), native_row)
            raws.append(bind_raw(saved_row.get("raw_acquisition"), baseline))
        raw = raws[0]
        if raws[1] != raw or reference.get("source_sha256") != raw["sha256"]:
            raise ValueError("Saved reference/history rows differ from the bound RAW baseline")
        bind_source(previous.get("source") or {}, raw, baseline, False)
        roles = dict(reference_native_row=dict(reference["native_row"]),
                     initial_previous_native_row=dict(previous["native_row"]))
        cases = set()
    if (baseline.get("previous_raw_sha256") is not None
            or run.get("initial_previous_raw_sha256") != raw["sha256"]):
        raise ValueError("Reference and initial history must retain the exact bound baseline")
    prior, last = raw["sha256"], (baseline["elapsed_h"], baseline["current_frame_index"])
    seen = {prior}
    for spec, declared in zip(config["frames"], frames):
        current = bind(spec, declared, True)
        position = (spec["elapsed_h"], spec["frame_index"])
        if (current["sha256"] in seen or not spec.get("case_id") or spec["case_id"] in cases
                or declared["previous_raw_sha256"] != prior
                or position[0] <= last[0] or position[1] <= last[1]):
            raise ValueError("Local selected cases changed their chronological predecessor chain")
        seen.add(current["sha256"])
        cases.add(spec["case_id"])
        prior, last = current["sha256"], position
    return selected[1:], dict(selection_record=dict(bound), original_camera_source=dict(raw),
                             elapsed_h=baseline["elapsed_h"], **roles)


def validate_current_selection(
    plan, *, series_root, series_id, selection_provider, admitter=None
):
    """Bind B4's actual resolver rows to the unchanged fixed execution selection.

    No new case IDs, frame filtering, predecessor edits or RAW hashing. A
    mismatching saved execution plan must be regenerated by its real producer.
    ``selection_provider(run_ids)`` must call B4 load_run_protocol_metadata and
    resolve_saved_selection for the currently selected registered series.
    """
    if not series_root or selection_provider is None:
        raise ValueError(
            "Create is blocked: the current series' saved B4 image selection is not connected to an execution-plan binding"
        )
    plan = {key: plan[key] for key in ("path", "sha256", "size_bytes")}
    if admitter is None:
        from darsia.fluidflower.pipeline.series import read_series

        admitter = read_series
    value, configs, actual_plan = admitter(
        Path(plan["path"]), expected_sha256=plan["sha256"]
    )
    if actual_plan != plan:
        raise ValueError("Execution plan descriptor changed")
    run_ids = [row["run"] for row in value["runs"]]
    resolved = selection_provider(run_ids)
    if (
        not isinstance(resolved, list)
        or [row.get("run_id") for row in resolved] != run_ids
    ):
        raise ValueError(
            "B4 selection must resolve exactly the execution plan's ordered runs"
        )
    reader, bindings = JobSnapshotReader(), []
    expected_manifest = Path(series_root) / "series.toml"
    for row, config, declaration in zip(resolved, configs, value["runs"]):
        if (
            row.get("format") != "fluidflower-resolved-series-image-selection-v1"
            or row.get("series_id") != series_id
        ):
            raise ValueError("B4 resolver output belongs to another series")
        if _key(row.get("manifest", {}).get("path", "")) != _key(expected_manifest):
            raise ValueError("B4 resolver manifest belongs to another series folder")
        if descriptor(expected_manifest) != row["manifest"]:
            raise ValueError(
                "Current series manifest changed after B4 selection was resolved"
            )
        if not row.get("source_metadata"):
            raise ValueError(
                "B4 selection lacks its original protocol/config metadata binding"
            )
        for bound in row["source_metadata"]:
            _bound(bound)
            if descriptor(Path(bound["path"])) != bound:
                raise ValueError("B4 protocol/config metadata changed")
        selected, baseline_binding = bind_local_selection_roles(row, config, reader)
        if (baseline_binding is None and row.get("selected_frame_count") != len(selected)) or len(
            config["frames"]
        ) != len(selected):
            raise ValueError(
                "B4 image count differs from the saved execution plan; no frame filtering is performed"
            )
        source_config, source_catalog = config, None
        if value.get("pipeline") == "n1":
            from .production_image_selection import selection_sources
            source_config, source_catalog = selection_sources(
                value, config, series_manifest=row["manifest"])
        aliases = config.get("data_aliases") or []

        def relocated(bound):
            if source_catalog is not None:
                actual = source_catalog.physical(bound)
                if "size_bytes" in bound and bound["size_bytes"] != actual.get("size_bytes"):
                    raise ValueError("Source relocation changes its original byte count")
                return dict(actual)
            result = dict(bound)
            for alias in aliases:
                logical, physical = alias["logical"], alias["physical"]
                if _key(logical["path"]) != _key(bound["path"]):
                    continue
                if (
                    logical["sha256"] != physical["sha256"]
                    or bound["sha256"] != physical["sha256"]
                ):
                    raise ValueError("Source relocation changes its original SHA")
                if logical.get("size_bytes") != physical.get("size_bytes") or (
                    "size_bytes" in bound
                    and bound["size_bytes"] != physical.get("size_bytes")
                ):
                    raise ValueError(
                        "Source relocation changes its original byte count"
                    )
                result = dict(physical)
            return result

        pairs = []
        for saved, frame in zip(selected, source_config["frames"]):
            task = frame.get("native_task") or {}
            if baseline_binding is not None and config.get("format") == "darsia-build-raw-phase-inputs-v1":
                raw = dict(frame["raw_acquisition"])
            elif task.get("kind") == "source":
                raw = relocated(task["arguments"]["camera_jpeg"])
            elif task.get("kind") == "raw_rig":
                bound = relocated(task["geometry"])
                geometry = (source_catalog.doc(
                    bound, "rig_model", "UI.selection." + row["run_id"] + "." + frame["case_id"] + ".geometry")
                    if source_catalog is not None else reader.read(Path(bound["path"]), bound))
                raw = relocated(geometry["acquisition"])
                from .image_formats import image_family
                if (image_family(raw["path"]) != "raw"
                        or frame["source"]["source_sha256"] != raw["sha256"]):
                    raise ValueError("RAW selected source differs from its exact acquisition")
            elif task.get("kind") == "rig":
                bound = relocated(task["geometry"])
                geometry = (source_catalog.doc(
                    bound, "rig_model", "UI.selection." + row["run_id"] + "." + frame["case_id"] + ".geometry")
                    if source_catalog is not None else reader.read(Path(bound["path"]), bound))
                raw = relocated(
                    {
                        "path": geometry["camera_jpeg_path"],
                        "sha256": geometry["camera_jpeg_sha256"],
                    }
                )
            else:
                raise ValueError(
                    "No supported original native camera-source binding for this execution frame"
                )
            if (
                _key(saved["source_path"]) != _key(raw["path"])
                or saved["elapsed_h"] != frame["elapsed_h"]
            ):
                raise ValueError(
                    "B4 source path/order/acquisition time differs from the fixed execution plan; regenerate that plan with its real producer"
                )
            pairs.append(
                {
                    "case_id": frame["case_id"],
                    "source_path": saved["source_path"],
                    "elapsed_h": saved["elapsed_h"],
                    "b4_input_index": saved.get("input_index"),
                    "original_camera_source": raw,
                }
            )
        bindings.append(
            {
                "run": row["run_id"],
                "manifest": row["manifest"],
                "source_metadata": row["source_metadata"],
                "configuration": declaration["configuration"],
                "cases": pairs,
            }
        )
        if baseline_binding is not None:
            bindings[-1]["baseline_roles"] = baseline_binding
    return {
        "format": "darsia-job-ui-image-selection-binding-v1",
        "execution_plan": dict(plan),
        "series_id": series_id,
        "series_root": str(series_root),
        "runs": bindings,
        "execution_entries_unchanged": True,
        "raw_hashed_by_this_UI": False,
    }


class JobSnapshotReader:
    """Follow explicit current-job metadata and original producer references only.

    Cache immutable JSON by actual file stat and expected descriptor. Changes
    invalidate it and are rechecked; new progress files are never hidden by a
    directory cache. No recursive NAS discovery, payload hashing or psutil lookup.
    """

    def __init__(self):
        self.cache = {}

    def read(self, path, expected=None):
        path = Path(path)
        if expected is not None:
            _bound(expected)
            if _key(path) != _key(expected["path"]):
                raise ValueError("Metadata path differs from its descriptor")
        stat = retry_transient(path.stat)
        stamp = (stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino)
        cache_key = (
            _key(path),
            None if expected is None else (expected["sha256"], expected["size_bytes"]),
        )
        cached = self.cache.get(cache_key)
        if cached and cached[0] == stamp:
            return cached[1]
        if expected is not None:
            raw = retry_transient(path.read_bytes)
            if (
                len(raw) != expected["size_bytes"]
                or hashlib.sha256(raw).hexdigest() != expected["sha256"]
            ):
                raise ValueError("Published metadata SHA/size differs: " + str(path))
        value = read_json(path)
        after = retry_transient(path.stat)
        if stamp != (after.st_size, after.st_mtime_ns, after.st_dev, after.st_ino):
            raise ValueError("Metadata changed while being read: " + str(path))
        self.cache[cache_key] = (stamp, value)
        return value

    def _publication(self, value, run_root, case_id, *, precision=None):
        if value.get("frame_id") != case_id:
            raise ValueError("Frame publication case identity differs")
        directory = Path(value["nas_directory"])
        if not _inside(directory, run_root) or directory.name != case_id:
            raise ValueError("Frame publication is outside its own run")
        declared = value["files"]["commit.json"]
        commit_path = directory / "commit.json"
        commit = self.read(commit_path, dict(declared, path=str(commit_path)))
        if commit.get("complete") is not True or commit.get("frame_id") != case_id:
            raise ValueError("Frame commit is not complete for this case")
        if (
            precision is not None
            and commit.get("products", {}).get("precision") != precision
        ):
            raise ValueError("Committed frame precision differs from this job")
        return commit

    def _producer(self, producer, run_root, case_id, precision):
        """Follow one explicitly saved original producer; no attempt-history scan."""
        attempt = Path(producer["attempt_directory"])
        if (
            not _inside(attempt, run_root / ".attempts")
            or producer.get("case_id") != case_id
        ):
            raise ValueError("Original frame producer belongs to another run/case")
        receipt = self.read(
            Path(producer["publication"]["path"]), producer["publication"]
        )
        commit_path = Path(receipt["nas_directory"]) / "commit.json"
        if not _inside(commit_path, attempt) or _key(
            producer["commit"]["path"]
        ) != _key(commit_path):
            raise ValueError("Original producer commit/publication paths differ")
        commit = self._publication(receipt, run_root, case_id, precision=precision)
        self.read(commit_path, producer["commit"])
        return commit

    def _completed_identity(self, root, run_root, task, job, done, completed):
        """Bind original completion DATA and the queue's exact reclaim lineage.

        Follow only known tokens forward from reserved task.json. No attempt
        discovery, current-process check or scientific payload is needed.
        """
        required = (
            "commit",
            "calibration_sha256",
            "configuration",
            "task",
            "attempt_id",
        )
        missing = [key for key in required if key not in completed]
        if missing or "attempt_id" not in done:
            raise ValueError(
                "Completion unavailable: original identity metadata is missing: "
                + ", ".join(
                    missing + ([] if "attempt_id" in done else ["done.attempt_id"])
                )
            )
        actual = completed["task"]
        if not isinstance(actual, dict):
            raise ValueError(
                "Completion unavailable: original completed.task is not recorded"
            )
        initial = task.get("attempt_id")
        final = completed["attempt_id"]
        if (
            not isinstance(initial, str)
            or re.fullmatch(r"[0-9a-f]{32}", initial) is None
            or not isinstance(final, str)
            or re.fullmatch(r"[0-9a-f]{32}", final) is None
            or done["attempt_id"] != final
            or actual.get("attempt_id") != final
        ):
            raise ValueError(
                "Completion attempt identity differs from done/task or is unavailable"
            )
        for key in ("commit", "calibration_sha256", "precision"):
            if completed.get(key) != task.get(key) or completed.get(key) != job.get(
                key
            ):
                raise ValueError(
                    "Completed " + key + " differs from its original task/job"
                )
        if completed["configuration"] != task["configuration"]:
            raise ValueError("Completed configuration differs from its original task")
        fixed = {key: value for key, value in task.items() if key != "attempt_id"}
        if {
            key: value for key, value in actual.items() if key != "attempt_id"
        } != fixed:
            raise ValueError(
                "Completed task science/output binding differs from the reserved task"
            )
        if completed.get("backend") not in {"cpu", "cuda"} or task.get(
            "backend", "any"
        ) not in {"any", completed["backend"]}:
            raise ValueError(
                "Completed backend differs from the original task or is unavailable"
            )
        token = initial
        lineage = {initial}
        while token != final:
            archive = root / "reclaimed" / name(task["run"]) / token
            record = self.read(archive / "reclaimed.json")
            successor = record.get("next_attempt")
            if (
                record.get("run") != task["run"]
                or record.get("previous_attempt") != token
                or not isinstance(successor, str)
                or re.fullmatch(r"[0-9a-f]{32}", successor) is None
                or successor in lineage
            ):
                raise ValueError(
                    "Original reclaimed completion lineage differs or cycles"
                )
            if record.get("interrupted_claim_before_science") is True:
                previous = self.read(archive / "task.json")
            else:
                owner = self.read(
                    retired_lease_directory(root, task["run"], token, reader=self.read)
                    / "owner.json"
                )
                if owner.get("token") != token or _key(
                    owner.get("output_path", "")
                ) != _key(run_root / ".attempts" / token):
                    raise ValueError("Original reclaimed owner/token/output differs")
                previous = owner.get("task")
            if (
                not isinstance(previous, dict)
                or previous.get("attempt_id") != token
                or {
                    key: value for key, value in previous.items() if key != "attempt_id"
                }
                != fixed
            ):
                raise ValueError(
                    "Original reclaimed task science/output binding differs"
                )
            lineage.add(successor)
            token = successor
        # An original carried frame may belong to an earlier bound generation;
        # arbitrary old attempt output is never adopted merely for matching IDs.
        for record in completed.get("records") or []:
            if not any(
                _inside(Path(record["nas_directory"]), run_root / ".attempts" / token)
                for token in lineage
            ):
                raise ValueError(
                    "Completed frame is outside its original bound attempt lineage"
                )
        producers = completed.get("frame_producers", [])
        if not isinstance(producers, list):
            raise ValueError("Original frame producer metadata is malformed")
        for producer in producers:
            attempt = Path(producer["attempt_directory"])
            if (
                _key(attempt.parent) != _key(run_root / ".attempts")
                or attempt.name not in lineage
            ):
                raise ValueError(
                    "Completed frame producer is outside its original bound attempt lineage"
                )
        return lineage

    def __call__(self, registration, *, now=None):
        now = time.time() if now is None else now
        root = Path(registration["root"])
        job = self.read(root / "job.json")
        runs = job.get("expected_runs")
        if (
            not isinstance(runs, list)
            or not runs
            or len({name(run) for run in runs}) != len(runs)
        ):
            raise ValueError("Job has no exact distinct expected_runs")
        ui = {}
        master_path = root / "ui" / "master.json"
        if master_path.exists():
            ui = self.read(master_path)
            if ui.get("format") == "darsia-job-ui-master-v2":
                from .pipeline.alias_job import job_metadata_reference
                if (job.get("format") != "darsia-alias-job-v1"
                        or ui.get("job") != job_metadata_reference(root,root/"job.json")):
                    raise ValueError("UI master metadata belongs to another job")
            elif ui.get("format") != "darsia-job-ui-master-v1" or ui.get(
                "job"
            ) != descriptor(root / "job.json"):
                raise ValueError("UI master metadata belongs to another job")
        from .storage_profile import (declared_storage_profile, expected_format_version,
                                      job_storage_profile)
        if job.get("execution_mode") == "current_build":
            from .pipeline.current_build_job import current_build_snapshot
            return current_build_snapshot(registration, job, ui, reader=self.read)
        if job.get("format") == "darsia-alias-job-v1":
            from .pipeline.alias_job import prepared_snapshot
            # Lean jobs are version 2 (N55); full and legacy jobs version 1.
            if job.get("format_version") != expected_format_version(job):
                raise ValueError("Unsupported alias-job metadata version")
            if job.get("execution_ready") is False:
                return prepared_snapshot(registration, job, ui, reader=self.read)
            if (job.get("execution_ready") is not True
                    or job.get("runtime_support") != "darsia-role-io-runtime-v1"):
                raise ValueError("Unsupported alias-job runtime version")
            from .published_jobs_snapshot import published_snapshot
            return published_snapshot(registration, job, ui, reader=self.read)
        job_storage_profile(job)  # A legacy job without the field is full.
        created = ui.get("created_utc") or job.get("created_utc")
        created_epoch = _epoch(created)
        snapshots, machines, errors = [], {}, []
        workers = []
        task_pipeline, task_job_id = set(), set()
        total, total_known = 0, True
        for run in runs:
            row = {
                "run": run,
                "state": "Not queued",
                "committed": 0,
                "total": None,
                "machine": "Not reported",
                "worker": "Not reported",
                "error": "",
                "case_ids": [],
            }
            run_root = root / "results" / name(run)
            counted = {}
            attempt_sources = {}
            try:
                task = self.read(run_root / "task.json")
                if (
                    task["run"] != run
                    or task.get("commit") != job.get("commit")
                    or task.get("calibration_sha256") != job.get("calibration_sha256")
                    or task.get("precision") != job.get("precision")
                ):
                    raise ValueError("Task code/calibration/precision differs from job")
                if declared_storage_profile(task) != declared_storage_profile(job):
                    raise ValueError("Task storage_profile differs from job")
                config = self.read(
                    Path(task["configuration"]["path"]), task["configuration"]
                )
                cases = [frame["case_id"] for frame in config["frames"]]
                if not cases or len(set(cases)) != len(cases):
                    raise ValueError("Exact distinct selected case IDs required")
                row["total"] = len(cases)
                retain_execution(row, task=task)
                total += len(cases)
                task_pipeline.add(task.get("pipeline", "n1"))
                task_job_id.add(task.get("job_id", ""))
                done_path = root / "done" / (name(run) + ".json")
                completed = None
                if done_path.exists():
                    done = self.read(done_path)
                    target = run_root / "completed.json"
                    if done.get("run") != run or _key(
                        done["completed"]["path"]
                    ) != _key(target):
                        raise ValueError(
                            "Done record points outside its canonical run completion"
                        )
                    completed = self.read(target, done["completed"])
                    if (
                        completed.get("complete") is not True
                        or completed.get("worker_references_closed") is not True
                        or completed.get("run") != run
                        or completed.get("precision") != job.get("precision")
                    ):
                        raise ValueError(
                            "A genuine closed per-run completion is required"
                        )
                    self._completed_identity(root, run_root, task, job, done, completed)
                    records = completed.get("records") or []
                    if [
                        record["frame_id"] for record in records
                    ] != cases or completed.get("frames") != len(cases):
                        raise ValueError("Completed selected-case order/count differs")
                    machine = completed.get("actual_worker", {}).get(
                        "machine", "Not reported"
                    )
                    for record in records:
                        self._publication(
                            record,
                            run_root,
                            record["frame_id"],
                            precision=job.get("precision"),
                        )
                        counted[record["frame_id"]] = machine
                    row.update(state="Complete", machine=machine)
                    retain_execution(row, task=task, result=completed)
                    for producer in completed.get("frame_producers") or []:
                        path = Path(producer["attempt_directory"])
                        if (
                            not _inside(path, run_root / ".attempts")
                            or producer["case_id"] not in cases
                        ):
                            raise ValueError(
                                "Frame producer points outside its explicit run attempts"
                            )
                        attempt_sources.setdefault(str(path), set()).add(
                            producer["case_id"]
                        )
                failed_path = root / "failed" / (name(run) + ".json")
                failure = self.read(failed_path) if failed_path.exists() else None
                claims = list((root / "claimed").glob("*/" + name(run) + ".json"))
                if len(claims) > 1:
                    raise ValueError("More than one current claim for a run")
                active_owner = None
                if claims and completed is None and failure is None:
                    claimed = self.read(claims[0])
                    token = claimed["attempt_id"]
                    owner_path = (
                        root / "leases" / (name(run) + "." + token) / "owner.json"
                    )
                    if owner_path.exists():
                        owner = self.read(owner_path)
                        attempt = run_root / ".attempts" / token
                        if (
                            owner.get("token") != token
                            or owner.get("task") != claimed
                            or _key(owner["output_path"]) != _key(attempt)
                        ):
                            raise ValueError("Claim owner/attempt generation differs")
                        if {k: v for k, v in claimed.items() if k != "attempt_id"} != {
                            k: v for k, v in task.items() if k != "attempt_id"
                        }:
                            raise ValueError("Current claim differs from reserved task")
                        active_owner = owner
                        retain_execution(row, task=claimed, owner=owner)
                        attempt_sources[str(attempt)] = None
                        actual = owner.get("actual_owner") or {}
                        row.update(
                            state="Claimed (liveness not established)",
                            machine=actual.get(
                                "machine", owner.get("machine", "Not reported")
                            ),
                            worker=row["worker"],
                        )
                    else:
                        row["state"] = "Claiming (owner not published)"
                elif completed is None and failure is None:
                    if (root / "publishing" / name(run)).exists():
                        row["state"] = "Publishing (not done)"
                    elif (root / "queue" / (name(run) + ".json")).exists():
                        row["state"] = "Queued"
                if failure is not None:
                    if failure.get("run") != run or not isinstance(
                        failure.get("traceback"), str
                    ):
                        raise ValueError("Malformed current run failure")
                    row.update(
                        state="Failed",
                        error=failure["traceback"],
                        machine=failure.get("actual_worker", {}).get(
                            "machine", "Not reported"
                        ),
                    )
                    retain_execution(row, task=task, result=failure)
                    token = failure.get("attempt_id")
                    if isinstance(token, str) and re.fullmatch(r"[a-f0-9]{32}", token):
                        attempt_sources[str(run_root / ".attempts" / token)] = None
                # These are explicit current/terminal producer paths, not a scan
                # for whichever older attempt happens to have the most output.
                for attempt_string, allowed_cases in list(attempt_sources.items()):
                    attempt = Path(attempt_string)
                    if not attempt.exists():
                        continue
                    resume_path = attempt / "resume.json"
                    if allowed_cases is None and resume_path.exists():
                        resume = self.read(resume_path)
                        prefix = resume.get("prefix_case_ids") or []
                        if cases[: len(prefix)] != prefix:
                            raise ValueError(
                                "Saved resume prefix differs from selected order"
                            )
                        producers = resume.get("frame_producers") or []
                        if prefix and [p["case_id"] for p in producers] != prefix:
                            raise ValueError(
                                "Resume prefix has no exact original producers"
                            )
                        for producer in producers:
                            commit = self._producer(
                                producer,
                                run_root,
                                producer["case_id"],
                                job.get("precision"),
                            )
                            counted[producer["case_id"]] = commit.get(
                                "settings", {}
                            ).get("host", row["machine"])
                    progress_paths = sorted(attempt.glob("progress_*.json"))
                    first, last = None, None
                    for sequence, progress_path in enumerate(progress_paths):
                        event_doc = self.read(progress_path)
                        if (
                            event_doc.get("sequence") != sequence
                            or event_doc.get("run") != run
                            or event_doc.get("attempt_id") != attempt.name
                        ):
                            raise ValueError("Progress sequence/run/attempt differs")
                        event = event_doc["event"]
                        case = event.get("case_id")
                        if case is not None and case not in cases:
                            raise ValueError(
                                "Progress case is outside the fixed selection"
                            )
                        if allowed_cases is not None and case not in allowed_cases:
                            continue
                        if (
                            type(event_doc.get("written_unix_ns")) is not int
                            or event_doc["written_unix_ns"] <= 0
                        ):
                            raise ValueError(
                                "Actual progress requires an exact integer timestamp"
                            )
                        timestamp = event_doc["written_unix_ns"] / 1e9
                        retain_event(row, event, datetime.fromtimestamp(timestamp, timezone.utc).isoformat())
                        if event.get("state") == "frame_started":
                            first = min(first or timestamp, timestamp)
                        if event.get("state") == "frame_completed":
                            if event.get("complete") is not True:
                                raise ValueError(
                                    "Incomplete frame progress is not a publication"
                                )
                            commit = self._publication(
                                event["publication"],
                                run_root,
                                case,
                                precision=job.get("precision"),
                            )
                            machine = commit.get("settings", {}).get(
                                "host", row["machine"]
                            )
                            counted[case] = machine
                            last = max(last or timestamp, timestamp)
                            if active_owner is not None and allowed_cases is None:
                                row["progress_machine"] = machine
                            facts = machines.setdefault(
                                machine,
                                {
                                    "machine": machine,
                                    "cases": set(),
                                    "first": None,
                                    "last": None,
                                    "active": False,
                                },
                            )
                            facts["cases"].add((run, case))
                            facts["first"] = min(
                                facts["first"] or first or timestamp, first or timestamp
                            )
                            facts["last"] = max(facts["last"] or timestamp, timestamp)
                            facts["active"] |= (
                                active_owner is not None and allowed_cases is None
                            )
                    if active_owner is not None and allowed_cases is None:
                        workers.append(
                            {
                                "machine": row["machine"],
                                "worker": row["worker"],
                                "run": run,
                                "state": (
                                    "Working (last actual frame progress; liveness not established)"
                                    if last is not None
                                    else row["state"]
                                ),
                                "last_progress_utc": (
                                    datetime.fromtimestamp(
                                        last, timezone.utc
                                    ).isoformat()
                                    if last
                                    else "Not reported"
                                ),
                                "error": "",
                            }
                        )
                        if last is not None:
                            row["state"] = (
                                "Working (last actual frame progress; liveness not established)"
                            )
                if failure is not None:
                    workers.append(
                        {
                            "machine": row["machine"],
                            "worker": "From failed.json",
                            "run": run,
                            "state": "Failed",
                            "last_progress_utc": failure.get(
                                "failed_utc", "Not reported"
                            ),
                            "error": row["error"],
                        }
                    )
                row["committed"] = len(counted)
                row["case_ids"] = sorted(counted)
                for case, machine in counted.items():
                    machines.setdefault(
                        machine,
                        {
                            "machine": machine,
                            "cases": set(),
                            "first": None,
                            "last": None,
                            "active": False,
                        },
                    )["cases"].add((run, case))
            except Exception:
                row["state"], row["error"] = "Metadata error", traceback.format_exc()
                errors.append(row["error"])
                for facts in machines.values():
                    if any(item[0] == run for item in facts["cases"]):
                        facts["cases"] = {
                            item for item in facts["cases"] if item[0] != run
                        }
                        facts["first"] = facts["last"] = None
                workers[:] = [worker for worker in workers if worker["run"] != run]
                if row["total"] is None:
                    total_known = False
            snapshots.append(row)
        committed = sum(row["committed"] for row in snapshots)
        completed_runs = sum(row["state"] == "Complete" for row in snapshots)
        failed_runs = sum(row["state"] == "Failed" for row in snapshots)
        complete = completed_runs == len(runs) and failed_runs == 0 and not errors
        elapsed = (
            now - created_epoch
            if created_epoch is not None and now > created_epoch
            else None
        )
        rate = committed * 60 / elapsed if elapsed and committed else None
        remaining = max(0, total - committed) if total_known else None
        eta = (
            remaining * 60 / rate
            if rate
            and remaining is not None
            and not complete
            and not errors
            and not failed_runs
            else (0 if complete else None)
        )
        machine_rows = []
        for facts in machines.values():
            end = now if facts["active"] else facts["last"]
            duration = (
                end - facts["first"]
                if end and facts["first"] and end > facts["first"]
                else None
            )
            rate_here = len(facts["cases"]) * 60 / duration if duration else None
            remaining_here = sum(
                max(0, row["total"] - row["committed"])
                for row in snapshots
                if row.get("progress_machine") == facts["machine"]
                and row["total"] is not None
                and row["state"].startswith("Working")
            )
            machine_rows.append(
                {
                    "machine": facts["machine"],
                    "committed": len(facts["cases"]),
                    "frames_per_minute": rate_here,
                    "currently_claimed_remaining": remaining_here,
                    "eta_seconds": (
                        remaining_here * 60 / rate_here if rate_here else None
                    ),
                    "rate_scope": (
                        "Observed frame-start window; ETA covers current claims only, unclaimed queue work is not assigned"
                        if duration
                        else "Timing not recorded; no per-machine ETA"
                    ),
                }
            )
        logs = []
        for path in sorted((root / "logs").glob("*.log")):
            with path.open("rb") as stream:
                stream.seek(0, 2)
                length = stream.tell()
                stream.seek(max(0, length - 128 * 1024))
                tail = stream.read().decode("utf-8", errors="replace")
            logs.append(
                {"path": str(path), "tail": tail, "truncated": length > 128 * 1024}
            )
            latest, stopped = {}, False
            for line in tail.splitlines():
                timestamp, separator, encoded = line.partition(" ")
                if not separator or _epoch(timestamp) is None:
                    continue
                try:
                    event = json.loads(encoded)
                except (ValueError, TypeError):
                    continue
                if event.get("kind") == "started":
                    latest.clear()
                    stopped = False
                if event.get("kind") == "stopped":
                    stopped = True
                slot = event.get("worker_slot")
                if type(slot) is int:
                    latest[slot] = (timestamp, event)
            for slot, (timestamp, event) in latest.items():
                if event.get("kind") != "completed":
                    continue
                if any(
                    w["run"] == event.get("run") and w["state"].startswith("Working")
                    for w in workers
                ):
                    continue
                workers.append(
                    {
                        "machine": path.stem,
                        "worker": f"slot {slot} · generation {event.get('worker_generation', '?')}",
                        "run": event.get("run", "Not reported"),
                        "state": (
                            "Stopped (last service report)"
                            if stopped
                            else "Idle (last reported completion; liveness not established)"
                        ),
                        "last_progress_utc": timestamp,
                        "error": "",
                    }
                )
        facts = dict(
            job,
            legacy_paths=job.get("legacy_paths", True),
            pipeline=next(iter(task_pipeline)) if len(task_pipeline) == 1 else None,
            job_id=next(iter(task_job_id)) if len(task_job_id) == 1 else None,
            name=ui.get("name") or registration.get("name") or root.name,
        )
        registered_machines = []
        for path in sorted((root / "ui" / "machines").glob("*.json")):
            bound = descriptor(path)
            record = self.read(path, bound)
            if record.get(
                "format"
            ) != "darsia-job-machine-registration-v1" or record.get(
                "job"
            ) != descriptor(
                root / "job.json"
            ):
                raise ValueError("Registered machine facts belong to another job")
            name(record["machine"])
            registered_machines.append(dict(record, source=bound))
        return retain_log_reports({
            "root": str(root),
            "name": facts["name"],
            "series": job.get("series_id") or "Not recorded",
            "created_utc": created or "Not recorded",
            "master_machine": ui.get("machine") or "Not recorded",
            "job_facts": facts,
            "runs": snapshots,
            "workers": workers,
            "machines": machine_rows,
            "logs": logs,
            "registered_machines": registered_machines,
            "launch_requests": read_launch_requests(root, reader=self.read),
            "run_count": len(runs),
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
            "committed": committed,
            "total_frames": total if total_known else None,
            "remaining": remaining,
            "frames_per_minute": rate,
            "eta_seconds": eta,
            "complete": complete,
            "state": (
                "Complete"
                if complete
                else (
                    "Metadata error"
                    if errors
                    else (
                        "Failed"
                        if failed_runs
                        else (
                            "Working"
                            if any(
                                row["state"].startswith("Working") for row in snapshots
                            )
                            else (
                                "Claimed"
                                if any(
                                    row["state"].startswith("Claim")
                                    for row in snapshots
                                )
                                else "Waiting"
                            )
                        )
                    )
                )
            ),
            "errors": errors,
            "rate_scope": "Unique committed selected cases / elapsed time since recorded job creation; includes queue waiting",
        })


def create_job_from_plan(
    plan,
    job_root,
    *,
    display_name,
    series_label,
    series_root=None,
    series_id=None,
    selection_provider=None,
    machine=None,
    preparer=None,
    admitter=None,
    storage_profile=None,
):
    """Only an explicit Create click calls this original metadata producer.

    A saved adopted execution plan is required. No selected image count, model,
    calibration or descriptor is manufactured from GUI strings. An explicit
    ``storage_profile`` ('full' or 'lean') is the job's choice; None leaves the
    producer's default for new jobs.
    """
    if storage_profile is not None:
        from .storage_profile import new_job_profile
        new_job_profile(storage_profile)  # Refuse an unknown choice before any work.
    _bound(plan)
    if not Path(plan["path"]).is_absolute():
        raise ValueError("An explicit absolute source plan path is required")
    from .job_filesystem import inspect_shared_folder
    existing = Path(job_root)
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    admission = inspect_shared_folder(existing)
    if not admission.allowed:
        raise ValueError(admission.reason + ": " + admission.evidence)
    plan = {key: plan[key] for key in ("path", "sha256", "size_bytes")}
    raw = Path(plan["path"]).read_bytes()
    if (
        len(raw) != plan["size_bytes"]
        or hashlib.sha256(raw).hexdigest() != plan["sha256"]
    ):
        raise ValueError("Selected saved execution plan changed")
    document = tomllib.loads(raw.decode("utf-8"))
    if (
        document.get("format") != "darsia-series-v1"
        or document.get("adopted_for_execution") is not True
    ):
        raise ValueError("Choose an explicitly adopted darsia-series-v1 execution plan")
    if document.get("job", {}).get("precision") != "float32":
        raise ValueError(
            "New production analysis jobs require explicit float32 in the saved plan"
        )
    series_manifest = descriptor(Path(series_root) / "series.toml")
    if admitter is None:
        from .pipeline.alias_job import read_alias_series
        def admitter(path, *, expected_sha256):
            return read_alias_series(path, expected_sha256=expected_sha256,
                                     series_manifest=series_manifest)
    selection_binding = validate_current_selection(
        plan,
        series_root=series_root,
        series_id=series_id,
        selection_provider=selection_provider,
        admitter=admitter,
    )
    root = Path(job_root)
    if root.exists():
        raise FileExistsError(
            "Choose a fresh job folder; existing jobs are registered separately"
        )
    if preparer is None:
        from darsia.fluidflower.pipeline.runner import prepare_job

        preparer = prepare_job
    from .pipeline.alias_job import portable_selection_binding, job_metadata_reference
    if descriptor(Path(series_root) / "series.toml") != series_manifest:
        raise ValueError("Series roots/manifest changed during plan admission")
    portable_options = {}
    if document.get("pipeline") == "n1":
        from .pipeline.job_inputs import InputCatalog

        # The original plan reader admits run configurations through these
        # byte-bound plan aliases. Retain that same locator mapping while
        # encoding the UI proof; its original declaration stays unchanged.
        def catalog_factory(roots):
            return InputCatalog(roots, document.get("data_aliases", ()))

        portable_options["catalog_factory"] = catalog_factory
    portable_binding = portable_selection_binding(
        selection_binding, series_manifest=series_manifest, **portable_options)
    choice = {} if storage_profile is None else {"storage_profile": storage_profile}
    result = preparer(Path(plan["path"]), root, expected_sha256=plan["sha256"],
                      series_manifest=series_manifest, **choice)
    if (
        result.get("complete") is not True
        or result.get("numerical_calls_started") is not False
    ):
        raise ValueError(
            "Existing metadata producer did not report a complete inert job"
        )
    # A changed GUI selection must not acquire a successful UI adoption receipt.
    # The original producer's metadata is preserved, never deleted or rewritten.
    for row in selection_binding["runs"]:
        if descriptor(Path(row["manifest"]["path"])) != row["manifest"] or any(
            descriptor(Path(bound["path"])) != bound for bound in row["source_metadata"]
        ):
            raise ValueError(
                "Job metadata preserved at "
                + str(root)
                + "; B4 selection changed during preparation, no worker was started"
            )
    binding_path = root / "ui" / "image_selection_binding.json"
    write_immutable(binding_path, portable_binding)
    master = {
        "format": "darsia-job-ui-master-v2",
        "job": job_metadata_reference(root,root / "job.json"),
        "machine": machine or socket.gethostname(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "name": str(display_name).strip() or document.get("job_id", root.name),
        "series_label": str(series_label),
        "execution_plan": dict(result["series"]),
        "numerical_calls_started": False,
    }
    master["image_selection_binding"] = job_metadata_reference(root,binding_path)
    write_immutable(root / "ui" / "master.json", master)
    return {
        "root": str(root),
        "name": master["name"],
        "series_label": master["series_label"],
        "active": True,
        "master": descriptor(root / "ui" / "master.json"),
        "producer_receipt": result,
        "worker_started": False,
    }
