"""Explicit, read-only historical timing import; no jobs, probes or RAW reads.

Only selected completion/trial-summary metadata and their exact bound packet or
code-lock metadata are followed. A local immutable registry remembers explicit
imports; it is not a NAS history scanner or a measurement generator.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from .pipeline.job_queue import write_immutable


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _read(path, expected=None):
    path = Path(path)
    before = path.stat()
    if before.st_size > 32 * 1024 * 1024:
        raise ValueError(
            "Selected measurement metadata exceeds 32 MiB; no payload is imported"
        )
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise ValueError("Measurement metadata changed while being read")
    bound = dict(
        path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)
    )
    if expected is not None and bound != expected:
        raise ValueError("Measurement metadata SHA/path/size changed: " + str(path))

    # Decode the exact hashed bytes, not a second read.
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate measurement metadata key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite measurement metadata: " + value)

    doc = json.loads(
        raw.decode("utf-8-sig"), object_pairs_hook=unique, parse_constant=invalid
    )
    if not isinstance(doc, dict):
        raise ValueError("Measurement receipt must be a JSON object")
    return doc, bound


def _duration(doc, key):
    value = doc.get(key)
    if value is not None and not _number(value):
        raise ValueError("Invalid measured duration: " + key)
    return value


def _memory(doc, key):
    value = doc.get(key)
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError("Invalid recorded peak-memory byte count: " + key)
    return value


def load_measurement_receipt(path, *, expected=None):
    """Return one honest sample from an explicitly chosen original JSON receipt.

    Accepted routes: current whole-run completed.json; original R4 completion
    whose exact packet names its run/settings; trial summary containing an exact
    whole_run_completion descriptor. No result arrays or referenced RAWs are read.
    Missing memory/timing subdivisions remain None. No machine/precision or CPU
    throughput is substituted for a different backend/precision.
    """
    doc, selected = _read(path, expected)
    completion = doc
    bound = selected
    sources = [selected]
    summary = None
    if "whole_run_completion" in doc:
        summary = doc
        if summary.get("complete") is not True:
            raise ValueError("Trial summary is not complete")
        completion, bound = _read(
            doc["whole_run_completion"]["path"], doc["whole_run_completion"]
        )
        sources.append(bound)
    if (
        completion.get("complete") is not True
        or completion.get("worker_references_closed") is not True
    ):
        raise ValueError(
            "A complete original run with closed worker references is required"
        )
    frames = completion.get("frames")
    records = completion.get("records")
    if (
        type(frames) is not int
        or frames < 1
        or not isinstance(records, list)
        or len(records) != frames
    ):
        raise ValueError("Exact original measured frame count/records are required")
    ids = [row.get("frame_id") for row in records]
    if (
        any(not isinstance(case, str) or not case for case in ids)
        or len(set(ids)) != frames
    ):
        raise ValueError("Original measured frame identities are missing or duplicated")
    if "ordered_case_ids" in completion and completion["ordered_case_ids"] != ids:
        raise ValueError("Measured completion case order differs")
    precision, backend = completion.get("precision"), completion.get("backend")
    if precision not in {"float32", "float64"} or backend not in {"cpu", "cuda"}:
        raise ValueError(
            "Original explicit CPU/CUDA backend and float32/float64 precision required"
        )
    run = completion.get("run")
    if not isinstance(run, str) or not run:
        raise ValueError("Measured run identity required")
    task = completion.get("task") or {}
    owner = completion.get("actual_worker") or {}
    machine = owner.get("machine")
    threads = task.get("threads")
    commit = completion.get("commit")
    pipeline = completion.get("pipeline") or task.get("pipeline")
    if task:
        if (
            task.get("run") != run
            or task.get("precision") != precision
            or task.get("commit") != commit
        ):
            raise ValueError("Measured task/code/precision differs from its completion")
        if task.get("backend", backend) not in {backend, "any"}:
            raise ValueError("Measured task backend differs")
        if pipeline is None:
            pipeline = "n1"  # The retained worker's explicit default route.
    elif completion.get("packet") and completion.get("stage"):
        packet, packet_bound = _read(completion["packet"]["path"], completion["packet"])
        sources.append(packet_bound)
        if packet.get("format") != "ffac-round4-main-operational-packet-v1":
            raise ValueError("Unsupported original operational packet schema")
        tasks = packet.get("tasks") or {}
        if (
            not isinstance(tasks, dict)
            or run not in tasks
            or tasks[run].get("run") != run
        ):
            raise ValueError("Original packet has no exact measured run")
        settings = tasks[run].get("settings") or {}
        if settings.get("precision") != precision or settings.get("backend") != backend:
            raise ValueError("Original packet backend/precision differs")
        machine = settings.get("host") or packet.get("host")
        threads = settings.get("threads")
        if packet.get("host") != machine:
            raise ValueError("Original packet host/settings differ")
        pipeline = "historical-round4-" + str(completion["stage"])
        lock = completion.get("code_lock")
        if lock:
            if packet.get("code_lock") != lock:
                raise ValueError("Original packet/code lock differs")
            code, code_bound = _read(lock["path"], lock)
            sources.append(code_bound)
            commit = code.get("commit")
    else:
        raise ValueError(
            "Unsupported measured completion schema; no machine or thread count is guessed"
        )
    if not isinstance(machine, str) or not machine.strip():
        raise ValueError("Original measured machine is not recorded")
    if threads is not None and (type(threads) is not int or threads < 1):
        raise ValueError("Invalid original thread count")
    if commit is not None and (
        not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None
    ):
        raise ValueError("Invalid original full Git commit")
    wall = _duration(completion, "wall_seconds")
    if wall is None:
        raise ValueError("Original measured wall time is not recorded")
    sample = dict(
        machine=machine,
        backend=backend,
        precision=precision,
        threads=threads,
        run=run,
        frames=frames,
        pipeline=pipeline,
        git_commit=commit,
        wall_seconds=wall,
        compute_and_input_seconds=_duration(completion, "compute_and_input_seconds"),
        publication_and_checkpoint_seconds=_duration(
            completion, "publication_and_checkpoint_seconds"
        ),
        peak_worker_rss_bytes=_memory(completion, "peak_worker_rss_bytes"),
        peak_device_memory_bytes=_memory(completion, "peak_device_memory_bytes"),
        source=bound,
        measurement_receipt=selected,
        source_bindings=sources,
        scope="Historical original receipt; not a current-job speed prediction",
    )
    if "device_id" in completion:
        sample["device_id"] = str(completion["device_id"])
    if summary is not None:
        if (
            summary.get("run") != run
            or summary.get("precision") != precision
            or summary.get("case_count") != frames
            or summary.get("code_commit") != commit
            or summary.get("actual_worker") != owner
        ):
            raise ValueError(
                "Trial summary identity differs from its original completion"
            )
        timings = summary.get("timings") or {}
        if _duration(timings, "pipeline_wall_seconds") != wall:
            raise ValueError("Trial wall time differs from its completion")
        sample["compute_and_input_seconds"] = _duration(
            timings, "input_read_preprocess_and_compute_seconds"
        )
        sample["publication_and_checkpoint_seconds"] = _duration(
            timings, "frame_publication_and_start_checkpoint_seconds"
        )
    return sample


class MeasurementRegistry:
    """Local UI-owned immutable registrations of explicit measurement imports.

    Constructor/browsing never creates directories. Source descriptors are
    reverified on refresh. The only write is import_receipt after explicit UI
    choice, one new local entry; source files are never copied or modified.
    """

    def __init__(self, root):
        self.root = Path(root)

    def import_receipt(self, path):
        sample = load_measurement_receipt(path)
        selected = sample["measurement_receipt"]
        target = self.root / (selected["sha256"] + ".json")
        record = dict(
            format="darsia-ui-measurement-registration-v1",
            source=selected,
            imported_utc=datetime.now(timezone.utc).isoformat(),
        )
        if target.exists():
            old, _ = _read(target)
            if old.get("format") != record["format"] or old.get("source") != selected:
                raise ValueError("Existing local measurement registration differs")
        else:
            write_immutable(target, record)
        return dict(sample=sample, registry_path=str(target), source=selected)

    def samples(self):
        if not self.root.exists():
            return []
        samples = {}
        for path in sorted(self.root.glob("*.json")):
            record, _ = _read(path)
            if record.get("format") != "darsia-ui-measurement-registration-v1":
                raise ValueError("Unsupported local measurement registration")
            selected = record["source"]
            sample = load_measurement_receipt(selected["path"], expected=selected)
            key = (sample["source"]["path"], sample["source"]["sha256"])
            old = samples.get(key)
            # A validated explicit trial summary supplies subdivisions; a bare
            # completion must not replace them simply because of filename order.
            if old is None or sample.get("compute_and_input_seconds") is not None:
                samples[key] = sample
        return list(samples.values())

    def merge_machines(self, machines):
        result = deepcopy(list(machines or []))
        by_name = {row["machine"]: row for row in result}
        if len(by_name) != len(result):
            raise ValueError("Duplicate registered machine facts")
        for sample in self.samples():
            row = by_name.get(sample["machine"])
            if row is None:
                row = dict(machine=sample["machine"], inventory={}, history=[])
                by_name[sample["machine"]] = row
                result.append(row)
            history = row.setdefault("history", [])
            identity = (sample["source"]["path"], sample["source"]["sha256"])
            history[:] = [
                s
                for s in history
                if (s.get("source", {}).get("path"), s.get("source", {}).get("sha256"))
                != identity
            ]
            history.append(sample)
        return result
