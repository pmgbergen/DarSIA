"""Alias-job master status from immutable shared publications only.

No input-store resolution and no path embedded by a producing machine is opened.
Numerical products remain passive SHA/size identities in their original index.
"""
from __future__ import annotations
from datetime import datetime, timezone
from collections import OrderedDict
import hashlib
import json
from pathlib import Path, PureWindowsPath, PurePosixPath
import re
import traceback

from .pipeline.job_queue import (
    descriptor, name, _own_asset, publication_directory, retired_lease_directory, validate_task,
)
from .storage_profile import declared_storage_profile
from .pipeline.products_publication import (
    _read_index, _bound, _json, _identity, _relative, _file, _stamp,
    _verified_bytes, _verify_file, ProductUnavailable, MAX_INDEX_BYTES, MAX_METADATA_BYTES,
)

def _same_bytes(left, right):
    return (left["sha256"], left.get("bytes", left.get("size_bytes"))) == (
        right["sha256"], right.get("bytes", right.get("size_bytes")))

def _read_reference(root, reference, reader):
    path = _own_asset(root, reference)
    return reader(path, descriptor(path))

def _task_identity(base, actual):
    return {k:v for k,v in base.items() if k != "attempt_id"} == {
        k:v for k,v in actual.items() if k != "attempt_id"}

def _commit(record, commit, task):
    case = record["frame_id"]
    if (case not in task["ordered_case_ids"] or commit.get("complete") is not True
            or commit.get("frame_id") != case
            or commit.get("manifest_sha256") != record.get("manifest_sha256")):
        raise ValueError("Published original commit differs from its selected frame")
    if commit.get("products",{}).get("precision") != task["precision"]:
        raise ValueError("Published original frame precision differs")
    files = record["files"]
    if set(commit.get("files",{})) != set(files) - {"commit.json"}:
        raise ValueError("Published commit inventory differs from original frame receipt")
    if any(not _same_bytes(asset, files[leaf]) for leaf,asset in commit["files"].items()):
        raise ValueError("Published commit product SHA/size differs")
    return case

def _lease_token(lease):
    """Attempt of a lease: leases/<run>.<attempt>, or the legacy reclaimed/<run>/<attempt>/lease."""
    return lease.parent.name if lease.name == "lease" else lease.name.split(".")[-1]


def _lease_chain(root, run, token, base_task, reader):
    """Follow exact task generations, never a highest-output scan."""
    base_token = base_task["attempt_id"]
    if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32}", value)
               for value in (token, base_token)):
        raise ValueError("Alias task attempt token is malformed")
    paths, visited = [], set()
    while token != base_token:
        if token in visited:
            raise ValueError("Cyclic alias attempt lineage")
        visited.add(token)
        matches = []
        for p in (root/"reclaimed"/name(run)).glob("*/reclaimed.json"):
            value = reader(p)
            if value.get("next_attempt") == token:
                matches.append((p,value))
        if len(matches) != 1:
            raise ValueError("Alias attempt has no unique actual predecessor")
        path, value = matches[0]
        previous = value["previous_attempt"]
        if (value.get("run") != run or path.parent.name != previous
                or not re.fullmatch(r"[0-9a-f]{32}", previous)):
            raise ValueError("Alias predecessor run/generation differs")
        task = reader(path.parent/"task.json")
        if (task.get("attempt_id") != previous or task.get("run") != run
                or not _task_identity(base_task, task)):
            raise ValueError("Archived predecessor task differs")
        paths.append(retired_lease_directory(root, run, previous, reader=reader))
        token = previous
    return list(reversed(paths))


def _phase1_progress(value, task, lease, reader):
    from .pipeline.phase1_protocol import PIPELINE
    from .pipeline.staged_data import FRAME_FORMAT, _same_json
    event = value["event"]
    case = event.get("case_id")
    original = event.get("phase1_commit")
    copied = value.get("published_phase1_frame")
    if (event.get("state") != "phase1_frame_completed" or event.get("stage") != PIPELINE
            or event.get("complete") is not True or event.get("run") != task["run"]
            or case not in task["ordered_case_ids"] or not isinstance(original, dict)
            or not isinstance(copied, dict) or not _same_bytes(original, copied)):
        raise ValueError("Actual positive phase-one shared progress identity required")
    record = _read_reference(lease, copied, reader)
    if (record.get("format") != FRAME_FORMAT or record.get("complete") is not True
            or record.get("run") != task["run"] or record.get("case_id") != case
            or record.get("precision") != task["precision"]
            or not _same_json(record.get("frame_index"), event.get("frame_index"))
            or not _same_bytes(record.get("configuration", {}), task["configuration"])):
        raise ValueError("Shared phase-one commit differs from the task/progress identity")
    return case, record, value["written_unix_ns"]


def _phase1_terminal(root, base_task, job, done, reader, *, display=None):
    """Inspect only original shared indexed metadata; never producer-local paths."""
    from .pipeline.phase1_protocol import validate_phase1_claim, RESULT_FORMAT, MANIFEST_FORMAT, MAP_SHA_FIELD
    from .pipeline.staged_data import COMPLETION_FORMAT, _same_json, _frame_record
    validate_phase1_claim(base_task, job)
    if (done.get("format") != "darsia-alias-done-v1" or done.get("run") != base_task["run"]
            or done["completed"]["relative_path"] != "results/" + base_task["run"] + "/completed.json"):
        raise ValueError("Alias phase-one done record differs from its canonical run")
    completed = _read_reference(root, done["completed"], reader)
    index_path = _own_asset(root, done["products_index"])
    index = descriptor(index_path)
    _index_root, document, products = _read_index(index)
    shared_metadata = _SnapshotMetadata(index, _index_root, products)
    if (document["run"] != base_task["run"] or document["attempt_id"] != done["attempt_id"]
            or not _same_bytes(document["job"], descriptor(root / "job.json"))
            or not _same_bytes(shared_metadata.read("completed.json")[0], done["completed"])):
        raise ValueError("Published phase-one index/job/completion identity differs")
    actual = completed.get("task") or {}
    validate_phase1_claim(actual, job)
    if (completed.get("format") != RESULT_FORMAT or completed.get("pipeline") != job["pipeline"]
            or completed.get("complete") is not True or completed.get("whole_run") is not True
            or completed.get("worker_references_closed") is not True
            or completed.get("run") != base_task["run"] or completed.get("attempt_id") != done["attempt_id"]
            or not _task_identity(base_task, actual) or actual.get("attempt_id") != done["attempt_id"]
            or completed.get("commit") != job["commit"] or completed.get("precision") != job["precision"]
            or completed.get("backend") not in {"cpu", "cuda"}
            or base_task.get("backend") not in {"any", completed.get("backend")}
            or completed.get(MAP_SHA_FIELD) != job[MAP_SHA_FIELD] or "calibration_sha256" in completed
            or completed.get("fitted_mass_calibration_read") is not False
            or type(completed.get("mass_evaluation_calls")) is not int or completed["mass_evaluation_calls"] != 0):
        raise ValueError("Original closed positive phase-one result identity differs")
    lineage = _lease_chain(root, base_task["run"], done["attempt_id"], base_task, reader)
    publication_owner = reader(index_path.parent.parent / "owner.json")
    if (publication_owner.get("task") != actual or publication_owner.get("token") != done["attempt_id"]
            or publication_owner.get("actual_owner") != completed.get("actual_worker")):
        raise ValueError("Phase-one publication owner differs from original terminal task/worker")
    machine = _owner_machine(publication_owner, base_task, done["attempt_id"])
    completion_product = products["completed.json"]
    if (not _same_bytes(completion_product, done["completed"])
            or name(completion_product["machine_id"]) != name(machine)
            or _pure(completion_product["local_path"]).name != "candidate_completed.json"):
        raise ValueError("Indexed phase-one completion producer differs")
    attempt = _pure(completion_product["local_path"]).parent
    if (attempt.name != done["attempt_id"] or attempt.parent.name != ".attempts"
            or name(attempt.parent.parent.name) != name(base_task["run"])
            or name(attempt.parent.parent.parent.name) != name(base_task["job_id"])):
        raise ValueError("Phase-one indexed producer is outside its genuine job/run/attempt namespace")
    def metadata(asset, product_id):
        row = products[product_id]
        if (not _same_bytes(row, asset) or _pure(row["local_path"]) != _pure(asset["path"])
                or name(row["machine_id"]) != name(machine)):
            raise ValueError("Named phase-one metadata product differs from its original descriptor")
        return _original_metadata(index, products, asset, attempt, machine, metadata=shared_metadata)
    manifest = metadata(completed["manifest"], "manifest.json")
    core = metadata(completed["phase1_completion"], "phase1_completed.json")
    expected_stage = dict(pipeline=job["pipeline"], configuration=base_task["configuration"],
        run=base_task["run"], commit=job["commit"], precision=job["precision"],
        backend=completed.get("backend"), threads=base_task["threads"],
        attempt_id=done["attempt_id"], phase1_configurations_sha256=job[MAP_SHA_FIELD])
    cases = base_task["ordered_case_ids"]
    if (manifest.get("format") != MANIFEST_FORMAT or manifest.get("run") != base_task["run"]
            or manifest.get("job_id") != base_task["job_id"]
            or not _same_json(manifest.get("stage_identity"), expected_stage)
            or manifest.get("code") != dict(commit=job["commit"], identity_algorithm="sha256-git-commit-ascii")
            or manifest.get("commit_identity_digest") != hashlib.sha256(job["commit"].encode("ascii")).hexdigest()
            or manifest.get("settings", {}).get("backend") != completed.get("backend")
            or manifest.get("settings", {}).get("precision") != job["precision"]
            or not _same_json(manifest.get("settings", {}).get("threads"), base_task["threads"])
            or not _same_bytes(manifest.get("configuration", {}), base_task["configuration"])
            or not manifest.get("native_code_admission", {}).get("entries")
            or any(row.get("commit") != job["commit"] for row in manifest["native_code_admission"]["entries"])
            or set(manifest.get("frames", {})) != set(cases)):
        raise ValueError("Original phase-one manifest/code/settings/configuration differs")
    if (core.get("format") != COMPLETION_FORMAT or core.get("complete") is not True
            or core.get("history_complete") is not True or core.get("run") != base_task["run"]
            or core.get("code_commit") != job["commit"] or core.get("precision") != job["precision"]
            or not _same_json(core.get("configuration"), manifest["configuration"])
            or not _same_json(completed.get("configuration"), manifest["configuration"])
            or not _same_json(core.get("ordered_case_ids"), cases)
            or not _same_json(completed.get("ordered_case_ids"), cases)
            or type(completed.get("case_count")) is not int or completed["case_count"] != len(cases)
            or not isinstance(core.get("frames"), list) or len(core["frames"]) != len(cases)
            or not _same_json(core["frames"], completed.get("phase1_frames"))):
        raise ValueError("Original phase-one core completion differs from its manifest/case order")
    expected_products = {"completed.json", "manifest.json", "phase1_completed.json"}
    for case, bound in zip(cases, core["frames"]):
        if _pure(bound["path"]) != attempt / "preparation" / case / "phase1_commit.json":
            raise ValueError("Phase-one core frame points outside its actual current attempt")
        record = metadata(bound, "phase1/" + case + "/phase1_commit.json")
        _frame_record(record, manifest["frames"][case], manifest["configuration"], job["precision"], bound)
        expected_products.add("phase1/" + case + "/phase1_commit.json")
        for leaf, asset in record["files"].items():
            key = "phase1/" + case + "/" + leaf
            row = products[key]
            if (not _same_bytes(row, asset) or _pure(row["local_path"]) != _pure(asset["path"])
                    or _pure(asset["path"]) != attempt / "preparation" / case / leaf
                    or name(row["machine_id"]) != name(machine)):
                raise ValueError("Indexed phase-one product differs from its original inventory")
            expected_products.add(key)
    if set(products) != expected_products:
        raise ValueError("Phase-one products index has missing or extra original entries")
    shared_metadata.finish()
    if display is not None:
        from .jobs_progress import retain_execution
        retain_execution(display, task=base_task, owner=publication_owner, result=completed)
    return completed, {case: machine for case in cases}, lineage


def _events(lease, task, reader, *, token=None):
    token = token or _lease_token(lease)
    events=[]
    for sequence,path in enumerate(sorted(lease.glob("progress_*.json"))):
        value=reader(path)
        if (value.get("sequence") != sequence or value.get("run") != task["run"]
                or value.get("attempt_id") != token
                or type(value.get("written_unix_ns")) is not int):
            raise ValueError("Published progress sequence/run/attempt/time differs")
        event=value["event"]
        if task.get("pipeline") == "native-phase-statistics":
            if event.get("state") == "frame_completed":
                raise ValueError("Mass frame progress cannot belong to a positive phase-one task")
            if event.get("state") == "phase1_frame_completed":
                events.append(_phase1_progress(value, task, lease, reader))
            continue
        if event.get("state") != "frame_completed":
            continue
        if event.get("complete") is not True:
            raise ValueError("An incomplete event is not a committed frame")
        if event.get("run") != task["run"]:
            raise ValueError("Original progress event belongs to another run")
        record=event["publication"]
        bound=value.get("published_frame")
        if not isinstance(bound,dict):
            raise ValueError("Alias progress has no shared original frame-commit copy")
        if not _same_bytes(bound,record["files"]["commit.json"]):
            raise ValueError("Shared frame commit is not the original byte identity")
        commit=_read_reference(lease,bound,reader)
        case=_commit(record,commit,task)
        if event.get("case_id") != case:
            raise ValueError("Progress case differs from its original published commit")
        events.append((case,commit,value["written_unix_ns"]))
    return events

def _pure(value):
    text = str(value)
    windows = PureWindowsPath(text)
    return windows if windows.drive or "\\" in text else PurePosixPath(text)


def _original_path(path, root):
    """Lexical identity only; do not resolve or stat an original local path."""
    path, root = _pure(path), _pure(root)
    if type(path) is not type(root) or not path.is_absolute() or ".." in path.parts:
        raise ValueError("Invalid original producer path")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("Original producer descriptor escapes its actual attempt") from error
    if not relative.parts:
        raise ValueError("Original producer descriptor must identify a file")
    return path


def _owner_machine(owner, task, token):
    output = owner.get("output")
    if (owner.get("format") != "darsia-alias-lease-owner-v1"
            or owner.get("token") != token or owner.get("task", {}).get("attempt_id") != token
            or not _task_identity(task, owner.get("task", {}))
            or not isinstance(output, dict)
            or set(output) != {"machine_id", "root_alias", "relative_path"}
            or output["root_alias"] != "results"
            or output["relative_path"] != "/".join((name(task["job_id"]),name(task["run"]),".attempts",token))
            or owner.get("actual_owner", {}).get("machine") != owner.get("machine")):
        raise ValueError("Original alias producer owner/task/output identity differs")
    name(output["machine_id"])
    return output["machine_id"]


class _SnapshotMetadata:
    """Strict shared metadata reused only during one terminal snapshot."""

    def __init__(self, index, root, products):
        """Index published metadata paths and create one bounded snapshot cache."""
        self.index = _bound(index)
        self.root = Path(root)
        self.by_path = {}
        for key, row in products.items():
            self.by_path.setdefault(_pure(row["local_path"]), []).append((key, row))
        self.products = products
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.observed = {}
        self.index_stamp = _stamp(_file(self.index["path"]).stat())

    def matches(self, path):
        """Return every indexed product bound to the exact original path."""
        return self.by_path.get(path, ())

    def read(self, key):
        """Read a verified metadata copy and reuse its bytes within this snapshot."""
        row = self.products[_relative(key)]
        copy = row.get("metadata_copy")
        if copy is None:
            raise ProductUnavailable("No published metadata copy: " + key)
        path = self.root.joinpath(*copy["relative_path"].split("/"))
        bound = dict(path=str(path), **_identity(copy))
        stamp = _stamp(_file(path).stat())
        previous = self.observed.get(str(path))
        if previous is not None and (previous[0] != stamp or previous[1] != bound):
            raise ValueError("Shared metadata identity changed during the snapshot")
        cache_key = (str(path), bound["sha256"], bound["bytes"])
        raw = self.cache.get(cache_key)
        if raw is None:
            raw = _verified_bytes(path, bound, limit=MAX_METADATA_BYTES)
            after = _stamp(_file(path).stat())
            if stamp != after:
                raise ValueError("Shared metadata changed during the snapshot")
            self.observed[str(path)] = (stamp, bound)
            while self.cache and self.cache_bytes + len(raw) > MAX_METADATA_BYTES:
                _old_key, old_raw = self.cache.popitem(last=False)
                self.cache_bytes -= len(old_raw)
            self.cache[cache_key] = raw
            self.cache_bytes += len(raw)
        else:
            self.cache.move_to_end(cache_key)
        return bound, raw

    def finish(self):
        """Recheck the index and all consumed metadata before accepting the snapshot."""
        # Rehash, do not trust stat alone: same-size/mtime edits must fail too.
        _verified_bytes(self.index["path"], self.index, limit=MAX_INDEX_BYTES)
        if _stamp(_file(self.index["path"]).stat()) != self.index_stamp:
            raise ValueError("Products index identity changed during the snapshot")
        for path, (stamp, bound) in self.observed.items():
            _verify_file(path, bound, limit=MAX_METADATA_BYTES)
            if _stamp(_file(path).stat()) != stamp:
                raise ValueError("Shared metadata identity changed during the snapshot")


def _original_metadata(index, products, asset, attempt, machine, *, metadata, decode_json=True):
    """Require the exact original descriptor and a verified shared byte copy."""
    original = _bound(asset)
    path = _original_path(original["path"], attempt)
    matches = metadata.matches(path)
    if not matches:
        raise ValueError("Original producer field has no exact indexed path")
    raw = None
    for key, row in matches:
        if not _same_bytes(row, original) or name(row["machine_id"]) != name(machine):
            raise ValueError("Original producer path/SHA/bytes/machine differs from index")
        actual, candidate = metadata.read(key)
        if not _same_bytes(actual, original):
            raise ValueError("Original producer field has no matching shared metadata copy")
        if raw is not None and candidate != raw:
            raise ValueError("Duplicate original metadata paths have different shared bytes")
        raw = candidate
    return _json(raw) if decode_json else raw


def _producer_inventory(index, products, record, producer, task, job, *,
                        current_attempt, current_token, owners, metadata):
    attempt = _pure(producer["attempt_directory"])
    token = attempt.name
    if (not re.fullmatch(r"[0-9a-f]{32}", token) or token not in owners
            or attempt.parent.name != ".attempts"
            or name(attempt.parent.parent.name) != name(task["run"])
            or name(attempt.parent.parent.parent.name) != name(task["job_id"])):
        raise ValueError("Original frame producer is outside its genuine attempt lineage")
    machine = owners[token]
    if token == current_token:
        if attempt != current_attempt:
            raise ValueError("Current frame producer differs from the original completion attempt")
    else:
        proof_path = attempt / "retirement.json"
        matches = [row for row in products.values() if _pure(row["local_path"]) == proof_path]
        if not matches:
            raise ValueError("Original prefix has no shared retirement origin proof")
        proof_row = matches[0]
        proof = _original_metadata(index, products,
            dict(path=proof_row["local_path"],sha256=proof_row["sha256"],bytes=proof_row["bytes"]),
            attempt, machine, metadata=metadata)
        if (proof.get("complete") is not True or proof.get("all_absent") is not True
                or proof.get("worker_os_references_closed") is not True
                or proof.get("run") != task["run"] or proof.get("attempt_id") != token
                or _pure(proof.get("attempt_directory", "")) != attempt
                or name(proof.get("machine_id")) != name(machine)):
            raise ValueError("Original prefix retirement proof differs from its owner/run/attempt")
    case = record["frame_id"]
    directory = _original_path(_pure(record["nas_directory"]) / "commit.json",attempt).parent
    if (_pure(producer["commit"]["path"]) != directory / "commit.json"
            or _pure(producer["result"]["path"]) != directory / "result.json"
            or _pure(producer["manifest"]["path"]) != attempt / "manifest.json"
            or not _same_bytes(producer["commit"],record["files"]["commit.json"])
            or producer["manifest"]["sha256"] != record["manifest_sha256"]):
        raise ValueError("Original frame producer fields do not match their actual frame paths")
    originals = {field: _original_metadata(index,products,producer[field],attempt,machine, metadata=metadata)
                 for field in ("manifest","commit","frame_registration","publication","result")}
    commit = originals["commit"]
    _commit(record,commit,task)
    manifest = originals["manifest"]
    if (manifest.get("run") != task["run"] or manifest.get("configuration") != task["configuration"]
            or manifest.get("code",{}).get("commit") != job["commit"]
            or manifest.get("settings",{}).get("precision") != task["precision"]
            or manifest.get("settings") != commit.get("settings")):
        raise ValueError("Original producer manifest code/configuration/settings differs")
    registration = originals["frame_registration"]
    registered_commit = registration.get("commit",{})
    if (registration.get("frame_id") != case
            or registration.get("manifest_sha256") != producer["manifest"]["sha256"]
            or _pure(registered_commit.get("path", "")) != _pure(producer["commit"]["path"])
            or not _same_bytes(registered_commit,producer["commit"])
            or registration.get("files") != commit["files"]
            or originals["publication"] != record):
        raise ValueError("Original frame registration/publication differs from its producer")
    scalar = originals["result"]
    if (scalar.get("complete") is not True or scalar.get("run") != task["run"]
            or scalar.get("case_id") != case or scalar.get("precision") != task["precision"]):
        raise ValueError("Original frame result identity differs")
    for leaf, asset in record["files"].items():
        row = products["frames/"+case+"/"+leaf]
        if (not _same_bytes(row,asset) or _pure(row["local_path"]) != directory / leaf
                or name(row["machine_id"]) != name(machine)):
            raise ValueError("Indexed final product differs from its exact original frame producer")
    return machine


def _terminal(root, run_root, base_task, job, done, reader, *, display=None):
    if base_task.get("pipeline") == "preset-wasserstein":
        from .preset_wasserstein_results import preset_wasserstein_terminal
        return preset_wasserstein_terminal(root, base_task, job, done, reader)
    if base_task.get("pipeline") == "series-setup":
        from .pipeline.series_setup_results import setup_terminal
        return setup_terminal(root, base_task, job, done, reader, display=display)
    if base_task.get("pipeline") == "native-phase-statistics":
        return _phase1_terminal(root, base_task, job, done, reader, display=display)
    if base_task.get("pipeline") == "n1-calibration-refit":
        from .pipeline.common_calibration import refit_terminal
        return refit_terminal(root, base_task, job, done, reader)
    if (done.get("format") != "darsia-alias-done-v1" or done.get("run") != base_task["run"]
            or done["completed"]["relative_path"] != "results/"+base_task["run"]+"/completed.json"):
        raise ValueError("Alias done record differs from its canonical run")
    completed=_read_reference(root,done["completed"],reader)
    index_path=_own_asset(root,done["products_index"])
    index=descriptor(index_path)
    _index_root,document,products=_read_index(index)
    shared_metadata = _SnapshotMetadata(index, _index_root, products)
    if (document["run"] != base_task["run"] or document["attempt_id"] != done["attempt_id"]
            or not _same_bytes(document["job"],descriptor(root/"job.json"))):
        raise ValueError("Published products index belongs to another job/run/attempt")
    if not _same_bytes(shared_metadata.read("completed.json")[0],done["completed"]):
        raise ValueError("Done completion differs from its original shared index")
    actual_task=completed.get("task") or {}
    if (completed.get("complete") is not True or completed.get("worker_references_closed") is not True
            or completed.get("run") != base_task["run"] or completed.get("attempt_id") != done["attempt_id"]
            or not _task_identity(base_task,actual_task) or actual_task.get("attempt_id") != done["attempt_id"]
            or completed.get("commit") != job["commit"]
            or completed.get("calibration_sha256") != job["calibration_sha256"]
            or completed.get("precision") != job["precision"]):
        raise ValueError("Original closed run completion identity differs")
    cases=base_task["ordered_case_ids"]; records=completed["records"]; producers=completed["frame_producers"]
    if (completed.get("frames") != len(cases) or completed.get("case_count") != len(cases)
            or completed.get("ordered_case_ids") != cases
            or [r["frame_id"] for r in records] != cases
            or [p["case_id"] for p in producers] != cases):
        raise ValueError("Completed selected-case order/count differs")
    lineage=_lease_chain(root,base_task["run"],done["attempt_id"],base_task,reader)
    publication_owner=reader(index_path.parent.parent/"owner.json")
    if (publication_owner.get("task") != actual_task or publication_owner.get("token") != done["attempt_id"]
            or publication_owner.get("actual_owner") != completed.get("actual_worker")):
        raise ValueError("Publication owner differs from original terminal task/worker")
    owners={done["attempt_id"]:_owner_machine(publication_owner,base_task,done["attempt_id"])}
    for lease in lineage:
        path=lease/"owner.json"
        if path.exists():
            owner=reader(path);token=_lease_token(lease)
            owners[token]=_owner_machine(owner,base_task,token)
    completion_product=products["completed.json"]
    if (not _same_bytes(completion_product,done["completed"])
            or name(completion_product["machine_id"]) != name(owners[done["attempt_id"]])
            or _pure(completion_product["local_path"]).name != "candidate_completed.json"):
        raise ValueError("Indexed original completion producer differs")
    current_attempt=_pure(completion_product["local_path"]).parent
    frame_machines={}
    for record,producer in zip(records,producers):
        frame_machines[record["frame_id"]]=_producer_inventory(index,products,record,producer,
            base_task,job,current_attempt=current_attempt,current_token=done["attempt_id"],owners=owners,
            metadata=shared_metadata)
    for field in ("manifest","table"):
        _original_metadata(index,products,completed[field],current_attempt,owners[done["attempt_id"]],
                           metadata=shared_metadata, decode_json=field != "table")
    shared_metadata.finish()
    if display is not None:
        from .jobs_progress import retain_execution
        retain_execution(display, task=base_task, owner=publication_owner, result=completed)
    return completed,frame_machines,lineage


def _failed_publication(root, base_task, failure_path, reader, *, display=None):
    """Keep literal failure and actual frame progress from its exact shared bundle."""
    # A legacy moved publishing directory, or the in-place lease its marker names.
    directory = publication_directory(root, base_task["run"])
    if directory is None:
        raise ValueError("Canonical failure has no shared publication")
    owner = reader(directory / "owner.json")
    publication = reader(directory / "publication.json")
    token = owner.get("token")
    physical = _owner_machine(owner, base_task, token)
    if (publication.get("format") != "darsia-alias-publication-v1"
            or publication.get("success") is not False
            or publication.get("run") != base_task["run"]
            or publication.get("attempt_id") != token
            or set(publication) != {"format", "run", "attempt_id", "success", "candidate"}):
        raise ValueError("Original failed publication differs from its bound owner/task")
    candidate = _own_asset(directory, publication["candidate"])
    failed_bound = descriptor(failure_path)
    if not _same_bytes(publication["candidate"], failed_bound):
        raise ValueError("Canonical failure is not the original published failure bytes")
    # Keep the caller's normal bounded metadata admission before buffering
    # either raw copy. No producer-local source is used for failure recovery.
    failure = reader(failure_path, failed_bound)
    if candidate.read_bytes() != Path(failure_path).read_bytes():
        raise ValueError("Original shared failed.json bytes differ")
    if (failure.get("complete") is not False or failure.get("run") != base_task["run"]
            or failure.get("attempt_id") != token or failure.get("task") != owner["task"]
            or failure.get("actual_worker") != owner.get("actual_owner")
            or not isinstance(failure.get("traceback"), str)):
        raise ValueError("Original failure task/run/attempt/worker or literal traceback differs")
    leases = _lease_chain(root, base_task["run"], token, base_task, reader)
    counted, times = {}, []
    for lease in [*leases, directory]:
        path = lease / "owner.json"
        if not path.exists():
            continue
        original = reader(path)
        original_token = token if lease == directory else _lease_token(lease)
        machine = _owner_machine(original, base_task, original_token)
        for case, _commit_document, stamp in _events(lease, base_task, reader, token=original_token):
            counted[case] = machine
            times.append(stamp)
    if display is not None:
        from .jobs_progress import retain_execution
        retain_execution(display, task=base_task, owner=owner, result=failure)
    return failure, counted, times, physical


def published_snapshot(registration,job,ui,*,reader):
    if job.get("pipeline") == "preset-wasserstein":
        from .preset_wasserstein_snapshot import preset_wasserstein_snapshot
        return preset_wasserstein_snapshot(registration, job, ui, reader=reader)
    if job.get("pipeline") == "series-setup":
        from .pipeline.series_setup_results import setup_snapshot
        return setup_snapshot(registration, job, ui, reader=reader)
    from .jobs_progress import retain_execution, retain_log_reports
    root=Path(registration["root"]);rows=[];workers=[];machines={};errors=[];all_times=[]
    for run in job["expected_runs"]:
        row=dict(run=run,state="Not queued",committed=0,total=None,machine="Not reported",
                 worker="Not reported",error="",case_ids=[])
        try:
            run_root=root/"results"/name(run);task=reader(run_root/"task.json");validate_task(task)
            if task.get("pipeline") == "native-phase-statistics" or job.get("pipeline") == "native-phase-statistics":
                from .pipeline.phase1_protocol import validate_phase1_claim
                validate_phase1_claim(task, job)
                if task.get("execution_ready") is not True or task["run"] != run:
                    raise ValueError("Positive phase-one alias task is not the original runnable run")
            elif (task.get("format")!="darsia-alias-task-v1" or task.get("execution_ready") is not True
                    or task["run"]!=run or task["commit"]!=job["commit"]
                    or task["calibration_sha256"]!=job["calibration_sha256"]
                    or task["precision"]!=job["precision"]
                    or declared_storage_profile(task)!=declared_storage_profile(job)):
                raise ValueError("Alias task identity differs from its immutable job")
            cases=task["ordered_case_ids"];row["total"]=len(cases);counted={};times=[]
            done_path=root/"done"/(run+".json");failure_path=root/"failed"/(run+".json")
            if done_path.exists():
                completed,counted,_=_terminal(root,run_root,task,job,reader(done_path),reader,display=row)
                row.update(state="Complete",machine=row.get("machine", "Not reported"))
                retain_execution(row, task=task, result=completed)
            elif failure_path.exists():
                failure,counted,times,physical=_failed_publication(root,task,failure_path,reader,display=row)
                row.update(state="Failed",error=failure["traceback"],machine=physical)
            else:
                claims=list((root/"claimed").glob("*/"+run+".json"))
                if len(claims)>1:
                    raise ValueError("More than one current claim for a run")
                if claims:
                    claimed=reader(claims[0]);token=claimed["attempt_id"]
                    if not _task_identity(task,claimed):
                        raise ValueError("Current claim differs from its reserved task")
                    lease=root/"leases"/(run+"."+token)
                    if (lease/"owner.json").exists():
                        owner=reader(lease/"owner.json")
                        if owner.get("task")!=claimed or owner.get("token")!=token:
                            raise ValueError("Current owner differs from its claim")
                        leases=_lease_chain(root,run,token,task,reader)+[lease]
                        for original in leases:
                            original_owner_path=original/"owner.json"
                            if not original_owner_path.exists():
                                continue  # An ownerless pre-science generation has no frames.
                            original_owner=reader(original_owner_path)
                            original_token=_lease_token(original)
                            physical=_owner_machine(original_owner,task,original_token)
                            retain_execution(row, owner=original_owner, current=False)
                            for case,commit,stamp in _events(original,task,reader):
                                counted[case]=physical;times.append(stamp)
                        retain_execution(row, task=claimed, owner=owner)
                        actual=owner.get("actual_owner") or {}
                        row.update(state=("Working (last actual frame progress; liveness not established)" if counted else "Claimed (liveness not established)"),
                                   machine=row["machine"],
                                   worker=row["worker"])
                        workers.append(dict(machine=row["machine"],worker=row["worker"],run=run,state=row["state"],
                            last_progress_utc=datetime.fromtimestamp(max(times)/1e9,timezone.utc).isoformat() if times else "Not reported",error=""))
                    else:row["state"]="Claiming (owner not published)"
                elif (root/"publishing"/run).exists():row["state"]="Publishing (not done)"
                elif (root/"queue"/(run+".json")).exists():row["state"]="Queued"
            row["committed"]=len(counted);row["case_ids"]=[c for c in cases if c in counted]
            for case,machine in counted.items():machines.setdefault(machine,set()).add((run,case))
            all_times.extend(times)
        except Exception:
            row.update(state="Metadata error",error=traceback.format_exc(),committed=0,case_ids=[])
            errors.append(row["error"])
        rows.append(row)
    committed=sum(r["committed"] for r in rows);known=all(r["total"] is not None for r in rows)
    total=sum(r["total"] or 0 for r in rows) if known else None
    completed_runs=sum(r["state"]=="Complete" for r in rows);failed_runs=sum(r["state"]=="Failed" for r in rows)
    complete=completed_runs==len(rows) and not errors
    logs=[]
    for p in sorted((root/"logs").glob("*.log")):
        with p.open("rb") as f:
            f.seek(0,2);size=f.tell();f.seek(max(0,size-128*1024));tail=f.read().decode("utf-8",errors="replace")
        logs.append(dict(path=str(p),tail=tail,truncated=size>128*1024))
    from .job_launch_state import read_launch_requests
    from .pipeline.alias_job import job_metadata_reference
    registered=[]
    for p in sorted((root/"ui/machines").glob("*.json")):
        value=reader(p)
        expected=job_metadata_reference(root,root/"job.json")
        if value.get("job")!=expected:
            raise ValueError("Alias registered machine facts belong to another job")
        registered.append(dict(value,source=descriptor(p)))
    state="Complete" if complete else ("Metadata error" if errors else ("Failed" if failed_runs else ("Working" if workers else "Waiting")))
    return retain_log_reports(dict(root=str(root),name=ui.get("name") or registration.get("name") or root.name,
        series=ui.get("series_label") or registration.get("series_label") or job["series_id"],
        created_utc=ui.get("created_utc") or job.get("created_utc") or "Not reported",
        master_machine=ui.get("machine") or "Not recorded",job_facts=dict(job),runs=rows,workers=workers,
        machines=[dict(machine=m,committed=len(v),frames_per_minute=None,currently_claimed_remaining=None,
                       eta_seconds=None,rate_scope="Published original frame receipts; no service liveness or timing inferred") for m,v in machines.items()],
        logs=logs,registered_machines=registered,launch_requests=read_launch_requests(root,reader=reader),
        run_count=len(rows),completed_runs=completed_runs,failed_runs=failed_runs,committed=committed,total_frames=total,
        remaining=max(0,total-committed) if known else None,frames_per_minute=None,eta_seconds=0 if complete else None,
        complete=complete,state=state,errors=errors,
        rate_scope="Only shared immutable original publications; no producer-local path or payload is opened"))
