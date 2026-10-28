"""Exact software closure for explicitly bound upstream reproduction stages.

The caller verifies the original stage jobs and supplies their documents. This
module never discovers jobs, substitutes installed libraries, or downloads code.
Every upstream stage uses the same software admission as the final saved job.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

from .archives import Member
from .export_software import software_members
from .layout import inside, record_relative

FORMAT = "darsia-staged-export-software-v1"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("A staged software inventory has duplicate keys: " + key)
        result[key] = value
    return result


def _document(raw):
    try:
        value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("A readable staged software JSON inventory is required") from error
    if not isinstance(value, dict):
        raise ValueError("A staged software inventory must be an object")
    return value


def _stage_ids(stage_jobs):
    if not isinstance(stage_jobs, Mapping):
        raise ValueError("Explicitly verified upstream stage jobs are required")
    names = []
    devices = {"con", "prn", "aux", "nul"} | {
        prefix + str(number) for prefix in ("com", "lpt") for number in range(1, 10)}
    for stage in stage_jobs:
        if (not isinstance(stage, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", stage) is None
                or stage.endswith(".") or stage.split(".", 1)[0].casefold() in devices):
            raise ValueError("Upstream stage ids must be portable single path components")
        names.append(stage.casefold())
    if len(names) != len(set(names)):
        raise ValueError("Upstream stage ids have conflicting portable names")
    return sorted(stage_jobs)


def _stage_job(job):
    if not isinstance(job, Mapping):
        raise ValueError("The original upstream stage job document is required")
    commit = job.get("commit")
    lock = job.get("environment_lock")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("The upstream stage job needs its exact original code commit")
    if (not isinstance(lock, Mapping)
            or not isinstance(lock.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", lock["sha256"]) is None):
        raise ValueError("The upstream stage job needs its original environment lock binding")
    return SimpleNamespace(job=deepcopy(dict(job))), commit


def _prefixed(path, prefix):
    value = record_relative(path)
    if not value.startswith("software/"):
        raise ValueError("An upstream software asset is outside its software directory")
    return record_relative(prefix + value.removeprefix("software/"))


def _relocate(value, prefix):
    """Move only the asset paths produced by the original software helper."""
    if isinstance(value, list):
        return [_relocate(item, prefix) for item in value]
    if isinstance(value, dict):
        return {key: _prefixed(item, prefix) if key == "path" else _relocate(item, prefix)
                for key, item in value.items()}
    return value


def staged_software_members(request, stage_jobs):
    """Return archive members and software sections keyed by upstream stage id.

    ``stage_jobs`` must already be verified through the original job bindings;
    values are the original job dictionaries, including commit and lock. The
    explicit ``request.software_stage_inventory`` JSON has this shape::

        {"format": "darsia-staged-export-software-v1",
         "jobs": {"phase1_r1": "phase1_r1/software.json"}}

    The relative files use the unchanged ``software_members`` inventory format.
    Their key set must exactly equal the upstream jobs, with no inferred stages.
    Parent ``software_mode`` applies to every stage. Original inventory bytes
    remain separately bound; relocated versions documents get new digests.
    """
    stages = _stage_ids(stage_jobs)
    selection_path = getattr(request, "software_stage_inventory", None)
    if not stages and selection_path is None:
        return [], {}
    if request.record != "R" or request.software_mode not in {"references", "include"}:
        raise ValueError("Upstream reproduction stages need exact software references or included software")
    if selection_path is None:
        raise ValueError("Choose an explicit software inventory for every upstream stage job")
    selection_path = Path(selection_path)
    selection_raw = selection_path.read_bytes()
    selection = _document(selection_raw)
    jobs = selection.get("jobs")
    if selection.get("format") != FORMAT or not isinstance(jobs, dict):
        raise ValueError("An explicit staged export software inventory is required")
    if set(jobs) != set(stages):
        raise ValueError("Staged software inventory must contain every upstream stage job exactly once")

    members, sections = [], {}
    for stage in stages:
        job, commit = _stage_job(stage_jobs[stage])
        path = inside(selection_path.parent, jobs[stage])
        source = path.read_bytes()
        _document(source)  # Refuse duplicate inventory keys before the original reader.
        stage_request = SimpleNamespace(record=request.record, software_mode=request.software_mode,
                                        software_inventory=path, software_folder=None)
        supplied, original = software_members(stage_request, job, commit)
        if path.read_bytes() != source:
            raise ValueError("An upstream software inventory changed while it was being verified")
        prefix = "software/stages/" + stage + "/"
        # The previous versions.json describes paths before relocation. Preserve
        # the original supplied inventory, and emit a newly hashed document.
        for member in supplied:
            if member.name != original["inventory"]["path"]:
                members.append(replace(member, name=_prefixed(member.name, prefix)))
        source_name = prefix + "input-inventory.json"
        source_binding = dict(path=source_name, sha256=hashlib.sha256(source).hexdigest(),
                              size_bytes=len(source))
        members.append(Member(source_name, data=source, sha256=source_binding["sha256"],
                              size_bytes=len(source)))
        document = _relocate({key: value for key, value in original.items() if key != "inventory"}, prefix)
        document["source_inventory"] = source_binding
        raw = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        name = prefix + "versions.json"
        digest = hashlib.sha256(raw).hexdigest()
        members.append(Member(name, data=raw, sha256=digest, size_bytes=len(raw)))
        sections[stage] = dict(document, inventory=dict(path=name, sha256=digest, size_bytes=len(raw)))
    if selection_path.read_bytes() != selection_raw:
        raise ValueError("The staged software selection changed while it was being verified")
    names = [member.name.casefold() for member in members]
    if len(names) != len(set(names)):
        raise ValueError("Upstream software files have conflicting portable names")
    return members, sections
