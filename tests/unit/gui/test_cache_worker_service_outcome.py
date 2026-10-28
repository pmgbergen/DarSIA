"""Actual outcome/publication metadata boundaries with isolated synthetic files."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import sys

import pytest

from darsia_cache import adapters
from darsia_cache.policy import CachePolicyError


TOKEN = "a" * 32
ATTEMPT = "b" * 32


@pytest.mark.parametrize("defect", [None, "epoch", "filetime", "lease", "job"])
def test_native_semantic_owner_uses_exact_fresh_psutil_and_native_lease(monkeypatch, defect):
    from darsia_cache import owned_process
    records = []
    native = dict(pid=70, process_create_time="125.0000001", creation_filetime_100ns="116444737250000001")
    context = dict(path="TESTONLY_observed", sha256="a" * 64, size_bytes=1)
    owner = dict(pid=70, process_create_time="125.0000002")
    lease = SimpleNamespace(manifest=dict(native_launch_context=context), owner=dict(native),
        reservation=dict(token="TESTONLY"), role_identifier="TESTONLY", entry_kind="worker",
        _record=lambda name, value: records.append((name, value)) or value)
    monkeypatch.setattr(adapters, "current_import_lease", lambda: lease)
    monkeypatch.setattr(adapters, "_owner", lambda: (dict(native), {}))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        Process=lambda pid: SimpleNamespace(create_time=lambda: 125.0000002)))
    monkeypatch.setattr(owned_process, "current_native_membership", lambda: context if defect != "job" else {})
    if defect == "epoch": owner["process_create_time"] = "125.0000001"
    if defect == "filetime": owner["creation_filetime_100ns"] = "116444737250000002"
    if defect == "lease": lease.owner["creation_filetime_100ns"] = "116444737250000002"
    if defect is None:
        value = adapters.bind_actual_worker_owner(owner)
        assert value["owner"] == owner and value["cache_native_owner"] == native
        assert len(records) == 1
    else:
        with pytest.raises(CachePolicyError): adapters.bind_actual_worker_owner(owner)
        assert records == []


def test_non_native_semantic_owner_keeps_original_exact_epoch_contract(monkeypatch):
    native = dict(pid=70, process_create_time="125.0000001")
    lease = SimpleNamespace(manifest={}, reservation=dict(token="TESTONLY"),
        role_identifier="TESTONLY", entry_kind="worker", _record=lambda name, value: value)
    monkeypatch.setattr(adapters, "current_import_lease", lambda: lease)
    monkeypatch.setattr(adapters, "_owner", lambda: (dict(native), {}))
    assert adapters.bind_actual_worker_owner(dict(native))["owner"] == native
    with pytest.raises(CachePolicyError):
        adapters.bind_actual_worker_owner(dict(native, process_create_time="125.0000002"))


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _alias(root, pin):
    from pathlib import Path
    return dict(relative_path=Path(pin["path"]).relative_to(root).as_posix(),
                sha256=pin["sha256"], bytes=pin["size_bytes"])


@pytest.fixture
def outcome_world(tmp_path, monkeypatch):
    root, receipts = tmp_path / "job", tmp_path / "receipts"
    native = dict(path=str(receipts / "native-observed.json"), sha256="c" * 64, size_bytes=1)
    _write(root / "job.json", dict(format="darsia-alias-job-v1", job_id="TESTONLY"))
    result = _write(root / "publishing/ac52/artifacts/completed.json", dict(complete=True))
    index = _write(root / "publishing/ac52/artifacts/products_index.json", dict(products=[]))
    _write(root / "done/ac52.json", dict(format="darsia-alias-done-v1", run="ac52",
        attempt_id=ATTEMPT, completed=_alias(root, result), products_index=_alias(root, index)))
    records, seals = [], []
    def record(name, document):
        pin = _write(receipts / name, document)
        records.append((name, deepcopy(document), pin))
        return pin
    lease = SimpleNamespace(profile={"native_supervision": {"TESTONLY": "not a real profile"}},
        entry_kind="worker", manifest=dict(token=TOKEN, native_launch_context=native),
        reservation=dict(token=TOKEN), entry_limit=32, _record=record)
    monkeypatch.setattr(adapters, "optional_import_lease", lambda: lease)
    monkeypatch.setattr(adapters, "seal_current_service", lambda **kw: seals.append(kw) or kw)
    return SimpleNamespace(root=root, receipts=receipts, records=records, seals=seals,
                           lease=lease, native=native, result=result, index=index)


def test_original_done_index_closure_seals_success_without_native_payload_read(outcome_world):
    w = outcome_world
    event = dict(kind="completed", run="ac52", attempt_id=ATTEMPT)
    result = adapters.seal_worker_service(job_root=w.root, exit_code=0, terminal_events=[event])
    assert result["outcome"] == "success" and result["publication"] == w.records[-1][2]
    assert result["traceback_text"] is None
    publication = w.records[-1][1]
    assert publication["completed"][0]["completed"] == w.result
    assert publication["completed"][0]["products_index"] == w.index
    assert publication["native_payload_copied_or_rehashed"] is False
    snapshot = dict(binding=dict(manifest=w.lease.manifest))
    assert adapters.verify_worker_publication(snapshot, result["publication"]) is True


@pytest.mark.parametrize("kind", ["failed", "lease_lost", "input_rejected", "paused", "control_stopped"])
def test_real_noncomplete_event_seals_failed_retention_without_relabeling_result(outcome_world, kind):
    w = outcome_world
    literal = "TESTONLY exact original failure or control boundary"
    result = adapters.seal_worker_service(job_root=w.root, exit_code=0,
        terminal_events=[dict(kind=kind, traceback=literal)])
    assert result["outcome"] == "failed" and result["publication"] is None
    assert literal in result["traceback_text"]
    assert w.result["sha256"] == hashlib.sha256((w.root / "publishing/ac52/artifacts/completed.json").read_bytes()).hexdigest()


def test_original_attempt_mismatch_refuses_success_seal(outcome_world):
    w = outcome_world
    with pytest.raises(CachePolicyError, match="differs"):
        adapters.seal_worker_service(job_root=w.root, exit_code=0,
            terminal_events=[dict(kind="completed", run="ac52", attempt_id="d" * 32)])
    assert w.seals == [] and w.records == []


@pytest.mark.parametrize("relative", ["../outside.json", "/absolute.json", "C:/foreign.json", "a//b.json"])
def test_publication_alias_never_escapes_or_normalizes_bad_identity(outcome_world, relative):
    w = outcome_world
    with pytest.raises(CachePolicyError):
        adapters._worker_alias(w.root, dict(relative_path=relative, sha256="e" * 64, bytes=1))
    assert w.seals == []


def test_changed_original_index_refuses_publication_verifier(outcome_world):
    w = outcome_world
    sealed = adapters.seal_worker_service(job_root=w.root, exit_code=0,
        terminal_events=[dict(kind="completed", run="ac52", attempt_id=ATTEMPT)])
    (w.root / "publishing/ac52/artifacts/products_index.json").write_text("{}", encoding="utf8")
    with pytest.raises(CachePolicyError, match="bytes differ"):
        adapters.verify_worker_publication(dict(binding=dict(manifest=w.lease.manifest)),
                                           sealed["publication"])
