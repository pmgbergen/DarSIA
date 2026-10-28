"""Unclaimed Setup candidates consume metadata; only an owned run hashes pixels."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from darsia.fluidflower.pipeline import store_runtime, worker
from darsia.fluidflower.pipeline.job_inputs import catalog_order_key
from .test_series_setup_queue_runtime import _queue_context, _claim


def _payload(context):
    path = context.controls / "large-raw.npz"
    raw = b"synthetic scientific payload" * 1024
    path.write_bytes(raw)
    pin = dict(root_alias="configuration", relative_path=path.name,
        sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), role="rig_model")
    context.task["input_catalog"]["inputs"].append(dict(reference=pin, uses=["setup.raw.native.fixture"]))
    context.task["input_catalog"]["inputs"].sort(key=lambda row: catalog_order_key(row["reference"]))
    context.pending.write_text(json.dumps(context.task))
    return path


def test_many_unclaimed_candidates_never_open_payloads(tmp_path, monkeypatch):
    context = _queue_context(tmp_path, monkeypatch)
    payload = _payload(context)
    opened, original = [], Path.open
    def read(path, *args, **kwargs):
        if path == payload:
            opened.append(str(path))
            raise AssertionError("An unclaimed candidate read scientific payload")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", read)
    for number in range(20):
        task = deepcopy(context.task)
        task["attempt_id"] = f"{number:032x}"
        with context.runtime.capacity_claim(context.owner) as candidate:
            assert candidate.availability(task)
            assert context.runtime.setup_verified_inputs == {}
    assert opened == []
    assert not list((context.queue.root / "leases").iterdir())


def test_only_own_claim_hashes_payload_once_and_rejects_content_change(tmp_path, monkeypatch):
    context = _queue_context(tmp_path, monkeypatch)
    payload = _payload(context)
    original, opened = Path.open, []
    def read(path, *args, **kwargs):
        if path == payload:
            opened.append(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", read)
    claim = _claim(context)
    assert opened == []
    store_runtime.verify_inputs(claim.task, claim.input_rows)
    assert opened == [str(payload)]
    # Same-size mutation after location admission must still fail actual SHA.
    raw = payload.read_bytes()
    payload.write_bytes(b"X" + raw[1:])
    with pytest.raises(store_runtime.InputHashMismatch, match="SHA/bytes differ"):
        store_runtime.verify_inputs(claim.task, claim.input_rows)


def test_stop_between_queue_candidates_leaves_every_pending_task_unclaimed(tmp_path, monkeypatch):
    context = _queue_context(tmp_path, monkeypatch)
    payload = _payload(context)
    before = context.pending.read_bytes()
    calls = []
    def checkpoint():
        calls.append(True)
        raise worker._JobControlBoundary("control_stopped")
    with pytest.raises(worker._JobControlBoundary):
        context.queue.claim(context.runtime.service_id, context.runtime.backend, context.owner,
            availability=lambda task: pytest.fail("Stop must precede candidate admission"),
            output_factory=context.runtime.output_factory, publication_factory=context.publisher,
            claim_controls=lambda *args: pytest.fail("Stop must precede claim controls"),
            checkpoint=checkpoint)
    assert calls == [True] and context.pending.read_bytes() == before
    assert not list((context.queue.root / "leases").iterdir())


def test_stop_interrupts_catalogue_metadata_walk(tmp_path, monkeypatch):
    context = _queue_context(tmp_path, monkeypatch)
    _payload(context)
    visited = []
    def checkpoint():
        visited.append(True)
        if len(visited) == 2:
            raise worker._JobControlBoundary("control_stopped")
    with pytest.raises(worker._JobControlBoundary):
        store_runtime.resolve_catalog(context.task, context.runtime.stores,
            context.runtime.job["series_id"], checkpoint=checkpoint)
    assert len(visited) == 2
