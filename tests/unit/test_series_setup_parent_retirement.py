"""Actual parent dispatch and finite supervisor controls on private queue files.

Process absence, spawning and service observation are explicit synthetic seams.
Original queue ownership, publication, capacity ledgers and retirement control
serialization remain real. These cases do not launch or qualify a service.
"""
from copy import deepcopy
import io
from pathlib import Path
import queue
import threading
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import job_queue, worker
from darsia.fluidflower.pipeline.series_setup_sink import current_writer
from darsia.fluidflower.pipeline.series_setup_worker_admission import retirement_allowance
from .test_series_setup_queue_runtime import _queue_context, _claim


class SyntheticProcess:
    """Expose only the process observations consumed by the original parent."""

    def __init__(self, *, alive=False):
        self.alive = alive
        self.exitcode = 0
        self.joins = 0

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.joins += 1


def _retained_claim(tmp_path, monkeypatch):
    """Publish an original failed claim and retain its actual output caps."""
    case = _queue_context(tmp_path, monkeypatch)
    claim = _claim(case)
    helper = case.runtime.setup_admissions[claim.token]
    with case.runtime.output_scope(claim):
        case.queue.fail(claim, "private operation failure", case.owner)
    case.runtime.release(claim, retain_capacity=True)
    return case, claim, helper


def _slot(case, claim, *, alive=False, attach=False):
    """Keep one original generation and its explicit synthetic process facts."""
    return dict(number=0, generation=1, process=SyntheticProcess(alive=alive), root=deepcopy(case.owner),
        observed=[], last_heartbeat=0., retired=False, waiting_reclaim=False, finished=False,
        claim=(dict(run=claim.task["run"], attempt_id=claim.token, task=deepcopy(claim.task)) if attach else None))


def _parent_seams(case, monkeypatch, slot):
    """Replace only actual spawning, process observation and service recording."""
    import darsia_cache.adapters as adapters
    events = []

    def retire(root, observed):
        assert root == case.owner
        slot["process"].alive = False
        events.append("owned-process-retired")
        return [deepcopy(case.owner)]

    monkeypatch.setattr(worker, "retire_owned_processes", retire)
    monkeypatch.setattr(worker, "_same_process", lambda _owner: None)
    monkeypatch.setattr(worker, "_identity", lambda _service: dict(case.owner, pid=999, process_create_time="999.125"))
    monkeypatch.setattr(adapters, "record_cache_observation", lambda kind, value: events.append(kind))
    return events


@pytest.mark.parametrize("fatal", [False, True])
def test_parent_attaches_flushed_claim_before_real_retirement_release(tmp_path, monkeypatch, fatal):
    """Heartbeat-first and fatal paths consume ownership before closing caps."""
    case, claim, helper = _retained_claim(tmp_path, monkeypatch)
    slot = _slot(case, claim, alive=fatal)
    observed = _parent_seams(case, monkeypatch, slot)
    messages = queue.Queue()
    common = dict(worker_slot=0, worker_generation=1, actual_worker=deepcopy(case.owner))
    first = dict(common, kind="watchdog_heartbeat")
    if fatal:
        first["actual_worker"] = dict(case.owner, machine="foreign-service")
    messages.put(first)
    # A previous generation remains passive provenance; it must not name the
    # current slot's claim or trigger an attempted lookup of this absent run.
    messages.put(dict(common, kind="claimed", worker_generation=0, run="stale-run", attempt_id="b" * 32))
    messages.put(dict(common, kind="claimed", run=claim.task["run"], attempt_id=claim.token))
    messages.put(dict(common, kind="failed", run=claim.task["run"], attempt_id=claim.token,
                      capacity_retained=True, traceback="private operation failure"))
    context = SimpleNamespace(Queue=lambda: messages, Event=threading.Event)
    monkeypatch.setattr(worker.mp, "get_context", lambda _method: context)
    monkeypatch.setattr(worker, "_start_slot", lambda *_args: slot)
    args = SimpleNamespace(job=case.queue.root, _alias_stores=case.runtime.stores,
        workers=1, watchdog=True, watchdog_seconds=600., once=True)
    result = worker._parent(args, case.runtime.service_id, case.runtime.job,
                            case.runtime.commit, io.StringIO())
    assert result == (1 if fatal else 0)
    assert slot["retired"] and slot["claim"]["task"] == claim.task
    assert slot["process"].joins > 0 and "owned-process-retired" in observed
    assert len(args._cache_terminal_events) == 1
    assert args._cache_terminal_events[0]["capacity_retained"] is True
    retirement = job_queue.read_json(claim.output / "retirement.json")
    assert retirement["attempt_id"] == claim.token and retirement["all_absent"] is True
    assert retirement["identities"] == [case.owner]
    writer = slot["setup_retirement_writers"][claim.token]
    assert writer.closed and writer.written_bytes > 0
    assert helper.preclaim_sink.maximum == helper.sink.maximum
    assert helper.sink.maximum + writer.maximum == helper.admission.metadata_limits["maximum_output_bytes"]
    assert helper.sink.written_bytes + writer.written_bytes <= helper.admission.metadata_limits["maximum_output_bytes"]
    assert all((Path(row["path"]) / "released.json").is_file()
               for row in helper.admission.bound_reservations)


def test_owner_delivered_after_first_closure_is_proved_with_distinct_wrapper(tmp_path, monkeypatch):
    """A newly flushed Python owner joins a second actual absence proof."""
    case, claim, helper = _retained_claim(tmp_path, monkeypatch)
    slot = _slot(case, claim)
    _parent_seams(case, monkeypatch, slot)
    wrapper = dict(case.owner, pid=321, process_create_time="321.125")
    slot["root"] = deepcopy(wrapper)
    messages = queue.Queue()
    messages.put(dict(worker_slot=0, worker_generation=1, kind="console", stream="stdout", text="private start\n"))
    common = dict(worker_slot=0, worker_generation=1, actual_worker=deepcopy(case.owner))
    proofs = []

    def retire(root, observed):
        """Prove only the exact identities presented to the original seam."""
        assert root == wrapper
        proofs.append(deepcopy(observed))
        if len(proofs) == 1:
            assert case.owner not in observed
            return [deepcopy(wrapper)]
        assert case.owner in observed and wrapper in observed
        return [deepcopy(wrapper), deepcopy(case.owner)]

    def flush_on_join(timeout=None):
        """Model a dead child's feeder flushing only after its first closure."""
        slot["process"].joins += 1
        if slot["process"].joins == 1:
            messages.put(dict(common, kind="claimed", run=claim.task["run"], attempt_id=claim.token))
            messages.put(dict(common, kind="failed", run=claim.task["run"], attempt_id=claim.token,
                              capacity_retained=True, traceback="private operation failure"))

    slot["process"].join = flush_on_join
    monkeypatch.setattr(worker, "retire_owned_processes", retire)
    monkeypatch.setattr(worker.mp, "get_context", lambda _method:
        SimpleNamespace(Queue=lambda: messages, Event=threading.Event))
    monkeypatch.setattr(worker, "_start_slot", lambda *_args: slot)
    args = SimpleNamespace(job=case.queue.root, _alias_stores=case.runtime.stores,
        workers=1, watchdog=True, watchdog_seconds=600., once=True)
    assert worker._parent(args, case.runtime.service_id, case.runtime.job,
                           case.runtime.commit, io.StringIO()) == 0
    assert len(proofs) == 2 and slot["retired"]
    document = job_queue.read_json(claim.output / "retirement.json")
    assert document["identities"] == [wrapper, case.owner]
    assert document["actual_launch_wrapper"] == wrapper
    assert all((Path(row["path"]) / "released.json").is_file()
               for row in helper.admission.bound_reservations)


def test_supervisor_partial_and_closed_writer_cannot_reset_same_slot_allowance(tmp_path, monkeypatch):
    """Retained bytes consume one reserved slice across original retries."""
    case, claim, helper = _retained_claim(tmp_path, monkeypatch)
    slot = _slot(case, claim, attach=True)
    _parent_seams(case, monkeypatch, slot)
    args = SimpleNamespace(job=case.queue.root, _alias_stores=case.runtime.stores)
    original = worker.write_immutable
    partial = claim.output / (".retirement.json." + "a" * 32 + ".partial")

    def fail_after_prefix(path, document):
        """Inject one retained write failure through the real finite stream."""
        sink = current_writer()
        sink.register_temporary(path, partial)
        with sink.open_exclusive(partial) as stream:
            stream.write(b"x" * (sink.maximum - 1))
        raise OSError("private supervisor write failure")

    monkeypatch.setattr(worker, "write_immutable", fail_after_prefix)
    with pytest.raises(OSError, match="supervisor write failure"):
        worker._record_retirement(slot, args, case.runtime.service_id)
    writer = slot["setup_retirement_writers"][claim.token]
    spent = writer.written_bytes
    assert spent == partial.stat().st_size == retirement_allowance(case.config["metadata_output"]) - 1
    assert not (claim.output / "retirement.json").exists()
    monkeypatch.setattr(worker, "write_immutable", original)
    with pytest.raises(ValueError, match="before write"):
        worker._record_retirement(slot, args, case.runtime.service_id)
    assert slot["setup_retirement_writers"][claim.token] is writer
    assert writer.written_bytes == spent and partial.stat().st_size == spent
    assert not (claim.output / "retirement.json").exists()
    with pytest.raises(ValueError, match="exact retirement"):
        writer.write_bytes(claim.output / "unrelated.json", b"{}")
    writer.close()
    with pytest.raises(ValueError, match="closed"):
        worker._record_retirement(slot, args, case.runtime.service_id)
    assert writer.written_bytes == spent and not slot["retired"]
    assert all(not (Path(row["path"]) / "released.json").exists()
               for row in helper.admission.bound_reservations)


def test_busy_capacity_does_not_abort_supervisor_child_retirement(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import capacity
    from darsia.fluidflower.pipeline.folder_capacity import CapacityLockBusy
    case, claim, helper = _retained_claim(tmp_path, monkeypatch)
    slot = _slot(case, claim, attach=True)
    _parent_seams(case, monkeypatch, slot)
    args = SimpleNamespace(job=case.queue.root, _alias_stores=case.runtime.stores)
    original = capacity.release_retired_capacity
    calls = []
    def release(*args, **kwargs):
        calls.append(kwargs["wait_seconds"])
        if len(calls) == 1:
            raise CapacityLockBusy("Another original child still holds admission")
        return original(*args, **kwargs)
    monkeypatch.setattr(capacity, "release_retired_capacity", release)
    worker._record_retirement(slot, args, case.runtime.service_id)
    assert slot["retired"] and (claim.output / "retirement.json").is_file()
    assert claim.token in args._pending_capacity_releases and calls == [0]
    assert all(not (Path(row["path"]) / "released.json").exists()
               for row in helper.admission.bound_reservations)
    assert worker._release_pending_capacity(args, final=True)
    assert calls == [0, 30] and not args._pending_capacity_releases
    assert all((Path(row["path"]) / "released.json").is_file()
               for row in helper.admission.bound_reservations)
