"""Actual immutable serialization retries remain covered after frame progress."""
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import storage_outputs
from darsia.fluidflower.pipeline import capacity, io_ops, job_queue, option_capacity
from . import test_capacity_declared_options as options
from . import test_n1_capacity as mass
from . import test_phase1_capacity as statistics
from .test_capacity import ledger_fixture, owner


def _estimate(route, monkeypatch):
    if route == "phase1":
        task, config, profile = statistics.phase1_task_TESTONLY(count=2)
        return task, statistics.estimate_TESTONLY(task, config, profile, monkeypatch)
    if route == "phase2":
        task, config, profile = mass.n1_task_TESTONLY(count=2)
        return task, mass.estimate_TESTONLY(task, config, profile, monkeypatch)
    task, config, profile = options.other_run_TESTONLY(count=2)
    monkeypatch.setattr(capacity, "_profile_writer_identity", lambda *args: None)
    monkeypatch.setattr(capacity, "_native_layout_for_task", lambda *args: options.run_layout_TESTONLY())
    monkeypatch.setattr(capacity.subprocess, "check_output", lambda args: b"TESTONLY")
    monkeypatch.setattr(capacity, "_writer_identity", lambda *args: None)
    return task, options.estimate_TESTONLY(task, dict(config, **options.OPTIONS), profile)


@pytest.mark.parametrize("route", ("phase1", "phase2", "option"))
def test_all_failed_frame_serializations_stay_reserved_after_committed_prefix(tmp_path, monkeypatch, route):
    task, estimate = _estimate(route, monkeypatch)
    assert type(estimate) is capacity.OptionSpaceEstimate
    assert estimate.retained_serialization_estimated_bytes > 0
    task = dict(task, job_id="serialization-budget")
    claim = SimpleNamespace(task=task, token=task["attempt_id"])
    ledger, _, _ = ledger_fixture(tmp_path, monkeypatch)
    actor = owner()
    monkeypatch.setattr(job_queue, "TRANSIENT_RETRY_DELAYS", (0, 0, 0))
    original_fsync = io_ops.os.fsync
    active = dict(serializing=False, calls=0)

    def fail_three_full_flushes(fd):
        if active["serializing"]:
            active["calls"] += 1
            if active["calls"] <= 3:
                error = PermissionError("synthetic transient flush interruption")
                error.winerror = 32
                raise error
        return original_fsync(fd)

    monkeypatch.setattr(io_ops.os, "fsync", fail_three_full_flushes)
    with ledger.locked():
        reservation = ledger.bind(tmp_path / "job", claim, actor, estimate)
        initial = ledger.remaining()
    retained = []
    for index, case in enumerate(task["ordered_case_ids"], 1):
        path = tmp_path / "products" / case / "frame.npz"
        arrays = dict(values=np.random.default_rng(index).random((23, 31)).astype("float32"))
        active.update(serializing=True, calls=0)
        try:
            storage_outputs._write_npz(path, arrays)
        finally:
            active["serializing"] = False
        assert active["calls"] == 4
        partials = list(path.parent.glob(".frame.npz.*.partial"))
        assert len(partials) == 3
        assert all(partial.read_bytes() == path.read_bytes() for partial in partials)
        retained.extend(partials)
        with ledger.locked():
            ledger.progress(reservation, actor, dict(
                state="phase1_frame_completed" if route == "phase1" else "frame_completed",
                complete=True, run=task["run"], case_id=case))
            remaining = ledger.remaining()
        assert remaining == initial - index * estimate.products_per_frame_estimated_bytes
        assert remaining >= estimate.retained_serialization_estimated_bytes
        assert sum(partial.stat().st_size for partial in retained) <= estimate.retained_serialization_estimated_bytes
    assert len(retained) == 6 and all(partial.is_file() for partial in retained)


def test_changed_serialization_retry_count_cannot_reuse_the_capacity_envelope(monkeypatch):
    monkeypatch.setattr(job_queue, "TRANSIENT_RETRY_DELAYS", (0, 0, 0, 0))
    with pytest.raises(ValueError, match="retry count"):
        option_capacity.retained_serialization_bytes(2, 100, run_bytes=20)
