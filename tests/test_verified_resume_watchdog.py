"""Synthetic verified-resume and owned-process watchdog contracts.

These tests are inert in the implementation patch until Root applies and runs it.
The existing tiny saved-model fixture is reused without changing old tests.
"""

from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

from tests.server_lease_clock_fixture import server_lease_clock

import psutil
import pytest

from tests.test_n1_pipeline import BASE_COMMIT, tiny_config as original_tiny_config
from darsia.fluidflower.mass import n1_pipeline as n1
from darsia.fluidflower.pipeline import checkpoint, worker
from darsia.fluidflower.pipeline.job_queue import JobQueue, read_json, write_immutable


@pytest.fixture
def tiny_config(original_tiny_config, tmp_path):
    """Supply explicit field DATA to the canonical core without editing old tests."""
    config, _ = original_tiny_config
    config = dict(
        config,
        geometry_fields={
            "source_to_native": "H_ac28_to_frame_native",
            "original_domain_to_native": "H_ac28_to_frame_native",
            "carried_homography_provenance": "original_H_ac28_to_native",
        },
    )
    path = tmp_path / "explicit_geometry_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return config, n1.asset(path)


class Interrupted(BaseException):
    pass


def _task(config, bound):
    return dict(
        run=config["run"],
        configuration=bound,
        commit=BASE_COMMIT,
        calibration_sha256=config["calibration"]["sha256"],
        threads=1,
    )


def _retire_fixture(task, output):
    # Synthetic process closure used only by these local data-contract tests.
    identities = [dict(pid=123, process_create_time="1.0")]
    proof = checkpoint.retirement_document(
        task, output, identities, dict(pid=321, process_create_time="2.0")
    )
    write_immutable(output / "retirement.json", proof)
    return dict(
        attempt_directory=str(output), retirement=n1.asset(output / "retirement.json")
    )


def _stop_after_first(config, bound, tmp_path):
    old = tmp_path / "attempts" / ("a" * 32)
    task = _task(config, bound)

    def progress(event):
        if event["state"] == "frame_completed":
            raise Interrupted()

    with pytest.raises(Interrupted):
        n1.run_n1(task, old, "cpu", "float32", progress)
    return task, old, _retire_fixture(task, old)


def test_real_sparse_resume_skips_verified_frame_and_preserves_original_producer(
    tiny_config, tmp_path, monkeypatch
):
    config, bound = tiny_config
    task, old, source = _stop_after_first(config, bound, tmp_path)
    prefix_bytes = {p.name: p.read_bytes() for p in (old / "frames/sample_1").iterdir()}
    original = n1.N1Pipeline.evaluate
    seen = []

    def evaluate(self, spec):
        seen.append(spec["case_id"])
        return original(self, spec)

    monkeypatch.setattr(n1.N1Pipeline, "evaluate", evaluate)
    new = old.parent / ("b" * 32)
    outcome = n1.run_n1(dict(task, resume_from=source), new, "cpu", "float32")
    assert seen == ["sample_3"]  # Configured sparse order, never range(1,n).
    assert outcome["ordered_case_ids"] == ["sample_1", "sample_3"]
    assert (
        outcome["resumed_frame_count"] == 1
        and outcome["frames_computed_this_attempt"] == 1
    )
    assert Path(outcome["records"][0]["nas_directory"]) == old / "frames/sample_1"
    assert not (new / "frames/sample_1").exists()
    assert {
        p.name: p.read_bytes() for p in (old / "frames/sample_1").iterdir()
    } == prefix_bytes
    assert outcome["frame_producers"][0]["manifest"]["path"] == str(
        old / "manifest.json"
    )


@pytest.mark.parametrize(
    "change", ["payload", "configuration", "backend", "retirement"]
)
def test_untrusted_resume_is_rejected_before_any_new_pixel_evaluation(
    tiny_config, tmp_path, monkeypatch, change
):
    config, bound = tiny_config
    task, old, source = _stop_after_first(config, bound, tmp_path)
    if change == "payload":
        path = old / "frames/sample_1/products.npz"
        path.write_bytes(path.read_bytes() + b"changed")
    elif change == "configuration":
        task = dict(task, configuration=dict(bound, sha256="0" * 64))
    elif change == "backend":
        task["pool_limit_bytes"] = 1024
    else:
        source["retirement"] = dict(source["retirement"], sha256="0" * 64)

    def forbidden(*args, **kwargs):
        raise AssertionError("An untrusted prefix reached numerical evaluation")

    monkeypatch.setattr(n1.N1Pipeline, "evaluate", forbidden)
    new = old.parent / ("b" * 32)
    with pytest.raises(ValueError):
        n1.run_n1(
            dict(task, resume_from=source),
            new,
            "cuda" if change == "backend" else "cpu",
            "float32",
        )
    assert not (new / "manifest.json").exists()


def test_resume_requires_contiguous_prefix_and_uses_original_input_not_labels(
    tiny_config, tmp_path
):
    config, bound = tiny_config
    task, old, source = _stop_after_first(config, bound, tmp_path)
    verified = checkpoint.verify_frame_prefix(
        task, config, old.parent / ("b" * 32), "cpu", "float32", source
    )
    assert verified["predecessor_index"] == 0
    assert verified["predecessor_input"]["current"] == config["frames"][0]["current"]
    assert not {"labels", "products", "prediction"}.intersection(
        verified["predecessor_input"]
    )
    frame = old / "frames/sample_1"
    # A valid-looking later final directory cannot skip the first acquisition.
    frame.rename(old / "frames/sample_3")
    with pytest.raises(ValueError, match="contiguous"):
        checkpoint.verify_frame_prefix(
            task, config, old.parent / ("b" * 32), "cpu", "float32", source
        )


def test_retired_reclaimed_source_is_bound_to_current_claim_generation(tmp_path, server_lease_clock):
    shared = JobQueue(tmp_path, server_clock=server_lease_clock.clock)
    task = dict(
        run="sample",
        commit="1" * 40,
        calibration_sha256="2" * 64,
        configuration=dict(path=r"\\server\share\config.json", sha256="3" * 64),
        precision="float32",
        backend="cpu",
    )
    shared.enqueue(task)
    old = shared.claim("first", "cpu", dict(pid=123, process_create_time="1.0"))
    server_lease_clock.stamp_progress(old)
    proof = checkpoint.retirement_document(
        old.task,
        old.output,
        [dict(pid=123, process_create_time="1.0")],
        dict(pid=321, process_create_time="2.0"),
    )
    write_immutable(old.output / "retirement.json", proof)
    server_lease_clock.advance(601)
    shared.reclaim_stale()
    current = shared.claim("second", "cpu", dict(pid=456, process_create_time="3.0"))
    source = checkpoint.resume_source_for_claim(shared, current)
    assert Path(source["attempt_directory"]) == old.output
    assert source["retirement"] == n1.asset(old.output / "retirement.json")


def _wait_forever():
    while True:
        time.sleep(0.05)


def test_owned_process_is_actually_absent_before_replacement_starts():
    context = mp.get_context("spawn")
    old = context.Process(target=_wait_forever)
    old.start()
    try:
        bound = worker._process_identity(psutil.Process(old.pid))
        retired = worker.retire_owned_processes(bound)
        assert worker._same_process(bound) is None
        old.join(timeout=2)
        replacement = context.Process(target=_wait_forever)
        replacement.start()
        try:
            assert replacement.is_alive()
            assert retired[0]["pid"] == old.pid
        finally:
            replacement.terminate()
            replacement.join(timeout=2)
    finally:
        if old.is_alive():
            old.terminate()
            old.join(timeout=2)


def test_pid_reuse_never_terminates_the_new_incarnation(monkeypatch):
    calls = []

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 999.0

        def terminate(self):
            calls.append(self.pid)

    monkeypatch.setattr(psutil, "Process", Process)
    monkeypatch.setattr(psutil, "wait_procs", lambda processes, timeout: ([], []))
    old = dict(pid=123, process_create_time="1.0")
    assert worker.retire_owned_processes(old) == [old]
    assert calls == []


def test_control_heartbeat_never_extends_queue_progress_lease(tmp_path, server_lease_clock):
    shared = JobQueue(tmp_path, server_clock=server_lease_clock.clock)
    shared.enqueue(
        dict(
            run="sample",
            commit="1" * 40,
            calibration_sha256="2" * 64,
            configuration=dict(path=r"\\server\share\config.json", sha256="3" * 64),
        )
    )
    claim = shared.claim("machine", "cpu", dict(pid=123, process_create_time="1.0"))
    server_lease_clock.stamp_progress(claim)
    # Parent liveness messages cannot call JobQueue.progress or create progress files.
    server_lease_clock.advance(601)
    assert not list(claim.output.glob("progress_*.json"))
    assert len(shared.reclaim_stale()) == 1


def test_second_interruption_retains_first_original_frame_origin(
    tiny_config, tmp_path, monkeypatch
):
    config, bound = tiny_config
    task, first, source = _stop_after_first(config, bound, tmp_path)
    second = first.parent / ("b" * 32)

    def stop_before_next_frame(event):
        if event["state"] == "frame_started":
            raise Interrupted()

    with pytest.raises(Interrupted):
        n1.run_n1(
            dict(task, resume_from=source),
            second,
            "cpu",
            "float32",
            stop_before_next_frame,
        )
    next_source = _retire_fixture(task, second)
    third = first.parent / ("c" * 32)
    outcome = n1.run_n1(dict(task, resume_from=next_source), third, "cpu", "float32")
    assert outcome["frames_computed_this_attempt"] == 1
    assert outcome["frame_producers"][0]["attempt_directory"] == str(first)
    assert not (second / "frames/sample_1").exists()
    assert not (third / "frames/sample_1").exists()
