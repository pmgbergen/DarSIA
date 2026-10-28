"""Local completed throughput and shared resource budgets drive batch choices."""

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from darsia.fluidflower.shared_worker_tuning import AutomaticWorkerCapacity


GIB = 1024 ** 3
NOW = 1_790_000_000


def inventory(*, gpu=False, ram=100 * GIB, cores=8):
    return dict(cpu=dict(physical_cores=cores, logical_cores=cores * 2),
                memory=dict(available_bytes=ram), accelerators=[dict(
                    family="cuda", supported_backends=["cuda-float32"], runtime_ready=True,
                    device_id="0", stable_id="GPU-actual", available_bytes=20 * GIB,
                )] if gpu else [])


def stamp(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def sample(identifier, *, pair=(1, 0), backend="cpu", frames=100, seconds=10,
           memory=True, workload="job-a", pipeline="native-phase-n1", ram=5 * GIB,
           available=100 * GIB, device=5 * GIB):
    value = dict(sample_id=identifier, job_id=workload, workload_key=workload,
                 backend=backend, workers=pair[0 if backend == "cpu" else 1],
                 completed_frames=frames, elapsed_seconds=seconds,
                 started_utc=stamp(NOW - seconds), completed_utc=stamp(NOW),
                 job_facts=dict(pipeline=pipeline, precision="float32", commit="a" * 40),
                 configured_counts=dict(cpu=pair[0], cuda=pair[1]))
    if memory:
        value.update(peak_worker_rss_bytes=ram, available_bytes=available, threads=1,
                     source=dict(path="owner.json", sha256="a" * 64, size_bytes=128))
        if backend == "cuda":
            value.update(device_id="GPU-actual", peak_device_memory_bytes=device,
                         available_device_bytes=20 * GIB)
    return value


def observe(tuner, *samples, machine="host", errors=None):
    tuner.observe(dict(machine=machine, samples=list(samples), errors=errors or []))


def tuner(**kwargs):
    return AutomaticWorkerCapacity(machine="host", now=lambda: NOW, **kwargs)


def test_starts_conservatively_only_on_available_devices():
    assert tuner(inventory=inventory()).counts() == (1, 0)
    assert tuner(inventory=inventory(gpu=True)).counts() == (1, 1)
    absent = inventory(gpu=True)
    absent["accelerators"][0]["runtime_ready"] = False
    assert tuner(inventory=absent).counts() == (1, 0)
    empty = tuner(inventory=inventory(ram=0))
    assert empty.counts() == (0, 0)
    assert "free memory" in empty.note


def test_increases_one_at_a_time_and_returns_to_better_throughput():
    adaptive = tuner(inventory=inventory())
    observe(adaptive, sample("one"))
    assert adaptive.counts() == (2, 0)
    observe(adaptive, sample("two", pair=(2, 0), frames=160))
    assert adaptive.counts() == (3, 0)
    observe(adaptive, sample("three", pair=(3, 0), frames=130))
    assert adaptive.counts() == (2, 0)
    assert "retaining" in adaptive.note
    # The same receipt cannot drive another change.
    observe(adaptive, sample("three", pair=(3, 0), frames=130))
    assert adaptive.counts() == (2, 0)


@pytest.mark.parametrize("mutation", [
    dict(completed_frames=0), dict(elapsed_seconds=float("inf")),
    dict(error="failed"), dict(stale=True), dict(completed_utc=stamp(NOW - 700)),
    dict(completed_utc=stamp(NOW + 40)), dict(started_utc=stamp(NOW + 1)),
    dict(workers=True), dict(job_facts={}), dict(workload_key=""),
])
def test_rejects_invalid_or_stale_samples(mutation):
    adaptive = tuner(inventory=inventory())
    observe(adaptive, dict(sample("bad"), **mutation))
    assert adaptive.counts() == (1, 0)
    assert not adaptive._seen


def test_other_machine_and_global_progress_cannot_change_capacity():
    adaptive = tuner(inventory=inventory())
    observe(adaptive, sample("remote"), machine="another")
    adaptive.observe(dict(machine="host", committed=10000, total_frames=10001))
    observe(adaptive, sample("error"), errors=["read failed"])
    assert adaptive.counts() == (1, 0)


def test_unknown_memory_does_not_increase_worker_count():
    adaptive = tuner(inventory=inventory())
    observe(adaptive, sample("unknown", memory=False))
    assert adaptive.counts() == (1, 0)
    assert "RAM/VRAM" in adaptive.note


def test_combination_uses_joint_wall_time_and_shared_ram_budget():
    adaptive = tuner(inventory=inventory(gpu=True, ram=55 * GIB))
    observe(adaptive, sample("cpu", pair=(1, 1), frames=50, seconds=10,
                             available=55 * GIB),
            sample("gpu", pair=(1, 1), backend="cuda", frames=100, seconds=20,
                   available=55 * GIB))
    # Both use the same RAM: 3*5GiB fits the 15GiB budget; four do not.
    assert adaptive.counts() == (2, 1)
    assert adaptive._best[1] == pytest.approx(150 / 20)
    observe(adaptive, sample("cpu2", pair=(2, 1), frames=100, seconds=10,
                             available=55 * GIB),
            sample("gpu2", pair=(2, 1), backend="cuda", frames=100, seconds=10,
                   available=55 * GIB))
    assert sum(adaptive.counts()) <= 3
    assert adaptive._best[0] == (2, 1)


def test_gpu_memory_uuid_matches_inventory_and_bounds_gpu_count():
    adaptive = tuner(inventory=inventory(gpu=True))
    observe(adaptive, sample("cpu", pair=(1, 1)),
            sample("gpu", pair=(1, 1), backend="cuda", device=15 * GIB))
    limits = adaptive._budget(sample("facts")["job_facts"])[0]
    assert limits[1] == 1
    assert adaptive.history[-1]["device_id"] == "0"


def test_new_workload_resets_counts_and_cannot_reuse_old_memory():
    adaptive = tuner(inventory=inventory())
    observe(adaptive, sample("first"))
    assert adaptive.counts() == (2, 0)
    observe(adaptive, sample("other", workload="job-b", pair=(2, 0), memory=False))
    assert adaptive.counts() == (1, 0)
    observe(adaptive, sample("other-baseline", workload="job-b", memory=False))
    assert adaptive.counts() == (1, 0)


def test_can_measure_cpu_only_after_a_mixed_batch_without_assuming_gpu_faster():
    adaptive = tuner(inventory=inventory(gpu=True, ram=50 * GIB))
    observe(adaptive, sample("cpu", pair=(1, 1), available=50 * GIB),
            sample("gpu", pair=(1, 1), backend="cuda", available=50 * GIB))
    assert adaptive.counts() in {(0, 1), (1, 0)}
    pure = adaptive.counts()
    backend = "cpu" if pure[0] else "cuda"
    observe(adaptive, sample("pure", pair=pure, backend=backend, frames=300, available=50 * GIB))
    assert adaptive._best[0] == pure


def test_memory_exhaustion_reduces_to_zero_instead_of_overcommitting():
    adaptive = tuner(inventory=inventory())
    observe(adaptive, sample("low", available=40 * GIB, ram=GIB))
    assert adaptive.counts() == (0, 0)
    assert "memory limits" in adaptive.note


def test_cpu_only_queue_does_not_wait_for_an_ineligible_gpu_batch():
    adaptive = tuner(inventory=inventory(gpu=True))
    measured = sample("cpu-only", pair=(1, 1))
    measured["eligible_backends"] = ["cpu"]
    observe(adaptive, measured)
    assert adaptive.counts() == (1, 0)
    assert adaptive._best is not None
