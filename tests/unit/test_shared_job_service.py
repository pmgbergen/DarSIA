"""Use real file reservations with an inert installed-worker launcher."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower.job_control import publish_control
from darsia.fluidflower.job_launch_state import read_launch_requests
from darsia.fluidflower.shared_job_folder import SharedJobFolder
from darsia.fluidflower.shared_job_service import SharedJobService, backend_blocker, sample_owned_resources
from darsia.fluidflower.shared_job_capabilities import job_capabilities


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_job(folder, identifier="job", *, pipeline="native-phase-n1", backend="any", runs=("a", "b")):
    root = folder / identifier
    put(root / "job.json", dict(pipeline=pipeline, expected_runs=list(runs),
                               commit="a" * 40, precision="float32", backend=backend))
    for run in runs:
        put(root / "queue" / (run + ".json"), dict(run=run, pipeline=pipeline, backend=backend))
    return root


class Harness:
    def __init__(self, tmp_path, **kwargs):
        self.folder = SharedJobFolder(tmp_path)
        self.root = make_job(tmp_path, **kwargs)
        self.entry = self.folder.publish(self.root, master_machine="master")
        self.calls = []
        self.snapshot = dict(state="Waiting", complete=False, errors=[], failed_runs=0,
                             committed=0, total_frames=20, run_count=2, completed_runs=0,
                             runs=[dict(run=run, state="Waiting", committed=0, machine="Not reported")
                                   for run in ("a", "b")])

    def launch(self, root, **options):
        self.calls.append((root, options))
        return dict(started=True)

    def service(self, **kwargs):
        return SharedJobService(self.folder, "worker", launcher=self.launch,
                                reader=lambda registration: deepcopy(self.snapshot), **kwargs)

    def retire(self, backend, kind="stopped"):
        request = next(row for row in read_launch_requests(self.root) if row["backend"] == backend)
        stamp = datetime.now(timezone.utc) + timedelta(seconds=1)
        path = self.root / "logs" / ("worker-" + backend + ".log")
        path.parent.mkdir(exist_ok=True)
        path.write_text(stamp.isoformat() + " " + json.dumps(dict(kind=kind)) + "\n", encoding="utf-8")
        return request


def test_cpu_cuda_start_together_and_survive_new_gui_session(tmp_path):
    h = Harness(tmp_path)
    service = h.service(cpu_workers=3, cuda_workers=1)
    result = service.tick()
    assert [(options["backend"], options["workers"]) for _, options in h.calls] == [("cpu", 3), ("cuda", 1)]
    assert all(options["once"] is True for _, options in h.calls)
    assert result["jobs"][0]["progress"]["committed"] == 0
    assert result["workers"][0]["role"] == "worker"
    assert result["workers"][0]["stale"] is False
    h.service(cpu_workers=8, cuda_workers=2).tick()
    assert len(h.calls) == 2
    assert len(read_launch_requests(h.root)) == 2


def _worker_report(request, *, seconds=1, **changes):
    report = dict(machine="worker-cpu", worker="slot 0", run="a", state="Build",
        last_progress_utc=(datetime.fromisoformat(request["requested_utc"]) + timedelta(seconds=seconds)).isoformat())
    report.update(changes)
    return report


def test_pending_launch_displays_newer_actual_progress_without_releasing_reservation(tmp_path):
    from darsia.fluidflower.job_launch_state import UNKNOWN
    h = Harness(tmp_path)
    service = h.service(cpu_workers=6, memory_sampler=lambda *args: {})
    service.tick()
    request = read_launch_requests(h.root)[0]
    h.snapshot.update(state="Running", workers=[_worker_report(request)])
    status = service.tick()["jobs"][0]["backends"]["cpu"]
    assert status["state"] == "Launch requested" and status["workers"] == 6
    assert "Worker reported Build in a" in status["message"] and "last progress" in status["message"]
    unchanged = read_launch_requests(h.root)[0]
    assert unchanged["state"] == UNKNOWN and not unchanged["released"]
    assert unchanged["worker_liveness_verified"] is False and unchanged["numerical_progress_claimed"] is False
    assert len(h.calls) == 1
    h.snapshot["workers"] = [_worker_report(request, run="build-ac17",
        state="Claimed (liveness not established) · Last report: build / ac17 / create native configuration")]
    message = service.tick()["jobs"][0]["backends"]["cpu"]["message"]
    assert "Worker reported Build in AC17 · create native configuration" in message
    assert len(h.calls) == 1


@pytest.mark.parametrize("changes,seconds", [
    ({"machine": "other-cpu"}, 1), ({"machine": "worker-cuda"}, 1),
    ({"backend": "cuda"}, 1), ({"machine": "worker"}, 1),
    ({}, -1), ({}, 0), ({"last_progress_utc": "Not reported"}, 1),
    ({"last_progress_utc": "2026-10-27T00:00:00"}, 1),
    ({"state": "Queued"}, 1), ({"state": ""}, 1), ({"run": ""}, 1),
])
def test_pending_launch_keeps_unknown_without_a_matching_new_worker_report(tmp_path, changes, seconds):
    from darsia.fluidflower.job_launch_state import UNKNOWN
    h = Harness(tmp_path)
    service = h.service(cpu_workers=1, memory_sampler=lambda *args: {})
    service.tick()
    request = read_launch_requests(h.root)[0]
    h.snapshot["workers"] = [_worker_report(request, seconds=seconds, **changes)]
    status = service.tick()["jobs"][0]["backends"]["cpu"]
    assert status["message"] == UNKNOWN and len(h.calls) == 1


def test_reported_progress_uses_latest_same_backend_and_preserves_terminal_refusals(tmp_path):
    from darsia.fluidflower.job_launch_state import record_launch_error
    h = Harness(tmp_path)
    service = h.service(cpu_workers=1, memory_sampler=lambda *args: {})
    service.tick()
    request = read_launch_requests(h.root)[0]
    h.snapshot["workers"] = [_worker_report(request, seconds=2, machine="worker", backend="cpu", run="b"),
        _worker_report(request, seconds=5, machine="worker-cuda", run="wrong backend"), _worker_report(request)]
    status = service.tick()["jobs"][0]["backends"]["cpu"]
    assert "Build in b" in status["message"] and "2 worker reports" in status["message"]
    assert "wrong backend" not in status["message"]
    record_launch_error(request, "Worker startup refused")
    status = service.tick()["jobs"][0]["backends"]["cpu"]
    assert status["message"] == "Worker startup refused" and len(h.calls) == 1
    h.retire("cpu", "startup_failure")
    status = service.tick()["jobs"][0]["backends"]["cpu"]
    assert status["state"] == "Blocked" and len(h.calls) == 1


def test_queued_job_with_no_worker_report_remains_unknown_after_launch(tmp_path):
    from darsia.fluidflower.job_launch_state import UNKNOWN
    h = Harness(tmp_path)
    service = h.service(cpu_workers=1, memory_sampler=lambda *args: {})
    service.tick()
    status = service.tick()["jobs"][0]["backends"]["cpu"]
    assert h.snapshot["state"] == "Waiting" and status["message"] == UNKNOWN
    assert not list((h.root / "ui/commands").glob("*.json")) and len(h.calls) == 1


def test_prestart_failure_displays_recorded_reason_instead_of_generic_failure(tmp_path):
    h = Harness(tmp_path)
    service = h.service(cpu_workers=1)
    service.tick()
    request = h.retire("cpu", "startup_failure")
    reason = "RuntimeError: Startup requires a clean Git worktree at job.json commit"
    stamp = datetime.now(timezone.utc) + timedelta(seconds=1)
    path = h.root / "logs" / "worker-cpu.log"
    path.write_text(stamp.isoformat() + " " + json.dumps(dict(
        kind="startup_failure", source="darsia_service.prestart", error=reason,
    )) + "\n", encoding="utf-8")
    status = service.tick()["jobs"][0]["backends"]["cpu"]
    assert status["state"] == "Blocked"
    assert status["message"] == reason
    assert len(h.calls) == 1
    assert read_launch_requests(h.root)[0]["source"] == request["source"]


def test_capacity_changes_wait_for_retirement_and_collect_owned_sample(tmp_path):
    h = Harness(tmp_path)
    service = h.service(cpu_workers=1)
    service.tick()
    service.update_capacity(4, 0)
    running = service.tick()["jobs"][0]["backends"]["cpu"]
    assert running["workers"] == 1 and running["configured_workers"] == 4
    assert len(h.calls) == 1
    h.retire("cpu")
    h.snapshot["runs"][0].update(state="Complete", committed=10, machine="worker-cpu")
    h.snapshot.update(committed=10, completed_runs=1)
    (h.root / "queue/a.json").unlink()
    result = service.tick()
    assert h.calls[-1][1]["workers"] == 4 and len(h.calls) == 2
    assert len(result["samples"]) == 1
    assert result["samples"][0]["completed_frames"] == 10
    assert result["samples"][0]["workers"] == 1
    assert result["samples"][0]["elapsed_seconds"] > 0


def test_setup_apply_batch_retirement_replenishes_full_parallel_build_capacity(tmp_path):
    h = Harness(tmp_path,pipeline="series-setup",backend="cpu",runs=("setup-apply","build-a","build-b"))
    put(h.root/"queue/setup-apply.json", dict(run="setup-apply", pipeline="series-setup", backend="cpu", operation="apply"))
    for child in ("build-a","build-b"):
        (h.root/"queue"/(child+".json")).unlink()
    service = SharedJobService(h.folder,"master",cpu_workers=6,launcher=h.launch,
        reader=lambda _:deepcopy(h.snapshot))
    result = service.tick()
    assert len(h.calls)==1 and h.calls[0][1]["workers"]==6 and h.calls[0][1]["once"]
    assert result["jobs"][0]["backends"]["cpu"]["configured_workers"] == 6
    assert service.cpu_workers == 6
    # The detached supervisor gets all six workers of capacity. It starts only
    # the serial prerequisite, then continues Build without the GUI timer.
    (h.root/"queue/setup-apply.json").unlink()
    service.tick()
    assert len(h.calls)==1
    for child in ("build-a","build-b"):
        put(h.root/"queue"/(child+".json"),dict(run=child,pipeline="series-setup",backend="cpu",operation="build"))
    request=read_launch_requests(h.root)[0]
    stamp=datetime.now(timezone.utc)+timedelta(seconds=1)
    log=Path(request["service_log"])
    log.parent.mkdir(exist_ok=True)
    log.write_text(stamp.isoformat()+" "+json.dumps(dict(kind="stopped"))+"\n",encoding="utf-8")
    service.tick()
    assert len(h.calls)==2 and h.calls[-1][1]["workers"]==6


def test_queued_build_does_not_inherit_serial_apply_allocation(tmp_path):
    h = Harness(tmp_path, pipeline="series-setup", backend="cpu", runs=("build-a",))
    put(h.root/"queue/build-a.json", dict(run="build-a", pipeline="series-setup", backend="cpu", operation="build"))
    service = SharedJobService(h.folder, "master", cpu_workers=6, launcher=h.launch,
        reader=lambda _: deepcopy(h.snapshot))
    service.tick()
    assert [options["workers"] for _, options in h.calls] == [6]


def test_another_machine_progress_is_not_local_tuning_evidence(tmp_path):
    h = Harness(tmp_path)
    service = h.service(cpu_workers=1)
    service.tick()
    h.retire("cpu")
    h.snapshot["runs"][0].update(state="Complete", committed=10, machine="other-cpu")
    result = service.tick()
    assert result["samples"] == []


def test_one_backend_job_at_a_time_and_leave_does_not_interrupt(tmp_path):
    h = Harness(tmp_path)
    h.folder.publish(make_job(tmp_path, "second"), master_machine="master")
    service = h.service(cpu_workers=2, cuda_workers=1)
    service.tick()
    assert len(h.calls) == 2
    assert h.calls[0][0] == h.calls[1][0]
    requests = sum(len(read_launch_requests(path)) for path in (h.root, tmp_path / "second"))
    assert requests == 2
    result = service.leave()
    assert result["state"] == "offline"
    assert result["workers"][0]["stale"] is True
    service.tick()
    assert len(h.calls) == 2
    assert not list(h.root.glob("ui/commands/*.json"))


def test_partial_start_failure_never_relaunches_successful_cpu(tmp_path):
    h = Harness(tmp_path)

    def launch(root, **options):
        h.calls.append((root, options))
        if options["backend"] == "cuda":
            raise RuntimeError("CUDA device unavailable")
        return dict(started=True)

    service = SharedJobService(h.folder, "worker", cpu_workers=2, cuda_workers=1,
                               launcher=launch, reader=lambda _: deepcopy(h.snapshot))
    result = service.tick()
    assert result["jobs"][0]["backends"]["cuda"]["message"] == "CUDA device unavailable"
    service.tick()
    assert len(h.calls) == 2
    assert "CUDA device unavailable" in read_launch_requests(h.root)[1]["error_traceback"] or any(
        "CUDA device unavailable" in row["error_traceback"] for row in read_launch_requests(h.root))


def test_explicit_new_join_can_retry_released_failure_without_polling_loop(tmp_path):
    h = Harness(tmp_path)
    service = h.service(cpu_workers=1)
    service.tick()
    h.retire("cpu", "startup_failure")
    failed = service.tick()["jobs"][0]["backends"]["cpu"]
    assert failed["state"] == "Blocked"
    assert len(h.calls) == 1
    h.service(cpu_workers=1).tick()
    assert len(h.calls) == 2
    assert len(read_launch_requests(h.root)) == 2


def test_master_stop_and_metadata_failure_do_not_start_work(tmp_path):
    h = Harness(tmp_path)
    publish_control(h.root, "stop", machine="master", expected_previous=None)
    service = h.service(cpu_workers=1)
    result = service.tick()
    assert not h.calls
    assert result["jobs"][0]["backends"]["cpu"]["state"] == "stopped"
    (h.root / "job.json").write_text("{}", encoding="utf-8")
    assert service.tick()["errors"]
    assert not h.calls


def test_reset_holds_active_service_until_explicit_recovery(tmp_path):
    from darsia.fluidflower.job_lifecycle import recover_job, user_status
    h = Harness(tmp_path)
    service = h.service(cpu_workers=1)
    service.tick()
    h.retire("cpu", "startup_failure")
    assert service.tick()["jobs"][0]["backends"]["cpu"]["state"] == "Blocked"
    assert len(h.calls) == 1
    reset = recover_job(h.root, machine="master", expected_previous=None, reset=True)
    for _ in range(3):
        assert service.tick()["jobs"][0]["state"] == "Pending"
    assert len(h.calls) == 1
    assert user_status(dict(h.snapshot, root=str(h.root)))["state"] == "Pending"
    recover_job(h.root, machine="master", expected_previous=reset)
    service.tick()
    assert len(h.calls) == 2


def test_cpu_only_task_and_comparison_are_respected(tmp_path):
    h = Harness(tmp_path, pipeline="preset-wasserstein", backend="cpu")
    result = h.service(cpu_workers=9, cuda_workers=2).tick()
    assert len(h.calls) == 1 and h.calls[0][1]["workers"] == 1
    assert result["jobs"][0]["backends"]["cuda"]["state"] == "Unavailable"


def test_automatic_setup_with_gpu_available_launches_only_cpu(tmp_path):
    from darsia.fluidflower.shared_worker_tuning import AutomaticWorkerCapacity

    h = Harness(tmp_path, pipeline="series-setup", backend="cpu")
    inventory = dict(cpu=dict(physical_cores=8, logical_cores=16),
        memory=dict(available_bytes=100 * 1024 ** 3), accelerators=[dict(
            family="cuda", supported_backends=["cuda-float32"], runtime_ready=True,
            device_id="0", stable_id="GPU-local", available_bytes=20 * 1024 ** 3)])
    tuner = AutomaticWorkerCapacity(inventory, machine="master")
    assert tuner.counts() == (1, 1)
    service = SharedJobService(h.folder, "master", role="master", tuner=tuner,
        cpu_workers=1, cuda_workers=1, launcher=h.launch,
        reader=lambda _: deepcopy(h.snapshot))
    result = service.tick()
    assert [(options["backend"], options["workers"]) for _, options in h.calls] == [("cpu", 1)]
    assert [request["backend"] for request in read_launch_requests(h.root)] == ["cpu"]
    assert result["jobs"][0]["backends"]["cpu"]["state"] == "Launch requested"
    assert result["jobs"][0]["backends"]["cuda"]["state"] == "Unavailable"
    assert result["jobs"][0]["state"] == "Waiting"
    assert result["errors"] == []


def test_only_compatible_queued_tasks_get_services(tmp_path):
    h = Harness(tmp_path)
    for run in ("a", "b"):
        put(h.root / "queue" / (run + ".json"), dict(run=run, pipeline="native-phase-n1", backend="cpu"))
    result = h.service(cpu_workers=1, cuda_workers=1).tick()
    assert len(h.calls) == 1 and h.calls[0][1]["backend"] == "cpu"
    assert result["jobs"][0]["backends"]["cuda"]["message"] == "No queued work for this backend"


@pytest.mark.parametrize("job,backend,machine,master", [
    ({"pipeline": "series-setup"}, "cpu", "worker", "master"),
    ({"pipeline": "n1-calibration-refit"}, "cuda", "worker", "master"),
    ({"pipeline": "native-phase-statistics", "execution_mode": "current_build", "target_machine": "master"}, "cpu", "worker", "master"),
    ({"pipeline": "native-phase-statistics", "execution_mode": "current_build", "target_machine": "master"}, "cuda", "master", "master"),
    ({"pipeline": "unknown"}, "cpu", "worker", "master"),
])
def test_local_and_pipeline_limits_remain(job, backend, machine, master):
    assert backend_blocker(job, backend, machine=machine, master_machine=master)


def test_tuning_observes_retired_batch_before_starting_next_batch(tmp_path):
    h = Harness(tmp_path)

    class Tuner:
        note = "Measured local batches"
        workers = 1

        def observe(self, snapshot):
            if snapshot["samples"]:
                assert len(h.calls) == 1
                self.workers = 3

        def counts(self):
            return self.workers, 0

    tuner = Tuner()
    service = h.service(cpu_workers=1, tuner=tuner)
    service.tick()
    h.retire("cpu")
    h.snapshot["runs"][0].update(state="Complete", committed=10, machine="worker-cpu")
    h.snapshot.update(committed=10, completed_runs=1)
    (h.root / "queue/a.json").unlink()
    result = service.tick()
    assert len(h.calls) == 2 and h.calls[-1][1]["workers"] == 3
    assert result["tuning_note"] == tuner.note
    assert result["workers"][0]["cpu_workers"] == 3
    service.set_tuner(None)
    service.update_capacity(2, 0)
    assert service.cpu_workers == 2


def test_automatic_waits_for_both_backends_before_starting_new_allocation(tmp_path):
    h = Harness(tmp_path, runs=("a", "b", "c", "d"))
    h.snapshot["runs"].extend(dict(run=run, state="Waiting", committed=0, machine="Not reported")
                              for run in ("c", "d"))
    h.snapshot.update(run_count=4, total_frames=40)

    class Tuner:
        note = "Automatic"
        pair = (1, 1)

        def observe(self, snapshot):
            if {sample["backend"] for sample in snapshot["samples"]} == {"cpu", "cuda"}:
                self.pair = (2, 2)

        def counts(self):
            return self.pair

    service = h.service(cpu_workers=1, cuda_workers=1, tuner=Tuner())
    service.tick()
    assert len(h.calls) == 2
    h.retire("cpu")
    h.snapshot["runs"][0].update(state="Complete", committed=10, machine="worker-cpu")
    h.snapshot.update(committed=10, completed_runs=1)
    (h.root / "queue/a.json").unlink()
    waiting = service.tick()
    assert len(h.calls) == 2
    statuses = waiting["jobs"][0]["backends"]
    assert statuses["cpu"]["message"] == "Waiting for current automatic batch to finish"
    assert statuses["cuda"]["workers"] == 1
    # Neither the coordinator nor the barrier sends a Stop to the GPU worker.
    assert not list((h.root / "ui/commands").glob("*.json"))
    h.retire("cuda")
    h.snapshot["runs"][1].update(state="Complete", committed=10, machine="worker-cuda")
    h.snapshot.update(committed=20, completed_runs=2)
    (h.root / "queue/b.json").unlink()
    service.tick()
    assert [(options["backend"], options["workers"]) for _, options in h.calls] == [
        ("cpu", 1), ("cuda", 1), ("cpu", 2), ("cuda", 2)]


def test_manual_cpu_can_continue_while_gpu_batch_is_running(tmp_path):
    h = Harness(tmp_path)
    service = h.service(cpu_workers=1, cuda_workers=1)
    service.tick()
    h.retire("cpu")
    h.snapshot["runs"][0].update(state="Complete", committed=10, machine="worker-cpu")
    (h.root / "queue/a.json").unlink()
    service.tick()
    assert [options["backend"] for _, options in h.calls] == ["cpu", "cuda", "cpu"]


@pytest.mark.parametrize("birth,expected", [(1234.0, True), (4321.0, False)])
def test_memory_sample_requires_exact_original_claim_owner_birth(tmp_path, monkeypatch, birth, expected):
    import psutil
    task = dict(run="a", threads=2, attempt_id="a" * 32)
    put(tmp_path / "claimed/worker-cpu/a.json", task)
    owner = tmp_path / "leases" / ("a." + "a" * 32) / "owner.json"
    put(owner, dict(task=task, token="a" * 32,
                    actual_owner=dict(machine="worker-cpu", pid=123, process_create_time="1234.0")))
    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(
        pid=pid, create_time=lambda: birth, memory_info=lambda: SimpleNamespace(rss=4096)))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=1000000))
    snapshot = dict(workers=[dict(machine="worker-cpu", run="a")])
    value = sample_owned_resources(tmp_path, "worker", "cpu", snapshot)
    assert bool(value) is expected
    if expected:
        assert value["peak_worker_rss_bytes"] == 4096
        assert value["threads"] == 2
        assert value["source"]["path"] == str(owner)
        assert "peak_device_memory_bytes" not in value


def test_capability_hints_keep_whole_run_execution_and_measure_gpu_preference():
    supported = job_capabilities(dict(pipeline="native-phase-n1"))
    assert supported["supported_backends"] == ["cpu", "cuda"]
    assert supported["scheduling_unit"] == "whole_run"
    assert supported["stages"][1]["preference"] == "measured"
    assert not any(stage["independently_scheduled"] for stage in supported["stages"])
    assert job_capabilities(dict(pipeline="preset-wasserstein"))["supported_backends"] == ["cpu"]
    assert job_capabilities(dict(pipeline="native-phase-n1", backend="cpu"))["supported_backends"] == ["cpu"]
    assert job_capabilities(dict(pipeline="native-phase-statistics", execution_mode="current_build"))["supported_backends"] == ["cpu"]
    assert job_capabilities(dict(pipeline="future"))["supported_backends"] == []
