"""Real typed worker dispatch with explicit lifecycle doubles, not science proof."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

from darsia.fluidflower import preset_wasserstein_runner
from darsia.fluidflower.pipeline import checkpoint, worker, worker_io
from .test_preset_wasserstein_results import _context


@pytest.mark.parametrize("pause", [False, True])
def test_original_worker_dispatch_closes_time_and_never_requests_frame_resume(tmp_path, monkeypatch, pause):
    case = _context(tmp_path)
    case.job["environment_lock"] = dict(path="requirements-worker.lock", sha256="2" * 64)
    events, calls = [], []
    state = dict(claimed=False, binding=False, boundaries=0)
    owner = dict(machine="fixture-cpu", pid=123)

    class Shared:
        def __init__(self, *args, **kwargs):
            pass

        def recover_publications(self):
            pass

        def reclaim_stale(self):
            return []

        def claim(self, *args, **kwargs):
            if state["claimed"]:
                return None
            state["claimed"] = True
            return case

        def progress(self, claim, event):
            assert claim is case and state["binding"]
            assert event["state"] == "preset_wasserstein_progress"
            calls.append("published-progress")

        def complete(self, claim, result):
            assert not state["binding"] and claim is case
            assert result["actual_worker"] == owner and result["task"] == case.task
            assert result["precision"] == "original" and "calibration_sha256" not in result
            calls.append("complete")

        def fail(self, *args):
            pytest.fail("Comparison dispatch or pause must not reach task failure")

    class Binding:
        @contextmanager
        def activate(self):
            state["binding"] = True
            try:
                yield self
            finally:
                state["binding"] = False
                calls.append("input-context-closed")

    class Runtime:
        def __init__(self, *args):
            self.publications = SimpleNamespace(publish_progress=lambda *args: None)

        @contextmanager
        def capacity_claim(self, *args):
            yield SimpleNamespace(availability=lambda task: (), bind=lambda claim: None)

        def output_factory(self, *args):
            pass

        def prepare(self, claim):
            assert claim is case
            return Binding()

        def capacity_progress(self, *args):
            pass

        def capacity_close(self, claim):
            assert not state["binding"]
            calls.append("capacity-closed")

        def finalize_common(self, claim):
            assert not state["binding"]

        def release(self, claim):
            assert not state["binding"]
            calls.append("released")

    def execute(task, output, backend, precision, progress):
        assert state["binding"] and output == case.output
        assert task == dict(case.task, environment_lock=case.job["environment_lock"])
        assert "resume_from" not in task
        assert (backend, precision) == ("cpu", "original")
        calls.append("original-runner-dispatched")
        progress(dict(state="preset_wasserstein_progress", stage="comparison_written",
                      completed_count=1, expected_count=1, run=task["run"]))
        return deepcopy(case.result)

    def control(root):
        state["boundaries"] += 1
        if pause and state["boundaries"] == 2:
            raise worker._JobControlBoundary("paused")

    def forbidden_resume(*args, **kwargs):
        pytest.fail("A requested Comparison time must never reuse a production frame prefix")

    monkeypatch.setattr(worker, "JobQueue", Shared)
    monkeypatch.setattr(worker_io, "WorkerRunIO", Runtime)
    monkeypatch.setattr(worker, "_identity", lambda machine: deepcopy(owner))
    monkeypatch.setattr(worker, "_job_control_boundary", control)
    monkeypatch.setattr(checkpoint, "resume_source_for_claim", forbidden_resume)
    monkeypatch.setattr(preset_wasserstein_runner, "run_preset_wasserstein", execute)
    from darsia_cache import adapters
    from darsia import _opencv_runtime
    monkeypatch.setattr(adapters, "bind_actual_worker_owner", lambda *args, **kwargs: None)
    monkeypatch.setattr(_opencv_runtime, "scientific_thread", lambda *args: {"test_boundary": True})
    monkeypatch.setattr(_opencv_runtime, "require_ipp_off", lambda *args: {"test_boundary": True})
    worker._run_tasks(str(tmp_path), case.job, case.task["commit"], "fixture-cpu", "cpu",
        SimpleNamespace(put=events.append), SimpleNamespace(is_set=lambda: False),
        once=True, exit_when_idle=True, stores=SimpleNamespace(machine_id="fixture"))
    assert calls[:3] == ["original-runner-dispatched", "published-progress", "input-context-closed"]
    if pause:
        assert calls[3:] == ["released"]
        assert [row["kind"] for row in events] == ["claimed", "progress", "paused"]
    else:
        assert calls[3:] == ["complete", "capacity-closed", "released"]
        assert [row["kind"] for row in events] == ["claimed", "progress", "completed"]


@pytest.mark.parametrize("route", ["measured", "first", "bounded", "other"])
def test_claim_capacity_refuses_only_measured_preset_before_estimate_or_reservation(monkeypatch, route):
    """UNRUN outer-dispatch coverage; qualification and reservation remain doubles."""
    from darsia.fluidflower import preset_wasserstein_capacity as preset
    from darsia.fluidflower.pipeline import capacity
    from darsia_cache.preset_qualification import PresetQualification

    task = dict(pipeline="n1" if route == "other" else "preset-wasserstein",
                run="whole-time", attempt_id="attempt")
    first = object.__new__(PresetQualification) if route == "first" else None
    profile = ("darsia-preset-wasserstein-bounded-profile-v1" if route == "bounded"
               else "darsia-cache-import-profile-v1")
    runtime = SimpleNamespace(stores=object(), job=dict(series_id="series"),
        availability=lambda task: (), capacity_refusals={},
        cache_adapter=SimpleNamespace(lease=SimpleNamespace(first_qualification=first,
                                                           profile=dict(format=profile))))
    calls = []
    estimate = object()

    def first_estimate(*args):
        calls.append("first-entitlement")
        return estimate

    def ordinary_estimate(*args, **kwargs):
        calls.append("ordinary-estimate")
        return estimate

    class ReachedReservationBoundary(Exception):
        pass

    def prepare(actual_task, actual_estimate):
        assert actual_task is task and actual_estimate is estimate
        calls.append("reservation-boundary")
        raise ReachedReservationBoundary

    monkeypatch.setattr(capacity, "configuration_for_task", lambda *args: {})
    monkeypatch.setattr(preset, "first_entitlement", first_estimate)
    monkeypatch.setattr(capacity, "load_estimate", ordinary_estimate)
    admission = capacity.ClaimCapacity(runtime,
        dict(machine="fixture-cpu", pid=123, process_create_time="1.0"))
    admission.cache = SimpleNamespace(unlock=lambda: None, prepare=prepare)
    if route == "measured":
        with pytest.raises(capacity.RunUnavailable) as error:
            admission.availability(task)
        assert str(error.value) == preset.MEASURED_UNAVAILABLE
        assert runtime.capacity_refusals == {task["run"]: preset.MEASURED_UNAVAILABLE}
        assert calls == []
    else:
        with pytest.raises(ReachedReservationBoundary):
            admission.availability(task)
        assert calls == ["first-entitlement" if route == "first" else "ordinary-estimate",
                         "reservation-boundary"]


@pytest.mark.parametrize("route", ["measured", "first", "bounded", "other"])
def test_worker_replay_refuses_only_measured_preset_before_binding(monkeypatch, route):
    """UNRUN prepare dispatch only; input replay and first proof are explicit doubles."""
    from darsia.fluidflower import preset_wasserstein_capacity as preset
    from darsia.fluidflower import preset_wasserstein_protocol as protocol
    from darsia_cache.preset_qualification import PresetQualification

    task = dict(pipeline="n1" if route == "other" else "preset-wasserstein")
    configuration, calls = {}, []
    first = object.__new__(PresetQualification) if route == "first" else None
    if first is not None:
        first.configuration = configuration
    profile = ("darsia-preset-wasserstein-bounded-profile-v1" if route == "bounded"
               else "darsia-cache-import-profile-v1")
    lease = SimpleNamespace(first_qualification=first, profile=dict(format=profile))
    runtime = object.__new__(worker_io.WorkerRunIO)
    runtime.job, runtime.commit, runtime.bindings = {}, "a" * 40, {}
    runtime.cache_adapter = SimpleNamespace(lease=lease)
    claim = SimpleNamespace(task=task, input_rows=(), local_output=object(),
                            output=object(), token="attempt")
    binding = SimpleNamespace(configuration=configuration)
    monkeypatch.setattr(worker_io, "verify_inputs", lambda *args: ())
    monkeypatch.setattr(worker_io.RunIOBinding, "replay", staticmethod(lambda *args, **kwargs: binding))
    monkeypatch.setattr(protocol, "validate_preset_wasserstein_claim",
                        lambda *args, **kwargs: calls.append("typed-claim"))
    monkeypatch.setattr(PresetQualification, "require_task", lambda *args: calls.append("first-task"))
    monkeypatch.setattr(PresetQualification, "require_output", lambda *args: calls.append("first-output"))
    if route == "measured":
        with pytest.raises(worker_io.InputHashMismatch) as error:
            runtime.prepare(claim)
        assert str(error.value) == "Input admission failed before science: " + preset.MEASURED_UNAVAILABLE
        assert runtime.bindings == {}
    else:
        assert runtime.prepare(claim) is binding
        assert runtime.bindings == {claim.token: binding}
    assert calls == ([] if route == "other" else
                     ["typed-claim", "first-task", "first-output"] if route == "first" else
                     ["typed-claim"])
