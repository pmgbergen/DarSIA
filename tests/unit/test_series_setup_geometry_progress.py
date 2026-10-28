"""Operation progress precedes original saved-source hashing and replay."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import corrected_setup as original
from darsia.fluidflower.pipeline import run_io
from darsia.fluidflower.pipeline import series_setup_operations as operations
from darsia.fluidflower.pipeline import series_setup_run_io as setup_io
from darsia.fluidflower.pipeline import worker
from darsia.fluidflower.pipeline.job_queue import JobQueue, LeaseLost, read_json
from darsia.fluidflower import job_control
from darsia.fluidflower.adoption_check_progress import observe_check_progress, report_check_progress
from tests.server_lease_clock_fixture import server_lease_clock  # noqa: F401
from .test_series_setup_operations import applied_plan  # noqa: F401
from .test_painted_setup_apply import named_setup  # noqa: F401
from .test_painted_adoption_table import imported, synthetic_payload_admission  # noqa: F401


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("changed_source", [False, True])
def test_build_reports_selected_run_before_saved_geometry_reads(
        tmp_path, monkeypatch, managed, changed_source):
    """Build retains its initial selected-run proof before the original reader."""
    operation = "build"
    runs = ["second", "first"]
    descriptors = {}
    for run in runs:
        path = tmp_path / (run + ".bin")
        path.write_bytes((run + " saved geometry").encode())
        descriptors[run] = original._descriptor(path)
    if changed_source:
        Path(descriptors["second"]["path"]).write_bytes(b"changed saved geometry")
    frame = dict(display_reference_run="first", runs={
        run: dict(configuration=dict(path=str(tmp_path / (run + ".toml")))) for run in runs})
    revision = dict(configuration=dict(frame_binding=frame))
    plan = dict(runs=runs, display_reference_run="first", control_root=str(tmp_path / "controls"),
                build_plans=[dict(run_id=run) for run in runs])
    config = dict(run="second", raw_build_request={})
    project = SimpleNamespace(root=tmp_path)
    guarded_runs = [config["run"]]
    controls_before = deepcopy((config, plan, revision))
    progress, reads, scopes = [], [], []

    def expected(run):
        return dict(state="setup_progress", operation=operation, run=run,
                    step="verify_saved_geometry", run_index=guarded_runs.index(run) + 1,
                    run_count=len(guarded_runs))

    descriptor = original._descriptor

    def observed_hash(path):
        run = Path(path).stem
        assert {key: value for key, value in progress[-1].items()
                if key not in {"message", "check_progress_sequence"}} == expected(run)
        if managed:
            assert scopes == ["entered"]
        reads.append(run)
        return descriptor(path)

    def registered(root, current, configuration, run):
        assert progress[-1] == expected(run)
        assert (root, current, configuration) == (project.root, revision, revision["configuration"])
        # A registered reader can hash large sources before returning its guards.
        original._verified(descriptors[run])
        return dict(guarded_sources=[descriptors[run]], files=dict(labels=descriptors[run]))

    @contextmanager
    def original_scope(actual_project, actual_revision):
        assert (actual_project, actual_revision) == (project, revision)
        scopes.append("entered")
        try:
            yield
        finally:
            scopes.append("exited")

    class ApplyBoundaryReached(RuntimeError):
        """Stop before the first export or Apply write in this progress fixture."""

    def stop_build(*args, **kwargs):
        raise ApplyBoundaryReached("Initial saved-geometry checks finished")

    def execute():
        return operations.run_build(config, plan, revision, project, {}, progress.append)

    monkeypatch.setattr(operations, "_sink", lambda: None)
    monkeypatch.setattr(run_io, "current_binding", lambda: object() if managed else None)
    monkeypatch.setattr(setup_io, "_original_scope", original_scope)
    monkeypatch.setattr(original, "_registered_binding", registered)
    monkeypatch.setattr(original, "_descriptor", observed_hash)
    monkeypatch.setattr("darsia.fluidflower.rig_raw_build.read_build_request", stop_build)

    if changed_source:
        with pytest.raises(ValueError, match="source identity mismatch"):
            execute()
        assert [event for event in progress if "message" not in event] == [expected("second")]
        assert any(event.get("message") == "Verifying saved source: " + Path(descriptors["second"]["path"]).name
                   for event in progress)
        assert reads == ["second"]
    else:
        with pytest.raises(ApplyBoundaryReached):
            execute()
        assert [event for event in progress if "message" not in event] == [
            expected(run) for run in guarded_runs]
        assert reads == [run for run in guarded_runs for _ in range(3)]
    assert scopes == (["entered", "exited"] if managed else [])
    assert (config, plan, revision) == controls_before


@pytest.mark.parametrize("managed", [False, True])
def test_apply_enters_original_selected_run_without_upfront_cohort_proof(
        tmp_path, monkeypatch, managed):
    """Apply starts the reference's original transaction before other runs."""
    runs = ["second", "first"]
    frame = dict(display_reference_run="first", runs={
        run: dict(configuration=dict(path=str(tmp_path / (run + ".toml")))) for run in runs})
    revision = dict(configuration=dict(frame_binding=frame))
    plan = dict(runs=runs, display_reference_run="first",
                build_plans=[dict(run_id=run) for run in runs])
    events, entered = [], []
    plan["control_root"] = str(tmp_path / "controls")

    class OriginalApplyEntered(RuntimeError):
        """Stop at the original selected-run entry point."""

    def cohort_proof(*args, **kwargs):
        raise AssertionError("Apply must retain the original export's verified binding")

    def original_entry(*args, **kwargs):
        reference = kwargs["reference"] if "reference" in kwargs else args[2]
        entered.append(reference.run_id)
        raise OriginalApplyEntered

    monkeypatch.setattr(operations, "_sink", lambda: None)
    monkeypatch.setattr(run_io, "current_binding", lambda: object() if managed else None)
    monkeypatch.setattr(operations, "_geometry_sources", cohort_proof)
    monkeypatch.setattr(setup_io, "original_geometry_sources", cohort_proof)
    monkeypatch.setattr(original, "export_corrected_setup", original_entry)
    monkeypatch.setattr(setup_io, "original_apply_run", original_entry)
    with pytest.raises(OriginalApplyEntered):
        operations.run_apply({}, plan, revision, SimpleNamespace(root=tmp_path), events.append)
    assert entered == ["first"]
    assert events == [dict(state="setup_progress", operation="apply", run="first", step="apply")]


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
@pytest.mark.parametrize("stop_at", [None, ("capture_configuration_sources", 1),
                                   ("reverify_saved_geometry", 1), ("reverify_saved_geometry", 2)])
def test_apply_refreshes_real_lease_and_obeys_stop_at_capture_and_final_reverify(
        applied_plan, tmp_path, monkeypatch, server_lease_clock, stop_at):
    """Actual operation callbacks keep one lease over many simulated lease periods."""
    case = applied_plan
    monkeypatch.setattr(operations, "monotonic", lambda: server_lease_clock.clock.now_ns() / 1e9)
    queue = JobQueue(case.root / "progress-queue", server_clock=server_lease_clock.clock)
    queue.enqueue(dict(run="apply-all", commit="1" * 40, calibration_sha256="2" * 64,
                       configuration=dict(path=r"\\synthetic-server\series\controls.json", sha256="3" * 64),
                       precision="float32", backend="cpu"))
    claim = queue.claim("fixture-machine", "cpu", dict(pid=123, process_create_time="1.0"))
    server_lease_clock.stamp_progress(claim)
    events, occurrences = [], {}
    monkeypatch.setattr(job_control, "read_control", lambda root: dict(action="run", sequence=0))
    require_export = original.require_corrected_export_for_build

    def capture_check(paths):
        report_check_progress("Verifying configuration source bindings…")
        return require_export(paths)

    monkeypatch.setattr(original, "require_corrected_export_for_build", capture_check)

    def progress(event):
        # Each real per-run event follows 500 seconds of simulated source I/O.
        server_lease_clock.advance(500)
        queue.progress(claim, event)
        for parent in (claim.output, claim.lease):
            os.utime(parent / f"progress_{len(events):08d}.json",
                     ns=(server_lease_clock.clock.now_ns(),) * 2)
        assert queue.reclaim_stale() == []
        events.append(deepcopy(event))
        if "message" not in event:
            key = event["step"]
            occurrences[key] = occurrences.get(key, 0) + 1
            if stop_at == (key, occurrences[key]):
                monkeypatch.setattr(job_control, "read_control",
                                    lambda root: dict(action="stop", sequence=1))
        worker._job_control_boundary(queue.root)

    with case.sink.activate():
        if stop_at is None:
            result = operations.run_apply(case.config, case.plan, case.revision, case.data.project, progress)
            coordinator = operations._read(result["coordinator_completion"])
            assert [row["run_id"] for row in coordinator["runs"]] == case.plan["runs"]
            assert [row["run_id"] for row in result["children"]] == case.plan["runs"]
        else:
            with pytest.raises(worker._JobControlBoundary) as stopped:
                operations.run_apply(case.config, case.plan, case.revision, case.data.project, progress)
            assert stopped.value.state == "control_stopped"
    expected = []
    for step in ("apply", "capture_configuration_sources", "reverify_saved_geometry", "prepare_build_requests"):
        selected = list(reversed(case.plan["runs"])) if step == "apply" else case.plan["runs"]
        for index, run in enumerate(selected, start=1):
            event = dict(state="setup_progress", operation="apply", run=run, step=step)
            if step != "apply":
                event.update(run_index=index, run_count=len(selected))
            expected.append(event)
    entered = [event for event in events if "message" not in event]
    assert entered == expected[:len(entered)]
    if stop_at is None:
        assert entered == expected
    else:
        assert (entered[-1]["step"], occurrences[entered[-1]["step"]]) == stop_at
    nested = [event for event in events if "message" in event]
    assert [event["check_progress_sequence"] for event in nested] == list(range(1, len(nested) + 1))
    if stop_at is None:
        assert any(event["step"] == "capture_configuration_sources"
                   and event["message"] == "Verifying configuration source bindings…" for event in nested)
    shared = sorted(claim.lease.glob("progress_*.json"))
    assert [read_json(path)["event"] for path in shared] == events
    queue.assert_owner(claim)


def test_build_reports_initial_and_final_guard_checks_as_build(tmp_path, monkeypatch):
    """Forward the callback through original RunIO and preserve the final guard hashes."""
    from darsia.fluidflower import rig_raw_build

    source = tmp_path / "saved-geometry.bin"
    source.write_bytes(b"saved geometry")
    bound = original._descriptor(source)
    certificate = tmp_path / rig_raw_build.CERTIFICATE_NAME
    certificate.write_text(json.dumps(dict(outputs=[], certificate_scope="synthetic metadata",
        complete_rig_model=False, production_qualification_claimed=False)), encoding="utf-8")
    config = dict(run="sample", raw_build_request=dict(path="synthetic-request"), configuration_files=[])
    revision, project = dict(configuration={}), SimpleNamespace(root=tmp_path)
    request = dict(output=dict(destination=str(tmp_path)))
    events, requests, hashes, builds = [], [], [], []
    descriptor = original._descriptor

    def observed_hash(path):
        hashes.append(events[-1]["step"])
        assert events[-1]["operation"] == "build"
        return descriptor(path)

    def read_request(binding, **kwargs):
        requests.append(deepcopy(binding))
        return deepcopy(request), []

    def build(*args):
        builds.append(events[-1])

    monkeypatch.setattr(operations, "_sink", lambda: None)
    monkeypatch.setattr(run_io, "current_binding", lambda: object())
    monkeypatch.setattr(setup_io, "_original_scope", lambda *args: nullcontext())
    monkeypatch.setattr(setup_io, "original_build", build)
    monkeypatch.setattr(original, "_registered_binding",
                        lambda *args: dict(guarded_sources=[bound], files={}))
    monkeypatch.setattr(original, "_descriptor", observed_hash)
    monkeypatch.setattr(rig_raw_build, "read_build_request", read_request)
    result = operations.run_build(config, {}, revision, project, request, events.append)
    assert [event for event in events if "message" not in event] == [
        dict(state="setup_progress", operation="build", run="sample", step="verify_saved_geometry",
             run_index=1, run_count=1),
        dict(state="setup_progress", operation="build", run="sample", step="build"),
        dict(state="setup_progress", operation="build", run="sample", step="reverify_saved_geometry",
             run_index=1, run_count=1)]
    assert hashes == ["verify_saved_geometry", "reverify_saved_geometry"]
    assert requests == [config["raw_build_request"]] * 2
    assert builds == [dict(state="setup_progress", operation="build", run="sample", step="build")]
    assert result["guarded_geometry"][0]["guarded_sources"] == [bound]
    assert source.read_bytes() == b"saved geometry"


@pytest.mark.parametrize("failure", [LeaseLost("original lease lost"), worker._JobControlBoundary("control_stopped")])
def test_strict_check_observer_propagates_original_failure_and_restores_default(failure):
    """Opted-in worker errors escape; the existing diagnostic mode still ignores them."""
    messages = []

    def failed_observer(message):
        raise failure

    with observe_check_progress(messages.append):
        with pytest.raises(type(failure)) as stopped:
            with observe_check_progress(failed_observer, strict=True):
                report_check_progress("entered original check")
        assert stopped.value is failure
        report_check_progress("restored observer")
        with observe_check_progress(failed_observer):
            report_check_progress("diagnostic display unavailable")
        report_check_progress("diagnostic observer restored")
    assert messages == ["restored observer", "diagnostic observer restored"]


def test_source_hash_progress_is_throttled_without_delaying_stage_notifications(monkeypatch):
    """All nested checks are sampled; stage transitions remain immediate."""
    now = [0.0]
    monkeypatch.setattr(operations, "monotonic", lambda: now[0])
    events = []
    def slow_shared_write(event):
        events.append(event)
        if event.get("message", "").startswith("Verifying saved source: "):
            now[0] += 4.0
    progress = operations._operation_progress(slow_shared_write)
    stage = dict(state="setup_progress", operation="apply", step="verify_saved_geometry")
    first = dict(stage, message="Verifying saved source: source-a.bin")
    second = dict(stage, message="Verifying saved source: source-b.bin")
    check = dict(stage, message="Verifying registration bindings…")
    progress(first)
    now[0] = 5.0
    progress(second)
    progress(stage)
    progress(check)
    assert events == [dict(first, check_progress_sequence=1), stage]
    now[0] = 14.0
    progress(second)
    now[0] = 23.0
    progress(first)
    now[0] = 28.0
    progress(first)
    assert events == [dict(first, check_progress_sequence=1), stage,
                      dict(second, check_progress_sequence=2), dict(first, check_progress_sequence=3)]
    separate = []
    operations._operation_progress(separate.append)(first)
    assert separate == [dict(first, check_progress_sequence=1)]


@pytest.mark.parametrize("outcome", ["stop", "lease-lost"])
def test_throttled_check_keeps_actual_queue_ownership_and_stop_checks(
        tmp_path, monkeypatch, server_lease_clock, outcome):
    queue = JobQueue(tmp_path / "queue", server_clock=server_lease_clock.clock)
    queue.enqueue(dict(run="sample", commit="1" * 40, calibration_sha256="2" * 64,
        configuration=dict(path=r"\\synthetic-server\series\controls.json", sha256="3" * 64),
        precision="float32", backend="cpu"))
    claim = queue.claim("fixture-machine", "cpu", dict(pid=123, process_create_time="1.0"))
    now, events = [0.0], []
    monkeypatch.setattr(operations, "monotonic", lambda: now[0])
    monkeypatch.setattr(job_control, "read_control", lambda root: dict(action="run", sequence=0))
    def publish(event):
        queue.progress(claim, event)
        events.append(event)
        server_lease_clock.stamp_progress(claim)
    def checkpoint():
        queue.assert_owner(claim)
        worker._job_control_boundary(queue.root)
    publish.checkpoint = checkpoint
    progress = operations._operation_progress(publish)
    event = dict(state="setup_progress", operation="apply", message="Verifying saved geometry source…")
    progress(event)
    if outcome == "stop":
        monkeypatch.setattr(job_control, "read_control", lambda root: dict(action="stop", sequence=1))
        expected = worker._JobControlBoundary
    else:
        server_lease_clock.advance(601)
        assert queue.reclaim_stale()
        expected = LeaseLost
    now[0] = 2.0
    with pytest.raises(expected):
        progress(event)
    assert len(events) == 1
    assert len(list(claim.output.glob("progress_*.json"))) == 1


@pytest.mark.parametrize("scope,managed,operation", [
    ("geometry", False, "apply"), ("geometry", True, "build"), ("reverify", False, "apply")])
@pytest.mark.parametrize("outcome", ["complete", "stop", "lease-lost"])
def test_nested_check_steps_refresh_real_lease_and_preserve_worker_failures(
        tmp_path, monkeypatch, server_lease_clock, scope, managed, operation, outcome):
    """One outer run remains owned across nested checks lasting several lease periods."""
    monkeypatch.setattr(operations, "monotonic", lambda: server_lease_clock.clock.now_ns() / 1e9)
    queue = JobQueue(tmp_path / "queue", server_clock=server_lease_clock.clock)
    queue.enqueue(dict(run="sample", commit="1" * 40, calibration_sha256="2" * 64,
        configuration=dict(path=r"\\synthetic-server\series\controls.json", sha256="3" * 64),
        precision="float32", backend="cpu"))
    claim = queue.claim("fixture-machine", "cpu", dict(pid=123, process_create_time="1.0"))
    server_lease_clock.stamp_progress(claim)
    source = tmp_path / "saved-geometry.bin"
    source.write_bytes(b"saved geometry")
    bound = original._descriptor(source)
    project, revision = SimpleNamespace(root=tmp_path), dict(configuration={})
    events, finished, failures, restored, checks_entered = [], [], [], [], []
    lease_lost = LeaseLost("original nested worker lease lost")
    monkeypatch.setattr(job_control, "read_control", lambda root: dict(action="run", sequence=0))

    def progress(event):
        queue.progress(claim, event)
        for parent in (claim.output, claim.lease):
            os.utime(parent / f"progress_{len(events):08d}.json",
                     ns=(server_lease_clock.clock.now_ns(),) * 2)
        events.append(deepcopy(event))
        if event.get("check_progress_sequence") == 2:
            if outcome == "lease-lost":
                failures.append(lease_lost)
                raise lease_lost
            if outcome == "stop":
                monkeypatch.setattr(job_control, "read_control",
                                    lambda root: dict(action="stop", sequence=1))
        try:
            worker._job_control_boundary(queue.root)
        except worker._JobControlBoundary as failure:
            failures.append(failure)
            raise

    def nested_checks():
        for index in range(3):
            checks_entered.append(index)
            report_check_progress("Verifying saved geometry sources…")
            server_lease_clock.advance(500)
            assert queue.reclaim_stale() == []
        finished.append("original checks complete")

    def registered(*args):
        nested_checks()
        return dict(guarded_sources=[bound], files={})

    verified = original._verified
    descriptor = original._descriptor
    hashes = []

    def observed_hash(path):
        hashes.append(Path(path))
        return descriptor(path)

    def reverified(row):
        nested_checks()
        return verified(row)

    monkeypatch.setattr(setup_io, "_original_scope", lambda *args: nullcontext())
    monkeypatch.setattr(original, "_registered_binding", registered)
    monkeypatch.setattr(original, "_descriptor", observed_hash)
    if scope == "reverify":
        monkeypatch.setattr(original, "_verified", reverified)

    def execute():
        if scope == "reverify":
            return operations._reverify([dict(run_id="sample", guarded_sources=[bound])],
                                        progress, operation=operation)
        read = setup_io.original_geometry_sources if managed else operations._geometry_sources
        return read(project, revision, ["sample"], progress, operation=operation)

    with observe_check_progress(restored.append):
        if outcome == "complete":
            execute()
        else:
            expected = LeaseLost if outcome == "lease-lost" else worker._JobControlBoundary
            with pytest.raises(expected) as stopped:
                execute()
            assert stopped.value is failures[0]
        report_check_progress("after worker checks")
    nested = [event for event in events if "message" in event]
    assert [event["check_progress_sequence"] for event in nested] == list(range(1, len(nested) + 1))
    if outcome == "complete":
        assert checks_entered == [0, 1, 2]
        assert 1 <= sum(event["message"] == "Verifying saved geometry sources…" for event in nested) <= 3
    else:
        assert len(nested) == 2
    assert all(event["operation"] == operation and event["run"] == "sample" for event in nested)
    assert {event["message"] for event in nested} <= {
        "Verifying saved geometry sources…", "Verifying saved geometry source…",
        "Verifying saved source: " + source.name}
    assert finished == (["original checks complete"] if outcome == "complete" else [])
    assert hashes == ([source] if outcome == "complete" else [])
    assert restored == ["after worker checks"]
    shared = sorted(claim.lease.glob("progress_*.json"))
    assert [read_json(path)["event"] for path in shared] == events
    queue.assert_owner(claim)
