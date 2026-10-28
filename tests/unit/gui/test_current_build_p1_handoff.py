"""Exercise the new completed-Build action and original Start/refit signal."""
from pathlib import Path

from darsia.fluidflower.pipeline import current_build_job
from .test_phase_one_refit_handoff import application, launch_fixture
from darsia.fluidflower.jobs_panel import JobsPanel


def test_completed_build_prepares_without_picker_then_original_start_arms_refit(launch_fixture, monkeypatch):
    f = launch_fixture
    panel = f.panel
    panel.series_id = "test"
    panel.current = dict(root="completed-build", complete=True,
                         job_facts={"pipeline": "series-setup"})
    calls, delivered = [], []
    registration = dict(root=f.phase_root, ui_series_root=f.series, ui_series_id="test", name="Current Build P1")
    def prepare(snapshot, **kwargs):
        calls.append((snapshot, kwargs))
        return registration
    monkeypatch.setattr(current_build_job, "prepare_current_build_job", prepare)
    monkeypatch.setattr(panel, "_start_call", lambda call, done, closed: (delivered.append((call, done)) or (None, None)))
    panel.prepare_build_p1()
    assert len(delivered) == 1 and not f.events
    call, done = delivered[0]
    result = call()
    assert calls[0][0]["root"] == "completed-build"
    assert calls[0][1]["series_root"] == f.series
    done(result)
    panel.current = dict(root=f.phase_root, errors=[], job_facts=dict(pipeline="native-phase-statistics",
        execution_mode="current_build", expected_runs=f.order, target_machine="Main"))
    panel.start_local_workers()
    assert [row[0] for row in f.events] == ["launcher", "refit"]
    assert f.events[1] == ("refit", str(Path(f.phase_root).resolve()), f.order)


def test_parallel_default_is_shown_and_preserves_a_later_manual_choice(launch_fixture, monkeypatch):
    f = launch_fixture
    panel = f.panel
    panel.current = dict(root=f.phase_root, errors=[], job_facts=dict(pipeline="native-phase-statistics",
        execution_mode="current_build", expected_runs=f.order, target_machine="Main"))
    monkeypatch.setattr(panel, "_target_changed", lambda *args: None)
    original = panel.workers_spin.value()
    result = dict(available=True, recommended_workers=6,
        message="6 parallel run workers; 20 GiB each plus 40 GiB reserve; rechecked at Start")
    panel._capacity_ready(dict(generation=panel._generation, root=f.phase_root,
                              result=result, selected_workers=original))
    assert panel.workers_spin.value() == 6
    assert "6 parallel" in panel.capacity_label.text()
    panel.workers_spin.setValue(3)
    panel._capacity_ready(dict(generation=panel._generation, root=f.phase_root,
                              result=dict(result, recommended_workers=4), selected_workers=3))
    assert panel.workers_spin.value() == 3
    assert JobsPanel._start_allowed(panel, "Main")


def _prepare_with(panel, monkeypatch, prepare):
    """Run the phase-map preparation call at once; return its result slot."""
    monkeypatch.setattr(current_build_job, "prepare_current_build_job", prepare)
    delivered = []
    monkeypatch.setattr(panel, "_start_call", lambda call, done, closed: (delivered.append((call, done)) or (None, None)))
    panel.prepare_build_p1()
    (call, done), = delivered
    return call, done


def test_production_photographs_and_other_machines_are_explicit_choices(launch_fixture, monkeypatch):
    panel = launch_fixture.panel
    panel.series_id = "test"
    panel.current = dict(root="completed-build", complete=True, job_facts={"pipeline": "series-setup"})
    calls = []
    call, _done = _prepare_with(panel, monkeypatch, lambda snapshot, **kwargs: calls.append(kwargs) or {})
    call()
    assert calls[0]["purpose"] == "calibration" and calls[0]["any_connected_machine"] is False
    panel.phase_map_purpose.setCurrentIndex(panel.phase_map_purpose.findData("production"))
    panel.phase_map_shared.setChecked(True)
    call, _done = _prepare_with(panel, monkeypatch, lambda snapshot, **kwargs: calls.append(kwargs) or {})
    assert panel.progress_label.text() == "Preparing production phase maps from the completed Setup…"
    call()
    assert calls[1]["purpose"] == "production" and calls[1]["any_connected_machine"] is True
    assert calls[1]["on_progress"] == panel.phase_map_progress.emit


def test_progress_counts_photographs_and_a_refusal_is_shown_without_a_traceback(launch_fixture, monkeypatch):
    panel = launch_fixture.panel
    panel.series_id = "test"
    panel.current = dict(root="completed-build", complete=True, job_facts={"pipeline": "series-setup"})
    reason = "This job fits common colour paths between its stages, so it runs on its Build machine only."

    def refuse(snapshot, **kwargs):
        raise ValueError(reason)

    call, done = _prepare_with(panel, monkeypatch, refuse)
    panel._create_thread = object()  # Progress shows only while the preparation runs.
    panel._phase_map_progress(dict(state="raw_identity", completed=12, total=3090))
    panel._create_thread = None
    assert panel.progress_label.text().endswith("reading photograph identities, 12 of 3090…")
    before = len(panel.registrations)
    done(call())
    assert panel.progress_label.text() == "Phase map job not prepared: " + reason
    assert launch_fixture.errors == [reason]  # The plain reason, not a traceback.
    assert len(panel.registrations) == before

