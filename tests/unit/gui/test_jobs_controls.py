"""Local Qt command publication only; no worker, executor, SSH or science."""

import json
import socket
from pathlib import Path
import threading
import time

import pytest
from PySide6.QtCore import QTimer

from darsia.fluidflower import job_control
from darsia.fluidflower import jobs_panel as ui
from .test_jobs_metadata import bytes_of, job, put
from .test_jobs_panel import app, wait


def publish_and_wait(app, panel, action=None):
    """Separate producer I/O from the unchanged three-second Qt wait."""
    state = panel._test_publication
    ticks = [time.monotonic()]
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start(20)
    try:
        if action is not None:
            action()
        wait(app, lambda: state["started"].is_set() or panel.can_close())
        if state["started"].is_set():
            remaining = max(0, 15 - (time.monotonic() - state["started_at"]))
            wait(app, state["finished"].is_set, timeout=remaining)
            assert state["finished_at"] - state["started_at"] <= 15
        wait(app, panel.can_close)
    finally:
        timer.stop()
    ticks.append(time.monotonic())
    assert max(b - a for a, b in zip(ticks, ticks[1:])) < 3, (
        "Qt event-loop heartbeat stalled during control publication"
    )
    state["started"].clear()
    state["finished"].clear()


@pytest.fixture
def controls(app, job, request):
    root, _cases, _task = job
    launches = []

    def launcher(job_root, **kwargs):
        launches.append((job_root, kwargs))
        return dict(job_root=job_root, numerical_progress_claimed=False)

    panel = ui.JobsPanel(local_machine="Main", local_launcher=launcher)
    state = dict(
        started=threading.Event(), finished=threading.Event(), io_durations_seconds=[]
    )

    def publisher(*args, **kwargs):
        state["started_at"] = time.monotonic()
        state["started"].set()
        try:
            return job_control.publish_control(*args, **kwargs)
        finally:
            state["finished_at"] = time.monotonic()
            state["io_durations_seconds"].append(
                state["finished_at"] - state["started_at"]
            )
            state["finished"].set()

    panel._test_publication = state
    panel.control_publisher = publisher
    panel.set_context(jobs_provider=lambda: [dict(root=str(root), active=True)])
    wait(app, panel.can_close)
    yield panel, launches, root
    try:
        wait(app, panel.can_close)
        panel.close()
        panel.deleteLater()
        app.processEvents()
    finally:
        print("N27_CONTROL_PRODUCER_IO " + json.dumps(dict(
            nodeid=request.node.nodeid, host=socket.gethostname(), test_root=str(root),
            publisher_calls=len(state["io_durations_seconds"]),
            io_durations_seconds=state["io_durations_seconds"],
        ), sort_keys=True), flush=True)


def test_open_refresh_only_read_bound_master_and_displayed_chain(controls, app):
    panel, launches, root = controls
    before = bytes_of(root)
    panel.refresh()
    wait(app, panel.can_close)
    assert bytes_of(root) == before and not launches
    assert panel.current["control"]["state"] == {"action": "run", "sequence": 0}
    assert panel.pause_button.isEnabled() and panel.stop_button.isEnabled()
    assert not panel.resume_button.isEnabled()
    assert "Recorded master: Main" in panel.control_state_label.text()
    assert "sequence 0" in panel.control_state_label.text()
    assert not (root / "ui" / "commands").exists()


def test_pause_and_stop_append_requested_states_without_task_or_lease_changes(controls, app):
    panel, launches, root = controls
    original = bytes_of(root)
    publish_and_wait(app, panel, panel.pause_job)
    first = job_control.read_control(root)
    first_bytes = Path(first["command"]["path"]).read_bytes()
    assert first["action"] == "pause" and first["sequence"] == 1
    assert first["previous"] is None
    assert panel.current["state"] == "Paused" and panel.current["committed"] == 0
    assert panel.resume_button.isEnabled() and not panel.start_button.isEnabled()
    assert first["command"]["sha256"] in panel.control_state_label.text()
    assert "exit is not inferred" in panel.control_state_label.text()
    publish_and_wait(app, panel, panel.stop_job)
    last = job_control.read_control(root)
    assert last["action"] == "stop" and last["previous"] == first["command"]
    assert Path(first["command"]["path"]).read_bytes() == first_bytes
    assert all((root / path).read_bytes() == raw for path, raw in original.items())
    assert not launches and not (root / "failed").exists()


@pytest.mark.parametrize("action", ["pause", "stop"])
def test_passive_refresh_cannot_block_control_publication(controls, app, action):
    panel, launches, root = controls
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    original_reader = panel.reader
    def delayed(row):
        entered.set()
        try:
            assert release.wait(10)
            value = original_reader(row)
            value["name"] = "Obsolete display must not be installed"
            return value
        finally:
            returned.set()
    panel._show_control_state()
    restart_without_read = panel.restart_button.isEnabled()
    panel.reader = delayed
    try:
        panel.refresh()
        wait(app, entered.is_set)
        assert panel._read_thread is not None and not panel.can_close()
        panel._show_control_state()
        assert panel.stop_button.isEnabled() and panel.pause_button.isEnabled()
        # A passive read does not change which commands are offered.
        assert panel.restart_button.isEnabled() == restart_without_read
        # The normal refresh after publication must use the available reader.
        panel.reader = original_reader
        publish_and_wait(app, panel, panel.stop_job if action == "stop" else panel.pause_job)
        assert job_control.read_control(root)["action"] == action
        assert not release.is_set() and not launches
        assert panel.current["name"] != "Obsolete display must not be installed"
        release.set()
        wait(app, returned.is_set)
        app.processEvents()
        assert panel.current["name"] != "Obsolete display must not be installed"
    finally:
        release.set()


def test_stop_does_not_interrupt_an_existing_control_publication(controls, app):
    panel, _, root = controls
    entered, release = threading.Event(), threading.Event()
    original_publisher = panel.control_publisher
    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original_publisher(*args, **kwargs)
    panel.control_publisher = delayed
    try:
        panel.pause_job()
        wait(app, entered.is_set)
        panel.stop_job()
        assert panel._control_pending and not panel.stop_button.isEnabled()
        assert "control publication" in panel.error_text.toPlainText()
        assert job_control.read_control(root)["sequence"] == 0
    finally:
        release.set()
        publish_and_wait(app, panel)
    assert job_control.read_control(root)["action"] == "pause"
    assert job_control.read_control(root)["sequence"] == 1


def test_resume_publishes_run_then_uses_normal_selected_service_and_original_lease(controls, app):
    panel, launches, root = controls
    publish_and_wait(app, panel, panel.pause_job)
    pause = job_control.read_control(root)["command"]
    immutable_tasks = {
        path: raw for path, raw in bytes_of(root).items()
        if path.startswith(("queue/", "results/", "leases/", "claimed/"))
    }
    panel.backend_combo.setCurrentIndex(panel.backend_combo.findData("cuda"))
    panel.workers_spin.setValue(3)
    publish_and_wait(app, panel, panel.resume_job)
    current = job_control.read_control(root)
    assert current["action"] == "run" and current["previous"] == pause
    assert len(launches) == 1
    assert launches[0][0] == str(root)
    assert {
        key: value for key, value in launches[0][1].items()
        if key != "launch_directory"
    } == dict(backend="cuda", workers=3, machine="Main")
    assert ui.CONTROL_LEASE in panel.control_state_label.text()
    assert "600 s actual-progress lease" in panel.control_state_label.text()
    assert all((root / path).read_bytes() == raw for path, raw in immutable_tasks.items())
    assert panel.current["committed"] == 0
    assert "unknown" in panel.launch_state_label.text()
    assert not panel.start_button.isEnabled()  # the original reservation still applies


def test_stale_window_supplies_its_displayed_previous_not_a_fresh_retry(controls, app):
    first, _launches, root = controls
    second = ui.JobsPanel(local_machine="Main")
    errors = []
    second.error_reported.connect(errors.append)
    second.set_context(jobs_provider=lambda: [dict(root=str(root))])
    wait(app, second.can_close)
    try:
        assert second.current["control"]["state"]["sequence"] == 0
        publish_and_wait(app, first, first.pause_job)
        requested = job_control.read_control(root)["command"]
        second.stop_job()
        wait(app, second.can_close)
        assert errors and "refresh before applying" in errors[-1]
        assert job_control.read_control(root)["command"] == requested
        assert len(list((root / "ui" / "commands").glob("*.json"))) == 1
    finally:
        wait(app, second.can_close)
        second.close()
        second.deleteLater()
        app.processEvents()


def test_non_master_and_legacy_without_master_never_publish(controls, app):
    panel, _launches, root = controls
    before = bytes_of(root)
    panel.local_machine = "Second"
    panel._target_changed()
    assert not any(
        button.isEnabled()
        for button in (panel.pause_button, panel.stop_button, panel.resume_button)
    )
    panel.pause_job()
    assert "recorded master machine" in panel.error_text.toPlainText()
    assert bytes_of(root) == before
    panel.local_machine = "Main"
    legacy = root.parent / "legacy"
    put(legacy / "job.json", dict(expected_runs=["ac52"]))
    panel.reader = lambda row: dict(
        root=row["root"], name="Legacy", state="Waiting", job_facts={}, errors=[]
    )
    panel.set_context(jobs_provider=lambda: [dict(root=str(legacy))])
    wait(app, panel.can_close)
    legacy_before = bytes_of(legacy)
    panel.stop_job()
    assert bytes_of(legacy) == legacy_before
    assert "recorded master" in panel.control_state_label.text()


def test_control_read_failure_blocks_direct_start_and_command_before_reservation(controls, app):
    panel, launches, root = controls
    publish_and_wait(app, panel, panel.pause_job)
    command = Path(job_control.read_control(root)["command"]["path"])
    document = json.loads(command.read_bytes())
    document["machine"] = "foreign"
    command.write_bytes(json.dumps(document).encode())
    before = bytes_of(root)
    panel.refresh()
    wait(app, panel.can_close)
    assert panel.current["control"]["error"] is True
    panel.start_local_workers()
    panel.pause_job()
    assert bytes_of(root) == before and not launches
    assert "identity or chain" in panel.error_text.toPlainText()
    assert not (root / "ui" / "launches").exists()


def test_pending_publication_holds_owner_and_navigation_cannot_launch_for_new_context(controls, app):
    panel, launches, root = controls
    publish_and_wait(app, panel, panel.pause_job)
    entered, release = threading.Event(), threading.Event()
    publisher = panel.control_publisher

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return publisher(*args, **kwargs)

    panel.control_publisher = delayed
    panel.resume_job()
    wait(app, entered.is_set)
    assert not panel.can_close()
    responsive = []
    QTimer.singleShot(0, lambda: responsive.append(
        not release.is_set() and not panel.can_close()
    ))
    wait(app, lambda: bool(responsive))
    assert responsive == [True]
    panel.set_context(jobs_provider=lambda: [])
    release.set()
    publish_and_wait(app, panel)
    assert job_control.read_control(root)["action"] == "run"
    assert panel.current is None and not launches


def test_repeated_start_cannot_bypass_pause_or_existing_reservation(controls, app):
    panel, launches, root = controls
    publish_and_wait(app, panel, panel.pause_job)
    before = bytes_of(root)
    panel.start_selected_machine()
    assert bytes_of(root) == before and not launches
    publish_and_wait(app, panel, panel.resume_job)
    assert len(launches) == 1
    panel.start_selected_machine()
    assert len(launches) == 1


def test_a_refusal_while_a_publication_finishes_is_cleared_by_the_next_read(controls, app):
    panel, _, root = controls
    cleared = []
    panel.error_cleared.connect(cleared.append)
    entered, release = threading.Event(), threading.Event()
    original_publisher = panel.control_publisher

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original_publisher(*args, **kwargs)

    panel.control_publisher = delayed
    try:
        panel.pause_job()
        wait(app, entered.is_set)
        panel.stop_job()
        refusal = panel.error_text.toPlainText()
        assert "control publication" in refusal
    finally:
        release.set()
        publish_and_wait(app, panel)
    panel.refresh()
    wait(app, panel.can_close)
    assert panel.error_text.toPlainText() == "" and cleared == [refusal]
    # A real error is not withdrawn by a read.
    panel._error("The shared job folder could not be read.")
    panel.refresh()
    wait(app, panel.can_close)
    assert cleared == [refusal]
