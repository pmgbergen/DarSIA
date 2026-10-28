"""Actual offscreen Qt, injected metadata/producers; no worker or SSH process."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import threading
import time

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QWidget
from shiboken6 import isValid

from .test_jobs_metadata import model, put

from darsia.fluidflower import remote_worker_launch as remote
from darsia.fluidflower import jobs_panel as ui


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait(app, predicate, timeout=30):
    # A slower or loaded test machine needs more than a few seconds for the
    # panel's background reads to retire; the predicate is unchanged.
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        app.processEvents()
        if predicate():
            return
        QTest.qWait(2)
    assert predicate(), "bounded local Qt operation did not retire"


def snapshot(root="local-fixture"):
    return dict(
        root=root,
        name="Tiny",
        series="Local",
        state="Waiting",
        job_facts=dict(precision="float32"),
        master_machine="Main",
        run_count=1,
        completed_runs=0,
        committed=0,
        total_frames=2,
        remaining=2,
        frames_per_minute=None,
        eta_seconds=None,
        workers=[],
        machines=[],
        logs=[],
        errors=[],
        runs=[],
    )


@pytest.fixture
def panel(app):
    calls = []
    widget = ui.JobsPanel(
        reader=lambda row: snapshot(row["root"]),
        local_machine="Main",
        job_creator=lambda *a, **k: calls.append((a, k)),
        local_launcher=lambda *a, **k: calls.append((a, k)),
    )
    yield widget, calls
    wait(app, widget.can_close)
    widget.close()
    widget.deleteLater()
    app.processEvents()
    if isValid(widget):
        QCoreApplication.sendPostedEvents(widget, QEvent.DeferredDelete)
    assert not isValid(widget)


def test_passive_navigation_reads_only_explicit_roots_no_creation_or_launch(panel, app):
    widget, calls = panel
    observed = []
    widget.reader = lambda row: (observed.append(row["root"]) or snapshot(row["root"]))
    widget.set_context(
        series_label="Local",
        jobs_provider=lambda: [dict(root="explicit-root", active=True)],
    )
    wait(app, widget.can_close)
    assert observed == ["explicit-root"] and not calls
    assert widget.current["root"] == "explicit-root"
    assert not widget.create_button.isEnabled()
    assert widget.start_button.isEnabled() and widget.add_machine_button.isEnabled()
    assert all(
        not button.isEnabled()
        for button in [widget.pause_button, widget.resume_button, widget.stop_button]
    )
    assert "recorded master" in widget.pause_button.toolTip()
    assert "Unknown" in widget.progress_label.text()


def test_read_worker_is_freed_by_its_owner_in_the_gui_thread(panel, app):
    """Q185: a call worker is never deleted in its own thread.

    Deleted there, Qt calls the Python QThread's disconnectNotify under a
    signal-slot lock, which needs the GIL. QTest.qWait keeps the GIL while it
    deletes widgets that earlier tests left; it waited for that lock, and the
    focused GUI run hung. The worker's Python owner frees it in the GUI thread.
    """
    widget, _ = panel
    finished = []
    thread, worker = widget._start_call(lambda: None, lambda value: None,
                                         lambda: finished.append(True))
    wait(app, lambda: bool(finished))
    assert isValid(worker), "the call worker was deleted in its own thread"
    watcher = QObject()
    worker.destroyed.connect(watcher.deleteLater)
    del worker  # The last owner lets go here, in the GUI thread.
    QCoreApplication.sendPostedEvents(watcher, QEvent.DeferredDelete)
    assert not isValid(watcher)


def test_close_cancels_blocked_metadata_read_without_wait_or_late_result(app):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    def reader(row):
        entered.set()
        try:
            assert release.wait(5)
            return snapshot(row["root"])
        finally:
            returned.set()
    widget = ui.JobsPanel(reader=reader, local_machine="Main")
    widget.set_context(series_label="Local", jobs_provider=lambda: [dict(root="r")])
    wait(app, entered.is_set)
    assert not widget.can_close()
    widget.prepare_close()
    assert widget.can_close() and not release.is_set()
    assert widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(widget, QEvent.DeferredDelete)
    assert not isValid(widget)
    release.set()
    wait(app, returned.is_set)
    app.processEvents()  # The old read cannot install a snapshot on its deleted display.


@pytest.mark.parametrize("field", ["_create_thread", "_control_thread", "_raw_review_thread"])
def test_prepare_close_does_not_remove_mutation_or_copy_guards(panel, field):
    widget, _ = panel
    active = object()
    generation = widget._generation
    setattr(widget, field, active)
    try:
        widget.prepare_close()
        assert getattr(widget, field) is active and not widget.can_close()
        assert widget._generation == generation
    finally:
        setattr(widget, field, None)


def test_context_switch_discards_old_read_and_does_not_drop_new_registration(
    panel, app
):
    widget, _ = panel
    entered = threading.Event()
    release = threading.Event()
    reads = []

    def reader(row):
        reads.append(row["root"])
        if row["root"] == "old":
            entered.set()
            assert release.wait(2)
        return snapshot(row["root"])

    widget.reader = reader
    widget.set_context(series_label="Old", jobs_provider=lambda: [dict(root="old")])
    wait(app, entered.is_set)
    widget.set_context(
        series_label="New", jobs_provider=lambda: [dict(root="new", active=True)]
    )
    assert not widget.can_close()
    release.set()
    wait(app, widget.can_close)
    assert widget.current["root"] == "new" and reads == ["old", "new"]
    assert widget.context_label.text() == "Jobs — New"


def test_cancel_create_dialog_has_no_producer_side_effect(panel, app, monkeypatch):
    widget, calls = panel
    widget.set_context(
        series_label="Local",
        series_root="series",
        series_id="unit",
        selection_provider=lambda runs: [],
        plans_provider=lambda: [],
    )

    class Cancel:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(ui, "StartAnalysisDialog", Cancel)
    widget.open_create_dialog()
    assert not calls and widget.can_close()


def test_explicit_create_passes_captured_series_b4_context_without_launch(panel, app):
    widget, calls = panel
    provider = lambda runs: []
    registrations = []

    def creator(plan, root, **kwargs):
        calls.append((plan, root, kwargs))
        return dict(
            root=root,
            name="Created",
            series_label=kwargs["series_label"],
            active=True,
            worker_started=False,
        )

    widget.job_creator = creator
    widget.job_created.connect(registrations.append)
    widget.set_context(
        series_label="Local",
        series_root="selected-series",
        series_id="unit",
        selection_provider=provider,
    )
    widget.create_job(dict(path="explicit-plan"), "fresh-job", "Name")
    wait(app, widget.can_close)
    assert len(calls) == 1 and registrations[0]["worker_started"] is False
    assert (
        calls[0][2]["series_root"] == "selected-series"
        and calls[0][2]["selection_provider"] is provider
    )
    assert widget.current["root"] == "fresh-job"


def test_explicit_local_launch_uses_selected_backend_and_count_not_alive_claim(
    panel, app, tmp_path
):
    widget, calls = panel
    launches = []
    put(tmp_path / "job.json", dict(expected_runs=["ac52"]))

    def launcher(root, **kwargs):
        launches.append((root, kwargs))
        return dict(
            job_root=root,
            started=True,
            launcher_pid=123,
            numerical_progress_claimed=False,
        )

    widget.local_launcher = launcher
    widget.set_context(jobs_provider=lambda: [dict(root=str(tmp_path), active=True)])
    wait(app, widget.can_close)
    widget.backend_combo.setCurrentIndex(widget.backend_combo.findData("cuda"))
    widget.workers_spin.setValue(3)
    widget.start_local_workers()
    wait(app, widget.can_close)
    assert len(launches) == 1 and launches[0][0] == str(tmp_path)
    assert {k: v for k, v in launches[0][1].items() if k != "launch_directory"} == dict(
        backend="cuda", workers=3, machine="Main"
    )
    assert (
        Path(launches[0][1]["launch_directory"]).parent == tmp_path / "ui" / "launches"
    )
    assert widget.current["committed"] == 0 and not calls


def test_resource_choice_for_other_machine_cannot_change_local_controls(panel, app):
    widget, _ = panel

    class Resource(QWidget):
        workers_selected = Signal(object)

        def set_context(self, *a, **k):
            self.context = (a, k)

        def can_close(self):
            return True

    resource = Resource()
    widget.set_resource_panel(resource)
    received = []
    widget.workers_selected.connect(received.append)
    resource.workers_selected.emit(dict(machine="Second", backend="cuda", workers=5))
    assert (
        widget.workers_spin.value() == 1 and widget.backend_combo.currentData() == "cpu"
    )
    resource.workers_selected.emit(dict(machine="Main", backend="cuda", workers=2))
    assert (
        widget.workers_spin.value() == 2
        and widget.backend_combo.currentData() == "cuda"
    )
    assert len(received) == 2


def test_reader_traceback_displayed_verbatim_without_producer(panel, app):
    widget, calls = panel

    def reader(row):
        raise ValueError("literal Ω metadata error")

    widget.reader = reader
    widget.set_context(jobs_provider=lambda: [dict(root="explicit")])
    wait(app, widget.can_close)
    assert "ValueError: literal Ω metadata error" in widget.error_text.toPlainText()
    assert not widget.start_button.isEnabled() and not calls
    assert "ValueError: literal Ω metadata error" in widget.start_button.toolTip()
    assert "See Errors" in widget.start_button.toolTip()


def test_start_dialog_chooses_existing_plan_only(app):
    plan = dict(
        path=r"\\server\share\fixed.toml",
        sha256="a" * 64,
        size_bytes=15,
        label="Frozen",
    )
    dialog = ui.StartAnalysisDialog([plan])
    assert not dialog.buttons.button(ui.QDialogButtonBox.StandardButton.Ok).isEnabled()
    dialog.job_edit.setText(r"\\server\share\jobs\new")
    dialog.name_edit.setText("Local-name")
    assert dialog.buttons.button(ui.QDialogButtonBox.StandardButton.Ok).isEnabled()
    assert dialog.choices()["plan"] == plan
    assert "no worker" in dialog.findChildren(ui.QLabel)[0].text()
    dialog.close()
    dialog.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("once", [False, True])
def test_current_worker_command_detached_and_immutable_receipt_only(
    tmp_path, monkeypatch, once
):
    # The detached legacy launch requires an actual bound job document.
    put(tmp_path / "job.json", dict(expected_runs=["ac52"], precision="float32"))
    monkeypatch.setattr(ui, "require_unc", lambda path: None)
    observed = []
    # This test owns only the detached argv/receipt boundary, not real policy.
    from darsia.fluidflower import store_map
    from darsia_cache import adapters
    import os

    selected_map = object()
    metadata_calls = []
    environment_calls = []
    declared_environment = {"TEST_ONLY_LOCAL_WORKER_ENV": "original-default-map"}

    def selected_metadata(path):
        assert path is None
        metadata_calls.append(path)
        return selected_map

    def selected_environment(stores, *, machine, series_id, explicit_stores,
                             parent_environment):
        assert stores is selected_map
        assert machine == "Main" and series_id is None and explicit_stores is False
        assert parent_environment is os.environ
        environment_calls.append(dict(declared_environment))
        return dict(declared_environment)

    monkeypatch.setattr(store_map, "read_store_map", selected_metadata)
    monkeypatch.setattr(adapters, "independent_worker_environment", selected_environment)

    class FakeProcess:
        ProcessChannelMode = ui.QProcess.ProcessChannelMode

        def __init__(self):
            self.values = {}
            observed.append(self)

        def setProcessEnvironment(self, value):
            self.values["environment"] = value

        def setProgram(self, value):
            self.values["program"] = value

        def setArguments(self, value):
            self.values["arguments"] = value

        def setWorkingDirectory(self, value):
            self.values["cwd"] = value

        def setProcessChannelMode(self, value):
            self.values["channels"] = value

        def setStandardOutputFile(self, value):
            self.values["console"] = value

        def startDetached(self):
            return True, 321

        def errorString(self):
            return "not called"

    monkeypatch.setattr(ui, "QProcess", FakeProcess)
    record = ui.launch_local_worker(
        tmp_path,
        backend="cpu",
        workers=2,
        machine="Main",
        python="existing-python",
        cwd="existing-checkout",
        once=once,
    )
    assert metadata_calls == [None]
    assert environment_calls == [declared_environment]
    assert observed[0].values["environment"].keys() == list(declared_environment)
    assert observed[0].values["environment"].value("TEST_ONLY_LOCAL_WORKER_ENV") == "original-default-map"
    # N109: the stdlib pre-start supervisor wraps the unchanged worker command.
    assert observed[0].values["program"] == "existing-python"
    assert observed[0].values["arguments"] == [
        "-B",
        "-m",
        "darsia_service.prestart",
        "--service-log",
        str(tmp_path / "logs" / "main-cpu.log"),
        "--",
        "existing-python",
        "-m",
        "darsia.fluidflower.worker",
        "--job",
        str(tmp_path),
        "--backend",
        "cpu",
        "--workers",
        "2",
        "--machine",
        "Main",
        "--watchdog",
    ] + (["--once"] if once else [])
    assert (
        record["detached"]
        and not record["numerical_progress_claimed"]
        and record["launcher_pid"] == 321
    )
    assert list(tmp_path.rglob("launch.json")) and not (tmp_path / "queue").exists()
    assert not any(p.suffix == ".py" for p in tmp_path.rglob("*"))


def test_machine_registration_records_facts_without_contact_or_overwrite(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ui, "require_unc", lambda path: None)
    put(tmp_path / "job.json", dict(job="fixture"))
    facts = dict(
        machine="Second",
        host="existing-host",
        checkout="existing-repo",
        python="existing-env",
        ssh_alias="existing-alias",
    )
    receipt = ui.register_machine_facts(tmp_path, facts, local_machine="Second")
    value = model.read_json(Path(receipt["path"]))
    assert value["remote_contacted"] is False and value["installed"] is False
    assert value["checkout"] == "existing-repo" and value["job"] == model.descriptor(
        tmp_path / "job.json"
    )
    with pytest.raises(FileExistsError):
        ui.register_machine_facts(tmp_path, facts, local_machine="Second")


@pytest.mark.parametrize("state", ["missing", "error", "ready"])
def test_start_explains_execution_metadata_without_changing_its_gate(panel, app, tmp_path, state):
    """A grey Start names the missing input; valid passive selection stays ready."""
    widget, calls = panel
    data = snapshot(str(tmp_path / "private-job"))
    if state == "missing":
        data["job_facts"] = None
    elif state == "error":
        data["errors"] = ["Traceback (most recent call last):\nValueError: Selected job source binding differs"]
    widget._show_snapshot(data)
    tooltip = widget.start_button.toolTip()
    if state == "ready":
        assert widget.start_button.isEnabled()
        assert tooltip == "Explicit start of the existing whole-run worker; frame publications alone show progress."
    else:
        assert not widget.start_button.isEnabled()
        assert "See Errors" in tooltip
        if state == "missing":
            assert "readable execution metadata" in tooltip
        else:
            assert "ValueError: Selected job source binding differs" in tooltip
            assert widget.error_text.toPlainText() == data["errors"][0]
            assert "Traceback" not in tooltip
        widget.start_button.click()
    assert not calls and widget.can_close()
