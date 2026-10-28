"""Folder participation keeps the GUI responsive and worker role read-only."""

import threading
from copy import deepcopy

import pytest
from PySide6.QtCore import QTimer

from .test_native_jobs_owner import local_registry, snapshot, workspace, put  # noqa: F401


class Folder:
    published = []

    def __init__(self, path):
        self.path = path

    def publish(self, root, **options):
        self.published.append((root, options))


class Tuner:
    note = "Measured capacity: CPU 2, GPU 1"

    def __init__(self, **options):
        pass

    def counts(self):
        return 2, 1

    def observe(self, value):
        pass


class Service:
    def __init__(self, folder, machine, **options):
        self.folder, self.machine, self.options = folder, machine, options
        self.role = options["role"]
        self.left = False

    def set_tuner(self, tuner):
        self.tuner = tuner

    def tick(self):
        if getattr(self, "tuner", None) is not None:
            self.update_capacity(*self.tuner.counts())
        return dict(role=self.role, workers=[dict(machine=self.machine, role=self.role,
            cpu_workers=2, cuda_workers=1, state="online", jobs=[], reported_utc="now")],
            jobs=[], errors=[], tuning_note=getattr(getattr(self, "tuner", None), "note", ""))

    def update_capacity(self, cpu, cuda):
        self.capacity = cpu, cuda

    def leave(self):
        self.left = True
        return {}


@pytest.fixture
def controls(workspace, tmp_path):
    controls = workspace.shared_jobs
    class Settings:
        def setValue(self, *args):
            pass
    controls._settings = Settings()
    controls.folder.setText(str(tmp_path))
    controls._folder_factory, controls._service_factory, controls._tuner_factory = Folder, Service, Tuner
    Folder.published = []
    yield controls
    controls.timer.stop()


def test_worker_joins_without_series_and_disables_master_actions(controls, workspace, qtbot):
    assert workspace.series_root is None
    assert controls.automatic.isChecked() and controls.manual_section.isHidden()
    controls.join.click()
    qtbot.waitUntil(lambda: controls._thread is None)
    assert controls._service.role == "worker"
    assert not Folder.published
    assert not controls.publish.isEnabled()
    assert not workspace.inline_layout.legacy_section.isEnabled()
    assert not workspace.inline_layout.master_controls.isEnabled()
    assert not workspace.new_analysis.isEnabled()
    assert workspace.result_views.currentWidget() is controls.results
    assert controls.workers.item(0, 2).text() == "2 / 1"
    service = controls._service
    controls.leave.click()
    qtbot.waitUntil(lambda: controls._thread is None)
    assert service.left and controls._service is None
    assert workspace.inline_layout.master_controls.isEnabled()
    assert not controls.timer.isActive()


def test_series_jobs_choice_follows_series_without_starting_participation(controls, tmp_path):
    controls.folder.clear()
    first, second = tmp_path / "first", tmp_path / "second"
    controls.set_series(first)
    assert controls.folder.text() == str(first / "jobs")
    controls.set_series(second)
    assert controls.folder.text() == str(second / "jobs")
    chosen = tmp_path / "chosen shared jobs"
    controls.folder.setText(str(chosen))
    controls.set_series(first)
    assert controls.folder.text() == str(first / "jobs")
    assert controls._service is None and not controls.timer.isActive() and not Folder.published


def test_master_publishes_selected_job_and_may_run_cpu_gpu_together(controls, workspace, qtbot, tmp_path):
    job = tmp_path / "job"
    workspace.panel._show_snapshot(snapshot(dict(root=str(job))))
    controls.publish.click()
    qtbot.waitUntil(lambda: controls._thread is None)
    assert len(Folder.published) == 1
    assert Folder.published[0] == (str(job), dict(master_machine=workspace.panel.local_machine,
        cpu_workers=2, cuda_workers=1))
    assert controls._service.options["cpu_workers"] == 2
    assert controls._service.options["cuda_workers"] == 1
    assert controls._service.role == "master"
    assert workspace.inline_layout.master_controls.isEnabled()
    controls._tick()
    qtbot.waitUntil(lambda: controls._thread is None)
    assert controls._service.capacity == (2, 1)
    assert controls.capacity_note.text() == Tuner.note


def test_legacy_admission_recovery_is_an_explicit_gui_action(controls, workspace, qtbot, tmp_path, monkeypatch):
    from darsia.fluidflower import shared_admission_lock
    calls = []
    monkeypatch.setattr(shared_admission_lock, 'recover_legacy',
        lambda folder, machine, **options: calls.append((folder, machine, options)) or dict(recovered=True))
    controls.recover_admission.click()
    assert not controls.admission_recovery.isHidden()
    assert not controls.admission_recover_now.isEnabled()
    controls.admission_confirmation.setChecked(True)
    controls.admission_recover_now.click()
    qtbot.waitUntil(lambda: controls._thread is None)
    assert len(calls) == 1 and calls[0][2]['confirmed'] is True
    assert 'Click Start' in controls.status.text()
    assert not Folder.published


def _setup_selection(workspace, tmp_path, *, commit="a" * 40, **changes):
    root = tmp_path / "setup-job"
    put(root / "job.json", dict(pipeline="series-setup", expected_runs=["setup-apply"],
        precision="float32", commit=commit))
    workspace.series_root, workspace.series_id = tmp_path / "series", "fixture"
    current = snapshot(dict(root=str(root)))
    current.update(state="Pending", job_facts=dict(pipeline="series-setup", commit=commit))
    current.update(changes)
    workspace.panel._show_snapshot(current)
    return root, current


def test_old_setup_start_is_blocked_and_reprepare_visible_in_job_control(controls, workspace, tmp_path):
    workspace.job_management._source_commit = "b" * 40
    root, current = _setup_selection(workspace, tmp_path)
    original = (root / "job.json").read_bytes()
    workspace.resize(1280, 900)
    workspace.show()
    workspace.inline_layout.show_section(workspace.inline_layout.control_section)
    assert controls.reprepare_setup.isVisible() and controls.reprepare_setup.isEnabled()
    assert controls.setup_version_note.isVisible()
    assert "aaaaaaaa" in controls.setup_version_note.text() and "bbbbbbbb" in controls.setup_version_note.text()
    assert "then click Start" in controls.setup_version_note.text()
    assert not controls.publish.isEnabled()
    for _ in range(2):
        controls.start_selected_job()
        controls._connect("master", str(root))
    assert not Folder.published and controls._service is None and controls._thread is None
    assert "Reprepare" in controls.status.text()
    assert workspace.panel.current == current
    assert (root / "job.json").read_bytes() == original


def test_first_failed_apply_offers_same_job_update_even_at_current_code(controls, workspace, tmp_path):
    workspace.job_management._source_commit = "b" * 40
    _setup_selection(workspace, tmp_path, commit="b" * 40, state="Failed", failed_runs=1,
        recovery_available=True, workers=[], runs=[dict(run="setup-apply", state="Failed", operation="apply", committed=0)])
    workspace.show()
    workspace.inline_layout.show_section(workspace.inline_layout.control_section)
    controls.sync()
    assert controls.reprepare_setup.isVisible() and controls.reprepare_setup.isEnabled()
    assert not controls.publish.isEnabled()
    controls.start_selected_job()
    assert not Folder.published
    assert "Update this failed Setup job" in controls.status.text()


def test_job_control_reprepare_uses_existing_callback_then_requires_start(
        controls, workspace, tmp_path, qtbot, monkeypatch):
    from darsia.fluidflower import setup_job_reprepare
    workspace.job_management._source_commit = "b" * 40
    root, current = _setup_selection(workspace, tmp_path)
    calls = []
    def reprepare(*args, **choices):
        calls.append((args, choices))
        return dict(root=str(root), paused=True)
    monkeypatch.setattr(setup_job_reprepare, "reprepare_unstarted_setup", reprepare)
    updated = dict(current, job_facts=dict(pipeline="series-setup", commit="b" * 40))
    def refresh():
        put(root / "job.json", dict(pipeline="series-setup", expected_runs=["setup-apply"],
            precision="float32", commit="b" * 40))
        workspace.panel._show_snapshot(updated)
    monkeypatch.setattr(workspace, "refresh_registered", refresh)
    controls.reprepare_setup.click()
    qtbot.waitUntil(workspace.job_management.can_close)
    assert len(calls) == 1 and calls[0][0][-1] == root
    assert calls[0][1]["machine"] == workspace.panel.local_machine
    assert not Folder.published and controls._service is None
    assert "Click Start" in controls.setup_version_note.text()
    assert not controls.reprepare_setup.isHidden() and controls.publish.isEnabled()
    controls.publish.click()
    qtbot.waitUntil(lambda: controls._thread is None)
    assert len(Folder.published) == 1 and Folder.published[0][0] == str(root)


@pytest.mark.parametrize("changes", [dict(state="Running", workers=[dict(machine="worker",
    worker="slot 1", run="setup-apply", state="Working", last_progress_utc="now", error="")]),
    dict(committed=1), dict(completed_operations=1), dict(failed_runs=1),
    dict(recovery_available=False)])
def test_old_started_or_unresolved_setup_is_not_reprepared(
        controls, workspace, tmp_path, monkeypatch, changes):
    from darsia.fluidflower import setup_job_reprepare
    workspace.job_management._source_commit = "b" * 40
    _setup_selection(workspace, tmp_path, **changes)
    monkeypatch.setattr(setup_job_reprepare, "reprepare_unstarted_setup",
                        lambda *args, **kwargs: pytest.fail("Started Setup must keep its source pin"))
    assert controls.reprepare_setup.isHidden() and not controls.publish.isEnabled()
    controls.start_selected_job()
    controls._reprepare_selected_setup()
    assert not Folder.published and controls._thread is None
    assert "reprepare is unavailable" in controls.setup_version_note.text()


def test_current_setup_version_uses_normal_start(controls, workspace, tmp_path, qtbot):
    workspace.job_management._source_commit = "a" * 40
    root, _ = _setup_selection(workspace, tmp_path)
    assert not controls.reprepare_setup.isHidden() and controls.setup_version_note.isHidden()
    assert controls.publish.isEnabled()
    controls.publish.click()
    qtbot.waitUntil(lambda: controls._thread is None)
    assert len(Folder.published) == 1 and Folder.published[0][0] == str(root)


def test_reprepare_refusal_stays_visible_in_job_control(controls, workspace, tmp_path, qtbot, monkeypatch):
    from darsia.fluidflower import setup_job_reprepare
    workspace.job_management._source_commit = "b" * 40
    _setup_selection(workspace, tmp_path, state="Failed")
    def refuse(*args, **kwargs):
        raise ValueError("Waiting for the original worker service to report stopping")
    monkeypatch.setattr(setup_job_reprepare, "reprepare_unstarted_setup", refuse)
    controls.reprepare_setup.click()
    qtbot.waitUntil(workspace.job_management.can_close)
    assert "original worker service" in controls.setup_version_note.text()
    controls._display(dict(role="master", workers=[], jobs=[], errors=[]))
    assert "original worker service" in controls.setup_version_note.text()
    assert not Folder.published and controls._service is None


def test_setup_start_rechecks_actual_job_before_publication(controls, workspace, tmp_path, qtbot):
    workspace.job_management._source_commit = "b" * 40
    _root, current = _setup_selection(workspace, tmp_path)
    current["job_facts"]["commit"] = "b" * 40  # Display changed before an old on-disk job was restored.
    workspace.panel._show_snapshot(current)
    controls.publish.click()
    qtbot.waitUntil(lambda: controls._thread is None)
    assert not Folder.published and controls._service is None
    assert "Refresh, reprepare" in controls.status.text()


def test_setup_source_version_is_cached_across_status_refreshes(controls, workspace, tmp_path, monkeypatch):
    import subprocess
    calls = []
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: calls.append(args) or "b" * 40)
    workspace.job_management._source_commit = None
    _setup_selection(workspace, tmp_path)
    controls.sync()
    controls.sync()
    controls.start_selected_job()
    assert len(calls) == 1 and not Folder.published


def test_job_control_keeps_one_capacity_choice_across_navigation(controls, workspace, qtbot):
    workspace.resize(1280, 900)
    workspace.show()
    inline = workspace.inline_layout
    inline.show_section(inline.control_section)
    assert controls.automatic.isVisible() and controls.automatic.isChecked()
    assert controls.publish.isHidden() and controls.leave.isHidden()
    assert controls.join.isVisible() and controls.show_status.isVisible()
    assert inline.master_controls.isHidden()
    assert inline.manual_launch_section.content.isHidden()
    choices = controls.automatic, controls.cpu, controls.gpu
    controls.automatic.setChecked(False)
    controls.cpu.setValue(4)
    controls.gpu.setValue(2)
    assert controls.cpu.isVisible() and controls.gpu.isVisible()
    inline.show_section(workspace.shared_jobs_section)
    assert controls.folder.isVisible() and not controls.automatic.isVisible()
    inline.open_job_control.click()
    assert controls.automatic.isVisible() and not controls.automatic.isChecked()
    assert (controls.automatic, controls.cpu, controls.gpu) == choices
    assert (controls.cpu.value(), controls.gpu.value()) == (4, 2)
    assert controls._service is None and not Folder.published


def test_job_control_shows_compact_relevant_actions_for_selected_state(controls, workspace, tmp_path):
    workspace.resize(1280, 900)
    workspace.show()
    inline, panel = workspace.inline_layout, workspace.panel
    inline.show_section(inline.control_section)
    data = snapshot(dict(root=str(tmp_path / "job")))
    data["control"] = dict(available=True, master_machine=panel.local_machine,
                           state=dict(action="run", sequence=0))
    panel._show_snapshot(data)
    assert controls.publish.isVisible() and controls.leave.isHidden()
    assert inline.master_controls.isVisible()
    assert panel.pause_button.isVisible() and panel.stop_button.isVisible()
    assert panel.resume_button.isHidden()
    assert panel.reset_button.isVisible() and panel.restart_button.isVisible()
    assert panel.pause_button.y() == panel.stop_button.y()
    assert panel.reset_button.y() == panel.restart_button.y()
    data["control"]["state"]["action"] = "pause"
    panel._show_snapshot(data)
    assert panel.pause_button.isHidden() and panel.resume_button.isVisible()
    assert panel.stop_button.isVisible()
    panel._show_snapshot(None)
    assert inline.master_controls.isHidden() and controls.publish.isHidden()
    assert inline.manual_launch_section.isHidden()
    assert workspace.width() == 1280


@pytest.mark.parametrize("automatic,expected", [(True, (2, 1)), (False, (4, 2))])
def test_job_control_start_passes_both_backends_to_existing_service(
        controls, workspace, qtbot, tmp_path, automatic, expected):
    workspace.show()
    workspace.inline_layout.show_section(workspace.inline_layout.control_section)
    root = tmp_path / "job"
    workspace.panel._show_snapshot(snapshot(dict(root=str(root))))
    controls.automatic.setChecked(automatic)
    controls.cpu.setValue(4)
    controls.gpu.setValue(2)
    controls.publish.click()
    qtbot.waitUntil(lambda: controls._thread is None)
    assert Folder.published == [(str(root), dict(master_machine=workspace.panel.local_machine,
        cpu_workers=expected[0], cuda_workers=expected[1]))]
    assert (controls._service.options["cpu_workers"], controls._service.options["cuda_workers"]) == expected


def test_worker_can_return_to_job_control_and_leave_without_master_actions(
        controls, workspace, qtbot):
    workspace.show()
    workspace.inline_layout.show_section(workspace.inline_layout.control_section)
    controls.join.click()
    qtbot.waitUntil(lambda: controls._thread is None)
    workspace.inline_layout.back_to_job_control.click()
    assert workspace.inline_layout.control_section.isEnabled()
    assert controls.leave.isVisible() and controls.leave.isEnabled()
    assert not workspace.inline_layout.master_controls.isEnabled()
    assert workspace.inline_layout.master_controls.isHidden()
    assert workspace.inline_layout.manual_launch_section.isHidden()
    assert not workspace.inline_layout.manual_launch_section.isEnabled()
    controls.leave.click()
    qtbot.waitUntil(lambda: controls._thread is None)
    assert controls._service is None


def test_manual_master_can_coordinate_without_local_workers(controls, workspace, qtbot, tmp_path):
    workspace.panel._show_snapshot(snapshot(dict(root=str(tmp_path / "job"))))
    controls.automatic.setChecked(False)
    controls.cpu.setValue(0)
    controls.gpu.setValue(0)
    controls.join.click()
    assert controls._service is None and "at least one worker" in controls.status.text()
    controls.publish.click()
    qtbot.waitUntil(lambda: controls._thread is None)
    assert controls._service.options["cpu_workers"] == controls._service.options["cuda_workers"] == 0


def test_shared_start_explicitly_releases_reset_while_active_polling_does_not(controls, workspace, qtbot, tmp_path):
    from darsia.fluidflower.job_lifecycle import recover_job, user_status
    from darsia.fluidflower.shared_job_folder import SharedJobFolder
    from darsia.fluidflower.shared_job_service import SharedJobService
    root = tmp_path / "pending-job"
    put(root / "job.json", dict(pipeline="native-phase-n1", expected_runs=["sample"],
        commit="a" * 40, precision="float32"))
    put(root / "queue" / "sample.json", dict(run="sample", pipeline="native-phase-n1", backend="cpu"))
    progress = dict(state="Waiting", complete=False, errors=[], failed_runs=0,
        committed=0, total_frames=1, completed_runs=0, run_count=1, runs=[], workers=[])
    calls = []

    def launch(root, **options):
        calls.append((root, options))
        return dict(started=True)

    def service(folder, machine, **options):
        return SharedJobService(folder, machine, reader=lambda _: deepcopy(progress), **options)

    controls._folder_factory, controls._service_factory = SharedJobFolder, service
    workspace.panel.local_launcher = launch
    controls.automatic.setChecked(False)
    controls.cpu.setValue(0)
    controls.gpu.setValue(0)
    workspace.panel._show_snapshot(snapshot(dict(root=str(root))))
    controls.publish.click()
    qtbot.waitUntil(lambda: controls._thread is None)
    assert controls._service is not None and not calls
    reset = recover_job(root, machine=workspace.panel.local_machine, expected_previous=None, reset=True)
    assert user_status(dict(progress, root=str(root)))["state"] == "Pending"
    controls.cpu.setValue(1)
    controls._service.update_capacity(1, 0)
    for _ in range(2):
        controls._tick()
        qtbot.waitUntil(lambda: controls._thread is None)
    assert not calls
    current = snapshot(dict(root=str(root)))
    current["control"] = workspace.panel._read_control_snapshot(root)
    assert current["control"]["state"]["command"] == reset
    workspace.panel._show_snapshot(current)
    controls.publish.click()
    qtbot.waitUntil(lambda: controls._thread is None)
    assert len(calls) == 1 and calls[0][0] == root
    assert workspace.panel._read_control_snapshot(root)["state"]["action"] == "run"


def test_network_wait_is_background_and_close_waits_for_file_operation(controls, workspace, qtbot):
    entered, release = threading.Event(), threading.Event()

    class SlowService(Service):
        def tick(self):
            entered.set()
            assert release.wait(5)
            return super().tick()

    controls._service_factory = SlowService
    controls.join.click()
    try:
        qtbot.waitUntil(entered.is_set)
        assert not workspace.can_close()
        assert not controls.join.isEnabled()
        painted = []
        QTimer.singleShot(0, lambda: painted.append(True))
        qtbot.waitUntil(lambda: bool(painted))
    finally:
        release.set()
    qtbot.waitUntil(lambda: controls._thread is None)
    assert workspace.can_close()


def test_failed_connection_is_visible_and_can_retry(controls, qtbot):
    class FailedFolder(Folder):
        def __init__(self, path):
            raise OSError("Share unavailable")

    controls._folder_factory = FailedFolder
    controls.join.click()
    qtbot.waitUntil(lambda: controls._thread is None)
    assert "Share unavailable" in controls.status.text()
    assert controls.join.isEnabled() and not controls.timer.isActive()


def test_stale_worker_is_not_reported_online(controls):
    controls._display(dict(role="master", workers=[dict(machine="Other", role="worker",
        cpu_workers=4, cuda_workers=1, stale=True, state="online")], jobs=[], errors=[]))
    assert controls.workers.item(0, 3).text() == "Stale — connection unknown"


def test_old_gui_sessions_do_not_duplicate_the_current_machine(controls):
    base = dict(machine="worker-host", role="master", cpu_workers=1, cuda_workers=0)
    reports = [dict(base, session_id="old", stale=True, state="online",
                    reported_utc="2024-01-01T00:00:00+00:00"),
               dict(base, session_id="current", stale=False, state="online",
                    reported_utc="2024-01-01T01:00:00+00:00")]
    controls._display(dict(role="master", workers=reports, jobs=[], errors=[]))
    assert controls.workers.rowCount() == 1
    assert controls.workers.item(0, 3).text() == "online"
    assert "1 earlier GUI session" in controls.workers.item(0, 3).toolTip()
    assert len(reports) == 2 and reports[0]["session_id"] == "old"


def test_two_guis_share_jobs_capacity_and_progress_without_ssh(controls, workspace, tmp_path, qtbot):
    from darsia.fluidflower.native_jobs_workspace import NativeJobsWorkspace
    from darsia.fluidflower.native_jobs_registry import JobsUIRegistry
    from darsia.fluidflower.shared_job_folder import SharedJobFolder
    from darsia.fluidflower.shared_job_service import SharedJobService

    root = tmp_path / "calculation"
    put(root / "job.json", dict(pipeline="native-phase-n1", series_id="FF_UM",
        expected_runs=["a", "b"], commit="a" * 40, precision="float32"))
    for run in ("a", "b"):
        put(root / "queue" / (run + ".json"), dict(run=run, pipeline="native-phase-n1", backend="any"))
    progress = dict(state="Waiting", complete=False, errors=[], failed_runs=0,
        committed=0, total_frames=20, completed_runs=0, run_count=2, runs=[], workers=[])
    calls = []

    def launch(root, **options):
        calls.append(options)
        return dict(started=True)

    def service(folder, machine, **options):
        return SharedJobService(folder, machine, reader=lambda registration: deepcopy(progress), **options)

    controls._folder_factory, controls._service_factory = SharedJobFolder, service
    workspace.panel.local_launcher = launch
    controls.automatic.setChecked(False)
    controls.cpu.setValue(0)
    controls.gpu.setValue(0)
    workspace.panel._show_snapshot(snapshot(dict(root=str(root))))
    controls.publish.click()
    qtbot.waitUntil(lambda: controls._thread is None)
    assert not calls

    worker_owner = NativeJobsWorkspace(registry=JobsUIRegistry(tmp_path / "worker-registry"),
        local_machine="Second", reader=snapshot, local_launcher=launch,
        remote_launcher=lambda *args, **kwargs: pytest.fail("No SSH may be used"))
    qtbot.addWidget(worker_owner)
    worker = worker_owner.shared_jobs
    worker._settings = controls._settings
    worker._folder_factory, worker._service_factory = SharedJobFolder, service
    worker.folder.setText(str(tmp_path))
    worker.automatic.setChecked(False)
    worker.cpu.setValue(2)
    worker.gpu.setValue(1)
    try:
        worker.join.click()
        qtbot.waitUntil(lambda: worker._thread is None)
        assert {(call["backend"], call["workers"]) for call in calls} == {("cpu", 2), ("cuda", 1)}
        assert all(call["machine"] == "second" for call in calls)
        progress.update(state="Running", committed=14)
        worker._tick()
        qtbot.waitUntil(lambda: worker._thread is None)
        controls._tick()
        qtbot.waitUntil(lambda: controls._thread is None)
        assert len(calls) == 2
        assert controls.workers.rowCount() == 2
        assert controls.jobs.item(0, 0).text() == "FF_UM"
        assert controls.jobs.item(0, 3).text() == "14 / 20"
        assert any("14/20" in controls.workers.item(row, 5).text()
                   for row in range(controls.workers.rowCount()))
        workspace.resize(1600, 1000)
        workspace.show()
        qtbot.waitExposed(workspace)
        assert workspace.grab().save(str(tmp_path / "shared-jobs.png"))
    finally:
        qtbot.waitUntil(lambda: worker._thread is None and controls._thread is None)
        worker.leave_folder()
        controls.leave_folder()
        qtbot.waitUntil(lambda: worker._thread is None and controls._thread is None)
