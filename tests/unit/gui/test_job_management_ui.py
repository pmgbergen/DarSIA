import json
from types import SimpleNamespace

from darsia.fluidflower import job_control
from darsia.fluidflower.job_launch_state import begin_launch_request
from ..test_job_management import retired_claim
from ..test_series_setup_jobs import setup_job, controls  # noqa: F401
from ..test_setup_image_preparation_update import image_admission_failure
from .test_native_jobs_owner import workspace, local_registry, job, series


def test_completed_setup_offers_explicit_image_preparation_without_repeating_geometry():
    from darsia.fluidflower.job_management_ui import JobManagementControls
    controls = SimpleNamespace(source_commit=lambda: "b" * 40)
    current = dict(job_facts=dict(pipeline="series-setup", commit="a" * 40),
        state="Completed", failed_runs=0, completed_operations=2, completed_runs=2,
        committed=0, workers=[], runs=[
            dict(run="setup-apply", operation="apply", state="Complete"),
            dict(run="sample", operation="build", state="Complete")])
    status = JobManagementControls.setup_source_status(controls, current)
    assert status["can_reprepare"] and status["action_label"] == "Update image correction"
    assert "only the selected image correction is added" in status["message"]
    # Finished immutable claims still exist; this flag describes generic Reset,
    # not whether the completed Setup may add one new typed operation.
    assert JobManagementControls.setup_source_status(controls,
        dict(current, recovery_available=False))["can_reprepare"]
    assert not JobManagementControls.setup_source_status(controls,
        dict(current, workers=[dict(run="sample", state="Running")]))["can_reprepare"]


def test_failed_image_preparation_offers_retry_while_geometry_stays_complete():
    from darsia.fluidflower.job_management_ui import JobManagementControls
    controls = SimpleNamespace(source_commit=lambda: "b" * 40)
    current = dict(job_facts=dict(pipeline="series-setup", commit="a" * 40),
        state="Failed", failed_runs=1, recovery_available=False, completed_operations=2,
        committed=0, workers=[], runs=[
            dict(run="setup-apply", operation="apply", state="Completed"),
            dict(run="sample", operation="build", state="Completed"),
            dict(run="setup-image-preparation", operation="image-preparation", state="Failed")])
    result = JobManagementControls.setup_source_status(controls, current)
    assert result["can_reprepare"] and result["action_label"] == "Retry image correction"
    assert result["message"].startswith("Retry image correction in this job")
    assert "partial image files are kept in job history" in result["message"]
    assert not JobManagementControls.setup_source_status(controls,
        dict(current, workers=[dict(run="setup-image-preparation", state="Running")]))["can_reprepare"]


def test_completed_images_offer_same_job_per_run_colour_path_rebuild():
    from darsia.fluidflower.job_management_ui import JobManagementControls
    controls = SimpleNamespace(source_commit=lambda: "b" * 40)
    current = dict(job_facts=dict(pipeline="series-setup", commit="a" * 40),
        state="Completed", failed_runs=0, recovery_available=False,
        completed_operations=3, completed_runs=3, committed=0, workers=[], runs=[
            dict(run="setup-apply", operation="apply", state="Complete"),
            dict(run="sample", operation="build", state="Complete"),
            dict(run="setup-image-preparation", operation="image-preparation", state="Complete")])
    status = JobManagementControls.setup_source_status(controls, current)
    assert status["can_reprepare"] and status["action_label"] == "Rebuild colour paths per run"
    assert "completed corrected image samples" in status["message"]
    assert "existing Apply, Builds, image corrections" in status["message"]
    assert not JobManagementControls.setup_source_status(controls,
        dict(current, workers=[dict(run="setup-image-preparation", state="Running")]))["can_reprepare"]


def test_pre_owner_image_admission_exposes_real_update_action_after_pause_and_retirement(
        setup_job, workspace, qtbot, monkeypatch):
    from pathlib import Path
    from darsia.fluidflower import job_lifecycle, setup_job_reprepare
    from darsia.fluidflower.jobs_metadata import JobSnapshotReader
    from darsia.fluidflower.pipeline import job_queue
    from darsia.fluidflower.setup_completed_builds import retained_operations
    from ..test_setup_job_reprepare import _bytes
    case, claim, lease = image_admission_failure(setup_job, monkeypatch)
    raw_outputs = {build["run_id"]: _bytes(Path(build["request_template"]["output"]["destination"]))
                   for build in case.plan["build_plans"]}
    original = JobSnapshotReader()(case.registry.jobs(case.series, "synthetic")[0])
    current = job_lifecycle.user_status(original, local_machine="private-host",
                                        stores_path=case.stores.source["path"])
    assert current["state"] == "Starting" and not current["recovery_available"]
    workspace.registry = case.registry
    workspace.series_root, workspace.series_id = case.series, "synthetic"
    workspace.stores.path = Path(case.stores.source["path"])
    workspace.panel.local_machine = "private-host"
    workspace.panel.current = current
    controls = workspace.job_management
    controls._source_commit = case.identity["commit"]
    controls.selection_changed()
    assert controls.reprepare.isHidden()  # A launcher/claim alone grants nothing.
    prior = job_control.read_control(case.job_root)
    job_control.publish_control(case.job_root, "pause", machine="private-host",
                                expected_previous=prior.get("command"))
    paused = job_lifecycle.user_status(original, local_machine="private-host",
                                      stores_path=case.stores.source["path"])
    workspace.panel.current = paused
    controls.selection_changed()
    assert paused["state"] == "Starting" and not paused["recovery_available"]
    assert not controls.reprepare.isHidden() and controls.reprepare.isEnabled()
    assert controls.reprepare.text() == "Retry image correction"
    assert "before worker ownership" in controls.setup_source_status()["message"]
    # A later started service report revokes the review offer, even if the
    # selected snapshot still describes the same ownerless claim.
    log = case.job_root / "logs/private-host-cpu.log"
    original_log = log.read_bytes()
    with log.open("a", encoding="utf-8") as stream:
        stream.write(job_control.utc() + ' {"kind":"started"}\n')
    controls.selection_changed()
    assert controls.reprepare.isHidden()
    log.write_bytes(original_log)
    controls.selection_changed()
    assert not controls.reprepare.isHidden()
    assert not list(lease.iterdir()) and claim.exists()
    monkeypatch.setattr(setup_job_reprepare, "current_setup_identity", lambda: case.identity)
    monkeypatch.setattr(workspace, "refresh_registered", lambda: None)
    controls.actor.setText("fixture-user")
    controls.reprepare.click()
    qtbot.waitUntil(controls.can_close, timeout=15000)
    assert "Completed geometry is retained" in controls.note.text()
    assert "run image correction with the saved choices" in controls.note.text()
    assert {path.stem for path in (case.job_root / "queue").glob("*.json")} == {"setup-image-preparation"}
    assert set(retained_operations(case.job_root)) == {"setup-apply", *case.plan["runs"]}
    for build in case.plan["build_plans"]:
        assert _bytes(Path(build["request_template"]["output"]["destination"])) == raw_outputs[build["run_id"]]


def test_update_job_offered_for_pending_partial_build_without_fake_failures():
    from darsia.fluidflower.job_management_ui import JobManagementControls
    controls = SimpleNamespace(source_commit=lambda: "b" * 40)
    current = dict(job_facts=dict(pipeline="series-setup", commit="a" * 40),
        state="Pending", failed_runs=0, completed_runs=12, completed_operations=12,
        workers=[], committed=0, runs=[
            dict(run="setup-apply", operation="apply", state="Completed"),
            *[dict(run=f"run{i}", operation="build", state="Complete") for i in range(11)],
            dict(run="remaining", operation="build", state="Pending")])
    status = JobManagementControls.setup_source_status(controls, current)
    assert status["can_reprepare"] and "Completed Builds" in status["message"]
    assert not JobManagementControls.setup_source_status(controls,
        dict(current, workers=[dict(run="remaining", state="Working")]))["can_reprepare"]
    assert not JobManagementControls.setup_source_status(controls,
        dict(current, recovery_available=False))["can_reprepare"]


def test_update_job_offered_for_backend_review_after_any_build_failure():
    from darsia.fluidflower.job_management_ui import JobManagementControls
    from ..test_setup_failed_build import PARSER_TRACE
    controls = SimpleNamespace(source_commit=lambda: "b" * 40)
    current = dict(job_facts=dict(pipeline="series-setup", commit="a" * 40),
        state="Cancelled", failed_runs=2, completed_runs=1, completed_operations=1,
        workers=[], committed=0, runs=[
            dict(run="setup-apply", operation="apply", state="Completed", detail_state="Complete"),
            dict(run="run1", operation="build", state="Failed", error=PARSER_TRACE),
            dict(run="run2", operation="build", state="Failed", error=PARSER_TRACE),
            dict(run="run3", operation="build", state="Cancelled", detail_state="Queued")])
    status = JobManagementControls.setup_source_status(controls, current)
    assert status["can_reprepare"] and status["blocked"]
    assert "Saved Apply files are retained" in status["message"]
    changed_error = dict(current, runs=[current["runs"][0],
        dict(run="run1", operation="build", state="Failed", error="ValueError: protocols differ"),
        dict(run="run2", operation="build", state="Failed", error="A different pre-output failure")])
    assert JobManagementControls.setup_source_status(controls, changed_error)["can_reprepare"]
    retired = dict(current, recovery_available=True,
        workers=[dict(run="run3", state="Claimed (liveness not established)")])
    assert JobManagementControls.setup_source_status(controls, retired)["can_reprepare"]
    assert not JobManagementControls.setup_source_status(controls,
        dict(retired, recovery_available=False))["can_reprepare"]
    with_completed = dict(current, completed_runs=6, completed_operations=6,
        runs=[*current["runs"], *[dict(run="done"+str(i), operation="build", state="Complete") for i in range(5)]])
    assert JobManagementControls.setup_source_status(controls, with_completed)["can_reprepare"]
    for change in (dict(workers=[dict(run="run3")]), dict(committed=1),
                   dict(runs=[current["runs"][0], dict(run="run1", operation="apply", state="Failed")])):
        assert not JobManagementControls.setup_source_status(controls, dict(current, **change))["can_reprepare"]


def test_setup_reprepare_action_uses_captured_selection_and_hides_for_other_jobs(workspace, tmp_path, qtbot, monkeypatch, qapp):
    from PySide6.QtCore import QThread
    from darsia.fluidflower import setup_job_reprepare
    root = series(tmp_path / "series", "SERIES")
    selected = job(root / "jobs" / "fixture-job")
    workspace.registry.register_job(root, "SERIES", selected)
    workspace.set_series("SERIES", root)
    qtbot.waitUntil(workspace.can_close, timeout=5000)
    controls = workspace.job_management
    controls.actor.setText("fixture-user")
    controls._source_commit = "b" * 40
    current = dict(workspace.panel.current, state="Pending", workers=[],
                   job_facts=dict(pipeline="series-setup", commit="a" * 40))
    workspace.panel.current = current
    controls.selection_changed()
    assert not controls.reprepare.isHidden()
    calls = []
    def retry(*args, **choices):
        calls.append((args, choices, QThread.currentThread() is not qapp.thread()))
        return dict(root=str(selected))
    monkeypatch.setattr(setup_job_reprepare, "reprepare_unstarted_setup", retry)
    controls.reprepare.click()
    workspace.panel.current = dict(current, root=str(root / "jobs/another-job"))
    qtbot.waitUntil(controls.can_close, timeout=5000)
    assert len(calls) == 1 and calls[0][2]
    assert calls[0][0][-1] == selected
    assert calls[0][1]["actor"] == "fixture-user"
    workspace.panel.current = dict(current, job_facts=dict(pipeline="other", commit="a" * 40))
    controls.selection_changed()
    assert controls.reprepare.isHidden()
    workspace.panel.current = dict(current, job_facts=dict(pipeline="series-setup", commit="b" * 40))
    controls.selection_changed()
    assert not controls.reprepare.isHidden()
    status=controls.setup_source_status()
    assert status["can_reprepare"] and not status["blocked"] and not status["mismatch"]
    workspace.panel.current = dict(workspace.panel.current,workers=[dict(run="working")])
    controls.selection_changed()
    assert controls.reprepare.isHidden()


def test_archive_restore_and_delete_review_are_inline_and_do_not_delete_without_confirmation(workspace, tmp_path, qtbot):
    root = series(tmp_path / "series", "SERIES")
    selected = job(root / "jobs" / "fixture-job")
    workspace.registry.register_job(root, "SERIES", selected)
    workspace.set_series("SERIES", root)
    qtbot.waitUntil(workspace.can_close, timeout=5000)
    controls = workspace.job_management
    controls.actor.setText("fixture-user")
    controls.archive.click()
    qtbot.waitUntil(workspace.can_close, timeout=5000)
    assert workspace.panel.job_table.rowCount() == 0 and (selected / "job.json").exists()
    controls.show_archived.setChecked(True)
    qtbot.waitUntil(workspace.can_close, timeout=5000)
    assert workspace.panel.current["archived"] and controls.restore.isEnabled()
    controls.restore.click()
    qtbot.waitUntil(workspace.can_close, timeout=5000)
    assert not workspace.panel.current["archived"]
    controls.delete.click()
    qtbot.waitUntil(controls.can_close, timeout=5000)
    assert controls.review is not None and not controls.confirmation.isWindow()
    assert (selected / "job.json").exists()
    controls.cancel_delete.click()
    assert controls.review is None and (selected / "job.json").exists()
    controls.delete.click()
    qtbot.waitUntil(controls.can_close, timeout=5000)
    controls.confirm_delete.click()
    qtbot.waitUntil(workspace.can_close, timeout=5000)
    assert not selected.exists() and (root / "series.toml").exists()
    assert workspace.panel.job_table.rowCount() == 0


def test_stopped_unfinished_job_has_inline_archive_action_that_keeps_all_files(workspace, tmp_path, qtbot):
    root = series(tmp_path / "series", "SERIES")
    selected = job(root / "jobs" / "fixture-job")
    claim = selected / "claimed" / "worker" / "ac52.json"
    claim.parent.mkdir(parents=True)
    claim.write_text(json.dumps(dict(run="ac52", attempt_id="a" * 32)), encoding="utf-8")
    logs = selected / "logs"
    logs.mkdir()
    (logs / "worker.log").write_text('timestamp {"kind":"stopped"}\n', encoding="utf-8")
    original = {path.relative_to(selected): path.read_bytes() for path in selected.rglob("*") if path.is_file()}
    workspace.registry.register_job(root, "SERIES", selected)
    workspace.set_series("SERIES", root)
    qtbot.waitUntil(workspace.can_close, timeout=5000)
    controls = workspace.job_management
    controls.actor.setText("fixture-user")
    assert controls.archive_instead.isHidden()
    controls.delete.click()
    qtbot.waitUntil(controls.can_close, timeout=5000)
    assert "Stopping a job preserves unfinished work" in controls.note.text()
    assert controls.review is None and controls.confirmation.isHidden()
    assert not controls.archive_instead.isHidden() and controls.archive_instead.isEnabled()
    controls.archive_instead.click()
    qtbot.waitUntil(workspace.can_close, timeout=5000)
    assert "Job archived. Its files are kept." == controls.note.text()
    assert workspace.panel.job_table.rowCount() == 0 and controls.archive_instead.isHidden()
    assert {path.relative_to(selected): path.read_bytes() for path in selected.rglob("*") if path.is_file()} == original
    controls.show_archived.setChecked(True)
    qtbot.waitUntil(workspace.can_close, timeout=5000)
    assert workspace.panel.current["archived"] and controls.restore.isEnabled()
    assert controls.archive_instead.isHidden()


def test_stop_delete_requires_explicit_review_and_original_service_report(workspace, tmp_path, qtbot):
    root = series(tmp_path / "series", "SERIES")
    selected = job(root / "jobs" / "fixture-job")
    job_control.write_immutable(selected / "ui" / "master.json", dict(
        format="darsia-job-ui-master-v1", machine="Main", job=job_control.descriptor(selected / "job.json")))
    begin_launch_request(selected, machine="Main", backend="cpu", workers=1, kind="local")
    workspace.registry.register_job(root, "SERIES", selected)
    workspace.set_series("SERIES", root)
    qtbot.waitUntil(workspace.can_close, timeout=5000)
    controls = workspace.job_management
    controls.actor.setText("fixture-user")
    controls.stop_delete.click()
    qtbot.waitUntil(controls.can_close, timeout=5000)
    assert controls.stop_review is not None and str(selected) in controls.review_text.text()
    assert "dependent consumers" in controls.review_text.text()
    assert not controls.confirm_stop_delete.isEnabled()
    assert job_control.read_control(selected)["sequence"] == 0
    controls.destructive_ack.setChecked(True)
    controls.confirm_stop_delete.click()
    qtbot.waitUntil(controls.can_close, timeout=5000)
    assert job_control.read_control(selected)["action"] == "stop"
    controls.check_retirement.click()
    qtbot.waitUntil(controls.can_close, timeout=5000)
    assert "waiting for the original service" in controls.note.text()
    assert controls.review is None and (selected / "job.json").exists()
    logs = selected / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "Main-cpu.log").write_text(job_control.utc() + ' {"kind":"stopped"}\n', encoding="utf-8")
    controls.check_retirement.click()
    qtbot.waitUntil(controls.can_close, timeout=5000)
    assert controls.review is not None and not controls.confirm_delete.isHidden()
    assert not controls.retirement_timer.isActive()
    assert (selected / "job.json").exists()
    controls.cancel_delete.click()
    assert controls.review is None and controls.stop_review is None


def test_gui_confirmed_delete_abandons_only_the_retired_selected_job(workspace, tmp_path, qtbot, monkeypatch):
    root = series(tmp_path / "series", "SERIES")
    selected = job(root / "jobs" / "fixture-job")
    job_control.write_immutable(selected / "ui" / "master.json", dict(
        format="darsia-job-ui-master-v1", machine="Main", job=job_control.descriptor(selected / "job.json")))
    claimed = retired_claim(selected, monkeypatch, service="main-cpu")
    raw = root / "raw"
    raw.mkdir()
    (raw / "source.ARW").write_bytes(b"original raw")
    geometry = root / "approved_geometry.json"
    geometry.write_bytes(b"approved geometry")
    workspace.registry.register_job(root, "SERIES", selected)
    workspace.set_series("SERIES", root)
    qtbot.waitUntil(workspace.can_close, timeout=5000)
    controls = workspace.job_management
    controls.actor.setText("fixture-user")
    controls.stop_delete.click()
    qtbot.waitUntil(controls.can_close, timeout=5000)
    controls.destructive_ack.setChecked(True)
    controls.confirm_stop_delete.click()
    qtbot.waitUntil(controls.can_close, timeout=5000)
    controls.check_retirement.click()
    qtbot.waitUntil(controls.can_close, timeout=5000)
    assert controls.review is not None
    assert "permanently cancels" in controls.review_text.text()
    assert claimed.exists() and not (selected / "decisions").exists()
    controls.confirm_delete.click()
    qtbot.waitUntil(workspace.can_close, timeout=5000)
    assert not selected.exists() and workspace.panel.job_table.rowCount() == 0
    assert (raw / "source.ARW").read_bytes() == b"original raw"
    assert geometry.read_bytes() == b"approved geometry"
