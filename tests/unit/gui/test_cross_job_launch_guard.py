"""Cross-job service reservations, tiny local metadata and fake launches only."""

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from .test_jobs_metadata import launches, model, put, job as metadata_job
from .test_jobs_panel import ui, app, panel, wait
from .test_job_launch_state import begin, append_log


def job_pair(metadata_job, tmp_path):
    root_a, _, _ = metadata_job
    root_b = tmp_path / "other-job"
    shutil.copytree(root_a, root_b)
    # The tiny fixture's master binding/path is job-specific, as in production.
    master = model.read_json(root_b / "ui" / "master.json")
    master["job"] = model.descriptor(root_b / "job.json")
    master["name"] = "Job B"
    put(root_b / "ui" / "master.json", master)
    return root_a, root_b


def registered(root, label):
    return dict(root=str(root), name=label, job=model.descriptor(root / "job.json"))


@pytest.mark.parametrize("terminal", ["stopped", "startup_failure"])
def test_other_series_registered_request_blocks_b_until_a_own_terminal(
    panel, app, metadata_job, tmp_path, terminal
):
    widget, calls = panel
    root_a, root_b = job_pair(metadata_job, tmp_path)
    request = begin(root_a)
    widget.reader = model.JobSnapshotReader()
    widget.known_jobs_provider = lambda: [
        registered(root_a, "Job A"),
        registered(root_b, "Job B"),
    ]
    # Reopened B context lists only B; A is known solely through local registry.
    widget.set_context(jobs_provider=lambda: [registered(root_b, "Job B")])
    wait(app, widget.can_close)
    assert widget.current["root"] == str(root_b)
    assert (
        not widget.start_button.isEnabled()
        and str(root_a) in widget.launch_state_label.text()
    )
    widget.start_local_workers()
    widget.start_selected_machine()
    assert calls == [] and not (root_b / "ui" / "launches").exists()
    append_log(root_b, request, "stopped", machine="main-cpu")
    append_log(root_a, request, "completed")
    widget.refresh()
    wait(app, widget.can_close)
    assert (
        not widget.start_button.isEnabled()
    )  # Neither foreign stop nor progress releases A.
    append_log(root_a, request, terminal)
    widget.refresh()
    wait(app, widget.can_close)
    assert widget.start_button.isEnabled() and calls == []
    assert model.JobSnapshotReader()({"root": str(root_a)})["committed"] == 0


def test_cross_job_explicit_declaration_binds_a_without_touching_b_or_science(
    panel, app, metadata_job, tmp_path, monkeypatch
):
    widget, calls = panel
    root_a, root_b = job_pair(metadata_job, tmp_path)
    request = begin(root_a)
    original = Path(request["source"]["path"]).read_bytes()
    widget.reader = model.JobSnapshotReader()
    widget.known_jobs_provider = lambda: [
        registered(root_a, "Job A"),
        registered(root_b, "Job B"),
    ]
    widget.set_context(jobs_provider=lambda: [registered(root_b, "Job B")])
    wait(app, widget.can_close)
    questions = []
    monkeypatch.setattr(
        ui.QMessageBox,
        "question",
        lambda *a, **k: (questions.append(a[2]) or ui.QMessageBox.StandardButton.No),
    )
    widget.declare_previous_service_stopped()
    assert not widget.start_button.isEnabled() and not list(
        (root_a / "ui" / "launches").glob("*.declaration.*.json")
    )
    monkeypatch.setattr(
        ui.QMessageBox,
        "question",
        lambda *a, **k: (questions.append(a[2]) or ui.QMessageBox.StandardButton.Yes),
    )
    widget.declare_previous_service_stopped()
    wait(app, widget.can_close)
    assert widget.start_button.isEnabled() and calls == []
    assert all(str(root_a) in text for text in questions)
    declaration = list((root_a / "ui" / "launches").glob("*.declaration.*.json"))
    assert len(declaration) == 1 and not (root_b / "ui" / "launches").exists()
    saved = model.read_json(declaration[0])
    assert saved["job"] == request["job"] and saved["request"] == request["source"]
    assert (
        saved["scientific_evidence"] is False
        and saved["worker_liveness_verified"] is False
    )
    assert Path(request["source"]["path"]).read_bytes() == original
    assert widget.current["committed"] == 0


def test_unreadable_registered_job_never_assumed_dead_and_direct_start_stays_blocked(
    panel, app, metadata_job, tmp_path
):
    widget, calls = panel
    root_a, root_b = job_pair(metadata_job, tmp_path)
    begin(root_a)
    binding = registered(root_a, "Job A")
    widget.reader = model.JobSnapshotReader()
    widget.known_jobs_provider = lambda: [binding, registered(root_b, "Job B")]
    widget.set_context(jobs_provider=lambda: [registered(root_b, "Job B")])
    wait(app, widget.can_close)
    put(
        root_a / "job.json",
        dict(expected_runs=["ac52"], precision="float32", externally_changed=True),
    )
    widget.refresh()
    wait(app, widget.can_close)
    assert not widget.start_button.isEnabled()
    assert (
        "Launch state unavailable" in widget.launch_state_label.text()
        and str(root_a) in widget.launch_state_label.text()
    )
    widget.start_local_workers()
    widget.start_selected_machine()
    assert calls == [] and not (root_b / "ui" / "launches").exists()


def test_known_reservations_match_normalized_alias_and_backend_only(
    panel, app, metadata_job, tmp_path
):
    widget, calls = panel
    root_a, root_b = job_pair(metadata_job, tmp_path)
    launches.begin_launch_request(
        root_a, machine="mAiN", backend="cuda", workers=1, kind="local"
    )
    widget.reader = model.JobSnapshotReader()
    widget.known_jobs_provider = lambda: [
        registered(root_a, "A"),
        registered(root_b, "B"),
    ]
    widget.set_context(jobs_provider=lambda: [registered(root_b, "B")])
    wait(app, widget.can_close)
    assert widget.start_button.isEnabled()  # CPU remains independent.
    widget.backend_combo.setCurrentIndex(widget.backend_combo.findData("cuda"))
    assert not widget.start_button.isEnabled() and calls == []


def test_explicit_start_revalidates_other_registered_roots_after_snapshot(
    panel, app, metadata_job, tmp_path
):
    widget, calls = panel
    root_a, root_b = job_pair(metadata_job, tmp_path)
    widget.reader = model.JobSnapshotReader()
    widget.known_jobs_provider = lambda: [
        registered(root_a, "A"),
        registered(root_b, "B"),
    ]
    widget.set_context(jobs_provider=lambda: [registered(root_b, "B")])
    wait(app, widget.can_close)
    assert widget.start_button.isEnabled()
    begin(root_a)  # Another explicit UI request appeared after the last refresh.
    widget.start_local_workers()
    assert calls == [] and not widget.start_button.isEnabled()
    assert not (root_b / "ui" / "launches").exists()


def test_local_registry_reopen_lists_explicit_other_series_roots_without_nas_scan(
    metadata_job, tmp_path, monkeypatch
):
    from darsia.fluidflower import native_jobs_registry as registry_module

    monkeypatch.setattr(registry_module, "require_unc", lambda value: None)
    root_a, root_b = job_pair(metadata_job, tmp_path)
    local = tmp_path / "local-ui-registry"
    registry = registry_module.JobsUIRegistry(local)
    registry.register_job(tmp_path / "series-a", "series-a", root_a)
    registry.register_job(tmp_path / "series-b", "series-b", root_b)
    unregistered = tmp_path / "not-registered"
    unregistered.mkdir()
    put(unregistered / "job.json", dict(expected_runs=["ac52"], precision="float32"))
    before = {str(path): path.read_bytes() for path in local.glob("*.json")}
    reopened = registry_module.JobsUIRegistry(local)
    roots = {row["root"] for row in reopened.known_jobs()}
    assert roots == {str(root_a), str(root_b)} and str(unregistered) not in roots
    assert before == {str(path): path.read_bytes() for path in local.glob("*.json")}
    assert {
        row["root"] for row in reopened.jobs(tmp_path / "series-b", "series-b")
    } == {str(root_b)}


def test_malformed_local_registry_is_not_silently_ignored(tmp_path):
    from darsia.fluidflower import native_jobs_registry as registry_module

    local = tmp_path / "local-ui-registry"
    local.mkdir()
    put(local / "foreign.json", dict(format="unproved-source", jobs=[]))
    with pytest.raises(ValueError, match="Malformed local Jobs"):
        registry_module.JobsUIRegistry(local).known_jobs()
