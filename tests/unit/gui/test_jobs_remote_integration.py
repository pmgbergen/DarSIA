"""Local Qt and fake launchers only; never executes local worker or SSH."""

from copy import deepcopy
from pathlib import Path

import pytest

from .test_jobs_panel import ui, remote, model, put, snapshot, wait, app, panel
from .test_jobs_metadata import job as metadata_job


def facts(tmp_path, *, shell="windows-powershell"):
    ssh = tmp_path / "existing-SSH" / "ssh.exe"
    ssh.parent.mkdir(exist_ok=True)
    ssh.write_bytes(b"not an executable; fixture only")
    return dict(
        machine="Second",
        host="existing-host",
        checkout=r"C:\Existing repo\DarSIA-worker",
        python=r"C:\Existing env\Scripts\python.exe",
        ssh_alias="existing-second-alias",
        ssh_executable=str(ssh),
        default_shell=shell,
    )


def record(root, facts, *, suffix=""):
    value = dict(
        facts,
        format="darsia-job-machine-registration-v1",
        job=model.descriptor(root / "job.json"),
        registered_utc="2026-10-16T02:00:00+00:00",
        remote_contacted=False,
        installed=False,
    )
    bound = put(root / "ui" / "machines" / (facts["machine"] + suffix + ".json"), value)
    return dict(value, source=bound)


def test_remote_form_starts_unknown_requires_explicit_shell_and_chosen_existing_ssh(
    app, tmp_path, monkeypatch
):
    dialog = ui.RegisterMachineDialog(local_machine="Main")
    assert (
        dialog.default_shell_combo.currentData() == ""
        and dialog.ssh_executable_edit.isReadOnly()
    )
    values = facts(tmp_path)
    for field, key in [
        (dialog.machine_edit, "machine"),
        (dialog.host_edit, "host"),
        (dialog.checkout_edit, "checkout"),
        (dialog.python_edit, "python"),
        (dialog.ssh_edit, "ssh_alias"),
    ]:
        field.setText(values[key])
    assert not dialog.ok_button.isEnabled()
    monkeypatch.setattr(
        ui.QFileDialog,
        "getOpenFileName",
        lambda *a, **k: (values["ssh_executable"], ""),
    )
    dialog._choose_ssh()
    assert not dialog.ok_button.isEnabled() and dialog.facts()["default_shell"] == ""
    dialog.default_shell_combo.setCurrentIndex(1)
    assert (
        dialog.ok_button.isEnabled()
        and dialog.facts()["ssh_executable"] == values["ssh_executable"]
    )
    assert not dialog.new_version.isChecked()
    dialog.close()
    dialog.deleteLater()
    app.processEvents()


def test_complete_remote_registration_is_immutable_new_version_explicit(
    tmp_path, monkeypatch
):
    put(tmp_path / "job.json", dict(expected_runs=["ac52"]))
    monkeypatch.setattr(ui, "require_unc", lambda path: None)
    monkeypatch.setattr(remote, "require_unc", lambda path: None)
    complete = facts(tmp_path)
    original = dict(complete)
    receipt = ui.register_machine_facts(tmp_path, complete, local_machine="Main")
    stored = Path(receipt["path"]).read_bytes()
    assert (
        complete == original
        and not model.read_json(Path(receipt["path"]))["remote_contacted"]
    )
    with pytest.raises(FileExistsError):
        ui.register_machine_facts(tmp_path, complete, local_machine="Main")
    second = ui.register_machine_facts(
        tmp_path,
        dict(complete, ssh_alias="existing-other-alias"),
        local_machine="Main",
        new_version=True,
    )
    assert (
        second["path"] != receipt["path"]
        and Path(receipt["path"]).read_bytes() == stored
    )
    assert len(list((tmp_path / "ui" / "machines").glob("*.json"))) == 2


def test_missing_remote_shell_refuses_before_any_registration_or_launcher(
    tmp_path, monkeypatch
):
    put(tmp_path / "job.json", dict(expected_runs=["ac52"]))
    monkeypatch.setattr(ui, "require_unc", lambda path: None)
    monkeypatch.setattr(remote, "require_unc", lambda path: None)
    with pytest.raises(ValueError, match="PowerShell default shell"):
        ui.register_machine_facts(
            tmp_path, facts(tmp_path, shell=""), local_machine="Main"
        )
    assert not (tmp_path / "ui").exists()


def test_current_job_reader_includes_exact_bound_registration_versions(
    metadata_job, tmp_path
):
    root, _, _ = metadata_job
    saved = record(root, facts(tmp_path))
    saved2 = record(root, facts(tmp_path), suffix=".explicit-version")
    value = model.JobSnapshotReader()({"root": str(root)})
    assert sorted(
        row["source"]["path"] for row in value["registered_machines"]
    ) == sorted([saved["source"]["path"], saved2["source"]["path"]])
    foreign = model.read_json(Path(saved2["source"]["path"]))
    foreign["job"]["sha256"] = "f" * 64
    put(saved2["source"]["path"], foreign)
    with pytest.raises(ValueError, match="another job"):
        model.JobSnapshotReader()({"root": str(root)})


def test_start_selected_machine_routes_registered_remote_and_local_separately(
    panel, app, tmp_path, monkeypatch
):
    widget, oldcalls = panel
    root = tmp_path / "job"
    put(root / "job.json", dict(expected_runs=["ac52"]))
    saved = record(root, facts(tmp_path))
    launches = []
    local = []
    monkeypatch.setattr(remote, "require_unc", lambda path: None)
    widget.reader = lambda row: dict(snapshot(row["root"]), registered_machines=[saved])
    widget.local_launcher = lambda root, **kwargs: (
        local.append((root, kwargs))
        or dict(job_root=root, started=True, numerical_progress_claimed=False)
    )

    def remote_launcher(root, **kwargs):
        launches.append((root, kwargs))
        return dict(
            job_root=root,
            transport_started=True,
            transport_pid=321,
            worker_startup_verified=False,
            numerical_progress_claimed=False,
        )

    widget.remote_launcher = remote_launcher
    delivered = []
    widget.remote_worker_launch_requested.connect(delivered.append)
    widget.set_context(jobs_provider=lambda: [dict(root=str(root), active=True)])
    wait(app, widget.can_close)
    assert (
        widget.target_combo.count() == 2 and widget.target_combo.currentData()["local"]
    )
    widget.target_combo.setCurrentIndex(1)
    widget.backend_combo.setCurrentIndex(widget.backend_combo.findData("cuda"))
    widget.workers_spin.setValue(2)
    assert widget.start_button.isEnabled() and not launches and not local
    widget.start_selected_machine()
    wait(app, widget.can_close)
    assert len(launches) == 1 and not local and not oldcalls and len(delivered) == 1
    assert launches[0][1]["target"].ssh_alias == "existing-second-alias"
    assert launches[0][1]["backend"] == "cuda" and launches[0][1]["workers"] == 2
    assert (
        not delivered[0]["numerical_progress_claimed"]
        and widget.current["committed"] == 0
    )
    widget.target_combo.setCurrentIndex(0)
    widget.start_selected_machine()
    wait(app, widget.can_close)
    assert len(local) == 1 and len(launches) == 1 and local[0][1]["machine"] == "Main"


def test_legacy_unknown_shell_visible_but_disabled_and_direct_slot_still_refuses(
    panel, app, tmp_path, monkeypatch
):
    widget, _ = panel
    root = tmp_path / "job"
    put(root / "job.json", dict(expected_runs=["ac52"]))
    saved = record(root, facts(tmp_path, shell=""))
    calls = []
    widget.remote_launcher = lambda *a, **k: calls.append((a, k))
    widget.reader = lambda row: dict(snapshot(row["root"]), registered_machines=[saved])
    monkeypatch.setattr(remote, "require_unc", lambda path: None)
    widget.set_context(jobs_provider=lambda: [dict(root=str(root))])
    wait(app, widget.can_close)
    widget.target_combo.setCurrentIndex(1)
    assert (
        not widget.start_button.isEnabled()
        and "PowerShell default shell" in widget.start_button.toolTip()
    )
    widget.start_selected_machine()
    assert calls == [] and "PowerShell default shell" in widget.error_text.toPlainText()


def test_changed_registration_refuses_before_remote_launch(
    panel, app, tmp_path, monkeypatch
):
    widget, _ = panel
    root = tmp_path / "job"
    put(root / "job.json", dict(expected_runs=["ac52"]))
    saved = record(root, facts(tmp_path))
    calls = []
    widget.reader = lambda row: dict(snapshot(row["root"]), registered_machines=[saved])
    widget.remote_launcher = lambda *a, **k: calls.append((a, k))
    monkeypatch.setattr(remote, "require_unc", lambda path: None)
    widget.set_context(jobs_provider=lambda: [dict(root=str(root))])
    wait(app, widget.can_close)
    widget.target_combo.setCurrentIndex(1)
    Path(saved["source"]["path"]).write_bytes(b"external change")
    widget.start_selected_machine()
    assert not calls and "SHA/size differs" in widget.error_text.toPlainText()


def test_resource_remote_choice_cannot_guess_between_explicit_registration_versions(
    panel, app, tmp_path, monkeypatch
):
    widget, _ = panel
    root = tmp_path / "job"
    put(root / "job.json", dict(expected_runs=["ac52"]))
    one = record(root, facts(tmp_path))
    two = record(root, facts(tmp_path), suffix=".second")
    widget.reader = lambda row: dict(
        snapshot(row["root"]), registered_machines=[one, two]
    )
    monkeypatch.setattr(remote, "require_unc", lambda path: None)
    widget.set_context(jobs_provider=lambda: [dict(root=str(root))])
    wait(app, widget.can_close)
    widget._resource_choice(dict(machine="Second", backend="cuda", workers=3))
    assert (
        widget.target_combo.currentData()["local"] and widget.workers_spin.value() == 1
    )
    widget.target_combo.setCurrentIndex(2)
    widget.refresh()
    wait(app, widget.can_close)
    assert widget.target_combo.currentData()["key"] == two["source"]["path"]
