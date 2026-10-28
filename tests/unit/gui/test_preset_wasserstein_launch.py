"""Synthetic argv/metadata checks only; source phase UNRUN, no real launch."""
from pathlib import Path

import pytest

from darsia.fluidflower import jobs_panel as ui, remote_worker_launch as remote
from darsia.fluidflower import portable_job_launch, store_map
from darsia.fluidflower.pipeline import job_queue as jobs
from darsia_cache import adapters
from darsia_service import prestart
from .test_alias_local_stores_launch import FakeProcess
from .test_jobs_panel import app


def _job(tmp_path):
    root = tmp_path / "shared-folder"
    jobs.write_immutable(root / "job.json", dict(format="darsia-alias-job-v1", format_version=1,
        pipeline="preset-wasserstein", job_id="comparison"))
    return root


def test_missing_alias_version_refuses_portable_metadata_reference(tmp_path):
    root = tmp_path / "unversioned-job"
    jobs.write_immutable(root / "job.json", dict(format="darsia-alias-job-v1",
        pipeline="preset-wasserstein", job_id="comparison"))
    # Keep the original own-job metadata/version guard in this negative case.
    with pytest.raises(ValueError, match="Unknown alias job version"):
        ui._job_reference(root, root / "job.json")


def _target(ssh=r"C:\Windows\System32\OpenSSH\ssh.exe"):
    return remote.RemoteWorkerTarget(machine="second", ssh_alias="second",
        python=r"C:\Existing\python.exe", checkout=r"C:\Existing\repo",
        ssh_executable=ssh, default_shell="windows-powershell")


def _worker(arguments):
    return arguments[arguments.index("--") + 1:]


def _assert_finite(command):
    assert command[1:4] == ["-B", "-m", "darsia.fluidflower.worker"]
    assert command.count("--once") == 1
    assert command[command.index("--workers") + 1] == "1"
    assert command[command.index("--backend") + 1] == "cpu"
    assert "--once" in prestart.retry_refusal(command, {})


def test_preset_remote_preview_uses_one_finite_original_service():
    launch = remote.build_remote_worker_launch(r"\\NAS\jobs\comparison", target=_target(),
        backend="cpu", workers=1, pipeline="preset-wasserstein")
    _assert_finite(list(launch.worker_argv))
    assert "--stores" not in launch.worker_argv, "Keep the remote machine's own default map"


@pytest.mark.parametrize("pipeline", [None, "native-phase-n1", "series-setup"])
def test_existing_remote_routes_keep_worker_count_and_loop(pipeline):
    launch = remote.build_remote_worker_launch(r"\\NAS\jobs\existing", target=_target(),
        backend="cpu", workers=3, pipeline=pipeline)
    assert "--once" not in launch.worker_argv
    assert launch.worker_argv[launch.worker_argv.index("--workers") + 1] == "3"


@pytest.mark.parametrize("backend,workers", [("cuda", 1), ("cpu", 2)])
def test_invalid_preset_choices_refuse_before_request_or_launcher(app, tmp_path, monkeypatch, backend, workers):
    root = _job(tmp_path)
    monkeypatch.setattr(ui, "_admit_gui_job_root", lambda root: None)
    monkeypatch.setattr(ui, "QProcess", lambda: pytest.fail("No process may be constructed"))
    monkeypatch.setattr(store_map, "read_store_map", lambda value: pytest.fail("Choice must refuse before map admission"))
    monkeypatch.setattr(remote, "require_unc", lambda value: None)
    with pytest.raises(ValueError, match="exactly one worker"):
        ui.launch_local_worker(root, backend=backend, workers=workers, machine="fixture")
    with pytest.raises(ValueError, match="exactly one worker"):
        remote.build_remote_worker_launch(r"\\NAS\jobs\comparison", target=_target(),
            backend=backend, workers=workers, pipeline="preset-wasserstein")
    with pytest.raises(ValueError, match="exactly one worker"):
        remote.launch_remote_worker(root, target=_target(), backend=backend, workers=workers,
            process_factory=lambda: pytest.fail("No SSH process may be constructed"))
    with pytest.raises(ValueError, match="exactly one worker"):
        portable_job_launch.begin(root, machine="fixture", backend=backend, workers=workers, kind="local")
    assert not (root / "ui").exists()


def test_actual_local_metadata_selects_once_and_keeps_local_map_private(app, tmp_path, monkeypatch):
    root = _job(tmp_path)
    selected = tmp_path / "private-stores.toml"
    monkeypatch.setattr(ui, "_admit_gui_job_root", lambda root: None)
    monkeypatch.setattr(store_map, "read_store_map", lambda value: object())
    monkeypatch.setattr(adapters, "independent_worker_environment", lambda *args, **kwargs: {})
    monkeypatch.setattr(FakeProcess, "instances", [])
    monkeypatch.setattr(FakeProcess, "starts", True)
    monkeypatch.setattr(ui, "QProcess", FakeProcess)
    receipt = ui.launch_local_worker(root, backend="cpu", workers=1, machine="fixture",
        python="fixture-python", cwd=tmp_path, stores=selected)
    command = _worker(FakeProcess.instances[-1].arguments)
    _assert_finite(command)
    assert command[command.index("--stores") + 1] == str(selected)
    assert receipt["machine"] == "fixture" and receipt["workers"] == 1
    assert receipt["numerical_progress_claimed"] is False
    assert str(selected) not in (Path(FakeProcess.instances[-1].console).parent / "launch.json").read_text()


def test_actual_shared_metadata_selects_once_without_local_store_path(app, tmp_path, monkeypatch):
    root = _job(tmp_path)
    ssh = tmp_path / "ssh.exe"
    ssh.write_text("Synthetic non-executable fixture")
    monkeypatch.setattr(remote, "require_unc", lambda value: None)
    monkeypatch.setattr(remote, "_local_windows_path", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(FakeProcess, "instances", [])
    monkeypatch.setattr(FakeProcess, "starts", True)
    receipt = remote.launch_remote_worker(root, target=_target(str(ssh)), backend="cpu", workers=1,
        process_factory=FakeProcess)
    _assert_finite(receipt["worker_argv"])
    assert "--stores" not in receipt["worker_argv"]
    assert receipt["machine"] == "second"
    assert receipt["worker_startup_verified"] is False and receipt["numerical_progress_claimed"] is False
