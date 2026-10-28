"""GUI Start selects ordinary folders even beside an obsolete Setup policy."""
import json
from pathlib import Path

from darsia_cache import setup_service
from darsia.fluidflower import jobs_panel, store_map, phase_workflow, job_filesystem
from darsia.fluidflower.pipeline import job_queue
from .test_alias_local_stores_launch import FakeProcess
from .test_jobs_panel import app  # noqa: F401


def test_start_uses_selected_folder_without_vhd_service(app, tmp_path, monkeypatch):
    jobs = tmp_path / "Jobs"
    root = jobs / "fixture-job"
    job_queue.write_immutable(root / "job.json", dict(format="darsia-alias-job-v1",
        format_version=1, pipeline="series-setup", series_id="synthetic"))
    path = tmp_path / "stores.toml"
    raw = ('[machine]\nid="fixture"\n[stores.synthetic]\nseries=' + json.dumps(str(tmp_path / "Series"))
        + '\nresults=' + json.dumps(str(jobs)) + '\ncache=' + json.dumps(str(tmp_path / "Cache"))
        + '\n[setup_cache.host_policy]\npath="Q:/obsolete-policy.json"\nsha256="' + "a" * 64 + '"\nsize_bytes=17\n').encode()
    path.write_bytes(raw)
    selected = store_map.parse_store_map(raw, source=dict(path=str(path)))
    monkeypatch.setattr(store_map, "read_store_map", lambda *_args: selected)
    monkeypatch.setattr(phase_workflow, "phase_python_executable", lambda: "science-python")
    monkeypatch.setattr(setup_service, "read_host", lambda *_args:
        (_ for _ in ()).throw(AssertionError("Ordinary folder Start must not prepare a VHD")))
    monkeypatch.setattr(FakeProcess, "instances", [])
    monkeypatch.setattr(FakeProcess, "starts", True)
    monkeypatch.setattr(jobs_panel, "QProcess", FakeProcess)
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda _path:
        job_filesystem.SharedFolderAdmission(True, "smb", "fixture", "private UI fixture"))
    result = jobs_panel.launch_local_worker(root, backend="cpu", workers=2, machine="fixture", stores=path)
    process = FakeProcess.instances[-1]
    command = [process.program, *process.arguments]
    assert command[:4] == ["science-python", "-B", "-m", "darsia_service.prestart"]
    assert "darsia_cache.setup_service" not in command and "--exit-when-idle" in command
    assert command[command.index("--stores") + 1] == str(path)
    assert Path(process.environment.value("NUMBA_CACHE_DIR")) == tmp_path / "Cache" / "fixture" / "numba"
    assert not process.environment.contains("DARSIA_CACHE_BOOTSTRAP_RECEIPT")
    assert result["started"] is True and result["numerical_progress_claimed"] is False
