"""The real GUI launch boundary, with a fake QProcess and no actual worker."""
from types import SimpleNamespace
import sys
import pytest

from darsia.fluidflower import jobs_panel as ui, store_map, job_filesystem
from darsia.fluidflower.pipeline import job_queue
from darsia_cache import setup_service, adapters
from .test_alias_local_stores_launch import FakeProcess
from .test_jobs_panel import app


@pytest.mark.parametrize("launcher_python", ["selected-python", sys.executable])
def test_gui_start_prepares_selected_setup_cache_in_background_and_preserves_six_workers(app, tmp_path, monkeypatch, launcher_python):
    root = tmp_path / "shared-job"
    job_queue.write_immutable(root / "job.json", dict(format="darsia-alias-job-v1", format_version=1,
        pipeline="series-setup", series_id="SERIES"))
    chosen = str(tmp_path / "chosen-stores.toml")
    selected = SimpleNamespace(document=dict(setup_cache=dict(host_policy=dict(TESTONLY="pin"))),
        machine_id="FIXTURE", source=dict(path=chosen), roots=lambda series: {"TESTONLY": series})
    monkeypatch.setattr(store_map, "read_store_map", lambda path: selected)
    seen = []
    monkeypatch.setattr(setup_service, "read_host", lambda doc: seen.append(doc))
    from darsia.fluidflower import phase_workflow
    monkeypatch.setattr(phase_workflow, "phase_python_executable", lambda: "selected-python")
    monkeypatch.setattr(adapters, "independent_worker_environment",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("No incomplete legacy policy admission")))
    monkeypatch.setattr(FakeProcess, "instances", [])
    monkeypatch.setattr(FakeProcess, "starts", True)
    monkeypatch.setattr(ui, "QProcess", FakeProcess)
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda path:
        job_filesystem.SharedFolderAdmission(True, "smb", "fixture", "synthetic transport"))
    record = ui.launch_local_worker(root, backend="cpu", workers=6, machine="fixture",
        python=launcher_python, cwd=tmp_path / "checkout", stores=chosen)
    process = FakeProcess.instances[-1]
    command = [process.program] + process.arguments
    assert command[:4] == ["selected-python", "-B", "-m", "darsia_cache.setup_service"]
    worker = command[command.index("--") + 1:]
    assert worker[:4] == ["selected-python", "-B", "-m", "darsia.fluidflower.worker"]
    assert worker[worker.index("--workers") + 1] == "6"
    assert worker[worker.index("--stores") + 1] == chosen
    assert "--exit-when-idle" in worker and "--watchdog" in worker
    assert "darsia_service.prestart" not in command  # No automatic repeat of a released Start.
    assert len(seen) == 1 and record["started"] is True
    from darsia.fluidflower.job_control import publish_control, read_control
    publish_control(root, "stop", machine="fixture", expected_previous=None)
    assert read_control(root)["action"] == "stop"
