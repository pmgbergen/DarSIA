import json
from pathlib import Path
import sys
import time

import pytest

from darsia_cache.adapters import _worker_metadata
from darsia_cache.policy import CachePolicyError, SETUP_CONTROL_MAX_BYTES
from darsia_cache.setup_qualification import setup_control_metadata, _control_identity
from darsia_cache.setup_service import source_pin, record_preparation_failure
from darsia.fluidflower.pipeline.job_inputs import MAX_METADATA_BYTES
from darsia.fluidflower import job_launch_state
from tests.unit.test_setup_service_qualification import protocol


def test_actual_sized_setup_control_uses_worker_catalogue_bound_but_publications_stay_small(tmp_path):
    path = tmp_path / "apply-operation.json"
    path.write_text(json.dumps(dict(payload="x" * (17 * 1024**2))), encoding="utf-8")
    pin = source_pin(path)
    assert SETUP_CONTROL_MAX_BYTES == MAX_METADATA_BYTES
    assert len(setup_control_metadata(pin)["payload"]) == 17 * 1024**2
    with pytest.raises(CachePolicyError, match="unsafe/unbounded"):
        _worker_metadata(path, expected=pin)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(CachePolicyError, match="bytes differ"):
        setup_control_metadata(pin)


def test_setup_plan_descriptor_uses_same_large_control_bound():
    first, _, _ = protocol()
    size = 17 * 1024**2
    first.task["setup_plan"]["bytes"] = size
    first.job["setup_plan"]["bytes"] = size
    first.configuration["setup_plan"]["size_bytes"] = size
    for row in first.task["input_catalog"]["inputs"]:
        if "setup.plan" in row["uses"]:
            row["reference"]["bytes"] = size
    assert _control_identity(first.task, first.job, first.configuration, first.provider["scope"], "apply")


def test_preparation_failure_releases_only_its_exact_unstarted_request(tmp_path):
    root = tmp_path / "job"
    root.mkdir()
    (root / "job.json").write_text(json.dumps(dict(format="darsia-alias-job-v1", format_version=1)), encoding="utf-8")
    request = job_launch_state.begin_launch_request(root, machine="fixture", backend="cpu", workers=6, kind="local")
    launch = Path(request["launch_directory"])
    launch.mkdir()
    record_preparation_failure([sys.executable, "--job", str(root)], launch, "literal preparation failure")
    row = job_launch_state.read_launch_requests(root)[0]
    assert row["released"] and row["error_traceback"] == "literal preparation failure"
    failure = launch / "preparation-failure.json"
    value = json.loads(failure.read_text())
    value["worker_started"] = True
    failure.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        job_launch_state.read_launch_requests(root)


def test_installed_qt_detached_instance_overload_launches_only_a_private_noop(tmp_path):
    from PySide6.QtCore import QProcess
    from darsia.fluidflower.jobs_panel import _start_detached
    target = tmp_path / "noop-finished.json"
    process = QProcess()
    selected = Path(sys.executable).with_name("pythonw.exe")
    process.setProgram(str(selected if selected.is_file() else Path(sys.executable)))
    process.setArguments(["-B", "-c", "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text('noop completed')", str(target)])
    process.setWorkingDirectory(str(tmp_path))
    process.setStandardOutputFile(str(tmp_path / "noop.log"))
    started, pid = _start_detached(process)
    assert started and (pid is None or pid > 0)
    deadline = time.monotonic() + 5
    while not target.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert target.read_text() == "noop completed"
