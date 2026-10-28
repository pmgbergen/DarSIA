"""Read-only progress from admitted counts, worker identities and actual clocks."""
from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from darsia.fluidflower.jobs_progress import (
    progress_view, retain_execution, retain_log_reports,
)


START = "2026-01-01T10:00:00Z"
END = "2026-01-01T10:02:00Z"
NOW = datetime(2026, 1, 1, 10, 3, tzinfo=timezone.utc).timestamp()


def test_image_preparation_is_an_operation_not_an_extra_experiment():
    snapshot = dict(work_unit="operations", complete=False, runs=[
        dict(run="setup-image-preparation", operation="image-preparation", state="Queued",
             completed_operations=0, total_operations=1)])
    result = progress_view(snapshot, now=NOW)
    assert result["rows"][0]["task"] == "Image correction"
    assert result["rows"][0]["operation"] == "Image correction"
    assert result["rows"][0]["percent"] != "100%"


def _row(**changes):
    return dict(dict(run="run-a", state="Working", committed=1, total=4,
                     started_utc=START, machine="worker-a", worker="PID 17"), **changes)


@pytest.mark.parametrize("row,expected", [
    (_row(), "Estimated 25.0%"),
    (_row(state="Failed", committed=2), "Estimated 50.0%"),
    (_row(state="Publishing", committed=4), "Estimated <100%"),
    (_row(state="Complete", committed=4), "100%"),
    (_row(total=None), "Unknown"),
    (_row(state="Metadata error"), "Unknown"),
])
def test_percent_uses_actual_counts_and_confirmed_completion(row, expected):
    snapshot = dict(runs=[row], complete=row["state"] == "Complete")
    before = deepcopy(snapshot)
    view = progress_view(snapshot, now=NOW)
    assert view["rows"][0]["percent"] == expected
    assert snapshot == before


def test_elapsed_uses_actual_start_and_fixed_finish_never_creation_or_selection():
    snapshot = dict(created_utc="2025-01-01T00:00:00Z", runs=[_row()])
    assert "Elapsed: 00:03:00" in progress_view(snapshot, now=NOW)["summary"]
    snapshot.update(complete=True)
    snapshot["runs"][0].update(state="Complete", finished_utc=END)
    for now in (NOW, NOW + 600):
        view = progress_view(snapshot, now=now)
        assert view["rows"][0]["elapsed"] == "00:02:00"
        assert "Elapsed: 00:02:00" in view["summary"]
    snapshot["runs"][0].pop("finished_utc")
    assert "Elapsed: Unknown" in progress_view(snapshot, now=NOW)["summary"]
    snapshot["runs"][0].pop("started_utc")
    assert progress_view(snapshot, now=NOW)["rows"][0]["elapsed"] == "Unknown"


def test_assignments_and_stage_counts_never_use_selected_machine_or_fake_weights():
    rows = [_row(progress_event=dict(state="raw_prepared", position=2, total=5)),
            _row(run="run-b", machine="worker-b", worker="PID 28", committed=2)]
    snapshot = dict(runs=rows, selected_target="not-a-worker", master_machine="master")
    view = progress_view(snapshot, now=NOW)
    assert "Estimated 37.5%" in view["summary"]
    assert [row["machine"] for row in view["rows"]] == ["worker-a", "worker-b"]
    assert "2/5 (40.0% of this stage)" in view["rows"][0]["stage"]
    assert "not-a-worker" not in str(view) and "master" not in str(view)


def test_log_times_require_same_reported_worker_and_attempt():
    row = _row(state="Complete", committed=4)
    actual = dict(machine="worker-a-cpu", pid=17, process_create_time="1.5")
    retain_execution(row, owner=dict(token="a" * 32, claimed_utc=START,
        actual_owner=actual), task=dict(backend="cpu"))
    def line(stamp, **event):
        return stamp + " " + json.dumps(event) + "\n"
    right = line(END, kind="completed", run="run-a", attempt_id="a" * 32)
    wrong = line("2026-01-01T10:59:00Z", kind="completed", run="run-a", attempt_id="b" * 32)
    snapshot = dict(runs=[row], logs=[dict(path="worker-a-cpu.log", tail=right + wrong),
                                     dict(path="other-cpu.log", tail=wrong)])
    retain_log_reports(snapshot)
    assert row["finished_utc"] == END and row["started_utc"] == START
    assert row["worker"] == "PID 17"
    assert row["backend"] == "cpu"


def test_setup_completed_and_active_claim_keep_actual_owner_and_reported_clocks(tmp_path):
    from .test_series_setup_results import _published, _read, _put
    from darsia.fluidflower.published_jobs_snapshot import published_snapshot
    case = _published(tmp_path)
    case.owner["claimed_utc"] = START
    _put(case.lease / "owner.json", case.owner)
    log = case.root / "logs" / "fixture-cpu.log"
    log.parent.mkdir()
    log.write_text(END + " " + json.dumps(dict(kind="completed", run=case.task["run"],
        attempt_id=case.token)) + "\n", encoding="utf-8")
    snapshot = published_snapshot(dict(root=str(case.root)), case.job, {}, reader=_read)
    row = next(row for row in snapshot["runs"] if row["run"] == case.task["run"])
    assert row["machine"] == "fixture"
    assert row["worker"] == "PID 123"
    assert row["started_utc"] == START and row["finished_utc"] == END
    projected = next(row for row in progress_view(snapshot, now=NOW)["rows"] if row["task"] == case.task["run"])
    assert projected["percent"] == "100%" and projected["elapsed"] == "00:02:00"


def test_progress_projection_performs_no_io_or_hardware_lookup(monkeypatch):
    import builtins
    from pathlib import Path
    import socket
    def forbidden(*args, **kwargs):
        pytest.fail("A retained progress projection performed IO")
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    monkeypatch.setattr(socket, "gethostname", forbidden)
    assert "25.0%" in progress_view(dict(runs=[_row()]), now=NOW)["summary"]


def test_current_build_retains_reported_workers_partial_stage_and_end_time(tmp_path):
    from darsia.fluidflower.pipeline.current_build_job import current_build_snapshot
    root = tmp_path / "job"
    (root / "logs").mkdir(parents=True)
    owner = dict(machine="actual-cpu", pid=71)
    def line(stamp, event):
        return stamp + " " + json.dumps(event) + "\n"
    log = line(START, dict(kind="progress", run="run-a", state="preparing_sources",
                           actual_worker=owner, completed_count=0))
    log += line(END, dict(kind="progress", run="run-a", state="raw_prepared",
                          position=2, total=5, actual_worker=owner, completed_count=0))
    (root / "logs" / "service.log").write_text(log, encoding="utf-8")
    job = dict(target_machine="selected-only", created_utc="2025-01-01T00:00:00Z",
        series_id="fixture", inputs=[dict(run="run-a", selected_frames=4),
                                     dict(run="run-b", selected_frames=4)])
    (root / "job.json").write_text(json.dumps(job), encoding="utf-8")
    value = current_build_snapshot(dict(root=str(root)), job, {}, reader=lambda p: json.loads(p.read_text()))
    view = progress_view(value, now=NOW)
    assert view["rows"][0]["machine"] == "actual-cpu"
    assert view["rows"][0]["elapsed"] == "00:03:00"
    assert "2/5" in view["rows"][0]["stage"]
    assert view["rows"][1]["machine"] == "Not reported"
    assert "selected-only" not in str(view)
