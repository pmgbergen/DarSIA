"""A refit job built from finished runs is admitted and run by the worker's own I/O path."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

import darsia
from darsia.fluidflower import job_filesystem
from darsia.fluidflower.pipeline import alias_job, refit_jobs, refit_runner, store_runtime
from test_n1_refit import TRUE_BETA, TRUE_GAIN, make_plan, make_rows

RUNS = ["r1", "r2"]
REPOSITORY = Path(darsia.__file__).resolve().parents[2]


def bound(path):
    data = Path(path).read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


@pytest.fixture
def series(tmp_path, monkeypatch):
    """A series with two finished runs' statistics, a baseline and a shared job folder."""
    admitted = job_filesystem.SharedFolderAdmission(True, "smb", "unit transport", "no filesystem qualification")
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda path: admitted)
    root = tmp_path / "series"
    (root / "calibration").mkdir(parents=True)
    (root / "series.toml").write_text('schema_version = 1\nseries_id = "TEST"\n\n[roots]\nseries = "."\n',
                                      encoding="utf-8")
    (tmp_path / "out").mkdir()
    phase_job = tmp_path / "jobs" / "phase-one"
    for row in make_rows(RUNS):
        attempt = root / "results" / "phase-one" / row["run"] / ".attempts" / "a1"
        frame = attempt / "frames" / f"{row['run']}_f{row['frame_index']}"
        frame.mkdir(parents=True)
        (frame / "native_statistics.json.gz").write_bytes(gzip.compress(json.dumps(row).encode()))
        manifest = attempt / "manifest.json"
        if not manifest.exists():
            manifest.write_text("{}", encoding="utf-8")
            completion = phase_job / "results" / row["run"] / "completed.json"
            completion.parent.mkdir(parents=True)
            completion.write_text(json.dumps(dict(complete=True, run=row["run"], manifest=bound(manifest))),
                                  encoding="utf-8")
    rules = make_plan(RUNS)["rules"]
    baseline = root / "calibration" / "baseline.json"
    baseline.write_text(json.dumps(dict(parameters=rules["baseline"])), encoding="utf-8")
    return dict(root=root, phase_job=phase_job, rules=rules, baseline=bound(baseline), tmp=tmp_path)


def prepare(series, loro=True):
    statistics = {run: refit_jobs.statistics_entries(refit_jobs.completed_attempt(series["phase_job"], run))
                  for run in RUNS}
    plans = refit_jobs.build_plans(statistics=statistics, roles=dict(fit=RUNS, heldout=["held"], excluded=[]),
                                   rules=series["rules"], baseline_calibration=series["baseline"], loro=loro)
    job = dict(commit="c" * 40, calibration_sha256=series["baseline"]["sha256"], precision="float32",
               created_utc="2026-10-21T00:00:00+00:00", environment_lock=dict(path="uv.lock", sha256="d" * 64))
    execution = refit_jobs.write_refit_inputs(series["root"] / "calibration" / "refit" / "job1", plans,
                                              job=job, job_id="refit-job1")
    job_root = series["tmp"] / "jobs" / "refit-job1"
    prepared = refit_jobs.prepare_refit_job(execution, job_root, series_manifest=bound(series["root"] / "series.toml"))
    return prepared, job_root


def test_statistics_are_found_from_published_completions(series):
    entries = refit_jobs.statistics_entries(refit_jobs.completed_attempt(series["phase_job"], "r1"))
    assert [entry["frame_index"] for entry in entries] == list(range(1, 24))
    assert entries[0]["case_id"] == "r1_f1" and entries[0]["selected_target_h"] == .5


def test_the_job_has_a_refit_task_and_one_fold_per_fit_run(series):
    prepared, job_root = prepare(series)
    assert [task["run"] for task in prepared["tasks"]] == ["refit", "loro-r1", "loro-r2"]
    job = json.loads((job_root / "job.json").read_text(encoding="utf-8"))
    assert job["pipeline"] == refit_runner.PIPELINE and job["expected_runs"] == ["refit", "loro-r1", "loro-r2"]
    task = json.loads(Path(prepared["tasks"][0]["task"]["path"]).read_text(encoding="utf-8"))
    assert task["pipeline"] == refit_runner.PIPELINE and task["calibration_sha256"] == series["baseline"]["sha256"]
    uses = {use for item in task["input_catalog"]["inputs"] for use in item["uses"]}
    assert {"run.execution_series", "run.series_manifest", "refit.baseline_calibration"} <= uses
    assert sum(use.endswith(".statistics") for use in uses) == 2 * 23
    with pytest.raises(FileExistsError):
        refit_jobs.prepare_refit_job(job_root.parent.parent / "series" / "calibration" / "refit" / "job1" /
                                     "execution.toml", job_root,
                                     series_manifest=bound(series["root"] / "series.toml"))


def run_under_replay(series, task):
    """Admit a refit task with the worker's own replay and run it under that binding."""
    from darsia.fluidflower.pipeline.job_queue import LocalOutputBinding
    from darsia.fluidflower.pipeline.run_io import RunIOBinding
    from darsia.fluidflower.store_map import parse_store_map
    task = dict(task, attempt_id="e" * 32)  # the queue's claim gives each attempt its identity
    stores = parse_store_map(('[machine]\nid = "Main"\n\n[stores.TEST]\n'
                              f"series = '{series['root']}'\nresults = '{series['tmp'] / 'out'}'\n").encode())
    rows = store_runtime.resolve_catalog(task, stores, "TEST")
    verified = store_runtime.verify_inputs(task, rows)
    output = series["tmp"] / "out" / "refit-job1" / task["run"] / ".attempts" / task["attempt_id"]
    admission = job_filesystem.LocalFileAdmission(True, "NTFS", "unit results root", "unit",
                                                  checked_container=str(series["tmp"] / "out"))
    binding = RunIOBinding.replay(task, verified,
                                  LocalOutputBinding("Main", series["tmp"] / "out", output, admission),
                                  repository=REPOSITORY, admitted_commit=task["commit"])
    with binding.activate():
        return refit_runner.run_refit(task, output, "cpu", "float32")


def test_the_worker_replay_admits_a_refit_task_and_the_fit_runs_under_it(series):
    prepared, _ = prepare(series, loro=False)
    task = json.loads(Path(prepared["tasks"][0]["task"]["path"]).read_text(encoding="utf-8"))
    result = run_under_replay(series, task)
    assert result["complete"] and result["frozen_calibration"] is not None
    assert result["parameters"]["gain"] == pytest.approx(TRUE_GAIN, rel=1e-4)
    assert result["parameters"]["beta"] == pytest.approx(TRUE_BETA, rel=1e-3)


def test_status_follows_each_task_and_reads_the_verified_frozen_result(series):
    prepared, job_root = prepare(series, loro=False)
    # This finite inline-completion view exercises the unchanged legacy reader.
    # Genuine alias done/index/producer boundaries are covered by the real
    # original queue publication case in test_common_phase_calibration.
    legacy = json.loads((job_root / "job.json").read_text(encoding="utf8"))
    legacy.pop("format")
    (job_root / "job.json").write_text(json.dumps(legacy), encoding="utf8")
    status = refit_jobs.refit_status(job_root)
    assert status["tasks"]["refit"]["state"] == "waiting" and status["frozen_calibration"] is None
    # A claim moves the queued task and records progress under its lease.
    claimed = job_root / "claimed" / "main" / "refit.json"
    claimed.parent.mkdir(parents=True)
    (job_root / "queue" / "refit.json").rename(claimed)
    for index, step in enumerate(("statistics_read", "stage1_grid")):
        lease = job_root / "leases" / ("refit." + "e" * 32)
        lease.mkdir(exist_ok=True)
        (lease / f"progress_{index:08d}.json").write_text(json.dumps(dict(
            written_unix_ns=index, event=dict(state="refit_progress", step=step))), encoding="utf-8")
    status = refit_jobs.refit_status(job_root)
    assert status["tasks"]["refit"]["state"] == "running" and status["tasks"]["refit"]["step"] == "stage1_grid"
    # The queue publishes the worker's result as the run's completion.
    task = json.loads(claimed.read_text(encoding="utf-8"))
    result = run_under_replay(series, task)
    published = job_root / "results" / "refit" / "completed.json"
    published.write_text(json.dumps(result), encoding="utf-8")
    status = refit_jobs.refit_status(job_root)
    assert status["complete"] and status["parameters"] == result["parameters"]
    document, frozen = refit_jobs.read_frozen_calibration(job_root)
    assert frozen == result["frozen_calibration"] and document["parameters"] == result["parameters"]
    Path(frozen["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after it was published"):
        refit_jobs.read_frozen_calibration(job_root)


def test_start_uses_the_jobs_page_launcher_and_refuses_other_jobs(series, monkeypatch, tmp_path):
    from darsia.fluidflower import jobs_panel
    _, job_root = prepare(series, loro=False)
    calls = []
    monkeypatch.setattr(jobs_panel, "launch_local_worker", lambda root, **kwargs: calls.append((root, kwargs)) or "ok")
    assert refit_jobs.start_refit_job(job_root, machine="main", stores="stores.toml") == "ok"
    assert calls[0][0] == job_root
    assert {key: calls[0][1][key] for key in ("backend", "workers", "machine", "stores")} == \
        dict(backend="cpu", workers=1, machine="main", stores="stores.toml")
    from darsia.fluidflower.job_launch_state import read_launch_requests
    requests = read_launch_requests(job_root)
    assert len(requests) == 1 and requests[0]["released"] is False
    assert calls[0][1]["launch_directory"] == requests[0]["launch_directory"]
    with pytest.raises(ValueError, match="worker launch is already recorded"):
        refit_jobs.start_refit_job(job_root, machine="main", stores="stores.toml")
    assert len(calls) == 1  # Unknown launch state never becomes permission to relaunch.
    other = tmp_path / "other"
    other.mkdir()
    (other / "job.json").write_text(json.dumps(dict(pipeline="native-phase-n1")), encoding="utf-8")
    with pytest.raises(ValueError, match="Not a mass calibration job"):
        refit_jobs.start_refit_job(other, machine="main")


@pytest.mark.parametrize("state", ["running", "failed"])
def test_start_refuses_real_claim_and_failure_without_launching(series, monkeypatch, state):
    from darsia.fluidflower import jobs_panel
    _, root = prepare(series)
    launches = []
    monkeypatch.setattr(jobs_panel, "launch_local_worker", lambda *args, **kwargs: launches.append(args))
    if state == "running":
        claimed = root / "claimed" / "main" / "refit.json"
        claimed.parent.mkdir(parents=True)
        (root / "queue" / "refit.json").rename(claimed)
    elif state == "failed":
        failed = root / "failed" / "refit.json"
        failed.parent.mkdir(exist_ok=True)
        failed.write_text(json.dumps(dict(run="refit", traceback="original failure")), encoding="utf8")
    reason = {"running": "running task", "failed": "failed task"}[state]
    with pytest.raises(ValueError, match=reason):
        refit_jobs.start_refit_job(root, machine="main")
    assert launches == [] and not list((root / "ui" / "launches").glob("*.request.json"))


def test_launcher_failure_preserves_original_request_and_literal_error(series, monkeypatch):
    from darsia.fluidflower import jobs_panel
    from darsia.fluidflower.job_launch_state import read_launch_requests
    _, root = prepare(series, loro=False)
    def fail(*args, **kwargs):
        raise RuntimeError("finite launcher refused before any worker")
    monkeypatch.setattr(jobs_panel, "launch_local_worker", fail)
    with pytest.raises(RuntimeError, match="finite launcher refused"):
        refit_jobs.start_refit_job(root, machine="main")
    rows = read_launch_requests(root)
    assert len(rows) == 1 and not rows[0]["released"]
    assert "finite launcher refused before any worker" in rows[0]["error_traceback"]
    with pytest.raises(ValueError, match="already recorded"):
        refit_jobs.start_refit_job(root, machine="main")
