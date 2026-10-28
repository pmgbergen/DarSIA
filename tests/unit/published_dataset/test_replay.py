"""The launcher's replay: a moved dataset's prepared tasks run with the production I/O, without a job.

Real here:
- the dataset;
- the store map and its reader;
- ``resolve_catalog`` and ``verify_inputs`` (every input hashed);
- the local results admission;
- ``RunIOBinding`` with its output proof;
- the run records and the comparison.

Three parts are test doubles:
- the role graph: the synthetic configurations are not native ones, so the
  replay's consumer walk mirrors the task's own catalogue;
- the Git code admission (it is tested with the job layer);
- the science: a stand-in writes each frame's result and stored layer as the
  production run would.
"""

import json
from pathlib import Path
import shutil
import tomllib
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from darsia.fluidflower.pipeline import run_io
from darsia.fluidflower.published_dataset import builder, read_only, replay, scan
from darsia.fluidflower.published_dataset.comparison import IDENTICAL
from darsia.fluidflower.published_dataset.job_reader import LAYER_FILE
from darsia.fluidflower.published_dataset.reproduce import Reproduction, Settings, choose_scope, execute
from tests.unit.published_dataset.synthetic import COMMIT, PRODUCER, frame_masses, layer_arrays, make_series

INSTALLED = {"commit": COMMIT, "clean": True, "source": "Git checkout"}
ENVIRONMENT = {"format": "fixture", "python": "3", "file_digests": "recorded", "sha256": "e" * 64}
INVENTORY = {"cpu": {"physical_cores": 4, "logical_cores": 8}, "memory": {"available_bytes": 64 * 1024**3}}


def mirrored_collect(self, config_descriptor, config):
    """The replay's consumer walk for synthetic configurations: the task's own catalogue, use by use."""
    catalog = self.binding.task["input_catalog"]
    for entry in catalog["inputs"]:
        reference = entry["reference"]
        for use in entry["uses"]:
            if use not in self.used:
                self.add({"path": reference["relative_path"], "sha256": reference["sha256"],
                          "size_bytes": reference["bytes"]}, reference["role"], use)
    configuration = self.add(config_descriptor, "configuration", "run.configuration")
    return {"format": catalog["format"], "complete": True, "run": config["run"], "configuration": configuration,
            "inputs": [self.entries[key] for key in sorted(self.entries)],
            "payload_hashes_verified": False, "numerical_calls_started": False}


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """A built R record, downloaded elsewhere; series and expected results unpacked, raw images packed."""
    monkeypatch.setattr(builder, "machine_names", lambda: (
        scan.ScanName("example-build-host", "build machine host name"),))
    series = make_series(tmp_path / "producer-site")
    report = builder.build_record(builder.BuildRequest(series.series, series.job, "R", "S50", tmp_path / "record",
                                                       deny=(), fingerprint=False, peak_memory_bytes=2 * 1024**3))
    assert report.built, report.text()
    folder = tmp_path / "downloads" / "dataset"
    shutil.copytree(tmp_path / "record", folder)
    for name in ("series.zip", "expected.zip"):
        with zipfile.ZipFile(folder / name) as archive:
            archive.extractall(folder)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-app-data"))
    monkeypatch.setattr(run_io._Replay, "collect", mirrored_collect)
    admitted = []
    monkeypatch.setattr(run_io.RunIOBinding, "_admit_native_code", lambda self, native: admitted.append(native))
    (tmp_path / "checkout").mkdir()
    series.dataset, series.checkout, series.admitted = folder, tmp_path / "checkout", admitted
    return series


def stand_in(series, calls, *, on_frame=None):
    """The science: each frame's result and stored layer, written into the admitted attempt folder."""
    def execute(task, output, backend, precision, progress):
        binding = run_io.for_task(task)
        binding.require_output(output)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        assert not any(output.iterdir())
        run = task["run"]
        for position, case in enumerate(task["ordered_case_ids"]):
            progress({"state": "frame_started", "run": run, "case_id": case})
            arrays = layer_arrays(series.runs.index(run), position)
            (output / "frames" / case).mkdir(parents=True)
            (output / "frames" / case / "result.json").write_text(json.dumps({"case_id": case, **frame_masses(arrays)}))
            (output / "preparation" / case).mkdir(parents=True)
            np.savez_compressed(output / "preparation" / case / LAYER_FILE, **arrays)
            calls.append((run, case, task["threads"], task["job_id"], task["attempt_id"]))
            if on_frame is not None:
                on_frame(run, case)
            progress({"state": "frame_completed", "run": run, "case_id": case})
        return {"precision": precision, "backend": backend, "frames": len(task["ordered_case_ids"]),
                "wall_seconds": 0.5}
    return execute


def in_process(execute, checkout):
    """Run each worker in this process, one after the other."""
    def launch(spec_path, log_path):
        code = replay.run_worker(spec_path, execute=execute, checkout=checkout)
        return SimpleNamespace(poll=lambda: code)
    return launch


def runner_for(series, calls, **changes):
    values = dict(launch=in_process(stand_in(series, calls), series.checkout), poll_seconds=0,
                  sleep=lambda seconds: None, installed=INSTALLED, inventory=INVENTORY, environment=ENVIRONMENT)
    values.update(changes)
    return replay.ReplayRunner(**values)


def snapshot(folder):
    return {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in Path(folder).rglob("*")}


def test_a_moved_dataset_reproduces_identically_without_a_job_and_stays_read_only(dataset, tmp_path):
    calls = []
    results = tmp_path / "results"
    reproduction = Reproduction(dataset.dataset, results, runner=runner_for(dataset, calls))
    check = reproduction.open(installed=INSTALLED, fingerprint={})
    assert check.problems == [] and check.code["same"] is True
    scope = choose_scope(check.manifest, "all")
    settings = Settings(workers=2, threads=3)
    assert reproduction.preflight(scope, settings) == []
    before = snapshot(dataset.dataset)
    read_only.protect(dataset.dataset)  # Any write into the dataset would raise.
    try:
        result = execute(reproduction, scope, settings, machine={"cpu": "fixture"})
    finally:
        read_only.unprotect(dataset.dataset)
    assert result["summary"]["dataset"] == IDENTICAL and result["summary"]["reproduced"] is True
    assert snapshot(dataset.dataset) == before
    # No job folder, queue, claim or ledger: only the replay's own folders and the report.
    assert sorted(path.name for path in results.iterdir()) == ["outputs", "replay", "report", "scratch"]
    assert not any(results.rglob("job.json")) and not any(path.name in {"queue", "claimed", "leases"}
                                                          for path in results.rglob("*"))
    assert not any(results.joinpath("scratch").rglob("*.ARW"))  # Each run's raw images left after it.
    # Every frame ran once, with the user's threads, under one reproduction job id.
    assert len(calls) == 6 and {call[2] for call in calls} == {3}
    job_ids = {call[3] for call in calls}
    assert len(job_ids) == 1 and next(iter(job_ids)).startswith("reproduction-")
    assert len(dataset.admitted) == 2
    record = json.loads((results / "replay" / "r1.json").read_text(encoding="utf-8"))
    assert record["route"] == "library replay" and record["reproduction"] is True
    assert record["qualified"] is False and record["production"] is False
    assert record["machine"] == "reproduction" and record["code"] == {"commit": COMMIT, "clean_checkout": True}
    assert record["environment"] == ENVIRONMENT and record["task"]["path"] == "plan/tasks/r1.json"
    attempt = record["attempts"][-1]
    assert attempt["state"] == "done" and attempt["frames"] == 3 and attempt["frames_completed"] == 3
    assert attempt["output"].startswith("outputs/reproduction-") and attempt["output"].endswith(attempt["attempt_id"])
    stores = list((tmp_path / "local-app-data" / "DarSIA" / "reproductions").rglob("stores.toml"))
    assert len(stores) == 1
    roots = tomllib.loads(stores[0].read_text(encoding="utf-8"))["stores"]["syn"]
    assert Path(roots["series"]) == dataset.dataset.resolve() and Path(roots["results"]) == results / "outputs"
    report = json.loads(result["files"]["json"].read_text(encoding="utf-8"))
    assert report["settings"]["route"] == "library replay" and report["verdict"]["dataset"] == IDENTICAL
    html = result["files"]["html"].read_text(encoding="utf-8")
    assert "Route: library replay on this machine, without the job queue" in html
    text = result["files"]["json"].read_text(encoding="utf-8") + html
    assert PRODUCER not in text and str(tmp_path) not in text


def test_the_quick_check_stops_each_run_after_its_frames(dataset, tmp_path):
    calls = []
    results = tmp_path / "results"
    reproduction = Reproduction(dataset.dataset, results, runner=runner_for(dataset, calls))
    check = reproduction.open(installed=INSTALLED, fingerprint={})
    scope = choose_scope(check.manifest, "quick", run="r2", quick_frames=2)
    result = execute(reproduction, scope, Settings(workers=1, threads=2), machine={})
    assert result["summary"]["dataset"] == IDENTICAL
    assert [call[:2] for call in calls] == [("r2", "r2_f1"), ("r2", "r2_f2")]
    attempt = json.loads((results / "replay" / "r2.json").read_text(encoding="utf-8"))["attempts"][-1]
    assert attempt["state"] == "done" and attempt["quick_check"] is True and attempt["frames"] == 2
    assert not (results / "replay" / "r1.json").exists()


def test_pause_stops_at_a_frame_boundary_and_resume_replays_the_run_afresh(dataset, tmp_path):
    calls, paused = [], []
    results = tmp_path / "results"

    def pause_once(run, case):
        if not paused:
            paused.append(case)
            replay.publish_control(results, "pause")

    settings = Settings(workers=1, threads=2)
    runner = runner_for(dataset, calls)
    runner.launch = in_process(stand_in(dataset, calls, on_frame=pause_once), dataset.checkout)
    runner.sleep = lambda seconds: runner.resume(settings) if replay.read_control(results) == "pause" else None
    reproduction = Reproduction(dataset.dataset, results, runner=runner)
    check = reproduction.open(installed=INSTALLED, fingerprint={})
    result = execute(reproduction, choose_scope(check.manifest, "all"), settings, machine={})
    assert result["summary"]["dataset"] == IDENTICAL
    attempts = json.loads((results / "replay" / "r1.json").read_text(encoding="utf-8"))["attempts"]
    assert [attempt["state"] for attempt in attempts] == ["paused", "done"]
    assert attempts[0]["frames_completed"] == 1 and attempts[0]["attempt_id"] != attempts[1]["attempt_id"]
    assert (results / attempts[0]["output"]).is_dir()  # The paused attempt folder is kept.
    assert [call[:2] for call in calls if call[0] == "r1"] == [("r1", "r1_f1"), ("r1", "r1_f1"), ("r1", "r1_f2"),
                                                                ("r1", "r1_f3")]


def test_a_worker_that_ends_early_fails_its_runs_clearly(dataset, tmp_path):
    results = tmp_path / "results"
    runner = runner_for(dataset, [], launch=lambda spec, log: SimpleNamespace(poll=lambda: 1))
    reproduction = Reproduction(dataset.dataset, results, runner=runner)
    check = reproduction.open(installed=INSTALLED, fingerprint={})
    with pytest.raises(RuntimeError, match="Runs failed: r1, r2"):
        execute(reproduction, choose_scope(check.manifest, "all"), Settings(workers=1, threads=2), machine={})
    attempt = json.loads((results / "replay" / "r1.json").read_text(encoding="utf-8"))["attempts"][-1]
    assert attempt["state"] == "failed" and attempt["error"] == "No worker replayed the run"


def test_a_failing_run_is_recorded_with_its_error(dataset, tmp_path):
    def broken(task, output, backend, precision, progress):
        raise ValueError("numerical stand-in failure")

    results = tmp_path / "results"
    runner = runner_for(dataset, [], launch=in_process(broken, dataset.checkout))
    reproduction = Reproduction(dataset.dataset, results, runner=runner)
    check = reproduction.open(installed=INSTALLED, fingerprint={})
    with pytest.raises(RuntimeError, match="Runs failed"):
        execute(reproduction, choose_scope(check.manifest, "run", run="r1"), Settings(workers=1, threads=2),
                machine={})
    attempt = json.loads((results / "replay" / "r1.json").read_text(encoding="utf-8"))["attempts"][-1]
    assert attempt["state"] == "failed" and attempt["error"] == "ValueError: numerical stand-in failure"


def test_preflight_refuses_changed_tasks_checkouts_memory_and_disk(dataset, tmp_path, monkeypatch):
    results = tmp_path / "results"
    reproduction = Reproduction(dataset.dataset, results, runner=runner_for(dataset, []))
    check = reproduction.open(installed=INSTALLED, fingerprint={})
    scope, settings = choose_scope(check.manifest, "all"), Settings(workers=2, threads=2)
    small = {"memory": {"available_bytes": 3 * 1024**3}}
    reproduction.runner.inventory = small
    assert reproduction.preflight(scope, settings) == [
        "Memory: 2 worker(s) × 2.0 GiB measured on the reference machine + 1 GiB reserve = 5.0 GiB, but 3.0 GiB "
        "is available now. Choose fewer workers or close other programs."]
    reproduction.runner.inventory = INVENTORY
    monkeypatch.setattr("darsia.fluidflower.storage_profile.estimate_prepared_job", lambda profile, tasks, configs: {
        "complete": True, "runs": [{"run": task["run"], "frames": 3, "needed_bytes": 10**18} for task in tasks]})
    assert reproduction.preflight(scope, settings)[0].startswith("Disk: the job layer estimates 2000000000")
    monkeypatch.undo()
    assert replay.checkout_problems({"commit": None, "source": "no Git checkout (an installed copy)"}, COMMIT) == [
        "The replay needs a Git clone of DarSIA at the published commit (no Git checkout (an installed copy))"]
    assert replay.checkout_problems({"commit": "e" * 40, "clean": True}, COMMIT) == [
        f"The DarSIA checkout is at {'e' * 40}, not at {COMMIT}"]
    assert "local changes" in replay.checkout_problems({"commit": COMMIT, "clean": False}, COMMIT)[0]
    task = dataset.dataset / "plan" / "tasks" / "r1.json"
    task.write_bytes(task.read_bytes() + b" ")
    problems = Reproduction(dataset.dataset, results, runner=runner_for(dataset, [])).open(
        installed=INSTALLED, fingerprint={}) and reproduction.preflight(scope, settings)
    assert problems == ["The dataset's prepared tasks: The prepared task of run r1 differs from the manifest"]


def test_helpers_bind_roots_tasks_and_pipelines(dataset, tmp_path, monkeypatch):
    text = replay.store_map_text("SYN", {"series": Path("C:/data/set"), "raw": Path("C:/data/set/raw"),
                                         "results": Path("C:/results/outputs")})
    assert '[stores."syn"]' in text and 'id = "reproduction"' in text
    task = {"run": "r1", "attempt_id": "a" * 32, "job_id": "job-1", "threads": 2, "commit": COMMIT, "x": 1}
    assert replay.replay_task(task, attempt_id="b" * 32, job_id="reproduction-1", threads=4) == dict(
        task, attempt_id="b" * 32, job_id="reproduction-1", threads=4)
    assert replay.replay_task(task, attempt_id="b" * 32, job_id="j", commit="e" * 40)["commit"] == "e" * 40
    from darsia.fluidflower.pipeline import lattice_runner, runner

    assert replay.science("native-phase-n1") is runner.run_series
    assert replay.science("lattice-phase-n1") is lattice_runner.run_lattice_series
    with pytest.raises(ValueError, match="cannot replay"):
        replay.science("n1")
    check = SimpleNamespace(root=dataset.dataset, manifest=json.loads(
        (dataset.dataset / "dataset_manifest.json").read_text(encoding="utf-8")))
    tasks = replay.published_tasks(check)
    assert set(tasks) == {"r1", "r2"} and tasks["r1"]["pipeline"] == "native-phase-n1"
    roots = replay.raw_roots(check.root, check.manifest, tmp_path / "scratch")
    assert roots == {"raw": tmp_path / "scratch" / "raw"}
    sizes = [row["archive"]["size_bytes"] for row in check.manifest["runs"]]
    assert replay.scratch_bytes(check, SimpleNamespace(runs=["r1"], frames={}), 2) == sizes[0]
    assert replay.scratch_bytes(check, SimpleNamespace(runs=["r1", "r2"], frames={}), 1) == max(sizes)
    assert replay.scratch_bytes(check, SimpleNamespace(runs=["r1", "r2"], frames={}), 2) == sum(sizes)
    unpacked = replay.unpack_run(check.root, check.manifest, "r1", roots)
    assert len(unpacked) == 4 and all(path.is_relative_to(tmp_path / "scratch") for path in unpacked)
    assert replay.unpack_run(check.root, check.manifest, "r1", roots) == []  # Already here: read in place.
    replay.remove_unpacked(check.manifest, "r1", roots, tmp_path / "scratch")
    assert not any((tmp_path / "scratch").rglob("*.ARW"))
    assert replay.read_control(tmp_path) == "run"
    replay.publish_control(tmp_path, "pause")
    assert replay.read_control(tmp_path) == "pause"
    with pytest.raises(ValueError):
        replay.publish_control(tmp_path, "stop")
    assert replay.run_state(tmp_path, "r1") == "queued"
    started = []
    monkeypatch.setattr(replay, "run_worker", lambda path: started.append(path) or 0)
    assert replay.main(["--worker", str(tmp_path / "worker.json")]) == 0
    assert started == [tmp_path / "worker.json"]
    assert sorted(replay.environment_identity()) == ["file_digests", "format", "python", "sha256"]
