"""Local/original refit parity on the existing synthetic numerical fixture.

The original side uses the real catalogue/verification/RunIOBinding fixture.
Its phase-one records and filesystem admission are explicitly test metadata,
not evidence of RAW execution, a campaign completion or a qualified share.
The numerical parity tests retain the real fitter, optimizer, grid audits,
selection, anchors, predictions and frozen writer on both entry points.
Lifecycle tests below isolate source identity and host resource observations;
they do not qualify the checkout, available RAM or an actual local campaign.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from threadpoolctl import threadpool_limits

from darsia.fluidflower import execution
from darsia.fluidflower.mass import n1_refit
from darsia.fluidflower.pipeline import local_refit, local_refit_source, refit_runner
from test_refit_jobs import RUNS, bound, prepare, run_under_replay, series  # noqa: F401
from test_refit_runner import write_plan
from test_n1_refit import TRUE_BETA, TRUE_GAIN


def _local_task(configuration, run="refit", **changes):
    return dict(pipeline=refit_runner.PIPELINE, run=run, configuration=configuration,
                backend="cpu", precision="float32", threads=1, commit="c" * 40, **changes)


def _document(descriptor):
    assert bound(descriptor["path"]) == descriptor
    return json.loads(Path(descriptor["path"]).read_bytes())


@pytest.mark.parametrize("run", ["refit", "loro-r1"])
def test_original_bound_runner_and_local_runner_have_exact_numerical_payloads(series, run):
    """Both calls execute the real original numerical functions on equal bytes."""
    prepared, _ = prepare(series, loro=run != "refit")
    task_row = next(row for row in prepared["tasks"] if row["run"] == run)
    original_task = json.loads(Path(task_row["task"]["path"]).read_bytes())
    assert original_task["configuration"]["root_alias"] == "series"
    configuration = bound(series["root"] / original_task["configuration"]["relative_path"])
    before = _document(configuration)
    events = []
    with threadpool_limits(limits=1):
        original = run_under_replay(series, original_task)
        local = refit_runner.run_local_refit(_local_task(configuration, run),
            (series["tmp"] / ("local-" + run)).resolve(), on_progress=events.append)

    # Full fit-byte equality covers parameters, losses, all frame inclusion /
    # omission decisions, source-bound anchors, grids and optimizer records.
    assert set(original["files"]) == set(local["files"])
    assert Path(original["files"]["fit.json"]["path"]).read_bytes() == \
        Path(local["files"]["fit.json"]["path"]).read_bytes()
    original_fit, local_fit = (_document(result["files"]["fit.json"]) for result in (original, local))
    assert original_fit == local_fit
    assert original_fit["parameters"] == local["parameters"]
    assert local["parameters"]["gain"] == pytest.approx(TRUE_GAIN, rel=1e-4)
    assert local["parameters"]["beta"] == pytest.approx(TRUE_BETA, rel=1e-3)
    assert original_fit["anchors_by_run"] and original_fit["frame_decisions"]
    assert set(original_fit["grid_audits"]) == {"stage1", "stage2"}
    result_observations = {"host", "wall_seconds", "files", "frozen_calibration"}
    assert {key: value for key, value in original.items() if key not in result_observations} == \
        {key: value for key, value in local.items() if key not in result_observations}

    if run == "refit":
        original_frozen, local_frozen = (_document(result["frozen_calibration"])
                                         for result in (original, local))
        # The original frozen writer deliberately retains the alias task
        # reference; the local task retains its full physical descriptor.
        # Verify the declared locator and all byte identity fields before
        # normalizing this one producer-owned configuration reference.
        original_configuration = original_frozen["configuration"]
        assert original_configuration == original_task["configuration"]
        assert original_configuration["root_alias"] == "series"
        assert original_configuration["role"] == "configuration"
        assert (series["root"] / original_configuration["relative_path"]).resolve() == \
            Path(configuration["path"]).resolve()
        assert original_configuration["sha256"] == configuration["sha256"]
        assert original_configuration["bytes"] == configuration["size_bytes"]
        assert local_frozen["configuration"] == configuration
        assert original_frozen["code_commit"] == local_frozen["code_commit"] == original_task["commit"] == "c" * 40
        original_frozen["configuration"] = deepcopy(configuration)
        for document, result in ((original_frozen, original), (local_frozen, local)):
            assert document["calibration_files"] == {"fit.json": result["files"]["fit.json"]}
            document.pop("frozen_utc")
            # Only this producer-owned output locator moves. Its SHA and size
            # remain exact, and the full fit payload was compared above.
            document["calibration_files"]["fit.json"].pop("path")
        assert original_frozen == local_frozen
        assert [event["step"] for event in events][-2:] == ["freeze", "complete"]
    else:
        assert original["frozen_calibration"] is local["frozen_calibration"] is None
        assert Path(original["files"]["predictions.json"]["path"]).read_bytes() == \
            Path(local["files"]["predictions.json"]["path"]).read_bytes()
        assert _document(local["files"]["predictions.json"])["heldout_fit_run"] == "r1"
    assert [event["step"] for event in events][:len(RUNS)] == ["statistics_read"] * len(RUNS)
    assert {"stage1", "stage1_grid", "stage2", "stage2_grid"} <= {event["step"] for event in events}
    assert all(event["run"] == run and event["state"] == "refit_progress" for event in events)
    assert _document(configuration) == before


@pytest.mark.parametrize("fault", ["statistics_bytes", "baseline_bytes", "missing_fit_run", "wrong_frame"])
def test_local_runner_refuses_changed_inputs_before_fitting(tmp_path, monkeypatch, fault):
    plan, configuration = write_plan(tmp_path / "source")
    if fault == "statistics_bytes":
        path = Path(plan["statistics"]["r1"][0]["statistics"]["path"])
        path.write_bytes(path.read_bytes() + b"tampered")
    elif fault == "baseline_bytes":
        Path(plan["baseline"]["calibration"]["path"]).write_text("{}", encoding="utf-8")
    else:
        if fault == "missing_fit_run":
            del plan["statistics"]["r3"]
        else:
            plan["statistics"]["r1"][0]["frame_index"] = 999
        Path(configuration["path"]).write_text(json.dumps(plan), encoding="utf-8")
        configuration = bound(configuration["path"])
    monkeypatch.setattr(n1_refit, "fit_candidate", lambda *a, **k: pytest.fail("Refused input reached the fitter"))
    output = (tmp_path / "output").resolve()
    with pytest.raises(ValueError):
        refit_runner.run_local_refit(_local_task(configuration), output)
    assert not (output / "fit.json").exists() and not (output / "frozen_calibration.json").exists()


@pytest.fixture
def local_metadata(series, monkeypatch):
    """Original local preparation with synthetic identity/resource observations."""
    identity = dict(commit="local-source-sha256:" + "d" * 64, calibration_sha256=series["baseline"]["sha256"],
        precision="float32", created_utc="2026-10-23T00:00:00+00:00",
        source_binding=dict(format=local_refit_source.FORMAT, sha256="d" * 64,
                            modules=[], runtime={}, scope="synthetic lifecycle fixture only"))
    original_identity = local_refit.source_identity
    monkeypatch.setattr(local_refit, "source_identity", lambda *a, **k: deepcopy(identity))
    monkeypatch.setattr(execution, "detect_host_resources",
                        lambda: {"memory": {"available_bytes": 1024 ** 4}})
    monkeypatch.setattr(local_refit.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024 ** 4))
    arguments = dict(series_root=series["root"], roles=dict(fit=RUNS, heldout=["held"], excluded=[]),
        rules=series["rules"], baseline=series["baseline"], phase_job=series["phase_job"],
        jobs_root=series["tmp"] / "local-jobs", job_id="local-test", loro=False)
    return SimpleNamespace(series=series, arguments=arguments,
        root=arguments["jobs_root"] / arguments["job_id"], identity=identity,
        original_identity=original_identity)


@pytest.mark.parametrize("fault", ["incomplete", "wrong_run", "manifest_changed"])
def test_local_creation_refuses_partial_or_wrong_phase_one_completion(local_metadata, fault):
    fixture = local_metadata
    path = fixture.series["phase_job"] / "results" / "r1" / "completed.json"
    value = json.loads(path.read_bytes())
    if fault == "incomplete":
        value["complete"] = False
    elif fault == "wrong_run":
        value["run"] = "another-run"
    else:
        Path(value["manifest"]["path"]).write_text('{"changed":true}', encoding="utf-8")
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        local_refit.create_local_refit_job(**fixture.arguments)
    assert not fixture.root.exists()


def test_local_job_is_fresh_and_existing_launch_never_restarts(local_metadata, monkeypatch):
    fixture = local_metadata
    prepared = local_refit.create_local_refit_job(**fixture.arguments)
    assert prepared["numerical_calls_started"] is False
    snapshot = local_refit.read_local_refit_job(fixture.root)
    assert snapshot["status"]["tasks"]["refit"]["state"] == "waiting"
    assert snapshot["frozen"] is None and not snapshot["start"]["blocker"]
    with pytest.raises(FileExistsError):
        local_refit.create_local_refit_job(**fixture.arguments)
    marker = fixture.root / "launch.json"
    marker.write_text('{"test":"already reserved"}', encoding="utf-8")
    before = marker.read_bytes(), marker.stat().st_mtime_ns
    monkeypatch.setattr(refit_runner, "run_local_refit", lambda *a, **k: pytest.fail("Duplicate fit started"))
    with pytest.raises(FileExistsError, match="already been launched"):
        local_refit.run_local_refit_job(fixture.root)
    assert (marker.read_bytes(), marker.stat().st_mtime_ns) == before
    assert not (fixture.root / "results").exists()
    assert local_refit.read_local_refit_job(fixture.root)["start"]["blocker"]


def test_local_job_limits_actual_fit_rows_and_retains_phase_sources(local_metadata):
    fixture = local_metadata
    manifest = fixture.series["root"] / "series.toml"
    manifest.write_text(manifest.read_text() + "\n[calibration.mass.image_selection]\nmax_images_per_run = 3\n")
    phase_sources = fixture.series["root"] / "results" / "phase-one"
    before = {str(path): path.read_bytes() for path in phase_sources.rglob("*") if path.is_file()}
    local_refit.create_local_refit_job(**fixture.arguments)
    job = json.loads((fixture.root / "job.json").read_bytes())
    plan = _document(job["plans"]["refit"])
    assert len(plan["rules"]["target_times_h"]) == 3
    assert plan["image_selection"] == job["image_selection"]
    rows = refit_runner.load_rows(plan, local_refit._reader(fixture.series["root"]))
    rules = n1_refit.RefitRules.from_plan(plan)
    anchors = n1_refit.anchors_by_run(rows, rules)
    stages, decisions = n1_refit.select_frames(rows, rules, stage1_gain=TRUE_GAIN, anchors=anchors)
    assert len(decisions) == 3 * len(RUNS)
    assert all(row["selected_target_h"] in plan["rules"]["target_times_h"]
               for stage in stages.values() for row in stage)
    assert before == {str(path): path.read_bytes() for path in phase_sources.rglob("*") if path.is_file()}


def test_exclusive_launch_creation_closes_the_check_to_create_race(local_metadata, monkeypatch):
    """Deterministic metadata interleaving; no real concurrent fit is launched."""
    fixture = local_metadata
    local_refit.create_local_refit_job(**fixture.arguments)
    original_write = local_refit._write_new
    marker = fixture.root / "launch.json"

    def interleave(path, value):
        if path == marker:
            original_write(marker, {"test": "another caller acquired the exclusive marker"})
        return original_write(path, value)

    monkeypatch.setattr(local_refit, "_write_new", interleave)
    monkeypatch.setattr(refit_runner, "run_local_refit", lambda *a, **k: pytest.fail("Losing caller started fitting"))
    with pytest.raises(FileExistsError):
        local_refit.run_local_refit_job(fixture.root)
    assert json.loads(marker.read_bytes()) == {"test": "another caller acquired the exclusive marker"}
    assert not (fixture.root / "results").exists()


def test_local_failure_is_retained_and_cannot_be_relaunched(local_metadata, monkeypatch):
    fixture = local_metadata
    local_refit.create_local_refit_job(**fixture.arguments)
    events = []

    def failed(task, output, on_progress=None):
        on_progress({"state": "refit_progress", "run": task["run"], "step": "statistics_read"})
        raise RuntimeError("deliberate lifecycle-only failure before fitting")

    monkeypatch.setattr(refit_runner, "run_local_refit", failed)
    with pytest.raises(RuntimeError, match="lifecycle-only"):
        local_refit.run_local_refit_job(fixture.root, on_progress=events.append)
    snapshot = local_refit.read_local_refit_job(fixture.root)
    assert snapshot["status"]["failed"] == ["refit"] and snapshot["status"]["complete"] is False
    assert snapshot["frozen"] is None and snapshot["start"]["blocker"]
    assert [event["step"] for event in events] == ["statistics_read"]
    assert "lifecycle-only" in json.loads((fixture.root / "failed.json").read_bytes())["error"]
    with pytest.raises(FileExistsError):
        local_refit.run_local_refit_job(fixture.root)


@pytest.mark.parametrize("fault", ["statistics", "checkout"])
def test_local_start_rechecks_input_bytes_and_actual_checkout_before_launch(local_metadata, monkeypatch, fault):
    fixture = local_metadata
    local_refit.create_local_refit_job(**fixture.arguments)
    path = fixture.root / "job.json"
    job = json.loads(path.read_bytes())
    if fault == "statistics":
        plan = _document(job["plans"]["refit"])
        statistics = Path(plan["statistics"]["r1"][0]["statistics"]["path"])
        statistics.write_bytes(statistics.read_bytes() + b"changed after preparation")
    else:
        job["repository"] = str(fixture.series["tmp"] / "another-checkout")
        path.write_text(json.dumps(job), encoding="utf-8")
        # The real identity reader owns source-root admission now; do not let
        # the isolated lifecycle identity stub hide that check.
        monkeypatch.setattr(local_refit, "source_identity", fixture.original_identity)
    monkeypatch.setattr(refit_runner, "run_local_refit", lambda *a, **k: pytest.fail("Refused source reached fitting"))
    with pytest.raises(ValueError):
        local_refit.run_local_refit_job(fixture.root)
    assert not (fixture.root / "launch.json").exists()
    assert not (fixture.root / "results").exists()
