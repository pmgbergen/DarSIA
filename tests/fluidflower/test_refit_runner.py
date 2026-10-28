"""The refit worker pipeline: plan admission, catalogue, dispatch and one run each kind."""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import darsia
from darsia.fluidflower.pipeline import job_inputs, refit_runner, run_io
from test_n1_refit import TRUE_BETA, TRUE_GAIN, make_plan, make_rows

RUNS = ["r1", "r2", "r3"]


def descriptor(path):
    data = Path(path).read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def write_plan(root, run="refit", fold=None, **plan_changes):
    """Statistics files, a baseline calibration and a plan for the synthetic runs."""
    root = Path(root)
    (root / "stats").mkdir(parents=True, exist_ok=True)
    statistics = {}
    for row in make_rows(RUNS):
        path = root / "stats" / f"{row['run']}_f{row['frame_index']}.json.gz"
        if not path.exists():
            path.write_bytes(gzip.compress(json.dumps(row).encode()))
        statistics.setdefault(row["run"], []).append(dict(
            frame_index=row["frame_index"], selected_target_h=row["selected_target_h"], statistics=descriptor(path)))
    plan = make_plan(RUNS)
    baseline_path = root / "baseline.json"
    baseline_path.write_text(json.dumps(dict(parameters=plan["rules"]["baseline"])), encoding="utf-8")
    plan.update(format=refit_runner.PLAN_FORMAT, run=run, fold=fold, statistics=statistics,
                baseline=dict(calibration=descriptor(baseline_path)), **plan_changes)
    path = root / (run + "_plan.json")
    path.write_text(json.dumps(plan), encoding="utf-8")
    return plan, descriptor(path)


def run(monkeypatch, tmp_path, run_name, fold):
    plan, configuration = write_plan(tmp_path, run_name, fold)
    task = dict(pipeline=refit_runner.PIPELINE, precision="float32", run=run_name, commit="c" * 40,
                configuration=configuration)
    binding = SimpleNamespace(configuration_descriptor=configuration, require_output=lambda output: None)
    monkeypatch.setattr(run_io, "for_task", lambda value: binding if value is task else None)
    events = []
    result = refit_runner.run_refit(task, tmp_path / ("attempt_" + run_name), "cpu", "float32", events.append)
    return result, events


def test_the_global_task_fits_and_freezes_with_sha256(monkeypatch, tmp_path):
    result, events = run(monkeypatch, tmp_path, "refit", None)
    assert result["complete"] and not result["stop_required"] and result["fold"] is None
    assert result["parameters"]["gain"] == pytest.approx(TRUE_GAIN, rel=1e-4)
    assert result["parameters"]["beta"] == pytest.approx(TRUE_BETA, rel=1e-3)
    frozen_path = Path(result["frozen_calibration"]["path"])
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest() == result["frozen_calibration"]["sha256"]
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert frozen["format"] == refit_runner.FREEZE_FORMAT and frozen["frozen"] is True
    assert frozen["parameters"] == result["parameters"] and frozen["calibration_runs"] == RUNS
    fit = Path(frozen["calibration_files"]["fit.json"]["path"])
    assert hashlib.sha256(fit.read_bytes()).hexdigest() == frozen["calibration_files"]["fit.json"]["sha256"]
    assert frozen["colour_paths_refitted"] is False and frozen["variant_f"] is False
    steps = [event["step"] for event in events]
    assert steps[:3] == ["statistics_read"] * 3 and steps[-2:] == ["freeze", "complete"]
    assert all(event["state"] == "refit_progress" and event["run"] == "refit" for event in events)


def test_a_fold_task_predicts_its_run_and_freezes_nothing(monkeypatch, tmp_path):
    result, _ = run(monkeypatch, tmp_path, "loro-r2", "r2")
    assert result["fold"] == "r2" and result["frozen_calibration"] is None
    predictions = json.loads(Path(result["files"]["predictions.json"]["path"]).read_text(encoding="utf-8"))
    assert predictions["heldout_fit_run"] == "r2"
    assert {p["run"] for p in predictions["predictions"]} == {"r2"} and len(predictions["predictions"]) == 23


def test_the_plan_is_refused_when_run_fold_or_statistics_disagree(tmp_path):
    plan, _ = write_plan(tmp_path)
    refit_runner.validate_plan(plan)
    with pytest.raises(ValueError, match="run and fold"):
        refit_runner.validate_plan(dict(plan, fold="r1"))
    with pytest.raises(ValueError, match="exactly its fit runs"):
        refit_runner.validate_plan(dict(plan, statistics={k: v for k, v in plan["statistics"].items() if k != "r3"}))
    with pytest.raises(ValueError, match="'refit' or 'loro-<fit run>'"):
        refit_runner.fold_of("other")


def test_a_statistics_row_that_differs_from_its_plan_entry_is_refused(monkeypatch, tmp_path):
    plan, configuration = write_plan(tmp_path)
    entry = plan["statistics"]["r1"][0]
    entry["frame_index"] = 99
    Path(configuration["path"]).write_text(json.dumps(plan), encoding="utf-8")
    configuration = descriptor(configuration["path"])
    task = dict(pipeline=refit_runner.PIPELINE, precision="float32", run="refit", commit="c" * 40,
                configuration=configuration)
    binding = SimpleNamespace(configuration_descriptor=configuration, require_output=lambda output: None)
    monkeypatch.setattr(run_io, "for_task", lambda value: binding)
    with pytest.raises(ValueError, match="differs from the plan"):
        refit_runner.run_refit(task, tmp_path / "attempt", "cpu", "float32")


def test_the_catalogue_names_the_baseline_and_every_statistics_file(tmp_path):
    plan, configuration = write_plan(tmp_path)
    catalog = job_inputs.InputCatalog({"series": str(tmp_path)}).collect(configuration, plan)
    uses = {use: item["reference"] for item in catalog["inputs"] for use in item["uses"]}
    assert uses["refit.baseline_calibration"]["relative_path"] == "baseline.json"
    statistics = [use for use in uses if use.endswith(".statistics")]
    assert len(statistics) == 3 * 23 and "refit.r2.frame.17.statistics" in statistics
    assert all(uses[use]["role"] == "calibration" for use in statistics)
    assert catalog["run"] == "refit" and catalog["configuration"]["relative_path"] == "refit_plan.json"
    job_inputs.validate_catalog(catalog)


def test_the_worker_dispatches_the_refit_pipeline_lazily():
    source = Path(darsia.__file__).parent / "fluidflower" / "pipeline" / "worker.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    branches = [node for node in ast.walk(tree) if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and any(isinstance(c, ast.Constant) and c.value == refit_runner.PIPELINE
                        for c in node.test.comparators)]
    assert len(branches) == 1
    imports = [node for node in ast.walk(branches[0]) if isinstance(node, ast.ImportFrom)]
    assert [(node.module, [alias.name for alias in node.names]) for node in imports][:1] == \
        [("refit_runner", ["run_refit"])]
    top_level = {node.module for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert "refit_runner" not in top_level
