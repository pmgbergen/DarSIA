"""Metadata-only chemistry choice, through original synthetic refit contracts."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from darsia.fluidflower.pipeline import refit_jobs, refit_runner
from test_refit_jobs import series, bound, RUNS, run_under_replay  # noqa: F401


def _plans(series, **options):
    statistics = {run: refit_jobs.statistics_entries(refit_jobs.completed_attempt(series["phase_job"], run))
                  for run in RUNS}
    return refit_jobs.build_plans(statistics=statistics, roles=dict(fit=RUNS, heldout=["held"], excluded=[]),
        rules=series["rules"], baseline_calibration=series["baseline"], loro=False, **options)


def test_explicit_titration_preserves_every_original_plan_value_and_legacy_absence(series):
    legacy = _plans(series)
    explicit = _plans(series, chemistry_model="titration")
    assert "chemistry_model" not in legacy["refit"]
    removed = deepcopy(explicit)
    assert removed["refit"].pop("chemistry_model") == "titration"
    assert removed == legacy
    refit_runner.validate_plan(legacy["refit"])
    refit_runner.validate_plan(explicit["refit"])


@pytest.mark.parametrize("model", ["simple_flash", "both", "unknown", None])
def test_original_plan_admission_rejects_unsupported_or_explicit_null_chemistry(series, model):
    plan = _plans(series)["refit"]
    with pytest.raises(ValueError, match="chemistry model is not available"):
        refit_runner.validate_plan(dict(plan, chemistry_model=model))
    if model is not None:  # None at the builder API means the unchanged legacy absence.
        with pytest.raises(ValueError, match="chemistry model is not available"):
            _plans(series, chemistry_model=model)


def test_named_titration_job_runs_original_replay_and_freezes_model_identity(series):
    root = series["root"]
    job = dict(commit="c" * 40, calibration_sha256=series["baseline"]["sha256"], precision="float32",
        created_utc="2026-10-21T00:00:00+00:00", environment_lock=dict(path="uv.lock", sha256="d" * 64))
    prepared = refit_jobs.create_refit_job(bound(root / "series.toml"), phase_job=series["phase_job"],
        roles=dict(fit=RUNS, heldout=["held"], excluded=[]), rules=series["rules"],
        baseline_calibration=series["baseline"], job_id="refit-job1", jobs_root=series["tmp"] / "jobs",
        job=job, loro=False, chemistry_model="titration")
    task = json.loads(Path(prepared["tasks"][0]["task"]["path"]).read_bytes())
    result = run_under_replay(series, task)
    frozen = json.loads(Path(result["frozen_calibration"]["path"]).read_bytes())
    assert frozen["chemistry_model"] == "titration"
    assert frozen["parameters"] == result["parameters"]
    assert frozen["colour_paths_refitted"] is False and frozen["overflow_in_objective"] is False
    assert frozen["variant_f"] is False and frozen["gas_branch_fitted"] is False

