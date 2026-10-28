"""Real F numerical fit/freeze on finite rows, independent of actual experiment jobs."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from darsia.fluidflower.mass import n1_refit, overflow_statistics
from darsia.fluidflower.mass.n1_pipeline import _write_json, asset
from darsia.fluidflower.pipeline import local_overflow_refit as subject, local_refit_source
from test_n1_refit import make_plan, make_rows, TRUE_GAIN
from test_local_refit import local_metadata
from test_refit_jobs import series


def source(tmp_path):
    plan = make_plan(["r1", "r2"], variant_f=True)
    plan["runs"]["descriptive"] = []
    plan["baseline"] = {"fixture": "finite numerical baseline"}
    plan.update(format=subject.PLAN_FORMAT, run="refit", fold=None, statistics={})
    for row in make_rows(["r1", "r2"]):
        row["contract"] = overflow_statistics.STATISTICS_FORMAT
        row["overflow_steering"] = dict(format="darsia-overflow-statistics-derivation-v1")
        # An explicit finite F gas/water change must reach the gain objective.
        row["main_gas_mass_g"] *= .5
        folder = tmp_path / row["run"]
        folder.mkdir(exist_ok=True)
        path = folder / (str(row["frame_index"]) + ".json.gz")
        path.write_bytes(gzip.compress(json.dumps(row).encode(), mtime=0))
        plan["statistics"].setdefault(row["run"], []).append(dict(frame_index=row["frame_index"], statistics=asset(path)))
    plan["overflow_preparation"] = _write_json(tmp_path / "preparation.json", dict(
        format=subject.PREPARATION_FORMAT, statistics=plan["statistics"], production_admitted=False,
        pending_nonfit_runs=["held"]))
    bound = _write_json(tmp_path / "plan.json", plan)
    task = dict(run="refit", configuration=bound, commit="local-source-sha256:" + "a" * 64)
    return plan, task


def test_actual_overflow_adjusted_mass_reaches_the_gain_fit_and_distinct_freeze(tmp_path):
    plan, task = source(tmp_path)
    result = subject.run_fit(task, plan, tmp_path / "result")
    assert not result["stop_required"]
    assert result["parameters"]["gain"] > TRUE_GAIN
    frozen = json.loads(Path(result["frozen_calibration"]["path"]).read_bytes())
    assert frozen["format"] == subject.FROZEN_FORMAT
    assert frozen["variant_f"] is True and frozen["configuration"] == task["configuration"]
    assert frozen["free_gas_target"].startswith("reviewed overflow")
    assert frozen["production_admitted"] is False and frozen["pending_nonfit_runs"] == ["held"]
    with pytest.raises(ValueError, match="Using overflow"):
        n1_refit.fit_candidate(make_rows(["r1", "r2"]), n1_refit.RefitRules.from_plan(plan))


@pytest.mark.parametrize("fault", ["row-bytes", "preparation-bytes", "changed-plan"])
def test_missing_or_changed_actual_f_inputs_do_not_reach_the_optimizer(tmp_path, monkeypatch, fault):
    plan, task = source(tmp_path)
    if fault == "row-bytes":
        Path(plan["statistics"]["r1"][0]["statistics"]["path"]).write_bytes(b"changed")
    elif fault == "preparation-bytes":
        Path(plan["overflow_preparation"]["path"]).write_bytes(b"changed")
    else:
        plan["rules"]["stage2_start_h"] += .5
    monkeypatch.setattr(subject.overflow_refit, "fit_adjusted_rows", lambda *a, **kw: pytest.fail("Changed source reached optimizer"))
    with pytest.raises(ValueError):
        subject.run_fit(task, plan, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_local_f_identity_also_binds_the_original_overflow_numerics():
    root = Path(subject.__file__).resolve().parents[4]
    identity = local_refit_source.source_identity(root, calibration_sha256="a" * 64, overflow=True)
    modules = {row["module"] for row in identity["source_binding"]["modules"]}
    assert {"darsia.fluidflower.mass.overflow_steering", "darsia.fluidflower.mass.overflow_refit"} <= modules


def test_q255_only_completed_p1_is_refused_before_local_f_creation(tmp_path):
    from darsia.fluidflower.mass.n1_pipeline import DataReader
    config = _write_json(tmp_path / "old-p1.json", dict(run="r1", ordered_case_ids=["one"],
        storage=dict(gas_score_encoding="uint8-la-q255-v1")))
    core = _write_json(tmp_path / "old-core.json", dict(ordered_case_ids=["one"], frames=[{}]))
    done = _write_json(tmp_path / "old-completed.json", dict(execution_mode="local", complete=True,
        run="r1", configuration=config, phase1_completion=core))
    with pytest.raises(ValueError, match="lossless gas probabilities"):
        subject.preparation_budget(dict(phase_sources=dict(r1=dict(completion=done))),
            dict(runs=dict(fit=["r1"])), {}, reader=DataReader(unc_only=False))


@pytest.mark.parametrize("legacy", [True, False])
def test_local_job_dispatches_real_f_fit_and_retains_separate_reference(local_metadata, monkeypatch, legacy):
    """Isolate only the P1-preparation boundary; both optimizer/freeze calls are real."""
    from darsia.fluidflower.pipeline import local_refit, refit_runner
    from darsia.fluidflower.mass.n1_pipeline import DataReader
    f = local_metadata
    f.arguments["roles"]["descriptive"] = []
    inputs = _write_json(f.series["tmp"] / "routing-only-overflow-inputs.json", {"TESTONLY": "source-admission tested separately"})
    monkeypatch.setattr(subject, "read_request", lambda *a, **kw: {})
    monkeypatch.setattr(subject, "preparation_budget", lambda *a, **kw: dict(
        products_estimated_bytes=1024, transient_estimated_bytes=1024, workspace_estimated_bytes=1024))
    def prepare(job, plans, root, **kwargs):
        entries = {}
        for run in plans["refit"]["statistics"]:
            entries[run] = []
            for item in plans["refit"]["statistics"][run]:
                row = refit_runner.read_statistics(DataReader(unc_only=False), item["statistics"])
                row["contract"] = overflow_statistics.STATISTICS_FORMAT
                row["overflow_steering"] = dict(format="darsia-overflow-statistics-derivation-v1")
                row["main_gas_mass_g"] *= .5
                path = root / (run + "-" + str(row["frame_index"]) + ".json.gz")
                path.write_bytes(gzip.compress(json.dumps(row).encode(), mtime=0))
                entries[run].append(dict(item, statistics=asset(path)))
        bound = _write_json(root / "routing-only-preparation.json", dict(format=subject.PREPARATION_FORMAT,
            statistics=entries, production_admitted=False, pending_nonfit_runs=["held"], frames=[]))
        return {key: dict(deepcopy(value), format=subject.PLAN_FORMAT, statistics=deepcopy(entries),
                         variant_f={"enabled": True}, overflow_preparation=bound,
                         **({"overflow_phase_split_policy": job["overflow_phase_split_policy"]}
                            if job.get("overflow_phase_split_policy") else {})) for key, value in plans.items()}
    monkeypatch.setattr(subject, "prepare", prepare)
    local_refit.create_local_refit_job(**f.arguments, overflow=inputs)
    if legacy:
        # Replay of the retained legacy comparison route.
        saved_job = json.loads((f.root / "job.json").read_bytes())
        saved_job.pop("overflow_phase_split_policy")
        (f.root / "job.json").write_text(json.dumps(saved_job), encoding="utf-8")
    ordinary_calls = []
    original = refit_runner.run_local_refit
    def ordinary(*args, **kwargs):
        ordinary_calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(refit_runner, "run_local_refit", ordinary)
    snapshot = local_refit.run_local_refit_job(f.root)
    assert snapshot["status"]["complete"] and snapshot["local_variant_f"]
    frozen, _ = snapshot["frozen"]
    assert frozen["variant_f"] and frozen["format"] == subject.FROZEN_FORMAT
    if legacy:
        reference = DataReader(unc_only=False).json(frozen["reference_without_overflow"])
        assert reference["pipeline"] == "n1-calibration-refit"
        assert reference["parameters"]["gain"] < frozen["parameters"]["gain"]
        assert ordinary_calls
    else:
        assert not ordinary_calls
        assert "reference_without_overflow" not in frozen
        assert not (f.root / "reference_without_overflow").exists()
        assert frozen["overflow_phase_split_policy"]
    assert snapshot["overflow_inputs"] == inputs


def test_freeze_records_the_plan_chemistry_and_every_field_phase_two_reads(tmp_path):
    from darsia.fluidflower.production_selection import production_calibration_fields
    plan, task = source(tmp_path)
    plain = json.loads(Path(subject.run_fit(task, plan, tmp_path / "plain")["frozen_calibration"]["path"]).read_bytes())
    assert "chemistry_model" not in plain and production_calibration_fields(plain) is None
    plan["chemistry_model"] = "titration"
    task = dict(task, configuration=_write_json(tmp_path / "titration-plan.json", plan))
    result = subject.run_fit(task, plan, tmp_path / "titration")
    frozen = json.loads(Path(result["frozen_calibration"]["path"]).read_bytes())
    assert frozen["chemistry_model"] == "titration" and production_calibration_fields(frozen) == "titration"
    for field, reason in (("anchors_by_run", "anchor for every fit run"), ("rules", "anchor rules"),
                          ("heldout_runs", "held-out runs"), ("parameters", "gain and beta")):
        broken = {key: value for key, value in frozen.items() if key != field}
        with pytest.raises(ValueError, match=reason):
            production_calibration_fields(broken)
