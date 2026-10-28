from copy import deepcopy
import json
import tomllib

import pytest
import toml

from darsia.fluidflower.shared_calibration_cohort import (
    explicit_mass_cohort, mass_cohort_roles, read_mass_cohort, save_mass_cohort,
)
from darsia.fluidflower.series_method import ManifestChangedError


def series(tmp_path):
    value = dict(runs=dict(ids=["one", "two", "three", "skip"], excluded={"skip": "omitted"}),
        method=dict(active="phase_model", phase_model=dict(mass_calibration="baseline.json")),
        image_selection=dict(runs={"one": dict(selected=[dict(image_id="preserve")])}),
        future=dict(unknown=[1, 2]))
    (tmp_path / "baseline.json").write_text(json.dumps(dict(calibration_runs=["one", "two"], heldout_runs=["three"])))
    raw = b"\xef\xbb\xbf" + toml.dumps(value).replace("\n", "\r\n").encode()
    (tmp_path / "series.toml").write_bytes(raw)
    return value, raw


def test_explicit_mass_split_is_independent_and_individual_uses_every_active_run(tmp_path):
    original, raw = series(tmp_path)
    assert mass_cohort_roles(tmp_path)[0]["fit"] == ["one", "two"]
    value = save_mass_cohort(tmp_path, ["one", "three"], expected_bytes=raw)
    assert value["roles"] == dict(fit=["one", "three"], heldout=["two"], descriptive=[], excluded=["skip"])
    saved_raw = (tmp_path / "series.toml").read_bytes()
    assert saved_raw.startswith(b"\xef\xbb\xbf") and b"\r\n" in saved_raw
    saved = tomllib.loads(saved_raw.decode("utf-8-sig"))
    for key in ("runs", "method", "image_selection", "future"):
        assert saved[key] == original[key]
    assert mass_cohort_roles(tmp_path, scope="individual")[0] == dict(
        fit=["one", "two", "three"], heldout=[], descriptive=[], excluded=["skip"])
    with pytest.raises(ManifestChangedError):
        save_mass_cohort(tmp_path, ["one"], expected_bytes=raw)
    before = saved_raw
    with pytest.raises(ValueError, match="at least one"):
        save_mass_cohort(tmp_path, [], expected_bytes=before)
    assert (tmp_path / "series.toml").read_bytes() == before


def test_refit_plans_never_read_heldout_statistics(tmp_path):
    from darsia.fluidflower.pipeline.refit_jobs import build_plans
    from tests.unit.gui.test_refit_job_panel import _rules, BASELINE
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(dict(parameters=BASELINE)))
    from darsia.fluidflower.mass.n1_pipeline import asset
    roles = dict(fit=["one", "two"], heldout=["three"], descriptive=[], excluded=[])
    entry = dict(case_id="frame", frame_index=1, selected_target_h=.5,
                 statistics=dict(path="unused", sha256="a" * 64, size_bytes=1))
    class Inputs(dict):
        def __getitem__(self, key):
            assert key != "three", "A holdout must never be consumed by the objective"
            return super().__getitem__(key)
    plans = build_plans(statistics=Inputs(one=[entry], two=[entry]), roles=roles,
        rules=_rules(), baseline_calibration=asset(baseline), loro=False)
    assert plans["refit"]["runs"]["heldout"] == ["three"]
    assert set(plans["refit"]["statistics"]) == {"one", "two"}


def test_changed_overflow_cohort_uses_only_current_fit_measurements():
    from darsia.fluidflower.mass import overflow_method as method, overflow_levels
    analysis = list(dict.fromkeys([*method.USABLE_RUNS, *method.NO_USABLE_OVERFLOW]))
    roles = dict(fit=["ac28", "ac30"], heldout=[run for run in analysis if run not in {"ac28", "ac30"}],
                 descriptive=[], excluded=[])
    cohort = dict(format="darsia-current-mass-cohort-v1", roles=roles, analysis_runs=analysis)
    observations = {run: dict(ratio=ratio, sigma=.1, selected_cases=[run + "-observed"])
                    for run, ratio in (("ac28", .4), ("ac30", .6), ("ac35", 100.))}
    fallback = {run: dict(code=(overflow_levels.NO_OVERFLOW if run in method.NO_USABLE_OVERFLOW else
        overflow_levels.ANCHOR_UNPROVED if run in method.CONDITIONAL_ANCHOR_RUNS else overflow_levels.NO_CURRENT_VOLUME),
        reason="Actual source has no eligible observation") for run in analysis if run not in observations}
    with pytest.raises(ValueError, match="unchanged accepted"):
        method.validate_roles(roles, analysis)
    result = overflow_levels.observations_to_levels(observations, fallback,
        roles=roles, analysis_runs=analysis, cohort=cohort)
    assert set(result["group_fit_runs"]) == {"ac28", "ac30"}
    changed = deepcopy(observations)
    changed["ac35"]["ratio"] = 10000.
    other = overflow_levels.observations_to_levels(changed, fallback,
        roles=roles, analysis_runs=analysis, cohort=cohort)
    assert result["group_ratio"] == other["group_ratio"]
    assert result["between_run_variance"] == other["between_run_variance"]
