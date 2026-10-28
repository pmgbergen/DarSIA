"""Original scalar evaluation and prescribed check boundaries, no campaign claim.

Finite original statistics are evaluated at known frozen parameters. These tests
exercise the original prediction kernel and unchanged score maths; they do not
substitute publication, original P1 verification, or a 26-fold campaign result.
"""
from copy import deepcopy
from pathlib import Path

import pytest

from darsia.fluidflower import calibration_check_calculation as checks
from darsia.fluidflower.calibration_checks import normalise_check_scores
from darsia.fluidflower.mass import n1_refit
from test_n1_refit import make_plan, make_row, TRUE_GAIN, TRUE_BETA


@pytest.fixture
def evaluated():
    fit_runs = ["finite-fit-" + str(index) for index in range(26)]
    rules = n1_refit.RefitRules.from_plan(make_plan(fit_runs))
    times = sorted(set(checks.CHECK1_T + checks.CHECK2_T + checks.CHECK3_T))
    runs = checks.HELDOUT + fit_runs
    rows = [make_row(run, index + 1, time) for run in runs for index, time in enumerate(times)]
    anchors = n1_refit.anchors_by_run(rows, rules)
    fixed = dict(parameters=n1_refit.parameters(TRUE_GAIN, TRUE_BETA), anchors_by_run=anchors)
    predictions = {run: n1_refit.predict_run(rows, fixed, run) for run in runs}
    held = [row for run in checks.HELDOUT for row in predictions[run]]
    loro = [row for run in fit_runs for row in predictions[run]]
    return held, loro, fit_runs, rows, fixed


def test_original_frozen_evaluation_scores_all_prescribed_times_without_a_fit(evaluated):
    held, loro, fit_runs, rows, fixed = evaluated
    original = deepcopy((held, loro, rows, fixed))
    for run in checks.HELDOUT + fit_runs:
        own = sorted((row for row in rows if row["run"] == run), key=lambda row: row["frame_index"])
        assert checks._predict_rows(lambda: iter(own), fixed, run) == n1_refit.predict_run(rows, fixed, run)
    scored = checks.score_predictions(held, loro, fit_runs=fit_runs)
    assert scored["verdict"] == "PASS"
    assert scored["check1"]["status"] == "PASS"
    assert scored["check2"]["primary"]["assessable_times"] == 5
    assert scored["check3"]["primary"]["assessable_times"] == 4
    assert scored["check2"]["decision"]["deciding_rule"] == "primary_U_rest_capacity"
    assert all(not scored[key] for key in ("numerical_fit_performed", "calibration_adopted", "production_started"))
    assert (held, loro, rows, fixed) == original
    assert normalise_check_scores(scored).verdict == "PASS"  # Original display reader ABI.
    sample = held[0]
    source = next(row for row in rows if row["run"] == sample["run"] and row["frame_index"] == sample["frame_index"])
    assert sample["total_mass_g"] == n1_refit.evaluate(source, TRUE_GAIN, TRUE_BETA,
        fixed["anchors_by_run"][sample["run"]])["total_mass_g"]


@pytest.mark.parametrize("mode", ["primary_fail", "fallback_only", "too_few_even_fallback"])
def test_primary_fail_is_never_rescued_and_fallback_only_decides_ineligibility(evaluated, mode):
    held, loro, fit_runs, _rows, _fixed = evaluated
    for value in held + loro:
        if value["selected_target_h"] not in checks.CHECK2_T + checks.CHECK3_T:
            continue
        d = value["injected_mass_g"]
        value.update(total_mass_g=d, E_lo_g=-.01*d, E_hi_g=.01*d, U_rest_g=0., W_g=.02*d)
        if mode == "primary_fail":
            # Genuine narrow eligible interval: d is 20% below its nearest end.
            value.update(total_mass_g=1.21*d, E_lo_g=-.01*d, E_hi_g=.01*d)
        else:
            value.update(U_rest_g=.10*d, W_g=.12*d)
            if mode == "too_few_even_fallback":
                value.update(E_lo_g=-.03*d, E_hi_g=.03*d, W_g=.16*d)
    scored = checks.score_predictions(held, loro, fit_runs=fit_runs)
    for key in ("check2", "check3"):
        rule = scored[key]
        if mode == "primary_fail":
            assert rule["primary"]["status"] == "FAIL"
            assert rule["decision"] == dict(deciding_rule="primary_U_rest_capacity", status="FAIL")
            assert next(iter(rule["primary"]["per_time"].values()))["median_delta"] == pytest.approx(.20)
        elif mode == "fallback_only":
            assert rule["primary"]["status"] == "NOT_ASSESSABLE"
            assert rule["decision"] == dict(deciding_rule="U_rest_zero_fallback", status="PASS")
        else:
            assert rule["decision"]["status"] == "NOT_ASSESSABLE"
            assert next(iter(rule["fallback"]["per_time"].values()))["runs_counted"] == 0
    assert scored["verdict"] == ("PASS" if mode == "fallback_only" else "FAIL")


def test_check1_lighting_group_cannot_be_hidden_by_the_pooled_median(evaluated):
    held, loro, fit_runs, _rows, _fixed = evaluated
    for value in held:
        if value["run"] in checks.GROUPS["new_lamps"] and value["selected_target_h"] in checks.CHECK1_T:
            value["total_mass_g"] = 1.12 * value["injected_mass_g"]
    scored = checks.score_predictions(held, loro, fit_runs=fit_runs)
    assert scored["check1"]["status"] == "FAIL"
    for value in scored["check1"]["per_time"].values():
        assert value["median_abs_error"] < .10
        assert value["group_median_ratio"]["new_lamps"] == pytest.approx(1.12)
        assert value["time_pass"] is False


@pytest.mark.parametrize("fault", ["duplicate", "missing_time", "wrong_role", "nonfinite", "zero_dose"])
def test_unusable_predictions_refuse_instead_of_inventing_a_score(evaluated, fault):
    held, loro, fit_runs, _rows, _fixed = evaluated
    if fault == "duplicate":
        held.append(deepcopy(held[0]))
    elif fault == "missing_time":
        held[:] = [row for row in held if not (row["run"] == checks.HELDOUT[0] and row["selected_target_h"] == 1.)]
    elif fault == "wrong_role":
        loro[0]["run"] = checks.HELDOUT[0]
    elif fault == "nonfinite":
        held[0]["total_mass_g"] = float("nan")
    else:
        held[0]["injected_mass_g"] = 0.
    with pytest.raises(ValueError):
        checks.score_predictions(held, loro, fit_runs=fit_runs)


def test_producer_refuses_output_conflict_before_any_input_can_be_opened(tmp_path):
    existing = tmp_path / "existing.json"
    existing.write_bytes(b"retain")
    with pytest.raises(FileExistsError, match="fresh"):
        checks.calculate_check_scores("missing-series", "missing-calibration", "missing-phase", existing, protocol={})
    assert existing.read_bytes() == b"retain"


def test_missing_uncompleted_job_cannot_be_replaced_by_a_plausible_frozen_file(tmp_path):
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_bytes(b'schema_version=1\nseries_id="finite"\n[roots]\nseries="."\n')
    inputs = checks._Inputs(root)
    job = root / "job"
    job.mkdir()
    (job / "job.json").write_bytes(b'{"pipeline":"n1-calibration-refit","precision":"float32","expected_runs":["refit"]}')
    with pytest.raises(ValueError, match="original completed"):
        inputs.terminal(job, "refit", "n1-calibration-refit")
    assert not (root / "scores.json").exists()


def test_original_unselected_target_rows_are_retained_but_never_counted(evaluated):
    held, loro, fit_runs, _rows, _fixed = evaluated
    before = checks.score_predictions(held, loro, fit_runs=fit_runs)
    unselected = deepcopy(held[0])
    unselected.update(frame_index=999, selected_target_h=None, injected_mass_g=0.)
    held.append(unselected)
    after = checks.score_predictions(held, loro, fit_runs=fit_runs)
    assert all(after[key] == before[key] for key in ("check1", "check2", "check3", "verdict"))
    assert after["counts"]["heldout_rows"] == before["counts"]["heldout_rows"] + 1


def test_original_tiny_fit_publication_is_consumed_via_exact_shared_metadata(tmp_path, monkeypatch):
    """Original numerical fit/queue publication; declared finite I/O transport.

    The same explicit I/O specimens as the existing original publication tests
    model a local filesystem and availability. No publication/source identity
    predicate is replaced, and this is not current campaign qualification.
    """
    import test_refit_jobs as fixture
    from types import SimpleNamespace
    from tests.unit.test_phase1_worker_protocol import SyntheticServerClock
    from darsia.fluidflower import job_filesystem
    from darsia.fluidflower.pipeline import job_queue, publication_runtime, refit_runner, run_io
    from darsia.fluidflower.pipeline.store_runtime import ResolvedInput
    data = fixture.series.__wrapped__(tmp_path, monkeypatch)
    prepared, root = fixture.prepare(data, loro=False)
    job = job_queue.read_json(root / "job.json")
    task = job_queue.read_json(Path(prepared["tasks"][0]["task"]["path"]))
    unit_asset = Path(data["baseline"]["path"])
    def availability(value):
        return tuple(ResolvedInput(deepcopy(row["reference"]), unit_asset, tuple(row["uses"]))
            for row in value["input_catalog"]["inputs"])
    def output(value, token):
        local = tmp_path / "published-local"
        attempt = local / value["job_id"] / value["run"] / ".attempts" / token
        admission = job_filesystem.LocalFileAdmission(True, "finite-local", "unit protocol",
            "No physical filesystem qualification", str(tmp_path))
        return job_queue.LocalOutputBinding("finite", local, attempt, admission)
    def no_foreign(_attempt):
        raise AssertionError("Fresh result cannot borrow another producer")
    queue = job_queue.JobQueue(root, machine_id="finite", server_clock=SyntheticServerClock())
    publisher = publication_runtime.make_publication_factory(root, job, "finite", producer_origin_for_attempt=no_foreign)
    owner = dict(machine="finite", pid=123, process_create_time="finite-check-input")
    claim = queue.claim("finite", "cpu", owner, availability=availability,
        output_factory=output, publication_factory=publisher)
    configuration = fixture.bound(data["root"] / "calibration/refit/job1/refit_plan.json")
    binding = SimpleNamespace(configuration_descriptor=configuration, require_output=lambda _output: None)
    with monkeypatch.context() as patch:
        patch.setattr(run_io, "for_task", lambda value: binding if value is claim.task else None)
        result = refit_runner.run_refit(claim.task, claim.output, "cpu", "float32")
    assert result["stop_required"] is False
    result = dict(result, task=claim.task, attempt_id=claim.token, commit=task["commit"],
        calibration_sha256=task["calibration_sha256"], actual_worker=owner)
    queue.complete(claim, result)
    inputs = checks._Inputs(data["root"])
    actual_job, actual_result, product = inputs.terminal(root, "refit", refit_runner.PIPELINE)
    assert actual_job == job and actual_result == result
    frozen, frozen_pin = product("frozen_calibration.json")
    assert frozen["parameters"] == result["parameters"]
    assert not Path(frozen_pin["path"]).is_relative_to(claim.output)
    bindings = inputs.finish()
    assert frozen_pin in bindings
    old = Path(frozen_pin["path"]).read_bytes()
    Path(frozen_pin["path"]).write_bytes(old + b" ")
    with pytest.raises(ValueError, match="changed|differs"):
        inputs.finish()


def test_protocol_descriptor_is_exact_and_bounded_before_opening_a_job(tmp_path):
    import hashlib
    accepted = tmp_path / "accepted-protocol.md"
    accepted.write_bytes(b"# finite accepted protocol\n")
    pin = dict(sha256=hashlib.sha256(accepted.read_bytes()).hexdigest(), size_bytes=accepted.stat().st_size)
    series = tmp_path / "series"
    series.mkdir()
    (series / "series.toml").write_text('schema_version=1\nseries_id="finite"\n[decisions.calibration_protocol]\n'
        f'sha256="{pin["sha256"]}"\nsize_bytes={pin["size_bytes"]}\n', encoding="utf-8")
    assert checks.protocol_binding(accepted, series_root=series) == dict(path=str(accepted), **pin)
    wrong = tmp_path / "wrong-protocol.md"
    wrong.write_bytes(b"x" * (pin["size_bytes"] + 1))
    with pytest.raises(ValueError, match="exact current"):
        checks.protocol_binding(wrong, series_root=series)
    wrong.write_bytes(b"x" * pin["size_bytes"])
    with pytest.raises(ValueError, match="exact current"):
        checks.protocol_binding(wrong, series_root=series)
    # A series that pins no accepted protocol says which setting is missing.
    (series / "series.toml").write_text('schema_version=1\nseries_id="finite"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="decisions.calibration_protocol"):
        checks.protocol_binding(accepted, series_root=series)
    assert not (tmp_path / "scores.json").exists()


def test_raw_readiness_bindings_keep_original_toml_text_and_source_formats(tmp_path):
    root = tmp_path / "series"
    root.mkdir()
    manifest = root / "series.toml"
    manifest.write_bytes(b'schema_version=1\nseries_id="finite"\n[roots]\nseries="."\n')
    protocol = root / "protocol.md"
    protocol.write_bytes(b"# finite protocol bytes, not JSON\n")
    source = root / "source.py"
    source.write_bytes(b"def frozen(): return 1\n")
    inputs = checks._Inputs(root)  # Original TOML manifest consumer.
    for path in (manifest, protocol, source):
        pin = inputs.pin(path)
        assert inputs.recheck(pin) == pin
    before = inputs.finish()
    assert len(before) == 3
    malformed = root / "malformed.json"
    malformed.write_bytes(b"not JSON")
    with pytest.raises(ValueError):
        inputs.metadata(malformed)  # A JSON consumer still requires valid JSON.
    original_protocol = protocol.read_bytes()
    protocol.write_bytes(original_protocol.replace(b"finite", b"FINITE"))
    assert protocol.stat().st_size == len(original_protocol)
    with pytest.raises(ValueError, match="changed"):
        inputs.finish()
