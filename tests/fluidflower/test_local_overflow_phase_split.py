from copy import deepcopy
import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.mass import overflow_statistics
from darsia.fluidflower.mass.n1_pipeline import DataReader, _write_json
from darsia.fluidflower.mass.overflow_phase_split import POLICY
from darsia.fluidflower.pipeline import local_overflow_phase_split as subject
from test_overflow_statistics import steering_case, prepared  # noqa: F401
from tests.fixtures.tiny_saved_model_fixture import tiny_config  # noqa: F401


def test_new_policy_admits_named_series_sources_without_fixed_experiment_lists(tmp_path):
    from darsia.fluidflower.pipeline.local_overflow_refit import REQUEST_FORMAT, read_request
    source = _write_json(tmp_path / "source.json", dict(metadata=dict(run="sample"), records=[]))
    request = _write_json(tmp_path / "request.json", dict(format=REQUEST_FORMAT, scale_reviews={"sample": source}))
    roles = dict(fit=["sample"], heldout=["holdout"], descriptive=[], excluded=[])
    cohort = dict(format="darsia-current-mass-cohort-v1", roles=roles, analysis_runs=["sample", "holdout"])
    value = read_request(request, roles=roles, cohort=cohort, policy=POLICY, reader=DataReader(unc_only=False))
    assert value["scale_reviews"] == {"sample": source}
    with pytest.raises(ValueError):
        read_request(request, roles=roles, cohort=cohort, reader=DataReader(unc_only=False))


def test_preparation_consumes_own_target_and_publishes_exact_phase_split_without_raw(steering_case, tmp_path, monkeypatch):
    case = steering_case
    frame = prepared(case)
    run = frame.prepared["frame"].run_id
    spec = dict(case["spec"])
    time = spec["elapsed_h"]
    baseline = overflow_statistics.baseline_volume(frame)
    target = .7*baseline
    review = _write_json(tmp_path / "own.json", dict(metadata=dict(run=run)))
    monkeypatch.setattr(subject.overflow_measurements, "reviewed_curve", lambda document, **kwargs: dict(
        usable=True, curve=dict(t=np.array([time-1., time+1.]), vC=np.array([target*1e6]*2)), qa={}))
    class Saved:
        def __init__(self, phase_job, selected, **kwargs):
            assert selected == run
            self.config = dict(statistics=dict(dose=dict(injection_stages=[dict(start_h=0., end_h=2.)])))
            self.physical = SimpleNamespace(context=SimpleNamespace(provenance=dict(adopted_revision=dict(sha256="same"))))
            self.bindings = dict(completion={"path": "original-completed-p1.json"})
        def frames(self, **kwargs):
            yield spec, frame
        def finish(self):
            pass
    from darsia.fluidflower.pipeline import local_overflow_sources
    monkeypatch.setattr(local_overflow_sources, "LocalOverflowRun", Saved)
    selected_time = case["original"]["selected_target_h"]
    plan = dict(runs=dict(fit=[run], heldout=[], descriptive=[], excluded=[]), statistics={run: [dict(
        frame_index=spec["frame_index"], selected_target_h=selected_time, statistics=case["statistics"]) ]})
    output = tmp_path / "prepared"
    output.mkdir()
    result = subject.prepare(dict(phase_job="original-p1", overflow={"path": "request"}),
        dict(refit=plan), output, dict(scale_reviews={run: review}), reader=DataReader(unc_only=False))
    assert result["refit"]["overflow_phase_split_policy"] == POLICY
    binding = result["refit"]["statistics"][run][0]["statistics"]
    row = json.loads(gzip.decompress(open(binding["path"], "rb").read()))
    assert row["injected_mass_g"] == case["original"]["injected_mass_g"]
    assert row["overflow_steering"]["phase_split_target"]["kind"] == "own_measurement"
    preparation = DataReader(unc_only=False).json(result["refit"]["overflow_preparation"])
    assert preparation["policy"] == POLICY
    assert preparation["frames"][0]["distribution"]["actual_volume_m3"] == pytest.approx(target, rel=2e-7)
    assert not preparation["raw_decoded"] and not preparation["phase_inference_run"]
    from darsia.fluidflower.mass.gas_contribution_review import read_gas_contributions
    review = read_gas_contributions(preparation, run, spec["case_id"])
    assert review["status"] == "available"
    assert review["summary"]["reference_kind"] == "own_measurement"
    assert review["summary"]["estimated"] is False
    assert review["summary"]["elapsed_h"] == time
    assert review["maps"]["cell_model_before_g"].sum() == pytest.approx(
        review["summary"]["model_gas_mass_g"], rel=2e-7)
    assert review["maps"]["cell_final_gas_g"].sum() == pytest.approx(
        review["summary"]["final_gas_mass_g"], rel=2e-7)
    assert review["summary"]["overflow_change_g"] < 0


def test_unavailable_overflow_retains_original_phase_fields_and_explicit_status(steering_case, tmp_path):
    frame = prepared(steering_case)
    result = subject._model_result(frame, "No measurement at this time")
    row = result["statistics"]
    ignored = {"contract", "overflow_steering"}
    assert {key: value for key, value in row.items() if key not in ignored} == {
        key: value for key, value in frame.prepared["statistics"].items() if key not in ignored}
    assert row["overflow_steering"]["kind"] == "model_phase_split"
    assert row["overflow_steering"]["overflow_reference_available"] is False
    assert "no overflow reference" in result["report"]["status"]
    from darsia.fluidflower.mass.gas_contribution_review import save_gas_contributions, read_gas_contributions
    record = save_gas_contributions(tmp_path, frame, result)
    saved_document = dict(frames=[dict(run="run", case_id="image", gas_contribution=record)])
    review = read_gas_contributions(saved_document, "run", "image")
    assert review["summary"]["reference_kind"] == "model_phase_split"
    assert review["summary"]["overflow_change_g"] == 0.
    assert review["maps"]["cell_overflow_added_g"].sum() == 0.
    assert review["maps"]["cell_overflow_removed_g"].sum() == 0.
