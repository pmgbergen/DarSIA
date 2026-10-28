"""Phase-two source manifests from published phase-one jobs.

The publication is B6's real local queue fixture with synthetic, undecoded
payloads. The dry run hands the written manifest to the original phase-two
preparation, verifier and sealer. These tests qualify no science or host.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import analysis_preparation as preparation
from darsia.fluidflower import phase_two_inputs as inputs
from darsia.fluidflower.mass.n1_pipeline import DataReader
from darsia.fluidflower.mass.n1_refit import RULES_FORMAT
from darsia.fluidflower.mass.n1_statistics import CONTRACT
from darsia.fluidflower.pipeline import refit_runner
from darsia.fluidflower.pipeline.job_queue import descriptor

from .test_phase1_enqueue_alias_master import owned_publication


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return descriptor(path)


def series(root, runs):
    root.mkdir(parents=True)
    ids = ", ".join(json.dumps(run) for run in runs)
    (root / "series.toml").write_text(
        f'schema_version = 1\nseries_id = "Synthetic"\n[roots]\nseries = "."\n[runs]\nids = [{ids}]\n',
        encoding="utf-8")
    return root


def frozen(root, *, fit, heldout=(), anchors=None, anchor_target_h=2.247):
    """A frozen calibration with original N1 parameters and refit rules."""
    rules = dict(format=RULES_FORMAT, stage1_window_h=[.5, 2.375], stage2_start_h=2.5,
        post_bin_edges_h=[8., 48.], width_limit=.05, anchor_target_h=anchor_target_h,
        anchor_tolerance_h=.13, baseline=dict(candidate="N1", gain=24., beta=.6),
        gain_starts=[1.], beta_starts=[.6], gain_grid={}, beta_grid={},
        require_frame_list_evidence=False)
    return put(root / "frozen_calibration.json", dict(format="unit-frozen-calibration", frozen=True,
        parameters=dict(candidate="N1", gain=1.7, beta=.4), calibration_runs=list(fit),
        heldout_runs=list(heldout), descriptive_runs=[], excluded_runs=[],
        anchors_by_run=anchors or {}, rules=rules))


@pytest.fixture
def published(tmp_path):
    return owned_publication(tmp_path / "publication", complete=True)


def test_fit_run_manifest_is_sealed_by_the_original_preparation(tmp_path, published):
    run = published.task["run"]
    anchor = dict(elapsed_h=2.25, affected_pore_volume_m3=.001, frame_index=3,
                  source_binding=dict(source_sha256="d" * 64))
    calibration = frozen(tmp_path / "calibration", fit=[run], anchors={run: anchor})
    root = series(tmp_path / "series", [run])
    destination = tmp_path / "inputs" / "phase_two_inputs.json"
    destination.parent.mkdir()
    result = inputs.collect_phase2_source_inputs([published.root], calibration, destination,
                                                 series_root=root)
    assert result["runs"] == [run] and result["plan_prepared"] is False
    document = json.loads(destination.read_bytes())
    assert document["format"] == inputs.FORMAT
    assert document["series_manifest"] == descriptor(root / "series.toml")
    assert document["anchor_calibration"] == calibration
    assert document["phase_one_results"] == {
        run: descriptor(published.root / "results" / run / "completed.json")}
    assert document["runs"] == [dict(run=run,
        completion_bound=published.result["phase1_completion"],
        configuration_bound=published.result["configuration"],
        expected_phase1_commit=published.job["commit"],
        fitted_anchor=dict(elapsed_h=2.25, affected_pore_volume_m3=.001))]
    # Dry run: the unchanged phase-two preparation verifies and seals these rows.
    job = dict(commit=published.job["commit"], precision="float32",
        environment_lock=dict(path="requirements-worker.lock", sha256="a" * 64),
        created_utc="2026-10-27T12:00:00+00:00")
    plan = preparation.prepare_phase2_inputs(result["inputs"], tmp_path / "plan", series_root=root,
        fitted_calibration=calibration, job_id="production", job=job)
    assert plan["pipeline"] == "n1" and plan["adopted_for_execution"] is False
    assert [row["run"] for row in plan["runs"]] == [run]
    sealed = json.loads(Path(plan["runs"][0]["configuration"]["path"]).read_bytes())
    assert sealed["anchor"] == document["runs"][0]["fitted_anchor"]
    assert sealed["calibration"] == calibration
    assert sealed["phase1"]["completion"] == published.result["phase1_completion"]


@pytest.mark.parametrize("target,expected", [(1., dict(elapsed_h=1., affected_pore_volume_m3=.002)),
                                             (2.247, None)])
def test_held_out_run_takes_its_own_frame_nearest_the_anchor_time(tmp_path, published, monkeypatch,
                                                                  target, expected):
    run = published.task["run"]
    calibration = frozen(tmp_path / "calibration", fit=["fitted"], heldout=[run],
        anchors={"fitted": dict(elapsed_h=2.25, affected_pore_volume_m3=.5)}, anchor_target_h=target)
    root = series(tmp_path / "series", [run])
    seen = []
    def statistics(reader, bound):
        seen.append(bound)
        spec = json.loads(Path(published.result["configuration"]["path"]).read_bytes())["frames"][0]
        return dict(contract=CONTRACT, run=run, frame_index=spec["frame_index"], elapsed_h=spec["elapsed_h"],
                    source_binding=dict(source_sha256=spec["source"]["source_sha256"]),
                    affected_pore_volume_m3=.002)
    monkeypatch.setattr(refit_runner, "read_statistics", statistics)
    destination = tmp_path / "phase_two_inputs.json"
    if expected is None:
        with pytest.raises(ValueError, match="injection-end volume anchor"):
            inputs.collect_phase2_source_inputs([published.root], calibration, destination, series_root=root)
        assert not destination.exists()
        return
    inputs.collect_phase2_source_inputs([published.root], calibration, destination, series_root=root)
    assert json.loads(destination.read_bytes())["runs"][0]["fitted_anchor"] == expected
    assert len(seen) == 1


def test_missing_unassigned_and_twice_published_runs_refuse_before_writing(tmp_path, published):
    run = published.task["run"]
    calibration = frozen(tmp_path / "calibration", fit=[run], anchors={run: dict(
        elapsed_h=2.25, affected_pore_volume_m3=.001)})
    destination = tmp_path / "phase_two_inputs.json"
    root = series(tmp_path / "series", [run, "run-two"])
    with pytest.raises(ValueError, match="run-two: no chosen phase-one job has published this run"):
        inputs.collect_phase2_source_inputs([published.root], calibration, destination, series_root=root)
    other = owned_publication(tmp_path / "other", complete=True)
    single = series(tmp_path / "single", [run])
    with pytest.raises(ValueError, match="two chosen phase-one jobs"):
        inputs.collect_phase2_source_inputs([published.root, other.root], calibration, destination,
                                            series_root=single)
    unassigned = frozen(tmp_path / "unassigned", fit=["fitted"],
                        anchors={"fitted": dict(elapsed_h=2.25, affected_pore_volume_m3=.5)})
    with pytest.raises(ValueError, match="no held-out or descriptive role"):
        inputs.collect_phase2_source_inputs([published.root], unassigned, destination, series_root=single)
    assert not destination.exists()


def test_an_existing_manifest_is_never_replaced(tmp_path, published):
    run = published.task["run"]
    calibration = frozen(tmp_path / "calibration", fit=[run], anchors={run: dict(
        elapsed_h=2.25, affected_pore_volume_m3=.001)})
    root = series(tmp_path / "series", [run])
    destination = tmp_path / "phase_two_inputs.json"
    destination.write_bytes(b"earlier manifest")
    with pytest.raises(FileExistsError):
        inputs.collect_phase2_source_inputs([published.root], calibration, destination, series_root=root)
    assert destination.read_bytes() == b"earlier manifest"


def test_an_incomplete_published_result_is_refused(tmp_path, published):
    run = published.task["run"]
    result = published.root / "results" / run / "completed.json"
    value = json.loads(result.read_bytes())
    value["pipeline"] = "native-phase-n1"
    copy = tmp_path / "copy"
    (copy / "results" / run).mkdir(parents=True)
    (copy / "job.json").write_bytes((published.root / "job.json").read_bytes())
    (copy / "results" / run / "completed.json").write_text(json.dumps(value), encoding="utf-8")
    calibration = frozen(tmp_path / "calibration", fit=[run], anchors={run: dict(
        elapsed_h=2.25, affected_pore_volume_m3=.001)})
    with pytest.raises(ValueError, match="not a complete phase-one run"):
        inputs.collect_phase2_source_inputs([copy], calibration, tmp_path / "out.json",
                                            series_root=series(tmp_path / "series", [run]))
    assert not (tmp_path / "out.json").exists()


def test_prepare_binds_the_inputs_to_the_calibration_it_uses(tmp_path, published, monkeypatch):
    run = published.task["run"]
    calibration = frozen(tmp_path / "calibration", fit=[run], anchors={run: dict(
        elapsed_h=2.25, affected_pore_volume_m3=.001)})
    root = series(tmp_path / "series", [run])
    destination = tmp_path / "phase_two_inputs.json"
    calls = []
    monkeypatch.setattr(preparation, "prepare_phase2_inputs",
                        lambda bound, folder, **kwargs: calls.append((bound, folder, kwargs)) or "plan")
    arguments = dict(series_root=root, job_id="production", job=dict(precision="float32"), threads=2)
    assert inputs.prepare_phase_two_plan([published.root], calibration, destination, tmp_path / "plan",
                                         **arguments) == "plan"
    first = descriptor(destination)
    assert calls[0][0] == first and calls[0][1] == tmp_path / "plan"
    assert calls[0][2]["fitted_calibration"] == calibration and calls[0][2]["threads"] == 2
    # Preparing again with the same calibration reuses the saved inputs unchanged.
    inputs.prepare_phase_two_plan([published.root], calibration, destination, tmp_path / "again", **arguments)
    assert descriptor(destination) == first and calls[1][0] == first
    other = frozen(tmp_path / "other", fit=[run], anchors={run: dict(
        elapsed_h=2.5, affected_pore_volume_m3=.002)})
    with pytest.raises(ValueError, match="made for another mass calibration"):
        inputs.prepare_phase_two_plan([published.root], other, destination, tmp_path / "third", **arguments)
    second = owned_publication(tmp_path / "second", complete=True)
    with pytest.raises(ValueError, match="other phase-one results"):
        inputs.bind_phase_two_inputs([second.root], calibration, destination, series_root=root)
    assert len(calls) == 2 and descriptor(destination) == first


def _phase_one(run, times, *, rate=.01):
    """A verified phase one in the shape overflow_targets reads (metadata only)."""
    rows = [dict(time_hours=t, pressure_bar=1. + .01 * t, temperature_c=23., pressure_gradient_bar_per_m=0.,
                 temperature_gradient_c_per_m=0.) for t in (0., 40.)]
    statistics = dict(physical_correction=dict(path=run + "-operation.json", sha256="0" * 64, size_bytes=1),
                      atmosphere_series=dict(rows=rows), recipe=dict(alkalinity_M=.01, btb_M=.001),
                      dose=dict(dose_kind="nominal_setpoint", injection_stages=[dict(
                          start_h=0., end_h=40., mass_rate_g_s=rate, location_x=0., location_y=0., location_id="1")]))
    frames = [dict(case_id=f"{run}-{index}", elapsed_h=t, mass_spec=dict(atmosphere=None))
              for index, t in enumerate(times)]
    return dict(configuration=dict(run=run, statistics=statistics, frames=frames))


def test_production_targets_follow_own_measurement_then_nearest_pressure_fit_donor(tmp_path, monkeypatch):
    from darsia.fluidflower.mass import native_optical_source, overflow_measurements
    from darsia.fluidflower.mass.overflow_phase_split import POLICY
    from darsia.fluidflower.pipeline import setup_image_preparation
    request = put(tmp_path / "request.json", dict(scale_reviews={
        "fit-a": put(tmp_path / "review-a.json", dict(run="fit-a")),
        "held-b": put(tmp_path / "review-b.json", dict(run="held-b"))}))
    preparation = put(tmp_path / "preparation.json", dict(policy=POLICY, inputs=request))
    frozen = dict(overflow_preparation=preparation, calibration_runs=["fit-a"])
    monkeypatch.setattr(native_optical_source, "load_build_physical_context", lambda operation, reader:
                        SimpleNamespace(provenance=dict(adopted_revision=dict(sha256="a" * 64))))
    monkeypatch.setattr(setup_image_preparation, "retained_series_aliases", lambda operations, reader: [])
    curves = {"fit-a": dict(usable=True, curve=dict(t=[0., 10.], vC=[0., 100.]), qa={}),
              "held-b": dict(usable=False, reasons=["drift"])}
    monkeypatch.setattr(overflow_measurements, "reviewed_curve", lambda document, run: curves[run])
    verified = {"fit-a": _phase_one("fit-a", [1., 2.]), "held-b": _phase_one("held-b", [1.5, 35.])}
    result = inputs.overflow_targets(verified, frozen, reader=DataReader())
    own = result["fit-a"]["targets"]
    assert result["fit-a"]["policy"] == POLICY and list(own) == ["fit-a-0", "fit-a-1"]
    assert own["fit-a-0"]["kind"] == "own_measurement"
    assert own["fit-a-0"]["target_volume_m3"] == pytest.approx(10e-6)
    donor = result["held-b"]["targets"]["held-b-0"]  # Its own review is not usable.
    assert donor["kind"] == "pressure_matched_estimate" and donor["estimated"] is True
    assert [row["run"] for row in donor["donors"]] == ["fit-a"]
    none = result["held-b"]["targets"]["held-b-1"]  # No donor measured around 35 h.
    assert none["kind"] == "none" and "comparable" in none["reason"]


def test_the_switch_is_off_by_default_and_binds_the_saved_inputs(tmp_path, published, monkeypatch):
    run = published.task["run"]
    calibration = frozen(tmp_path / "calibration", fit=[run], anchors={run: dict(
        elapsed_h=2.25, affected_pore_volume_m3=.001)})
    root = series(tmp_path / "series", [run])
    plain = tmp_path / "plain.json"
    explicit = tmp_path / "explicit.json"
    inputs.collect_phase2_source_inputs([published.root], calibration, plain, series_root=root)
    inputs.collect_phase2_source_inputs([published.root], calibration, explicit, series_root=root,
                                        overflow_phase_split=False)
    assert plain.read_bytes() == explicit.read_bytes()  # Off writes today's manifest.
    assert "overflow_phase_split" not in json.loads(plain.read_bytes())
    with pytest.raises(ValueError, match="Choose explicitly"):
        inputs.collect_phase2_source_inputs([published.root], calibration, tmp_path / "x.json",
                                            series_root=root, overflow_phase_split="on")
    split = dict(policy="unit", targets={})
    monkeypatch.setattr(inputs, "overflow_targets", lambda verified, frozen, reader: {run: split})
    switched = tmp_path / "switched.json"
    inputs.collect_phase2_source_inputs([published.root], calibration, switched, series_root=root,
                                        overflow_phase_split=True)
    document = json.loads(switched.read_bytes())
    assert document["overflow_phase_split"] is True and document["runs"][0]["overflow_phase_split"] == split
    with pytest.raises(ValueError, match="made with the overflow phase split on"):
        inputs.bind_phase_two_inputs([published.root], calibration, switched, series_root=root)
    with pytest.raises(ValueError, match="made with the overflow phase split off"):
        inputs.bind_phase_two_inputs([published.root], calibration, plain, series_root=root,
                                     overflow_phase_split=True)


def test_the_seal_is_unchanged_without_targets_and_needs_stored_probabilities(tmp_path, published):
    from darsia.fluidflower.pipeline.staged_data import seal_phase2_configuration
    run = published.task["run"]
    calibration = frozen(tmp_path / "calibration", fit=[run], anchors={run: dict(
        elapsed_h=2.25, affected_pore_volume_m3=.001)})
    arguments = dict(expected_phase1_commit=published.job["commit"], fitted_calibration=calibration,
                     fitted_anchor=dict(elapsed_h=2.25, affected_pore_volume_m3=.001), reader=DataReader())
    sources = (published.result["phase1_completion"], published.result["configuration"])
    seal_phase2_configuration(tmp_path / "plain.json", *sources, **arguments)
    seal_phase2_configuration(tmp_path / "none.json", *sources, overflow_phase_split=None, **arguments)
    assert (tmp_path / "plain.json").read_bytes() == (tmp_path / "none.json").read_bytes()
    assert "overflow_phase_split" not in json.loads((tmp_path / "plain.json").read_bytes())
    case = published.task["ordered_case_ids"][0]
    with pytest.raises(ValueError, match="stored gas probabilities"):
        seal_phase2_configuration(tmp_path / "split.json", *sources, overflow_phase_split=dict(
            policy="unit", targets={case: dict(kind="none", reason="unit")}), **arguments)
    from darsia.fluidflower.mass.overflow_phase_split import POLICY
    from darsia.fluidflower.pipeline.staged_data import validate_phase2_run
    config = json.loads((tmp_path / "plain.json").read_bytes())
    image = dict(target=None, reason="No comparable donor", probability_roles={}, phase_source={})
    config["overflow_phase_split"] = dict(policy=POLICY, frames={
        spec["case_id"]: dict(image) for spec in config["frames"]})
    assert validate_phase2_run(config) is config
    config["overflow_phase_split"]["frames"] = {"another-case": dict(image)}
    with pytest.raises(ValueError, match="exact phase-two case order"):
        validate_phase2_run(config)
