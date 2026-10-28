"""Preparation options and original P1 lineage reach the unchanged plan seal.

Capture tests isolate forwarding and pre-writer refusals. The integrated tests
use the original writer/sealer and B6's owned-publication fixture: its local
queue publication is real, but ownership and undecoded payloads are explicitly
synthetic. These tests do not qualify science, hosts, workers or RAW reproduction.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from darsia.fluidflower import analysis_preparation as preparation
from darsia.fluidflower.pipeline import reproduction_lineage as lineage
from darsia.fluidflower.pipeline.job_queue import descriptor
from darsia.fluidflower.storage_outputs import LA_ENCODING, REQUEST_FORMAT

from .test_phase1_enqueue_alias_master import owned_publication


def put(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return descriptor(path)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    series = tmp_path / "series"
    series.mkdir()
    (series / "series.toml").write_text('series_id="unit"\n', encoding="utf-8")
    config = put(tmp_path / "phase1.json", dict(run="run1"))
    calibration = put(tmp_path / "frozen.json", dict(synthetic=True))
    row = dict(run="run1", completion_bound=config, configuration_bound=config,
        expected_phase1_commit="a" * 40, fitted_anchor=dict(elapsed_h=1., affected_pore_volume_m3=.1),
        storage_data=dict(format=REQUEST_FORMAT, gas_score_encoding=LA_ENCODING,
                          resolution_mm=.5, extent_m=[0., 1., 0., 2.]),
        pixel_domain_data=dict(synthetic="preserve exact N16 sources"),
        storage_profile=dict(profile="full", colour_intermediates="keep", store_colour_coordinate=True))
    document = dict(format="darsia-analysis-phase2-inputs-v1",
                    series_manifest=descriptor(series / "series.toml"), runs=[row])
    calls = []
    monkeypatch.setattr(preparation, "write_phase2_plan",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or {"synthetic": True})
    return dict(series=series, document=document, calibration=calibration, calls=calls,
                source=tmp_path / "inputs.json", folder=tmp_path / "fresh")


def test_new_plan_uses_original_sealer_with_explicit_profile_and_storage_size(inputs):
    data = inputs
    original = deepcopy(data["document"])
    bound = put(data["source"], original)
    result = preparation.prepare_phase2_inputs(bound, data["folder"], series_root=data["series"],
        fitted_calibration=data["calibration"], job_id="production", job=dict(precision="float32"),
        storage_profile="lean", stored_resolution=1.)
    assert result == {"synthetic": True} and len(data["calls"]) == 1
    args, kwargs = data["calls"][0]
    source = args[1][0]
    expected = deepcopy(original["runs"][0])
    expected.pop("run")
    expected["fitted_calibration"] = data["calibration"]
    expected["storage_profile"]["profile"] = "lean"
    expected["storage_data"]["resolution_mm"] = 1.
    assert source == expected
    assert kwargs["resources"] == {"run1": dict(backend="cpu", threads=1)}
    assert kwargs["job"]["precision"] == "float32"
    assert descriptor(data["source"]) == bound and json.loads(data["source"].read_text()) == original
    assert not data["folder"].exists()


def test_default_preparation_retains_saved_options_and_missing_extent_blocks_before_sealer(inputs):
    data = inputs
    bound = put(data["source"], data["document"])
    kwargs = dict(series_root=data["series"], fitted_calibration=data["calibration"],
                  job_id="production", job=dict(precision="float32"))
    preparation.prepare_phase2_inputs(bound, data["folder"], **kwargs)
    source = data["calls"][0][0][1][0]
    for key in ("storage_data", "storage_profile", "pixel_domain_data"):
        assert source[key] == data["document"]["runs"][0][key]
    data["calls"].clear()
    data["document"]["runs"][0]["storage_data"].pop("extent_m")
    bound = put(data["source"], data["document"])
    with pytest.raises(ValueError, match="extent"):
        preparation.prepare_phase2_inputs(bound, data["folder"], stored_resolution=2., **kwargs)
    assert not data["calls"] and not data["folder"].exists()


@pytest.fixture
def published_inputs(tmp_path):
    """Existing original terminal publication with synthetic, undecoded payloads."""
    published = owned_publication(tmp_path / "publication", complete=True)
    series = tmp_path / "series"
    series.mkdir()
    (series / "series.toml").write_text('series_id="Synthetic"\n', encoding="utf-8")
    row = dict(run=published.task["run"],
        completion_bound=deepcopy(published.result["phase1_completion"]),
        configuration_bound=deepcopy(published.result["configuration"]),
        expected_phase1_commit=published.job["commit"],
        fitted_anchor=dict(elapsed_h=1., affected_pore_volume_m3=.001),
        storage_data=dict(format=REQUEST_FORMAT, gas_score_encoding=LA_ENCODING,
                          resolution_mm=.5, extent_m=[0., 1., 0., 1.]),
        storage_profile=dict(profile="full", colour_intermediates="keep",
                             store_colour_coordinate=True))
    document = dict(format="darsia-analysis-phase2-inputs-v1",
                    series_manifest=descriptor(series / "series.toml"), runs=[row])
    source = tmp_path / "inputs.json"
    bound = put(source, document)
    jobs = dict(format="darsia-phase1-reproduction-jobs-v1",
                runs={row["run"]: descriptor(published.root / "job.json")})
    jobs_path = tmp_path / "original-jobs.json"
    put(jobs_path, jobs)
    jobs_bound = preparation.source_metadata_descriptor(jobs_path)
    origin = lineage.bind_phase1_origin(jobs["runs"][row["run"]], row["run"],
        configuration=row["configuration_bound"], completion=row["completion_bound"],
        expected_commit=row["expected_phase1_commit"])
    # Plan metadata only: no repository/host admission is asserted by this lock.
    job = dict(commit=published.job["commit"], precision="float32",
        environment_lock=dict(path="requirements-worker.lock", sha256="a" * 64),
        created_utc="2026-10-23T12:00:00+00:00")
    calibration = descriptor(published.claim.output / "new_fitted_calibration.json")
    return dict(root=tmp_path, published=published, series=series, row=row,
        document=document, source=source, bound=bound, jobs=jobs, jobs_path=jobs_path,
        jobs_bound=jobs_bound, origin=origin, calibration=calibration,
        folder=tmp_path / "production", job=job)


def _prepare_arguments(data):
    return dict(series_root=data["series"], fitted_calibration=data["calibration"],
                job_id="production", job=deepcopy(data["job"]))


def _existing_bytes(data):
    return {path: path.read_bytes() for path in data["root"].rglob("*") if path.is_file()}


def _assert_unchanged(original):
    assert {path: path.read_bytes() for path in original} == original


def test_real_prepare_writer_and_seal_join_origin_and_storage_only(published_inputs):
    """No writer, sealer, DATA reader or original lineage verifier is replaced."""
    data = published_inputs
    original = _existing_bytes(data)
    source_before = deepcopy(data["document"])
    arguments = _prepare_arguments(data)
    baseline = preparation.prepare_phase2_inputs(data["bound"], data["root"] / "baseline", **arguments)
    combined = preparation.prepare_phase2_inputs(data["bound"], data["folder"],
        phase1_jobs=data["jobs_bound"], storage_profile="lean", stored_resolution=1., **arguments)
    configurations = []
    for result in (baseline, combined):
        assert result["pipeline"] == "n1"
        assert result["adopted_for_execution"] is False
        assert result["job_created"] is False and result["worker_started"] is False
        assert descriptor(Path(result["plan"]["path"])) == result["plan"]
        assert len(result["runs"]) == 1 and result["runs"][0]["run"] == data["row"]["run"]
        row = result["runs"][0]
        assert row["resources"] == dict(backend="cpu", threads=1)
        assert descriptor(Path(row["configuration"]["path"])) == row["configuration"]
        configurations.append(json.loads(Path(row["configuration"]["path"]).read_bytes()))
    saved, changed = configurations
    assert "origin" not in saved["phase1"]
    expected = deepcopy(saved)
    expected["phase1"]["origin"] = deepcopy(data["origin"])
    expected["storage_profile"]["profile"] = "lean"
    expected["storage"]["resolution_mm"] = 1.
    assert changed == expected  # Every other scientific field/descriptor is exact.
    assert changed["calibration"] == data["calibration"]
    assert changed["anchor"] == data["row"]["fitted_anchor"]
    assert changed["precision"] == "float32"
    assert changed["phase1"]["configuration"] == data["row"]["configuration_bound"]
    assert changed["phase1"]["completion"] == data["row"]["completion_bound"]
    assert changed["storage_profile"]["store_colour_coordinate"] is True
    assert json.loads(data["source"].read_bytes()) == source_before
    _assert_unchanged(original)


def test_existing_origin_and_frozen_refit_are_forwarded_exactly(published_inputs, monkeypatch):
    """Capture proves preservation; it does not certify this synthetic refit."""
    data = published_inputs
    data["row"]["phase1_origin"] = deepcopy(data["origin"])
    data["row"]["frozen_refit"] = dict(completion=deepcopy(data["calibration"]),
                                      plan=deepcopy(data["row"]["configuration_bound"]))
    bound = put(data["source"], data["document"])
    original = _existing_bytes(data)
    calls = []
    monkeypatch.setattr(preparation, "write_phase2_plan",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"captured": True})
    preparation.prepare_phase2_inputs(bound, data["folder"], phase1_jobs=data["jobs_bound"],
        storage_profile="lean", stored_resolution=1., **_prepare_arguments(data))
    assert len(calls) == 1
    source = calls[0][0][1][0]
    expected = deepcopy(data["row"])
    expected.pop("run")
    expected["fitted_calibration"] = data["calibration"]
    expected["storage_profile"]["profile"] = "lean"
    expected["storage_data"]["resolution_mm"] = 1.
    assert source == expected
    assert not data["folder"].exists()
    assert not data["folder"].with_name(data["folder"].name + "_sealed").exists()
    _assert_unchanged(original)


@pytest.mark.parametrize("fault", [
    "mapping_format", "missing_run", "extra_run", "job_sha", "job_size",
    "wrong_commit", "wrong_configuration", "wrong_completion", "conflicting_origin",
    "missing_extent", "malformed_extent", "drop_profile",
])
def test_mapping_binding_and_storage_refusals_precede_writer(published_inputs, monkeypatch, fault):
    data = published_inputs
    row, jobs = data["row"], data["jobs"]
    if fault == "mapping_format":
        jobs["format"] = "unsupported"
    elif fault == "missing_run":
        jobs["runs"].clear()
    elif fault == "extra_run":
        jobs["runs"]["another-run"] = deepcopy(jobs["runs"][row["run"]])
    elif fault == "job_sha":
        jobs["runs"][row["run"]]["sha256"] = "0" * 64
    elif fault == "job_size":
        jobs["runs"][row["run"]]["size_bytes"] += 1
    elif fault == "wrong_commit":
        row["expected_phase1_commit"] = "2" * 40
    elif fault == "wrong_configuration":
        row["configuration_bound"] = deepcopy(data["calibration"])
    elif fault == "wrong_completion":
        row["completion_bound"] = deepcopy(data["calibration"])
    elif fault == "conflicting_origin":
        row["phase1_origin"] = deepcopy(data["origin"])
        row["phase1_origin"]["task"]["sha256"] = "0" * 64
    elif fault == "missing_extent":
        row["storage_data"].pop("extent_m")
    elif fault == "malformed_extent":
        row["storage_data"]["extent_m"] = [0., 0., 0., 1.]
    else:
        assert fault == "drop_profile"
        row["storage_profile"].update(profile="lean", colour_intermediates="drop")
    bound = put(data["source"], data["document"])
    jobs_bound = put(data["jobs_path"], jobs)
    original = _existing_bytes(data)
    calls = []
    monkeypatch.setattr(preparation, "write_phase2_plan",
        lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(ValueError):
        preparation.prepare_phase2_inputs(bound, data["folder"], phase1_jobs=jobs_bound,
            storage_profile="lean", stored_resolution=1., **_prepare_arguments(data))
    assert calls == []
    assert not data["folder"].exists()
    assert not data["folder"].with_name(data["folder"].name + "_sealed").exists()
    _assert_unchanged(original)


@pytest.mark.parametrize("field,reason", [
    ("phase1_origin", "Explicit original phase-one reproduction origin required"),
    ("frozen_refit", "genuine global refit binding fields differ"),
])
def test_carried_lineage_guards_still_refuse_in_real_sealer(published_inputs, field, reason):
    data = published_inputs
    data["row"][field] = {}
    bound = put(data["source"], data["document"])
    original = _existing_bytes(data)
    with pytest.raises(ValueError, match=reason):
        preparation.prepare_phase2_inputs(bound, data["folder"],
            storage_profile="lean", stored_resolution=1., **_prepare_arguments(data))
    # The original writer creates this fresh namespace before seal validation.
    # This is deliberately not a claim that all failures precede mutation.
    sealed = data["folder"].with_name(data["folder"].name + "_sealed")
    assert sealed.is_dir() and not list(sealed.iterdir())
    assert not data["folder"].exists()
    _assert_unchanged(original)
