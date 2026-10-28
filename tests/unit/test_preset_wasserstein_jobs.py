"""Original metadata compiler through actual queue and per-read role replay.

Private tiny fixtures and synthetic source/filesystem admission only. These
cases are unrun source, not an operational or scientific qualification.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import analysis_preparation, job_filesystem, storage_paths
from darsia.fluidflower.preset_wasserstein_jobs import prepare_preset_wasserstein_job
from darsia.fluidflower.pipeline.job_queue import LocalOutputBinding, read_json
from darsia.fluidflower.pipeline.run_io import RunIOBinding
from darsia.fluidflower.pipeline.store_runtime import resolve_catalog, verify_inputs
from darsia.fluidflower.store_map import parse_store_map
from .test_preset_wasserstein_inputs import write_original_preset_fixture


@pytest.fixture
def original_job_source(tmp_path, monkeypatch):
    fixture = write_original_preset_fixture(tmp_path)
    root = fixture["config_path"].parent
    fixture["series_manifest"].write_text(
        'schema_version = 1\nseries_id = "campaign"\n[roots]\nseries = "."\n'
        '[runs]\nids = ["z", "a"]\n', encoding="utf-8")
    monkeypatch.setattr(storage_paths, "_alias_rows", lambda: [])
    identity = dict(commit="1" * 40, precision="float32", created_utc="2026-10-23T00:00:00+00:00",
        environment_lock=dict(path="requirements-worker.lock", sha256="2" * 64))
    monkeypatch.setattr(analysis_preparation, "job_identity", lambda repository: deepcopy(identity))
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda path:
        SimpleNamespace(allowed=True, reason="synthetic", evidence="not a transport qualification"))
    return SimpleNamespace(fixture=fixture, root=root, metadata=root / "prepared",
                           job_root=tmp_path / "shared-job", repository=tmp_path / "source")


def _prepare(case):
    return prepare_preset_wasserstein_job(case.fixture["config_path"], case.metadata, case.job_root,
        series_root=case.root, series_id="campaign", job_id="comparison", actor="Test operator",
        reason="Exercise original fixture inputs", repository=case.repository)


def _binding(case, task, tmp_path):
    results_root = tmp_path / "local-results"
    results_root.mkdir()
    stores = parse_store_map(('[machine]\nid="fixture"\n[stores.campaign]\nseries='
        + json.dumps(str(case.root)) + '\nresults=' + json.dumps(str(results_root)) + '\n').encode())
    rows = resolve_catalog(task, stores, "campaign")
    verified = verify_inputs(task, rows)
    output = results_root / task["job_id"] / task["run"] / ".attempts" / task["attempt_id"]
    admission = job_filesystem.LocalFileAdmission(True, "synthetic", "unit fixture",
        "Not an operational filesystem qualification", str(results_root))
    local = LocalOutputBinding("fixture", results_root, output, admission)
    return RunIOBinding.replay(task, verified, local, repository=case.repository,
                               admitted_commit=task["commit"])


def test_real_original_compiler_catalogue_and_queue_make_one_exact_time(original_job_source, monkeypatch):
    case = original_job_source
    import darsia
    monkeypatch.setattr(darsia, "imread", lambda *args, **kwargs: pytest.fail("Preparation decoded a raster"))
    result = _prepare(case)
    job = read_json(Path(result["job"]["path"]))
    queued = list((case.job_root / "queue").glob("*.json"))
    assert len(queued) == 1 and result["worker_started"] is False
    task = read_json(queued[0])
    assert job["pipeline"] == task["pipeline"] == "preset-wasserstein"
    assert job["precision"] == task["precision"] == "original"
    assert "calibration_sha256" not in job and "calibration_sha256" not in task
    assert task["ordered_case_ids"] == [task["run"]]
    assert task["comparison_time"] == dict(order=0, time_hours=1., tolerance_hours=0.1)
    assert result["comparison_count"] == 1 and result["reference_run"] == "z"
    assert job["created_by"] == "Test operator"
    assert not (case.root / "uncreated-comparison").exists()
    assert not (case.root / "results-z" / "cache").exists()


def test_real_replay_preserves_original_control_and_binds_every_original_input(original_job_source, tmp_path):
    case = original_job_source
    _prepare(case)
    task = read_json(next((case.job_root / "queue").glob("*.json")))
    binding = _binding(case, task, tmp_path)
    controls = binding.configuration["preset_inputs"]
    assert controls["runs"] == ["z", "a"] and controls["reference_run"] == "z"
    assert all(row["descriptor"]["sha256"] for row in controls["files"])
    with binding.activate():
        from darsia.fluidflower.preset_wasserstein_inputs import load_preset_wasserstein_runtime
        with load_preset_wasserstein_runtime(task) as runtime:
            assert runtime.reference_rig.baseline.img.dtype.name == "float32"
            for run in controls["runs"]:
                image = runtime.load_mass(runtime.config.runs.config[run], data="mass", time=1., tol=0.1)
                assert image.img.dtype.name == "float32"
            consumed = runtime.verify_consumed()
            assert consumed["reference_run"] == "z"
            paths = {row["descriptor"]["path"] for row in consumed["files"]}
            assert all(str(path) in paths for path in case.fixture["mass_paths"].values())


def test_missing_bound_source_fails_real_role_replay(original_job_source, tmp_path):
    case = original_job_source
    _prepare(case)
    task = read_json(next((case.job_root / "queue").glob("*.json")))
    task["input_catalog"]["inputs"] = [row for row in task["input_catalog"]["inputs"]
        if row["reference"]["role"] != "image"]
    with pytest.raises(ValueError):
        _binding(case, task, tmp_path)


def test_changed_bound_mass_is_rejected_by_actual_original_reader(original_job_source, tmp_path):
    case = original_job_source
    _prepare(case)
    task = read_json(next((case.job_root / "queue").glob("*.json")))
    binding = _binding(case, task, tmp_path)
    with binding.activate():
        from darsia.fluidflower.preset_wasserstein_inputs import load_preset_wasserstein_runtime
        with load_preset_wasserstein_runtime(task) as runtime:
            path = case.fixture["mass_paths"]["z"]
            raw = path.read_bytes()
            path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            with pytest.raises(ValueError, match="changed"):
                runtime.load_mass(runtime.config.runs.config["z"], data="mass", time=1., tol=0.1)


def test_unmapped_future_metadata_refuses_before_any_job_or_metadata_write(original_job_source, tmp_path):
    case = original_job_source
    case.metadata = tmp_path / "outside-registered-input-roots"
    with pytest.raises(ValueError, match="No declared root"):
        _prepare(case)
    assert not case.metadata.exists() and not case.job_root.exists()
