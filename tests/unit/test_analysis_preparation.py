"""Input-manifest compilation with a metadata-only catalogue protocol double."""
from copy import deepcopy
from pathlib import Path

import pytest

from darsia.fluidflower import analysis_preparation as preparation
from darsia.fluidflower.jobs_metadata import JobSnapshotReader
from darsia.fluidflower.pipeline.job_queue import descriptor
from .test_analysis_plan import _original, _job
from .test_staged_data import phase1_data, _write


def inputs(tmp_path, data, monkeypatch, *, original=None):
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text('series_id = "testonly"\n', encoding="utf-8")
    manifest = descriptor(root / "series.toml")
    original = _original(data) if original is None else deepcopy(original)
    method = {key: data["statistics"]["bindings"][key] for key in ("round4_spec_sha256", "amendment_sha256")}
    calibration = _write(tmp_path / "method.json", method)
    original["mass"]["calibration"] = calibration
    original["mass"]["template"]["calibration"] = calibration
    original_bound = _write(tmp_path / "run.json", original)
    frame = data["frames"][0]
    measured = dict(format="darsia-measured-frame-geometry-v1", run=data["run"], case_id=frame["case_id"],
        original_geometry=deepcopy(frame["geometry"]), source_to_frame_native=deepcopy(frame["phase_roles"]["reference_to_current_h"]),
        worker_geometry=deepcopy(frame["geometry"]), phase_area=deepcopy(frame["phase_roles"]["area"]),
        mass_shared_reference_matrix=deepcopy(data["statistics"]["shared"]["reference_matrix"]))
    measured_bound = _write(tmp_path / "frame.json", measured)
    frame_index = _write(tmp_path / "frames.json", dict(run=data["run"], complete_for_this_run=True,
        frames={frame["case_id"]: dict(frame_model=measured_bound)}))
    geometry = _write(tmp_path / "geometry.json", dict(format="darsia-measured-run-geometry-v1",
        reference_models_complete=True, runs={data["run"]: dict(frame_data=frame_index)}))
    atmosphere = _write(tmp_path / "atmosphere.json", dict(run=data["run"], complete=True, provisional_gap_default=False,
        rows=[dict(time_hours=t, pressure_bar=p, temperature_c=23., pressure_gradient_bar_per_m=0., temperature_gradient_c_per_m=0.)
              for t, p in [(0., 1.), (2., 1.2)]]))
    pressure = _write(tmp_path / "pressure.json", dict(runs={"testonly": {data["run"]: dict(atmosphere_series=atmosphere)}}))
    manifest_inputs = _write(tmp_path / "inputs.json", dict(format="darsia-analysis-plan-inputs-v1",
        series_manifest=manifest, runs=[dict(run=data["run"], configuration=original_bound)]))
    from darsia.fluidflower.pipeline import alias_job, job_inputs
    monkeypatch.setattr(alias_job, "_manifest", lambda bound: (Path(bound["path"]), dict(series_id="testonly"), {}))
    class MetadataCatalogue:
        def __init__(self, *args, **kwargs):
            pass
        def doc(self, bound, *args):
            return JobSnapshotReader().read(Path(bound["path"]), bound)
    monkeypatch.setattr(job_inputs, "InputCatalog", MetadataCatalogue)
    return root, manifest_inputs, geometry, pressure


def test_compiler_writes_one_inert_plan_with_running_values_and_explicit_sources(tmp_path, phase1_data, monkeypatch):
    root, source, geometry, pressure = inputs(tmp_path, phase1_data, monkeypatch)
    result = preparation.prepare_phase1_inputs(source, tmp_path / "prepared", series_root=root,
        geometry_index=geometry, pressure_index=pressure, gas_probability=True, job_id="testonly", job=_job())
    config = JobSnapshotReader().read(Path(result["runs"][0]["configuration"]["path"]), result["runs"][0]["configuration"])
    assert config["frames"][0]["mass_spec"]["atmosphere"]["pressure_bar"] == 1.1
    assert config["provenance"]["analysis_plan_inputs"]["source_manifest"] == source
    assert "gas_probability_storage" in config
    assert result["adopted_for_execution"] is False and result["worker_started"] is False
    assert not list(tmp_path.rglob("job.json"))


def test_missing_geometry_is_named_per_run_and_writes_no_plan(tmp_path, phase1_data, monkeypatch):
    root, source, geometry, pressure = inputs(tmp_path, phase1_data, monkeypatch)
    import json
    path = Path(geometry["path"])
    value = json.loads(path.read_bytes())
    value["runs"][phase1_data["run"]] = {}
    path.write_text(json.dumps(value), encoding="utf-8")
    target = tmp_path / "prepared"
    with pytest.raises(ValueError, match=phase1_data["run"]):
        preparation.prepare_phase1_inputs(source, target, series_root=root,
            geometry_index=descriptor(path), pressure_index=pressure, gas_probability=False, job_id="testonly", job=_job())
    assert not target.exists()


def test_preparation_binds_selected_metric_model_to_saved_current_build_plan(tmp_path, phase1_data, monkeypatch):
    from .test_analysis_plan import _current_build_original_TESTONLY
    from .test_staged_data import _bound
    import json

    original = _current_build_original_TESTONLY(phase1_data)
    original["current_build_binding"]["metric_model"] = _bound("obsolete-metric.json")
    root, source, geometry, pressure = inputs(tmp_path, phase1_data, monkeypatch, original=original)
    model = _write(tmp_path / "selected-metric.json", dict(
        format="darsia-measured-run-geometry-v1", run=phase1_data["run"]))
    value = json.loads(Path(geometry["path"]).read_bytes())
    value["runs"][phase1_data["run"]]["reference_model"] = model
    geometry = _write(tmp_path / "selected-geometry.json", value)

    result = preparation.prepare_phase1_inputs(source, tmp_path / "prepared", series_root=root,
        geometry_index=geometry, pressure_index=pressure, gas_probability=False,
        job_id="testonly", job=_job())
    bound = result["runs"][0]["configuration"]
    saved = JobSnapshotReader().read(Path(bound["path"]), bound)
    assert saved["current_build_binding"]["metric_model"] == model
    assert saved["current_build_binding"]["operation_result"] == original["current_build_binding"]["operation_result"]
    assert saved["provenance"]["analysis_plan_inputs"]["geometry_index"] == geometry
    source_run = JobSnapshotReader().read(Path(tmp_path / "run.json"), descriptor(tmp_path / "run.json"))
    assert source_run["current_build_binding"]["metric_model"] == _bound("obsolete-metric.json")
    assert not list(tmp_path.rglob("job.json"))


def test_identity_keeps_unrelated_user_files_but_refuses_untracked_source(tmp_path, monkeypatch):
    lock = b"exact-lock\n"
    (tmp_path / "requirements-worker.lock").write_bytes(lock)
    (tmp_path / "launcher.cmd").write_text("user file", encoding="utf-8")
    dirty_source = []
    def git(command, **kwargs):
        if "rev-parse" in command:
            return "a" * 40 + "\n"
        if "status" in command:
            assert "--untracked-files=no" in command
            return ""
        if "ls-files" in command:
            assert command[-1] == "src"
            return "\n".join(dirty_source)
        assert "show" in command
        return lock
    monkeypatch.setattr(preparation.subprocess, "check_output", git)
    result = preparation.job_identity(tmp_path)
    assert result["commit"] == "a" * 40
    assert (tmp_path / "launcher.cmd").read_text() == "user file"
    dirty_source.append("src/unknown.py")
    with pytest.raises(ValueError, match="Commit the analysis source"):
        preparation.job_identity(tmp_path)
