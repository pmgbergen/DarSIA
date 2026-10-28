"""Saved metadata specimens through original rooted source/plan admission.

All selected metadata files are real bound files under a schema-1 series root.
The nested native/phase payload descriptors are finite metadata specimens, not
campaign products or a claim of RAW, geometry, optics or physical qualification.
No source/geometry/pressure validator or input catalogue is substituted.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import analysis_preparation as preparation
from darsia.fluidflower.jobs_metadata import JobSnapshotReader
from darsia.fluidflower.pipeline.job_queue import descriptor

from .test_analysis_plan import _job, _original
from .test_staged_data import phase1_data  # noqa: F401


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, allow_nan=False) + "\n").encode("utf-8"))
    return descriptor(path)


@pytest.fixture
def saved_products(tmp_path, phase1_data):
    root = tmp_path / "series"
    root.mkdir()
    runs = ["run-one", "run-two"]
    (root / "series.toml").write_text(
        'schema_version = 1\nseries_id = "testonly"\n'
        '[roots]\nseries = "."\n[runs]\nids = ["run-one", "run-unused", "run-two"]\n',
        encoding="utf-8")
    method = _save(root / "method.json", {key: phase1_data["statistics"]["bindings"][key]
        for key in ("round4_spec_sha256", "amendment_sha256")})
    configurations, documents, geometry_runs, pressure_runs = [], {}, {}, {}
    for run in runs:
        data = deepcopy(phase1_data)
        data["run"] = data["frames"][0]["source"]["run"] = run
        original = _original(data)
        original["mass"]["calibration"] = deepcopy(method)
        original["mass"]["template"]["calibration"] = deepcopy(method)
        configuration = _save(root / run / "configuration.json", original)
        configurations.append(configuration)
        frame = original["frames"][0]
        measured = dict(format="darsia-measured-frame-geometry-v1", run=run,
            case_id=frame["case_id"], original_geometry=deepcopy(frame["geometry"]),
            source_to_frame_native=deepcopy(frame["phase_roles"]["reference_to_current_h"]),
            worker_geometry=deepcopy(frame["geometry"]), phase_area=deepcopy(frame["phase_roles"]["area"]),
            mass_shared_reference_matrix=deepcopy(data["statistics"]["shared"]["reference_matrix"]))
        measured_bound = _save(root / run / "frame.json", measured)
        frames = dict(run=run, complete_for_this_run=True,
            frames={frame["case_id"]: dict(frame_model=measured_bound)})
        frame_index = _save(root / run / "frames.json", frames)
        geometry_runs[run] = dict(frame_data=frame_index)
        atmosphere = dict(run=run, complete=True, provisional_gap_default=False,
            rows=[dict(time_hours=t, pressure_bar=p, temperature_c=23.,
                       pressure_gradient_bar_per_m=0., temperature_gradient_c_per_m=0.)
                  for t, p in ((0., 1.), (2., 1.2))])
        atmosphere_bound = _save(root / run / "atmosphere.json", atmosphere)
        pressure_runs[run] = dict(atmosphere_series=atmosphere_bound)
        documents[run] = SimpleNamespace(original=original, measured=measured,
            measured_bound=measured_bound, frames=frames, frame_index=frame_index,
            atmosphere=atmosphere, atmosphere_bound=atmosphere_bound)
    geometry_document = dict(format="darsia-measured-run-geometry-v1",
        reference_models_complete=True, runs=geometry_runs)
    pressure_document = dict(runs={"testonly": pressure_runs})
    geometry = _save(root / "geometry.json", geometry_document)
    pressure = _save(root / "pressure.json", pressure_document)
    return SimpleNamespace(root=root, configurations=configurations,
        geometry=geometry, pressure=pressure, data=phase1_data, runs=runs,
        documents=documents, geometry_document=geometry_document,
        pressure_document=pressure_document)


def _collect(case, destination):
    return preparation.collect_phase1_source_inputs(case.configurations, destination,
        series_root=case.root, geometry_index=case.geometry, pressure_index=case.pressure,
        gas_probability=True)


def test_saved_metadata_collects_exact_inputs_and_roundtrips_original_prepare_without_starting(
        tmp_path, saved_products):
    case = saved_products
    before = {path: path.read_bytes() for path in case.root.rglob("*") if path.is_file()}
    configurations = deepcopy(case.configurations)
    assert [preparation.source_metadata_descriptor(row["path"])
            for row in configurations] == configurations
    destination = tmp_path / "saved-inputs.json"
    result = _collect(case, destination)
    assert result == dict(inputs=descriptor(destination), runs=case.runs,
                          plan_prepared=False, worker_started=False)
    # A registered but unselected run needs no invented products. Preserve the
    # chosen cohort in registered relative order, like accepted39 of registered43.
    assert result["runs"] == ["run-one", "run-two"] and "run-unused" not in result["runs"]
    manifest = JobSnapshotReader().read(destination, result["inputs"])
    assert manifest == dict(format="darsia-analysis-plan-inputs-v1",
        series_manifest=descriptor(case.root / "series.toml"),
        runs=[dict(run=run, configuration=bound) for run, bound in zip(case.runs, configurations)])
    assert not list(tmp_path.rglob("execution.toml")) and not list(tmp_path.rglob("job.json"))

    prepared = preparation.prepare_phase1_inputs(result["inputs"], tmp_path / "prepared",
        series_root=case.root, geometry_index=case.geometry, pressure_index=case.pressure,
        gas_probability=True, job_id="testonly", job=_job())
    assert [row["run"] for row in prepared["runs"]] == case.runs
    for row in prepared["runs"]:
        configuration = JobSnapshotReader().read(Path(row["configuration"]["path"]), row["configuration"])
        original = case.documents[row["run"]].original
        assert configuration["ordered_case_ids"] == original["ordered_case_ids"]
        assert configuration["frames"][0]["native_task"] == original["frames"][0]["native_task"]
        assert configuration["frames"][0]["source"] == original["frames"][0]["source"]
        assert configuration["frames"][0]["mass_spec"]["atmosphere"]["pressure_bar"] == 1.1
        assert configuration["provenance"]["analysis_plan_inputs"]["source_manifest"] == result["inputs"]
        assert "gas_probability_storage" in configuration
    assert prepared["adopted_for_execution"] is False and prepared["worker_started"] is False
    assert case.configurations == configurations
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert not list(tmp_path.rglob("job.json"))


@pytest.mark.parametrize("fault", ["missing_frame", "native_matrix", "area_route", "provisional_pressure"])
def test_original_missing_or_changed_products_refuse_before_manifest_publication(
        tmp_path, saved_products, fault):
    case = saved_products
    docs = case.documents["run-one"]
    if fault == "missing_frame":
        docs.frames["frames"] = {}
    elif fault in {"native_matrix", "area_route"}:
        if fault == "native_matrix":
            docs.measured["source_to_frame_native"][0][2] = 2
        else:
            docs.measured["phase_area"]["role"] = "measured_parent"
        measured = _save(docs.measured_bound["path"], docs.measured)
        docs.frames["frames"]["case-one"]["frame_model"] = measured
    else:
        docs.atmosphere["provisional_gap_default"] = True
        atmosphere = _save(docs.atmosphere_bound["path"], docs.atmosphere)
        case.pressure_document["runs"]["testonly"]["run-one"]["atmosphere_series"] = atmosphere
        case.pressure = _save(case.pressure["path"], case.pressure_document)
    if fault != "provisional_pressure":
        frames = _save(docs.frame_index["path"], docs.frames)
        case.geometry_document["runs"]["run-one"]["frame_data"] = frames
        case.geometry = _save(case.geometry["path"], case.geometry_document)
    expected = {"missing_frame": "complete selected order", "native_matrix": "native positioning",
                "area_route": "area route", "provisional_pressure": "Provisional pressure gaps"}[fault]
    destination = tmp_path / "refused-inputs.json"
    with pytest.raises(ValueError, match=expected):
        _collect(case, destination)
    assert not destination.exists() and not list(tmp_path.rglob("execution.toml"))
    assert not list(tmp_path.rglob("job.json"))


def test_changed_selected_configuration_bytes_fail_the_original_descriptor_guard(tmp_path, saved_products):
    case = saved_products
    path = Path(case.configurations[0]["path"])
    original = path.read_bytes()
    changed = original.replace(b'"run-one"', b'"run-ome"', 1)
    assert changed != original and len(changed) == len(original)
    path.write_bytes(changed)
    destination = tmp_path / "refused-tampered-inputs.json"
    with pytest.raises(ValueError, match="Published metadata SHA/size differs"):
        _collect(case, destination)
    assert not destination.exists() and path.read_bytes() == changed


@pytest.mark.parametrize("selection", ["reordered", "duplicate"])
def test_original_registered_run_order_and_distinct_sources_are_preserved(
        tmp_path, saved_products, selection):
    case = saved_products
    case.configurations = (list(reversed(case.configurations)) if selection == "reordered"
        else [case.configurations[0], case.configurations[0]])
    destination = tmp_path / "refused-order.json"
    expected = "registered run order" if selection == "reordered" else "distinct registered runs"
    with pytest.raises(ValueError, match=expected):
        _collect(case, destination)
    assert not destination.exists() and not list(tmp_path.rglob("job.json"))


def test_source_collection_never_overwrites_a_source_or_an_existing_manifest(tmp_path, saved_products):
    case = saved_products
    source = Path(case.configurations[0]["path"])
    before = source.read_bytes()
    with pytest.raises(FileExistsError, match="fresh source-input manifest"):
        _collect(case, source)
    assert source.read_bytes() == before
    destination = tmp_path / "immutable-inputs.json"
    _collect(case, destination)
    manifest_bytes = destination.read_bytes()
    with pytest.raises(FileExistsError, match="fresh source-input manifest"):
        _collect(case, destination)
    assert destination.read_bytes() == manifest_bytes
    assert not list(tmp_path.rglob("execution.toml")) and not list(tmp_path.rglob("job.json"))
