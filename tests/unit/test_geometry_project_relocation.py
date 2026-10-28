"""Read an existing approved geometry package after a shared folder rename."""

from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import toml

from darsia.fluidflower import corrected_setup
from darsia.fluidflower import geometry_raster_sources as sources
from darsia.fluidflower import layer_painting as painting
from darsia.fluidflower import layer_painting_manual as manual
from darsia.fluidflower import painted_adoption_table
from darsia.fluidflower import painting_facies_approval
from darsia.fluidflower import painting_series_use_ui
from darsia.fluidflower.bound_verification_cache import BoundVerificationCache
from darsia.fluidflower.geometry_facies import GeometryRevisionStore, GEOMETRY_REVISION_FORMAT
from darsia.fluidflower.map_placement import GEOMETRY_FORMAT, PLACEMENT_FORMAT, read_placement, read_run_geometry
from darsia.fluidflower.painting_saved_transfers_ui import saved_transfers
from darsia.fluidflower.project import FluidFlowerProject, resolve_project_package_root
from darsia.fluidflower.series_geometry import geometry_package_root, load_series_geometry_project
from darsia.fluidflower.storage_paths import refresh_storage_paths


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return sources.descriptor(path)


def test_existing_geometry_and_indexed_models_move_outside_the_series_root(tmp_path, monkeypatch):
    from darsia.fluidflower import phase_model_library as library
    from darsia.fluidflower.series_paths import register_series

    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    for key in tuple(os.environ):
        if (key.startswith("DARSIA_") and key.endswith("_PROJECT_ROOT")) or key == "DARSIA_FLUIDFLOWER_DATA_ROOT":
            monkeypatch.delenv(key)
    series = tmp_path / "project" / "DarSIA"
    series.mkdir(parents=True)
    document = dict(schema_version=1, series_id="sample", name="Sample", paths=dict(
        geometry_facies="Geometry", phase_model="ExpertKnowledge/PhaseModels",
        models="ExpertKnowledge/PhaseModels/Models",
        phase_model_library="ExpertKnowledge/PhaseModels/Models/.index"))
    manifest = series / "series.toml"
    manifest.write_text(toml.dumps(document), encoding="utf-8")
    register_series(series)
    old_geometry, old_phase = series / "Geometry", series / "ExpertKnowledge/PhaseModels"
    project = FluidFlowerProject.create(old_geometry, name="Saved geometry", series_id="sample",
        template_id="custom", rig_type="custom")
    current = dict(format=GEOMETRY_REVISION_FORMAT, revision_id="saved-revision", configuration={})
    current_path = GeometryRevisionStore(project).current_path
    current_binding = _write(current_path, current)
    model = old_phase / "Models" / "phase-model-sample"
    _write(model / "summary.json", dict(model_id=model.name))
    for head in ("co2_presence", "gas_within_co2"):
        _write(model / "models" / "generic" / (head + "_signal.json"), dict(head=head))
    identity = library.source_identity(model)
    pointer = library.index_model(series, model)
    frozen_pointer, frozen_current = pointer.read_bytes(), current_path.read_bytes()
    results = tmp_path / "project" / "Results"
    geometry, phase = results / "setup/Geometry", results / "calibration/ExpertKnowledge/PhaseModels"
    geometry.parent.mkdir(parents=True)
    phase.parent.mkdir(parents=True)
    old_geometry.rename(geometry)
    old_phase.rename(phase)
    document["paths"].update(geometry_facies="../Results/setup/Geometry",
        phase_model="../Results/calibration/ExpertKnowledge/PhaseModels",
        models="../Results/calibration/ExpertKnowledge/PhaseModels/Models",
        phase_model_library="../Results/calibration/ExpertKnowledge/PhaseModels/Models/.index")
    document["storage"] = dict(path_aliases={str(old_geometry): str(geometry), str(old_phase): str(phase)})
    manifest.write_text(toml.dumps(document), encoding="utf-8")
    refresh_storage_paths(series)

    reopened = load_series_geometry_project(series)
    assert reopened.root == geometry and reopened.project_id == project.project_id
    assert GeometryRevisionStore(reopened).current() == current
    assert painted_adoption_table._checked(current_binding, metadata=True) == current
    moved_model = phase / "Models" / model.name
    moved_pointer = phase / "Models/.index" / pointer.name
    assert library.resolve_reference(pointer) == moved_model
    assert library.resolve_reference(moved_pointer) == moved_model
    for selected in (phase / "Models", moved_pointer.parent):
        records = library.records_for_location(selected)
        assert len(records) == 1 and records[0]["available"]
        assert records[0]["source_identity"] == identity
    assert moved_pointer.read_bytes() == frozen_pointer
    assert GeometryRevisionStore(reopened).current_path.read_bytes() == frozen_current


@pytest.mark.parametrize("kind", ["manual", "proposal"])
def test_approved_map_transfer_and_current_revision_survive_geometry_rename(
        tmp_path, monkeypatch, kind):
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    series = tmp_path / "project" / "DarSIA"
    series.mkdir(parents=True)
    manifest = series / "series.toml"
    document = dict(schema_version=1, paths=dict(geometry_facies="geometry_facies"))
    manifest.write_text(toml.dumps(document), encoding="utf-8")
    old = series / "geometry_facies"
    project = FluidFlowerProject.create(old, name="Synthetic geometry", series_id="sample",
        template_id="custom", rig_type="custom")
    session = painting.PaintingSession(run_id="sample", image_size=(3, 2))
    session.layers = [painting.PaintedLayer(1, "Sand", "#888888", "Sand")]
    session = painting.PaintingSession.from_dict(session.to_dict())
    # A tiny already painted map, with no classification or registration.
    pixels = np.ones((2, 3), dtype=np.uint8)
    if kind == "manual":
        value = manual.ManualMap(pixels, np.ones((2, 3), bool), np.zeros((2, 3), bool), (3, 2))
        record = manual.save_manual_version(project, session, value, name="Saved map")
        approval = manual.approve_manual_version(record)
        reference = painting_facies_approval.approved_reference(record, approval)
    else:
        value = painting.LayerProposal(pixels, np.full((2, 3), 255, np.uint8), (3, 2), "source")
        record = painting.save_version(project, session, value, [], name="Saved map")
        approval = painting.approve_version(record)
    version = dict(deepcopy(record), approval=deepcopy(approval))
    model = dict(format=GEOMETRY_FORMAT, shape_hw=[2, 3],
        H_metric_mm_to_camera_u=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    model_binding = _write(old / "results" / "measured_geometry" / "sample.json", model)
    geometry_index = old / "results" / "measured_geometry" / "index.json"
    _write(geometry_index, dict(format=GEOMETRY_FORMAT,
        runs=dict(sample=dict(reference_model=model_binding))))
    transfer_root = old / "annotations" / "map_placement"
    transfer = dict(format=PLACEMENT_FORMAT, created_utc="2026-01-01T00:00:00Z",
        reference_run="sample", map_version=version,
        runs=[dict(run_id="sample", status="placed")])
    transfer_binding = _write(transfer_root / "retained" / "index.json", transfer)
    preparation = dict(transfer, format=painting_series_use_ui._PREPARATION)
    preparation_binding = _write(old / "annotations" / "map_adoption" / "retained" / "index.json",
        preparation)
    store = GeometryRevisionStore(project)
    revision = dict(format=GEOMETRY_REVISION_FORMAT, revision_id="original-revision",
        configuration=dict(provenance=dict(preparation=preparation_binding)))
    current_binding = _write(store.current_path, revision)
    before = {path.relative_to(old): path.read_bytes() for path in old.rglob("*") if path.is_file()}

    moved = series / "Geometry"
    old.rename(moved)
    document["paths"]["geometry_facies"] = "Geometry"
    document["storage"] = dict(path_aliases={str(old): str(moved)})
    manifest.write_text(toml.dumps(document), encoding="utf-8")
    refresh_storage_paths(series)

    assert resolve_project_package_root(old) == moved
    reopened = FluidFlowerProject.load(old)
    assert reopened.project_id == project.project_id
    assert load_series_geometry_project(series).project_id == project.project_id
    assert geometry_package_root(series) == moved
    # A retained relative legacy reference also joins the root before aliases.
    document["paths"]["geometry_facies"] = "geometry_facies"
    manifest.write_text(toml.dumps(document), encoding="utf-8")
    refresh_storage_paths(series)
    assert geometry_package_root(series) == moved
    assert GeometryRevisionStore(reopened).current() == revision
    assert read_run_geometry(geometry_index)["sample"].model == model
    selected = painting.list_versions(reopened, "sample")[0]
    assert selected["sha256"] == record["sha256"]
    assert painting.map_approval(painting.read_map_approvals(reopened, "sample"), record["sha256"]) == approval
    if kind == "manual":
        restored, saved_map = manual.load_manual_version(record)
        assert restored.run_id == session.run_id
        assert manual.manual_map_digest(restored, saved_map) == record["sha256"]
        assert painting_facies_approval.approved_reference(record, approval) == reference
        painting_facies_approval.metadata_key(record, approval)
    else:
        restored, saved_map, corrections = painting.load_version(record)
        assert painting.map_digest(restored, saved_map, corrections) == record["sha256"]
    assert read_placement(transfer_root / "retained") == dict(transfer, folder=str(moved / "annotations/map_placement/retained"))
    rows, errors = saved_transfers(transfer_root)
    assert errors == [] and rows[0]["version"] == version
    progress = painting_series_use_ui.existing_progress(moved / "annotations", store.current_path, selected)
    assert progress["errors"] == []
    assert progress["transfer"]["map_version"] == version
    assert progress["adopted"]["revision_id"] == revision["revision_id"]
    assert painted_adoption_table._checked(transfer_binding, metadata=True) == transfer
    assert json.loads(corrected_setup._small(current_binding["path"])) == revision
    assert corrected_setup._verified(current_binding)["sha256"] == current_binding["sha256"]
    assert corrected_setup._same(current_binding["path"], GeometryRevisionStore(reopened).current_path)
    assert sources._carried_data(dict(document=current_binding, pointer="/revision_id")) == revision["revision_id"]
    cache = BoundVerificationCache()
    assert json.loads(cache.read(current_binding["path"], limit=1024 * 1024)) == revision
    assert {path.relative_to(moved): path.read_bytes() for path in moved.rglob("*") if path.is_file()} == before
    # Moving changes only locations: an altered file still fails its original SHA.
    (Path(selected["folder"]) / "session.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="changed"):
        (manual.load_manual_version if kind == "manual" else painting.load_version)(record)
