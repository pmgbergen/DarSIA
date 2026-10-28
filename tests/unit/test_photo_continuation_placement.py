"""Actual approved tiny maps -> V2 producer -> Store/export, private files only.

Positive native tests mock ONLY the deliberately unadmitted renderer-to-saved
corrected-NPZ certificate seam. All source, approval, model, replay and Store
guards are real. A separate negative test exercises the actual missing seam.
These synthetic tests qualify no real RAW renderer or real-data adoption.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_manual as manual
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower import map_placement as placement
from darsia.fluidflower import photo_edge_continuation as policy
from darsia.fluidflower import painted_adoption_table as approved
from darsia.fluidflower import painted_adoption_geometry as geometry_proof
from darsia.fluidflower import round5_geometry
from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import geometry_raster_sources as sources
from darsia.fluidflower import corrected_setup as corrected
from darsia.fluidflower.published_dataset import simulator_geometry
from darsia.fluidflower.series_geometry import RigFrame

from .test_geometry_imported_rasters import imported  # noqa: F401
from .test_geometry_registered_export_apply import registered_setup  # noqa: F401
from .test_map_adoption_store import MAP, IDENTITY, SHIFTED


_REAL_PAYLOAD_LINK = geometry_proof._require_corrected_payload_link


@pytest.fixture(autouse=True)
def synthetic_payload_link_only(monkeypatch):
    """Do not claim that a tiny old NPZ has an admitted renderer certificate."""
    def unit_link(config, run, model, renderer):
        from darsia.fluidflower.geometry_workspace_frame import selected_fields
        fields = selected_fields(config, run)
        return [sources._verify(fields[name]) for name in
                ("shape_corrected_baseline", "labels", "facies", "depth")]
    monkeypatch.setattr(geometry_proof, "_require_corrected_payload_link", unit_link)


def _approved_case(data, tmp_path):
    project = data.project
    # The older Store-only fixture never loads the series manifest. This
    # consumer exercises the actual native series loader, with its real schema.
    manifest_path = data.series / "series.toml"
    manifest_path.write_bytes(b"schema_version = 1\n" + manifest_path.read_bytes())
    photos, frames, entries = {}, {}, {}
    model_folder = tmp_path / "policy-models"
    model_folder.mkdir()
    edges = round5_geometry.write_new_json(model_folder / "edges.json", {"synthetic": True})
    for run, matrix in (("one", IDENTITY), ("two", SHIFTED)):
        photo = tmp_path / (run + ".png")
        Image.new("RGB", (3, 2), (70, 100, 130)).save(photo)
        photos[run] = photo
        frame = RigFrame(corners=dict(top_left=(0., 0.), top_right=(0., 2.),
            bottom_right=(1., 2.), bottom_left=(1., 0.)), width_m=.9, height_m=.49,
            run_id=run, source="private saved frame", image=str(photo), image_size=(3, 2))
        frames[run] = frame.to_dict()
        photograph = sources.descriptor(photo)
        measured_frame = dict(H_camera_to_mm=np.linalg.inv(matrix).tolist(),
            sand_area_factor14=1., explicit_certificate={"synthetic": True})
        saved = round5_geometry.write_new_json(model_folder / (run + "-source.json"),
            dict(run=run, conditional_frame=measured_frame))
        renderer = round5_geometry.write_new_json(model_folder / (run + "-renderer.json"),
            dict(shape_hw=[2, 3], H_rig_corrected_native_to_baseline_camera_u=IDENTITY,
                renderer=dict(raw_acquisition=photograph, camera_u_wh=[3, 2], native_shape_hw=[2, 3],
                              native_to_camera_u_h=IDENTITY)))
        value = round5_geometry.reference_model(run, measured_frame,
            reference_image=photograph, camera_profile={"camera_wh": [3, 2]},
            source_model=saved, edge_review=edges,
            baseline_configuration=data.config["frame_binding"]["runs"][run]["configuration"],
            baseline_native_renderer=renderer)
        bound = round5_geometry.write_new_json(model_folder / (run + ".json"), value)
        entries[run] = dict(reference_model=bound)
    index = model_folder / "index.json"
    index.write_text(json.dumps(dict(format=placement.GEOMETRY_FORMAT, runs=entries)), encoding="utf-8")
    project.manifest.setdefault("tools", {})["native_series"] = dict(
        series_root=str(data.series), series_id="private-series", rig_frames=frames)
    (project.root / "series.json").write_text(json.dumps(project.manifest), encoding="utf-8")
    session = lp.PaintingSession("one", image_path=str(photos["one"]), image_size=(3, 2),
        image_stamp=lp.image_stamp(photos["one"]), permeability_unit="mD",
        layers=[lp.PaintedLayer(5, "Upper", "#336699", "Sand1"), lp.PaintedLayer(6, "Lower", "#663399", "Sand2")],
        facies={"Sand1": lp.FaciesEntry("porous_medium", .4, 10.), "Sand2": lp.FaciesEntry("porous_medium", .3, 20.)})
    frame = RigFrame.from_dict(frames["one"])
    saved = manual.ManualMap(MAP.copy(), np.ones(MAP.shape, bool), np.zeros(MAP.shape, bool), (3, 2))
    version = manual.save_manual_version(project, session, saved, name="Private approved map")
    approval = manual.approve_manual_version(version, reviewer="private")
    binding = approved.capture_version(version, approval, frame, MAP, project=project)
    grid_path = data.config["canonical_import_parent"]["imported_configuration"]["labels_path"]
    grid = dict(metadata=geometry._darsia_metadata(Path(grid_path)), shape_hw=[2, 3], source=sources.descriptor(grid_path))
    return SimpleNamespace(data=data, photos=photos, frame=frame, version=version, approval=approval,
        binding=binding, index=index, grid=grid, session=session)


def _prepare(case, tmp_path):
    contracts = policy.build_contracts(case.data.project, case.index, "one", ["one", "two"], [5, 6], target_grid="native")
    return adoption.prepare_adoption(tmp_path / "native-preparation", MAP, reference_run="one",
        geometry_index=case.index, runs=["one", "two"], roles={5: "porous_medium", 6: "porous_medium"},
        names={5: "Upper", 6: "Lower"}, facies={5: "Sand1", 6: "Sand2"},
        version={"name": case.version["name"], "sha256": case.version["sha256"], "approval": case.approval},
        current_labels=lambda _run: case.grid, continuation=contracts, approved_source=case.binding)


def test_new_camera_placement_preserves_original_map_and_counts_continuation_separately(registered_setup, tmp_path, monkeypatch):
    case = _approved_case(registered_setup, tmp_path)
    original_files = {p: p.read_bytes() for p in Path(case.version["folder"]).rglob("*") if p.is_file()}
    progress = []
    contracts = policy.build_contracts(case.data.project, case.index, "one", ["one", "two"], [5, 6],
                                       run_progress=progress.append)
    result = placement.place_map_in_runs(tmp_path / "camera-review", MAP, reference_run="one",
        geometry_index=case.index, runs=["one", "two"], photo_loader=lambda run, _model: np.asarray(Image.open(case.photos[run])),
        roles={5: "porous_medium", 6: "porous_medium"}, version={"name": case.version["name"],
        "sha256": case.version["sha256"], "approval": case.approval}, measure_offsets=False,
        continuation=contracts, approved_source=case.binding, run_progress=progress.append)
    assert result["format"] == placement.PLACEMENT_FORMAT_V2
    assert result["runs"][1]["continuation"]["continued_pixels"] == 2
    assert result["runs"][1]["continuation"]["inside_gap_pixels"] == 0
    folder = Path(result["folder"])
    with np.load(folder / "two/layers.npz") as file:
        assert np.array_equal(file["layers"], np.asarray([[5, 5, 5], [5, 5, 6]], np.uint8))
        assert file["uncovered"][:, 0].all() and file["continued"][:, 0].all()
        assert not file["support"][:, 0].any() and file["effective_coverage"][:, 0].all()
        assert not file["gaps"].any()
    checked = placement.read_placement(folder)
    assert checked["format"] == result["format"]
    assert adoption.placement_provenance(folder)["format"] == adoption.PROVENANCE_FORMAT_V2
    from darsia.fluidflower import painted_analysis_exclusions as exclusions
    native_contracts = policy.build_contracts(case.data.project, case.index, "one", ["one", "two"], [5, 6],
                                              target_grid="native", run_progress=progress.append)
    # This synthetic renderer covers native products; it has no corrected-photo
    # asset for the final presentation writer, which is outside run progress.
    monkeypatch.setattr(exclusions, "build_gap_review", lambda snapshot, folder: snapshot)
    native_review = exclusions.build_transfer_gap_review(folder, MAP, painting_binding=case.binding, geometry_index=case.index,
        map_version=dict(case.version, approval=case.approval), contracts=native_contracts,
        run_progress=progress.append)
    assert [row["run_id"] for row in native_review["runs"]] == ["one", "two"]
    assert progress == [dict(phase=phase, completed=completed, total=2, run_id=run)
        for phase in ("frame_models_camera", "transfer", "frame_models_native", "native_review")
        for n, run in enumerate(("one", "two")) for completed in (n, n + 1)]
    assert all(path.read_bytes() == raw for path, raw in original_files.items())


def test_v2_masks_survive_actual_store_export_apply_and_simulator_read(registered_setup, tmp_path):
    data = registered_setup
    case = _approved_case(data, tmp_path)
    prepared = _prepare(case, tmp_path)
    assert prepared["format"] == adoption.PREPARATION_FORMAT_V2 and prepared["adopted"] is False
    before_csv = Path(data.config["canonical_import_parent"]["imported_configuration"]["facies_properties_source_path"]).read_bytes()
    candidate, summary = adoption.adoption_candidate(data.revision, prepared["folder"], {5: 1, 6: 2})
    assert all(row["registration"]["kind"] == sources.PAINTED_V2 for row in candidate["run_geometry"]["runs"])
    revision = geometry.GeometryRevisionStore(data.project).save_revision(candidate,
        actor="private synthetic test", reason="explicit synthetic V2 adoption", preserve_failed_artifacts=True)
    for run, expected_continued in (("one", 0), ("two", 2)):
        resolved = sources.registered_run_sources(data.project.root, revision, run)
        product = Path(resolved["files"]["labels"]["path"])
        with np.load(product, allow_pickle=True) as file:
            assert file["continued"].dtype == np.dtype(bool) and int(file["continued"].sum()) == expected_continued
            assert np.array_equal(file["effective_coverage"], file["inside"])
            assert not file["gaps"].any()
        assert Path(resolved["files"]["facies_properties"]["path"]).read_bytes() == before_csv
        exported = corrected.export_corrected_setup(data.project, reference=SimpleNamespace(run_id=run), actor="private synthetic test")
        corrected.apply_corrected_selected_run(data.project, exported)
        corrected.require_corrected_export_for_build(data.series / "config/merged" / (run + ".toml"))
        with np.load(exported.destination / "native_labels.npz", allow_pickle=True) as file:
            assert int(file["continued"].sum()) == expected_continued
        facies, _roles, _table, source = simulator_geometry.run_geometry(revision, run)
        assert facies.shape == (2, 3)
        assert source["photo_edge_continuation"]["continued_pixels"] == expected_continued
        assert source["photo_edge_continuation"]["adopted_by_this_export"] is False
    assert summary["runs"] == ["one", "two"]


def test_real_missing_renderer_payload_link_refuses_before_any_native_output(registered_setup, tmp_path, monkeypatch):
    case = _approved_case(registered_setup, tmp_path)
    monkeypatch.setattr(geometry_proof, "_require_corrected_payload_link", _REAL_PAYLOAD_LINK)
    with pytest.raises(ValueError, match="no admitted renderer-to-saved corrected-baseline payload link"):
        _prepare(case, tmp_path)
    assert not (tmp_path / "native-preparation").exists()


def test_native_reuse_rechecks_the_real_renderer_payload_gate(registered_setup, tmp_path, monkeypatch):
    case = _approved_case(registered_setup, tmp_path)
    result = _prepare(case, tmp_path)
    monkeypatch.setattr(geometry_proof, "_require_corrected_payload_link", _REAL_PAYLOAD_LINK)
    with pytest.raises(ValueError, match="no admitted renderer-to-saved corrected-baseline payload link"):
        adoption.read_preparation(result["folder"])


def test_positive_native_producer_refuses_guessed_grid_metadata(registered_setup, tmp_path):
    case = _approved_case(registered_setup, tmp_path)
    case.grid = dict(case.grid, metadata={"guessed": True})
    with pytest.raises(ValueError, match="exact imported parent's grid metadata and source"):
        _prepare(case, tmp_path)


def test_native_proof_cannot_be_supplied_from_legacy_v1_or_changed_configuration_copy(registered_setup, tmp_path):
    case = _approved_case(registered_setup, tmp_path)
    result = _prepare(case, tmp_path)
    folder = Path(result["folder"])
    path = folder / "index.json"
    value = json.loads(path.read_bytes())
    proof = value.pop("painting_geometry")
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="exact source/grid/model proof"):
        adoption.read_preparation(folder)
    value["painting_geometry"] = proof
    path.write_text(json.dumps(value), encoding="utf-8")
    copied = folder / proof["baseline_configurations"]["two"]["copy"]["path"]
    copied.write_bytes(copied.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        adoption.read_preparation(folder)


@pytest.mark.parametrize("field", ["continued", "gaps", "effective_coverage", "support", "filled"])
def test_v2_replay_refuses_tampered_masks_even_when_product_descriptor_is_updated(registered_setup, tmp_path, field):
    case = _approved_case(registered_setup, tmp_path)
    result = _prepare(case, tmp_path)
    folder = Path(result["folder"])
    path = folder / "two/labels.npz"
    with np.load(path, allow_pickle=True) as original:
        values = {key: original[key].copy() for key in original.files}
    values[field][0, 0] = not values[field][0, 0]
    np.savez_compressed(path, **values)
    index_path = folder / "index.json"
    index = json.loads(index_path.read_bytes())
    row = next(row for row in index["runs"] if row["run_id"] == "two")
    row["labels"].update({key: value for key, value in sources.descriptor(path).items() if key != "path"})
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="mask differs"):
        adoption.read_preparation(folder)


def test_source_version_file_or_actual_approval_history_drift_blocks_v2(registered_setup, tmp_path):
    case = _approved_case(registered_setup, tmp_path)
    result = _prepare(case, tmp_path)
    history = Path(case.version["folder"]).parent / "approvals.jsonl"
    original = history.read_bytes()
    history.write_bytes(b"")
    with pytest.raises(ValueError, match="absent from its actual history"):
        adoption.read_preparation(result["folder"])
    history.write_bytes(original)
    session = Path(case.version["folder"]) / "session.json"
    session.write_bytes(session.read_bytes() + b" ")
    with pytest.raises(ValueError):
        adoption.read_preparation(result["folder"])


def test_native_typed_metadata_drift_is_refused_even_with_updated_product_descriptor(registered_setup, tmp_path):
    case = _approved_case(registered_setup, tmp_path)
    result = _prepare(case, tmp_path)
    folder = Path(result["folder"])
    path = folder / "two/labels.npz"
    with np.load(path, allow_pickle=True) as original:
        values = {key: original[key].copy() for key in original.files}
    values["metadata"] = {"guessed": True}
    np.savez_compressed(path, **values)
    index_path = folder / "index.json"
    index = json.loads(index_path.read_bytes())
    row = next(row for row in index["runs"] if row["run_id"] == "two")
    row["labels"].update({key: value for key, value in sources.descriptor(path).items() if key != "path"})
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="typed grid metadata"):
        adoption.read_preparation(folder)


def test_missing_geometry_is_retained_as_diagnostic_without_an_invented_contract(registered_setup, tmp_path):
    case = _approved_case(registered_setup, tmp_path)
    contracts = policy.build_contracts(case.data.project, case.index, "one", ["one", "two"], [5, 6])
    result = placement.place_map_in_runs(tmp_path / "partial-camera-review", MAP, reference_run="one",
        geometry_index=case.index, runs=["one", "two", "not_analyzed"],
        photo_loader=lambda run, _model: np.asarray(Image.open(case.photos[run])),
        roles={5: "porous_medium", 6: "porous_medium"},
        version={"name": case.version["name"], "sha256": case.version["sha256"], "approval": case.approval},
        measure_offsets=False, continuation=contracts, approved_source=case.binding)
    checked = placement.read_placement(result["folder"])
    assert checked["runs"][-1] == {"run_id": "not_analyzed", "status": "no geometry"}
    assert set(policy.read_contracts(Path(result["folder"]) / "continuation.json")["runs"]) == {"one", "two"}
