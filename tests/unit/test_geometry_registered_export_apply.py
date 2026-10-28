"""Synthetic genuine Store -> registered view/export/Apply -> original Build guard.

Two tiny local runs only. The explicit fill and nominal displacement are fixture
inputs, not an answer to FF_AC's pending edge-fill or provisional decisions.
No real rig model, science execution, image registration, or NAS input is used.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest
import toml

from darsia.fluidflower import corrected_setup as export
from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import geometry_raster_sources as sources
from darsia.fluidflower import geometry_workspace_frame as frame
from tests.unit.test_geometry_imported_rasters import imported, save, snapshot


def _archive(path, values):
    np.savez(path, array=values, metadata=dict(space_dim=2, indexing="ij",
        dimensions=[.49, .9], origin=[0., .49], scalar=values.ndim == 2,
        series=False))


@pytest.fixture
def registered_setup(imported, monkeypatch):
    data = imported
    # Genuine existing NPZ readers/grid comparisons are used, without mocks.
    data.arrays["labels"] = np.array([[1, 1, 2], [1, 2, 2]], np.uint8)
    data.arrays["baseline"] = np.arange(18, dtype=np.float32).reshape(2, 3, 3) / 20.
    for name, values in data.arrays.items():
        _archive(data.original / (name + ".npz"), values)
    series = data.original.parent / "series"
    (series / "config/runs").mkdir(parents=True)
    (series / "config/merged").mkdir()
    (series / "series.toml").write_bytes(toml.dumps(dict(series_id="synthetic-two-runs",
        runs=dict(ids=["one", "two"]))).encode())
    second_photo, second_depth = data.original / "second-baseline.npz", data.original / "second-depth.npz"
    _archive(second_photo, data.arrays["baseline"] + np.float32(.05))
    _archive(second_depth, np.full((2, 3), .02, np.float64))
    config = deepcopy(data.config)
    config.update(width_m=.9, height_m=.49, frame="rig-corrected")
    rows, before = {}, {}
    for run, photo, depth in (("one", Path(config["baseline_path"]), Path(config["depth_path"])),
                              ("two", second_photo, second_depth)):
        fields = {name: {**sources.descriptor(path), "shape": shape}
                  for name, path, shape in (
                    ("shape_corrected_baseline", photo, [2, 3, 3]),
                    ("labels", Path(config["labels_path"]), [2, 3]),
                    ("facies", Path(config["facies_source_path"]), [2, 3]),
                    ("depth", depth, [2, 3]))}
        values = dict(data=dict(results=str(series / "results" / run)),
            depth=dict(depth_map=str(depth)),
            labeling=dict(labels=str(config["labels_path"]), colored_image="unmodified-original.png"),
            facies=dict(path=str(config["facies_source_path"]), props=config["facies_properties_source_path"]),
            corrections=dict(curvature=dict(crop=dict(top_left=[3, 8], width=.9))),
            calibration=dict(marker="unchanged"))
        raw = ("# exact original per-run configuration\r\n" + toml.dumps(values).replace("\n", "\r\n")).encode()
        for kind in ("runs", "merged"):
            path = series / "config" / kind / (run + ".toml")
            path.write_bytes(raw)
            before[(kind, run)] = raw
        rows[run] = dict(fields=fields,
            configuration=sources.descriptor(series / "config/merged" / (run + ".toml")),
            configured_depth_map=sources.descriptor(depth),
            facies_properties=sources.descriptor(config["facies_properties_source_path"]))
    correction = data.original / "bound-original-correction.json"
    correction.write_bytes(b'{"synthetic_original_correction_identity":true}\n')
    config["frame_binding"] = dict(format=frame.FRAME_BINDING_FORMAT,
        source_reference_run="original-source", display_reference_run="one",
        reference_shape_corrections=[sources.descriptor(correction)], runs=rows)
    config = sources.bind_imported_rasters(config, sources={
        key: sources.descriptor(config[sources.PATH_KEYS[key]]) for key in sources.KEYS},
        diagnosis=config["imported_rasters"]["diagnosis"])
    config["provenance"] = dict(adopted_content_sha256=frame.adopted_content_digest(config))
    data.config = config
    parent = save(data)
    draft = sources.prepare_canonical_edit(data.project, actor="synthetic owner",
        reason="explicit two-run integer translation fixture")
    authority = data.original / "explicit-synthetic-authority.json"
    authority.write_bytes(b'{"synthetic_only":true,"fill_label":1}\n')
    run_rows, expected = [], {}
    for run, column in (("one", 1), ("two", -1)):
        labels, metrics = sources.integer_label_translation(data.arrays["labels"],
            row_shift=0, column_shift=column, fill_label=1)
        target = data.original / (run + "-translated.npz")
        np.savez(target, array=labels, metadata=geometry._darsia_metadata(Path(config["labels_path"])))
        measured = data.original / (run + "-measurement.json")
        measured.write_bytes(json.dumps(dict(synthetic_only=True, column_shift=column)).encode())
        record = dict(parent_revision_id=parent["revision_id"],
            parent_labels=sources.descriptor(config["labels_path"]),
            labels=sources.descriptor(target), authority=sources.descriptor(authority),
            measurement=sources.descriptor(measured), source_image=rows[run]["fields"]["shape_corrected_baseline"].copy(),
            status="registered", row_shift=0, column_shift=column, fill_label=1,
            candidate=sources.registration_pixel_shift(dx_mm=column*.1645,dy_mm=0.,pixel_size_mm=.1645), metrics=metrics)
        run_rows.append(dict(run_id=run,geometry_choice="run_override",enabled=True,
            acquisition_state="unknown",labels_path=str(target),
            source_image_path=record["source_image"]["path"],registration=record))
        # Source descriptors intentionally carry no display-only shape metadata.
        record["source_image"] = sources.descriptor(record["source_image"]["path"])
        expected[run] = labels
    draft["run_geometry"]["runs"] = run_rows
    revision = save(data, draft)
    monkeypatch.setattr(export,"native_series_link",lambda _project: dict(series_root=str(series)))
    return SimpleNamespace(data=data,project=data.project,series=series,parent=parent,
        revision=revision,config=revision["configuration"],before=before,expected=expected,rows=rows)


def _export(data, run="one"):
    return export.export_corrected_setup(data.project,reference=SimpleNamespace(run_id=run),actor="synthetic owner")


def _mask(labels):
    boundary=geometry._layer_boundary_mask(labels,set(int(v) for v in np.unique(labels)))
    result=np.zeros((*labels.shape,4),np.uint8)
    result[boundary]=(255,255,255,235)
    return result


def test_two_run_store_export_apply_and_original_build_dispatch(registered_setup, monkeypatch):
    data=registered_setup
    originals=snapshot(data.data.original)
    reference=_export(data)
    reference_bytes=(reference.destination/"export.json").read_bytes()
    child=_export(data,"two")
    assert child.destination==reference.destination/"runs/two"
    assert (reference.destination/"export.json").read_bytes()==reference_bytes
    assert reference.revision_id==child.revision_id==data.revision["revision_id"]
    assert reference.revision_id!=data.parent["revision_id"]
    for run, output in (("one",reference),("two",child)):
        resolved=sources.registered_run_sources(data.project.root,data.revision,run)
        assert output.receipt["binding"]["registered_geometry"]==resolved
        for key in ("labels","facies"):
            assert output.receipt["binding"]["sources"][key]==resolved["files"][key]
            assert Path(output.receipt["files"][export.NAMES[key]]["path"]).read_bytes()==Path(resolved["files"][key]["path"]).read_bytes()
        labels=geometry._load_array(output.destination/"native_labels.npz")
        assert labels.dtype==np.int32 and np.array_equal(labels,data.expected[run])
        assert data.data.arrays["labels"].dtype==np.uint8
        assert output.receipt["binding"]["sources"]["depth"]==export._identity(data.rows[run]["fields"]["depth"])
        assert output.receipt["binding"]["configured_depth_source"]==data.rows[run]["configured_depth_map"]
        assert (output.destination/"facies_properties.csv").read_bytes()==Path(data.data.config["facies_properties_source_path"]).read_bytes()
    assert not np.array_equal(data.expected["one"],data.expected["two"])
    export.apply_corrected_selected_run(data.project,reference)
    assert (data.series/"config/merged/two.toml").read_bytes()==data.before[("merged","two")]
    export.apply_corrected_selected_run(data.project,child)
    from darsia.fluidflower import setup_rig as dispatch
    calls=[]
    monkeypatch.setattr(dispatch,"_original_setup_rig",lambda *args:calls.append(args))
    for run,output in (("one",reference),("two",child)):
        for kind in ("runs","merged"):
            path=data.series/"config"/kind/(run+".toml")
            assert path.read_bytes()==export.corrected_config_bytes(data.before[(kind,run)],output.receipt["binding"]["replacements"])
            parsed=tomllib.loads(path.read_bytes().decode())
            assert parsed["depth"]["depth_map"]==data.rows[run]["configured_depth_map"]["path"]
            assert parsed["corrections"]==tomllib.loads(data.before[(kind,run)].decode())["corrections"]
        path=data.series/"config/merged"/(run+".toml")
        assert export.require_corrected_export_for_build(path)==output.receipt
        dispatch.setup_rig(dispatch.Rig,path,False)
        assert calls[-1][1] is path and calls[-1][2] is False
        assert calls[-1][0].setup is dispatch.Rig.setup
        assert (output.destination/"apply-complete.json").is_file()
    assert len(calls)==2 and snapshot(data.data.original)==originals
    current=geometry.GeometryRevisionStore(data.project).current_path
    assert current.stat().st_size<frame.METADATA_LIMIT
    assert (Path(data.revision["artifacts"]["facies_path"]).parent/"revision.json").stat().st_size<frame.METADATA_LIMIT


def test_registered_apply_reuses_completed_bytes_without_new_claim(registered_setup):
    data=registered_setup;result=_export(data)
    first=export.apply_corrected_selected_run(data.project,result)
    before=snapshot(data.project.root)
    assert export.apply_corrected_selected_run(data.project,result)==first
    assert _export(data).reused
    assert snapshot(data.project.root)==before


def test_registered_boundary_view_uses_each_saved_run_not_parent(registered_setup):
    data=registered_setup;before=snapshot(data.project.root)
    for run in ("one","two"):
        assert np.array_equal(frame.saved_boundaries(data.config,run,(2,3),project=data.project),_mask(data.expected[run]))
    assert not np.array_equal(_mask(data.expected["one"]),_mask(data.expected["two"]))
    assert np.array_equal(frame.saved_boundaries(data.parent["configuration"],"one",(2,3)),_mask(data.data.arrays["labels"]))
    assert snapshot(data.project.root)==before


def test_actual_layer_view_route_passes_genuine_project_and_saved_config(registered_setup):
    from PySide6.QtWidgets import QApplication
    from darsia.fluidflower.series_geometry import SeriesRunInputs
    from darsia.fluidflower.series_layer_view import LayerPainterView
    app=QApplication.instance() or QApplication([])
    data=registered_setup;calls=[]
    state=SimpleNamespace(project=data.project,_workspace_frame_config=data.config,
        session=SimpleNamespace(image_size=None,strokes=[],polygons=[],image_path=""),
        canvas=SimpleNamespace(clear_photo=lambda:calls.append("clear")),
        propose_status=SimpleNamespace(setText=lambda text:calls.append(text)),
        _update_enhanced=lambda:calls.append("enhanced"),
        _show_reference=lambda **kwargs:calls.append(kwargs))
    run="two";reference=SeriesRunInputs(run,data.series/"config/merged/two.toml",
        Path(data.rows[run]["fields"]["shape_corrected_baseline"]["path"]))
    LayerPainterView._load_reference_photo(state,reference)
    assert app is not None and state.image.image_size==(3,2)
    assert np.array_equal(state._saved_corrected_boundaries,_mask(data.expected[run]))
    assert calls==["enhanced",dict(reset_view=True)]
    assert state.session.image_path==str(reference.baseline_image)


@pytest.mark.parametrize("mutation",["missing_project","unsaved_config"])
def test_registered_view_requires_actual_saved_project(registered_setup,mutation):
    data=registered_setup;config=deepcopy(data.config);project=data.project
    if mutation=="missing_project":project=None
    else:config["width_m"]+=.01
    before=snapshot(data.project.root)
    with pytest.raises(ValueError,match="actual project|genuine saved"):
        frame.saved_boundaries(config,"one",(2,3),project=project)
    assert snapshot(data.project.root)==before


def test_selected_source_validation_never_reads_other_run_photo(registered_setup):
    data=registered_setup
    path=Path(data.rows["two"]["fields"]["shape_corrected_baseline"]["path"])
    path.write_bytes(path.read_bytes()+b"changed unrelated run")
    resolved=sources.registered_run_sources(data.project.root,data.revision,"one")
    assert resolved["run_id"]=="one"
    with pytest.raises(ValueError,match="source changed"):
        sources.canonical_parent_payload(data.config)


@pytest.mark.parametrize("mutation",["stored_labels","stored_grid","full_csv","source_image","parent_record","current_record"])
def test_registered_export_rejects_changed_store_or_authority_without_output(registered_setup,mutation):
    data=registered_setup;run="one"
    artifact=geometry.resolve_run_geometry_artifacts(data.revision,run)
    if mutation=="stored_labels":
        path=Path(artifact["labels_path"]);values=geometry._load_array(path).copy();values[0,0]=2
        np.savez(path,array=values,metadata=geometry._darsia_metadata(path))
    elif mutation=="stored_grid":
        path=Path(artifact["labels_path"]);values=geometry._load_array(path);meta=geometry._darsia_metadata(path);meta["origin"]=[1.,.49]
        np.savez(path,array=values,metadata=meta)
    elif mutation=="full_csv":
        path=Path(artifact["facies_properties_path"]);path.write_bytes(path.read_bytes()+b"# lost-full-source-identity\n")
    elif mutation=="source_image":
        path=Path(data.rows[run]["fields"]["shape_corrected_baseline"]["path"]);path.write_bytes(path.read_bytes()+b"external")
    elif mutation=="parent_record":
        path=Path(data.config["canonical_import_parent"]["revision"]["path"]);path.write_bytes(path.read_bytes()+b"\n")
    else:
        path=geometry.GeometryRevisionStore(data.project).current_path
        value=json.loads(path.read_bytes());value["reason"]="external edit not in immutable revision";path.write_bytes(json.dumps(value).encode())
    before=snapshot(data.project.root)
    with pytest.raises(ValueError):_export(data)
    assert not (data.project.root/"exports").exists()
    assert snapshot(data.project.root)==before


def test_build_requires_real_apply_and_foreign_run_cannot_borrow_reference(registered_setup):
    data=registered_setup;result=_export(data)
    partial=data.series/"config/runs/one.toml"
    partial.write_bytes(export.corrected_config_bytes(partial.read_bytes(),result.receipt["binding"]["replacements"]))
    with pytest.raises(ValueError,match="Apply this run"):
        export.require_corrected_export_for_build(partial)
    foreign=data.series/"config/runs/two.toml"
    foreign.write_bytes(export.corrected_config_bytes(foreign.read_bytes(),result.receipt["binding"]["replacements"]))
    with pytest.raises(ValueError,match="different or ambiguous selected-run"):
        export.require_corrected_export_for_build(foreign)
    assert not (result.destination/"apply-claim.json").exists()


@pytest.mark.parametrize("stage",["before_apply","after_apply"])
def test_changed_registered_source_stops_apply_or_build_without_new_writes(registered_setup,stage):
    data=registered_setup;result=_export(data)
    if stage=="after_apply":export.apply_corrected_selected_run(data.project,result)
    authority=Path(data.config["run_geometry"]["runs"][0]["registration"]["authority"]["path"])
    authority.write_bytes(authority.read_bytes()+b"changed authority")
    before=snapshot(data.project.root)
    with pytest.raises(ValueError,match="source changed"):
        if stage=="before_apply":export.apply_corrected_selected_run(data.project,result)
        else:export.require_corrected_export_for_build(data.series/"config/merged/one.toml")
    assert snapshot(data.project.root)==before


def test_changed_apply_target_is_not_accepted_by_original_build_guard(registered_setup):
    data=registered_setup;result=_export(data)
    export.apply_corrected_selected_run(data.project,result)
    path=data.series/"config/merged/one.toml";path.write_bytes(path.read_bytes()+b"# later user edit\n")
    before=snapshot(data.project.root)
    with pytest.raises(ValueError,match="identity mismatch"):
        export.require_corrected_export_for_build(path)
    assert snapshot(data.project.root)==before


def test_child_rechecks_reference_registered_evidence_without_new_writes(registered_setup):
    data=registered_setup;reference=_export(data)
    path=Path(data.config["run_geometry"]["runs"][0]["registration"]["measurement"]["path"])
    path.write_bytes(path.read_bytes()+b"changed reference measurement")
    before=snapshot(data.project.root)
    with pytest.raises(ValueError,match="source changed"):
        _export(data,"two")
    assert snapshot(data.project.root)==before
    assert not (reference.destination/"runs/two").exists()


@pytest.fixture
def private_manual_reference(registered_setup):
    """Genuine tiny editor/session/materialization and new Store, no GUI or NAS."""
    import shutil
    from tests.unit.test_geometry_imported_rasters import Project
    data=registered_setup
    parent=data.revision
    root=data.project.root.parent/"private-series"/"geometry_facies"
    project=Project(root)
    parent_file=Path(parent["artifacts"]["labels_path"]).parent/"revision.json"
    context=dict(format=sources.MANUAL_LABEL_EDIT,authority_id="Q121",selected_run="one",
        parent_revision=sources.descriptor(parent_file),private_series_root=str(root.parent))
    project.manifest["tools"]={"native_darsia":{"manual_label_edit_validation":context},
        "native_series":{"series_root":str(root.parent)}}
    root.mkdir(parents=True)
    (root/"series.json").write_text(json.dumps(project.manifest))
    current=root/"overrides/geometry_facies/current.json";current.parent.mkdir(parents=True)
    current.write_text(json.dumps(parent))
    config=deepcopy(parent["configuration"])
    base=root/"annotations/source_labels.npz";base.parent.mkdir()
    shutil.copyfile(config["labels_path"],base);config["labels_path"]=str(base)
    session=geometry.GeometryLabelEditSession.create(config,root/"overrides/geometry_facies/edit_sessions")
    session.apply(kind="brush",points_m=[[.1,.1]],target_label=2,actor="TESTONLY manual owner",brush_radius_m=.01)
    session.undo(actor="TESTONLY");session.redo(actor="TESTONLY")
    materialization=session.materialize()
    config["labels_path"]=materialization["labels_path"];config["facies_source_path"]=""
    config["label_edit_session"]={**session.config_reference(),
        "materialization_id":materialization["materialization_id"],
        "materialization_manifest_path":materialization["manifest_path"],
        "materialized_active_operation_count":session.active_operation_count()}
    config=sources.bind_private_manual_label_edit(project,config,session,materialization)
    return SimpleNamespace(project=project,parent=parent,config=config,session=session,materialization=materialization)


def test_Q121_private_manual_edit_has_real_parent_event_replay_and_new_Store_UUID(private_manual_reference):
    data=private_manual_reference
    payload=sources.manual_label_edit_payload(data.config)
    assert payload["parent"]["revision_id"]==data.parent["revision_id"]
    saved=geometry.GeometryRevisionStore(data.project).save_revision(data.config,
        actor="TESTONLY",reason="Actual tiny event replay")
    assert saved["revision_id"]!=data.parent["revision_id"]
    actual=sources.private_manual_label_edit_sources(data.project.root,saved,"one")
    assert actual["format"]==sources.MANUAL_LABEL_EDIT
    assert actual["registration"]["not_a_registration_or_H"] is True
    assert np.array_equal(geometry._load_array(Path(actual["files"]["labels"]["path"])),data.session.labels)
    assert saved["configuration"]["run_geometry"]==data.config["run_geometry"]


@pytest.mark.parametrize("field",["parent_revision","selected_run","private_series_root"])
def test_Q121_manual_context_mismatch_refuses_without_old_adoption_fallback(private_manual_reference,field):
    data=private_manual_reference;config=deepcopy(data.config)
    if field=="parent_revision":config["manual_label_edit"]["context"][field]["sha256"]="0"*64
    elif field=="selected_run":config["manual_label_edit"]["context"][field]="two"
    else:config["manual_label_edit"]["context"][field]="//TESTONLY/share/not-private"
    with pytest.raises((ValueError,OSError)):sources.manual_label_edit_payload(config)


@pytest.mark.parametrize("role",["source_labels","session","events","materialization","materialized_labels"])
def test_Q121_every_actual_manual_provenance_role_is_SHA_bound(private_manual_reference,role):
    data=private_manual_reference;config=deepcopy(data.config)
    config["manual_label_edit"][role]["sha256"]="0"*64
    with pytest.raises(ValueError):sources.manual_label_edit_payload(config)


def test_Q121_altered_materialized_labels_cannot_be_accepted_by_repinning_only(private_manual_reference):
    data=private_manual_reference;config=deepcopy(data.config)
    path=Path(config["labels_path"])
    changed=data.session.labels.copy();changed[0,0]=99
    np.savez_compressed(path,array=changed,metadata=geometry._darsia_metadata(data.session.base_labels_path))
    config["manual_label_edit"]["materialized_labels"]=sources.descriptor(path)
    with pytest.raises(ValueError,match="exact original active-event replay"):
        sources.manual_label_edit_payload(config)


def test_Q121_manual_record_cannot_export_another_run_or_reuse_parent_Store_UUID(private_manual_reference):
    data=private_manual_reference
    saved=geometry.GeometryRevisionStore(data.project).save_revision(data.config,actor="TESTONLY",reason="tiny event")
    with pytest.raises(ValueError,match="new genuine private Store UUID/reference run"):
        sources.private_manual_label_edit_sources(data.project.root,saved,"two")
    old=deepcopy(saved);old["revision_id"]=data.parent["revision_id"]
    with pytest.raises(ValueError,match="new genuine private Store UUID/reference run"):
        sources.private_manual_label_edit_sources(data.project.root,old,"one")


def test_Q121_absent_private_context_leaves_original_materialization_config_unchanged():
    config={"TESTONLY":"ordinary raw/painted config"}
    project=SimpleNamespace(manifest={})
    assert sources.bind_private_manual_label_edit(project,config,object(),{}) is config


def test_Q121_private_project_cannot_keep_a_NAS_series_link(private_manual_reference):
    data=private_manual_reference
    manifest=json.loads((data.project.root/"series.json").read_bytes())
    manifest["tools"]["native_series"]["series_root"]="//TESTONLY/NAS/original-series"
    (data.project.root/"series.json").write_text(json.dumps(manifest))
    saved={"configuration":data.config}
    with pytest.raises(ValueError,match="exact local private series/project"):
        sources.private_manual_label_edit_sources(data.project.root,saved,"one")


def test_Q121_manual_facies_cannot_fall_back_to_original_unedited_raster(private_manual_reference):
    data=private_manual_reference;config=deepcopy(data.config)
    config["facies_source_path"]=data.parent["configuration"]["facies_source_path"]
    with pytest.raises(ValueError,match="actual parent display/reference grid"):
        sources.manual_label_edit_payload(config)
