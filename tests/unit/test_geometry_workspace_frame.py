"""Synthetic saved-frame display boundaries; no rig, registration or NAS job."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from darsia.fluidflower import geometry_workspace_frame as frame
from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import layer_painting as painting
from darsia.fluidflower.series_geometry import SeriesRunInputs


def asset(path, shape):
    path.write_bytes(path.name.encode())
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                size_bytes=len(raw), mtime_ns=path.stat().st_mtime_ns, shape=shape)


@pytest.fixture
def corrected(tmp_path):
    rows = {}
    for run in ["first", "second"]:
        folder = tmp_path/run
        folder.mkdir()
        fields = {name: asset(folder/(name+".npz"), [2, 3, 3] if name=="shape_corrected_baseline" else [2, 3])
                  for name in frame.FIELDS}
        rows[run] = dict(fields=fields, configured_depth_map=dict(path=str(folder/"original-depth.npz")))
    correction = asset(tmp_path/"shape_correction.npz", [2, 3])
    config = dict(frame="rig-corrected", width_m=.9, height_m=.49,
        frame_binding=dict(format=frame.FRAME_BINDING_FORMAT,source_reference_run="historical",
            display_reference_run="first", reference_shape_corrections=[correction],runs=rows))
    return config


def test_raw_default_uses_original_loader_without_saved_map_access(tmp_path, monkeypatch):
    original = SeriesRunInputs("one", tmp_path/"config.toml", tmp_path/"raw.JPG")
    expected = object()
    monkeypatch.setattr(painting,"load_painting_image",lambda path: expected if path==original.baseline_image else None)
    monkeypatch.setattr(frame,"selected_fields",lambda *a,**k: pytest.fail("raw opened corrected fields"))
    assert frame.frame_kind({})=="raw"
    assert frame.workspace_reference({},original) is original
    assert frame.load_painting_image({},original) is expected


def test_selected_run_preserves_its_own_baseline_and_distinct_depth(corrected):
    first = frame.selected_fields(corrected,"first")
    second = frame.selected_fields(corrected,"second")
    assert first["depth"]["path"] != second["depth"]["path"]
    assert first["shape_corrected_baseline"]["path"] != second["shape_corrected_baseline"]["path"]
    assert first["configured_depth_map"] != second["configured_depth_map"]
    with pytest.raises(ValueError,match="selected run missing"):
        frame.selected_fields(corrected,"missing")


def test_no_common_field_fallback_when_selected_run_loses_a_field(corrected):
    del corrected["frame_binding"]["runs"]["second"]["fields"]["depth"]
    with pytest.raises(ValueError,match="explicit absolute"):
        frame.selected_fields(corrected,"second")
    assert frame.selected_fields(corrected,"first")["depth"]["shape"]==[2,3]


def test_reference_replacement_changes_display_path_only(corrected,tmp_path):
    original = SeriesRunInputs("second",tmp_path/"original.toml",tmp_path/"raw.JPG",
                              crop={"original":True},width_m=.9,height_m=.49)
    actual = frame.workspace_reference(corrected,original)
    assert actual==replace(original,baseline_image=Path(corrected["frame_binding"]["runs"]["second"]["fields"]["shape_corrected_baseline"]["path"]))
    assert original.baseline_image==tmp_path/"raw.JPG"
    assert corrected["frame_binding"]["source_reference_run"]=="historical"


@pytest.mark.parametrize("mutation", ["unknown_frame","missing_correction","wrong_shape","changed_size"])
def test_invalid_frame_or_source_is_visible_without_raw_fallback(corrected,mutation):
    if mutation=="unknown_frame":corrected["frame"]="guessed"
    elif mutation=="missing_correction":corrected["frame_binding"]["reference_shape_corrections"]=[]
    elif mutation=="wrong_shape":corrected["frame_binding"]["runs"]["first"]["fields"]["depth"]["shape"]=[3,2]
    else:Path(corrected["frame_binding"]["runs"]["first"]["fields"]["labels"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):frame.selected_fields(corrected,"first")


def test_corrected_canvas_reads_saved_rgb_not_raw_photograph(corrected,tmp_path,monkeypatch):
    raw = np.full((2,3,3),.25,dtype=np.float32)
    monkeypatch.setattr(geometry,"_load_array",lambda path: raw)
    monkeypatch.setattr(painting,"load_painting_image",lambda *a,**k: pytest.fail("corrected canvas entered raw reader"))
    reference = SeriesRunInputs("second",tmp_path/"config.toml",tmp_path/"not-read.JPG")
    actual = frame.load_painting_image(corrected,reference,analysis_width=20,display_limit=20)
    assert actual.path==Path(corrected["frame_binding"]["runs"]["second"]["fields"]["shape_corrected_baseline"]["path"])
    assert actual.image_size==(3,2)
    assert np.array_equal(actual.display,raw)
    assert np.array_equal(raw,np.full((2,3,3),.25,dtype=np.float32))


def test_corrected_canvas_refuses_changed_decoded_shape(corrected,tmp_path,monkeypatch):
    monkeypatch.setattr(geometry,"_load_array",lambda path: np.zeros((3,3,3),np.float32))
    with pytest.raises(ValueError,match="bound grid"):
        frame.load_painting_image(corrected,SeriesRunInputs("first",tmp_path/"config.toml"))


def test_saved_boundary_view_calls_existing_mask_and_never_relabels(corrected,monkeypatch):
    labels=np.array([[0,0,8],[3,3,8]],np.uint8);before=labels.copy();calls=[]
    monkeypatch.setattr(geometry,"_load_array",lambda path: labels)
    original=geometry._layer_boundary_mask
    def observe(value, ids):
        calls.append((value.copy(),set(ids)))
        return original(value,ids)
    monkeypatch.setattr(geometry,"_layer_boundary_mask",observe)
    image=frame.saved_boundaries(corrected,"first",(2,3))
    assert image.shape==(2,3,4) and calls[0][1]=={0,3,8}
    assert np.array_equal(labels,before)
    assert np.array_equal(calls[0][0],labels)


def test_corrected_metric_edges_use_recorded_dimensions_and_pixel_count(corrected):
    value=frame.corrected_rig_frame(corrected,"first")
    assert value.image_size==(3,2) and value.width_m==.9 and value.height_m==.49
    assert value.corners["top_left"]==(-.5,-.5)
    assert value.corners["bottom_right"]==(1.5,2.5)
    assert value.run_id=="first"


def test_current_revision_read_is_bounded_and_does_not_scan_history(tmp_path,monkeypatch):
    path=tmp_path/"current.json"
    monkeypatch.setattr(geometry,"GeometryRevisionStore",lambda p: SimpleNamespace(current_path=path))
    path.write_bytes(b" "*(frame.METADATA_LIMIT+1))
    with pytest.raises(ValueError,match="1 MiB"):
        frame.current_configuration(object())
    path.write_text(json.dumps(dict(frame="rig-corrected",configuration={"frame":"raw"})))
    with pytest.raises(ValueError,match="different coordinate"):
        frame.current_configuration(object())
    path.unlink()
    assert frame.current_configuration(object())=={}


def test_all_selected_field_locations_follow_declared_archive_alias_without_rewriting_identity(tmp_path,monkeypatch):
    from darsia.fluidflower import storage_paths
    source, archived = tmp_path/"former-results", tmp_path/"archive-results"
    rows = {}
    for index in range(39):
        run = f"run{index:02}"
        folder = archived/run
        folder.mkdir(parents=True)
        fields = {}
        for field in frame.FIELDS:
            bound = asset(folder/(field+".npz"), [2,3,3] if field=="shape_corrected_baseline" else [2,3])
            fields[field] = dict(bound,path=str(source/run/(field+".npz")))
        rows[run] = dict(fields=fields,configured_depth_map=dict(path=str(folder/"depth.npz")))
    correction = asset(tmp_path/"shape.npz", [2,3])
    configuration = dict(frame="rig-corrected",frame_binding=dict(format=frame.FRAME_BINDING_FORMAT,
        source_reference_run="historical",display_reference_run="run00",reference_shape_corrections=[correction],runs=rows))
    before = deepcopy(configuration)
    registry = tmp_path/"registry.toml"
    registry.write_text("schema_version=1\n[storage.path_aliases]\n"+
        json.dumps(str(source))+"="+json.dumps(str(archived))+"\n",encoding="utf-8")
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY",str(registry))
    storage_paths.refresh_storage_paths()
    for run in rows:
        selected = frame.selected_fields(configuration,run)
        for field in frame.FIELDS:
            assert Path(selected[field]["path"]) == archived/run/(field+".npz")
            assert {key:value for key,value in selected[field].items() if key!="path"} == {
                key:value for key,value in rows[run]["fields"][field].items() if key!="path"}
    assert configuration == before
    (archived/"run38"/"depth.npz").write_bytes(b"changed-size")
    with pytest.raises(ValueError,match="size changed"):
        frame.selected_fields(configuration,"run38")
