"""Metadata-only preparation tests. No real rig or raster is imported."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import toml
from darsia.fluidflower import geometry_import as subject
from darsia.fluidflower.geometry_workspace_frame import FRAME_BINDING_FORMAT


def bound(path, shape=None):
    raw=path.read_bytes()
    result=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw),mtime_ns=path.stat().st_mtime_ns)
    if shape is not None:result["shape"]=shape
    return result


@pytest.fixture
def metadata_inputs(tmp_path,monkeypatch):
    root=tmp_path/"series";root.mkdir()
    (root/"series.toml").write_text('schema_version=1\nseries_id="Synthetic"\n[runs]\nids=["first","second"]\n')
    source_map=root/"original-map.png";source_map.write_bytes(b"original source map; not decoded")
    source_image=root/"historical.JPG";source_image.write_bytes(b"original drawing photo; not asserted pre-injection")
    table=root/"facies.csv";table.write_bytes(b'id,name,porosity,permeability,swi CO2\n1,Sand,.4,10,.12\n')
    rows={}
    for run in ["first","second"]:
        folder=root/run;folder.mkdir()
        fields={}
        for name in ["shape_corrected_baseline","labels","facies","depth"]:
            p=folder/(name+".npz");p.write_bytes((run+name).encode())
            fields[name]=bound(p,[2,3,3] if name=="shape_corrected_baseline" else [2,3])
        config=dict(rig=dict(path=str(folder),width=.9,height=.49),
            depth=dict(depth_map=str(folder/"original-depth.npz")),
            labeling=dict(colored_image=str(source_map)),
            facies={"props":str(table),"path":str(folder/"original-facies.npz"),"1":{"labels":[0]}})
        p=root/(run+".toml");p.write_text(toml.dumps(config))
        rows[run]=dict(fields=fields,configuration=bound(p),configured_depth_map=dict(path=config["depth"]["depth_map"]))
    correction=root/"shape_correction.npz";correction.write_bytes(b"old correction")
    binding=dict(format=FRAME_BINDING_FORMAT,source_reference_run="historical",display_reference_run="first",
        reference_shape_corrections=[bound(correction)],runs=rows)
    project=SimpleNamespace(root=root/"geometry_facies",manifest={})
    monkeypatch.setattr(subject,"native_series_link",lambda p:{"series_root":str(root)})
    class Store:
        def __init__(self,p):self.current_path=p.root/"current.json"
        def revisions(self):return []
        def save_revision(self,*a,**k):pytest.fail("Preparation must not save a revision")
    monkeypatch.setattr(subject,"GeometryRevisionStore",Store)
    args=dict(series_root=root,configuration=root/"first.toml",facies_table=table,
        source_map=source_map,source_image=source_image,frame_binding=binding,
        facies_roles={1:"porous_medium"},permeability_unit="darcy",actor="synthetic explicit adoption")
    return project,args


def test_preparation_keeps_every_original_run_and_depth_pointer_without_writes(metadata_inputs):
    project,args=metadata_inputs
    before={str(p):p.read_bytes() for p in args["series_root"].rglob("*") if p.is_file()}
    result=subject.prepare_corrected_adoption(project,**args)
    config=result["candidate"]
    assert config["frame"]=="rig-corrected"
    assert config["frame_binding"]==args["frame_binding"]
    assert config["provenance"]["reference_run"]=="historical"
    assert config["provenance"]["display_reference_run"]=="first"
    assert config["provenance"]["original_configured_depth_pointer"]==args["frame_binding"]["runs"]["first"]["configured_depth_map"]
    assert len(result["reference_copy_plan"])==4
    for row in result["reference_copy_plan"]:
        assert not Path(row["target"]).exists()
    assert result["actual_Store_UUID"] is None and result["Store_called"] is False
    assert result["applied"] is False and result["Build_qualified"] is False
    assert not project.root.exists()
    assert all(Path(path).read_bytes()==raw for path,raw in before.items())


def test_one_run_cannot_be_relabelled_as_another_run(metadata_inputs):
    project,args=metadata_inputs
    binding=deepcopy(args["frame_binding"])
    binding["runs"]["second"]["fields"]["labels"]=binding["runs"]["first"]["fields"]["labels"]
    args["frame_binding"]=binding
    with pytest.raises(ValueError,match="another original rig"):
        subject.prepare_corrected_adoption(project,**args)
    assert not project.root.exists()


def test_changed_configured_depth_pointer_is_not_substituted_by_saved_rig_depth(metadata_inputs):
    project,args=metadata_inputs
    args["frame_binding"]["runs"]["second"]["configured_depth_map"]={"path":args["frame_binding"]["runs"]["second"]["fields"]["depth"]["path"]}
    with pytest.raises(ValueError,match="production depth pointer"):
        subject.prepare_corrected_adoption(project,**args)
    assert not project.root.exists()


def test_missing_run_and_changed_configuration_are_refused_before_copy(metadata_inputs):
    project,args=metadata_inputs
    old=deepcopy(args["frame_binding"])
    del args["frame_binding"]["runs"]["second"]
    with pytest.raises(ValueError,match="every actual registered"):
        subject.prepare_corrected_adoption(project,**args)
    args["frame_binding"]=old
    Path(old["runs"]["second"]["configuration"]["path"]).write_bytes(b"external edit")
    with pytest.raises(ValueError,match="configuration changed"):
        subject.prepare_corrected_adoption(project,**args)
    assert not project.root.exists()


def test_existing_source_unit_is_preserved_in_corrected_plan_without_conversion(metadata_inputs):
    project, args = metadata_inputs
    args["permeability_unit"] = "source_unit"
    plan = subject.prepare_corrected_adoption(project, **args)
    candidate = plan["candidate"]
    assert candidate["permeability_unit"] == "source_unit"
    assert [(r["label_id"], r["facies_id"], r["porosity"], r["permeability"])
            for r in candidate["regions"]] == [(0, 1, .4, 10)]
    assert candidate["frame_binding"] == args["frame_binding"]
    assert plan["Store_called"] is False and not project.root.exists()


def test_unknown_source_unit_is_still_refused_in_corrected_preparation(metadata_inputs):
    project, args = metadata_inputs
    args["permeability_unit"] = "unknown_unit"
    with pytest.raises(ValueError, match="source permeability unit"):
        subject.prepare_corrected_adoption(project, **args)
    assert not project.root.exists()
