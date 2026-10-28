"""Tiny explicit Q43 edge replication; no measured shifts or real geometry."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import geometry_raster_sources as sources
from tests.unit.test_geometry_imported_rasters import imported, save, snapshot
from tests.unit.test_geometry_registered_runs import registered


@pytest.mark.parametrize("row,column,expected",[
    (1,1,[[1,1,2,3],[1,1,2,3],[5,5,6,7]]),
    (-1,-2,[[7,8,8,8],[11,255,255,255],[11,255,255,255]]),
    (1,-1,[[2,3,4,4],[2,3,4,4],[6,7,8,8]]),
    (-1,1,[[5,5,6,7],[9,9,10,11],[9,9,10,11]]),
    (1,0,[[1,2,3,4],[1,2,3,4],[5,6,7,8]]),
    (-1,0,[[5,6,7,8],[9,10,11,255],[9,10,11,255]]),
    (0,1,[[1,1,2,3],[5,5,6,7],[9,9,10,11]]),
    (0,-1,[[2,3,4,4],[6,7,8,8],[10,11,255,255]])])
def test_asymmetric_all_axes_and_corners_are_nearest_covered_values(row,column,expected):
    original=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,255]],dtype=np.uint16)
    before=original.copy()
    translated,detail=sources.integer_label_edge_replication(original,
        row_shift=row,column_shift=column,pixel_size_mm=.25)
    assert np.array_equal(translated,np.array(expected,dtype=np.uint16))
    assert translated.dtype==original.dtype and not np.shares_memory(translated,original)
    assert np.array_equal(original,before)
    assert set(map(int,np.unique(translated))) <= set(map(int,np.unique(original)))
    assert detail["interpolation"] is False and detail["wrapping"] is False
    assert detail["fill_mode"]=="edge-replication" and "fill_label" not in detail


def test_exact_per_edge_rectangles_area_and_corner_overlap():
    original=np.arange(1,13,dtype=np.uint8).reshape(3,4)
    _,detail=sources.integer_label_edge_replication(original,row_shift=1,column_shift=1,pixel_size_mm=.25)
    filled=detail["geometry_filled"]
    assert filled["shape"]==[3,4] and filled["pixel_count"]==6 and filled["area_mm2"]==6/16
    assert filled["edges"]=={
        "top":{"bounds_rc":[0,1,0,4],"pixel_count":4,"area_mm2":4/16},
        "bottom":{"bounds_rc":[3,3,0,4],"pixel_count":0,"area_mm2":0},
        "left":{"bounds_rc":[0,3,0,1],"pixel_count":3,"area_mm2":3/16},
        "right":{"bounds_rc":[0,3,4,4],"pixel_count":0,"area_mm2":0}}
    assert filled["corner_overlap_pixel_count"]==1 and filled["edge_counts_include_corner_overlap"] is True
    mask=np.zeros(original.shape,bool)
    for edge in filled["edges"].values():
        r0,r1,c0,c1=edge["bounds_rc"];mask[r0:r1,c0:c1]=True
    assert np.array_equal(mask,[[True,True,True,True],[True,False,False,False],[True,False,False,False]])
    assert detail["introduced_fill_pixel_count"]==detail["clipped_source_pixel_count"]==6


def test_zero_shift_preserves_existing_255_and_has_no_extrapolated_pixels():
    original=np.array([[255,1],[3,4]],np.uint8)
    actual,detail=sources.integer_label_edge_replication(original,row_shift=0,column_shift=0,pixel_size_mm=.25)
    assert np.array_equal(actual,original) and not np.shares_memory(original,actual)
    assert detail["geometry_filled"]["pixel_count"]==detail["clipped_source_pixel_count"]==0
    assert detail["clipped_source_label_counts"]=={}


@pytest.mark.parametrize("row,column",[(True,0),(0,False),(.5,0),(0,1.)])
def test_shifts_require_explicit_integers(row,column):
    with pytest.raises(ValueError,match="explicit integers"):
        sources.integer_label_edge_replication(np.ones((3,4),np.uint8),row_shift=row,column_shift=column,pixel_size_mm=.25)


@pytest.mark.parametrize("row,column",[(3,0),(-3,0),(0,4),(0,-4)])
def test_entire_parent_discard_is_still_forbidden(row,column):
    with pytest.raises(ValueError,match="overlapping"):
        sources.integer_label_edge_replication(np.ones((3,4),np.uint8),row_shift=row,column_shift=column,pixel_size_mm=.25)


@pytest.mark.parametrize("shape,dtype",[((2,3,1),np.uint8),((2,3),np.float32)])
def test_original_integer_scalar_grid_required(shape,dtype):
    with pytest.raises(ValueError,match="two-dimensional integer"):
        sources.integer_label_edge_replication(np.ones(shape,dtype),row_shift=0,column_shift=1,pixel_size_mm=.25)


@pytest.mark.parametrize("scale",[0.,-1.,float("nan"),True])
def test_per_edge_area_requires_explicit_positive_finite_pixel_size(scale):
    with pytest.raises(ValueError,match="finite"):
        sources.integer_label_edge_replication(np.ones((3,4),np.uint8),row_shift=0,column_shift=1,pixel_size_mm=scale)


def edge_registered(imported):
    draft,parent,_=registered(imported)
    translated,metrics=sources.integer_label_edge_replication(imported.arrays["labels"],
        row_shift=0,column_shift=1,pixel_size_mm=.1645)
    row=draft["run_geometry"]["runs"][0];record=row["registration"]
    np.savez(Path(row["labels_path"]),array=translated,
        metadata=geometry._darsia_metadata(Path(imported.config["labels_path"])))
    record.pop("fill_label");record["fill_mode"]="edge-replication"
    record["labels"]=sources.descriptor(row["labels_path"]);record["metrics"]=metrics
    return draft,parent,translated


def test_genuine_store_preserves_edge_flags_canonical_facies_and_full_csv(imported):
    original=snapshot(imported.original)
    draft,parent,expected=edge_registered(imported)
    parent_before=snapshot(Path(parent["artifacts"]["facies_path"]).parent)
    revision=save(imported,draft);actual=geometry.resolve_run_geometry_artifacts(revision,"one")
    assert np.array_equal(geometry._load_array(Path(actual["labels_path"])),expected)
    cfg=deepcopy(draft);cfg["labels_path"]=draft["run_geometry"]["runs"][0]["labels_path"]
    facies,_=geometry._canonical_facies(cfg)
    assert np.array_equal(geometry._load_array(Path(actual["facies_path"])),facies)
    assert Path(actual["facies_properties_path"]).read_bytes()==Path(imported.config["facies_properties_source_path"]).read_bytes()
    assert sources._full_csv(actual["facies_properties_path"])[2]==[1,2,3,4,6]
    assert revision["configuration"]["run_geometry"]["runs"][0]["registration"]["metrics"]["geometry_filled"]==draft["run_geometry"]["runs"][0]["registration"]["metrics"]["geometry_filled"]
    assert snapshot(imported.original)==original and snapshot(Path(parent["artifacts"]["facies_path"]).parent)==parent_before


@pytest.mark.parametrize("change",["missing_flags","area","edge_bounds","constant_fill","unknown_mode","nonrigid"])
def test_store_rejects_unbound_edge_flags_or_changed_rigid_translation(imported,change):
    draft,_,labels=edge_registered(imported);row=draft["run_geometry"]["runs"][0];record=row["registration"]
    if change=="missing_flags":record["metrics"].pop("geometry_filled")
    elif change=="area":record["metrics"]["geometry_filled"]["area_mm2"]+=1
    elif change=="edge_bounds":record["metrics"]["geometry_filled"]["edges"]["left"]["bounds_rc"][3]+=1
    elif change=="constant_fill":record["fill_label"]=1
    elif change=="unknown_mode":record["fill_mode"]="automatic"
    else:
        labels=labels.copy();labels[0,0]=2 if labels[0,0]!=2 else 1
        np.savez(row["labels_path"],array=labels,metadata=geometry._darsia_metadata(Path(imported.config["labels_path"])))
        record["labels"]=sources.descriptor(row["labels_path"])
    before=snapshot(imported.project.root)
    with pytest.raises(ValueError):save(imported,draft)
    assert snapshot(imported.project.root)==before
