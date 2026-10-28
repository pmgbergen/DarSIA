"""Synthetic integer categorical registration through the genuine revision Store.

No measured displacement, real map, export, Build, or migration is used here.
"""
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import darsia
import numpy as np
import pytest

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import geometry_raster_sources as sources
from tests.unit.test_geometry_imported_rasters import imported, save, snapshot


def registered(imported, *, row_shift=0, column_shift=1, run="one"):
    parent = save(imported)
    draft = sources.prepare_canonical_edit(imported.project, actor="synthetic reviewer",
                                            reason="explicit integer registration test")
    labels, metrics = sources.integer_label_translation(imported.arrays["labels"],
        row_shift=row_shift, column_shift=column_shift, fill_label=1)
    path = imported.project.root / (run + "-translated.npz")
    np.savez(path, array=labels, metadata=geometry._darsia_metadata(Path(imported.config["labels_path"])))
    evidence = imported.project.root / "explicit-authority.json"
    evidence.write_bytes(b'{"synthetic":true}\n')
    measurement = imported.project.root / "saved-measurement.json"
    measurement.write_bytes(b'{"synthetic_measured_shift":true}\n')
    record = dict(parent_revision_id=parent["revision_id"],
        parent_labels=sources.descriptor(imported.config["labels_path"]),
        labels=sources.descriptor(path), authority=sources.descriptor(evidence),
        measurement=sources.descriptor(measurement), source_image=sources.descriptor(imported.config["baseline_path"]),
        status="registered", row_shift=row_shift, column_shift=column_shift, fill_label=1,
        candidate=sources.registration_pixel_shift(dx_mm=column_shift*.1645,
            dy_mm=row_shift*.1645,pixel_size_mm=.1645), metrics=metrics)
    draft["run_geometry"]["runs"] = [dict(run_id=run, geometry_choice="run_override",
        enabled=True, acquisition_state="unknown", labels_path=str(path),
        source_image_path=imported.config["baseline_path"], registration=record)]
    return draft, parent, labels


def test_asymmetric_positive_translation_preserves_values_dtype_and_input():
    original = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,255]],dtype=np.uint16)
    before = original.copy()
    actual, detail = sources.integer_label_translation(original,row_shift=1,column_shift=1,fill_label=255)
    expected = np.array([[255,255,255,255],[255,1,2,3],[255,5,6,7]],dtype=np.uint16)
    assert np.array_equal(actual,expected) and actual.dtype==original.dtype
    assert np.array_equal(original,before) and not np.shares_memory(original,actual)
    assert detail["clipped_source_pixel_count"]==6
    assert detail["clipped_source_label_counts"]=={"4":1,"8":1,"9":1,"10":1,"11":1,"255":1}
    assert detail["introduced_fill_pixel_count"]==6
    assert detail["interpolation"] is False and detail["wrapping"] is False


def test_negative_translation_has_no_roll_or_interpolation():
    original=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,255]],dtype=np.uint16)
    actual,_=sources.integer_label_translation(original,row_shift=-1,column_shift=-2,fill_label=255)
    assert np.array_equal(actual,np.array([[7,8,255,255],[11,255,255,255],[255,255,255,255]],dtype=np.uint16))


def test_zero_translation_is_an_exact_independent_copy():
    x=np.array([[1,255],[3,4]],np.uint8)
    y,metrics=sources.integer_label_translation(x,row_shift=0,column_shift=0,fill_label=255)
    assert np.array_equal(y,x) and y.dtype==x.dtype and not np.shares_memory(x,y)
    assert metrics["clipped_source_pixel_count"]==0
    assert metrics["clipped_source_label_counts"]=={}


@pytest.mark.parametrize("kwargs",[dict(row_shift=True,column_shift=0,fill_label=1),
    dict(row_shift=.5,column_shift=0,fill_label=1),dict(row_shift=0,column_shift=False,fill_label=1),
    dict(row_shift=0,column_shift=0,fill_label=1.0)])
def test_offsets_and_fill_are_explicit_integers(kwargs):
    with pytest.raises(ValueError,match="explicit integers"):
        sources.integer_label_translation(np.ones((2,3),np.uint8),**kwargs)


@pytest.mark.parametrize("fill",[0,99])
def test_new_fill_category_is_never_invented(fill):
    with pytest.raises(ValueError,match="existing parent label"):
        sources.integer_label_translation(np.ones((2,3),np.uint8),row_shift=0,column_shift=1,fill_label=fill)


@pytest.mark.parametrize("shape,dtype",[((2,3,1),np.uint8),((2,3),np.float32)])
def test_translation_requires_the_original_integer_2d_grid(shape,dtype):
    with pytest.raises(ValueError,match="two-dimensional integer"):
        sources.integer_label_translation(np.ones(shape,dtype),row_shift=0,column_shift=0,fill_label=1)


@pytest.mark.parametrize("row,column",[(2,0),(0,-3)])
def test_translation_cannot_discard_the_entire_parent(row,column):
    with pytest.raises(ValueError,match="overlapping"):
        sources.integer_label_translation(np.ones((2,3),np.uint8),row_shift=row,column_shift=column,fill_label=1)


def test_metric_conversion_records_integer_grid_and_quantization_residuals():
    value=sources.registration_pixel_shift(dx_mm=.25,dy_mm=-.5,pixel_size_mm=.1645)
    assert value["column_shift"]==2 and value["row_shift"]==-3
    assert value["represented_dx_mm"]==2*.1645 and value["represented_dy_mm"]==-3*.1645
    assert value["residual_dx_mm"]==.25-2*.1645
    assert value["residual_dy_mm"]==-.5+3*.1645
    assert value["interpolation"] is False


@pytest.mark.parametrize("value",[0.,-1.,float("nan"),True])
def test_metric_conversion_requires_an_explicit_positive_finite_scale(value):
    with pytest.raises(ValueError,match="finite"):
        sources.registration_pixel_shift(dx_mm=1.,dy_mm=-1.,pixel_size_mm=value)


def test_real_store_materializes_run_labels_canonical_facies_and_entire_csv(imported):
    original=snapshot(imported.original)
    draft,parent,expected=registered(imported)
    parent_files=snapshot(Path(parent["artifacts"]["facies_path"]).parent)
    revision=save(imported,draft)
    actual=geometry.resolve_run_geometry_artifacts(revision,"one")
    assert actual["scope"]=="run_override"
    assert np.array_equal(geometry._load_array(Path(actual["labels_path"])),expected)
    run_config=deepcopy(draft);run_config["labels_path"]=draft["run_geometry"]["runs"][0]["labels_path"]
    expected_facies,_=geometry._canonical_facies(run_config)
    assert np.array_equal(geometry._load_array(Path(actual["facies_path"])),expected_facies)
    assert geometry._load_array(Path(actual["facies_path"])).dtype==np.dtype(np.int32)
    assert Path(actual["facies_properties_path"]).read_bytes()==Path(imported.config["facies_properties_source_path"]).read_bytes()
    assert sources._full_csv(actual["facies_properties_path"])[2]==[1,2,3,4,6]
    assert revision["configuration"]["run_geometry"]["runs"][0]["registration"]==draft["run_geometry"]["runs"][0]["registration"]
    assert snapshot(imported.original)==original
    assert snapshot(Path(parent["artifacts"]["facies_path"]).parent)==parent_files


@pytest.mark.parametrize("mutation",["missing_authority","missing_measurement","parent","unbound_row","independent_facies"])
def test_registration_never_admits_unbound_parent_or_evidence(imported,mutation):
    draft,_,_=registered(imported);row=draft["run_geometry"]["runs"][0]
    if mutation=="missing_authority":row["registration"].pop("authority")
    elif mutation=="missing_measurement":row["registration"].pop("measurement")
    elif mutation=="parent":row["registration"]["parent_revision_id"]="0"*32
    elif mutation=="unbound_row":row.pop("registration")
    else:row["facies_path"]=imported.config["facies_source_path"]
    before=snapshot(imported.project.root)
    with pytest.raises(ValueError):save(imported,draft)
    assert snapshot(imported.project.root)==before


@pytest.mark.parametrize("mutation",["file_changed","nonrigid","dtype","metadata"])
def test_registration_rejects_changed_or_noninteger_translation_artifacts(imported,mutation):
    draft,_,labels=registered(imported);row=draft["run_geometry"]["runs"][0];p=Path(row["labels_path"])
    if mutation=="file_changed":p.write_bytes(p.read_bytes()+b'changed')
    else:
        data=labels.copy()
        if mutation=="nonrigid":data[0,0]=2
        if mutation=="dtype":data=data.astype(np.int32)
        metadata={"dimensions":[999.,1.]} if mutation=="metadata" else {}
        np.savez(p,array=data,metadata=metadata)
        row["registration"]["labels"]=sources.descriptor(p)
    before=snapshot(imported.project.root)
    with pytest.raises(ValueError):save(imported,draft)
    assert snapshot(imported.project.root)==before


def test_registration_cannot_change_full_csv_physical_properties(imported):
    draft,_,_=registered(imported);draft["regions"][0]["porosity"]=.9
    with pytest.raises(ValueError,match="physical CSV properties"):save(imported,draft)


def test_corrected_run_source_cannot_borrow_another_runs_photograph(imported):
    source=sources.descriptor(imported.config["baseline_path"])
    imported.config["frame_binding"]={"runs":{"one":{"fields":{"shape_corrected_baseline":dict(source)}}}}
    imported.config=sources.bind_imported_rasters(imported.config,
        sources=imported.config["imported_rasters"]["sources"],
        diagnosis=imported.config["imported_rasters"]["diagnosis"])
    draft,_,_=registered(imported)
    sources.canonical_parent_payload(draft)
    other=imported.project.root/'other-baseline.npz';other.write_bytes(Path(source['path']).read_bytes())
    row=draft["run_geometry"]["runs"][0];row["source_image_path"]=str(other);row["registration"]["source_image"]=sources.descriptor(other)
    with pytest.raises(ValueError,match="existing bound corrected baseline"):
        sources.canonical_parent_payload(draft)


@pytest.mark.parametrize("missing",[False,True])
def test_recorded_metric_candidate_must_match_the_actual_integer_shift(imported,missing):
    draft,_,_=registered(imported);record=draft["run_geometry"]["runs"][0]["registration"]
    if missing:record.pop("candidate")
    else:record["candidate"]["represented_dx_mm"]+=.1645
    before=snapshot(imported.project.root)
    with pytest.raises(ValueError):save(imported,draft)
    assert snapshot(imported.project.root)==before


def test_recorded_clipping_metrics_must_match_actual_parent_pixels(imported):
    draft,_,_=registered(imported)
    draft["run_geometry"]["runs"][0]["registration"]["metrics"]["clipped_source_pixel_count"]+=1
    before=snapshot(imported.project.root)
    with pytest.raises(ValueError,match="translation metrics"):save(imported,draft)
    assert snapshot(imported.project.root)==before


def _real_image_metadata(*, dated=False):
    return darsia.Image(np.zeros((2, 3), np.float32), space_dim=2, indexing="ij",
        dimensions=[.49, .9], origin=np.array([0., .49], np.float64),
        scalar=False, series=False,
        date=datetime(2026, 10, 4, 12, 30) if dated else None).metadata()


@pytest.mark.parametrize("dated", [False, True])
def test_original_image_metadata_npz_roundtrip_is_exact_without_json(tmp_path, dated):
    metadata = _real_image_metadata(dated=dated)
    path = tmp_path / "genuine-image-metadata.npz"
    np.savez(path, array=np.zeros((2, 3), np.uint8), metadata=metadata)
    actual = geometry._darsia_metadata(path)
    assert type(actual["origin"]) is darsia.Coordinate
    assert actual["origin"].dtype == np.dtype(np.float64)
    assert actual["origin"].shape == (2,)
    assert actual["type"] is darsia.Image
    assert sources._same_grid_metadata(metadata, actual)
    assert sources._same_grid_metadata(actual, deepcopy(actual))


@pytest.mark.parametrize("change", ["origin_value", "origin_dtype", "origin_shape",
    "origin_class", "dimensions_type", "integer_type", "flag_type", "missing",
    "extra", "image_class", "date_value", "date_type", "reference_date", "numpy_scalar"])
def test_real_metadata_equality_rejects_every_changed_field_or_type(change):
    original = _real_image_metadata(dated=True)
    changed = deepcopy(original)
    if change == "origin_value": changed["origin"][0] = .001
    elif change == "origin_dtype": changed["origin"] = changed["origin"].astype(np.float32)
    elif change == "origin_shape": changed["origin"] = changed["origin"].reshape(1, 2)
    elif change == "origin_class": changed["origin"] = np.asarray(changed["origin"]).copy()
    elif change == "dimensions_type": changed["dimensions"] = tuple(changed["dimensions"])
    elif change == "integer_type": changed["space_dim"] = 2.0
    elif change == "flag_type": changed["scalar"] = 0
    elif change == "missing": changed.pop("name")
    elif change == "extra": changed["unbound"] = None
    elif change == "image_class": changed["type"] = np.ndarray
    elif change == "date_value": changed["date"] += timedelta(microseconds=1)
    elif change == "date_type": changed["date"] = changed["date"].date()
    elif change == "reference_date": changed["reference_date"] += timedelta(seconds=1)
    elif change == "numpy_scalar":
        original["time"] = np.float64(0.)
        changed["time"] = np.float32(0.)
    assert not sources._same_grid_metadata(original, changed)
    assert not sources._same_grid_metadata(changed, original)


def test_metadata_equality_never_forgives_nan_or_object_scalar_type_changes():
    assert not sources._same_grid_metadata(np.array([np.nan]), np.array([np.nan]))
    assert not sources._same_grid_metadata(float("nan"), float("nan"))
    assert not sources._same_grid_metadata(np.array([True], object), np.array([1], object))
    assert not sources._same_grid_metadata({True: "same"}, {1: "same"})


def _registered_with_real_metadata(data):
    metadata = _real_image_metadata()
    for name, array in data.arrays.items():
        np.savez(data.original / (name + ".npz"), array=array, metadata=metadata)
    data.config.update(width_m=.9, height_m=.49)
    data.config = sources.bind_imported_rasters(data.config,
        sources={key: sources.descriptor(data.config[sources.PATH_KEYS[key]]) for key in sources.KEYS},
        diagnosis=data.config["imported_rasters"]["diagnosis"])
    draft, parent, translated = registered(data)
    revision = save(data, draft)
    return revision, translated


def test_original_store_and_registered_reader_keep_genuine_coordinate_metadata(imported):
    revision, expected = _registered_with_real_metadata(imported)
    original_before = snapshot(imported.original)
    result = sources.registered_run_sources(imported.project.root, revision, "one")
    for name in ("labels", "facies"):
        path = Path(result["files"][name]["path"])
        assert sources._same_grid_metadata(geometry._darsia_metadata(path),
            geometry._darsia_metadata(Path(imported.config["labels_path"])))
    labels = sources._integer_map(result["files"]["labels"]["path"])
    assert labels.dtype == np.dtype(np.int32) and np.array_equal(labels, expected)
    assert snapshot(imported.original) == original_before


@pytest.mark.parametrize("field", ["labels", "facies"])
def test_registered_reader_rejects_changed_stored_coordinate_metadata(imported, field):
    revision, _ = _registered_with_real_metadata(imported)
    artifact = geometry.resolve_run_geometry_artifacts(revision, "one")
    path = Path(artifact[field + "_path"])
    values = geometry._load_array(path)
    metadata = geometry._darsia_metadata(path)
    metadata["origin"][0] += .001
    np.savez(path, array=values, metadata=metadata)
    with pytest.raises(ValueError, match="Stored registered " + field + " changed their grid metadata"):
        sources.registered_run_sources(imported.project.root, revision, "one")
