"""Synthetic byte/stat roles only; no image/NPZ, numerical import or real job."""
from copy import deepcopy
import hashlib
import json
import stat
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import job_inputs as inputs
from darsia.fluidflower.store_map import StoreMapError


class MemoryFiles:
    def __init__(self, root="/series"):
        self.root=str(root).replace("\\","/")
        self.files={}
        self.opened=[]
        self.statted=[]
    def file(self, name, raw=b"payload"):
        path=self.root+"/"+name
        self.files[path]=raw
        return {"path":path,"sha256":hashlib.sha256(raw).hexdigest(),"size_bytes":len(raw)}
    def doc(self,name,value):
        return self.file(name,json.dumps(value,sort_keys=True).encode())
    def read(self,path):
        path=str(path).replace("\\","/")
        self.opened.append(path)
        assert path.endswith(".json"),"Payload must never be opened during J2"
        return self.files[path]
    def stat(self,path):
        path=str(path).replace("\\","/")
        self.statted.append(path)
        return SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_size=len(self.files[path]))
    def catalog(self,roots=None,aliases=()):
        return inputs.InputCatalog(roots or {"series":self.root},aliases,
                                   metadata_bytes=self.read,file_stat=self.stat)


def mass_fixture(files):
    clear={"rgb":files.file("clear.npy"),"valid":files.file("clear_valid.npy")}
    shared={"physical_reference":files.file("physical.npz"),
            "floor_curve":files.file("floor.npy"),"depth_points":[[1,2,3]],
            "reference_matrix":[[1,0,0],[0,1,0],[0,0,1]],"clearwater":clear,
            "optics":{"metadata":files.doc("optics.json",{}),
                      "color_paths":{"0":files.doc("color.json",{})},
                      "signals":{"0":files.file("signal.csv",b"supports,values\n0,0\n")},
                      "flash":files.doc("flash.json",{})},
            "facies_properties":{"path":"/not-an-input/facies.csv","sha256":"a"*64}}
    full={"native_centre":{"log_linear_rgb":files.file("log.npy"),"valid_mask":files.file("valid.npy")},
          "ring_windows":[{"log_linear_rgb":files.file("ring.npy")}]}
    frame={"case_id":"ac52_f1","current":full,"baseline":deepcopy(full),
           **{key:files.file(key+".npz") for key in ("photo_top","labels","classes","area")},
           "original_domain":{"parent_labels_map":files.file("original_domain.png"),
                              "native_shape_hw":[2,2],"frame_index":0}}
    config={"format":"darsia-n1-run-v1","run":"ac52","frames":[frame],"shared":shared,
            "calibration":files.doc("calibration.json",{}),
            "physical_correction":files.doc("physical_correction.json",{}),"boxes":{},
            "provenance":{"not_runtime":{"path":"C:/missing/historical.json","sha256":"b"*64}}}
    return config,files.doc("run.json",config)


def test_catalogue_follows_consumers_not_all_path_shaped_metadata():
    files=MemoryFiles();config,bound=mass_fixture(files)
    before=deepcopy(config)
    catalogue=files.catalog().collect(bound,config)
    encoded=json.dumps(catalogue)
    assert catalogue["complete"] and not catalogue["payload_hashes_verified"]
    assert not catalogue["numerical_calls_started"]
    assert "historical.json" not in encoded and "facies.csv" not in encoded
    assert all("C:" not in path and "not-an-input" not in path for path in files.statted)
    assert all(path.endswith(".json") for path in files.opened)
    assert config==before
    assert all(set(row["reference"])=={"root_alias","relative_path","sha256","bytes","role"} for row in catalogue["inputs"])
    assert "/series/" not in encoded and "C:" not in encoded


def test_unknown_consumed_root_refuses_before_stat_or_metadata_open():
    files=MemoryFiles()
    catalog=files.catalog()
    with pytest.raises(StoreMapError,match="No declared root"):
        catalog.add({"path":"/unknown/input.npy","sha256":"a"*64},"image","current")
    assert not files.opened and not files.statted


def test_alias_keeps_original_sha_but_portable_reference_has_no_logical_path():
    files=MemoryFiles()
    physical=files.file("native/input.npy")
    logical=dict(physical,path="C:/old/input.npy")
    catalog=files.catalog(aliases=[{"logical":logical,"physical":physical}])
    ref=catalog.add(logical,"image","selected.camera")
    assert ref["sha256"]==logical["sha256"] and ref["relative_path"]=="native/input.npy"
    assert "C:" not in json.dumps(ref)
    with pytest.raises(ValueError,match="changes source SHA"):
        files.catalog(aliases=[{"logical":logical,"physical":dict(physical,sha256="0"*64)}])


def test_unknown_or_corrupt_metadata_does_not_get_complete_catalogue():
    files=MemoryFiles();config,bound=mass_fixture(files)
    files.files[config["calibration"]["path"]]=b"not the original bytes"
    with pytest.raises(ValueError):
        files.catalog().collect(bound,config)
    with pytest.raises(ValueError,match="Unsupported role catalogue"):
        files.catalog().collect(bound,dict(config,format="new-unqualified-method"))


def test_actual_stat_size_is_used_when_original_descriptor_had_no_size():
    files=MemoryFiles();bound=files.file("original.npy",b"1234567")
    bound.pop("size_bytes")
    ref=files.catalog().add(bound,"image","raw")
    assert ref["bytes"]==7


def cohort_fixture(files):
    source=files.doc("source.json",{"run":"ac52","frame_sources":[{},{}]})
    temporal=files.doc("temporal.json",{"frames":[{"temporal_registration_pass":True},{"temporal_registration_pass":True}]})
    rows=[]
    for index in range(2):
        receipt={}
        for key in ("native_review_jpeg","camera_jpeg","v5_B_native_jpeg","v5_B_arw","v5_B_camera_jpeg"):
            bound=files.file(str(index)+"/"+key+".jpg")
            receipt[key+"_path"],receipt[key+"_sha256"]=bound["path"],bound["sha256"]
        for key in ("v5_B_registration_certificate","photo_sand_reference_receipt"):
            bound=files.doc(str(index)+"/"+key+".json",{})
            receipt[key+"_path"],receipt[key+"_sha256"]=bound["path"],bound["sha256"]
        mask=files.file(str(index)+"/photo_mask.npz")
        receipt.update(photo_sand_reference_mask_path=mask["path"],photo_sand_reference_mask_sha256=mask["sha256"],
                       temporal_geometry_receipt_path=temporal["path"],temporal_geometry_receipt_sha256=temporal["sha256"])
        bound=files.doc(str(index)+"/receipt.json",receipt)
        rows.append({"receipt_path":bound["path"],"receipt_sha256":bound["sha256"]})
    row={"run":"ac52","source_manifest_path":source["path"],"source_manifest_sha256":source["sha256"],"frames":rows}
    parameters={"source":{"symbols":{"prepared_source_id":"another-run"}}}
    return row,parameters


def test_photo_cohort_includes_unselected_original_consumed_frames_without_decoding():
    files=MemoryFiles();row,parameters=cohort_fixture(files)
    catalog=files.catalog()
    catalog._cohort(row,parameters,"native.processor")
    relative=[entry["reference"]["relative_path"] for entry in catalog.entries.values()]
    assert any(value.startswith("0/") for value in relative)
    assert any(value.startswith("1/") for value in relative)
    assert "1/camera_jpeg.jpg" in relative
    assert all(path.endswith(".json") for path in files.opened)


def test_parent_cache_is_explicit_role_not_a_current_expected_output():
    files=MemoryFiles()
    original=files.file("parent.npy",b"saved parent bits")
    frame={"original_log_path":original["path"],"original_log_sha256":original["sha256"]}
    parent=files.doc("parent.json",{"frame_sources":[frame]})
    labels=files.file("old_labels.png")
    receipt=files.doc("parent_frame.json",{"labels_map":labels})
    masks=files.file("old_masks.npz")
    probability=files.doc("probability.json",{"metadata":{"native_manifest_path":parent["path"]},
        "binding":{"native_manifest_sha256":parent["sha256"]},"files":{"masks":masks}})
    binding={"frame_index":0,"parent_native_manifest":parent,"parent_frame_receipt":receipt,
             "parent_probability_receipt":probability}
    catalog=files.catalog()
    catalog.retained(binding,{"parent_array_names":["original_log"],"mask_file":"masks"},"frame.retained")
    refs=[row["reference"]["relative_path"] for row in catalog.entries.values()]
    assert "old_labels.png" in refs and "old_masks.npz" in refs and "parent.npy" in refs
    assert all(path.endswith(".json") for path in files.opened)


def test_catalogue_schema_refuses_an_absolute_input_or_unknown_role():
    files=MemoryFiles();config,bound=mass_fixture(files)
    valid=files.catalog().collect(bound,config)
    inputs.validate_catalog(valid)
    bad=deepcopy(valid);bad["inputs"][0]["reference"]["relative_path"]="../escape"
    with pytest.raises(ValueError): inputs.validate_catalog(bad)
    bad=deepcopy(valid);bad["inputs"][0]["reference"]["role"]="executable"
    with pytest.raises(ValueError): inputs.validate_catalog(bad)


def test_ambiguous_named_consumer_refuses_at_collection_and_at_catalogue_admission():
    files=MemoryFiles();config,bound=mass_fixture(files)
    catalog=files.catalog()
    first=files.file("one.npy",b"one");second=files.file("two.npy",b"two")
    catalog.add(first,"image","selected.current")
    catalog.add(first,"image","selected.current")  # Same exact binding is idempotent.
    with pytest.raises(ValueError,match="Ambiguous input consumer"):
        catalog.add(second,"image","selected.current")
    valid=files.catalog().collect(bound,config)
    changed=deepcopy(valid)
    changed["inputs"][1]["uses"]=[changed["inputs"][0]["uses"][0]]
    with pytest.raises(ValueError,match="Ambiguous input consumer"):
        inputs.validate_catalog(changed)


def test_catalogue_out_of_its_canonical_order_is_refused_when_made():
    """A real collection is in canonical order; a row moved by hand is refused."""
    files=MemoryFiles();config,bound=mass_fixture(files)
    valid=files.catalog().collect(bound,config)
    order=[inputs.catalog_order_key(row["reference"]) for row in valid["inputs"]]
    assert len(order)>=2 and order==sorted(order)
    inputs.validate_catalog(valid)
    moved=deepcopy(valid);moved["inputs"].append(moved["inputs"].pop(0))
    with pytest.raises(ValueError,match="canonical order"):
        inputs.validate_catalog(moved)


def test_catalog_order_key_is_alias_path_sha_role():
    """The order key is the InputCatalog entry key: alias, path, SHA-256, role."""
    reference=dict(root_alias="series",relative_path="a/b.json",sha256="0"*64,bytes=3,role="rig_model")
    assert inputs.catalog_order_key(reference)==("series","a/b.json","0"*64,"rig_model")


def test_toml_boxes_are_bound_as_original_roi_input_without_json_decoding():
    files = MemoryFiles()
    config, _ = mass_fixture(files)
    original = b'[[roi]]\nname = "box1"\nroi = [[0, 0], [1, 1]]\n'
    boxes = files.file("common.toml", original)
    config["boxes"] = boxes
    bound = files.doc("run.json", config)
    before = deepcopy(config)
    catalogue = files.catalog().collect(bound, config)
    rows = [row for row in catalogue["inputs"] if "mass.boxes" in row["uses"]]
    assert len(rows) == 1
    assert rows[0]["reference"] == {
        "root_alias": "series", "relative_path": "common.toml",
        "sha256": boxes["sha256"], "bytes": len(original), "role": "rig_model",
    }
    assert boxes["path"] in files.statted
    assert boxes["path"] not in files.opened
    assert files.files[boxes["path"]] == original and config == before
    inputs.validate_catalog(catalogue)


def test_original_domain_binds_only_original_parent_labels_not_the_metadata_container():
    files = MemoryFiles()
    config, bound = mass_fixture(files)
    catalogue = files.catalog().collect(bound, config)
    rows = [row for row in catalogue["inputs"]
            if "frame.ac52_f1.mass.original_domain.parent_labels_map" in row["uses"]]
    assert len(rows) == 1
    assert rows[0]["reference"]["relative_path"] == "original_domain.png"
    assert rows[0]["reference"]["sha256"] == config["frames"][0]["original_domain"]["parent_labels_map"]["sha256"]
    assert all(path.endswith(".json") for path in files.opened)
    inputs.validate_catalog(catalogue)
