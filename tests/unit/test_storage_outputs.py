"""Synthetic Q23 output closure tests; no camera/model inference or jobs."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import storage_outputs as output


def request(resolution="native"):
    return output.parse_request({"storage": dict(format=output.REQUEST_FORMAT,
        resolution_mm=resolution, gas_score_encoding=output.LA_ENCODING,
        extent_m=[0., .002, 0., .002])}, "float32")


def capture():
    return dict(g3_pixel=np.array([[0., 1.], [.5, np.nan]], np.float32),
                flatfield_valid=np.array([[True, True], [True, False]]),
                source_box_xywh=[1, 0, 2, 2], probability_encoding={"storage_layout": "already-decoded"})


def fixture(tmp_path, resolution="native"):
    root=tmp_path / "attempt"
    stage=root / "preparation" / "ac52_f1"
    stage.mkdir(parents=True)
    labels=np.array([[0,1],[2,255]],np.uint8)
    path=stage / "phase_labels.png"
    with path.open("xb") as stream: Image.fromarray(labels).save(stream,format="PNG")
    phase={"labels":output._asset(path)}
    eligible=labels!=255
    products={"mass_total":np.where(eligible,3.,0.).astype(np.float32),
              "mass_aqueous":np.where(eligible,1.,0.).astype(np.float32),
              "mass_free_gas":np.where(eligible,1.,0.).astype(np.float32),
              "phase_labels":labels.copy(),
              "concentration_aqueous":np.ones(labels.shape,np.float32),
              "saturation_free_gas":np.full(labels.shape,.5,np.float32)}
    frame=SimpleNamespace(phase_labels=labels.copy(),domain=eligible.copy(),
        pixel_area_m2=np.full(labels.shape,1e-6,np.float64),
        depth_m=np.full(labels.shape,.02,np.float32),
        porosity=np.full(labels.shape,.3,np.float32),facies=np.ones(labels.shape,np.uint8),
        excluded_masks={})
    result=dict(total_mass_g=9.,aqueous_mass_g=3.,gas_mass_g=3.,precision="float32",
                fields={"main_mass_eligible_mask":eligible.copy()})
    evaluated=dict(result=result,export={"products":products})
    inferred=dict(labels=labels.copy(),classes={"rig_domain":eligible.copy()})
    pc=dict(g3_pixel=np.array([[0.,1.],[.5,np.nan]],np.float32),
            flatfield_valid=np.array([[True,True],[True,False]]),source_box_xywh=[0,0,2,2],
            probability_encoding={"storage_layout":"already-decoded"})
    mass=dict(frame=frame,volume_m3=np.full(labels.shape,.001,np.float64),
        x_m=np.array([[.0005,.0015],[.0005,.0015]]),
        y_m=np.array([[.0005,.0005],[.0015,.0015]]))
    task=dict(job_id="synthetic",run="ac52",configuration={"path":"config.json","sha256":"a"*64,"size_bytes":1},
              commit="b"*40,calibration_sha256="c"*64,precision="float32")
    spec=dict(case_id="ac52_f1",source={"source_sha256":"d"*64})
    return root,task,spec,phase,inferred,pc,mass,evaluated,request(resolution)


def test_absent_storage_request_is_an_exact_bypass():
    assert output.parse_request({},"float32") is None


def test_encoding_is_never_chosen_implicitly():
    with pytest.raises(ValueError,match="Explicit bound"):
        output.parse_request({"storage":{"format":output.REQUEST_FORMAT,"resolution_mm":"native"}},"float32")


def test_no_second_precision_setting_is_accepted():
    value=dict(request()); value["precision"]="float64"
    with pytest.raises(ValueError,match="Unknown"):
        output.parse_request({"storage":value},"float32")


def test_la_p_one_is_not_a_missing_code_and_crop_remains_missing():
    source=capture(); before={key: value.copy() for key,value in source.items() if isinstance(value,np.ndarray)}
    la,status=output.encode_before(source,(3,4))
    assert status["state"]=="complete"
    assert tuple(la[0,2])==(255,255)
    assert tuple(la[0,1])==(0,255)
    assert tuple(la[1,2])==(0,0)
    assert not la[:,0].any() and not la[:,3].any() and not la[2].any()
    for key,old in before.items(): np.testing.assert_array_equal(source[key],old)


@pytest.mark.parametrize("value",[-.01,1.01,float("inf")])
def test_non_probability_only_stops_before_product_without_rescaling(value):
    source=capture(); source["g3_pixel"][0,0]=value
    la,status=output.encode_before(source,(3,4))
    assert la is None and status["state"]=="pending"
    assert status["rescaled"] is False and status["clipped"] is False
    assert source["g3_pixel"][0,0]==value


def test_absent_native_centre_is_explicit_pending():
    la,status=output.encode_before({},(3,4))
    assert la is None and status["reason"]=="native_centre_not_computed"


def test_shape_mismatch_is_not_downgraded_to_missing():
    source=capture();source["source_box_xywh"]=[1,0,3,2]
    with pytest.raises(ValueError,match="score/support/grid"):
        output.encode_before(source,(3,4))


def test_la_png_roundtrip_retains_literal_quantization_and_nan_support(tmp_path):
    la,_=output.encode_before(capture(),(3,4)); path=tmp_path/"score.png"
    with path.open("xb") as stream:Image.fromarray(la,mode="LA").save(stream,format="PNG")
    p=output.decode_before(path,(3,4),"float32")
    assert p.dtype==np.float32 and p[0,2]==1 and p[0,1]==0
    assert p[1,1]==np.float32(128/255)
    assert np.isnan(p[1,2]) and np.isnan(p[2]).all()


def test_pixel_mass_keeps_independently_rounded_total():
    values={"products":{"mass_total":np.array([[3.]],np.float32),
                        "mass_aqueous":np.array([[1.]],np.float32),
                        "mass_free_gas":np.array([[1.]],np.float32)}}
    masses,_,_=output.pixel_masses(values,np.array([[.001]],np.float64),(1,1))
    assert masses["total"][0,0]==3 and masses["aqueous"][0,0]+masses["gas"][0,0]==2


@pytest.mark.parametrize("resolution",["native",1.])
def test_output_closes_before_native_commit_and_preserves_old_bytes(tmp_path,resolution):
    args=fixture(tmp_path,resolution); root,task,spec,phase,inferred,pc,mass,evaluated,req=args
    old=Path(phase["labels"]["path"]).read_bytes()
    entry=output.write_frame(*args)
    assert not (root/"frames"/spec["case_id"]).exists()  # caller has not committed native frame yet
    assert Path(phase["labels"]["path"]).read_bytes()==old
    doc=output.verify_entry(entry,task,spec,req,phase["labels"],evaluated["result"],
                            evaluated["export"]["products"],mass["volume_m3"])
    assert doc["final_label_counts"]=={"0":1,"1":1,"2":1,"255":1}
    assert doc["after_rules"]["gas_pixels"]==1
    assert all(check["passed"] for check in doc["native_pixel_mass_checks"].values())
    assert (doc["coarse"] is None)==(resolution=="native")
    with pytest.raises(FileExistsError):output.write_frame(*args)


def test_pending_before_still_has_honest_final_only_closure(tmp_path):
    args=list(fixture(tmp_path,1.)); args[5]={}
    entry=output.write_frame(*args); doc=json.loads(Path(entry["receipt"]["path"]).read_bytes())
    assert doc["complete"] and doc["before_rules"]["state"]=="pending"
    assert doc["after_rules"]["gas_pixels"]==1
    assert not any(Path(value["path"]).name.endswith(".png") for value in entry["files"])
    with np.load(next(value["path"] for value in entry["files"] if value["path"].endswith(".npz"))) as stored:
        assert "gas_probability_before_rules" not in stored
        # Q28 counts every native pixel; excluded label255 is zero gas, not absent.
        np.testing.assert_array_equal(stored["gas_probability_after_rules"],(args[4]["labels"]==2).astype(np.float32))


def test_mass_mismatch_preserves_own_partial_and_old_file(tmp_path):
    args=list(fixture(tmp_path)); root=args[0]; old=Path(args[3]["labels"]["path"]).read_bytes()
    args[7]["result"]["total_mass_g"]=8.
    with pytest.raises(ValueError,match="original total mass"):output.write_frame(*args)
    assert list((root/"preparation"/args[2]["case_id"]).glob("*.partial"))
    assert Path(args[3]["labels"]["path"]).read_bytes()==old
    assert not (root/"preparation"/args[2]["case_id"]/"storage_completed.json").exists()


def test_missing_storage_closure_is_an_error_not_recreated(tmp_path):
    args=fixture(tmp_path); entry=output.write_frame(*args)
    missing=dict(entry,receipt=dict(entry["receipt"],path=str(tmp_path/"missing"/"completed.json")))
    with pytest.raises((ValueError,FileNotFoundError)):
        output.verify_entry(missing,args[1],args[2],args[8],args[3]["labels"],args[7]["result"],
                            args[7]["export"]["products"],args[6]["volume_m3"])


def test_resume_verification_rejects_changed_native_density(tmp_path):
    args=fixture(tmp_path);entry=output.write_frame(*args)
    args[7]["export"]["products"]["mass_total"][0,0]=4.
    with pytest.raises(ValueError,match="independent native mass density"):
        output.verify_entry(entry,args[1],args[2],args[8],args[3]["labels"],args[7]["result"],
                            args[7]["export"]["products"],args[6]["volume_m3"])


def test_native_inventory_anchors_every_storage_asset(tmp_path):
    args=fixture(tmp_path);entry=output.write_frame(*args)
    rows={Path(bound['path']).name:bound for bound in [entry['receipt'],*entry['files']]}
    binding=dict(case_id=args[2]['case_id'],phase=args[3],storage=entry,files=rows)
    assert output.validate_publication_binding(binding)==entry
    changed=dict(binding,files=dict(rows))
    changed['files']['storage_completed.json']=dict(entry['receipt'],sha256='e'*64)
    with pytest.raises(ValueError,match='not anchored'):output.validate_publication_binding(changed)


def test_unlisted_storage_inventory_row_cannot_be_silently_published(tmp_path):
    args=fixture(tmp_path);entry=output.write_frame(*args)
    rows={Path(bound['path']).name:bound for bound in [entry['receipt'],*entry['files']]}
    rows['storage_unlisted.json']={'path':'unlisted'}
    binding=dict(case_id=args[2]['case_id'],phase=args[3],storage=entry,files=rows)
    with pytest.raises(ValueError,match='Unlisted'):output.validate_publication_binding(binding)


def test_every_original_preparation_row_and_path_is_preserved(tmp_path):
    args=fixture(tmp_path);stage=Path(args[3]['labels']['path']).parent
    originals=('phase_classes.npz','phase_area.npz','phase_qa.json',
               'phase_operator_masks.npz','history_input.json','native_audit.json',
               'temporal_log.npy','temporal_valid.npy')
    for name in originals:
        with (stage/name).open('xb') as stream:stream.write(('unchanged original '+name).encode())
    before={path.name:output._asset(path) for path in stage.iterdir() if path.is_file()}
    output.write_frame(*args)
    for name,bound in before.items():assert output._asset(stage/name)==bound
    new={path.name for path in stage.iterdir() if path.is_file()}-set(before)
    assert new=={'storage_completed.json','storage_gas_probability_before_rules.png'}


def original_commit_fixture(args,entry=None):
    root,task,spec,phase,inferred,pc,mass,evaluated,req=args
    directory=root/'frames'/spec['case_id'];directory.mkdir(parents=True)
    rows={Path(phase['labels']['path']).name:phase['labels']}
    binding=dict(case_id=spec['case_id'],phase=phase,files=rows)
    if entry is not None:
        binding['storage']=entry
        for bound in [entry['receipt'],*entry['files']]:rows[Path(bound['path']).name]=bound
    result={name:evaluated['result'][name] for name in output.TOTALS.values()}
    result['prediction']={'pipeline':binding}
    output._write_json(directory/'result.json',result)
    output._write_npz(directory/'products.npz',evaluated['export']['products'])
    output._write_npz(directory/'native_geometry.npz',{'transport_weight_l':mass['volume_m3']*1000.})
    files={name:output._asset(directory/name) for name in ('result.json','products.npz','native_geometry.npz')}
    record=dict(frame_id=spec['case_id'],nas_directory=str(directory),files=files)
    producer=dict(case_id=spec['case_id'],attempt_directory=str(root))
    return dict(records=[record],frame_producers=[producer]),{'frames':[spec]}


def test_native_committed_prefix_without_preceding_storage_is_a_stop(tmp_path):
    args=fixture(tmp_path);prefix,config=original_commit_fixture(args)
    original=Path(prefix['records'][0]['files']['result.json']['path']).read_bytes()
    with pytest.raises(ValueError,match='lacks its required preceding storage closure'):
        output.verify_prefix(prefix,args[1],config,args[8])
    assert Path(prefix['records'][0]['files']['result.json']['path']).read_bytes()==original
    assert not list((args[0]/'preparation'/args[2]['case_id']).glob('storage_*'))


def test_resume_verifies_native_commit_anchored_storage_without_recreation(tmp_path):
    args=fixture(tmp_path);entry=output.write_frame(*args)
    prefix,config=original_commit_fixture(args,entry)
    before={Path(bound['path']):Path(bound['path']).read_bytes() for bound in [entry['receipt'],*entry['files']]}
    assert output.verify_prefix(prefix,args[1],config,args[8])==[entry]
    for path,raw in before.items():assert path.read_bytes()==raw


def test_changed_encoded_prefix_product_is_stop_not_replayed(tmp_path):
    args=fixture(tmp_path);entry=output.write_frame(*args)
    prefix,config=original_commit_fixture(args,entry)
    path=next(Path(bound['path']) for bound in entry['files'] if bound['path'].endswith('.png'))
    path.write_bytes(b'deliberately corrupted synthetic product')
    with pytest.raises(ValueError,match='SHA|size|identity'):
        output.verify_prefix(prefix,args[1],config,args[8])
    assert path.read_bytes()==b'deliberately corrupted synthetic product'


def test_q28_written_coarse_mean_uses_decoded_pixels_not_area_or_domain(tmp_path):
    args=list(fixture(tmp_path,2.))
    args[6]['frame'].pixel_area_m2=np.array([[1e-6,9e-6],[2e-6,3e-6]],np.float64)
    entry=output.write_frame(*args)
    doc=json.loads(Path(entry['receipt']['path']).read_bytes())
    assert doc['after_rules']['label_legend']=={'0':'water','1':'dissolved','2':'gas','255':'excluded'}
    assert doc['after_rules']['support_mask'] is None
    assert doc['coarse']['probability_weighting']=='arithmetic pixel count, not pixel area'
    assert doc['coarse']['grid']['probability_coordinate_counts']['inside_extent']==4
    archive=next(Path(row['path']) for row in entry['files'] if row['path'].endswith('.npz'))
    with np.load(archive) as stored:
        expected=np.float32(np.array([0.,1.,128/255],np.float32).mean(dtype=np.float64))
        assert stored['gas_probability_before_rules'][0,0]==expected
        assert stored['gas_probability_before_rules_valid_pixel_fraction'][0,0]==np.float32(.75)
        assert stored['gas_probability_after_rules'][0,0]==np.float32(.25)
        assert stored['phase_gas_area_fraction'][0,0]==np.float32(2/12)
        assert stored['gas_probability_after_rules'][0,0]!=stored['phase_gas_area_fraction'][0,0]
        assert 'gas_probability_after_rules_valid_pixel_fraction' not in stored
        assert stored['cell_mass_total_g'][0,0]==9.  # existing independent density/volume mass


def test_q28_written_before_without_any_valid_pixel_is_nan_not_zero(tmp_path):
    args=list(fixture(tmp_path,2.)); args[5]['flatfield_valid'][:]=False
    entry=output.write_frame(*args)
    archive=next(Path(row['path']) for row in entry['files'] if row['path'].endswith('.npz'))
    with np.load(archive) as stored:
        assert np.isnan(stored['gas_probability_before_rules'][0,0])
        assert stored['gas_probability_before_rules_valid_pixel_fraction'][0,0]==0.
        assert stored['gas_probability_after_rules'][0,0]==np.float32(.25)


@pytest.mark.parametrize('axis',['x_m','y_m'])
@pytest.mark.parametrize('value',[-.0001,.002,.003])
def test_q34_canvas_preflight_precedes_mass_and_any_storage_side_effect(tmp_path,axis,value):
    args=list(fixture(tmp_path,1.))
    root,task,spec,phase,inferred,pc,mass,evaluated,req=args
    stage=root/'preparation'/spec['case_id']; before={p.name:output._asset(p) for p in stage.iterdir()}
    assert not mass['frame'].domain[1,1]
    mass[axis][1,1]=value
    evaluated['export']['products']['mass_total'][:]=np.nan  # Coordinate STOP must happen before density validation.
    with pytest.raises(ValueError,match='Full-canvas native pixel centre lies outside declared half-open'):
        output.write_frame(*args)
    assert {p.name:output._asset(p) for p in stage.iterdir()}==before
    assert not list(stage.glob('.storage.*.partial'))


def test_q34_native_storage_keeps_original_coarse_coordinate_bypass(tmp_path):
    args=list(fixture(tmp_path,'native'))
    args[6]['x_m'][:]=np.nan; args[6]['y_m'][:]=999.
    entry=output.write_frame(*args)
    doc=json.loads(Path(entry['receipt']['path']).read_bytes())
    assert doc['coarse'] is None and doc['before_rules']['state']=='complete'
    assert {Path(row['path']).name for row in entry['files']}=={'storage_gas_probability_before_rules.png'}


def test_q34_count_arrays_are_exact_existing_writer_outputs(tmp_path):
    args=list(fixture(tmp_path,2.)); entry=output.write_frame(*args)
    doc=json.loads(Path(entry['receipt']['path']).read_bytes())
    archive=next(Path(row['path']) for row in entry['files'] if row['path'].endswith('.npz'))
    with np.load(archive,allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored['static_included_area_native_pixel_count'],np.array([[3]],np.int64))
        np.testing.assert_array_equal(stored['static_full_canvas_probability_pixel_count'],np.array([[4]],np.int64))
        assert stored['static_pixel_area_m2_sum'][0,0]==np.float32(3e-6)
    assert set(doc['coarse']['grid']['cell_count_fields'])=={
        'static_included_area_native_pixel_count','static_full_canvas_probability_pixel_count'}


@pytest.mark.parametrize('extent',[[.0001,.0021,0.,.002],[0.,.0025,0.,.002]])
def test_q34_request_rejects_shifted_origin_or_noninteger_width(extent):
    config={'storage':dict(format=output.REQUEST_FORMAT,resolution_mm=1.,
                           gas_score_encoding=output.LA_ENCODING,extent_m=extent)}
    with pytest.raises(ValueError,match='exact origin-zero metric lattice'):
        output.parse_request(config,'float32')


def test_q34_request_accepts_negative_integer_origin_and_fractional_h_without_snapping():
    extent=[-.001,.00025,-.0005,.001]
    config={'storage':dict(format=output.REQUEST_FORMAT,resolution_mm=.25,
                           gas_score_encoding=output.LA_ENCODING,extent_m=extent)}
    actual=output.parse_request(config,'float32')
    assert actual['extent_m']==extent and actual['resolution_mm']==.25
    assert config['storage']['extent_m']==extent
