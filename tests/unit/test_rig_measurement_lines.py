"""Two image endpoints define one measured distance, not invented x/y anchors."""
from copy import deepcopy
import numpy as np
import pytest
from darsia.fluidflower import rig_static_references as refs
from darsia.fluidflower import rig_measurement_lines as lines
from .test_rig_static_references import scene,packet,transformed_scene


def source_line(tmp_path):
    source,_,positions=scene(tmp_path)
    value=lines.capture_measurement_line(source,line_id='span',name='Measured span',
        points_u_xy=[[0.,40.],[575.,60.]],length_mm=195.,plane_id='bolt-heads')
    return source,value,positions


def test_measured_endpoints_follow_static_appearance_with_other_camera_profile(tmp_path):
    source,line,positions=source_line(tmp_path)
    h=np.array([[1.13,.04,42.],[.02,1.08,-9.],[.00008,-.0001,1.]])
    target=packet(tmp_path,transformed_scene(source,positions,h),run='target',origin=(-160,-80),profile='second')
    document=refs.empty_document()
    document['measurement_lines']=[line]
    result=refs.recognize_regions(document,target)
    observed=result['measurement_lines']['span']
    expected=refs._project(h,np.asarray(line['points_u_xy'])+[100,60])+[-160,-80]
    assert observed['status']=='recognized',observed
    np.testing.assert_allclose(observed['points_u_xy'],expected,atol=1.8)
    assert observed['length_mm']==195.
    assert result['model'] is None and not result['observations']
    endpoints=[item for item in line['endpoint_regions'] if item]
    fitted=refs.fit_reference_model(endpoints,{item['id']:dict(status='manual',anchor_u_xy=item['anchor_u_xy']) for item in endpoints})
    assert fitted['used_region_ids']==[]


def test_metadata_edits_keep_manual_endpoints_and_explicit_save(tmp_path):
    source,line,_=source_line(tmp_path)
    manual=lines.manual_measurement_observation(line,source,[[2.,42.],[570.,65.]])
    line=lines.update_measurement_line(line,name='Renamed',length_mm=200.,plane_id='rig')
    result=dict(photo=refs.photo_binding(source),measurement_lines={'span':manual},observations={})
    observed=lines.measurement_for_photo(line,source,result)
    assert observed['points_u_xy']==manual['points_u_xy']
    assert observed['length_mm']==200.
    document=refs.empty_document()
    document['measurement_lines']=[line]
    document['runs']={source['run_id']:result}
    refreshed=refs.recognize_regions(document,source)['measurement_lines']['span']
    assert refreshed['points_u_xy']==manual['points_u_xy'] and refreshed['status']=='manual'
    refs.save_references(tmp_path/'project',document,actor='Reviewer',reason='Save measured spans')
    restored=refs.load_references(tmp_path/'project')
    assert restored['measurement_lines'][0]['length_mm']==200.
    assert restored['runs'][source['run_id']]['measurement_lines']['span']['points_u_xy']==manual['points_u_xy']
    assert not restored['regions']


def test_line_on_plain_surface_is_measurable_even_without_trackable_texture(tmp_path):
    source=packet(tmp_path,np.full((200,300,3),90,np.uint8),origin=(0,0))
    line=lines.capture_measurement_line(source,line_id='plain',name='Frame width',
        points_u_xy=[[30.,60.],[260.,60.]],length_mm=899.)
    assert line['endpoint_regions']==[None,None]
    observed=lines.measurement_for_photo(line,source)
    assert observed['length_u_px']==230. and observed['length_mm']==899.
    target=packet(tmp_path,np.full((200,300,3),90,np.uint8),run='target',origin=(0,0))
    document=refs.empty_document()
    document['measurement_lines']=[line]
    assert refs.recognize_regions(document,target)['measurement_lines']['plain']['status']=='missing'


def test_incomplete_or_changed_measurement_sources_do_not_supply_false_observations(tmp_path):
    source,line,_=source_line(tmp_path)
    other=packet(tmp_path,source['image_rgb'],run='other')
    assert lines.measurement_for_photo(line,other) is None
    corrupted=deepcopy(line)
    corrupted['points_u_xy'][0][0]+=1
    with pytest.raises(ValueError,match='binding'):
        lines.validate_measurement_lines([corrupted])
    with pytest.raises(ValueError,match='different'):
        lines.capture_measurement_line(source,line_id='bad',name='Zero',points_u_xy=[[0,40],[0,40]])
    with pytest.raises(ValueError,match='length'):
        lines.update_measurement_line(line,length_mm=-1.)
