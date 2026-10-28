"""Guides cover the photograph, retain camera-U coordinates and stay photo-bound."""
import numpy as np
import pytest

from darsia.fluidflower.rig_guide_lines import make_guide, guide_segment
from darsia.fluidflower.rig_static_references import empty_document, save_references, load_references


def packet():
    return dict(run_id='run1', image_rgb=np.zeros((100,160,3),np.uint8),
        camera_u_to_photo_h=[[1.,0.,20.],[0.,1.,30.],[0.,0.,1.]])


@pytest.mark.parametrize('mode,expected',[
    ('horizontal',[[-20.,10.],[139.,10.]]),
    ('vertical',[[10.,-30.],[10.,69.]]),
    ('through',[[-20.,-5.],[128.,69.]])])
def test_full_guides_clip_to_original_photo_with_expanded_u_margin(mode,expected):
    photo=packet()
    guide=make_guide(photo,guide_id='g1',points_u_xy=[[10.,10.],[50.,30.]],
        mode=mode,extend=True,photo_identity='original')
    assert np.allclose(guide_segment(guide,photo,photo_identity='original'),expected)


def test_segment_keeps_clicked_endpoints_and_hides_on_other_photo_or_run():
    photo=packet()
    guide=make_guide(photo,guide_id='g1',points_u_xy=[[-10.,-10.],[50.,30.]],
        extend=False,photo_identity='original')
    assert np.allclose(guide_segment(guide,photo,photo_identity='original'),guide['points_u_xy'])
    assert guide_segment(guide,dict(photo,run_id='run2'),photo_identity='original') is None
    assert guide_segment(guide,photo,photo_identity='changed') is None


def test_camera_mapping_fallback_is_independent_of_viewport():
    from types import SimpleNamespace
    photo=packet()
    del photo['camera_u_to_photo_h']
    photo['mapping']=SimpleNamespace(camera_u_to_display_xy=lambda points:np.asarray(points)+[20.,30.])
    guide=make_guide(photo,guide_id='g1',points_u_xy=[[10.,10.],[50.,30.]],photo_identity='original')
    assert np.allclose(guide_segment(guide,photo,photo_identity='original'),[[-20.,-5.],[128.,69.]])


def test_guides_save_and_reload_without_measurement_or_model_changes(tmp_path):
    document=empty_document()
    document['guides']=[make_guide(packet(),guide_id='g1',points_u_xy=[[10.,10.],[50.,30.]],photo_identity='original')]
    document['guides_visible']=False
    saved=save_references(tmp_path,document,actor='Reviewer',reason='Save support lines')
    reloaded=load_references(tmp_path)
    assert reloaded['guides']==saved['guides']==document['guides']
    assert not reloaded['guides_visible']
    assert reloaded['regions']==document['regions']==[]
    assert reloaded['runs']==document['runs']=={}


@pytest.mark.parametrize('mode,points',[
    ('through',[[1.,1.],[1.,1.]]),
    ('horizontal',[[1.,1.],[1.,2.]]),
    ('vertical',[[1.,1.],[2.,1.]])])
def test_coincident_or_axis_locked_degenerate_clicks_are_rejected(mode,points):
    with pytest.raises(ValueError,match='different'):
        make_guide(packet(),guide_id='g1',points_u_xy=points,mode=mode,photo_identity='original')
