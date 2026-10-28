"""Display-only water review with synthetic photos; no project or actual approval."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsPathItem

from darsia.fluidflower import painted_water_surface as surface
from darsia.fluidflower.painting_water_surface_ui import WaterSurfaceControls

from .test_painting_gap_review_ui import _Backend, _dialog


def _basis(*, bolts=True, metric=True):
    return dict(format='darsia-water-surface-basis-v1',run_id='ac18',
        photo=dict(path='C:/synthetic/clearwater.ARW',sha256='a'*64,size_bytes=99),photo_shape=[32,40],
        photo_to_metric_h=[[1.,0.,-2.],[0.,-1.,28.],[0.,0.,1.]] if metric else None,
        own_geometry_origin_metric_mm=[0.,0.],own_geometry_up_metric_xy=[0.,1.],
        left_bolts_photo_xy=[[2.,28.],[2.,18.],[2.,8.]] if bolts else None,
        bolt_heights_mm=[0,195,390],bottom_edge_photo_xy=[[2.,28.],[37.,28.]],
        own_top_photo_xy=[[2.,2.],[37.,2.]],transported_top_photo_xy=[[2.,3.],[37.,3.]],
        port_photo_xy=None,provenance=dict(meaning='synthetic-existing-own-geometry-only'))


def _packet(*, bolts=True, metric=True, line=True):
    return dict(basis=_basis(bolts=bolts,metric=metric),proposal=dict(
        line_photo_xy=[[0.,12.],[39.,12.]] if line else None,
        diagnostics=dict(method='synthetic-unconfirmed-proposal',human_confirmation_required=True)),
        action=None,measurements=None,display_grid={'synthetic':'same-explicit-plane'},
        photo_to_display_h=[[1.,0.,3.],[0.,1.,5.],[0.,0.,1.]])


def _api():
    calls=[]
    def make(*args,**options):
        calls.append(('confirm',deepcopy(args),deepcopy(options)))
        return surface.make_water_surface_action(*args,**options)
    def move(*args,**options):
        calls.append(('move',deepcopy(args),deepcopy(options)))
        return surface.move_water_surface_to_photo_point(*args,**options)
    return SimpleNamespace(make_water_surface_action=make,move_water_surface_to_photo_point=move,
        water_surface_measurements=surface.water_surface_measurements),calls


def _controls(qtbot,**options):
    api,calls=_api()
    widget=WaterSurfaceControls(actor='Synthetic human',api=api)
    qtbot.addWidget(widget)
    widget.show()
    packet=_packet(**options)
    widget.set_packet(packet,display_grid=packet['display_grid'],active=True)
    return widget,packet,calls


def test_viewing_proposal_and_display_controls_never_confirm_or_change_input(qtbot):
    widget,packet,calls=_controls(qtbot)
    before=deepcopy(packet)
    assert widget.action is None and widget.measurements is None and calls==[]
    assert 'unconfirmed' in widget.status.text()
    assert widget.height_tolerance.value()==3. and widget.angle_tolerance.value()==.5
    widget.outline.setChecked(False);widget.outline.setChecked(True)
    widget.set_active(False);assert not widget.confirm.isEnabled()
    widget.set_active(True)
    assert calls==[] and packet==before and widget.action is None


def test_only_human_confirm_retains_exact_action_and_separates_own_geometry_from_observed_ruler(qtbot):
    widget,packet,calls=_controls(qtbot,bolts=False)
    records=[];widget.confirmed.connect(records.append)
    assert widget.confirm.isEnabled()
    widget.confirm.click()
    assert [row[0] for row in calls]==['confirm'] and len(records)==1
    assert records[0]['action']['actor']=='Synthetic human'
    assert records[0]['measurements']['height_basis']=='existing-own-geometry'
    assert records[0]['measurements']['height_mm']==16.
    assert records[0]['measurements']['bolt_height_mm'] is None
    assert 'own geometry (inner left-wall and bottom intersection): 16.000 mm' in widget.readout.text()
    assert 'observed bottom-left bolt centre (0/195/390 mm)' in widget.readout.text() and 'unavailable' in widget.readout.text()
    records[0]['action']['line_photo_xy'][0][1]=999.
    assert widget.action['line_photo_xy'][0][1]==12.


def test_no_metric_binding_keeps_physical_values_unavailable_but_two_human_endpoints_are_possible(qtbot):
    widget,packet,calls=_controls(qtbot,metric=False,line=False)
    assert not widget.move.isEnabled() and not widget.confirm.isEnabled() and widget.manual.isEnabled()
    picked=[];widget.pick_requested.connect(picked.append)
    widget.manual.click()
    widget.select_photo_point([1.,10.])
    assert widget.action is None and not widget.confirm.isEnabled()
    widget.select_photo_point([38.,13.])
    assert widget.line_photo_xy==[[1.,10.],[38.,13.]] and widget.confirm.isEnabled()
    assert calls==[] and picked[-1] is False
    widget.confirm.click()
    assert widget.measurements['height_mm'] is None and widget.measurements['angle_degrees'] is None
    assert 'unavailable' in widget.readout.text()


def test_old_confirmation_requires_exact_current_basis_and_manual_changes_need_fresh_confirmation(qtbot):
    widget,packet,calls=_controls(qtbot)
    widget.confirm.click()
    old=deepcopy(widget.action)
    changed=deepcopy(packet);changed['basis']['photo']['sha256']='b'*64;changed['action']=old
    widget.set_packet(changed,display_grid=changed['display_grid'],active=True)
    assert widget.action is None and widget.measurements is None
    assert 'The previous confirmation' in widget.status.text() and widget.confirm.isEnabled()
    widget.manual.click();widget.select_photo_point([2.,11.]);widget.select_photo_point([37.,14.])
    assert widget.action is None
    widget.confirm.click()
    assert widget.action['basis_sha256']!=old['basis_sha256']
    assert widget.action['line_photo_xy']==[[2.,11.],[37.,14.]]


@pytest.mark.parametrize('problem',['transform','plane','outside'])
def test_missing_display_binding_and_outside_click_never_guess_or_clamp(qtbot,problem):
    widget,packet,calls=_controls(qtbot)
    if problem=='outside':
        widget.manual.click();widget.select_photo_point([-1.,12.])
        assert widget.line_photo_xy==packet['proposal']['line_photo_xy']
        assert widget._manual_points==[] and 'coordinates are not clipped' in widget.status.text()
    else:
        changed=deepcopy(packet)
        if problem=='transform':
            changed.pop('photo_to_display_h')
        else:
            changed['display_grid']={'synthetic':'another-plane'}
        widget.set_packet(changed,display_grid=packet['display_grid'],active=True)
        assert widget.packet is None and not widget.confirm.isEnabled()
    assert widget.action is None and calls==[]


def test_bound_display_overlay_and_one_click_use_exact_inverse_photo_transform(qtbot):
    backend=_Backend()
    base=backend.packets['ac18']
    water=_packet()
    base.update(display_grid=water['display_grid'],image_choices=[dict(image_id='clearwater',available=True)])
    def comparison(review,run,component,image):
        assert run=='ac18' and image=='clearwater'
        return dict(image_rgb=base['image_rgb'].copy(),image_encoding=base['image_encoding'],
            native_to_photo_h=base['native_to_photo_h'],display_grid=base['display_grid'],water_surface=water)
    api,calls=_api()
    widget,_=_dialog(qtbot,backend=backend,comparison_reader=comparison,water_surface_api=api)
    widget.images.setCurrentIndex(widget.images.findData('clearwater'))
    qtbot.waitUntil(lambda:not widget.busy(),timeout=5000)
    assert widget.water_surface.confirm.isEnabled() and calls==[]
    blue=[item for item in widget._overlays if isinstance(item,QGraphicsPathItem) and item.pen().color().name()=='#00bfff']
    assert blue and blue[0].path().boundingRect().top()==17.  # photo y12 + explicit display shift5
    assert blue[0].brush().style()==Qt.BrushStyle.NoBrush
    widget.water_surface.move.click()
    assert widget.photo.pick_single_point
    widget.photo.point_clicked.emit((15.,16.))  # exact display inverse -> source photo (12,11)
    assert calls[0][0]=='move' and calls[0][1][1]==[12.,11.]
    assert widget.water_surface.action is None and not widget.photo.pick_single_point
    np.testing.assert_array_equal(base['image_rgb'],np.full((64,64,3),30,np.uint8))
    widget.water_surface.confirm.click()
    assert widget.water_surface_confirmations['ac18']['action']['line_photo_xy']==[[0.,11.],[39.,11.]]
