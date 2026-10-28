"""Saved measured spans can locate unseen rig corners without drawing new areas."""
from copy import deepcopy

import numpy as np
import pytest

from darsia.fluidflower import rig_static_references as refs
from darsia.fluidflower import rig_measurement_lines as lines
from .test_rig_measurement_line_controls import measurements, draw  # noqa: F401
from .test_rig_static_reference_ui import context  # noqa: F401


H = np.array([[1.1, .08, 40.], [.04, -1., 450.], [.0001, -.00008, 1.]])


def setup_lines(measurements):
    panel = measurements.panel
    physical = [np.array([[0.,60.],[400.,60.]]), np.array([[0.,180.],[400.,180.]])]
    records=[]
    for index, points in enumerate(physical):
        record=draw(measurements,name=f'Width {index+1}',length=400.,
                    points=refs._project(H,points))
        records.append(record)
    measurements.rig.width_m.setValue(.4)
    measurements.rig.height_m.setValue(.3)
    return records, physical


def place(panel, row, start, direction='x+'):
    panel.measurement_table.selectRow(row)
    panel.placement_x_mm.setValue(start[0])
    panel.placement_y_mm.setValue(start[1])
    panel.placement_direction.setCurrentIndex(panel.placement_direction.findData(direction))
    panel.apply_placement_button.click()


def test_existing_lines_fit_unseen_corners_after_placement_without_redrawing(measurements):
    panel, page = measurements.panel, measurements.page
    records, physical=setup_lines(measurements)
    before=[deepcopy(record) for record in records]
    assert not panel.use_button.isEnabled()
    place(panel,0,physical[0][0])
    assert not panel.use_button.isEnabled()
    place(panel,1,physical[1][0])
    assert panel.use_button.isEnabled(),panel.model_note.text()
    assert '2 lines' in panel.model_note.text()
    assert not panel.point_area_toggle.isChecked()
    assert panel.use_button.parentWidget() is panel.frame_model_box
    for record, old in zip(records,before):
        assert record['points_u_xy']==old['points_u_xy']
        assert record['tracking_identity']==old['tracking_identity']
        assert record['endpoint_regions']==old['endpoint_regions']
    assert sorted(item.text() for item in panel._items
                  if item.data(0)=='measurement-endpoint-number')==['1','2']
    calls=[]
    page.use_static_reference_frame=lambda run,corners,width,height:calls.append((run,corners,width,height)) or True
    panel.origin_x_mm.setValue(20.)
    panel.origin_y_mm.setValue(30.)
    panel.use_button.click()
    assert len(calls)==1 and calls[0][0]=='source' and calls[0][2:]==(.4,.3)
    metric_corners=[[20.,330.],[420.,330.],[420.,30.],[20.,30.]]
    np.testing.assert_allclose(calls[0][1],refs._project(H,metric_corners)[:,::-1],atol=2e-4)


def test_placement_and_origin_survive_save_reload_and_show_save_state(measurements,qtbot):
    panel=measurements.panel
    records, physical=setup_lines(measurements)
    for row, points in enumerate(physical):place(panel,row,points[0])
    panel.origin_x_mm.setValue(-12.5)
    panel.origin_y_mm.setValue(7.)
    assert panel.save_state.text().startswith('Unsaved references')
    assert panel.save()
    qtbot.waitUntil(lambda:not panel.busy(),timeout=5000)
    assert not panel.save_button.isEnabled() and panel.save_state.text()=='All references saved.'
    saved=refs.load_references(measurements.root)
    assert saved['rig_origin_xy_mm']==[-12.5,7.]
    assert saved['measurement_lines'][1]['placement']==dict(start_xy_mm=[0.,180.],direction='x+')
    panel.origin_x_mm.blockSignals(True)
    panel.origin_x_mm.setValue(123.)
    panel.origin_x_mm.blockSignals(False)
    assert panel.load()
    qtbot.waitUntil(lambda:not panel.busy(),timeout=5000)
    assert panel.origin_x_mm.value()==-12.5 and panel.origin_y_mm.value()==7.
    assert panel.use_button.isEnabled(),panel.model_note.text()
    assert panel.document['measurement_lines'][0]['tracking_identity']==records[0]['tracking_identity']


def test_target_frame_uses_target_endpoints_and_clear_placement_retains_distance(measurements):
    panel,page=measurements.panel,measurements.page
    records,physical=setup_lines(measurements)
    for row, points in enumerate(physical):place(panel,row,points[0])
    original=deepcopy(panel.document['runs']['source'])
    target_h=np.array([[1.,.03,85.],[.01,-.92,410.],[.00008,-.00003,1.]])
    page.current_run,page.canvas.preview='target',measurements.target
    panel.document['runs']['target']=dict(photo=refs.photo_binding(measurements.target),observations={},
        measurement_lines={record['id']:lines.manual_measurement_observation(record,measurements.target,
            refs._project(target_h,points)) for record,points in zip(records,physical)})
    panel.context_changed()
    assert panel.use_button.isEnabled(),panel.model_note.text()
    calls=[]
    page.use_static_reference_frame=lambda run,corners,width,height:calls.append((run,corners)) or True
    assert panel.use_for_rig_frame()
    assert calls[0][0]=='target'
    np.testing.assert_allclose(calls[0][1],refs._project(target_h,[[0,300],[400,300],[400,0],[0,0]])[:,::-1],atol=2e-4)
    assert panel.document['runs']['source']==original
    panel.measurement_table.selectRow(0)
    panel.clear_placement_button.click()
    assert 'placement' not in records[0] or records[0]['placement'] is None
    assert records[0]['length_mm']==400. and len(records[0]['points_u_xy'])==2
    assert not panel.use_button.isEnabled()
    assert panel.document['runs']['source']==original


def test_zero_length_or_missing_direction_never_claims_positioned_geometry(measurements):
    panel=measurements.panel
    records,physical=setup_lines(measurements)
    panel.measurement_table.selectRow(0)
    panel.apply_placement_button.click()
    assert 'Choose the physical direction' in panel.note.text()
    assert not records[0].get('placement')
    place(panel,0,physical[0][0])
    place(panel,1,physical[1][0])
    panel.measurement_table.item(1,1).setText('0')
    assert not panel.use_button.isEnabled()
    assert 'needs a positive physical length' in panel.placement_note.text()


def test_extrapolated_frame_cannot_cross_the_model_horizon(measurements,monkeypatch):
    panel,page=measurements.panel,measurements.page
    records,physical=setup_lines(measurements)
    for row,points in enumerate(physical):place(panel,row,points[0])
    model=panel._fit(panel._observations())
    model['H_metric_mm_to_camera_u']=[[1.,0.,40.],[0.,-1.,100.],[0.,-.004,1.]]
    monkeypatch.setattr(panel,'_fit',lambda observations:model)
    calls=[]
    page.use_static_reference_frame=lambda *args:calls.append(args) or True
    assert not panel.use_for_rig_frame()
    assert not calls and 'horizon' in panel.note.text()
