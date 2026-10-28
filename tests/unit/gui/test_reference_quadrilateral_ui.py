"""Four visible reference points replace the separate measured-line editor."""
from copy import deepcopy
import math

import numpy as np
from PySide6.QtCore import Qt

from darsia.fluidflower import series_rig_frame_panel as panel_module
from darsia.fluidflower.rig_reference_quadrilateral import reference_frame_geometry
from darsia.fluidflower.series_geometry import RigFrame
from .test_rig_full_raw_editor import packet, loaded
from .test_painting_raw_reference_on_u import raw_view, raw_file, profile, renders, settings  # noqa: F401


PHOTO=dict(top_left=(-5.,-10.),top_right=(-4.,70.),bottom_right=(50.,68.),bottom_left=(48.,-8.))


def show_frame(view, frame):
    panel=view.rig_panel
    panel.frame=frame
    panel._loading=True
    panel._show_frame()
    panel._loading=False
    panel._draw_overlay()
    return panel


def test_reference_table_has_descriptions_combined_pixels_and_metric_origin(raw_view,qtbot,monkeypatch):
    view=raw_view
    view._rig_photo_loader.reader=lambda path,profile:packet(path)
    view.set_workflow_scope('rig_frame')
    qtbot.waitUntil(lambda:loaded(view))
    values=reference_frame_geometry(PHOTO,measurements_m=dict(width=.899,height=.49))
    values['reference_geometry'].update(estimated=True,uncertainty_mm=5.,point_descriptions={
        'top_left':'top_of_free_water','top_right':'top_of_free_water',
        'bottom_right':'rig_frame','bottom_left':'rig_frame'})
    panel=show_frame(view,RigFrame(**values,run_id='ac52',source='estimated reference quadrilateral'))
    assert panel.frame_status.text()==''
    assert panel.frame_status.isHidden()
    assert panel.width_m.value()==.899 and panel.height_m.value()==.49
    assert panel.reference_shape.currentData()=='rectangle'
    assert panel.corners.verticalHeaderItem(0).text().startswith('1')
    assert panel.static_references is None
    assert panel.corners.columnCount()==5
    assert panel.description_boxes['top_left'].currentText()=='Top of free water'
    assert panel.description_boxes['bottom_left'].currentText()=='Rig frame'
    assert panel.reference_photo_corners()==PHOTO
    assert [[panel.corners.item(row,col).text() for col in (2,3)] for row in range(4)]==[
        ['0.0','490.0'],['899.0','490.0'],['899.0','0.0'],['0.0','0.0']]
    assert not (panel.corners.item(0,2).flags() & Qt.ItemFlag.ItemIsEditable)
    stored=[]
    monkeypatch.setattr(panel_module,'save_rig_frame',lambda project,frame,**kw:stored.append(frame))
    monkeypatch.setattr(panel,'_store_points',lambda:None)
    combo=panel.description_boxes['bottom_right']
    combo.setCurrentIndex(combo.findData('top_of_free_water'))
    assert stored[-1].complete() and stored[-1].reference_geometry['estimated']
    panel=show_frame(view,RigFrame.from_dict(stored[-1].to_dict()))
    assert panel.description_boxes['bottom_right'].currentData()=='top_of_free_water'
    assert panel.reference_photo_corners()==PHOTO


def test_combined_pixel_edit_preserves_precision_and_rejects_invalid_pairs(raw_view,qtbot,monkeypatch):
    view=raw_view
    view._rig_photo_loader.reader=lambda path,profile:packet(path)
    view.set_workflow_scope('rig_frame')
    qtbot.waitUntil(lambda:loaded(view))
    photo=deepcopy(PHOTO)
    photo['top_left']=(-5.123456789,-10.987654321)
    panel=show_frame(view,RigFrame(**reference_frame_geometry(photo,measurements_m=dict(width=.899,height=.49)),run_id='ac52'))
    saved=[]
    monkeypatch.setattr(panel_module,'save_rig_frame',lambda project,frame,**kw:saved.append(frame))
    monkeypatch.setattr(panel,'_store_points',lambda:None)
    panel._store_frame()
    assert saved[-1].reference_geometry['photo_corners_rc']['top_left']==list(photo['top_left'])
    panel.corners.item(0,1).setText('-5.13; -10.987654321')
    assert saved[-1].reference_geometry['photo_corners_rc']['top_left']==[-5.13,-10.987654321]
    panel.corners.item(0,1).setText('-6,25; -12,5')
    assert panel.reference_photo_corners()['top_left']==(-6.25,-12.5)
    before=deepcopy(panel.frame.to_dict())
    saved.clear()
    panel.corners.item(0,1).setText('not a pair')
    assert 'row; column' in panel.reference_error.text()
    assert not saved and panel.frame.to_dict()==before


def test_general_quad_uses_measured_sides_and_keeps_observed_points_after_reload(raw_view,qtbot,monkeypatch):
    view=raw_view
    view._rig_photo_loader.reader=lambda path,profile:packet(path)
    view.set_workflow_scope('rig_frame')
    qtbot.waitUntil(lambda:loaded(view))
    values=reference_frame_geometry(PHOTO,measurements_m=dict(width=.8,height=.4))
    panel=show_frame(view,RigFrame(**values,run_id='ac52'))
    saved=[]
    monkeypatch.setattr(panel_module,'save_rig_frame',lambda project,frame,**kw:saved.append(frame))
    monkeypatch.setattr(panel,'_store_points',lambda:None)
    panel.reference_shape.setCurrentIndex(panel.reference_shape.findData('quadrilateral'))
    # A known trapezoid: D(0,0), C(.8,0), A(.1,.4), B(.7,.4).
    lengths=dict(top=.6,right=math.hypot(.1,.4),bottom=.8,left=math.hypot(.1,.4),
                 diagonal_ac=math.hypot(.7,.4))
    for key,value in lengths.items():panel.side_lengths[key].setValue(value)
    panel._size_edited()
    assert not panel.reference_error.text(),panel.reference_error.text()
    frame=saved[-1]
    assert frame.reference_geometry['shape']=='quadrilateral'
    assert frame.reference_geometry['photo_corners_rc']=={key:list(value) for key,value in PHOTO.items()}
    assert any(not np.allclose(frame.corners[key],PHOTO[key]) for key in PHOTO)
    assert frame.complete()
    for key,point in PHOTO.items():
        np.testing.assert_allclose(frame.pixel_to_rig(*point),frame.reference_geometry['metric_corners_xy'][key],atol=1e-8)
    panel=show_frame(view,RigFrame.from_dict(frame.to_dict()))
    assert panel.reference_shape.currentData()=='quadrilateral'
    for row,key in enumerate(panel_module.CLICK_ORDER):
        np.testing.assert_allclose(panel.reference_photo_corners()[key],PHOTO[key])


def test_invalid_side_measurements_do_not_overwrite_a_valid_reference(raw_view,qtbot,monkeypatch):
    view=raw_view
    view._rig_photo_loader.reader=lambda path,profile:packet(path)
    view.set_workflow_scope('rig_frame')
    qtbot.waitUntil(lambda:loaded(view))
    values=reference_frame_geometry(PHOTO,measurements_m=dict(width=.8,height=.4))
    panel=show_frame(view,RigFrame(**values,run_id='ac52'))
    saved=[]
    monkeypatch.setattr(panel_module,'save_rig_frame',lambda project,frame,**kw:saved.append(frame))
    monkeypatch.setattr(panel,'_store_points',lambda:None)
    panel.reference_shape.setCurrentIndex(panel.reference_shape.findData('quadrilateral'))
    before=deepcopy(panel.frame.to_dict())
    saved.clear()
    panel.side_lengths['diagonal_ac'].setValue(5.)
    panel._size_edited()
    assert 'triangle' in panel.reference_error.text()
    assert not saved and panel.frame.to_dict()==before
    assert not panel.copy_crop.isEnabled()
    panel._draw_overlay()  # Invalid input stays editable and cannot crash the photo.
