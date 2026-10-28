"""Unseen frame corners stay unobserved without losing physical dimensions."""
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPolygonItem

from darsia.fluidflower import series_geometry as geometry
from darsia.fluidflower import series_rig_frame_panel as rig_panel
from darsia.fluidflower import series_layer_view as layer_view
from darsia.fluidflower.series_geometry import RigFrame, FRAME_FROM_STATIC_REFERENCES
from .test_rig_full_raw_editor import loaded, packet, click_u
from .test_painting_raw_reference_on_u import (  # noqa: F401
    raw_view, raw_file, profile, renders, settings,
)


def frame(run='ac52'):
    return RigFrame({'top_left':(-5.,-10.), 'top_right':(-5.,70.),
        'bottom_right':(50.,70.), 'bottom_left':(50.,-10.)},
        width_m=.899, height_m=.49, run_id=run)


def test_visibility_roundtrip_preserves_old_shape_and_requires_measured_geometry():
    original = frame()
    assert original.complete() and 'disabled_corners' not in original.to_dict()
    assert RigFrame.from_dict(original.to_dict()) == original
    hidden = replace(original, disabled_corners=('top_left','top_right'))
    restored = RigFrame.from_dict(hidden.to_dict())
    assert restored == hidden and restored.has_size() and not restored.complete()
    with pytest.raises(ValueError, match='Unobserved'):
        restored.native_crop()
    derived = replace(restored, source=FRAME_FROM_STATIC_REFERENCES)
    assert derived.complete() and derived.native_crop()['width'] == .899
    assert np.allclose(derived.pixel_to_rig(50.,-10.), [0.,0.])
    invalid = hidden.to_dict()
    invalid['disabled_corners'] = ['invented_corner']
    with pytest.raises(ValueError, match='distinct rig corners'):
        RigFrame.from_dict(invalid)


def test_marking_skips_hidden_corners_and_visibility_persists(raw_view, qtbot, monkeypatch):
    view = raw_view
    view._rig_photo_loader.reader = lambda path, profile: packet(path)
    view.set_workflow_scope('rig_frame')
    qtbot.waitUntil(lambda:loaded(view))
    panel = view.rig_panel
    saved=[]
    monkeypatch.setattr(rig_panel,'save_rig_frame',lambda project,value,**kw:saved.append(value))
    monkeypatch.setattr(panel,'_store_points',lambda:None)
    panel.frame=frame()
    panel._loading=True
    panel._show_frame()
    panel._loading=False
    for row in (0,1):
        panel.corners.item(row,4).setCheckState(Qt.CheckState.Unchecked)
    assert saved[-1].disabled_corners == ('top_left','top_right')
    assert not saved[-1].complete() and panel.width_m.value()==.899
    assert 'Not visible: Top left, Top right' in panel.frame_status.text()
    assert view.step_frame.status.text().startswith('2/4 visible corners')
    assert not panel.copy_crop.isEnabled()
    assert not (panel.corners.item(0,1).flags() & Qt.ItemFlag.ItemIsEditable)
    panel._draw_overlay()
    assert sum(isinstance(x,QGraphicsEllipseItem) and x.zValue()==21 for x in panel._overlay)==2
    assert sum(isinstance(x,QGraphicsLineItem) and x.zValue()==20 for x in panel._overlay)==1
    assert not any(isinstance(x,QGraphicsPolygonItem) for x in panel._overlay)
    panel.mark_corners.setChecked(True)
    assert panel._corner_rows==[2,3] and 'Bottom right' in panel.status.text()
    click_u(qtbot,view,48.,68.)
    assert 'Bottom left' in panel.status.text()
    click_u(qtbot,view,48.,-8.)
    assert panel._mode=='' and not panel.mark_corners.isChecked()
    assert np.allclose(saved[-1].corners['bottom_right'],[48.,68.],atol=.3)
    assert saved[-1].corners['top_left']==(-5.,-10.)
    panel.frame=RigFrame.from_dict(saved[-1].to_dict())
    panel._loading=True
    panel._show_frame()
    panel._loading=False
    assert panel.corners.item(0,4).checkState()==Qt.CheckState.Unchecked
    for row in (0,1):
        panel.corners.item(row,4).setCheckState(Qt.CheckState.Checked)
    assert saved[-1].complete() and not saved[-1].disabled_corners
    assert panel.corners.item(0,1).flags() & Qt.ItemFlag.ItemIsEditable


def test_measured_frame_keeps_visibility_and_model_on_checkbox_change(raw_view, qtbot, monkeypatch):
    view=raw_view
    view._rig_photo_loader.reader=lambda path,profile:packet(path)
    view.set_workflow_scope('rig_frame')
    qtbot.waitUntil(lambda:loaded(view))
    prior=replace(frame(),disabled_corners=('top_left','top_right'))
    monkeypatch.setattr(layer_view,'saved_rig_frames',lambda project:{prior.run_id:prior})
    saved=[]
    monkeypatch.setattr(geometry,'save_rig_frame',lambda project,value,**kw:saved.append(value))
    monkeypatch.setattr(rig_panel,'save_rig_frame',lambda project,value,**kw:saved.append(value))
    monkeypatch.setattr(view.rig_panel,'_store_points',lambda:None)
    points=[prior.corners[name] for name in rig_panel.CLICK_ORDER]
    assert view.use_static_reference_frame(prior.run_id,points,.899,.49)
    assert saved[-1].complete() and saved[-1].disabled_corners==prior.disabled_corners
    assert view.rig_panel.corners.item(0,4).checkState()==Qt.CheckState.Unchecked
    view.rig_panel.corners.item(0,4).setCheckState(Qt.CheckState.Checked)
    assert saved[-1].source==FRAME_FROM_STATIC_REFERENCES and saved[-1].complete()
    assert saved[-1].disabled_corners==('top_right',)
