"""Static references use the visible full field and original run coordinates."""
import hashlib
import numpy as np
from PIL import Image

from darsia.fluidflower.rig_photo_loader import read_static_reference_packet
from darsia.fluidflower.series_geometry import SeriesRunInputs
from .test_rig_full_raw_editor import loaded, packet, click_u
from .test_painting_raw_reference_on_u import (  # noqa: F401
    raw_view, raw_file, profile, renders, settings,
)


def test_static_areas_receive_full_packet_and_negative_u_clicks(raw_view, qtbot):
    view = raw_view
    view._rig_photo_loader.reader = lambda path, profile: packet(path)
    view.set_workflow_scope('rig_frame')
    qtbot.waitUntil(lambda: loaded(view))
    run, ready = view.static_reference_context()
    assert ready and run == view.reference().run_id
    assert view.canvas.preview['image_rgb'].shape == (95, 120, 3)
    assert view.canvas.preview['run_id'] == run
    selected = []
    view.canvas.raw_point_selected.connect(lambda row, col: selected.append([row, col]))
    view.canvas.place_mode = True
    click_u(qtbot, view, -5, -10)
    assert np.allclose(selected, [[-5, -10]], atol=.3)
    assert view.static_reference_runs() == [view.reference()]
    view.set_workflow_scope('layers')
    assert view.static_reference_context() == (run, False)


def test_registered_image_reader_keeps_all_pixels_and_source_identity(tmp_path):
    rgb = np.arange(20*30*3, dtype=np.uint8).reshape(20,30,3)
    photo = tmp_path/'reference.png'
    Image.fromarray(rgb).save(photo)
    reference = SeriesRunInputs('run7', tmp_path/'run.toml', baseline_image=photo)
    result = read_static_reference_packet(reference, None)
    assert result['run_id'] == 'run7'
    assert np.array_equal(result['image_rgb'], rgb)
    assert result['source_raw']['sha256'] == hashlib.sha256(photo.read_bytes()).hexdigest()
    assert result['valid_support'].all()
    assert result['extent_u_xywh'] == [0,0,30,20]
    assert result['camera_u_wh'] == [30,20]


def test_static_area_run_preview_keeps_map_reference_and_saves_target_frame(raw_view, qtbot, tmp_path, monkeypatch):
    from dataclasses import replace
    from darsia.fluidflower import series_geometry
    view = raw_view
    source = view.reference()
    target_photo = tmp_path/'target.ARW'
    target_photo.write_bytes(b'private target photograph')
    target = replace(source, run_id='ac53', baseline_image=target_photo)
    view._static_reference_runs = lambda:[source,target]
    view._rig_photo_loader.reader = lambda path, profile:packet(path)
    view.set_workflow_scope('rig_frame')
    qtbot.waitUntil(lambda:loaded(view))
    before = view.session.to_dict()
    assert view.show_static_reference_run('ac53')
    qtbot.waitUntil(lambda:view.static_reference_context()==('ac53',True))
    assert view.reference() == source
    assert view.session.to_dict() == before
    assert not view.rig_panel.corners.parentWidget().isEnabled()
    assert not view.rig_panel.points.parentWidget().isEnabled()
    assert view.rig_panel.static_references is None
    assert view.canvas.preview['source_raw']['path']==str(target_photo)
    saved=[]
    monkeypatch.setattr(series_geometry,'save_rig_frame',lambda project,frame,**kw:saved.append(frame))
    corners=[[-5.,-10.],[-5.,70.],[50.,70.],[50.,-10.]]
    assert view.use_static_reference_frame('ac53',corners,.899,.49)
    assert len(saved)==1 and saved[0].run_id=='ac53'
    assert saved[0].image==str(target_photo)
    assert saved[0].source=='computed from static reference areas'
    assert not view.use_static_reference_frame(source.run_id,corners,.899,.49)
    # A display adjustment must retain the selected target and correct profile.
    view._apply_rig_photo(view._rig_photo_packet,target_photo,reset_view=False)
    assert view.static_reference_context()==('ac53',True)
    assert view.show_static_reference_run(source.run_id)
    qtbot.waitUntil(lambda:view.static_reference_context()==(source.run_id,True))
    assert view.rig_panel.corners.parentWidget().isEnabled()
    assert view.reference()==source and view.session.to_dict()==before
