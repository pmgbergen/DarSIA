"""Layers, saved versions and other-run previews retain the complete RAW field."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
from PySide6.QtCore import QPointF

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import painting_run_preview_ui as preview_ui
from darsia.fluidflower.painting_run_preview import RunPreview
import pytest
from .test_rig_full_raw_editor import packet, loaded
from .test_painting_manual_step4_route import _saved_manual
from .test_painting_raw_reference_on_u import raw_view, raw_file, profile, renders, settings  # noqa: F401


def test_layers_and_reference_points_share_full_photo_and_exact_polygon_grid(raw_view,qtbot):
    view=raw_view
    calls=[]
    def read(path,profile):
        calls.append(path)
        return packet(path)
    view._rig_photo_loader.reader=read
    polygon=lp.Polygon(1,[(-5.123,-8.456),(20.,30.),(40.,-8.)])
    view.add_polygon(polygon)
    before=deepcopy(view.session.to_dict())
    view.set_workflow_scope('layers')
    qtbot.waitUntil(lambda:loaded(view))
    assert view.canvas.pixmap_item.pixmap().size().toTuple()==(120,95)
    assert view.canvas.paint_enabled and view.canvas._marks_visible
    assert view.canvas.preview is view._reference_token
    assert view.analysis_boxes._canvas_problem() == ''
    assert view.canvas.raw_size==view.image.image_size==(80,60)
    path=view.canvas._polygon_items[0].path()
    expected=view.canvas.raw_to_scene(*polygon.points[0])
    np.testing.assert_allclose((path.elementAt(0).x,path.elementAt(0).y),(expected.x(),expected.y()))
    np.testing.assert_allclose(view.canvas.scene_to_raw(expected),polygon.points[0])
    view.set_workflow_scope('rig_frame')
    assert loaded(view) and not view.canvas.paint_enabled
    view.set_workflow_scope('layers')
    assert loaded(view) and view.canvas.paint_enabled
    assert len(calls)==1
    assert view.session.to_dict()==before


def test_saved_manual_version_uses_full_reference_without_changing_archive(raw_view,qtbot):
    view=raw_view
    view._rig_photo_loader.reader=lambda path,profile:packet(path)
    record=_saved_manual(view,qtbot,'Full-photo saved map')
    qtbot.waitUntil(lambda:not view.run_preview.busy())
    before={path.name:path.read_bytes() for path in Path(record['folder']).iterdir()}
    view.set_workflow_scope('layers')
    qtbot.waitUntil(lambda:loaded(view) and view.run_preview.canvas.photo_display_mapping is not None)
    preview=view.run_preview
    assert preview.stack.currentWidget() is preview.canvas
    assert preview.canvas.pixmap_item.pixmap().size().toTuple()==(120,95)
    assert preview.canvas.raw_size==(80,60)
    shown=preview.canvas.proposal_item.mapToScene(QPointF(.5,.5))
    expected=preview.canvas.raw_to_scene(0.,0.)
    np.testing.assert_allclose((shown.x(),shown.y()),(expected.x(),expected.y()))
    assert {path.name:path.read_bytes() for path in Path(record['folder']).iterdir()}==before
    assert preview._reference_version()['folder']==record['folder']


def test_other_run_full_photo_repositions_reduced_labels_without_changing_gaps(raw_view,raw_file):
    view=raw_view
    full=packet(raw_file)
    result=RunPreview('ac53',np.zeros((30,40,3),np.uint8),np.ones((30,40),np.uint8),
        np.zeros((30,40),bool),str(raw_file),'ac52','Current map',native_shape_hw=(60,80),full_photo=full)
    before=result.layers.copy(),result.uncovered.copy()
    view.run_preview._render_result(result,view.session)
    canvas=view.run_preview.canvas
    assert canvas.pixmap_item.pixmap().size().toTuple()==(120,95)
    expected=canvas.raw_to_scene(.5,.5)
    shown=canvas.proposal_item.mapToScene(QPointF(.5,.5))
    np.testing.assert_allclose((shown.x(),shown.y()),(expected.x(),expected.y()))
    np.testing.assert_array_equal(result.layers,before[0])
    np.testing.assert_array_equal(result.uncovered,before[1])


def test_full_photo_failure_leaves_no_cropped_working_photo(raw_view,qtbot):
    view=raw_view
    def failed(path,profile):raise ValueError('full field unavailable')
    view._rig_photo_loader.reader=failed
    view.set_workflow_scope('layers')
    qtbot.waitUntil(lambda:'full field unavailable' in view.photo_note.text())
    assert view.canvas.preview is None and not view.canvas.paint_enabled


def test_saved_version_waits_for_full_photo_without_displaying_cropped_fallback(raw_view,qtbot):
    view=raw_view
    record=_saved_manual(view,qtbot,'Waiting for full photo')
    qtbot.waitUntil(lambda:not view.run_preview.busy())
    assert view.run_preview.canvas.preview is not None
    hold,started=Event(),Event()
    def read(path,profile):
        started.set()
        hold.wait(5)
        return packet(path)
    view._rig_photo_loader.reader=read
    try:
        view.set_workflow_scope('layers')
        qtbot.waitUntil(started.is_set)
        assert view.run_preview.canvas.preview is None
        assert view.run_preview.canvas.proposal_item.pixmap().isNull()
        view.run_preview.redraw_display()
        assert view.run_preview.canvas.proposal_item.pixmap().isNull()
        hold.set()
        qtbot.waitUntil(lambda:view.run_preview.canvas.photo_display_mapping is not None)
        assert view.run_preview._reference_version()['folder']==record['folder']
    finally:
        hold.set()
        qtbot.waitUntil(lambda:not view._rig_photo_loader.busy())


@pytest.mark.parametrize('wrong_grid',[False,True])
def test_other_run_worker_reads_full_photo_once_and_preserves_the_native_grid(raw_view,raw_file,profile,monkeypatch,wrong_grid):
    view=raw_view
    values=np.ones((60,80),np.uint8)
    reduced=RunPreview('ac53',np.zeros((30,40,3),np.uint8),values[::2,::2],
        np.zeros((30,40),bool),str(raw_file),'ac52','Saved map',native_shape_hw=(60,80))
    monkeypatch.setattr(preview_ui.preview,'preview_layers',lambda session,*args,**kwargs:(session,values))
    monkeypatch.setattr(preview_ui.frame_domain,'attach_frame_domain',lambda result,*args:result)
    def prepare(*args,**kwargs):
        kwargs['on_preview'](reduced)
        return reduced
    monkeypatch.setattr(preview_ui.preview,'prepare_preview',prepare)
    calls=[]
    def read(path,bound_profile):
        calls.append((path,bound_profile))
        result=packet(path)
        if wrong_grid:result['camera_u_wh']=(81,60)
        return result
    worker=preview_ui._PreviewWorker(view.project,deepcopy(view.session),None,[],None,
        dict(source='map',record={'kind':'manual'},name='Saved map',run='ac53',index='unused',config={},
            photo_loader=lambda *args:None,full_photo_profile=profile,full_photo_reader=read))
    worker.run()
    assert len(calls)==1 and calls[0]==(raw_file,profile)
    if wrong_grid:
        assert 'different camera grids' in worker.error and worker.result is None
    else:
        assert not worker.error
        result,session=worker.result
        assert result.full_photo is not None and result.photo.shape==(30,40,3)
        np.testing.assert_array_equal(result.layers,reduced.layers)
        assert reduced.full_photo is None
