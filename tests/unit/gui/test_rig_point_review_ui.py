"""Each full photograph must be reviewed; original and corrected points remain visible."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from threading import Event, Timer

import pytest
from PySide6.QtCore import Qt, QPoint

from darsia.fluidflower import rig_point_review as review
from darsia.fluidflower import rig_point_review_ui as ui
from darsia.fluidflower.map_placement import GEOMETRY_FORMAT
from darsia.fluidflower.painting_frame_models import capture_frame_model_selection
from darsia.fluidflower.rig_reference_quadrilateral import reference_frame_geometry
from darsia.fluidflower.series_geometry import RigFrame, save_rig_frame
from .test_rig_full_raw_editor import packet
from .test_painting_raw_reference_on_u import raw_view, raw_file, profile, renders, settings  # noqa: F401

PHOTO=dict(top_left=[5.,5.],top_right=[5.,70.],bottom_right=[50.,70.],bottom_left=[50.,5.])


@pytest.fixture
def context(raw_view,raw_file,monkeypatch,qtbot):
    view=raw_view
    def metadata(name,value,**kwargs):
        view.project.manifest.setdefault(name,{}).update(deepcopy(value))
    view.project.update_metadata=metadata
    view.project.manifest.setdefault('tools',{})['native_series']={'series_root':str(raw_file.parent)}
    source=view.reference()
    other=raw_file.with_name('second.ARW')
    other.write_bytes(b'private second photograph')
    target=replace(source,run_id='ac53',baseline_image=other)
    view._static_reference_runs=lambda:[source,target]
    frame=RigFrame(**reference_frame_geometry(PHOTO,measurements_m=dict(width=.9,height=.49)),
        run_id=source.run_id,image=str(raw_file),image_size=(80,60))
    save_rig_frame(view.project,frame,actor='Reviewer')
    view.rig_panel.load()
    folder=view.project.root/'point-models'
    folder.mkdir(parents=True)
    entries={}
    for run,path,shift in ((source.run_id,raw_file,0.),(target.run_id,other,2.)):
        model=dict(format=GEOMETRY_FORMAT,run_id=run,shape_hw=[60,80],
            H_metric_mm_to_camera_u=[[1.,0.,shift],[0.,1.,0.],[0.,0.,1.]],
            reference_image=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size_bytes=path.stat().st_size))
        model_path=folder/(run+'.json')
        model_path.write_text(json.dumps(model),encoding='utf8')
        entries[run]={'reference_model':str(model_path)}
    index=folder/'index.json'
    index.write_text(json.dumps(dict(format=GEOMETRY_FORMAT,runs=entries)),encoding='utf8')
    selection=capture_frame_model_selection(view.project,override_path=str(index))
    view.run_preview._index=str(index)
    view.run_preview._geometry_selection=selection
    monkeypatch.setattr(ui,'_read_photo',lambda path,profile:packet(path))
    view._rig_photo_loader.reader=lambda path,profile:packet(path)
    view.open_reference_point_review()
    panel=view._point_review_panel
    qtbot.waitUntil(lambda:panel._viewed_identity is not None)
    return view,panel,other


def test_early_review_is_embedded_and_requires_each_own_photo_then_preserves_corrections(context,qtbot):
    view,panel,other=context
    assert not panel.isWindow()
    assert view.run_preview.stack.currentWidget() is panel
    assert panel.current_run()=='ac52'
    assert panel.canvas.pixmap_item.pixmap().size().toTuple()==(120,95)
    assert not panel.approve.isEnabled()
    assert view._reference_point_review_blocker()
    original=deepcopy(panel.document['runs']['ac52']['proposed_rc'])
    panel.auto_next.setChecked(False)
    panel.points.item(0,2).setText('6.25; 7.5')
    assert panel.document['runs']['ac52']['before_rc']==original
    assert panel.document['runs']['ac52']['current_rc']['top_left']==[6.25,7.5]
    assert float(panel.points.item(0,3).text())>0
    panel.checked.setChecked(True)
    assert panel.approve.isEnabled()
    panel.approve.click()
    assert panel.document['runs']['ac52']['status']=='approved'
    assert panel.runs.item(0,1).text()=='Approved with changes'
    assert view._reference_point_review_blocker()  # The target still needs its own review.
    panel.runs.selectRow(1)
    qtbot.waitUntil(lambda:panel.current_run()=='ac53' and panel._viewed_identity is not None)
    assert Path(panel.canvas.preview['path'])==other.resolve()
    assert not panel.checked.isChecked() and not panel.approve.isEnabled()
    panel.points.selectRow(2)
    panel.place.setChecked(True)
    panel.canvas.raw_point_selected.emit(49.,71.)
    assert not panel.place.isChecked()
    assert panel.document['runs']['ac53']['current_rc']['bottom_right']==[49.,71.]
    panel.checked.setChecked(True)
    panel.approve.click()
    qtbot.waitUntil(lambda:view._reference_point_review_blocker() is None)
    assert view.run_preview.geometry_selection().source=='reviewed reference points'
    selected=view.run_preview.geometry_selection()
    preview_key=view.run_preview._key()
    view._reviewed_point_selection=None
    assert view.run_preview._key()!=preview_key
    view._restore_reviewed_point_selection()
    qtbot.waitUntil(lambda:view._reference_point_review_blocker() is None)
    assert view.run_preview.geometry_selection()==selected
    assert view.run_preview._key()==preview_key
    stored=review.load_review(view.project)
    assert stored['runs']['ac52']['proposed_rc']==original
    assert all(row['status']=='approved' for row in stored['runs'].values())
    assert '2 of 2 approved' in panel.summary.text()
    panel.return_button.click()
    assert view.run_preview.stack.currentIndex()==0
    assert not view._point_review_active


def test_correcting_an_approved_source_can_be_saved_resumed_and_approved_again(context,qtbot):
    view,panel,_=context
    panel.auto_next.setChecked(False)
    panel.checked.setChecked(True)
    panel.approve.click()
    old=deepcopy(panel.document['runs']['ac52']['proposed_rc'])
    panel.points.item(0,2).setText('7; 6')
    assert panel.document['runs']['ac52']['status']=='corrected'
    panel.return_button.click()
    view.open_reference_point_review()
    qtbot.waitUntil(lambda:panel._viewed_identity is not None)
    assert panel.document['runs']['ac52']['current_rc']['top_left']==[7.,6.]
    panel.checked.setChecked(True)
    panel.approve.click()
    assert panel.document['runs']['ac52']['status']=='approved'
    assert panel.document['runs']['ac52']['before_rc']==old


def test_excluded_run_explains_missing_points_and_review_list_stays_in_left_section(context,qtbot):
    view,panel,other=context
    frame=view.current_frame()
    panel.document=review.prepare_review(frame,[
        dict(run_id='ac52',path=frame.image,image_size=(80,60)),
        dict(run_id='ac53',path=str(other),image_size=(80,60)),
    ])
    panel.document['required_runs']=['ac52']
    panel._populate()
    qtbot.waitUntil(lambda:panel._viewed_identity is not None)
    assert view.point_review_host.isAncestorOf(panel.runs)
    assert not panel.isAncestorOf(panel.runs)
    assert panel.runs.columnCount()==2
    assert panel.runs.item(1,1).text()=='Outside transfer · No point proposal'
    assert 'AC53' in panel.scope_note.text()
    panel.runs.selectRow(1)
    qtbot.waitUntil(lambda:panel.current_run()=='ac53' and panel._viewed_identity is not None)
    assert 'outside the selected geometry' in panel.note.text()
    assert 'not required' in panel.note.text()
    assert not panel._items and not panel.approve.isEnabled()
    assert panel.run_title.text()=='AC53'
    # This inventory row can still be placed explicitly; no coordinates are invented.
    panel.points.selectRow(0)
    panel.place.setChecked(True)
    panel.canvas.raw_point_selected.emit(8.,9.)
    assert panel.document['runs']['ac53']['current_rc']=={'top_left':[8.,9.]}


def test_unchanged_approval_status_is_retained_when_reopening_the_left_review_section(context,qtbot):
    view,panel,_=context
    panel.auto_next.setChecked(False)
    panel.checked.setChecked(True)
    panel.approve.click()
    assert panel.runs.item(0,1).text()=='Approved directly'
    approved=deepcopy(panel.document['runs']['ac52'])
    view.close_reference_point_review()
    view.open_reference_point_review()
    qtbot.waitUntil(lambda:panel._viewed_identity is not None)
    assert panel.document['runs']['ac52']==approved
    assert panel.runs.item(0,1).text()=='Approved directly'


def test_cyan_point_can_be_dragged_without_changing_yellow_proposal(context,qtbot):
    _,panel,_=context
    marker=next(item for item in panel._items if isinstance(item,ui._Point) and item.name=='top_left')
    original=deepcopy(panel.document['runs']['ac52']['before_rc'])
    start=panel.canvas.mapFromScene(marker.pos())
    end=start+QPoint(18,12)
    qtbot.mousePress(panel.canvas.viewport(),Qt.MouseButton.LeftButton,pos=start)
    qtbot.mouseMove(panel.canvas.viewport(),pos=end)
    qtbot.mouseRelease(panel.canvas.viewport(),Qt.MouseButton.LeftButton,pos=end)
    qtbot.waitUntil(lambda:panel.document['runs']['ac52']['current_rc']!=original)
    assert panel.document['runs']['ac52']['before_rc']==original
    assert not panel.checked.isChecked()


def test_changed_photo_cannot_be_approved_with_an_old_point_review(context,qtbot):
    _,panel,other=context
    other.write_bytes(other.read_bytes()+b' changed')
    panel.runs.selectRow(1)
    qtbot.waitUntil(lambda:'photograph changed' in panel.note.text().lower())
    assert panel._viewed_identity is None and not panel.approve.isEnabled()


def test_points_outside_actual_photo_support_cannot_be_approved(context):
    _,panel,_=context
    panel.points.item(0,2).setText('-100; -100')
    panel.checked.setChecked(True)
    assert not panel.approve.isEnabled()


def test_changing_reference_dimensions_invalidates_point_approvals_and_transfer(context):
    view,panel,_=context
    panel.auto_next.setChecked(False)
    panel.checked.setChecked(True)
    panel.approve.click()
    source=view.current_frame()
    source.width_m=.8
    assert 'changed' in view._reference_point_review_blocker().lower()
    panel.points.item(0,2).setText('7; 7')
    assert 'reference changed' in panel.note.text().lower()


def test_close_joins_the_finite_point_photo_reader_without_applying_a_late_image(context,qtbot):
    view,panel,_=context
    hold,started=Event(),Event()
    def reader(path,profile):
        started.set()
        hold.wait(5)
        return packet(path)
    panel.loader.reader=reader
    try:
        panel.runs.selectRow(1)
        qtbot.waitUntil(started.is_set)
        assert not view.can_close()
        view.close()
        assert view.isVisible()
        release=Timer(.05,hold.set)
        release.start()
        view.wait_for_workers()
        release.join()
        assert not panel.busy() and view.can_close()
        assert panel._viewed_identity is None and panel.canvas.preview is None
    finally:
        hold.set()
        panel.wait_for_workers()
