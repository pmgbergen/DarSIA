"""Target editor event ownership only; backend pixel/proof tests are separate."""
from copy import deepcopy
import hashlib
import json
import threading

import numpy as np
from PySide6.QtCore import QPointF, QTimer, Qt
from PySide6.QtTest import QTest
import pytest

from darsia.fluidflower.painting_target_correction_ui import TargetCorrectionDialog

from .test_painting_gap_review_ui import _packet, _review


class _Draft:
    def __init__(self, *, labels, editable_mask, allowed_ids, basis):
        self.labels=np.array(labels,copy=True)
        self.mask=np.array(editable_mask,copy=True)
        self.allowed_ids=tuple(allowed_ids)
        self.basis=deepcopy(basis)
        self.actions=[];self.saves=[];self.history=[];self.future=[]
        self.dirty=False;self.snapshots=[]

    @property
    def can_undo(self):
        return bool(self.history)

    @property
    def can_redo(self):
        return bool(self.future)

    def paint(self, kind, points_xy, label_id, radius_px=1):
        action = dict(kind=kind, points_xy=points_xy, label_id=label_id, radius_px=radius_px)
        self.actions.append(deepcopy(action))
        self.history.append(self.labels.copy());self.future=[]
        # A spy pixel only: real polygon/brush rasterization belongs to the
        # separately tested backend. This verifies the GUI supplies native xy.
        column,row=np.rint(action["points_xy"][0]).astype(int)
        if self.mask[row,column]:
            self.labels[row,column]=action["label_id"]
        self.dirty=True

    def undo(self):
        self.future.append(self.labels.copy());self.labels=self.history.pop()

    def redo(self):
        self.history.append(self.labels.copy());self.labels=self.future.pop()

    def snapshot(self):
        frozen=deepcopy(self)
        self.snapshots.append(frozen)
        return frozen

    def save_and_approve(self, destination, **action):
        self.saves.append((str(destination),deepcopy(action),threading.current_thread() is threading.main_thread()))
        return dict(version={"path":"C:/synthetic/new-version.json","sha256":"a"*64,"size_bytes":1},
                    approval={"path":"C:/synthetic/new-approval.json","sha256":"b"*64,"size_bytes":1})


def _paint_packet():
    packet=dict(_packet("ac18"),component_id="c"*64)
    packet.update(target_labels=np.ones((32,40),np.uint8),editable_mask=packet["gap_mask"].copy(),
        allowed_layers=[dict(layer_id=2,name="Human chosen sand",facies="Fine",role="sand")],
        target_correction_basis=dict(run_id="ac18",component_id="c"*64,review={"synthetic":"bound"},
                                     review_row_sha256="c"*64,science={"synthetic":"exact",
        "native_photo": dict(path="C:/synthetic/native.npz",sha256="1"*64,size_bytes=100),
        "native_photo_pixels": dict(dtype="|u1",shape=[64,64,3],sha256="2"*64),
        "target_model": dict(path="C:/synthetic/model.json",sha256="3"*64,size_bytes=101)}))
    return packet


def _display_for(packet, role="current"):
    science=packet["target_correction_basis"]["science"]
    return dict(image_role=role, camera_jpg_display_only=True,
        image_rgb=np.full_like(packet["image_rgb"],70 if role=="current" else 130),image_encoding="rgb_uint8",
        native_to_photo_h=deepcopy(packet["native_to_photo_h"]),
        valid_support=np.ones(packet["image_rgb"].shape[:2],bool),
        display_grid=dict(format="darsia-painted-review-display-grid-v1",
            native_photo=deepcopy(science["native_photo"]),native_photo_pixels=deepcopy(science["native_photo_pixels"]),
            model=deepcopy(science["target_model"]),shape_hw=list(packet["target_labels"].shape),
            native_to_photo_h=np.asarray(packet["native_to_photo_h"]).tolist()))


def _dialog(qtbot, *, reader=None, current_display_reader=None, comparison_reader=None, review=None):
    packet=_paint_packet();reads=[]
    def read(review,run,component):
        reads.append((run,component,threading.current_thread() is threading.main_thread()))
        return packet
    widget=TargetCorrectionDialog(_review() if review is None else review,"ac18","c"*64,actor="Synthetic reviewer",
        destination="C:/synthetic/target_corrections",reader=reader or read,draft_factory=_Draft,
        point_native_xy=[4.,1.],
        current_display_reader=current_display_reader or (lambda *_args:_display_for(packet)),
        comparison_reader=comparison_reader or (lambda _review,_run,_component,role:_display_for(packet,role)))
    qtbot.addWidget(widget,before_close_func=lambda owned:owned.wait_for_workers())
    widget.show()
    qtbot.waitUntil(lambda:not widget.busy(),timeout=5000)
    return widget,packet,reads


def test_editor_never_selects_a_layer_or_saves_from_viewing_or_display_controls(qtbot):
    widget,packet,reads=_dialog(qtbot)
    assert reads==[("ac18","c"*64,False)]
    assert widget.layer.currentData() is None and not widget.save.isEnabled()
    assert widget.draft.actions==[] and widget.draft.saves==[]
    widget.mode.setCurrentIndex(widget.mode.findData("brush"))
    assert widget.mode.currentData()=="pan"
    original=packet["image_rgb"].copy();labels=packet["target_labels"].copy()
    widget.brighten.setChecked(True)
    widget.outline.setChecked(False);widget.boundaries.setChecked(False)
    assert widget.draft.actions==[] and widget.draft.saves==[]
    np.testing.assert_array_equal(packet["image_rgb"],original)
    np.testing.assert_array_equal(packet["target_labels"],labels)


def test_original_canvas_gesture_is_mapped_to_native_pixels_and_undo_redo_remain_local(qtbot):
    widget,packet,_=_dialog(qtbot)
    original=packet["target_labels"].copy()
    widget.layer.setCurrentIndex(1)
    widget.mode.setCurrentIndex(widget.mode.findData("brush"))
    widget.canvas.label_edit_drawn.emit("brush",[[7/63,6/63]])
    action=widget.draft.actions[-1]
    np.testing.assert_allclose(action["points_xy"],[[4.,1.]],rtol=0,atol=1e-12)
    assert action["label_id"]==2 and action["radius_px"]==1.
    assert widget.draft.labels[1,4]==2 and widget.undo.isEnabled()
    assert all(item.brush().style()==Qt.BrushStyle.NoBrush for item in widget._outlines)
    widget.undo.click();assert widget.draft.labels[1,4]==1 and widget.redo.isEnabled()
    widget.redo.click();assert widget.draft.labels[1,4]==2
    np.testing.assert_array_equal(packet["target_labels"],original)
    assert widget.draft.saves==[] and not widget.save.isEnabled()  # A reason is still required.


def _real_fixture_packet(tmp_path):
    from ..test_painted_target_corrections import _case, _pin
    from pathlib import Path

    admitted, base, editable, _science = _case(tmp_path)
    basis = deepcopy(admitted.basis)
    packet = dict(run_id=basis["run_id"], component_id=basis["component_id"],
        image_rgb=np.full((*base.shape, 3), 90, np.uint8), image_encoding="rgb_uint8",
        native_to_photo_h=np.eye(3).tolist(), target_labels=base,
        editable_mask=editable, target_correction_basis=basis,
        allowed_layers=[dict(layer_id=value, name="Fixture sand " + str(value), facies="Sand", role="sand")
                        for value in admitted.allowed_ids])
    from PIL import Image
    raw=tmp_path / "DSC00001.ARW"
    raw.write_bytes(b"synthetic exposure identity; never decoded")
    jpeg=raw.with_suffix(".JPG")
    Image.fromarray(packet["image_rgb"]).save(jpeg,quality=100,subsampling=0)
    native=tmp_path / "native.npy"
    np.save(native,packet["image_rgb"])
    model=tmp_path / "model.json"
    model.write_text(json.dumps(dict(run_id=basis["run_id"],reference_image=_pin(raw),baseline_native_grid=dict(
        H_baseline_camera_u_to_rig_corrected_native=np.eye(3).tolist(),shape_hw=list(base.shape)))),encoding="utf8")
    basis["science"].update(native_photo=_pin(native),target_model=_pin(model),
        native_photo_pixels=dict(dtype=packet["image_rgb"].dtype.str,shape=list(packet["image_rgb"].shape),
            sha256=hashlib.sha256(packet["image_rgb"].tobytes()).hexdigest()))
    display=dict(_display_for(packet),run_id=basis["run_id"],component_id=basis["component_id"],
        source_raw=_pin(raw),camera_jpeg=_pin(jpeg),display_transform=np.eye(3).tolist())
    with Image.open(jpeg) as photograph:
        display["image_rgb"]=np.array(photograph.convert("RGB"),copy=True)
    return packet,display,base,editable,basis


def test_original_canvas_stroke_uses_real_target_draft_native_points_api(qtbot, tmp_path):
    from darsia.fluidflower.painted_target_corrections import TargetCorrectionDraft
    from darsia.fluidflower.pages.geometry_facies.page import GeometryImageView
    from ..test_painted_target_corrections import _pin
    from pathlib import Path
    packet,display,base,editable,basis=_real_fixture_packet(tmp_path)
    original=base.copy()
    destination = tmp_path / "unsaved_target_corrections"
    # This is the original backend's real small, pinned synthetic basis. Only
    # the photograph reader is injected; no draft or source guard is mocked.
    widget = TargetCorrectionDialog(basis["review"], basis["run_id"], basis["component_id"],
        actor="Synthetic reviewer", destination=destination, reader=lambda *_args: packet,
        current_display_packet=display)
    qtbot.addWidget(widget, before_close_func=lambda owned: owned.wait_for_workers())
    widget.show()
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    assert type(widget.draft) is TargetCorrectionDraft and type(widget.canvas) is GeometryImageView
    widget.layer.setCurrentIndex(widget.layer.findData(3))
    widget.mode.setCurrentIndex(widget.mode.findData("brush"))
    position = widget.canvas.mapFromScene(QPointF(5., 1.))
    QTest.mousePress(widget.canvas.viewport(), Qt.MouseButton.LeftButton, pos=position)
    QTest.mouseRelease(widget.canvas.viewport(), Qt.MouseButton.LeftButton, pos=position)
    assert widget.draft.dirty and widget.draft.labels[1, 5] == 3
    assert widget.draft.events[-1]["kind"] == "brush" and widget.draft.events[-1]["label_id"] == 3
    assert "points_xy" in widget.draft.events[-1] and "points_xy_native" not in widget.draft.events[-1]
    assert widget.draft.painted_mask.any() and not np.any(widget.draft.painted_mask & ~editable)
    np.testing.assert_array_equal(widget.draft.labels[~editable], original[~editable])
    np.testing.assert_array_equal(base, original)
    widget.undo.click()
    np.testing.assert_array_equal(widget.draft.labels, original)
    widget.redo.click()
    assert widget.draft.labels[1, 5] == 3
    assert _pin(Path(basis["review"]["path"])) == basis["review"]
    assert not widget.save.isEnabled() and widget.binding is None and not destination.exists()


@pytest.mark.parametrize("damage",["jpeg","placement","wrong_run","wrong_exposure"])
def test_cached_experiment_jpg_is_rechecked_after_the_scientific_reader(qtbot,tmp_path,damage):
    from pathlib import Path
    packet,display,_base,_editable,basis=_real_fixture_packet(tmp_path)
    if damage=="jpeg":Path(display["camera_jpeg"]["path"]).write_bytes(b"changed bound JPG")
    elif damage=="placement":display["display_transform"][0][2]+=1
    elif damage=="wrong_run":display["run_id"]="another-run"
    else:display["source_raw"]["path"]=str(tmp_path / "another-exposure.ARW")
    reads=[]
    def scientific(*_args):
        reads.append("scientific packet read")
        return packet
    widget=TargetCorrectionDialog(basis["review"],basis["run_id"],basis["component_id"],
        actor="Synthetic reviewer",destination=tmp_path / "never_saved",
        reader=scientific,current_display_packet=display)
    qtbot.addWidget(widget,before_close_func=lambda owned:owned.wait_for_workers())
    widget.show()
    qtbot.waitUntil(lambda:not widget.busy(),timeout=5000)
    assert reads==["scientific packet read"]
    assert widget.draft is None and not widget.canvas.isEnabled() and not widget.save.isEnabled()
    assert widget.binding is None and "could not" in widget.status.text()


def test_only_human_save_and_approve_creates_new_binding_in_finite_worker(qtbot):
    widget,packet,_=_dialog(qtbot)
    saved=[];widget.saved.connect(saved.append)
    widget.layer.setCurrentIndex(1)
    widget.canvas.label_edit_drawn.emit("brush",[[7/63,6/63]])
    widget.reason.setText("I found sand here and painted this target")
    assert widget.draft.saves==[] and widget.save.isEnabled()
    widget.save.click()
    qtbot.waitUntil(lambda:not widget.busy(),timeout=5000)
    assert widget.draft.saves==[] and len(widget.draft.snapshots)==1 and len(saved)==1
    frozen=widget.draft.snapshots[0]
    assert len(frozen.saves)==1
    destination,action,on_gui=frozen.saves[0]
    assert destination=="C:/synthetic/target_corrections" and not on_gui
    assert action["actor"]=="Synthetic reviewer" and action["approved_utc"]
    assert action["reason"]=="I found sand here and painted this target"
    assert saved[0]["run_id"]=="ac18" and saved[0]["component_id"]=="c"*64
    assert saved[0]["binding"]==widget.binding and not widget.save.isEnabled()
    saved[0]["binding"]["version"]["sha256"]="mutable recipient"
    assert widget.binding["version"]["sha256"]=="a"*64


def test_target_photo_switch_keeps_draft_polygon_history_and_exact_view(qtbot):
    packet=_paint_packet();photo_reads=[]
    def current(review,run,component):
        photo_reads.append(("current",run,component,threading.current_thread() is threading.main_thread()))
        return _display_for(packet)
    def comparison(review,run,component,role):
        photo_reads.append((role,run,component,threading.current_thread() is threading.main_thread()))
        display=_display_for(packet,role)
        display["valid_support"][0,0]=False
        return display
    widget,scientific,_=_dialog(qtbot,current_display_reader=current,comparison_reader=comparison)
    assert photo_reads==[("current","ac18","c"*64,False)]
    assert np.all(widget._plain==70) and np.all(scientific["image_rgb"]==30)
    widget.layer.setCurrentIndex(1)
    widget.mode.setCurrentIndex(widget.mode.findData("brush"))
    point=widget.canvas.mapFromScene(QPointF(7.,6.))
    QTest.mousePress(widget.canvas.viewport(),Qt.MouseButton.LeftButton,pos=point)
    QTest.mouseRelease(widget.canvas.viewport(),Qt.MouseButton.LeftButton,pos=point)
    assert len(widget.draft.actions)==1
    widget.reason.setText("I explicitly found sand in this area")
    widget.canvas.scale(2.,2.)
    # QGraphicsView posts its automatic-scrollbar layout after zooming. The
    # previous immediate baseline included the soon-to-disappear scrollbar
    # footprint (7 viewport pixels per axis), before either photo was read.
    # Compare settled views exactly; do not loosen the photo-position oracle.
    horizontal,vertical=widget.canvas.horizontalScrollBar(),widget.canvas.verticalScrollBar()
    qtbot.waitUntil(lambda:horizontal.isVisible() and vertical.isVisible()
        and widget.canvas.viewport().width()+vertical.width()==widget.canvas.maximumViewportSize().width()
        and widget.canvas.viewport().height()+horizontal.height()==widget.canvas.maximumViewportSize().height(),
        timeout=1000)
    widget.canvas.centerOn(12.,12.)
    widget.mode.setCurrentIndex(widget.mode.findData("polygon"))
    for x,y in ((8.,6.),(8.,7.)):
        QTest.mouseClick(widget.canvas.viewport(),Qt.MouseButton.LeftButton,
            pos=widget.canvas.mapFromScene(QPointF(x,y)))
    polygon=list(widget.canvas._edit_points)
    assert len(polygon)==2
    transform=widget.canvas.transform()
    centre=widget.canvas.mapToScene(widget.canvas.viewport().rect().center())
    actions=deepcopy(widget.draft.actions)
    labels=widget.draft.labels.copy()
    widget.photo_toggle.click()
    assert widget.busy() and not widget.canvas.isEnabled() and not widget.save.isEnabled()
    qtbot.waitUntil(lambda:not widget.busy(),timeout=5000)
    assert widget._photo_role=="clearwater" and np.all(widget._plain[0,0]==64)
    assert widget.canvas.transform()==transform and widget.canvas.mapToScene(widget.canvas.viewport().rect().center())==centre
    assert widget.canvas._edit_points==polygon and widget.draft.actions==actions
    np.testing.assert_array_equal(widget.draft.labels,labels)
    assert widget.save.isEnabled() and widget.draft.saves==[] and widget.binding is None
    widget.activateWindow()
    qtbot.waitUntil(widget.isActiveWindow,timeout=1000)
    widget.canvas.setFocus()
    QTest.keyClick(widget.canvas,Qt.Key.Key_I,Qt.KeyboardModifier.AltModifier)
    qtbot.waitUntil(lambda:not widget.busy() and widget._photo_role=="current",timeout=5000)
    assert widget.canvas.transform()==transform and widget.canvas.mapToScene(widget.canvas.viewport().rect().center())==centre
    assert widget.canvas._edit_points==polygon and widget.draft.actions==actions
    assert photo_reads==[("current","ac18","c"*64,False),("clearwater","ac18","c"*64,False),
                         ("current","ac18","c"*64,False)]
    np.testing.assert_array_equal(scientific["image_rgb"],packet["image_rgb"])


@pytest.mark.parametrize("damage",["grid","transform","role","numeric_kind","support_dtype","support_shape"])
def test_target_photo_drift_pauses_editing_without_changing_science_or_draft(qtbot,damage):
    packet=_paint_packet()
    def comparison(_review,_run,_component,role):
        display=_display_for(packet,role)
        if damage=="grid":display["display_grid"]["model"]["sha256"]="4"*64
        elif damage=="transform":display["native_to_photo_h"][0][2]+=1
        elif damage=="role":display["image_role"]="current"
        elif damage=="numeric_kind":display["display_grid"]["native_photo"]["size_bytes"]=100.
        elif damage=="support_dtype":display["valid_support"]=display["valid_support"].astype(np.uint8)
        else:display["valid_support"]=np.ones((2,3),bool)
        return display
    widget,scientific,_=_dialog(qtbot,comparison_reader=comparison)
    before=widget._plain.copy()
    widget.photo_toggle.click()
    qtbot.waitUntil(lambda:not widget.busy(),timeout=5000)
    assert not widget._display_valid and not widget.canvas.isEnabled() and not widget.save.isEnabled()
    assert widget._photo_role=="current" and "could not" in widget.status.text()
    np.testing.assert_array_equal(widget._plain,before)
    assert widget.draft.actions==[] and widget.draft.saves==[] and widget.binding is None
    np.testing.assert_array_equal(scientific["target_labels"],packet["target_labels"])
    # A valid retry is explicit and uses the same owned reader, not a fallback.
    widget.comparison_reader=lambda _review,_run,_component,role:_display_for(packet,role)
    widget.photo_toggle.click()
    qtbot.waitUntil(lambda:not widget.busy(),timeout=5000)
    assert widget._display_valid and widget._photo_role=="clearwater" and widget.canvas.isEnabled()


def test_configured_scientific_target_reader_runs_even_when_initial_jpg_fails(qtbot):
    def refuse(*_args):
        raise ValueError("The bound camera JPG changed")
    widget,_packet_value,reads=_dialog(qtbot,current_display_reader=refuse)
    assert reads==[("ac18","c"*64,False)]
    assert widget.draft is None and not widget.save.isEnabled() and not widget.canvas.isEnabled()
    assert "camera JPG changed" in widget.status.text()


@pytest.mark.parametrize("shutdown",[False,True])
def test_close_or_shutdown_during_target_comparison_retires_original_reader(qtbot,shutdown):
    entered,release=threading.Event(),threading.Event()
    packet=_paint_packet()
    def finite(_review,_run,_component,role):
        entered.set()
        assert release.wait(5),"The finite photo read must be released"
        return _display_for(packet,role)
    widget,_packet_value,_=_dialog(qtbot,comparison_reader=finite)
    widget.photo_toggle.click()
    try:
        qtbot.waitUntil(entered.is_set,timeout=2000)
        if shutdown:
            timer=threading.Timer(.02,release.set)
            timer.start()
            widget.wait_for_workers()
            timer.join()
        else:
            widget.reject()
            assert widget.busy() and widget._closing and not widget.photo_shortcut.isEnabled()
            ticks=[]
            QTimer.singleShot(0,lambda:ticks.append("responsive"))
            qtbot.waitUntil(lambda:bool(ticks),timeout=2000)
            release.set()
            qtbot.waitUntil(lambda:not widget.busy(),timeout=5000)
        assert widget.thread is None and widget.worker is None and not widget.isVisible()
        assert widget._photo_role=="current" and widget.draft.actions==[] and widget.draft.saves==[]
        assert widget.binding is None
    finally:
        release.set()
        qtbot.waitUntil(lambda:not widget.busy(),timeout=5000)


def test_wrong_target_packet_refuses_before_a_draft_or_any_save(qtbot):
    def wrong(*_args):
        return dict(_paint_packet(),run_id="another-run")
    widget,packet,_=_dialog(qtbot,reader=wrong)
    assert widget.draft is None and not widget.save.isEnabled()
    assert "another run" in widget.status.text()


def test_close_retires_reader_without_creating_or_approving_a_draft(qtbot):
    entered,release=threading.Event(),threading.Event()
    def finite(*_args):
        entered.set()
        assert release.wait(5),"The finite private read must be released"
        return _paint_packet()
    widget=TargetCorrectionDialog(_review(),"ac18","c"*64,actor="Synthetic reviewer",
        destination="C:/synthetic/target_corrections",reader=finite,draft_factory=_Draft,
        current_display_reader=lambda *_args:_display_for(_paint_packet()))
    qtbot.addWidget(widget,before_close_func=lambda owned:owned.wait_for_workers())
    try:
        qtbot.waitUntil(entered.is_set,timeout=2000)
        widget.reject()  # Escape uses the same nonblocking deferred-close path.
        assert widget.busy() and widget._closing and widget.draft is None
        ticks=[]
        QTimer.singleShot(0,lambda:ticks.append("responsive"))
        qtbot.waitUntil(lambda:bool(ticks),timeout=2000)
        assert widget.busy() and not release.is_set()
        release.set()
        qtbot.waitUntil(lambda:not widget.busy(),timeout=5000)
        assert widget.draft is None and widget.binding is None
    finally:
        release.set()
        qtbot.waitUntil(lambda:not widget.busy(),timeout=5000)


def test_save_captures_detached_draft_and_destination_before_worker_dispatch(qtbot,monkeypatch):
    widget,packet,_=_dialog(qtbot)
    widget.layer.setCurrentIndex(1)
    widget.canvas.label_edit_drawn.emit("brush",[[7/63,6/63]])
    widget.reason.setText("The reviewer explicitly approved this local stroke")
    dispatch=[]
    monkeypatch.setattr(widget,"_start",lambda operation,function:dispatch.append((operation,function)))
    widget.save.click()
    assert len(widget.draft.snapshots)==1 and dispatch[0][0]=="save"
    frozen=widget.draft.snapshots[0]
    widget.draft.labels[1,4]=9
    widget.draft.actions[0]["label_id"]=9
    widget.destination="C:/synthetic/changed-after-click"
    widget.reason.setText("Changed after dispatch")
    dispatch[0][1]()
    assert frozen.labels[1,4]==2 and frozen.actions[0]["label_id"]==2
    destination,action,_=frozen.saves[0]
    assert destination=="C:/synthetic/target_corrections"
    assert action["reason"]=="The reviewer explicitly approved this local stroke"
    assert widget.draft.saves==[]


def test_close_during_explicit_save_is_nonblocking_and_retains_binding_before_closed(qtbot,monkeypatch):
    entered,release=threading.Event(),threading.Event()
    original=_Draft.save_and_approve
    def finite(draft,*args,**options):
        entered.set()
        assert release.wait(5),"The finite private save must be released"
        return original(draft,*args,**options)
    monkeypatch.setattr(_Draft,"save_and_approve",finite)
    widget,packet,_=_dialog(qtbot)
    widget.layer.setCurrentIndex(1)
    widget.canvas.label_edit_drawn.emit("brush",[[7/63,6/63]])
    widget.reason.setText("The human explicitly saves and approves this correction")
    order=[]
    widget.saved.connect(lambda binding:order.append(("saved",binding,widget.isVisible())))
    widget.finished.connect(lambda result:order.append(("closed",result)))
    try:
        widget.save.click()
        qtbot.waitUntil(entered.is_set,timeout=2000)
        widget.close()
        assert widget.busy() and widget.isVisible() and widget._closing
        ticks=[]
        QTimer.singleShot(0,lambda:ticks.append("responsive"))
        qtbot.waitUntil(lambda:bool(ticks),timeout=2000)
        assert widget.busy() and not release.is_set() and order==[]
        release.set()
        qtbot.waitUntil(lambda:not widget.busy() and not widget.isVisible(),timeout=5000)
        assert [row[0] for row in order]==["saved","closed"]
        assert order[0][1]["binding"]==widget.binding and order[0][2] is True
        assert widget.binding["version"]["sha256"]=="a"*64
        assert widget.draft.saves==[] and len(widget.draft.snapshots[0].saves)==1
    finally:
        release.set()
        qtbot.waitUntil(lambda:not widget.busy(),timeout=5000)


def test_application_shutdown_barrier_retires_reader_and_save_without_gui_delivery(qtbot,monkeypatch):
    from .test_painting_gap_review_ui import _controller
    host,backend,controller=_controller(qtbot,monkeypatch)
    review=_review()
    # The current controller joins a completed target Save to the exact owned
    # review row. Supply the finite reader's explicit descriptor and preserve
    # its genuine captured GUI context, as a normal opened dialog does.
    review["review_binding"]=dict(path="C:/synthetic/shutdown-gap-review.json",sha256="f"*64,size_bytes=101)
    gaps=controller.open_gap_review(review)
    qtbot.waitUntil(lambda:not gaps.busy(),timeout=5000)
    saved,packet,_=_dialog(qtbot,review=review)
    controller.gap_dialog=gaps
    assert controller._current() and saved.review["review_binding"]==gaps.review["review_binding"]
    host._target_correction_dialog=saved
    saved.saved.connect(controller._target_saved)
    read_entered,save_entered,release=threading.Event(),threading.Event(),threading.Event()
    original_read=gaps._read_preview
    original_save=_Draft.save_and_approve
    def finite_read(*args):
        read_entered.set()
        assert release.wait(5),"Finite reader must finish at shutdown"
        return original_read(*args)
    def finite_save(draft,*args,**options):
        save_entered.set()
        assert release.wait(5),"Explicit finite save must finish at shutdown"
        return original_save(draft,*args,**options)
    gaps._read_preview=finite_read
    monkeypatch.setattr(_Draft,"save_and_approve",finite_save)
    saved.layer.setCurrentIndex(1)
    saved.canvas.label_edit_drawn.emit("brush",[[7/63,6/63]])
    saved.reason.setText("The human explicitly approved before application shutdown")
    try:
        gaps.runs.setCurrentCell(1,0)
        saved.save.click()
        qtbot.waitUntil(lambda:read_entered.is_set() and save_entered.is_set(),timeout=2000)
        # A Python timer releases both workers even while the GUI event loop
        # is unavailable in the application's aboutToQuit shutdown barrier.
        timer=threading.Timer(.02,release.set)
        timer.start()
        controller.wait_for_workers()
        timer.join()
        assert not gaps.busy() and not saved.busy()
        assert gaps.packet is None and gaps._pending is None
        assert saved.binding is not None and controller.context is None
        assert controller._target_bindings[("ac18","c"*64)]==saved.binding
        assert len(saved.draft.snapshots[0].saves)==1
        # Delivering the old queued terminal signals cannot publish twice.
        qtbot.wait(1)
        assert len(saved.draft.snapshots[0].saves)==1
    finally:
        release.set()
        controller.wait_for_workers()
