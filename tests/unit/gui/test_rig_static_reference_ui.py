"""Physical reference actions remain explicit and bind observations to their run."""
from copy import deepcopy
from types import SimpleNamespace
import threading

import numpy as np
import pytest
from PySide6.QtCore import QPointF, QThread, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QWidget

from darsia.fluidflower import rig_static_reference_ui as ui
from darsia.fluidflower import series_rig_frame_panel as rig_ui


class Canvas(QGraphicsView):
    image_point_selected = Signal(float, float)
    raw_point_selected = Signal(float, float)

    def __init__(self, parent):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        pixmap = QPixmap(120, 120)
        pixmap.fill()
        self.pixmap_item = self.scene().addPixmap(pixmap)
        self.place_mode = False
        self.photo_display_mapping = None
        self.preview = None

    def raw_to_scene(self, row, col):
        return QPointF(col+20., row+20.)


class Backend:
    def __init__(self):
        self.saved = []
        self.loads = []
        self.recognizer = None

    def empty_document(self):
        return dict(format="test", regions=[], runs={})

    def load_references(self, root):
        self.loads.append(QThread.currentThread())
        return self.empty_document()

    def save_references(self, root, document, **kwargs):
        self.saved.append((deepcopy(document), QThread.currentThread()))
        return deepcopy(document)

    def photo_binding(self, packet):
        return dict(identity=packet["source_raw"]["sha256"], source_raw=deepcopy(packet["source_raw"]))

    def capture_region(self, packet, *, region_id, name, rect_u_xywh, anchor_u_xy, metric_xy_mm, plane_id):
        return dict(id=region_id, name=name, rect_u_xywh=rect_u_xywh, anchor_u_xy=anchor_u_xy,
            metric_xy_mm=metric_xy_mm, plane_id=plane_id, identity=region_id+"-v1")

    def update_region(self, region, **values):
        result = deepcopy(region)
        result.update(values)
        result["identity"] += "-edited"
        return result

    def fit_reference_model(self, regions, observations, *, plane_id, measurement_uncertainty_mm=0.):
        used = [region["id"] for region in regions if region["plane_id"] == plane_id
            and observations.get(region["id"], {}).get("status") in ("recognized", "manual")]
        model = dict(status="insufficient", reason="Four spread anchors required", used_region_ids=used,
            H_metric_mm_to_camera_u=None)
        if len(used) >= 4:
            model.update(status="valid", reason="", rms_u_px=0.,
                H_metric_mm_to_camera_u=[[1., 0., -10.], [0., -1., 100.], [0., 0., 1.]])
        return model

    def recognize_regions(self, document, packet):
        if self.recognizer:
            return self.recognizer(document, packet)
        return dict(run_id=packet["run_id"], photo=self.photo_binding(packet),
            observations={region["id"]: dict(anchor_u_xy=region["anchor_u_xy"], score=.94,
                status="recognized", method="features") for region in document["regions"]})


def packet(run_id):
    return dict(run_id=run_id, image_rgb=np.zeros((120, 120, 3), np.uint8),
        valid_support=np.ones((120, 120), bool),
        mapping=SimpleNamespace(camera_u_to_display_xy=lambda points: np.asarray(points)+20.),
        source_raw=dict(path=f"{run_id}.CR2", sha256=run_id, size_bytes=17),
        profile=dict(sha256="profile", profile_id="lens"))


@pytest.fixture
def context(qtbot, monkeypatch, tmp_path):
    backend = Backend()
    monkeypatch.setattr(ui, "_backend", lambda: backend)
    monkeypatch.setattr(rig_ui, "saved_rig_frames", lambda project: {})
    monkeypatch.setattr(rig_ui, "saved_injection_points", lambda project: [])
    monkeypatch.setattr(rig_ui, "rig_frame_from_inputs", lambda reference: None)
    monkeypatch.setattr(rig_ui, "saved_crop_choice", lambda project: rig_ui.CROP_REFERENCE_QUAD)
    monkeypatch.setattr(rig_ui, "saved_crop_areas", lambda project: {})
    page = QWidget()
    qtbot.addWidget(page)
    page.canvas = Canvas(page)
    page.current_run, page.ready = "run1", True
    page.runs = [SimpleNamespace(run_id=f"run{number}", baseline_image=None) for number in range(1, 6)]
    page.static_reference_context = lambda: (page.current_run, page.ready)
    page.static_reference_runs = lambda: page.runs
    page.read_static_reference_packet = lambda reference: packet(reference.run_id)
    page.canvas.preview = packet("run1")
    panel = rig_ui.RigFramePanel(page, SimpleNamespace(root=tmp_path),
        lambda: next(run for run in page.runs if run.run_id == page.current_run),
        scrollable=False, legacy_static_references=True)
    qtbot.addWidget(panel)
    references = panel.static_references
    references.context_changed()
    qtbot.waitUntil(lambda: not references.busy(), timeout=5000)
    yield SimpleNamespace(page=page, rig=panel, ui=references, backend=backend)
    references.wait_for_workers()


def define(context, x=1., y=2.):
    panel = context.ui
    panel.name.setText("Frame corner")
    panel.x_mm.setValue(x)
    panel.y_mm.setValue(y)
    panel.define_button.click()
    context.page.canvas.raw_point_selected.emit(-10., -10.)
    context.page.canvas.raw_point_selected.emit(20., 20.)
    context.page.canvas.raw_point_selected.emit(0., 0.)
    return panel.document["regions"][-1]


def test_three_native_u_clicks_share_measurements_and_save_only_explicitly(context, qtbot):
    region = define(context, 91., 47.)
    panel, backend = context.ui, context.backend
    assert region["rect_u_xywh"] == [-10., -10., 30., 30.]
    assert region["anchor_u_xy"] == [0., 0.]
    assert region["metric_xy_mm"] == [91., 47.]
    assert panel.document["runs"]["run1"]["observations"][region["id"]]["status"] == "manual"
    assert not backend.saved and not context.page.canvas.place_mode
    assert len(panel._items) == 3
    assert all(thread != context.page.thread() for thread in backend.loads)
    assert panel.save()
    qtbot.waitUntil(lambda: not panel.busy(), timeout=5000)
    assert len(backend.saved) == 1 and backend.saved[0][1] != context.page.thread()
    assert not panel.dirty


def test_anchor_outside_area_is_rejected_and_context_switch_cancels_interaction(context):
    panel = context.ui
    panel.define_button.click()
    context.page.canvas.raw_point_selected.emit(0., 0.)
    context.page.canvas.raw_point_selected.emit(20., 20.)
    context.page.canvas.raw_point_selected.emit(25., 25.)
    assert not panel.document["regions"] and len(panel._clicks) == 2
    assert "inside" in panel.note.text()
    context.page.ready = False
    panel.context_changed()
    assert not context.page.canvas.place_mode and not panel._mode
    context.page.canvas.raw_point_selected.emit(10., 10.)
    assert not panel.document["regions"] and not panel.define_button.isEnabled()


def test_shared_measurement_edit_invalidates_run_models_and_other_photo_is_hidden(context):
    region = define(context)
    panel = context.ui
    panel.document["runs"]["run2"] = deepcopy(panel.document["runs"]["run1"])
    for result in panel.document["runs"].values():
        result["model"] = dict(status="valid")
    panel.table.selectRow(0)
    panel.x_mm.setValue(51.)
    panel.update_button.click()
    assert region["metric_xy_mm"][0] == 51.
    assert all("model" not in result and region["id"] not in result["observations"]
        for result in panel.document["runs"].values())
    panel.table.selectRow(0)
    panel.correct_button.click()
    context.page.canvas.raw_point_selected.emit(3., -7.)
    assert panel._observations()[region["id"]]["anchor_u_xy"] == [-7., 3.]
    context.page.canvas.preview = packet("run1")
    context.page.canvas.preview["source_raw"]["sha256"] = "new-photo"
    panel.context_changed()
    assert not panel._observations() and not panel._items


def test_display_only_context_change_keeps_unsaved_regions_and_rig_mode_cancels_clicks(context):
    define(context)
    panel = context.ui
    before = deepcopy(panel.document)
    # A replacement display packet with the same source identity is the same context.
    context.page.canvas.preview = packet("run1")
    panel.context_changed()
    assert panel.document == before and panel.dirty
    panel.define_button.click()
    assert panel._mode == "define"
    context.rig._set_mode("corners")
    assert not panel._mode and context.rig._mode == "corners"


def test_all_runs_is_bounded_parallel_responsive_and_bound_to_original_runs(context, qtbot):
    define(context)
    panel = context.ui
    entered, release = threading.Event(), threading.Event()
    lock = threading.Lock()
    active, maximum = 0, 0
    beats = []
    timer = QTimer(context.page)
    timer.setInterval(5)
    timer.timeout.connect(lambda: beats.append(1))

    def reader(reference):
        nonlocal active, maximum
        assert QThread.currentThread() != context.page.thread()
        with lock:
            active += 1
            maximum = max(maximum, active)
            if active == 2:
                entered.set()
        assert release.wait(5)
        with lock:
            active -= 1
        return packet(reference.run_id)

    context.page.read_static_reference_packet = reader
    timer.start()
    try:
        assert panel.find_regions(True)
        qtbot.waitUntil(entered.is_set, timeout=3000)
        qtbot.waitUntil(lambda: len(beats) >= 3, timeout=3000)
        assert panel.busy() and not panel.can_close()
        release.set()
        qtbot.waitUntil(lambda: not panel.busy(), timeout=5000)
        assert maximum == 2
        assert set(panel.document["runs"]) == {run.run_id for run in context.page.runs}
        assert panel.table.item(0, 4).text() == "recognized"
        assert panel.table.item(0, 5).text() == "0.940"
        assert not context.backend.saved
    finally:
        release.set()
        panel.wait_for_workers()
        timer.stop()


def test_stale_worker_is_discarded_on_run_switch(context, qtbot):
    define(context)
    before = deepcopy(context.ui.document)
    entered, release = threading.Event(), threading.Event()

    def reader(reference):
        entered.set()
        assert release.wait(5)
        return packet(reference.run_id)

    context.page.read_static_reference_packet = reader
    try:
        assert context.ui.find_regions(True)
        qtbot.waitUntil(entered.is_set, timeout=3000)
        context.page.current_run = "run2"
        context.page.canvas.preview = packet("run2")
        context.ui.context_changed()
        release.set()
        qtbot.waitUntil(lambda: not context.ui.busy(), timeout=5000)
        assert context.ui.document == before
        assert "discarded" in context.ui.note.text()
    finally:
        release.set()
        context.ui.wait_for_workers()


def test_one_failed_run_does_not_keep_its_previous_recognition(context, qtbot):
    region = define(context)
    panel = context.ui
    panel.document["runs"]["run3"] = deepcopy(panel.document["runs"]["run1"])
    panel.document["runs"]["run3"]["observations"][region["id"]]["status"] = "recognized"

    def reader(reference):
        if reference.run_id == "run3":
            raise OSError("Photograph unavailable")
        return packet(reference.run_id)

    context.page.read_static_reference_packet = reader
    assert panel.find_regions(True)
    qtbot.waitUntil(lambda: not panel.busy(), timeout=5000)
    assert panel.document["runs"]["run3"]["observations"] == {}
    assert panel.document["runs"]["run3"]["status"] == "failed"
    assert "run3: Photograph unavailable" in panel.note.text()
    assert panel.document["runs"]["run2"]["observations"][region["id"]]["status"] == "recognized"


def test_explicit_rig_frame_uses_origin_and_existing_size_and_rejects_other_planes(context, monkeypatch):
    panel, rig = context.ui, context.rig
    for number in range(4):
        define(context, number*20., number*10.)
    rig._loading = True
    rig.width_m.setValue(.09)
    rig.height_m.setValue(.05)
    rig._loading = False
    panel.origin_x_mm.setValue(30.)
    panel.origin_y_mm.setValue(40.)
    calls = []
    monkeypatch.setattr(rig, "_store_frame", lambda: calls.append(True))
    assert panel.use_for_rig_frame()
    assert calls == [True]
    # mm -> native U: x-10, 100-y, in the rig panel's TL, TR, BR, BL order.
    assert rig.reference_photo_corners() == dict(zip(rig_ui.CLICK_ORDER,
        [(10., 20.), (10., 110.), (60., 110.), (60., 20.)]))
    panel.plane.setCurrentText("bolt-heads")
    assert not panel.use_for_rig_frame() and calls == [True]
    assert "plane conversion" in panel.note.text()


def test_real_backend_capture_measurement_update_and_manual_point_survive_explicit_save(context, qtbot, monkeypatch, tmp_path):
    from darsia.fluidflower import rig_static_references as refs
    from ..test_rig_static_references import scene

    source, _, _ = scene(tmp_path)
    source["mapping"] = SimpleNamespace(camera_u_to_display_xy=lambda points: np.asarray(points)+[100., 60.])
    monkeypatch.setattr(ui, "_backend", lambda: refs)
    panel, page = context.ui, context.page
    panel.document = refs.empty_document()
    panel.document["measurement_uncertainty_mm"] = 2.
    page.current_run = "source"
    page.runs = [SimpleNamespace(run_id="source", baseline_image=None)]
    page.canvas.preview = source
    panel.context_changed()
    panel.name.setText("Measured frame corner")
    panel.x_mm.setValue(90.)
    panel.y_mm.setValue(1110.)
    panel.define_button.click()
    for row, col in ((-15., -55.), (96., 56.), (40., 0.)):
        page.canvas.raw_point_selected.emit(row, col)
    assert len(panel.document["regions"]) == 1, panel.note.text()
    region = panel.document["regions"][0]
    first_identity = region["identity"]
    panel.table.selectRow(0)
    panel.x_mm.setValue(91.)
    panel.update_button.click()
    assert region["identity"] != first_identity
    panel.table.selectRow(0)
    panel.correct_button.click()
    page.canvas.raw_point_selected.emit(41., 1.)
    panel.uncertainty_mm.setValue(2.5)
    path = tmp_path / "annotations" / "rig_static_references" / "references.json"
    assert not path.exists()
    assert panel.save()
    qtbot.waitUntil(lambda: not panel.busy(), timeout=5000)
    saved = refs.load_references(tmp_path)
    assert saved["regions"][0]["metric_xy_mm"] == [91., 1110.]
    assert saved["measurement_uncertainty_mm"] == 2.5
    observation = saved["runs"]["source"]["observations"][region["id"]]
    assert observation["status"] == "manual" and observation["anchor_u_xy"] == [1., 41.]
    assert observation["region_identity"] == region["identity"]


def test_run_combo_previews_target_and_find_this_run_keeps_global_map_reference(context, qtbot, monkeypatch):
    define(context)
    page, panel, rig = context.page, context.ui, context.rig
    reference = page.runs[0]
    monkeypatch.setattr(rig, "_reference", lambda: reference)
    page.static_reference_current_run = lambda: next(run for run in page.runs if run.run_id == page.current_run)
    shown, read = [], []

    def show(run_id):
        shown.append(run_id)
        page.current_run = run_id
        page.canvas.preview = packet(run_id)
        return True

    def reader(target):
        read.append(target.run_id)
        return packet(target.run_id)

    page.show_static_reference_run = show
    page.read_static_reference_packet = reader
    panel.run_combo.setCurrentIndex(panel.run_combo.findData("run3"))
    assert shown == ["run3"] and panel.run_combo.currentData() == "run3"
    assert rig.reference() is reference
    assert panel.find_regions(False)
    assert not panel.run_combo.isEnabled()
    qtbot.waitUntil(lambda: not panel.busy(), timeout=5000)
    assert read == ["run3"] and panel.run_combo.isEnabled()
    assert panel.document["runs"]["run3"]["run_id"] == "run3"
    assert panel.table.item(0, 4).text() == "recognized"


def test_target_rig_frame_uses_host_api_without_overwriting_reference_corners(context, monkeypatch):
    panel, page, rig = context.ui, context.page, context.rig
    for number in range(4):
        define(context, number*20., number*10.)
    reference = page.runs[0]
    monkeypatch.setattr(rig, "_reference", lambda: reference)
    page.current_run = "run4"
    page.canvas.preview = packet("run4")
    page.static_reference_current_run = lambda: page.runs[3]
    result = deepcopy(panel.document["runs"]["run1"])
    result.update(run_id="run4", photo=context.backend.photo_binding(page.canvas.preview))
    panel.document["runs"]["run4"] = result
    panel.context_changed()
    rig._loading = True
    rig.width_m.setValue(.09)
    rig.height_m.setValue(.05)
    rig._loading = False
    before = rig.reference_photo_corners()
    calls = []
    page.use_static_reference_frame = lambda *args: calls.append(args) or True
    monkeypatch.setattr(rig, "_store_frame", lambda: pytest.fail("Global reference frame was overwritten"))
    assert panel.use_for_rig_frame()
    assert len(calls) == 1 and calls[0][0] == "run4" and calls[0][2:] == (.09, .05)
    assert calls[0][1] == [(50., -10.), (50., 80.), (100., 80.), (100., -10.)]
    assert rig.reference_photo_corners() == before
    assert rig.reference() is reference
