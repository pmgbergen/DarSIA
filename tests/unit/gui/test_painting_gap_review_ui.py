"""N434 GUI behavior with finite synthetic readers; no real project or RAW."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QObject, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import QLabel, QMessageBox, QWidget

from darsia.fluidflower import painting_gap_review_ui as gaps
from darsia.fluidflower import painting_series_adoption_ui as action
from darsia.fluidflower import painting_series_use_ui as status
from darsia.fluidflower.painting_frame_models import FrameModelSelection

from .test_painting_corrections import view  # noqa: F401


def _review():
    version = dict(folder="C:/synthetic/versions/v1", sha256="a" * 64,
                   name="Synthetic", kind="manual", run_id="ac17", approval={"version": "v1"})
    return dict(format="darsia-painted-gap-review-v1", map_version=version,
        geometry_index=dict(path="C:/synthetic/index.json", sha256="b" * 64, size_bytes=101),
        source=dict(reference_run="ac17", painting_facies={"synthetic": "source"}),
        runs=[dict(run_id=run, gap=dict(pixels=count), review_row_sha256=digit * 64,
            components=([dict(component_id=digit*64, pixels=count, identity={"sha256":digit*64},
                        review_row_sha256=digit*64)] if count else []))
              for run, count, digit in (("ac18", 9, "c"), ("ac61", 25, "d"), ("ac17", 0, "e"))])


def _packet(run):
    mask = np.zeros((32, 40), bool)
    if run == "ac18":
        mask[1:4, 4:7] = True
    elif run == "ac61":
        mask[3:28, 39] = True  # The one-column, 25-pixel edge case remains visible.
    return dict(run_id=run, image_rgb=np.full((64, 64, 3), 30, np.uint8), image_encoding="rgb_uint8",
        gap_mask=mask, layer_boundary_mask=np.eye(32,40,dtype=bool),
        painting_shape_hw=[64,64], painting_frame_xy=[[0,0],[63,0],[63,63],[0,63]], native_to_photo_h=np.array([[1, 0, 3], [0, 1, 5], [0, 0, 1.]]),
        own_frame_native_xy=[[0, 0], [39, 0], [39, 31], [0, 31]],
        transported_frame_native_xy=[[1, 2], [38, 2], [38, 30], [1, 30]],
        native_to_painting_h=np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1.]]),
        painting_run_id="ac17", painting_photograph=dict(path="C:/synthetic/source.ARW", sha256="e"*64, size_bytes=99),
        map_version=deepcopy(_review()["map_version"]))


class _Backend:
    def __init__(self):
        self.reads, self.approvals = [], []
        self.packets = {run: _packet(run) for run in ("ac18", "ac61", "ac17")}

    def read(self, value):
        self.reads.append(("review", threading.current_thread() is threading.main_thread()))
        return deepcopy(value)

    def preview(self, review, run, component_id=None):
        self.reads.append((run, threading.current_thread() is threading.main_thread()))
        return dict(self.packets[run], component_id=component_id)

    def approve(self, review, selected, **options):
        # The worker module tests scientific identities; this spy isolates the
        # human event boundary and preserves previous action values exactly.
        self.approvals.append((deepcopy(review), list(selected), deepcopy(options)))
        rows = {(row["run_id"],row["component_id"]):deepcopy(row) for row in (options["previous"] or {}).get("runs", [])}
        wanted = {(row["run_id"],row["component_id"]) for row in selected}
        for run in review["runs"]:
            for row in run["components"]:
                key=(run["run_id"],row["component_id"])
                if key in wanted:
                    rows[key] = dict(run_id=key[0],component_id=key[1],review_row_sha256=row["review_row_sha256"],
                        **{name:options[name] for name in ("actor","approved_utc","reason","decision_ref","classification","other_text")})
        return dict(format="darsia-painted-gap-exclusion-approval-v1",
            runs=[rows[(run["run_id"],row["component_id"])] for run in review["runs"]
                  for row in run["components"] if (run["run_id"],row["component_id"]) in rows])


def _dialog(qtbot, *, backend=None, source=None, **options):
    backend = backend or _Backend()
    widget = gaps.GapReviewDialog(source or _review(), actor="Synthetic reviewer",
        review_reader=backend.read, preview_reader=backend.preview, approval_builder=backend.approve, **options)
    qtbot.addWidget(widget, before_close_func=lambda owned: owned.wait_for_workers())
    widget.show()
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    return widget, backend


@pytest.mark.parametrize("encoding", ["rgb_uint8", "rgb_float01"])
def test_photo_is_original_with_unfilled_transformed_contours_and_display_only_controls(qtbot, encoding):
    backend = _Backend()
    if encoding == "rgb_float01":
        packet = backend.packets["ac18"]
        packet["image_rgb"] = packet["image_rgb"].astype(np.float32) / 255.
        packet["image_encoding"] = encoding
    widget, backend = _dialog(qtbot, backend=backend)
    assert [widget.runs.item(row, 0).text() for row in range(2)] == ["AC18", "AC61"]
    assert widget.runs.item(1, 2).text() == "25"
    assert backend.reads == [("review", False), ("ac18", False)]
    assert len(widget._overlays) == 1
    contour = widget._overlays[0]
    assert contour.brush().style() == Qt.BrushStyle.NoBrush
    assert contour.pen().isCosmetic()
    assert contour.path().boundingRect().left() == 7  # native x=4 + declared photo translation 3
    picture = widget.photo.item.pixmap().toImage()
    assert picture.pixelColor(8, 7).red() == 30  # Gap interior is still the original photograph.
    original = backend.packets["ac18"]["image_rgb"].copy()
    original_gap = backend.packets["ac18"]["gap_mask"].copy()
    widget.own_frame.setChecked(True)
    widget.transported_frame.setChecked(True)
    assert len(widget._overlays) == 3
    assert all(item.brush().style() == Qt.BrushStyle.NoBrush for item in widget._overlays)
    widget.outline.setChecked(False)
    assert len(widget._overlays) == 2
    transform = widget.photo.transform()
    center = widget.photo.mapToScene(widget.photo.viewport().rect().center())
    candidate, report = object(), object()
    widget._candidate, widget._report = candidate, report
    assert widget.brighten.value() == 100
    widget.brighten.setValue(150)
    assert widget.photo.item.pixmap().toImage().pixelColor(8, 7).red() == 45
    widget.brighten.setValue(200)
    assert widget.photo.item.pixmap().toImage().pixelColor(8, 7).red() == 60
    assert widget.brightness_value.text() == "200%"
    assert widget.photo.transform() == transform
    after = widget.photo.mapToScene(widget.photo.viewport().rect().center())
    assert abs(after.x() - center.x()) < 1 and abs(after.y() - center.y()) < 1
    assert widget._candidate is candidate and widget._report is report
    assert not widget._display_preferences_dirty and not widget._preparation_changed
    widget.brighten.setValue(100)
    assert widget.photo.item.pixmap().toImage().pixelColor(8, 7).red() == 30
    np.testing.assert_array_equal(backend.packets["ac18"]["image_rgb"], original)
    np.testing.assert_array_equal(backend.packets["ac18"]["gap_mask"], original_gap)
    assert backend.approvals == [] and widget.approval is None
    assert not widget.recheck.isEnabled()


def test_series_decision_reference_is_written_with_each_approval(qtbot):
    widget, backend = _dialog(qtbot, decision_ref="series-gap-review")
    widget.classification.setCurrentIndex(1)
    widget.approve_one.click()
    assert backend.approvals[0][2]["decision_ref"] == "series-gap-review"
    assert _dialog(qtbot)[0].decision_ref == gaps.FINDINGS_DECISION_REF

def test_thin_edge_gap_is_listed_and_outlined_without_fill(qtbot):
    widget, backend = _dialog(qtbot)
    widget.runs.setCurrentCell(1, 0)
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    assert widget.packet["run_id"] == "ac61"
    assert np.count_nonzero(widget.packet["gap_mask"]) == 25
    assert widget._overlays[0].brush().style() == Qt.BrushStyle.NoBrush
    assert widget._overlays[0].path().boundingRect().left() == 42
    assert widget._overlays[0].path().boundingRect().height() == 24
    assert backend.approvals == []


def test_only_explicit_one_all_clicks_record_reason_actor_time_and_exact_selection(qtbot):
    widget, backend = _dialog(qtbot)
    approvals, rechecks = [], []
    widget.approval_changed.connect(approvals.append)
    widget.recheck_requested.connect(rechecks.append)
    assert widget.reason.text() == "water and air above the sand"
    widget.reason.setText("  reviewed water/air edge  ")
    widget.classification.setCurrentIndex(1)
    widget.approve_one.click()
    assert backend.approvals[0][1] == [dict(run_id="ac18",component_id="c"*64)]
    options = backend.approvals[0][2]
    assert options["actor"] == "Synthetic reviewer" and options["approved_utc"]
    assert options["reason"] == "reviewed water/air edge" and options["decision_ref"] == gaps.FINDINGS_DECISION_REF
    assert approvals[0]["runs"][0]["review_row_sha256"] == "c" * 64
    first_action = deepcopy(widget.approval["runs"][0])
    widget.runs.setCurrentCell(1, 0)
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    widget.reason.setText("separate edge reason")
    widget.classification.setCurrentIndex(1)
    widget.approve_one.click()
    assert widget.approval["runs"][0] == first_action
    assert widget.approval["runs"][1]["reason"] == "separate edge reason"
    assert rechecks == []  # Approval and navigating never start Prepare.
    widget.reason.setText("explicitly all listed runs")
    widget.classification.setCurrentIndex(1)
    widget.approve_all.click()
    assert backend.approvals[-1][1] == [dict(run_id="ac18",component_id="c"*64),dict(run_id="ac61",component_id="d"*64)]
    assert all(row["reason"] == "explicitly all listed runs" for row in widget.approval["runs"])
    _human_state(qtbot, widget)
    widget.recheck.click()
    assert rechecks == [widget.approval]
    rechecks[0]["runs"][0]["reason"] = "mutable receiver"
    assert widget.approval["runs"][0]["reason"] == "explicitly all listed runs"


def test_empty_reason_or_missing_actor_cannot_approve(qtbot):
    widget, backend = _dialog(qtbot)
    widget.reason.setText(" ")
    assert not widget.approve_one.isEnabled() and not widget.approve_all.isEnabled()
    widget._approve(False)
    widget.reason.setText("explicit reason")
    widget.actor = ""
    widget._buttons()
    widget._approve(True)
    assert not widget.approve_all.isEnabled() and backend.approvals == []


def test_caller_metadata_snapshot_and_paint_location_use_exact_declared_transforms(qtbot):
    source = _review()
    widget, backend = _dialog(qtbot, source=source)
    source["runs"][0]["components"][0]["review_row_sha256"] = "f" * 64
    locations = []
    widget.paint_requested.connect(locations.append)
    widget.paint.click()
    assert locations[0]["run_id"] == "ac18"
    assert locations[0]["component_id"] == "c"*64
    assert locations[0]["point_native_xy"] == [4., 1.]
    widget._point_selected((8., 7.))  # explicit photo inverse maps to native (x5,y2)
    widget.paint.click()
    assert locations[1]["point_native_xy"] == [5., 2.]
    widget._point_selected((60., 60.))  # Not a gap: no changed selection or approval.
    widget.paint.click()
    assert locations[2] == locations[1]
    widget.classification.setCurrentIndex(1)
    widget.approve_one.click()
    assert backend.approvals[0][0]["runs"][0]["components"][0]["review_row_sha256"] == "c" * 64


@pytest.mark.parametrize("failure", ["reader", "transform", "encoding"])
def test_failed_preview_never_leaves_a_stale_photo_or_approval_enabled(qtbot, failure):
    backend = _Backend()
    if failure == "transform":
        backend.packets["ac61"]["native_to_photo_h"] = np.zeros((3, 3))
    elif failure == "encoding":
        backend.packets["ac61"]["image_encoding"] = "guess_from_shape"
    else:
        original = backend.preview
        def refuse(review, run, component_id=None):
            if run == "ac61":
                raise ValueError("bound photograph changed")
            return original(review, run, component_id)
        backend.preview = refuse
    widget, _ = _dialog(qtbot, backend=backend)
    widget.runs.setCurrentCell(1, 0)
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    assert widget.packet is None and widget.photo.item.pixmap().isNull()
    assert not widget.approve_one.isEnabled() and not widget.approve_all.isEnabled()
    assert not widget.paint.isEnabled() and backend.approvals == []


def test_close_retires_finite_reader_without_loading_queued_run_or_publishing_stale_result(qtbot):
    backend = _Backend()
    widget, _ = _dialog(qtbot, backend=backend)
    entered, release = threading.Event(), threading.Event()
    original = widget._read_preview
    def finite(review, run, component_id=None):
        entered.set()
        assert release.wait(5), "The finite private read must be released"
        return original(review, run, component_id)
    widget._read_preview = finite
    widget.runs.setCurrentCell(1, 0)
    qtbot.waitUntil(entered.is_set, timeout=2000)
    widget.runs.setCurrentCell(0, 0)
    ticks=[]
    try:
        widget.close()
        assert widget.busy() and widget.isVisible() and widget._closing
        assert widget._pending is None
        QTimer.singleShot(0,lambda:ticks.append("GUI remained responsive"))
        qtbot.waitUntil(lambda:bool(ticks),timeout=2000)
        assert widget.busy() and not release.is_set()
        release.set()
        qtbot.waitUntil(lambda:not widget.busy() and not widget.isVisible(),timeout=5000)
        assert widget._pending is None and widget.packet is None
        assert backend.reads == [("review", False), ("ac18", False), ("ac61", False)]
        assert backend.approvals == []
    finally:
        release.set()
        qtbot.waitUntil(lambda:not widget.busy(),timeout=5000)


class _Host(QWidget):
    def __init__(self):
        super().__init__()
        review = _review()
        self.record = deepcopy(review["map_version"])
        self.project = SimpleNamespace(root=Path("C:/synthetic/project"))
        self.approval = deepcopy(self.record["approval"])
        self.frame = SimpleNamespace(to_dict=lambda: {"frame": "unchanged"})
        self.selection = FrameModelSelection(review["geometry_index"]["path"], "synthetic", str(self.project.root),
            "synthetic", "C:/synthetic", "synthetic", "b"*64, 101)
        self.run_preview = SimpleNamespace(geometry_selection=lambda: self.selection)
        self.use_map_progress = QLabel(self)
        self.versions_note = QLabel(self)
        self._adoption_dialog = self._adoption_thread = None
        self.calls = []
        self.adoption_controller = action.SeriesAdoptionController(self)

    def busy(self):
        return False

    def _read_approvals(self, run):
        pass

    def _approval_for_version(self, record):
        return self.approval

    def _selected_version(self):
        return self.record

    def current_frame(self):
        return self.frame

    def _reviewer(self):
        return "Synthetic reviewer"

    def prepare_adoption_of(self, record, path, **options):
        self.calls.append((deepcopy(record), str(path), deepcopy(options)))
        return True


def _controller(qtbot, monkeypatch):
    host, backend = _Host(), _Backend()
    qtbot.addWidget(host, before_close_func=lambda owned: owned.adoption_controller.wait_for_workers())
    original = gaps.GapReviewDialog
    def factory(source, **options):
        return original(source, review_reader=backend.read, preview_reader=backend.preview,
            approval_builder=backend.approve, **options)
    monkeypatch.setattr(gaps, "GapReviewDialog", factory)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("A gap review cannot ask the final adoption question")
    monkeypatch.setattr(QMessageBox, "question", forbidden)
    return host, backend, host.adoption_controller


def test_typed_failure_waits_retirement_and_only_human_recheck_forwards_snapshotted_approval(qtbot, monkeypatch):
    host, backend, controller = _controller(qtbot, monkeypatch)
    from .test_findings_before_check import _literal_review
    review = _literal_review()  # Finite UI DATA; not a scientifically accepted review.
    requests = []
    controller.catalogue_refresher = requests.append
    assert controller.start(host.record, host.selection)
    assert "analysis_exclusions" not in host.calls[0][2]  # Original absent argument remains absent.
    controller.gap_review_ready(review)
    controller.failed("original interior gap refusal")
    assert controller.gap_dialog is None
    controller.preparation_retired()
    dialog = controller.gap_dialog
    qtbot.waitUntil(lambda: not dialog.busy(), timeout=5000)
    assert controller.dialog is None and controller.preparation is None and len(host.calls) == 1
    dialog.classification.setCurrentIndex(1)
    dialog.approve_all.click()
    assert len(host.calls) == 1
    dialog.recheck.click()
    assert len(host.calls) == 1  # N461 also requires the human source-water answer.
    _human_state(qtbot, dialog)
    qtbot.waitUntil(lambda: bool(requests) and not dialog.busy(), timeout=5000)
    request = requests[-1]
    assert controller.install_findings_result(dict(generation=request['generation'],
        review=deepcopy(dialog.review), report=dict(issues=[],
            input_bindings=deepcopy(request['input_bindings']))))
    qtbot.waitUntil(lambda: not dialog.busy(), timeout=5000)
    expected = deepcopy(dialog.approval)
    dialog.recheck.click()
    assert len(host.calls) == 2
    assert host.calls[1][2]["analysis_exclusions"] == expected
    dialog.approval["runs"][0]["reason"] = "later mutation"
    assert host.calls[1][2]["analysis_exclusions"] == expected
    assert controller.dialog is None  # No invented Check or prepared index.


@pytest.mark.parametrize("change", ["version", "approval", "geometry"])
def test_changed_context_after_human_choice_does_not_start_fresh_prepare(qtbot, monkeypatch, change):
    host, backend, controller = _controller(qtbot, monkeypatch)
    dialog = controller.open_gap_review(_review())
    qtbot.waitUntil(lambda: not dialog.busy(), timeout=5000)
    dialog.classification.setCurrentIndex(1)
    dialog.approve_all.click()
    if change == "version":
        host.record["folder"] += "-changed"
    elif change == "approval":
        host.approval = {"version": "changed"}
    else:
        host.selection = replace(host.selection, sha256="f"*64)
    controller._recheck_gaps(dialog.approval)  # Direct calls retain the current-context guard.
    assert host.calls == []
    assert "current project, map, approval and geometry" in host.use_map_progress.text()


def test_other_version_review_is_refused_before_human_actions(qtbot, monkeypatch):
    host, backend, controller = _controller(qtbot, monkeypatch)
    review = _review()
    review["map_version"]["folder"] += "-another"
    dialog = controller.open_gap_review(review)
    qtbot.waitUntil(lambda: not dialog.busy(), timeout=5000)
    assert dialog.review is None and not dialog.approve_all.isEnabled()
    assert "another selected painting" in dialog.status.text()
    assert backend.approvals == [] and host.calls == []


def test_step5_lists_existing_v2_preparation_without_admitting_or_reading_arrays(tmp_path):
    import json
    root = tmp_path / "annotations"
    folder = root / "map_adoption" / "existing-v2"
    folder.mkdir(parents=True)
    value = dict(format="darsia-fluidflower-painted-map-adoption-preparation-v2", created_utc="2026-01-01T00:00:00Z",
                 map_version=_review()["map_version"], runs=[dict(run_id="ac18", status="prepared")])
    path = folder / "index.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    before = path.read_bytes()
    result = status.existing_progress(root, tmp_path / "absent-current.json", value["map_version"])
    assert result["preparation"]["format"] == value["format"]
    assert result["adopted"] is None and path.read_bytes() == before


def test_ordinary_view_snapshots_optional_payload_and_receives_typed_review_on_gui_thread(view, qtbot, monkeypatch):
    from darsia.fluidflower import series_layer_view as editor
    from .test_painting_manual_step4_route import _saved_manual
    record = _saved_manual(view, qtbot)
    assert view.approve_version(record) is not None
    entered, release = threading.Event(), threading.Event()
    observed, delivered = [], []

    class Finite(QObject):
        progress = Signal(str)
        finished = Signal(object)
        failed = Signal(str)
        gap_review_ready = Signal(object)

        def __init__(self, record, frame, options):
            super().__init__()
            self.options = options

        def run(self):
            entered.set()
            assert release.wait(5), "The finite private worker must be released"
            observed.append(deepcopy(self.options["analysis_exclusions"]))
            self.gap_review_ready.emit({"synthetic": "typed review"})
            self.failed.emit("Synthetic preparation refusal after typed review")

    monkeypatch.setattr(editor, "_AdoptionWorker", Finite)
    class Probe(QObject):
        @Slot(object)
        def receive(self, value):
            delivered.append((value, threading.current_thread() is threading.main_thread()))
    probe = Probe(view)
    monkeypatch.setattr(view.adoption_controller, "gap_review_ready", probe.receive)
    payload = {"runs": [{"run_id": "ac18", "reason": "explicit private action"}]}
    expected = deepcopy(payload)
    try:
        assert view.prepare_adoption_of(record, "C:/synthetic/index.json", analysis_exclusions=payload)
        qtbot.waitUntil(entered.is_set, timeout=2000)
        payload["runs"][0]["reason"] = "mutable caller"
        release.set()
        qtbot.waitUntil(lambda: view._adoption_thread is None, timeout=5000)
        assert observed == [expected]
        assert delivered == [({"synthetic": "typed review"}, True)]
        assert "Synthetic preparation refusal" in view.use_map_progress.text()
    finally:
        release.set()
        view.wait_for_workers()


def _select_kind(qtbot, widget, kind, occurrence=0):
    rows = [index for index, row in enumerate(widget._rows) if row["kind"] == kind]
    widget.runs.setCurrentCell(rows[occurrence], 0)
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)


def _report(widget, issues):
    widget.set_check_result(dict(painted_adoption_table={"synthetic": True}),
        dict(config_sha256="f"*64, issues=deepcopy(issues)))


def _human_state(qtbot, widget):
    _select_kind(qtbot, widget, "source_state")
    assert widget.source_state.currentData() is None
    widget.reason.setText("I reviewed the source photograph myself")
    widget.source_state.setCurrentIndex(widget.source_state.findData("indicator"))
    widget.set_state.click()


def test_classes_are_not_preselected_and_other_requires_explicit_free_text(qtbot):
    widget, backend = _dialog(qtbot)
    assert widget.classification.currentData() is None
    assert not widget.approve_one.isEnabled()
    widget.approve_one.click()
    assert backend.approvals == []
    widget.classification.setCurrentIndex(widget.classification.findData("other"))
    assert not widget.approve_one.isEnabled()
    widget.other_text.setText("reviewed non-sand fixture")
    widget.approve_one.click()
    options = backend.approvals[-1][2]
    assert options["classification"] == "other" and options["other_text"] == "reviewed non-sand fixture"


def test_two_connected_areas_in_one_run_have_separate_rows_and_human_actions(qtbot):
    source, backend = _review(), _Backend()
    source["runs"][0]["components"].append(dict(component_id="f"*64, pixels=1,
        identity={"sha256":"f"*64}, review_row_sha256="f"*64))
    original = backend.preview
    def preview(review, run, component_id=None):
        packet = original(review, run, component_id)
        if component_id == "f"*64:
            packet = dict(packet, gap_mask=np.zeros((32,40),bool))
            packet["gap_mask"][8,8] = True
        return packet
    backend.preview = preview
    widget, _ = _dialog(qtbot, backend=backend, source=source)
    assert [row["run_id"] for row in widget._rows if row["kind"] == "gap"] == ["ac18","ac18","ac61"]
    widget.runs.setCurrentCell(1,0)
    qtbot.waitUntil(lambda:not widget.busy(),timeout=5000)
    assert np.count_nonzero(widget.packet["gap_mask"]) == 1
    widget.classification.setCurrentIndex(widget.classification.findData("air"))
    widget.approve_one.click()
    assert backend.approvals[-1][1] == [dict(run_id="ac18",component_id="f"*64)]
    assert not widget._resolved(widget._rows[0]) and widget._resolved(widget._rows[1])


def test_source_state_is_a_human_choice_and_each_duplicate_warning_needs_its_own_reason(qtbot):
    widget, backend = _dialog(qtbot)
    widget.classification.setCurrentIndex(1)
    widget.approve_all.click()
    warning = dict(severity="warning",code="indicator_facies_carryover_risk",message="Original indicator warning",
                   details={"run_id":"ac18"})
    _report(widget,[warning,warning])
    assert widget.source_state.currentData() is None and not widget.adopt.isEnabled()
    choices, adopted = [], []
    widget.state_requested.connect(choices.append)
    widget.adopt_requested.connect(adopted.append)
    _human_state(qtbot,widget)
    assert choices[0]["acquisition_state"] == "indicator" and choices[0]["actor"] == "Synthetic reviewer"
    assert len(widget._acquisition_actions) == 1 and widget._candidate is None
    _report(widget,[warning,warning])  # A genuine worker Check must finish after the choice.
    _select_kind(qtbot,widget,"warning",0)
    widget.reason.setText("I accept the first original warning")
    widget.approve_warning.click()
    assert not widget.adopt.isEnabled()
    _select_kind(qtbot,widget,"warning",1)
    widget.reason.setText("I accept the second original warning")
    widget.approve_warning.click()
    assert widget.adopt.isEnabled()
    assert widget._report["issues"] == [warning,warning]  # Nothing is hidden or downgraded.
    widget.adopt.click()
    assert [row["issue_index"] for row in adopted[0]["warnings"]] == [0,1]
    assert adopted[0]["acquisition"] == choices
    adopted[0]["warnings"][0]["reason"]="mutable recipient"
    assert next(iter(widget._warning_actions.values()))["reason"] != "mutable recipient"


def test_errors_cannot_be_approved_and_unknown_state_cannot_be_approved_instead_of_chosen(qtbot):
    widget,_ = _dialog(qtbot)
    unknown=dict(severity="warning",code="run_geometry_acquisition_unknown",message="Choose the state",details={"run_id":"ac18"})
    error=dict(severity="error",code="invalid_width",message="Original error",details={})
    _report(widget,[unknown,error])
    _select_kind(qtbot,widget,"warning")
    widget.reason.setText("An acknowledgement is not a state choice")
    assert not widget.approve_warning.isEnabled()
    widget._approve_warning()
    assert widget._warning_actions == {}
    _select_kind(qtbot,widget,"error")
    widget.reason.setText("Errors cannot be accepted")
    assert not widget.approve_warning.isEnabled()
    widget._approve_warning()
    assert widget._warning_actions == {} and not widget.adopt.isEnabled()
    assert "Original error" in widget.details.toPlainText()


def test_unchanged_warning_resolution_survives_index_shift_but_changed_evidence_reopens_it(qtbot):
    widget,_ = _dialog(qtbot)
    warning=dict(severity="warning",code="source_warning",message="Original cause",details={"run_id":"ac18","evidence":"same"})
    _report(widget,[warning])
    _select_kind(qtbot,widget,"warning")
    widget.reason.setText("I reviewed this exact evidence")
    widget.approve_warning.click()
    before=deepcopy(next(iter(widget._warning_actions.values())))
    info=dict(severity="info",code="frame_measurement",message="Bound frame distance",details={"max_edge_distance_mm":12.5})
    _report(widget,[info,warning])
    action=next(iter(widget._warning_actions.values()))
    assert action["issue_index"]==1 and action["approved_utc"]==before["approved_utc"]
    changed=deepcopy(warning);changed["details"]["evidence"]="changed"
    _report(widget,[info,changed])
    assert widget._warning_actions=={} and not widget.adopt.isEnabled()


def test_unbound_comparison_options_are_disabled_without_guessing_or_starting_another_read(qtbot):
    widget,backend=_dialog(qtbot)
    assert widget.images.count()==4
    assert all(not widget.images.model().item(index).isEnabled() for index in range(1,4))
    widget.image_shortcut.activated.emit()
    assert widget.images.currentData()=="current"
    assert backend.reads==[("review",False),("ac18",False)]
    assert "no verified placement binding" in widget.position.text()
    assert not widget.apply_tolerances.isEnabled()


@pytest.mark.parametrize("same_grid",[True,False])
def test_bound_image_toggle_preserves_view_or_refuses_a_different_common_grid(qtbot,same_grid):
    backend=_Backend();base=backend.packets["ac18"]
    base.update(display_grid={"synthetic":"exact common plane"},
        image_choices=[dict(image_id="clearwater",label="Synthetic verified clearwater",available=True)],
        position_readout=dict(method="camera registration (map placement)",reference=dict(shift_mm=10.,rotation_degrees=.1),previous=dict(shift_mm=2.,rotation_degrees=.03)),
        position_tolerances=dict(shift_mm=3.,rotation_degrees=.2))
    calls=[]
    def comparison(review,run,component,image_id):
        calls.append((run,component,image_id,threading.current_thread() is threading.main_thread()))
        return dict(base,image_rgb=np.full((64,64,3),80,np.uint8),
                    display_grid=base["display_grid"] if same_grid else {"synthetic":"different plane"})
    widget,_=_dialog(qtbot,backend=backend,comparison_reader=comparison)
    widget.photo.scale(2.,2.);widget.photo.centerOn(15.,15.)
    transform=widget.photo.transform()
    center=widget.photo.mapToScene(widget.photo.viewport().rect().center())
    assert "10.000 mm" in widget.position.text() and "2.000 mm" in widget.position.text()
    widget.image_shortcut.activated.emit()
    qtbot.waitUntil(lambda:not widget.busy(),timeout=5000)
    assert calls==[("ac18","c"*64,"clearwater",False)]
    assert widget.photo.transform()==transform
    after=widget.photo.mapToScene(widget.photo.viewport().rect().center())
    assert abs(after.x()-center.x())<1 and abs(after.y()-center.y())<1
    assert np.all(base["image_rgb"]==30) and widget.approval is None
    if same_grid:
        assert widget.photo.item.pixmap().toImage().pixelColor(8,7).red()==80
        assert widget.images.currentData()=="clearwater"
    else:
        assert widget.photo.item.pixmap().toImage().pixelColor(8,7).red()==30
        assert widget.images.currentData()=="current" and "exact common run display grid" in widget.status.text()


@pytest.mark.parametrize("valid_support", [True, False])
def test_comparison_support_is_display_only_or_refused_when_not_exact_boolean_grid(qtbot, valid_support):
    backend = _Backend()
    base = backend.packets["ac18"]
    base.update(display_grid={"synthetic": "same native grid"},
        image_choices=[dict(image_id="previous-run", available=True)])
    original = np.full((64, 64, 3), 80, np.uint8)
    support = np.ones((64, 64), bool)
    support[:8, :8] = False
    def comparison(*_args):
        return dict(base, image_rgb=original,
            valid_support=support if valid_support else support.astype(np.uint8))
    widget, _ = _dialog(qtbot, backend=backend, comparison_reader=comparison)
    widget.images.setCurrentIndex(widget.images.findData("previous-run"))
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    picture = widget.photo.item.pixmap().toImage()
    if valid_support:
        assert picture.pixelColor(2, 2).red() == 64 and picture.pixelColor(10, 10).red() == 80
        assert "display only" in widget.status.text()
    else:
        assert widget.images.currentData() == "current" and picture.pixelColor(2, 2).red() == 30
        assert "exact boolean display support" in widget.status.text()
    assert np.all(original == 80) and backend.approvals == []


def test_reopening_review_restores_human_water_tolerances_without_confirming_a_line(qtbot, monkeypatch):
    host, backend, controller = _controller(qtbot, monkeypatch)
    first = controller.open_gap_review(_review())
    qtbot.waitUntil(lambda: not first.busy(), timeout=5000)
    first.water_surface.tolerances_changed.emit(dict(height_tolerance_mm=5.5, angle_tolerance_degrees=.75))
    first.close()
    qtbot.waitUntil(lambda: not first.isVisible() and not first.busy(), timeout=5000)
    reopened = controller.open_gap_review(_review())
    qtbot.waitUntil(lambda: not reopened.busy(), timeout=5000)
    assert reopened.water_surface.height_tolerance.value() == 5.5
    assert reopened.water_surface.angle_tolerance.value() == .75
    assert controller._water_surface_confirmations == {} and backend.approvals == [] and host.calls == []


@pytest.mark.parametrize("change", ["version", "geometry", "approval", "binding", "suspended"])
def test_existing_review_reveal_requires_exact_live_context(qtbot, monkeypatch, change):
    host, backend, controller = _controller(qtbot, monkeypatch)
    review = _review()
    review["review_binding"] = dict(path="C:/synthetic/gap_review.json",
                                    sha256="f" * 64, size_bytes=1234)
    findings = controller.open_gap_review(review)
    qtbot.waitUntil(lambda: not findings.busy(), timeout=5000)
    assert controller.current_gap_review() is findings
    findings.hide()
    if change == "version":
        host.record = dict(host.record, folder="C:/synthetic/another-version")
    elif change == "geometry":
        host.selection = replace(host.selection, sha256="c" * 64)
    elif change == "approval":
        host.approval = dict(host.approval, approved_utc="2026-01-02T00:00:00Z")
    elif change == "binding":
        findings.review["review_binding"]["sha256"] = "e" * 64
    else:
        controller._suspended = True
    before = list(backend.reads)
    assert controller.current_gap_review() is None
    assert not findings.isVisible() and backend.reads == before
    assert backend.approvals == [] and host.calls == []


@pytest.mark.parametrize("has_records",[False,True])
def test_save_worker_preserves_legacy_call_shape_or_seals_immutable_human_records(qtbot,monkeypatch,has_records):
    from darsia.fluidflower import map_adoption_view as adoption_ui
    from darsia.fluidflower import geometry_facies as geometry
    from darsia.fluidflower import painted_analysis_exclusions as core
    calls=[]
    class Store:
        def __init__(self,project):
            pass
        def save_revision(self,candidate,**options):
            calls.append((deepcopy(candidate),deepcopy(options)))
            return {"revision_id":"synthetic"}
    def seal(candidate,report,records):
        assert candidate["nested"]["value"]==1 and records["warnings"][0]["reason"]=="human reason"
        return {"synthetic":"sealed","records":deepcopy(records)}
    monkeypatch.setattr(geometry,"GeometryRevisionStore",Store)
    monkeypatch.setattr(core,"seal_findings_resolutions",seal)
    candidate={"nested":{"value":1}}
    records={"warnings":[{"reason":"human reason"}],"acquisition":[],"analysis_exclusions":None}
    options={} if not has_records else dict(validation_report={"issues":[]},review_resolutions=records)
    worker=adoption_ui._SaveRevisionWorker(object(),candidate,"human actor","reason",**options)
    candidate["nested"]["value"]=99;records["warnings"][0]["reason"]="mutable caller"
    result=[];worker.finished.connect(result.append)
    worker.run()
    assert result==[{"revision_id":"synthetic"}] and calls[0][0]["nested"]["value"]==1
    assert ("review_resolutions" in calls[0][1]) is has_records
    assert calls[0][1]["preserve_failed_artifacts"] is True
    if has_records:
        assert calls[0][1]["review_resolutions"]["records"]["warnings"][0]["reason"]=="human reason"


def test_check_worker_builds_state_declaration_only_from_snapshotted_explicit_gui_choice(qtbot,tmp_path,monkeypatch):
    from darsia.fluidflower import map_adoption_view as adoption_ui
    from darsia.fluidflower import geometry_facies as geometry
    from darsia.fluidflower import painted_analysis_exclusions as core
    helper_calls=[];candidate_calls=[]
    choice=dict(acquisition_state="indicator",actor="human",approved_utc="2026-01-01T00:00:00Z",reason="human inspected",decision_ref="N438")
    def declaration(folder,state,**action):
        helper_calls.append((folder,state,deepcopy(action)))
        return {"synthetic":"bound human declaration"}
    def candidate(current,folder,facies,**options):
        candidate_calls.append(deepcopy(options))
        return {"synthetic":"candidate"},{"synthetic":"summary"}
    monkeypatch.setattr(core,"make_facies_source_acquisition",declaration)
    monkeypatch.setattr(adoption_ui.ma,"adoption_candidate",candidate)
    monkeypatch.setattr(geometry,"validate_geometry_config",lambda value:{"issues":[]})
    worker=adoption_ui._CheckWorker({},tmp_path,None,facies_source_choice=choice)
    choice["acquisition_state"]="clearwater";choice["reason"]="mutable caller"
    results=[];worker.finished.connect(results.append);worker.run()
    assert helper_calls[0][0]==tmp_path and helper_calls[0][1]=="indicator"
    assert helper_calls[0][2]["reason"]=="human inspected"
    assert candidate_calls==[{"facies_source_acquisition":{"synthetic":"bound human declaration"}}]
    assert results[0]["facies_source_acquisition"]=={"synthetic":"bound human declaration"}
    legacy=adoption_ui._CheckWorker({},tmp_path,None);legacy.run()
    assert len(helper_calls)==1 and candidate_calls[-1]=={}


def test_changed_area_resolution_invalidates_checked_candidate_until_explicit_recheck(qtbot):
    widget,backend=_dialog(qtbot)
    widget.classification.setCurrentIndex(widget.classification.findData("water_above_sand"))
    widget.approve_all.click()
    source_row=next(i for i,row in enumerate(widget._rows) if row["kind"]=="source_state")
    widget.runs.setCurrentCell(source_row,0)
    qtbot.waitUntil(lambda:not widget.busy(),timeout=5000)
    widget.source_state.setCurrentIndex(widget.source_state.findData("unknown"))
    widget.reason.setText("The human inspected the source")
    widget.set_state.click()
    widget.set_check_result({"synthetic":"checked"},{"issues":[]})
    qtbot.waitUntil(lambda:not widget.busy(),timeout=5000)
    assert widget.adopt.isEnabled()
    widget.runs.setCurrentCell(0,0)
    qtbot.waitUntil(lambda:not widget.busy(),timeout=5000)
    widget.classification.setCurrentIndex(widget.classification.findData("air"))
    widget.reason.setText("The human changed the class and its explanation")
    rechecks=[];adoptions=[]
    widget.recheck_requested.connect(rechecks.append)
    widget.adopt_requested.connect(adoptions.append)
    widget.approve_one.click()
    assert widget._candidate is None and widget._report is None
    assert not widget.adopt.isEnabled() and widget.recheck.isEnabled()
    widget._adopt()
    assert adoptions==[]
    widget.recheck.click()
    assert len(rechecks)==1
    assert rechecks[0]["runs"][0]["classification"]=="air"
    assert rechecks[0]["runs"][0]["reason"]=="The human changed the class and its explanation"
    assert widget._report is None  # Clicking is not a synthetic clean Check.
