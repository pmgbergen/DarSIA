"""B3 real widgets with manually completed asset reads; prepared, unrun tests."""

import pytest
from PySide6.QtGui import QColor, QImage

from darsia.fluidflower import calibration_review_panels as review


class _Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in list(self.callbacks):
            callback(*args)


class _ManualRead:
    """No thread runs; tests choose the arrival order of actual Qt images."""

    def __init__(self, generation, result, path, index, requests, parent):
        self.generation, self.result, self.path = generation, result, path
        self.index, self.requests = index, requests
        self.ready, self.finished = _Signal(), _Signal()
        self.started = self.interruptions = self.waits = self.deleted = 0

    def start(self):
        self.started += 1

    def requestInterruption(self):
        self.interruptions += 1

    def wait(self):
        self.waits += 1

    def deleteLater(self):
        self.deleted += 1


@pytest.fixture
def panels(qtbot, monkeypatch):
    reads = []

    def reader(*args):
        worker = _ManualRead(*args)
        reads.append(worker)
        return worker

    monkeypatch.setattr(review, "_PanelRead", reader)
    widget = review.CalibrationReviewPanels()
    qtbot.addWidget(widget)
    widget.show()
    yield widget, reads
    widget.stop()


def _select(combo, value):
    index = combo.findData(value)
    assert index >= 0
    combo.setCurrentIndex(index)


def _result():
    return {"complete": True, "selected_inputs": {"candidate": {
        "dataset": "ffac", "run": "ac28", "candidate_id": "phase-a"}},
        "method": {"settings": {"photometry_variant_id": "none_cc_off"}},
        "frames": [{"frame_index": index, "elapsed_h": index / 2} for index in range(3)]}


def _image(red):
    image = QImage(8, 5, QImage.Format.Format_RGB888)
    image.fill(QColor(red, 37, 82))
    return image


def test_layouts_are_real_grid_positions_and_keep_independent_choices(panels):
    widget, reads = panels
    _select(widget.panels[0].content, "overflow_curve")
    _select(widget.panels[1].content, "photograph")
    choices = [panel.content.currentData() for panel in widget.panels]
    expected_layouts = [
        ("one", [(0, 0, 1, 2)]),
        ("side_by_side", [(0, 0, 1, 1), (0, 1, 1, 1)]),
        ("pyramid", [(0, 0, 1, 2), (1, 0, 1, 1), (1, 1, 1, 1)]),
    ]
    for key, positions in expected_layouts:
        _select(widget.layout_choice, key)
        assert widget.grid.count() == len(positions)
        for index, position in enumerate(positions):
            panel = widget.panels[index]
            assert widget.grid.getItemPosition(widget.grid.indexOf(panel)) == position
            assert not panel.isHidden()
        assert all(panel.isHidden() for panel in widget.panels[len(positions):])
        assert [panel.content.currentData() for panel in widget.panels] == choices
    assert reads == []


def test_light_and_colour_controls_follow_only_each_photograph_panel(panels):
    widget, _ = panels
    first, second, third = widget.panels
    assert [panel.photo_controls.isHidden() for panel in widget.panels] == [False, True, True]
    _select(second.content, "photograph")
    assert not first.photo_controls.isHidden() and not second.photo_controls.isHidden()
    assert third.photo_controls.isHidden()
    _select(first.content, "concentration")
    assert first.photo_controls.isHidden() and not second.photo_controls.isHidden()
    assert third.content.currentData() == "mass_curve"
    for content in ("mass_curve", "dilution_curve", "overflow_curve", "concentration"):
        _select(second.content, content)
        assert second.photo_controls.isHidden()


def test_mass_map_requires_explicit_fixed_units_and_keeps_photo_filters_hidden(panels, tmp_path):
    widget, reads = panels
    panel = widget.panels[0]
    _select(panel.content, "mass_total")
    assert panel.photo_controls.isHidden()
    assert not panel.mass_controls.isHidden()
    assert panel.mass_limit.text() == ""
    widget.set_result(_result(), tmp_path / "result.json", 0, [])
    widget.set_active(True)
    assert "fixed positive" in reads[0].requests[0]["error"]
    panel.mass_limit.setText("0.002")
    panel.mass_limit.editingFinished.emit()
    reads[0].finished.emit()
    assert reads[-1].requests[0]["mass_limit"] == .002
    assert reads[-1].requests[0]["error"] == ""
    _select(panel.content, "photograph")
    assert panel.mass_controls.isHidden() and not panel.photo_controls.isHidden()
    assert panel.mass_limit.text() == "0.002"


def test_photo_filter_does_not_replace_anchor_result_or_other_panel_content(panels, tmp_path):
    widget, reads = panels
    path, alternate_path = tmp_path / "anchor.json", tmp_path / "corrected.json"
    result, original = _result(), _result()
    entry = dict(path=str(alternate_path), dataset="ffac", run="ac28", model="phase-a",
                 created_utc="2026-10-23T00:00:00Z",
                 method={"photometry_variant_id": "facies_cc_on"})
    widget.set_result(result, path, 2, [entry])
    photograph, concentration, graph = widget.panels
    _select(photograph.light, "facies")
    _select(photograph.colour, "on")
    assert photograph.photograph.currentData() == str(alternate_path)
    assert concentration.content.currentData() == "concentration"
    assert graph.content.currentData() == "mass_curve"
    widget.set_active(True)
    request = reads[-1]
    assert request.path == path and request.result is result and request.index == 2
    assert request.requests[0]["photo_path"] == str(alternate_path)
    assert request.requests[0]["entry"] == entry
    assert [item["content"] for item in request.requests] == ["photograph", "concentration", "mass_curve"]
    assert all(item["entry"] is None for item in request.requests[1:])
    assert result == original and widget.path == path and widget.index == 2


def test_no_matching_photo_is_explicit_and_does_not_fall_back(panels, tmp_path):
    widget, reads = panels
    widget.set_result(_result(), tmp_path / "result.json", 0, [])
    _select(widget.panels[0].light, "facies")
    assert widget.panels[0].photograph.count() == 0
    assert not widget.panels[0].photograph.isEnabled()
    widget.set_active(True)
    requests = reads[-1].requests
    assert requests[0]["photo_path"] is None
    assert "No saved photograph matches" in requests[0]["error"]
    assert requests[1]["error"] == requests[2]["error"] == ""


def test_navigation_coalesces_to_latest_frame_and_discards_stale_images(panels, tmp_path):
    widget, reads = panels
    result, path = _result(), tmp_path / "result.json"
    widget.set_result(result, path, 0, [])
    widget.set_active(True)
    first = reads[0]
    assert first.started == 1
    widget.set_result(result, path, 1, [])
    widget.set_result(result, path, 2, [])
    assert len(reads) == 1 and first.interruptions == 2
    first.ready.emit(first.generation, [(_image(255), "")] * 3)
    assert all(panel.image._source_pixmap.isNull() for panel in widget.panels)
    assert all(panel.notice.text() == "Loading saved display…" for panel in widget.panels)
    first.finished.emit()
    assert first.deleted == 1 and len(reads) == 2
    latest = reads[1]
    assert latest.index == 2 and latest.generation == widget._generation
    latest.ready.emit(latest.generation, [(_image(17), "")] * 3)
    assert all(panel.image._source_pixmap.toImage().pixelColor(0, 0).red() == 17
               for panel in widget.panels)
    assert all(panel.notice.text() == "" for panel in widget.panels)


def test_current_read_failure_clears_old_image_without_hiding_sibling(panels, tmp_path):
    widget, reads = panels
    widget.set_result(_result(), tmp_path / "result.json", 0, [])
    widget.set_active(True)
    first = reads[-1]
    first.ready.emit(first.generation, [(_image(17), "")] * 3)
    first.finished.emit()
    _select(widget.panels[0].content, "overflow_curve")
    assert widget.panels[0].image._source_pixmap.isNull()
    current = reads[-1]
    current.ready.emit(current.generation,
                       [(QImage(), "Saved graph missing"), (_image(29), ""), (_image(31), "")])
    assert widget.panels[0].image._source_pixmap.isNull()
    assert widget.panels[0].notice.text() == "Unavailable: Saved graph missing"
    assert widget.panels[1].image._source_pixmap.toImage().pixelColor(0, 0).red() == 29
    assert widget.panels[2].image._source_pixmap.toImage().pixelColor(0, 0).red() == 31


def test_deactivation_ignores_queued_completion_and_does_not_launch_pending_read(panels, tmp_path):
    widget, reads = panels
    widget.set_result(_result(), tmp_path / "result.json", 0, [])
    widget.set_active(True)
    worker = reads[0]
    widget.set_active(False)
    worker.ready.emit(worker.generation, [(_image(255), "")] * 3)
    worker.finished.emit()
    assert len(reads) == 1 and widget._worker is None
    assert all(panel.image._source_pixmap.isNull() for panel in widget.panels)
    assert widget.isHidden()


def test_stop_interrupts_and_waits_without_starting_pending_navigation(panels, tmp_path):
    widget, reads = panels
    result = _result()
    widget.set_result(result, tmp_path / "result.json", 0, [])
    widget.set_active(True)
    worker = reads[0]
    widget.set_result(result, tmp_path / "result.json", 2, [])
    widget.stop()
    assert worker.interruptions == 2 and worker.waits == 1
    assert not widget._active and not widget._pending
    worker.finished.emit()
    assert len(reads) == 1 and widget._worker is None


def test_overflow_off_clears_its_panel_and_respects_the_mass_calibration_choice(panels, tmp_path):
    widget, reads = panels
    widget.set_result(_result(), tmp_path / "result.json", 0, [])
    _select(widget.panels[2].content, "overflow_curve")
    widget.set_overflow_enabled(False)
    widget.set_active(True)
    worker = reads[-1]
    assert "Overflow comparison is off" in worker.requests[2]["error"]
    for panel in widget.panels:
        assert not panel.content.model().item(panel.content.findData("overflow_curve")).isEnabled()
    worker.finished.emit()
    widget.set_overflow_enabled(True)
    assert reads[-1].requests[2]["error"] == ""
