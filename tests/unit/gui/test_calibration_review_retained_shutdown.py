"""Queued retained reader completions cannot revive a closed review workspace.

Deterministic manual workers use real queued Qt signals; no lookup is executed.
"""
from copy import deepcopy

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject, Qt, Signal

from darsia.fluidflower import calibration_review_workspace as workspace
from darsia.fluidflower.tools import phase_studio_series_ui as retained


class _Reader(QObject):
    finished = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.running = False
        self.starts = self.interruptions = self.waits = 0

    def start(self):
        self.starts += 1
        self.running = True

    def isRunning(self):
        return self.running

    def requestInterruption(self):
        self.interruptions += 1

    def wait(self):
        self.waits += 1
        self.running = False


@pytest.fixture
def retained_review(qtbot, monkeypatch, tmp_path):
    series_readers, model_readers, selected_calls = [], [], []

    class SeriesReader(_Reader):
        ready = Signal(str, object, str)

        def __init__(self, dataset, parent, cache_root):
            super().__init__(parent)
            self.dataset = dataset
            series_readers.append(self)

    class ModelReader(_Reader):
        ready = Signal(object, str)

        def __init__(self, root, parent):
            super().__init__(parent)
            self.root = root
            model_readers.append(self)

    monkeypatch.setattr(retained, "studio_series_keys", lambda: ("old", "latest"))
    monkeypatch.setattr(retained, "SavedSeriesResultLookup", SeriesReader)
    monkeypatch.setattr(retained, "ModelMassLookup", ModelReader)
    monkeypatch.setattr(retained, "series_path", lambda dataset, folder: tmp_path / dataset / folder)
    monkeypatch.setattr(retained, "load_archive", lambda *args: {})
    def selected_model():
        selected_calls.append(True)
        return {"root": tmp_path / "model"}
    saved = workspace.CalibrationReviewWorkspace(cache_root=tmp_path / "cache", selected_model=selected_model)
    qtbot.addWidget(saved)
    yield saved, series_readers, model_readers, selected_calls
    saved.stop_review_reads()


def _queue_series_result(saved, reader, entries=()):
    # Simulate a ready signal posted by the retained worker just before it
    # returns. Delivery remains pending until the test drains Qt MetaCalls.
    reader.ready.disconnect(saved._series_results_ready)
    reader.ready.connect(saved._series_results_ready, Qt.ConnectionType.QueuedConnection)
    reader.ready.emit(reader.dataset, list(entries), "")


def _drain_callbacks():
    QCoreApplication.sendPostedEvents(None, QEvent.Type.MetaCall)


def _view(saved, path):
    saved.result = {"retained-view": True}
    saved.result_path = path
    saved.slider.blockSignals(True)
    saved.slider.setRange(0, 3)
    saved.slider.setValue(2)
    saved.slider.blockSignals(False)
    saved.status.setText("Retained image and status")
    saved.details.setPlainText("Retained scientific notes")


def _snapshot(saved):
    return deepcopy((saved.entries, saved.visible, saved._series_catalog_loaded,
        saved.phase_catalog, saved.result, saved.result_path, saved.slider.value(),
        saved.status.text(), saved.details.toPlainText(), saved.import_model.isEnabled()))


@pytest.mark.parametrize("change_series", [False, True])
def test_queued_same_or_old_series_result_after_stop_preserves_view_and_stays_retired(
        retained_review, tmp_path, change_series):
    saved, readers, model_readers, selected_calls = retained_review
    saved._series_changed()
    reader = readers[0]
    late = dict(path=str(tmp_path / "never-read-result.json"), dataset="old", run="ac01",
                model="Late model", created_utc="", method={})
    _queue_series_result(saved, reader, [late])
    if change_series:
        saved.series.setCurrentIndex(saved.series.findData("latest"))
    assert len(readers) == 1 and saved.series_lookup is reader
    _view(saved, tmp_path / "retained-result.json")
    before = _snapshot(saved)
    saved.stop_review_reads()
    assert reader.interruptions == reader.waits == 1
    assert saved.lookup is None and saved.series_lookup is None
    _drain_callbacks()
    saved._series_changed()
    saved.load_model_results()
    assert _snapshot(saved) == before
    assert len(readers) == 1 and model_readers == [] and selected_calls == []
    assert saved.lookup is None and saved.series_lookup is None
    assert saved.native_catalog._pending is None and saved.native_collections._pending is None


def test_open_old_series_completion_starts_only_the_latest_selected_series(retained_review):
    saved, readers, _models, _selected = retained_review
    saved._series_changed()
    first = readers[0]
    _queue_series_result(saved, first)
    saved.series.setCurrentIndex(saved.series.findData("latest"))
    assert len(readers) == 1 and saved.series_lookup is first
    first.running = False
    _drain_callbacks()
    assert [reader.dataset for reader in readers] == ["old", "latest"]
    assert all(reader.starts == 1 for reader in readers)
    assert saved.series_lookup is readers[1] and saved._series_catalog_loaded == {"old"}
    _queue_series_result(saved, readers[1])
    readers[1].running = False
    _drain_callbacks()
    assert saved.series_lookup is None and saved._series_catalog_loaded == {"old", "latest"}
    assert len(readers) == 2 and not saved._review_closed


def test_queued_model_ready_and_finished_after_stop_cannot_change_view_or_start_series(
        retained_review, tmp_path):
    saved, series_readers, readers, selected_calls = retained_review
    saved.load_model_results()
    reader = readers[0]
    _view(saved, tmp_path / "retained-result.json")
    saved.phase_catalog = {"retained-model": True}
    before = _snapshot(saved)
    reader.ready.emit(dict(dataset="latest", jobs=[], results=[], model_id="late-model",
                           saved_results_deferred=True), "")
    reader.finished.emit()
    saved.stop_review_reads()
    assert saved.lookup is None and saved.series_lookup is None
    _drain_callbacks()
    saved.load_model_results()
    assert _snapshot(saved) == before
    assert reader.interruptions == reader.waits == 1
    assert len(readers) == 1 and series_readers == [] and len(selected_calls) == 1
