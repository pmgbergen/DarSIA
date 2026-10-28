"""Native index attachment and retained legacy view coexist without projection."""
import pytest

from darsia.fluidflower import calibration_review_panels as panels
from ..test_calibration_review_native import native_attachment, _review, tiny_config, phase1_data
from .test_calibration_review_approval import _page, _series
from .test_calibration_review_panels import _ManualRead


def test_native_attachment_selects_exact_freeze_and_run_and_cannot_approve_series(qtbot, monkeypatch, tmp_path, native_attachment):
    _config, completion, index = native_attachment
    reader = _review(completion, index)
    monkeypatch.setattr(panels, "_PanelRead", _ManualRead)
    page, experiments = _page(qtbot, monkeypatch, _series(tmp_path))
    saved = page.saved_results
    try:
        saved.register_native_review(reader, dataset=page.series, calibration_name="Native named fit")
        page.show_view("review")
        key = "sha256:" + reader.expected_calibration_sha256
        saved.mass_calibration.setCurrentIndex(saved.mass_calibration.findData(key))
        assert saved._native_review is reader
        assert saved.result is None  # No invented Studio result dictionary.
        assert experiments.currentData() == reader.expected_run
        assert saved.review_panels.result is reader
        assert saved.review_panels.panels[1].content.currentData() == "mass_total"
        assert saved.review_panels.panels[1].mass_controls.isVisibleTo(saved)
        assert not page.approve_button.isEnabled()
        assert "no registered source binding" in page.approval_label.text()
        saved.slider.setValue(1)
        saved.refresh(keep_view=True)
        assert saved._native_review is reader and saved.slider.value() == 1
        assert saved.review_panels.index == 1
        page.show_view("results")
        assert saved._native_review is None
        assert all(entry["path"] != str(saved._native_entries[next(iter(saved._native_entries))]["path"])
                   for entry in saved.visible)
        assert saved.result is not None
        page.show_view("review")
        saved.mass_calibration.setCurrentIndex(saved.mass_calibration.findData(key))
        assert saved._native_review is reader
    finally:
        page._stop_lookups()


def test_changing_root_for_same_series_hides_old_native_results_and_rejects_late_lookup(qtbot, monkeypatch, tmp_path, native_attachment):
    _config, completion, index = native_attachment
    reader = _review(completion, index)
    monkeypatch.setattr(panels, "_PanelRead", _ManualRead)
    page, _ = _page(qtbot, monkeypatch, _series(tmp_path))
    saved, lookups = page.saved_results, []
    monkeypatch.setattr(saved, "_start_native_lookup", lambda **kwargs: lookups.append(saved._native_context()))
    try:
        saved.register_native_review(reader, dataset=page.series, calibration_name="First root fit")
        page.show_view("review")
        saved.mass_calibration.setCurrentIndex(saved.mass_calibration.findData("sha256:" + reader.expected_calibration_sha256))
        original_context, original_root = saved._native_context(), saved._native_series_root
        old_path = str(saved.result_path)
        assert saved._native_review is reader
        next_root = tmp_path / "another-root"
        next_root.mkdir()
        saved.set_native_series_root(next_root)
        assert saved._native_context() != original_context
        assert saved._native_review is None
        assert all(entry["path"] != old_path for entry in saved.visible)
        assert lookups[-1] == saved._native_context()
        with pytest.raises(ValueError, match="another series root"):
            saved.load_result(old_path)
        saved._native_catalog_ready(original_context, [dict(reader=reader, dataset=page.series,
            calibration_name="Late old root fit")], ["Old root error"])
        assert saved.native_notice.text() == ""
        assert all(entry["path"] != old_path for entry in saved.visible)
        saved.set_native_series_root(original_root)
        assert any(entry["path"] == old_path for entry in saved.visible)
        assert lookups[-1] == original_context
    finally:
        page._stop_lookups()


def test_native_approval_waits_for_current_binding_and_uses_explicit_action_guard(qtbot, monkeypatch, tmp_path, native_attachment):
    from darsia.fluidflower import calibration_review_native_approval as approval
    _config, completion, index = native_attachment
    reader = _review(completion, index)
    monkeypatch.setattr(panels, "_PanelRead", _ManualRead)
    page, _ = _page(qtbot, monkeypatch, _series(tmp_path))
    saved, requests, writes = page.saved_results, [], []
    monkeypatch.setattr(page._native_approval_lookup, "request", lambda *args, **kwargs: requests.append((args, kwargs)))
    try:
        saved.register_native_review(reader, dataset=page.series, calibration_name="Native fit",
            registration={"root": "explicit-job", "job": {"sha256": "bound-original"}})
        page.show_view("review")
        saved.mass_calibration.setCurrentIndex(saved.mass_calibration.findData("sha256:" + reader.expected_calibration_sha256))
        context = page._native_approval_request
        assert context is not None and requests
        assert not page.approve_button.isEnabled()
        binding = dict(calibration={"reference": "fits/frozen.json", "sha256": reader.expected_calibration_sha256},
                       result=reader.index["path"], native_evidence={"index": reader.index})
        page._native_approval_ready(("different-source",), binding, "")
        assert not page.approve_button.isEnabled()
        page._native_approval_ready(context, binding, "")
        assert page.approve_button.isEnabled()
        previous_requests = len(requests)
        page.show_view("review")  # A lock/source may change on another page.
        assert len(requests) > previous_requests and not page.approve_button.isEnabled()
        context = page._native_approval_request
        page._native_approval_ready(context, binding, "")
        def approve(*args, **kwargs):
            assert kwargs["expected_binding"] is binding
            assert kwargs["action_guard"]()
            writes.append(kwargs)
        monkeypatch.setattr(approval, "approve_native_run", approve)
        page.approve_selected_run()
        assert len(writes) == 1
        page.show_view("results")
        assert not writes[0]["action_guard"]()
        page._native_approval_ready(context, binding, "")
        assert page._native_approval_binding is None
    finally:
        page._stop_lookups()


def test_collection_dialog_cannot_attach_to_a_series_changed_while_it_was_open(qtbot, monkeypatch, tmp_path, native_attachment):
    from darsia.fluidflower import calibration_review_workspace as workspace
    _config, completion, index = native_attachment
    reader = _review(completion, index)
    monkeypatch.setattr(panels, "_PanelRead", _ManualRead)
    page, _ = _page(qtbot, monkeypatch, _series(tmp_path))
    saved, requests = page.saved_results, []
    monkeypatch.setattr(saved.native_collections, "request", lambda *args: requests.append(args))
    monkeypatch.setattr(saved, "_start_native_lookup", lambda **kwargs: None)
    try:
        saved.register_native_review(reader, dataset=page.series, calibration_name="Native fit")
        page.show_view("review")
        saved.mass_calibration.setCurrentIndex(saved.mass_calibration.findData("sha256:" + reader.expected_calibration_sha256))
        context = (saved._native_context(), reader)
        replacement_root = tmp_path / "another-selected-root"
        replacement_root.mkdir()
        def selected(*args):
            saved.set_native_series_root(replacement_root)
            return "never-open-this-receipt.json", ""
        monkeypatch.setattr(workspace.QFileDialog, "getOpenFileName", selected)
        saved._choose_native_collection()
        assert requests == [] and saved._native_review is None
        saved._native_collection_ready(context, reader, "")
        assert saved._native_review is None and not saved.open_collection.isEnabled()
        page._stop_lookups()
        current_context = saved._native_context()
        saved._native_catalog_ready(current_context, [dict(reader=reader, dataset=page.series,
            calibration_name="Late after close")], [])
        assert saved._native_review is None
    finally:
        page._stop_lookups()


def test_queued_outer_model_catalogues_after_stop_cannot_mutate_or_restart_page(qtbot, monkeypatch, tmp_path):
    from copy import deepcopy
    from PySide6.QtCore import QObject, Qt, Signal
    from darsia.fluidflower.tools import phase_studio_mass_analysis_workspace as module
    from .test_calibration_review_retained_shutdown import _Reader, _drain_callbacks

    readers = []
    class Lookup(_Reader):
        ready = Signal(object, str)

        def __init__(self, *args):
            super().__init__(args[-1])
            readers.append(self)

    class QueuedDelivery(QObject):
        models = Signal(int, object, object, str)
        catalog = Signal(int, object, str)

    monkeypatch.setattr(module, "_ModelChoicesLookup", Lookup)
    monkeypatch.setattr(module, "ModelMassLookup", Lookup)
    page, _ = _page(qtbot, monkeypatch, _series(tmp_path))
    try:
        previous = str(tmp_path / "selected-model")
        page.model.blockSignals(True)
        page.model.addItem("Retained model", previous)
        page.model.setCurrentIndex(page.model.findData(previous))
        page.model.blockSignals(False)
        page._selected_model_changed()
        assert len(readers) == 2
        page.phase_catalog = {"retained-model": True}
        page.status.setText("Retained page status")
        generation = page._generation
        rows = [dict(root=previous, model_id="late-choice")]
        catalog = dict(dataset=page.series, jobs=[], results=[], model_id="late-catalogue",
                       saved_results_deferred=True)
        delivery = QueuedDelivery(page)
        delivery.models.connect(page._models_ready, Qt.ConnectionType.QueuedConnection)
        delivery.catalog.connect(page._catalog_ready, Qt.ConnectionType.QueuedConnection)
        before = deepcopy((page.phase_catalog, page.status.text(), page.model.currentData(),
                           page.model.count(), page.saved_results.phase_catalog,
                           page.saved_results.result_path, page.refresh_button.isEnabled()))
        delivery.models.emit(generation, previous, rows, "")
        delivery.catalog.emit(generation, catalog, "")
        for reader in readers:
            reader.finished.emit()
        page._stop_lookups()
        assert page._generation > generation and page._background_lookups == []
        assert page._model_lookup is None and page._catalog_lookup is None
        _drain_callbacks()
        # The closed-state guard is independent of the stale generation guard.
        page._models_ready(page._generation, previous, rows, "")
        page._catalog_ready(page._generation, catalog, "")
        page.refresh_models()
        page._selected_model_changed()
        after = (page.phase_catalog, page.status.text(), page.model.currentData(),
                 page.model.count(), page.saved_results.phase_catalog,
                 page.saved_results.result_path, page.refresh_button.isEnabled())
        assert after == before and len(readers) == 2
        assert all(reader.starts == reader.interruptions == reader.waits == 1 for reader in readers)
        assert page.saved_results.lookup is None and page.saved_results.series_lookup is None
    finally:
        page._stop_lookups()
