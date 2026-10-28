"""Original finite GUI collection of explicit metadata specimens, not science.

The real collector and its original source validators run. Only file choices
and a scheduling barrier are injected; no producer, queue or approval is faked.
"""
import threading

import pytest
from PySide6.QtWidgets import QFileDialog

from darsia.fluidflower import analysis_preparation as preparation
from darsia.fluidflower.analysis_preparation_ui import AnalysisPreparationDialog
from darsia.fluidflower.jobs_panel import JobsPanel
from ..test_analysis_source_inputs import saved_products, phase1_data  # noqa: F401


def _dialog(qtbot, products):
    panel = JobsPanel()
    panel.series_root = products.root
    qtbot.addWidget(panel)
    dialog = AnalysisPreparationDialog(panel, panel)
    qtbot.addWidget(dialog)
    dialog.geometry.setText(products.geometry["path"])
    dialog.pressure.setText(products.pressure["path"])
    dialog.inputs.setText(str(products.root / "previous-manual-inputs.json"))
    dialog.show()
    return panel, dialog


@pytest.mark.parametrize("stale", [False, True])
def test_real_metadata_collection_uses_owned_thread_and_never_prepares_or_adopts(
        qtbot, saved_products, tmp_path, monkeypatch, stale):
    products = saved_products
    panel, dialog = _dialog(qtbot, products)
    target = tmp_path / "collected-inputs.json"
    before = dialog.inputs.text()
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *args, **kwargs:
        ([row["path"] for row in products.configurations], ""))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(target), ""))
    original = preparation.collect_phase1_source_inputs
    entered, release = threading.Event(), threading.Event()
    calls, adopted = [], []
    owner_thread = threading.get_ident()
    def collect(*args, **kwargs):
        calls.append(threading.get_ident())
        entered.set()
        if not release.wait(10):
            raise RuntimeError("The finite metadata reader was not released")
        return original(*args, **kwargs)
    monkeypatch.setattr(preparation, "collect_phase1_source_inputs", collect)
    dialog.adopted.connect(adopted.append)
    try:
        dialog.collect_button.click()
        qtbot.waitUntil(entered.is_set, timeout=10000)
        assert len(calls) == 1 and calls[0] != owner_thread
        assert not dialog.collect_button.isEnabled() and not dialog.adopt_button.isEnabled()
        dialog.reject()
        assert dialog.isVisible() and not dialog.can_close()
        if stale:
            dialog.pressure.setText(str(tmp_path / "new-current-pressure.json"))
        release.set()
        qtbot.waitUntil(lambda: dialog._thread is None, timeout=10000)
        assert target.is_file() and dialog.can_close()
        if stale:
            assert dialog.inputs.text() == before and "earlier choices" in dialog.status.text()
        else:
            assert dialog.inputs.text() == str(target.resolve())
            assert "Collected existing inputs" in dialog.status.text()
        assert dialog.prepared_result is None and not dialog.adopt_button.isEnabled()
        assert adopted == [] and panel.series_root == products.root
        assert not list(tmp_path.rglob("execution.toml"))
        assert not list(tmp_path.rglob("job.json"))
    finally:
        release.set()
        qtbot.waitUntil(lambda: dialog._thread is None, timeout=10000)
        dialog.close()


def test_source_collection_refuses_changed_modal_choices_and_other_stage(
        qtbot, saved_products, tmp_path, monkeypatch):
    products = saved_products
    panel, dialog = _dialog(qtbot, products)
    target = tmp_path / "never-written.json"
    before = dialog.inputs.text()
    def choose(*args, **kwargs):
        dialog.probability.setChecked(not dialog.probability.isChecked())
        return [row["path"] for row in products.configurations], ""
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", choose)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(target), ""))
    dialog.collect_button.click()
    assert dialog._thread is None and "changed while choosing files" in dialog.status.text()
    assert dialog.inputs.text() == before and not target.exists()
    dialog.stage.setCurrentIndex(1)
    assert not dialog.collect_button.isEnabled()
    dialog.collect_sources()  # A direct slot call cannot enable this unimplemented stage.
    assert dialog._thread is None and not target.exists()
    assert "not available here" in dialog.collect_note.text()
    dialog.stage.setCurrentIndex(0)
    panel.series_root = tmp_path / "different-series"
    dialog.collect_sources()
    assert "selected series changed" in dialog.status.text()
    assert dialog._thread is None and dialog.inputs.text() == before and not target.exists()
