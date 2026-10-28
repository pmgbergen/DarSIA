"""Finite export UI fixtures; no real shared reads, solver or filesystem export."""
from copy import deepcopy
from pathlib import Path
import threading

from PySide6.QtGui import QCloseEvent

from darsia.fluidflower.jobs_panel import JobsPanel
from .test_jobs_preset_wasserstein_projection import _comparison
from .test_jobs_setup_projection import _image, _ready


def test_export_is_explicit_retains_close_guard_and_captured_job(qtbot, tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []
    main_thread = threading.get_ident()
    def export(root, destination):
        calls.append((root, destination, threading.get_ident()))
        entered.set()
        if not release.wait(10):
            raise RuntimeError("Export fixture was not released")
        return dict(csv=dict(path=str(Path(destination) / "tables" / "wasserstein_distances_all.csv")),
            receipt=dict(path=str(Path(destination) / "export_receipt.json")),
            comparison_count=6, nonfinite_distances=1)
    panel = JobsPanel(local_machine="fixture", preset_wasserstein_exporter=export)
    qtbot.addWidget(panel, before_close_func=lambda widget: qtbot.waitUntil(widget.can_close, timeout=10000))
    value = dict(_comparison(tmp_path / "job-a"), complete=True)
    _ready(panel, [value])
    page = panel.comparison_export
    page.destination.setText(str(tmp_path / "export"))
    assert page.export_button.isEnabled() and not calls
    page.export_button.click()
    try:
        qtbot.waitUntil(entered.is_set, timeout=5000)
        assert not page.can_close() and not panel.can_close()
        page.export()
        event = QCloseEvent()
        panel.closeEvent(event)
        assert not event.isAccepted()
        other = dict(deepcopy(value), root=str(tmp_path / "job-b"))
        _ready(panel, [other])
    finally:
        release.set()
        qtbot.waitUntil(panel.can_close, timeout=10000)
    assert len(calls) == 1 and calls[0][:2] == (value["root"], str(tmp_path / "export"))
    assert calls[0][-1] != main_thread
    assert "job-a" not in page.status.text()
    _ready(panel, [value])
    assert "Original comparison CSV exported" in page.status.text()
    assert "wasserstein_distances_all.csv" in page.status.text()
    assert "Nonfinite distances: 1" in page.status.text()
    assert panel._session_requests == []


def test_export_requires_complete_comparison_and_failure_never_claims_success(qtbot, tmp_path):
    calls = []
    def export(root, destination):
        calls.append(root)
        raise ValueError("Original shared receipt changed")
    panel = JobsPanel(local_machine="fixture", preset_wasserstein_exporter=export)
    qtbot.addWidget(panel, before_close_func=lambda widget: qtbot.waitUntil(widget.can_close, timeout=10000))
    page = panel.comparison_export
    _ready(panel, [_image(tmp_path / "image")])
    assert page.isHidden()
    value = _comparison(tmp_path / "comparison")
    _ready(panel, [value])
    assert not page.export_button.isEnabled()
    page.export()
    assert not calls
    value["complete"] = True
    _ready(panel, [value])
    page.destination.setText(str(tmp_path / "export"))
    page.export_button.click()
    qtbot.waitUntil(panel.can_close, timeout=10000)
    assert calls == [value["root"]]
    assert "export failed" in page.status.text() and "Original shared receipt changed" in page.status.text()
    assert "CSV exported:" not in page.status.text()
