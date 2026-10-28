"""A saved disagreement queue is checked off the GUI thread before its review opens."""
import threading
from types import SimpleNamespace as NS

from PySide6.QtWidgets import QCheckBox, QLabel, QLineEdit, QMessageBox, QPushButton, QWidget

from darsia.fluidflower.ui import evidence_audit_library as library

Library = library.EvidenceAuditLibrary


def _host(qtbot, tmp_path):
    host = QWidget()
    qtbot.addWidget(host)
    host.queues = NS(currentData=lambda: str(tmp_path / "queue.json"))
    host.owner = NS(reviewer=QLineEdit("reviewer"), review_save_root=QLineEdit(str(tmp_path)),
                    review_gpu=QCheckBox())
    host.start, host.status, host.launched = QPushButton(), QLabel(), []
    host.selection_changed = lambda: host.start.setEnabled(True)
    host.launch = host.launched.append
    host._review_queue_checked = lambda value, error: Library._review_queue_checked(host, value, error)
    return host


def test_the_queue_is_checked_in_the_background_then_the_review_opens(qtbot, monkeypatch, tmp_path):
    host, calls = _host(qtbot, tmp_path), []

    def load(path, verify_sources=False):
        calls.append((threading.current_thread() is threading.main_thread(), verify_sources))
        return dict(model_id="M", binding=dict(ffac_root=str(tmp_path), ffum_root=str(tmp_path),
                                               combined_root=str(tmp_path), selected_candidates={}))

    monkeypatch.setattr(library, "load_queue", load)
    monkeypatch.setattr(library, "build_phase_review_command", lambda **kwargs: ["review", kwargs["target_model_id"]])
    Library.start_review(host)
    assert not host.start.isEnabled()
    assert host.status.text() == "Checking the saved disagreement queue and its phase maps…"
    qtbot.waitUntil(lambda: bool(host.launched), timeout=10000)
    assert calls == [(False, True)] and host.launched == [["review", "M"]]
    assert host.start.isEnabled() and host.status.text() == ""


def test_a_queue_that_fails_its_check_says_why(qtbot, monkeypatch, tmp_path):
    host, shown = _host(qtbot, tmp_path), []

    def load(path, verify_sources=False):
        raise ValueError("A pinned phase map changed")

    monkeypatch.setattr(library, "load_queue", load)
    monkeypatch.setattr(QMessageBox, "critical", lambda _parent, title, text: shown.append((title, text)))
    Library.start_review(host)
    qtbot.waitUntil(lambda: bool(shown), timeout=10000)
    assert shown == [("Cannot open evidence review", "ValueError: A pinned phase map changed")]
    assert not host.launched and host.start.isEnabled()
