"""The saved adoption preparation is read off the GUI thread before its dialog opens."""
import threading
from types import SimpleNamespace as NS

from PySide6.QtWidgets import QLabel, QWidget

from darsia.fluidflower import map_adoption as ma
from darsia.fluidflower import painting_series_adoption_ui as ui

Controller = ui.SeriesAdoptionController


def _host(qtbot, tmp_path, context):
    view = QWidget()
    qtbot.addWidget(view)
    view.use_map_progress = QLabel()
    host = NS(context=context, _gap_review_source=None, _suspended=False, error=None,
              preparation=dict(folder=str(tmp_path)), view=view, _current=lambda: True, opened=[])
    host._open_adoption_dialog = host.opened.append
    host._preparation_read = lambda value, error: Controller._preparation_read(host, value, error)
    return host


def test_the_dialog_opens_on_the_preparation_read_in_the_background(qtbot, monkeypatch, tmp_path):
    threads = []

    def read(folder):
        threads.append(threading.current_thread() is threading.main_thread())
        return dict(read_from=str(folder))

    monkeypatch.setattr(ma, "read_preparation", read)
    host = _host(qtbot, tmp_path, dict(version="v"))
    Controller.preparation_retired(host)
    assert host.view.use_map_progress.text() == "Reading the prepared adoption and its saved labels…"
    qtbot.waitUntil(lambda: bool(host.opened), timeout=10000)
    assert threads == [False] and host.opened == [dict(read_from=str(tmp_path))]


def test_a_manual_map_needs_no_reading_and_a_restart_drops_the_old_answer(qtbot, monkeypatch, tmp_path):
    release, reads = threading.Event(), []

    def read(folder):
        reads.append(folder)
        assert release.wait(10)
        return dict(read_from=str(folder))

    monkeypatch.setattr(ma, "read_preparation", read)
    host = _host(qtbot, tmp_path, dict(manual_map_approved=True))
    Controller.preparation_retired(host)
    assert host.opened == [host.preparation] and not reads
    host.opened.clear()
    host.context = first = dict(version="v1")
    Controller.preparation_retired(host)
    host.context = dict(version="v2")  # Adoption started again meanwhile.
    release.set()
    qtbot.waitUntil(lambda: not host._preparation_reader.busy, timeout=10000)
    assert reads and first is not host.context and not host.opened
