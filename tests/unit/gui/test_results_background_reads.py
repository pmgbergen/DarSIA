"""The Results page reads a job, its runs and frames off the GUI thread.

A synthetic production job in a temporary folder; no real series or NAS input
is used.
"""
import threading

from darsia.fluidflower import production_results_view as view
from .test_production_results_view import _settled, production_job


def _page(qtbot):
    page = view.ProductionResultsPage()
    qtbot.addWidget(page)
    page.resize(1200, 900)
    return page


def _on_threads(monkeypatch, cls, name, release=None):
    """Record whether ``cls.name`` runs on the GUI thread; hold it until ``release`` is set."""
    original, calls = getattr(cls, name), []

    def recorded(*args, **kwargs):
        calls.append(threading.current_thread() is threading.main_thread())
        if release is not None:
            assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(cls, name, recorded)
    return calls


def test_the_job_and_its_frames_are_read_off_the_gui_thread(qtbot, monkeypatch, tmp_path):
    data = production_job(tmp_path, states=(("r1", "complete"), ("r2", "queued")))
    opened = _on_threads(monkeypatch, view.ProductionJob, "__init__")
    read = _on_threads(monkeypatch, view.ProductionJob, "field_image")
    page = _page(qtbot)
    assert page.open_job(data.job)
    assert page.loading and page.runs_table.rowCount() == 0
    _settled(page, qtbot)
    assert opened == [False] and read and not any(read)
    assert page.runs_table.rowCount() == 2 and page.shown == ("r1", "r1_f1", "phase_labels", "products")


def test_only_the_last_chosen_frame_is_shown(qtbot, monkeypatch, tmp_path):
    data = production_job(tmp_path, states=(("r1", "complete"),))
    page = _page(qtbot)
    page.open_job(data.job)
    _settled(page, qtbot)
    release = threading.Event()
    _on_threads(monkeypatch, view.ProductionJob, "field_image", release)
    page.show_frame(1)
    page.show_frame(2)
    assert page.frame_label.text() == "Frame 3 of 3 · r1_f3 · 1.50 h"
    assert page.status_label.text() == "Reading r1_f3…"
    release.set()
    _settled(page, qtbot)
    assert page.shown == ("r1", "r1_f3", "phase_labels", "products") and page.status_label.text() == ""


def test_a_scale_typed_while_a_frame_is_read_is_used(qtbot, monkeypatch, tmp_path):
    data = production_job(tmp_path, states=(("r1", "complete"),))
    page = _page(qtbot)
    page.open_job(data.job)
    _settled(page, qtbot)
    page.field_combo.setCurrentIndex(page.field_combo.findData("concentration"))
    _settled(page, qtbot)
    release, scales = threading.Event(), []
    _on_threads(monkeypatch, view.ProductionJob, "field_image", release)
    original = view.render_map
    monkeypatch.setattr(view, "render_map", lambda *args, **kwargs: scales.append(kwargs["scale_max"])
                        or original(*args, **kwargs))
    page.show_frame(2)
    page.scale_edit.setText("2.5")
    page.scale_edit.editingFinished.emit()
    release.set()
    _settled(page, qtbot)
    assert page.shown[1] == "r1_f3" and scales[-1] == 2.5


def test_a_run_chosen_while_another_is_read_is_the_one_shown(qtbot, monkeypatch, tmp_path):
    data = production_job(tmp_path, states=(("r1", "complete"), ("r2", "complete")))
    page = _page(qtbot)
    page.open_job(data.job)
    _settled(page, qtbot)
    release = threading.Event()
    _on_threads(monkeypatch, view.ProductionJob, "sources", release)
    page.select_run("r1")
    assert page.map_view.text() == "Reading run r1…"
    page.select_run("r2")
    release.set()
    _settled(page, qtbot)
    assert page.run == "r2" and page.shown[0] == "r2"
