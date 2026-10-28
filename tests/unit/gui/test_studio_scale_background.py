"""Series mass workspace: an alternate concentration scale is drawn off the GUI thread and kept.

Synthetic 6 x 8 saved fields in a temporary folder; no saved result of a real
series is read.
"""
import threading

import numpy as np
from PySide6.QtWidgets import QComboBox, QSlider, QWidget

from darsia.fluidflower.tools import phase_studio_series_ui as studio

Workspace = studio.SeriesMassWorkspace


def _host(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(studio, "local_cache_root", lambda: tmp_path / "cache")
    host = QWidget()
    qtbot.addWidget(host)
    host.slider = QSlider()
    host.slider.setRange(0, 3)
    host.color_scale = QComboBox()
    host.color_scale.addItem("Nonlinear", "power")
    host.result_path = tmp_path / "result" / "result.json"
    host._scale_errors, host.shown = {}, []
    host.show_frame = host.shown.append
    for name in ("_saved_figure_path", "_scale_cache_path", "_start_scale_render", "_scale_rendered",
                 "_scaled_concentration_preview"):
        setattr(host, name, getattr(Workspace, name).__get__(host))
    folder = tmp_path / "result"
    folder.mkdir()
    np.savez(folder / "fields.npz", concentration_g_l=np.linspace(0, 2, 48, dtype=np.float32).reshape(6, 8),
             rgb=np.full((6, 8, 3), .5, np.float32))
    return host


def test_an_alternate_scale_is_drawn_in_the_background_and_kept(qtbot, tmp_path, monkeypatch):
    host = _host(qtbot, tmp_path, monkeypatch)
    threads = []

    def drawn(self, row, mode, base=None):
        threads.append(threading.current_thread() is threading.main_thread())
        return Workspace._scaled_concentration_preview(self, row, mode, base)

    host._scaled_concentration_preview = drawn.__get__(host)
    row = dict(fields="fields.npz", color_limit_g_l=2.0)
    fields, kept = host._scale_cache_path(row, "power")
    assert fields == str(tmp_path / "result" / "fields.npz") and not kept.exists()
    host._start_scale_render(0, row, "power")
    qtbot.waitUntil(lambda: host.shown == [0], timeout=20000)
    assert threads == [False] and kept.is_file()
    # Another result shown meanwhile: the drawing still uses the result it started from.
    host.result_path = tmp_path / "other" / "result.json"
    assert host._scale_cache_path(row, "power", tmp_path / "result" / "result.json")[1] == kept


def test_a_drawing_that_fails_is_said_and_not_tried_again(qtbot, tmp_path, monkeypatch):
    host = _host(qtbot, tmp_path, monkeypatch)

    def broken(self, row, mode, base=None):
        raise ValueError("the saved fields lack concentration_g_l")

    host._scaled_concentration_preview = broken.__get__(host)
    host._start_scale_render(0, dict(fields="fields.npz"), "power")
    qtbot.waitUntil(lambda: host.shown == [0], timeout=20000)
    assert host._scale_errors == {(0, "power"): "ValueError: the saved fields lack concentration_g_l"}
    # A drawing for a frame no longer shown is kept quiet.
    host.slider.setValue(2)
    host._start_scale_render(1, dict(fields="fields.npz"), "power")
    qtbot.waitUntil(lambda: (1, "power") in host._scale_errors, timeout=20000)
    assert host.shown == [0]
