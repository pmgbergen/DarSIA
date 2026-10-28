"""Owned GUI save action with finite publisher; no source qualification claim."""
from copy import deepcopy
import threading

import pytest

from .test_guided_water_line_step import WaterBackend, _review_for
from .test_painting_guided_review_ui import _dialog, _settle, _stage
from .test_painting_water_surface_ui import _api


def _widget(qtbot, builder):
    builder.run_id = "ac45"  # The series' recorded choice names the run the builder applies to.
    backend = WaterBackend(("ac45", "ac18"))
    api, _ = _api()
    widget, _ = _dialog(qtbot, backend=backend, review=_review_for(("ac45", "ac18")),
        editor=False, water_surface_comparison_reader=backend.water,
        water_surface_api=api, clearwater_review_builder=builder)
    _stage(qtbot, widget, "water_line")
    return widget


def _pickers(monkeypatch, tmp_path, *, during_save=None):
    from darsia.fluidflower import painting_guided_review_ui as ui
    authority = tmp_path / "decision.json"
    authority.write_bytes(b'{"finite": "publisher fixture only"}')
    monkeypatch.setattr(ui.QFileDialog, "getOpenFileName", lambda *a: (str(authority), ""))
    def save(*args):
        if during_save:
            during_save()
        return str(tmp_path / "new-review.json"), ""
    monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", save)
    return authority


def test_recorded_ac45_action_uses_original_owned_worker_and_keeps_open_review_and_answers(qtbot, tmp_path, monkeypatch):
    calls = []
    def builder(review, destination, *, authority):
        calls.append((deepcopy(review), destination, deepcopy(authority),
                      threading.current_thread() is threading.main_thread()))
        return dict(gap_review=dict(path=destination, sha256="a" * 64, size_bytes=10),
            source_policy=dict(path=destination + ".policy", sha256="b" * 64, size_bytes=20),
            run_id="ac45", original_source_files_changed=False, clearwater_selection_changed=True,
            current_review_changed=False, adopted=False, worker_started=False)
    widget = _widget(qtbot, builder)
    authority = _pickers(monkeypatch, tmp_path)
    before = deepcopy(widget.review), deepcopy(widget._rows), deepcopy(widget._questions)
    widget.reason.setText("Keep this unfinished human explanation")
    signals = []
    for signal in (widget.approval_changed, widget.recheck_requested, widget.adopt_requested,
                   widget.water_surface_confirmed):
        signal.connect(lambda *a: signals.append(a))
    assert widget.water_source_button.isVisible() and widget.water_source_button.isEnabled()
    widget.water_source_button.click()
    _settle(qtbot, widget)
    assert len(calls) == 1 and calls[0][0] == before[0] and calls[0][3] is False
    assert calls[0][2]["path"] == str(authority) and calls[0][2]["size_bytes"] == authority.stat().st_size
    assert (widget.review, widget._rows, widget._questions) == before
    assert widget.reason.text() == "Keep this unfinished human explanation"
    assert "current review and answers are unchanged" in widget.water_source_note.text()
    assert widget.saved_clearwater_review["clearwater_selection_changed"] is True
    assert signals == [] and widget.approval is widget._candidate is None
    widget.water_run_choice.setCurrentIndex(widget.water_run_choice.findData("ac18"))
    _settle(qtbot, widget)
    assert not widget.water_source_button.isVisible()


def test_changed_modal_run_cannot_publish_a_different_review(qtbot, tmp_path, monkeypatch):
    calls = []
    widget = _widget(qtbot, lambda *a, **kw: calls.append((a, kw)))
    _pickers(monkeypatch, tmp_path, during_save=lambda:
        widget.water_run_choice.setCurrentIndex(widget.water_run_choice.findData("ac18")))
    widget._prepare_water_source_review()
    _settle(qtbot, widget)
    assert calls == [] and widget.saved_clearwater_review is None
    assert "Nothing was created" in widget.water_source_note.text()
    assert widget.approval is widget._candidate is None


def test_close_waits_for_the_real_owned_publisher_without_switching_review(qtbot, tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def builder(review, destination, **options):
        entered.set()
        assert release.wait(5)
        raise ValueError("finite publisher refused the selected authority")
    widget = _widget(qtbot, builder)
    _pickers(monkeypatch, tmp_path)
    before = deepcopy(widget.review)
    try:
        widget.water_source_button.click()
        qtbot.waitUntil(entered.is_set, timeout=3000)
        assert widget.busy() and not widget.close()
        assert widget.review == before and widget.saved_clearwater_review is None
    finally:
        release.set()
        widget.wait_for_workers()
    assert not widget.busy() and widget.review == before
    assert widget.saved_clearwater_review is None and widget.approval is widget._candidate is None
