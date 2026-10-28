"""Ordinary guided photos fit their visible viewport once, without decisions.

Finite review/photo DATA exercise the original owned Qt readers, photo canvas,
display switches and water controls. This is presentation/lifetime evidence,
not source, physical geometry or RAW decoder qualification.
"""
from copy import deepcopy

import pytest

from .test_painting_guided_review_ui import GuidedBackend, _bound_review, _dialog, _settle
from .test_guided_water_line_step import _water_dialog


def _ready(qtbot, widget):
    _settle(qtbot, widget)
    qtbot.waitUntil(lambda: widget._pending_photo_fit is None, timeout=3000)


def _whole_photo_fits_visible_viewport(widget):
    view = widget.photo
    mapped = view.transform().mapRect(view.item.boundingRect())
    viewport = view.viewport()
    assert view.isVisible() and view.item.pixmap().width() > 0
    assert mapped.width() <= viewport.width() + 4
    assert mapped.height() <= viewport.height() + 4
    assert max(mapped.width()/viewport.width(), mapped.height()/viewport.height()) > .95


def _keep_user_view(qtbot, widget):
    view = widget.photo
    view.scale(4., 4.)
    view.centerOn(30., 26.)
    horizontal, vertical = view.horizontalScrollBar(), view.verticalScrollBar()
    qtbot.waitUntil(lambda: horizontal.isVisible() and vertical.isVisible()
        and view.viewport().width()+vertical.width() == view.maximumViewportSize().width()
        and view.viewport().height()+horizontal.height() == view.maximumViewportSize().height(), timeout=2000)
    transform = view.transform()
    pan = horizontal.value(), vertical.value()
    widget.brighten.setValue(160)
    widget.image_shortcut.activated.emit()
    _ready(qtbot, widget)
    assert view.transform() == transform and (horizontal.value(), vertical.value()) == pan
    widget.image_shortcut.activated.emit()
    _ready(qtbot, widget)
    assert view.transform() == transform and (horizontal.value(), vertical.value()) == pan
    widget._buttons()  # Same context redraw does not arm another fit.
    widget.resize(widget.width()+100, widget.height()+70)
    qtbot.wait(1)  # Deliver native resize/layout; no fit remains queued.
    assert view.transform() == transform and widget._pending_photo_fit is None


@pytest.mark.parametrize("stage", ["source_water", "position", "warnings"])
def test_first_ordinary_photo_fits_after_leaving_hidden_editor_then_keeps_user_view(qtbot, monkeypatch, stage):
    review = _bound_review()
    review["issues"] = [
        dict(severity="warning", code="painted_map_frame_disagreement", run_id="ac18", message="Finite frame finding"),
        dict(severity="warning", code="finite_other_warning", run_id="ac18", message="Finite other finding"),
    ]
    widget, backend = _dialog(qtbot, review=review, backend=GuidedBackend())
    widget.resize(1800, 1200)
    assert widget.image_stack.currentWidget() is widget.deviation_editor
    # Reproduce the real hidden view's earlier small geometry, before its
    # stacked page is displayed and given the current large canvas extent.
    widget.photo.resize(360, 260)
    assert not widget.photo.isVisible()
    visible_fits = []
    original_fit = widget.photo.fit
    def fit():
        if widget.photo.viewport().isVisible():
            visible_fits.append((widget._preview_key, widget.photo.viewport().size()))
        return original_fit()
    monkeypatch.setattr(widget.photo, "fit", fit)
    index = next(index for index, (kind, _key) in enumerate(widget._questions) if kind == stage)
    widget._show_question(index)
    _ready(qtbot, widget)
    assert len(visible_fits) == 1 and visible_fits[0][0] == widget._preview_key
    _whole_photo_fits_visible_viewport(widget)
    before = deepcopy(widget._rows), deepcopy(widget.review)
    _keep_user_view(qtbot, widget)
    assert (widget._rows, widget.review) == before
    assert backend.approvals == [] and widget.approval is widget._candidate is None
    assert widget._acquisition_actions == [] and widget._warning_actions == {}


def test_first_hidden_water_photo_waits_for_original_clearwater_reader_before_fitting(qtbot, monkeypatch):
    widget, backend, actions = _water_dialog(qtbot)
    _ready(qtbot, widget)
    widget.resize(1800, 1200)
    widget.image_stack.setCurrentWidget(widget.photo_placeholder)
    widget.photo.resize(360, 260)
    assert not widget.photo.isVisible()
    fits = []
    original_fit = widget.photo.fit
    def fit():
        if widget.photo.viewport().isVisible() and not widget.busy() and widget._pending_photo_fit is None:
            fits.append((widget.images.currentData(), widget.water_surface.packet["basis"]["run_id"]
                         if widget.water_surface.packet else None))
        return original_fit()
    monkeypatch.setattr(widget.photo, "fit", fit)
    index = next(index for index, (stage, _key) in enumerate(widget._questions) if stage == "water_line")
    widget._show_question(index)
    _ready(qtbot, widget)
    assert fits[-1] == ("clearwater", "ac17")
    assert backend.water_reads == [("ac17", None, "clearwater", False)]
    assert widget._current_row() is None
    _whole_photo_fits_visible_viewport(widget)
    _keep_user_view(qtbot, widget)
    assert actions == [] and widget.water_surface.action is None and widget.water_surface_confirmations == {}
    assert widget.approval is widget._candidate is None
    assert widget._acquisition_actions == [] and widget._warning_actions == {}
