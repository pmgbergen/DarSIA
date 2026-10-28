"""Real hidden-stack/layout fitting; finite photo and draft DATA only."""
from copy import deepcopy

import numpy as np
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget

from darsia.fluidflower.painting_deviation_step_ui import DeviationPaintingStep

from .test_painting_deviation_step_ui import CHOICES, DraftSpy, _packet


def test_hidden_editor_fits_actual_visible_gap_once_then_preserves_the_owned_user_view(qtbot, monkeypatch):
    host = QWidget()
    layout = QVBoxLayout(host)
    stack = QStackedWidget()
    layout.addWidget(stack)
    placeholder = QLabel("The original photo read is pending")
    stack.addWidget(placeholder)
    step = DeviationPaintingStep(class_choices=CHOICES, sand_class="sand")
    stack.addWidget(step)
    qtbot.addWidget(host)
    host.resize(1800, 1200)
    host.show()
    qtbot.waitUntil(host.isVisible)
    fits = []
    original_fit = step.canvas.fitInView
    def fit(bounds, *options):
        if isinstance(bounds, QRectF):
            fits.append((QRectF(bounds), step.canvas.viewport().size(), step.isVisible()))
        return original_fit(bounds, *options)
    monkeypatch.setattr(step.canvas, "fitInView", fit)
    packet = _packet()
    draft = DraftSpy(packet["gap_mask"])
    step.set_context(packet, draft)  # The actual guided order: prepare while hidden.
    hidden_width = step.canvas.viewport().width()
    assert not step.isVisible() and fits == [] and step._pending_gap_fit is not None
    stack.setCurrentWidget(step)
    qtbot.waitUntil(lambda: step._pending_gap_fit is None, timeout=3000)
    assert len(fits) == 1 and fits[0][2]
    assert fits[0][1].width() > hidden_width * 2
    # Exact original mask pixels (48:97,36:65), existing (+6,+4) photo
    # transform, half-pixel outline and existing 12px minimum display margin.
    bounds = QRectF(41.5, 27.5, 73., 53.)
    assert fits[0][0] == bounds
    viewport = step.canvas.viewport()
    expected = min((viewport.width()-4)/bounds.width(), (viewport.height()-4)/bounds.height())
    assert step.canvas.transform().m11() == pytest.approx(expected, rel=.05)
    assert step.canvas.mapToScene(viewport.rect()).boundingRect().contains(bounds.center())
    assert draft.actions == [] and step.classification.currentData() is None

    step.canvas.scale(1.8, 1.8)
    step.canvas.centerOn(82., 56.)
    horizontal, vertical = step.canvas.horizontalScrollBar(), step.canvas.verticalScrollBar()
    qtbot.waitUntil(lambda: horizontal.isVisible() and vertical.isVisible()
        and viewport.width()+vertical.width() == step.canvas.maximumViewportSize().width()
        and viewport.height()+horizontal.height() == step.canvas.maximumViewportSize().height(), timeout=2000)
    transform = step.canvas.transform()
    center = step.canvas.mapToScene(viewport.rect().center())
    before = draft.snapshot()
    step.set_brightness(170)
    step.set_context(packet, draft)  # Same bound context uses the original update path.
    requests = []
    step.photo_requested.connect(requests.append)
    step.photo_shortcut.activated.emit()  # The original Alt+I action, no alternative photo path.
    step.set_photo(dict(packet, image_role="clearwater", image_rgb=np.full_like(packet["image_rgb"], 130)))
    step.photo_shortcut.activated.emit()
    step.set_photo(packet)
    assert requests == ["clearwater", "current"] and len(fits) == 1
    assert step.canvas.transform() == transform
    assert step.canvas.mapToScene(viewport.rect().center()) == center
    assert draft.snapshot()["actions"] == before["actions"]
    for name, mask in before["class_masks"].items():
        np.testing.assert_array_equal(draft.class_masks[name], mask)
    host.resize(1950, 1280)
    qtbot.waitUntil(lambda: host.width() == 1950)
    qtbot.wait(1)  # Deliver the real resize/layout without another fit.
    assert step.canvas.transform() == transform and len(fits) == 1

    another = deepcopy(packet)
    another.update(run_id="ac19", component_id="d"*64)
    step.set_context(another, DraftSpy(another["gap_mask"]))
    qtbot.waitUntil(lambda: step._pending_gap_fit is None, timeout=3000)
    assert len(fits) == 2 and fits[-1][0] == bounds and fits[-1][2]
