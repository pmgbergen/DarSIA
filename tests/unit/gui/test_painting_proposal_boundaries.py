"""Calculated proposal edges are display overlays, separate from drawn marks."""

from copy import deepcopy

import numpy as np

from .test_painting_photo_saturation import settings, make_view, view  # noqa: F401
from .test_painting_corrections import _proposal, _show
from .test_painting_manual_map import LEFT, RIGHT, MIDDLE, _drawn, _rendered


def _white(rgba):
    return np.all(rgba == (255, 255, 255, 235), axis=-1)


def test_proposal_boundaries_toggle_changes_only_the_display_edges(view, monkeypatch):
    rendered = _rendered(view, monkeypatch)
    assert not view.show_proposal_boundaries.isEnabled()
    _show(view, _proposal())
    classes = view.proposal.classes.copy()
    confidence = view.proposal.confidence.copy()
    session = deepcopy(view.session.to_dict())
    assert view.show_proposal_boundaries.text() == "Proposal boundaries"
    assert view.show_proposal_boundaries.isChecked()
    assert view.show_proposal_boundaries.isEnabled()
    outlined = rendered[-1]
    white = _white(outlined)
    assert white.any()
    view.show_proposal_boundaries.setChecked(False)
    plain = rendered[-1]
    assert not _white(plain).any()
    np.testing.assert_array_equal(plain[~white], outlined[~white])
    # The former internal white edge now has its layer's own fill colour.
    np.testing.assert_array_equal(plain[15, 20], plain[15, 25])
    view.show_proposal_boundaries.setChecked(True)
    np.testing.assert_array_equal(rendered[-1], outlined)
    np.testing.assert_array_equal(view.proposal.classes, classes)
    np.testing.assert_array_equal(view.proposal.confidence, confidence)
    assert view.session.to_dict() == session and not view._dirty


def test_manual_map_never_adds_proposal_edges_and_keeps_drawn_polygon_outlines(view, qtbot, monkeypatch):
    rendered = _rendered(view, monkeypatch)
    _show(view, _proposal())
    assert _white(rendered[-1]).any()
    for polygon in (LEFT, RIGHT, MIDDLE):
        view.add_polygon(deepcopy(polygon))
    view.map_shown.setCurrentIndex(view.map_shown.findText("Manual map"))
    _drawn(view, qtbot)
    view.save_session()
    proposal_classes = view.proposal.classes.copy()
    manual_classes = view._manual.classes.copy()
    session = deepcopy(view.session.to_dict())
    polygon_outlines = [item.pen().color() for item in view.canvas._polygon_items]
    manual_overlay = rendered[-1]
    assert not _white(manual_overlay).any()
    assert view.show_proposal_boundaries.isChecked()
    assert not view.show_proposal_boundaries.isEnabled()
    assert not view.canvas._polygon_fill
    assert len(polygon_outlines) == 3
    assert all(item.isVisible() for item in view.canvas._polygon_items)
    # Even a programmatic toggle cannot decorate a manual map with proposal edges.
    for checked in (False, True):
        view.show_proposal_boundaries.setChecked(checked)
        np.testing.assert_array_equal(rendered[-1], manual_overlay)
        assert [item.pen().color() for item in view.canvas._polygon_items] == polygon_outlines
    view.map_shown.setCurrentIndex(view.map_shown.findData("proposal"))
    assert view.show_proposal_boundaries.isEnabled() and _white(rendered[-1]).any()
    np.testing.assert_array_equal(view.proposal.classes, proposal_classes)
    np.testing.assert_array_equal(view._manual.classes, manual_classes)
    assert view.session.to_dict() == session and not view._dirty
