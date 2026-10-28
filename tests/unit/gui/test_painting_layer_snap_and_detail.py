"""Layer painting tab: a layer's own snap distance and the proposal detail.

A right-click on a layer gives it its own snap distance (0 keeps thin layers
exact); the step "Propose boundaries" chooses Normal or Fine detail. Both are
stored in the session. A synthetic 80 x 60 photograph and a project in a
temporary folder; no real series is read.
"""

import json

import numpy as np
from PySide6.QtWidgets import QInputDialog

from darsia.fluidflower import layer_painting as lp
from .test_painting_background_photo import (  # noqa: F401  (fixtures)
    POLYGON, _Project, paths, settings, view,
)


def _saved(view):
    view.save_session()
    path = lp.session_path(view.project, "ac52")
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_layer_gets_its_own_snap_distance_and_back(view):
    view.add_polygon(lp.Polygon(2, list(POLYGON)))
    view.set_layer_snap(2, 0.0)
    assert view.session.layer(2).snap_mm == 0.0
    assert view.session.layer_snap_mm(2) == 0.0
    assert view.session.layer_snap_mm(3) == view.session.snap_mm
    row = [layer.layer_id for layer in view.session.layers].index(2)
    assert view.layer_table.item(row, 2).text().endswith("· 0 mm")
    stored = {row["layer_id"]: row for row in _saved(view)["layers"]}
    assert stored[2]["snap_mm"] == 0.0 and "snap_mm" not in stored[3]
    view.set_layer_snap(2, None)
    assert view.session.layer(2).snap_mm is None
    assert "·" not in view.layer_table.item(row, 2).text()


def test_the_layer_menu_asks_for_the_snap_distance(view, monkeypatch):
    asked = []

    def answer(parent, title, label, value, low, high, decimals):
        asked.append((value, low, high))
        return 0.0, True

    monkeypatch.setattr(QInputDialog, "getDouble", answer)
    view._ask_layer_snap(2)
    assert asked == [(view.session.snap_mm, 0.0, lp.MAX_SNAP_MM)]
    assert view.session.layer(2).snap_mm == 0.0
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *args: (5.0, False))
    view._ask_layer_snap(2)
    assert view.session.layer(2).snap_mm == 0.0  # Cancel keeps it.


def test_fine_detail_proposes_on_the_display_grid(view):
    # A photograph whose analysis grid is coarser than its display.
    display = view.image.display
    view.image = lp.PaintingImage(
        view.image.path, view.image.image_size, display, np.ascontiguousarray(display[::2, ::2])
    )
    assert view._proposal_image() is view.image  # Normal detail.
    view.detail.setCurrentIndex(view.detail.findData("fine"))
    assert view.session.detail == "fine"
    fine = view._proposal_image()
    assert fine.analysis_shape == display.shape[:2]
    assert view._proposal_image() is fine  # Kept, so features are computed once.
    assert _saved(view)["proposal_settings"]["detail"] == "fine"
    view.detail.setCurrentIndex(view.detail.findData("normal"))
    assert view._proposal_image() is view.image
