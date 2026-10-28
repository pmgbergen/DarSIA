"""Painting review canvases keep the 8-bit photograph while the same photograph is shown (synthetic arrays)."""
import numpy as np

from darsia.fluidflower.pages.geometry_facies.page import GeometryImageView


def test_a_redraw_of_the_same_photograph_keeps_its_8_bit_copy(qtbot):
    view = GeometryImageView()
    qtbot.addWidget(view)
    view.set_options(overlay="none", show_injectors=False)
    baseline = np.full((6, 8, 3), .5, np.float32)
    view.set_model(dict(baseline=baseline), {})
    kept = view._base8_kept
    assert kept[0] is baseline and kept[1].dtype == np.uint8 and int(kept[1][0, 0, 0]) == 128
    view.update_preview(dict(baseline=baseline), {})
    assert view._base8_kept is kept  # A stroke's redraw: no new conversion.
    brighter = np.full((6, 8, 3), 1., np.float32)
    view.update_preview(dict(baseline=brighter), {})
    assert view._base8_kept[0] is brighter and int(view._base8_kept[1][0, 0, 0]) == 255
    assert not view.pixmap_item.pixmap().isNull()
