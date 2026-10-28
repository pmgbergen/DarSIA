"""Model comparison view: an opacity change blends again without hatching again (synthetic images)."""
import numpy as np
from PySide6.QtGui import QColor, QImage

from darsia.fluidflower import phase_filled_display
from darsia.fluidflower.tools import phase_model_comparison_view as view_module


def _image(color):
    image = QImage(8, 6, QImage.Format.Format_RGBA8888)
    image.fill(QColor(color))
    return image


def test_an_opacity_change_reuses_the_hatched_overlay(qapp, monkeypatch):
    calls, original = [], phase_filled_display.hatch_filled_overlay

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(phase_filled_display, "hatch_filled_overlay", counted)
    photo, overlay = _image("blue"), _image("red")
    overlay._filled_phase_mask = np.zeros((6, 8), bool)
    hatched = {}
    half = view_module.compose(photo, overlay, .5, hatched=hatched)
    most = view_module.compose(photo, overlay, .8, hatched=hatched)
    assert len(calls) == 1  # Only the opacity changed: the hatched copy is reused.
    assert half.pixelColor(0, 0) != most.pixelColor(0, 0)
    view_module.compose(photo, overlay, .8, filled=False, hatched=hatched)
    assert len(calls) == 2 and len(hatched) == 1  # One copy per overlay, for its latest switches.
    view_module.compose(photo, overlay, .5)
    view_module.compose(photo, overlay, .5)
    assert len(calls) == 4  # Without a cache, every blend hatches.
