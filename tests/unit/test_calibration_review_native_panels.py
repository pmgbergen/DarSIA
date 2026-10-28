"""Native saved display integration; genuine tiny producer fixtures, unrun."""
import math

import numpy as np
import pytest

from darsia.fluidflower.calibration_review_native_panels import mass_curve_figure, read_native_panel
from darsia.fluidflower.storage_saved_rendering import release_stored_figure
from .test_calibration_review_native import native_attachment, _review, tiny_config, phase1_data


def test_native_mass_image_owns_pixels_after_figure_disposal(native_attachment):
    _config, completion, index = native_attachment
    reader = _review(completion, index)
    stored = reader.read_frame(0)
    before = {key: value.tobytes() for key, value in stored["arrays"].items()}
    image = read_native_panel(reader, stored, "mass_total", limit_g_per_cell=1.)
    assert not image.isNull() and image.width() > 100 and image.height() > 100
    assert image.copy().pixelColor(20, 20) == image.pixelColor(20, 20)
    assert {key: value.tobytes() for key, value in stored["arrays"].items()} == before


def test_native_mass_graph_uses_original_verified_scalar_time_and_mass(native_attachment, monkeypatch):
    _config, completion, index = native_attachment
    reader = _review(completion, index)
    original, scalars, calls = reader.read_frame, [], []
    def tracked(position):
        stored = original(position)
        scalars.append(stored["native_scalar_reference"])
        calls.append(position)
        return stored
    monkeypatch.setattr(reader, "read_frame", tracked)
    figure = mass_curve_figure(reader, cancelled=lambda: False)
    try:
        assert calls == list(range(len(reader.case_ids)))
        lines = {line.get_label(): line for line in figure.axes[0].lines}
        for label, key in (("Total", "total_mass_g"), ("Dissolved", "aqueous_mass_g"), ("Free gas", "gas_mass_g")):
            np.testing.assert_array_equal(lines[label].get_xdata(), [row["elapsed_h"] for row in scalars])
            np.testing.assert_array_equal(lines[label].get_ydata(), [row[key] for row in scalars])
    finally:
        release_stored_figure(figure)


def test_unknown_mass_is_a_gap_and_quantified_contributions_are_named_subtotals():
    class Reader:
        expected_run, case_ids = "display-contract", ("complete", "partial")
        def read_frame(self, index):
            return {"native_scalar_reference": dict(elapsed_h=float(index),
                total_mass_g=3. if index == 0 else None, aqueous_mass_g=1. if index == 0 else None,
                gas_mass_g=2. if index == 0 else None, injected_mass_g=4.,
                quantified_total_mass_g=2.5 if index else None,
                quantified_aqueous_mass_g=.8 if index else None,
                quantified_gas_mass_g=1.7 if index else None,
                analysis_scope={"unknown_mass_g": None} if index else None)}
    figure = mass_curve_figure(Reader(), cancelled=lambda: False)
    try:
        lines = {line.get_label(): line for line in figure.axes[0].lines}
        assert math.isnan(lines["Total"].get_ydata()[1])
        assert list(lines["Total · quantified subtotal"].get_ydata())[1] == 2.5
        assert lines["Total"].get_marker() == "o"
        assert lines["Total · quantified subtotal"].get_marker() == "o"
        assert "unknown mass remains" in figure.axes[0].get_title()
    finally:
        release_stored_figure(figure)


def test_native_graph_cancellation_does_not_read_a_frame(native_attachment, monkeypatch):
    _config, completion, index = native_attachment
    reader = _review(completion, index)
    monkeypatch.setattr(reader, "read_frame", lambda *_: pytest.fail("Cancelled read loaded a frame"))
    with pytest.raises(InterruptedError):
        mass_curve_figure(reader, cancelled=lambda: True)
