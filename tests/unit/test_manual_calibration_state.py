"""Unit tests for the state of the manual calibration (no figure involved)."""

from types import SimpleNamespace

import numpy as np
import pytest

import darsia
from darsia.presets.workflows.calibration.manual_calibration import CalibrationState

SHAPE = (8, 12)
META = {"dimensions": [1.0, 1.5], "space_dim": 2}


def _scalar(array) -> darsia.ScalarImage:
    return darsia.ScalarImage(np.asarray(array, dtype=float), **META)


class FakeSignalFunction:
    """Signal function with values at key colors; the signal scales with the last."""

    def __init__(self, values, support_indices=None):
        self.supports = np.linspace(0, 1, len(values))
        self.values = list(values)
        if support_indices is not None:
            self.support_indices = support_indices

    def update(self, values):
        self.values = list(values)


class FakeFlash:
    min_value_aq, max_value_aq, min_value_g, max_value_g = 0.0, 1.0, 1.0, 2.0

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, signal):
        c_aq = _scalar(np.clip(signal.img, 0, 1))
        s_g = _scalar(np.clip(signal.img - 1, 0, 1))
        return c_aq, s_g


def _make_state(has_gas=True, num_images=3):
    """State for two labels (left: 1, right: 2) and images of uniform brightness."""
    labels_array = np.ones(SHAPE, dtype=int)
    labels_array[:, SHAPE[1] // 2 :] = 2
    labels = darsia.ScalarImage(labels_array, **META)
    functions = {
        1: FakeSignalFunction([0.0, 0.5, 1.0]),
        2: FakeSignalFunction([0.0, 1.0, 2.0]),
    }
    states = []

    def call_pH_analysis(color_interpretation):
        scale = np.where(
            labels_array == 1, functions[1].values[-1], functions[2].values[-1]
        )
        return _scalar(color_interpretation.img * scale)

    analysis = SimpleNamespace(
        has_gas=has_gas,
        labels=labels,
        color_analysis=SimpleNamespace(base=labels),
        geometry=SimpleNamespace(
            depth=_scalar(np.full(SHAPE, 0.02)),
            porosity=_scalar(np.full(SHAPE, 0.4)),
            weight=_scalar(np.full(SHAPE, 0.008)),
        ),
        evaluation_mode=SimpleNamespace(projects_onto_path=True),
        call_color_interpretation=lambda image: _scalar(image.img.mean(axis=-1)),
        call_pH_analysis=call_pH_analysis,
        signal_model=SimpleNamespace(model=[None, functions]),
        flash=FakeFlash(),
        fluid_model=SimpleNamespace(
            mass_analysis=lambda c_aq, s_g: SimpleNamespace(
                mass=_scalar(1.5 * c_aq.img + 2.0 * s_g.img)
            ),
            density_gas=2.0,
            aqueous_capacity=1.5,
        ),
        expert_knowledge_adapter=None,
        contour_smoother=None,
        update_state=lambda date, protocol=None: states.append(date),
    )
    images = [
        darsia.Image(np.full((*SHAPE, 3), 0.25 * (i + 1)), **META)
        for i in range(num_images)
    ]
    experiment = SimpleNamespace(pressure_temperature_protocol=None)
    state = CalibrationState(analysis, images, experiment)
    return state, analysis, states


def test_setup_adds_full_roi_and_coarse_data():
    state, analysis, _ = _make_state()
    assert list(state.rois) == ["full"]
    assert state.coarse_shape == (200, 300)
    assert state.coarse_labels.img.shape == (200, 300)
    assert set(np.unique(state.coarse_labels.img)) == {1, 2}
    assert len(state.coarse_images) == len(state.coarse_color_interpretations) == 3
    assert state.contour_thresholds == [0.05, 0.05]
    assert state.initial_flash == (0.0, 1.0, 1.0, 2.0)
    np.testing.assert_array_equal(state.initial_values[2], [0.0, 1.0, 2.0])


def test_single_phase_has_no_gas_options():
    state, _, _ = _make_state(has_gas=False)
    assert "s_g" not in state.signal_options
    assert "rho_co2(g)" not in state.signal_options
    assert state.signal_options["c_aq"] == "Concentration"
    assert state.contour_thresholds == [0.05]
    assert state.initial_flash is None


def test_navigation_is_cyclic():
    state, _, _ = _make_state()
    assert [state.next_image() for _ in range(3)] == [1, 2, 0]

    assert state.signal_key == "pH"
    for _ in range(len(state.signal_option_ptr)):
        state.next_signal()
    assert state.signal_key == "pH"
    state.next_signal()
    assert (state.signal_key, state.signal_title) == ("color", "Color Interpretation")

    # Label 0 has no signal function: start at the first label, then cycle
    assert [state.next_label() for _ in range(3)] == [1, 2, 1]


def test_select_label_in_region():
    state, _, _ = _make_state()
    box = (slice(0, 4), slice(4, 12))  # 2 columns of label 1, 6 of label 2
    assert state.select_label_in(box) == 2
    assert state.label_idx == 2
    assert state.select_label_in(state.analysis.labels.img == 1) == 1


def test_compute_maps_of_current_image():
    state, _, states = _make_state()
    state.next_image()  # brightness 0.5
    maps = state.compute_maps()
    assert maps is state.maps and maps["image_idx"] == 1
    assert len(states) == 1  # thermodynamic state set for the image
    # Signal: brightness times the last value of the label (1 and 2)
    signal = maps["signals"]["pH"].img
    assert np.isclose(signal[:, :100], 0.5).all()
    assert np.isclose(signal[:, -100:], 1.0).all()
    assert np.isclose(maps["signals"]["rho_co2"].img[:, :100], 0.75).all()
    assert set(maps["signals"]) == set(state.signal_options)
    # Color scale: fixed for the signal, min-max for the other maps
    assert state.signal_range() == (0.0, 1.0)
    state.next_signal()
    assert np.allclose(state.signal_range(), (0.5, 0.5))


def test_single_phase_maps_have_no_gas_entries():
    state, _, _ = _make_state(has_gas=False)
    maps = state.compute_maps()
    assert set(maps["signals"]) == set(state.signal_options)


def test_contour_overlay_is_cached_per_thresholds(monkeypatch):
    state, _, _ = _make_state()
    state.compute_maps()
    calls = []

    def fake_plot(image, mask, **kwargs):
        calls.append(len(mask))
        return image

    monkeypatch.setattr(darsia, "plot_contour_on_image", fake_plot)
    state.contour_overlay()
    state.contour_overlay([0.05, 0.05])
    assert calls == [6]  # five c_aq contours and one for s_g, computed once
    state.contour_overlay([0.2, 0.05])
    assert calls == [6, 6]
    state.compute_maps()  # new maps: the cache is gone
    state.contour_overlay([0.2, 0.05])
    assert len(calls) == 3


def test_label_highlight_grays_only_the_rest():
    state, _, _ = _make_state()
    image = state.coarse_image
    image.img[..., 0] = 1.0  # reddish
    highlighted = state.label_highlight(image, 1)
    inside = state.coarse_labels.img == 1
    np.testing.assert_array_equal(highlighted[inside], image.img[inside])
    outside_red = highlighted[~inside][:, 0]
    assert (outside_red < 1.0).all()


def test_check_values_respects_unused_key_colors():
    state, analysis, _ = _make_state()
    assert state.check_values([0.0, 0.4, 0.8], label=1) is None
    assert "must not decrease" in state.check_values([0.0, 0.8, 0.4], label=1)
    # Unused key colors do not count
    analysis.signal_model.model[1][1].support_indices = [0, 2]
    assert state.check_values([0.0, 0.9, 0.4], label=1) is None


@pytest.mark.parametrize("has_gas", [True, False])
def test_apply_parameters(has_gas):
    state, analysis, _ = _make_state(has_gas=has_gas)
    function = analysis.signal_model.model[1][2]
    flash = [0.0, 1.0, 1.0, 2.0]

    # Unchanged values
    assert state.apply_parameters([0.0, 1.0, 2.0], flash, label=2) is False
    # Invalid values: nothing is applied
    assert state.apply_parameters([0.0, 2.0, 1.0], flash, label=2) is None
    assert function.values == [0.0, 1.0, 2.0]
    # Changed signal values of the selected label (default)
    state.label_idx = 2
    assert state.apply_parameters([0.0, 0.6, 1.2], flash) is True
    assert function.values == [0.0, 0.6, 1.2]
    assert analysis.signal_model.model[1][1].values == [0.0, 0.5, 1.0]
    # Flash parameters count only with a gas phase
    changed = state.apply_parameters([0.0, 0.6, 1.2], [0.1, 1.0, 1.0, 2.0])
    assert changed is has_gas
    assert analysis.flash.min_value_aq == (0.1 if has_gas else 0.0)
    # The start values for Reset are untouched
    np.testing.assert_array_equal(state.initial_values[2], [0.0, 1.0, 2.0])
