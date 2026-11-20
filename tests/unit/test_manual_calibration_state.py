"""Unit tests for the state of the manual calibration (no figure involved)."""

import copy
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


def _fake_analysis(shape, functions, flash, has_gas, calls, states, reducible):
    """Analysis for two labels (left: 1, right: 2); signal = brightness * last value."""
    labels_array = np.ones(shape, dtype=int)
    labels_array[:, shape[1] // 2 :] = 2
    labels = darsia.ScalarImage(labels_array, **META)

    def call_pH_analysis(color_interpretation):
        calls.append(shape)
        scale = np.where(
            labels_array == 1, functions[1].values[-1], functions[2].values[-1]
        )
        return _scalar(color_interpretation.img * scale)

    def call_flash_and_mass_analysis(signal, phase_masses=True):
        c_aq, s_g = flash(signal)
        return SimpleNamespace(
            mass=_scalar(1.5 * c_aq.img + 2.0 * s_g.img),
            concentration_aq=c_aq,
            saturation_g=s_g,
        )

    def reduced(factor):
        # Like the real analysis: own copies of the signal functions, same flash
        copies = {label: copy.copy(function) for label, function in functions.items()}
        reduced_shape = (shape[0] // factor, shape[1] // factor)
        return _fake_analysis(
            reduced_shape, copies, flash, has_gas, calls, states, reducible
        )

    return SimpleNamespace(
        calls=calls,
        call_flash_and_mass_analysis=call_flash_and_mass_analysis,
        has_gas=has_gas,
        labels=labels,
        color_analysis=SimpleNamespace(base=labels),
        geometry=SimpleNamespace(
            depth=_scalar(np.full(shape, 0.02)),
            porosity=_scalar(np.full(shape, 0.4)),
            weight=_scalar(np.full(shape, 0.008)),
        ),
        evaluation_mode=SimpleNamespace(projects_onto_path=True),
        call_color_interpretation=lambda image: _scalar(image.img.mean(axis=-1)),
        call_pH_analysis=call_pH_analysis,
        signal_model=SimpleNamespace(model=[None, functions]),
        flash=flash,
        fluid_model=SimpleNamespace(density_gas=2.0, aqueous_capacity=1.5),
        expert_knowledge_adapter=None,
        contour_smoother=None,
        update_state=lambda date, protocol=None: states.append(date),
        supports_reduction=reducible,
        reduced=reduced,
    )


def _make_state(
    has_gas=True,
    num_images=3,
    resolution_level=1,
    reducible=True,
    images=None,
    stored=False,
):
    """State for two labels (left: 1, right: 2) and images of uniform brightness."""
    functions = {
        1: FakeSignalFunction([0.0, 0.5, 1.0]),
        2: FakeSignalFunction([0.0, 1.0, 2.0]),
    }
    states = []
    analysis = _fake_analysis(
        SHAPE, functions, FakeFlash(), has_gas, [], states, reducible
    )
    if images is None:
        images = [
            darsia.Image(np.full((*SHAPE, 3), 0.25 * (i + 1)), **META)
            for i in range(num_images)
        ]
    experiment = SimpleNamespace(pressure_temperature_protocol=None)
    state = CalibrationState(
        analysis, images, experiment, resolution_level=resolution_level, stored=stored
    )
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


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_images_in_native_type_give_the_images_of_float_images(dtype):
    """Images may be stored as the integers of the files; what the state derives from
    them (display copies, reduced images) must be as for the float images."""
    rng = np.random.default_rng(0)
    native = [
        darsia.Image(
            rng.integers(0, np.iinfo(dtype).max, (*SHAPE, 3), dtype=dtype), **META
        )
        for _ in range(2)
    ]
    floats = [image.img_as(float) for image in native]
    state, _, _ = _make_state(resolution_level=2, images=native)
    reference, _, _ = _make_state(resolution_level=2, images=floats)
    assert state.resolution_level == reference.resolution_level == 2
    derived = state.coarse_images + state.level.images
    expected = reference.coarse_images + reference.level.images
    assert len(derived) == len(expected) == 4
    dtypes = [np.uint8, np.uint8, np.float64, np.float64]  # display copies 8-bit
    for image, expected_image, expected_dtype in zip(derived, expected, dtypes):
        assert image.img.dtype == expected_image.img.dtype == expected_dtype
        np.testing.assert_array_equal(image.img, expected_image.img)
    # The storage stays native; floats on request
    assert state.images[0].img.dtype == dtype
    assert state.float_image(0).img.dtype == np.float64
    np.testing.assert_array_equal(state.float_image(0).img, floats[0].img)


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


def test_maps_reuse_the_analysis_until_something_changes():
    state, analysis, _ = _make_state()
    state.compute_maps()
    state.compute_maps()
    assert len(analysis.calls) == 1  # same image, same parameters
    state.apply_parameters([0.0, 0.2, 0.4], [0.0, 1.0, 1.0, 2.0], label=1)
    maps = state.compute_maps()
    assert len(analysis.calls) == 2  # new parameters
    assert np.isclose(maps["signals"]["pH"].img[:, :100], 0.25 * 0.4).all()
    state.next_image()
    state.compute_maps()
    assert len(analysis.calls) == 3  # new image
    state.apply_parameters([0.0, 0.2, 0.4], [0.0, 1.0, 1.5, 2.0], label=1)
    state.compute_maps()
    assert len(analysis.calls) == 4  # new flash parameters


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
    image.img[..., 0] = 255  # reddish (display copies are 8-bit)
    highlighted = state.label_highlight(image, 1)
    inside = state.coarse_labels.img == 1
    np.testing.assert_array_equal(highlighted[inside], image.as_float().img[inside])
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


@pytest.mark.parametrize("has_gas", [True, False])
def test_unsaved_changes(has_gas):
    """Closing the window asks to save only if the parameters differ from the stored
    calibration."""
    state, _, _ = _make_state(has_gas=has_gas, stored=True)
    flash = [0.0, 1.0, 1.0, 2.0]
    assert not state.has_unsaved_changes()

    # A changed signal value, or flash bound, is unsaved
    assert state.apply_parameters([0.0, 1.5, 2.0], flash, label=2)
    assert state.has_unsaved_changes()
    state.mark_saved()
    assert not state.has_unsaved_changes()
    if has_gas:
        assert state.apply_parameters([0.0, 1.5, 2.0], [0.0, 0.8, 1.0, 2.0], label=2)
        assert state.has_unsaved_changes()
        state.apply_parameters([0.0, 1.5, 2.0], flash, label=2)

    # Back to the saved values: nothing to save
    assert not state.has_unsaved_changes()

    # A calibration which is not stored (e.g. new) is unsaved from the start
    new_state, _, _ = _make_state(has_gas=has_gas, stored=False)
    assert new_state.has_unsaved_changes()
    new_state.mark_saved()
    assert not new_state.has_unsaved_changes()


def test_starts_at_the_requested_resolution_level():
    state, analysis, _ = _make_state(resolution_level=2)
    assert state.available_levels == (1, 2, 4, 8, 16)
    assert state.resolution_level == 2
    assert state.labels.img.shape == (4, 6)
    assert list(state._levels) == [2]  # full resolution is not prepared
    maps = state.compute_maps()
    assert analysis.calls == [(4, 6)]
    # The display keeps its size and the values are those of full resolution
    signal = maps["signals"]["pH"].img
    assert signal.shape == state.coarse_shape
    assert np.isclose(signal[:, :100], 0.25).all()
    assert np.isclose(signal[:, -100:], 0.5).all()
    assert state.select_label_in((slice(0, 4), slice(4, 6))) == 2


def test_switching_the_resolution_level():
    state, analysis, _ = _make_state()
    state.compute_maps()
    state.mass_analysis_done = True
    assert state.set_resolution(1) is False
    assert state.set_resolution(4) is True
    assert state.resolution_level == 4 and state.labels.img.shape == (2, 3)
    assert state.maps is None and state.mass_analysis_done is False
    state.compute_maps()
    assert analysis.calls == [SHAPE, (2, 3)]
    # Parameters reach all levels, also when set at a reduced level
    state.apply_parameters([0.0, 0.2, 0.4], [0.0, 1.0, 1.0, 2.0], label=1)
    reduced_maps = state.compute_maps()
    assert np.isclose(reduced_maps["signals"]["pH"].img[:, :50], 0.1).all()
    assert analysis.signal_model.model[1][1].values == [0.0, 0.2, 0.4]
    assert state.set_resolution(1) is True
    full_maps = state.compute_maps()
    assert np.isclose(full_maps["signals"]["pH"].img[:, :100], 0.1).all()
    # A level built later starts from the current values
    state.set_resolution(2)
    assert np.isclose(state.compute_maps()["signals"]["pH"].img[:, :100], 0.1).all()


def test_only_full_resolution_without_reduction():
    state, _, _ = _make_state(resolution_level=4, reducible=False)
    assert state.available_levels == (1,)
    assert state.resolution_level == 1
    assert state.set_resolution(8) is False
