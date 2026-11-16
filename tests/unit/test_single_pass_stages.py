"""The compiled single-pass stages of the color to mass analysis must reproduce the
generic implementation exactly.

`HeterogeneousColorToMassAnalysis` evaluates color interpretation, signal and flash
with mass by kernels (`color_path_kernels`, `signal_kernels`, `mass_kernels`) if its
components are in their standard form, and by the generic classes otherwise. The
kernels mirror the arithmetic of these classes; this test fails if one side changes
without the other.

"""

import numpy as np
import pytest

import darsia
import darsia.signals.color.color_path_kernels as color_kernels
from darsia.presets.workflows.analysis.expert_knowledge import ExpertKnowledgeAdapter
from darsia.presets.workflows.heterogeneous_color_to_mass_analysis import (
    HeterogeneousColorToMassAnalysis,
)

pytestmark = pytest.mark.skipif(
    not color_kernels.NUMBA_AVAILABLE, reason="The kernels require numba."
)

SHAPE = (24, 36)
META = {"dimensions": [0.6, 0.9], "space_dim": 2}
FAST = ("_fast_color_interpretation", "_fast_signal", "_fast_flash_and_mass")
RELATIVE = darsia.ColorMode.RELATIVE

# Key colors (relative to the baseline) of the color path of each label
PATHS = {
    0: [[0.0, 0.0, 0.0], [0.3, -0.1, -0.2]],
    1: [[0.0, 0.0, 0.0], [0.2, 0.1, -0.1], [0.5, -0.2, -0.3]],
    2: [[0.0, 0.0, 0.0], [0.1, 0.2, 0.0], [0.3, 0.3, -0.2], [0.6, 0.1, -0.4]],
}
# Signal at the key colors of each label
VALUES = {0: [0.0, 0.4], 1: [0.0, 0.3, 0.9], 2: [0.0, 0.2, 0.7, 1.2]}


class Regions:
    """Expert knowledge with given masks (the real rule of applying them)."""

    def __init__(self, keep_aq, keep_g):
        self.masks = {"concentration_aq": keep_aq, "saturation_g": keep_g}

    def mask_for(self, image, mode):
        return self.masks[mode]

    apply = ExpertKnowledgeAdapter.apply


def _labels() -> darsia.ScalarImage:
    array = np.zeros(SHAPE, dtype=int)
    first, second = SHAPE[1] // 3, 2 * SHAPE[1] // 3
    array[:, first:] = 1
    array[:, second:] = 2
    return darsia.ScalarImage(array, **META)


def _spectrum() -> darsia.ColorSpectrum:
    """Colors of the baseline: the bins around zero relative color."""
    spectrum = np.zeros((7, 7, 7), dtype=bool)
    spectrum[2:4, 2:5, 3:5] = True
    return darsia.ColorSpectrum(
        base_color=np.zeros(3),
        spectrum=spectrum,
        histogram=spectrum.astype(float),
        color_range=darsia.ColorRange(
            min_color=np.array([-0.3, -0.4, -0.5]),
            max_color=np.array([0.7, 0.5, 0.3]),
            color_mode=RELATIVE,
        ),
    )


def _analysis(ignore_mode: str = "norm", expert=None, ignore_labels=None):
    rng = np.random.default_rng(1)
    baseline = darsia.OpticalImage(
        0.3 + 0.3 * rng.random((*SHAPE, 3)), color_space="RGB", **META
    )
    labels = _labels()
    color_paths = {
        label: darsia.ColorPath(
            base_color=np.zeros(3),
            relative_colors=[np.array(color) for color in colors],
        )
        for label, colors in PATHS.items()
    }
    interpretations = {
        label: darsia.ColorPathInterpolation(
            color_path=path,
            color_mode=RELATIVE,
            values=path.equidistant_distances,
            ignore_spectrum=_spectrum(),
            ignore_mode=ignore_mode,
        )
        for label, path in color_paths.items()
    }
    signal_functions = {
        label: darsia.PWTransformation(path.equidistant_distances, VALUES[label])
        for label, path in color_paths.items()
    }
    fluid_model = darsia.FluidMassAnalysis(
        baseline,
        atmospheric_pressure=1.01,
        atmospheric_temperature=20.0,
        atmospheric_temperature_gradient=3.0,
        system=darsia.FluidSystem.two_phase(),
    )
    analysis = HeterogeneousColorToMassAnalysis(
        baseline=baseline,
        labels=labels,
        color_mode=RELATIVE,
        color_path_interpretation=interpretations,
        signal_functions=signal_functions,
        flash=darsia.SimpleFlash(0.05, 0.6, 0.5, 1.1),
        fluid_model=fluid_model,
        geometry=None,
        ignore_labels=ignore_labels,
        expert_knowledge_adapter=expert,
    )
    return analysis, baseline


def _image(baseline: darsia.Image) -> darsia.Image:
    """Baseline plus colors along (and beyond) the color paths, noise and clean parts."""
    rng = np.random.default_rng(2)
    labels = _labels().img
    position = rng.uniform(-0.3, 1.4, SHAPE)
    change = np.zeros((*SHAPE, 3))
    for label, colors in PATHS.items():
        nodes = np.linspace(0, 1, len(colors))
        for channel in range(3):
            values = np.interp(position, nodes, np.array(colors)[:, channel])
            change[..., channel] = np.where(
                labels == label, values, change[..., channel]
            )
    change += 0.05 * rng.standard_normal(change.shape)
    change[: SHAPE[0] // 4] *= 0.1  # nearly clean
    change[0] = 0.0  # clean
    return darsia.OpticalImage(baseline.img + change, color_space="RGB", **META)


def _stages(analysis, image, fast: bool, monkeypatch) -> tuple[dict, list]:
    """Results of all stages, and which single-pass stages were used."""
    used = []
    with monkeypatch.context() as patch:
        for name in FAST:
            original = getattr(analysis, name)

            def stage(*args, name=name, original=original):
                result = original(*args) if fast else None
                used.append(result is not None)
                return result

            patch.setattr(analysis, name, stage)
        interpretation = analysis.call_color_interpretation(image)
        signal = analysis.call_pH_analysis(interpretation)
        result = analysis.call_flash_and_mass_analysis(signal)
    fields = {
        "interpretation": interpretation,
        "signal": signal,
        "concentration_aq": result.concentration_aq,
        "saturation_g": result.saturation_g,
        "mass": result.mass,
        "mass_g": result.mass_g,
        "mass_aq": result.mass_aq,
    }
    return fields, used


def _assert_identical(analysis, image, monkeypatch, expected_used=(True, True, True)):
    generic, _ = _stages(analysis, image, False, monkeypatch)
    single_pass, used = _stages(analysis, image, True, monkeypatch)
    assert tuple(used) == tuple(expected_used)
    for name, reference in generic.items():
        result = single_pass[name]
        assert type(result) is type(reference), name
        assert result.img.dtype == reference.img.dtype, name
        assert result.img.shape == reference.img.shape, name
        # Bitwise, including signs of zero
        assert result.img.tobytes() == reference.img.tobytes(), name
        assert np.array_equal(result.dimensions, reference.dimensions), name
    # The test image must exercise the stages
    assert np.ptp(generic["signal"].img) > 0.5
    assert np.any(generic["saturation_g"].img > 0)
    assert np.any(generic["concentration_aq"].img == 0)


@pytest.mark.parametrize("ignore_mode", ["norm", "spectrum"])
def test_single_pass_stages_reproduce_generic_stages(ignore_mode, monkeypatch):
    analysis, baseline = _analysis(ignore_mode)
    _assert_identical(analysis, _image(baseline), monkeypatch)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_single_pass_stages_with_images_in_native_type(dtype, monkeypatch):
    """Images in the type of the corrections (8-bit, or float32 after a type
    correction) give the results of the image converted beforehand."""
    analysis, baseline = _analysis()
    array = _image(baseline).img
    if dtype is np.uint8:
        array = np.round(255 * np.clip(array, 0, 1)).astype(np.uint8)
    else:
        array = array.astype(np.float32)
    native = darsia.OpticalImage(array, color_space="RGB", **META)
    assert native.img.dtype == dtype
    _assert_identical(analysis, native, monkeypatch)

    # The same as for the image converted as at the end of reading
    converted = native.img_as(float)
    reference, _ = _stages(analysis, converted, False, monkeypatch)
    result, used = _stages(analysis, native, True, monkeypatch)
    assert used == [True, True, True]
    for name, field in reference.items():
        assert result[name].img.tobytes() == field.img.tobytes(), name


def test_single_pass_stages_with_expert_knowledge(monkeypatch):
    keep_aq = np.zeros(SHAPE, dtype=bool)
    keep_aq[: SHAPE[0] // 2] = True
    keep_g = np.ones(SHAPE, dtype=bool)
    keep_g[:, : SHAPE[1] // 2] = False
    for regions in (Regions(keep_aq, keep_g), Regions(None, keep_g)):
        analysis, baseline = _analysis(expert=regions)
        _assert_identical(analysis, _image(baseline), monkeypatch)


def test_single_pass_stages_with_ignored_label(monkeypatch):
    analysis, baseline = _analysis(ignore_labels=[1])
    _assert_identical(analysis, _image(baseline), monkeypatch)


def test_single_pass_stages_with_restoration(monkeypatch):
    analysis, baseline = _analysis()
    analysis.signal_model.restoration = lambda signal: 0.5 * (
        signal + np.roll(signal, 1, axis=0)
    )
    _assert_identical(analysis, _image(baseline), monkeypatch)


def test_single_pass_stages_after_changing_parameters_and_state(monkeypatch):
    analysis, baseline = _analysis()
    analysis.signal_model.model[1][2].update(values=[0.0, 0.1, 0.5, 0.8])
    analysis.flash.update(min_value_aq=0.0, max_value_aq=0.4, min_value_g=0.4)
    analysis.fluid_model.update_state(
        atmospheric_pressure=0.98, atmospheric_temperature=24.0
    )
    _assert_identical(analysis, _image(baseline), monkeypatch)


def test_generic_stages_are_used_for_other_setups(monkeypatch):
    """Setups beyond the standard form must not be evaluated by the kernels."""
    # Signal functions which are not piecewise linear
    analysis, baseline = _analysis()
    image = _image(baseline)
    analysis.set_evaluation_mode(darsia.SplineEvaluation())
    _assert_identical(analysis, image, monkeypatch, (True, False, True))

    # Fluid properties which vary along rows
    analysis, _ = _analysis()
    density = np.array(analysis.fluid_model.density_gas)
    density[:, ::2] *= 1.01
    analysis.fluid_model.density_gas = density
    _assert_identical(analysis, image, monkeypatch, (True, True, False))

    # Flash with a restoration
    analysis, _ = _analysis()
    analysis.flash.restoration = lambda field: field
    _assert_identical(analysis, image, monkeypatch, (True, True, False))

    # Analyses with further steps
    analysis, _ = _analysis()
    analysis.color_analysis.balancing = lambda signal: signal
    analysis.signal_model.balancing = lambda signal: signal
    _assert_identical(analysis, image, monkeypatch, (False, False, True))
