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

    def __init__(self, keep_aq, keep_g, bounds=None):
        self.masks = {"concentration_aq": keep_aq, "saturation_g": keep_g}
        self.bounds = {} if bounds is None else bounds

    def mask_for(self, image, mode):
        return self.masks[mode]

    constrains = ExpertKnowledgeAdapter.constrains
    constrain = ExpertKnowledgeAdapter.constrain
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


def _analysis(
    ignore_mode: str = "norm", expert=None, ignore_labels=None, geometry=None
):
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
        geometry=geometry,
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


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_single_pass_stages_with_images_in_native_type(dtype, monkeypatch):
    """Images in the type of the corrections (8-bit or 16-bit integers of the files, or
    float32 after a type correction) give the results of the image converted
    beforehand."""
    analysis, baseline = _analysis()
    array = _image(baseline).img
    if np.issubdtype(dtype, np.integer):
        array = np.round(np.iinfo(dtype).max * np.clip(array, 0, 1)).astype(dtype)
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


def test_images_of_other_types_are_converted_beforehand(monkeypatch):
    """Types the kernel cannot read exactly (e.g. signed integers, which map to
    [-1, 1]) take the generic conversion and still give its results."""
    analysis, baseline = _analysis()
    array = np.round(32767 * np.clip(_image(baseline).img, 0, 1)).astype(np.int16)
    native = darsia.OpticalImage(array, color_space="RGB", **META)
    _assert_identical(analysis, native, monkeypatch, (False, True, True))


def test_single_pass_stages_with_expert_knowledge(monkeypatch):
    keep_aq = np.zeros(SHAPE, dtype=bool)
    keep_aq[: SHAPE[0] // 2] = True
    keep_g = np.ones(SHAPE, dtype=bool)
    keep_g[:, : SHAPE[1] // 2] = False
    for regions in (Regions(keep_aq, keep_g), Regions(None, keep_g)):
        analysis, baseline = _analysis(expert=regions)
        _assert_identical(analysis, _image(baseline), monkeypatch)


def test_single_pass_stages_with_bounds(monkeypatch):
    """Bounds which the ramps of the flash meet anyway keep the single pass; others are
    left to the generic path, which applies them."""
    keep_aq = np.zeros(SHAPE, dtype=bool)
    keep_aq[: SHAPE[0] // 2] = True
    met = {"concentration_aq": (0.0, None), "saturation_g": (None, 1.0)}
    analysis, baseline = _analysis(expert=Regions(keep_aq, None, met))
    image = _image(baseline)
    _assert_identical(analysis, image, monkeypatch)

    analysis, _ = _analysis(
        expert=Regions(keep_aq, None, {"concentration_aq": (0, 0.5)})
    )
    _assert_identical(analysis, image, monkeypatch, (True, True, False))
    fields, _ = _stages(analysis, image, True, monkeypatch)
    assert fields["concentration_aq"].img.max() == 0.5


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


def _geometry() -> darsia.ExtrudedPorousGeometry:
    """Geometry with a porosity and a depth varying over the image."""
    rng = np.random.default_rng(3)
    return darsia.ExtrudedPorousGeometry(
        porosity=0.2 + 0.2 * rng.random(SHAPE),
        depth=0.01 + 0.01 * rng.random(SHAPE),
        space_dim=2,
        num_voxels=SHAPE,
        dimensions=META["dimensions"],
    )


@pytest.mark.parametrize("with_expert_knowledge", [False, True])
def test_integrated_flash_and_mass_equals_integrated_fields(with_expert_knowledge):
    """The integrals of the mass series, computed without the fields, equal the
    integrals of the fields to the last digit (mirrors `Geometry.integrate`)."""
    expert = None
    if with_expert_knowledge:
        keep_aq = np.zeros(SHAPE, dtype=bool)
        keep_aq[: SHAPE[0] // 2] = True
        keep_g = np.ones(SHAPE, dtype=bool)
        keep_g[:, : SHAPE[1] // 2] = False
        expert = Regions(keep_aq, keep_g)
    analysis, baseline = _analysis(expert=expert)
    geometry = _geometry()
    signal = analysis.call_pH_analysis(
        analysis.call_color_interpretation(_image(baseline))
    )
    result = analysis.call_flash_and_mass_analysis(signal)
    expected = (geometry.integrate(result.mass_g), geometry.integrate(result.mass_aq))
    assert expected[0] > 0 and expected[1] > 0

    integrals = analysis.integrated_flash_and_mass(signal, geometry)

    assert integrals is not None
    assert integrals[0] == expected[0]
    assert integrals[1] == expected[1]

    # Setups beyond the standard form are left to the generic path
    analysis.flash.restoration = lambda field: field
    assert analysis.integrated_flash_and_mass(signal, geometry) is None


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


def test_reduced_analysis_takes_labels_at_the_centers_and_averages_the_geometry():
    """Labels (also of the expert knowledge) at the center of each reduced pixel,
    depth and porosity averaged."""
    rng = np.random.default_rng(3)
    depth = darsia.ScalarImage(rng.random(SHAPE), **META)
    porosity = darsia.ScalarImage(rng.random(SHAPE), **META)
    geometry = darsia.ExtrudedPorousGeometry(
        depth=depth, porosity=porosity, **depth.shape_metadata()
    )
    numbered = darsia.ScalarImage(np.arange(np.prod(SHAPE)).reshape(SHAPE), **META)
    expert = ExpertKnowledgeAdapter(labels_image=numbered)
    analysis, _ = _analysis(expert=expert, geometry=geometry)

    reduced = analysis.reduced(4)

    centers = np.s_[2::4, 2::4]
    assert np.array_equal(reduced.labels.img, analysis.labels.img[centers])
    assert reduced.labels.img.dtype == analysis.labels.img.dtype
    assert np.array_equal(
        reduced.expert_knowledge_adapter.labels_image.img, numbered.img[centers]
    )
    for reduced_field, field in (
        (reduced.geometry.depth, depth),
        (reduced.geometry.porosity, porosity),
    ):
        expected = darsia.resize(field, shape=(6, 9), interpolation="inter_area")
        assert np.allclose(reduced_field.img, expected.img)
