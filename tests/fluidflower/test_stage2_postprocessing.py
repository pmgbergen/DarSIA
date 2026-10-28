"""Stage 2 post-processing of stored per-frame maps."""

import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower.concentration_flags import dissolved_interval
from darsia.fluidflower.cycle_budget import (
    CycleBudgetInputs,
    cycle_budget,
    free_gas_mass_g,
    injected_mass_g,
)
from darsia.fluidflower.stage2_postprocessing import (
    BudgetTerms,
    FrameInputs,
    Stage2Config,
    StoredField,
    StoredFrameMapping,
    blue_fraction_from_colour,
    load_npz_fields,
    process_cycle,
    process_frame,
    read_stage1_frame,
)
from darsia.fluidflower.titration import (
    M_CO2_G_PER_MOL,
    IndicatorRecipe,
    TitrationModel,
)

# Test values only; the series values are the experimenter's decision.
PARAMS = dict(
    naoh_g_per_l=0.05,
    btb_g_per_l=0.452,
    btb_form="sodium_salt",
    pka_indicator=7.10,
    temperature_c=25.0,
    blue_fraction_floor=1e-9,
    clip_negative_dic=True,
    f_min=0.05,
    p_co2_atm=1.0,
    quantified_labels=(2, 3),
    injection_gas_density_kg_per_m3=1.872,
    free_gas_density_kg_per_m3=1.842,
    water_density_kg_per_m3=997.0,
    gas_density_reference="test values: injection 15 C 1 atm, rig 20 C 1 atm",
)
SAND, WATER = 2, 1


def config(**changes):
    return Stage2Config(**{**PARAMS, **changes})


def model(**changes):
    """The titration model built directly, independent of Stage2Config."""
    p = {**PARAMS, **changes}
    recipe = IndicatorRecipe(p["naoh_g_per_l"], p["btb_g_per_l"], form=p["btb_form"])
    return TitrationModel(
        recipe,
        temperature_c=p["temperature_c"],
        pka_indicator=p["pka_indicator"],
        blue_fraction_floor=p["blue_fraction_floor"],
        clip_negative_dic=p["clip_negative_dic"],
    )


def frame(
    f,
    labels=None,
    volume=None,
    *,
    frame_id="t0",
    valid=None,
    gas=None,
    outside=None,
    area=None,
):
    f = np.asarray(f, dtype=float)
    labels = (
        np.full(f.shape, SAND, np.uint8)
        if labels is None
        else np.asarray(labels, np.uint8)
    )
    volume = np.full(f.shape, 1e-6) if volume is None else np.asarray(volume, float)
    return FrameInputs(
        frame_id=frame_id,
        blue_fraction=f,
        labels=labels,
        pore_volume_m3=volume,
        valid_colour_mask=valid,
        gas_mask=gas,
        outside_mask=outside,
        pixel_area_m2=area,
    )


def grams(dic_mol_per_l, volume_m3):
    """Integration by hand: mol/L x 1000 L/m3 x m3 x g/mol."""
    return np.asarray(dic_mol_per_l) * 1000.0 * np.asarray(volume_m3) * M_CO2_G_PER_MOL


def dic(f, **changes):
    return float(model(**changes).dic_from_blue_fraction(f))


def test_uniform_blue_fraction_reproduces_the_scalar_titration_per_pixel():
    result = process_frame(config(), frame(np.full((3, 4), 0.5)))
    expected = dic(0.5)
    for name in ("dic_mol_per_l", "dic_lower_mol_per_l", "dic_upper_mol_per_l"):
        assert np.all(getattr(result, name) == expected)
    assert not result.saturated.any() and not result.not_quantified.any()
    assert np.allclose(result.mass_lower_g, grams(expected, 1e-6), rtol=1e-12, atol=0)
    total = result.summary.total
    assert total.mass_lower_g == total.mass_upper_g
    assert total.mass_lower_g == pytest.approx(12 * grams(expected, 1e-6), rel=1e-12)
    assert (
        total.pixels,
        total.quantified_pixels,
        total.saturated_pixels,
        total.not_quantified_pixels,
    ) == (12, 12, 0, 0)
    assert result.summary.saturated_fraction == 0.0
    assert result.summary.not_quantified_fraction == 0.0
    assert math.isnan(total.not_quantified_area_m2)  # no pixel areas given: unknown


@pytest.mark.parametrize(
    "changes",
    [
        dict(pka_indicator=7.3),
        dict(temperature_c=20.0),
        dict(btb_form="free_acid"),
        dict(naoh_g_per_l=0.06),
        dict(clip_negative_dic=False),
        dict(blue_fraction_floor=1e-3, clip_negative_dic=False),
    ],
)
def test_every_chemistry_parameter_reaches_the_titration_model(changes):
    f = np.array([[1.0, 0.9, 0.5, 0.2]])
    result = process_frame(config(**changes), frame(f))
    assert np.array_equal(
        result.dic_mol_per_l, model(**changes).dic_from_blue_fraction(f)
    )
    assert not np.allclose(result.dic_mol_per_l, model().dic_from_blue_fraction(f))


def test_masks_exclude_pixels_from_maps_and_masses():
    labels = np.array([[SAND, SAND, SAND], [SAND, WATER, 4]])
    gas = np.array([[True, False, False], [False, False, False]])
    outside = np.array([[False, True, False], [False, False, False]])
    result = process_frame(
        config(), frame(np.full((2, 3), 0.5), labels, gas=gas, outside=outside)
    )
    excluded = np.array([[True, True, False], [False, True, True]])
    assert np.array_equal(result.not_quantified, excluded)
    for name in ("dic_mol_per_l", "mass_lower_g", "mass_upper_g"):
        values = getattr(result, name)
        assert np.isnan(values[excluded]).all() and np.isfinite(values[~excluded]).all()
    s = result.summary
    assert s.total.mass_lower_g == pytest.approx(2 * grams(dic(0.5), 1e-6), rel=1e-12)
    assert (
        s.total.pixels,
        s.total.quantified_pixels,
        s.total.not_quantified_pixels,
    ) == (4, 2, 2)
    assert s.not_quantified_fraction == pytest.approx(0.5) and s.sand_pixels == 4
    assert set(s.by_label) == {SAND, WATER, 4}
    water = s.by_label[WATER]  # no colour path: not quantified, mass unknown
    assert math.isnan(water.mass_lower_g) and math.isnan(water.mass_upper_g)
    assert (water.pixels, water.quantified_pixels, water.not_quantified_pixels) == (
        1,
        0,
        1,
    )


def test_invalid_colour_is_not_quantified_and_never_zero_mass():
    f = np.array([[0.5, np.nan, 0.5, 0.5]])
    valid = np.array([[True, True, False, True]])
    labels = np.array([[SAND, SAND, 3, SAND]])
    result = process_frame(
        config(), frame(f, labels, valid=valid, area=np.full((1, 4), 2e-6))
    )
    assert result.not_quantified.tolist() == [[False, True, True, False]]
    assert (
        np.isnan(result.mass_lower_g[0, 1:3]).all()
        and np.isnan(result.mass_upper_g[0, 1:3]).all()
    )
    assert np.isnan(result.dic_mol_per_l[0, 1:3]).all()
    s = result.summary
    label3 = s.by_label[
        3
    ]  # sand whose only pixel has invalid colour: unknown, not zero
    assert math.isnan(label3.mass_lower_g) and label3.not_quantified_pixels == 1
    assert s.total.quantified_pixels == 2
    assert s.total.mass_lower_g == pytest.approx(2 * grams(dic(0.5), 1e-6), rel=1e-12)
    assert s.total.not_quantified_area_m2 == pytest.approx(4e-6)
    assert s.not_quantified_fraction == pytest.approx(0.5)


def test_saturated_pixels_take_the_interval_of_dissolved_interval():
    f = np.array([[0.01, 0.05, 0.3, 0.0]])
    volume = np.array([[1e-6, 2e-6, 1e-6, 3e-6]])
    result = process_frame(config(), frame(f, volume=volume))
    m = model()
    reference = dissolved_interval(
        f, m, f_min=0.05, p_co2_atm=1.0, valid_mask=np.ones(f.shape, bool)
    )
    assert result.saturated.tolist() == [[True, False, False, True]]
    assert np.array_equal(result.dic_lower_mol_per_l, reference.lower)
    assert np.array_equal(result.dic_upper_mol_per_l, reference.upper)
    assert result.dic_lower_mol_per_l[0, 0] == pytest.approx(dic(0.05))
    assert result.dic_upper_mol_per_l[0, 0] == pytest.approx(m.saturation_dic(1.0))
    # the colour no longer measures DIC there: no point value
    assert np.isnan(result.dic_mol_per_l[0, [0, 3]]).all()
    assert np.isfinite(result.dic_mol_per_l[0, 1:3]).all()
    s = result.summary
    assert s.total.mass_lower_g == pytest.approx(
        grams(reference.lower, volume).sum(), rel=1e-12
    )
    assert s.total.mass_upper_g == pytest.approx(
        grams(reference.upper, volume).sum(), rel=1e-12
    )
    assert s.total.saturated_pixels == 2 and s.saturated_fraction == pytest.approx(0.5)
    # f_min and the pressure of the solubility bound are the configuration's
    other = process_frame(config(f_min=0.4, p_co2_atm=0.5), frame(f, volume=volume))
    assert other.saturated.all()
    assert other.dic_upper_mol_per_l[0, 2] == pytest.approx(m.saturation_dic(0.5))
    assert other.dic_lower_mol_per_l[0, 2] == pytest.approx(dic(0.4))


def test_masses_integrate_with_the_stored_pore_volume():
    f = np.array([[0.9, 0.5], [0.2, 0.7]])
    volume = np.array([[1e-6, 2e-6], [3e-6, 4e-6]])
    labels = np.array([[SAND, 3], [SAND, WATER]])
    box = np.array([[True, True], [False, True]])  # its water pixel is not sand
    result = process_frame(config(), frame(f, labels, volume), regions={"box": box})
    expected = grams(model().dic_from_blue_fraction(f), volume)
    sand = labels != WATER
    assert np.allclose(result.mass_lower_g[sand], expected[sand], rtol=1e-12, atol=0)
    s = result.summary
    assert s.total.mass_lower_g == pytest.approx(expected[sand].sum(), rel=1e-12)
    assert s.by_label[SAND].mass_lower_g == pytest.approx(
        expected[0, 0] + expected[1, 0]
    )
    assert s.by_label[3].mass_upper_g == pytest.approx(expected[0, 1])
    assert s.by_region["box"].mass_lower_g == pytest.approx(
        expected[0, 0] + expected[0, 1]
    )
    assert s.by_region["box"].pixels == 2
    # the same mass with porosity, depth and area as separate factors
    split = TitrationModel.mass_per_pixel_g(
        model().dic_from_blue_fraction(f), 0.4, 0.01, volume / (0.4 * 0.01)
    )
    assert np.allclose(expected, split, rtol=1e-12, atol=0)


def test_a_cycle_of_frames_produces_budget_rows():
    shape = (2, 3)
    baseline = frame(np.full(shape, 0.9), frame_id="c1_baseline")
    f1 = np.full(shape, 0.9)
    f1[0, :2] = 0.3
    f2 = np.full(shape, 0.9)
    f2[0, 0], f2[0, 1] = 0.01, 0.3
    gas2 = np.zeros(shape, bool)
    gas2[1, 2] = True  # gas where the baseline had dissolved CO2: outside the support
    intervals = ((10.0, 20.0),)
    pairs = [
        (
            frame(f1, frame_id="t1"),
            BudgetTerms(
                injection_intervals=intervals,
                displaced_water_g=100.0,
                left_overflow_g=0.0,
            ),
        ),
        (
            frame(f2, frame_id="t2", gas=gas2),
            BudgetTerms(
                injection_intervals=intervals,
                displaced_water_g=None,
                left_overflow_g=None,
                quality_flags=("no scale record",),
            ),
        ),
    ]
    out = process_cycle(
        config(), cycle_run="e2_c1", baseline=baseline, frames=(pair for pair in pairs)
    )
    assert out.baseline.frame_id == "c1_baseline" and len(out.frames) == 2
    assert [row["frame_id"] for row in out.rows] == ["t1", "t2"]
    base = grams(dic(0.9), 1e-6)
    injected = injected_mass_g(10.0, 20.0, 1.872)
    new1 = 2 * (grams(dic(0.3), 1e-6) - base)
    sat_lower = grams(dic(0.05), 1e-6)
    sat_upper = grams(model().saturation_dic(1.0), 1e-6)
    new2 = (
        sat_lower - base + grams(dic(0.3), 1e-6) - base,
        sat_upper - base + grams(dic(0.3), 1e-6) - base,
    )
    expected = [
        cycle_budget(
            CycleBudgetInputs(
                "e2_c1",
                injected,
                new1,
                new1,
                free_gas_g=free_gas_mass_g(100.0, 1.842, 997.0),
                left_overflow_g=0.0,
                background_dissolved_g=6 * base,
            )
        ),
        cycle_budget(
            CycleBudgetInputs(
                "e2_c1",
                injected,
                *new2,
                background_dissolved_g=6 * base,
                quality_flags=("no scale record",),
            )
        ),
    ]
    for row, budget in zip(out.rows, expected, strict=True):
        for key, value in budget.as_row().items():
            if isinstance(value, float) and math.isnan(value):
                assert math.isnan(row[key]), key
            elif isinstance(value, float):
                assert row[key] == pytest.approx(value, rel=1e-9, abs=1e-15), key
            else:
                assert row[key] == value, key
        assert row["gas_density_reference"] == PARAMS["gas_density_reference"]
        assert row["injected_volume_ml"] == 200.0
        assert (
            row["background_dissolved_lower_g"] == row["background_dissolved_upper_g"]
        )
    first, second = out.rows
    assert first["free_gas_known"] and first["overflow_known"]
    assert not second["free_gas_known"] and math.isnan(second["free_gas_g"])
    assert first["new_dissolved_support_pixels"] == 6
    assert second["new_dissolved_support_pixels"] == 5
    assert second["saturated_fraction"] == pytest.approx(1 / 6)
    assert second["not_quantified_fraction"] == pytest.approx(1 / 6)


def test_a_saturated_baseline_leaves_the_background_as_bounds_only():
    baseline = frame(np.array([[0.01, 0.9]]), frame_id="b")
    terms = BudgetTerms(
        injection_intervals=((10.0, 20.0),),
        displaced_water_g=None,
        left_overflow_g=None,
    )
    out = process_cycle(
        config(),
        cycle_run="c",
        baseline=baseline,
        frames=[(frame(np.array([[0.01, 0.5]]), frame_id="t1"), terms)],
    )
    row = out.rows[0]
    assert math.isnan(row["background_dissolved_g"])
    assert row["background_dissolved_lower_g"] < row["background_dissolved_upper_g"]
    # saturated in both: the new dissolved mass of that pixel spans [lower-upper, upper-lower]
    width = grams(model().saturation_dic(1.0), 1e-6) - grams(dic(0.05), 1e-6)
    assert row["dissolved_new_upper_g"] - row["dissolved_new_lower_g"] == pytest.approx(
        2 * width
    )


def test_a_cycle_refuses_frames_whose_new_dissolved_mass_is_unknown():
    baseline = frame(np.full((1, 2), 0.9), frame_id="b")
    terms = BudgetTerms(
        injection_intervals=((10.0, 20.0),),
        displaced_water_g=None,
        left_overflow_g=None,
    )
    run = dict(cycle_run="c", baseline=baseline)
    with pytest.raises(ValueError, match="quantified in both"):
        process_cycle(
            config(),
            frames=[(frame(np.full((1, 2), np.nan), frame_id="t1"), terms)],
            **run,
        )
    with pytest.raises(ValueError, match="grid"):
        process_cycle(
            config(),
            frames=[(frame(np.full((2, 2), 0.5), frame_id="t1"), terms)],
            **run,
        )
    twice = frame(np.full((1, 2), 0.5), frame_id="t1")
    with pytest.raises(ValueError, match="twice"):
        process_cycle(config(), frames=[(twice, terms), (twice, terms)], **run)
    with pytest.raises(ValueError):
        process_cycle(config(), frames=[], **run)
    with pytest.raises(ValueError):
        process_cycle(config(), frames=[(twice, None)], **run)
    for intervals in ((), ((10.0, -1.0),), ((10.0,),), ((10.0, float("nan")),)):
        with pytest.raises(ValueError):
            BudgetTerms(
                injection_intervals=intervals,
                displaced_water_g=None,
                left_overflow_g=None,
            )
    for flags in ("one flag", (1,), None):
        with pytest.raises(ValueError):
            replace(terms, quality_flags=flags)
    for unknown_is_not_infinite in (
        replace(terms, displaced_water_g=math.inf),
        replace(terms, left_overflow_g=math.inf),
    ):
        with pytest.raises(ValueError):
            process_cycle(config(), frames=[(twice, unknown_is_not_infinite)], **run)


@pytest.mark.parametrize("name", sorted(PARAMS))
def test_every_parameter_is_required(name):
    values = {key: value for key, value in PARAMS.items() if key != name}
    with pytest.raises(TypeError):
        Stage2Config(**values)
    with pytest.raises(ValueError, match=name):
        Stage2Config.from_mapping(values)
    with pytest.raises(ValueError):
        Stage2Config(**values, **{name: None})


def test_invalid_parameter_values_are_refused():
    for changes in (
        dict(pka_indicator=float("nan")),
        dict(temperature_c=float("inf")),
        dict(temperature_c=-300.0),
        dict(naoh_g_per_l=True),
        dict(naoh_g_per_l=-1.0),
        dict(btb_form="unknown"),
        dict(clip_negative_dic=1),
        dict(blue_fraction_floor=0.6),
        dict(f_min=0.0),
        dict(f_min=1.0),
        dict(p_co2_atm=0.0),
        dict(p_co2_atm=1e-4),
        dict(quantified_labels=()),
        dict(quantified_labels=(2, 2)),
        dict(quantified_labels=(2.0,)),
        dict(quantified_labels="23"),
        dict(free_gas_density_kg_per_m3=0.0),
        dict(water_density_kg_per_m3=-997.0),
        dict(gas_density_reference=" "),
    ):
        with pytest.raises(ValueError):
            config(**changes)
    with pytest.raises(ValueError, match="unknown"):
        Stage2Config.from_mapping({**PARAMS, "extra": 1})
    assert (
        Stage2Config.from_mapping({**PARAMS, "quantified_labels": [2, 3]}) == config()
    )


def test_frame_inputs_enforce_the_input_contract():
    good = dict(
        frame_id="t",
        blue_fraction=np.full((1, 2), 0.5),
        labels=np.full((1, 2), SAND, np.uint8),
        pore_volume_m3=np.full((1, 2), 1e-6),
        valid_colour_mask=None,
        gas_mask=None,
        outside_mask=None,
        pixel_area_m2=None,
    )
    FrameInputs(**good)
    for changes in (
        dict(blue_fraction=np.array([[0.5, 1.2]])),
        dict(blue_fraction=np.array([[0.5, -0.01]])),
        dict(blue_fraction=np.array([[0.5, np.inf]])),
        dict(blue_fraction=np.array([[0, 1]])),  # integers cannot mark invalid colour
        dict(blue_fraction=np.full((2, 2), 0.5)),
        dict(labels=np.full((1, 2), 2.0)),
        dict(labels=np.full(2, SAND)),
        dict(gas_mask=np.array([[0, 1]])),
        dict(outside_mask=np.zeros((2, 2), bool)),
        dict(pore_volume_m3=np.array([[1e-6, -1e-6]])),
        dict(pore_volume_m3=np.array([[1e-6, np.inf]])),
        dict(pixel_area_m2=np.array([[1, 1]])),
        dict(frame_id=""),
    ):
        with pytest.raises(ValueError):
            FrameInputs(**{**good, **changes})
    with pytest.raises(TypeError):  # every mask is explicit, None included
        FrameInputs(**{key: value for key, value in good.items() if key != "gas_mask"})
    # unknown or zero pore volume is allowed only where nothing is quantified
    for volume in (np.array([[1e-6, np.nan]]), np.array([[1e-6, 0.0]])):
        with pytest.raises(ValueError, match="pore volume"):
            process_frame(config(), FrameInputs(**{**good, "pore_volume_m3": volume}))
        outside = np.array([[False, True]])
        result = process_frame(
            config(),
            FrameInputs(**{**good, "pore_volume_m3": volume, "outside_mask": outside}),
        )
        assert result.not_quantified.tolist() == [[False, True]]
    with pytest.raises(ValueError):
        process_frame(config(), FrameInputs(**good), regions={"box": np.ones((1, 2))})
    with pytest.raises(ValueError):
        process_frame(
            config(), FrameInputs(**{**good, "pixel_area_m2": np.array([[1e-6, 0.0]])})
        )


def test_the_adapter_is_explicit_and_its_output_is_checked():
    colour = np.array([[10.0, 20.0, -1.0]])

    def test_calibration(c):  # a test fixture, not a calibration
        return np.where(c < 0, np.nan, c / 40.0)

    f = blue_fraction_from_colour(colour, test_calibration)
    assert f[0, :2].tolist() == [0.25, 0.5] and np.isnan(f[0, 2])
    for broken in (
        lambda c: c,
        lambda c: np.full(c.shape, np.inf),
        lambda c: c.astype(int),
    ):
        with pytest.raises(ValueError):
            blue_fraction_from_colour(colour, broken)
    with pytest.raises(ValueError):
        blue_fraction_from_colour(colour, None)


def test_the_npz_reader_loads_only_mapped_fields_and_refuses_missing_ones(tmp_path):
    shape = (2, 2)
    products, geometry = tmp_path / "products.npz", tmp_path / "native_geometry.npz"
    np.savez(
        products,
        colour=np.array([[10.0, 20.0], [30.0, np.nan]]),
        phase_labels=np.array([[0, 2], [1, 255]], np.uint8),
        domain_mask=np.array([[True, True], [True, False]]),
        unmapped=np.array([{"never": "loaded"}], dtype=object),
    )  # would need pickle
    np.savez(
        geometry,
        facies=np.full(shape, SAND, np.uint8),
        transport_weight_l=np.full(shape, 2e-3),
        pixel_area_m2=np.full(shape, 1e-6),
    )
    mapping = StoredFrameMapping(
        colour=StoredField("products", "colour"),
        labels=StoredField("geometry", "facies"),
        pore_volume=StoredField("geometry", "transport_weight_l"),
        pore_volume_unit="L",
        valid_colour=None,
        gas=StoredField("products", "phase_labels", true_values=(2,)),
        outside=(
            StoredField("products", "domain_mask", true_values=(False,)),
            StoredField("products", "phase_labels", true_values=(255,)),
        ),
        pixel_area_m2=StoredField("geometry", "pixel_area_m2"),
    )
    archives = {"products": products, "geometry": geometry}

    def adapter(c):  # a test fixture, not a calibration
        return c / 40.0

    inputs = read_stage1_frame(archives, mapping, frame_id="t1", adapter=adapter)
    assert inputs.gas_mask.tolist() == [[False, True], [False, False]]
    assert inputs.outside_mask.tolist() == [[False, False], [False, True]]
    assert inputs.valid_colour_mask is None
    assert np.allclose(inputs.pore_volume_m3, 2e-6, rtol=1e-15, atol=0)
    assert inputs.blue_fraction[0].tolist() == [0.25, 0.5] and np.isnan(
        inputs.blue_fraction[1, 1]
    )
    result = process_frame(config(), inputs)
    assert result.summary.total.quantified_pixels == 2
    assert result.summary.total.not_quantified_area_m2 == pytest.approx(2e-6)

    loaded = load_npz_fields(products, {"f": "colour", "labels": "phase_labels"})
    assert set(loaded) == {"f", "labels"}
    with pytest.raises(ValueError):  # the unmapped member really is unloadable
        load_npz_fields(products, {"x": "unmapped"})
    with pytest.raises(ValueError, match="blue_fraction"):
        load_npz_fields(products, {"f": "blue_fraction"})
    with pytest.raises(ValueError, match="missing_field"):
        read_stage1_frame(
            archives,
            replace(mapping, colour=StoredField("products", "missing_field")),
            frame_id="t1",
            adapter=adapter,
        )
    with pytest.raises(ValueError, match="geometry"):
        read_stage1_frame(
            {"products": products}, mapping, frame_id="t1", adapter=adapter
        )
    with pytest.raises(
        ValueError, match="true_values"
    ):  # a category map is no boolean mask
        read_stage1_frame(
            archives,
            replace(mapping, gas=StoredField("products", "phase_labels")),
            frame_id="t1",
            adapter=adapter,
        )
    with pytest.raises(ValueError):
        replace(mapping, pore_volume_unit="ml")
    with pytest.raises(ValueError):
        replace(mapping, labels=StoredField("geometry", "facies", true_values=(2,)))
    with pytest.raises(TypeError):  # every role is explicit, None included
        StoredFrameMapping(
            colour=mapping.colour,
            labels=mapping.labels,
            pore_volume=mapping.pore_volume,
            pore_volume_unit="L",
        )


def test_the_module_imports_without_the_execution_framework():
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys\n"
        "import darsia.fluidflower.stage2_postprocessing\n"
        "heavy = [m for m in ('darsia.fluidflower.execution', "
        "'darsia.fluidflower.calibration_review', 'darsia.fluidflower.mass_materialization') "
        "if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root / "src")),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
