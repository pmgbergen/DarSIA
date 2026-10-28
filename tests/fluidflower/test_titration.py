"""Titration model: recipe, reference values and round trips."""
import numpy as np
import pytest

from darsia.fluidflower.titration import FFRE_RECIPE, IndicatorRecipe, TitrationModel


def test_ffre_recipe_concentrations():
    assert FFRE_RECIPE.sodium_m == pytest.approx(1.250e-3, rel=1e-3)
    assert FFRE_RECIPE.btb_total_m == pytest.approx(0.699e-3, rel=1e-2)
    assert FFRE_RECIPE.alkalinity_m == pytest.approx(1.250e-3, rel=1e-3)
    acid = IndicatorRecipe(1.0 / 20.0, 9.04 / 20.0, form="free_acid")
    assert acid.alkalinity_m == pytest.approx(0.526e-3, rel=1e-2)
    with pytest.raises(ValueError):
        IndicatorRecipe(1.0, 1.0, form="unknown")


def test_constants_at_25_c():
    c = TitrationModel(temperature_c=25.0).constants
    assert -np.log10(c.k1) == pytest.approx(6.35, abs=0.01)
    assert -np.log10(c.k2) == pytest.approx(10.33, abs=0.01)
    assert -np.log10(c.kw) == pytest.approx(14.00, abs=0.01)
    assert c.henry_mol_per_l_atm == pytest.approx(0.034, rel=1e-3)


def test_fresh_and_air_equilibrated_solution():
    model = TitrationModel()
    assert model.fresh_ph() == pytest.approx(10.74, abs=0.02)
    ph, dic, f = model.air_equilibrium()
    assert ph == pytest.approx(7.99, abs=0.03)
    assert dic == pytest.approx(0.64e-3, rel=0.03)
    assert f == pytest.approx(0.886, abs=0.01)
    # the free acid would already be green without CO2
    assert TitrationModel(IndicatorRecipe(1.0 / 20.0, 9.04 / 20.0, "free_acid")).fresh_ph() == pytest.approx(7.52, abs=0.03)


@pytest.mark.parametrize(
    "f, ph, dic_mm",
    [(0.95, 8.38, 0.58), (0.90, 8.05, 0.63), (0.70, 7.47, 0.82), (0.50, 7.10, 1.06),
     (0.30, 6.73, 1.47), (0.10, 6.15, 3.08), (0.05, 5.82, 5.3), (0.02, 5.41, 12.1), (0.01, 5.10, 23.4)],
)
def test_reference_values_at_25_c(f, ph, dic_mm):
    model = TitrationModel(temperature_c=25.0)
    dic = float(model.dic_from_blue_fraction(f))
    assert dic * 1e3 == pytest.approx(dic_mm, rel=0.01)
    assert float(model.ph_from_dic(dic)) == pytest.approx(ph, abs=0.01)
    assert float(model.blue_fraction_from_dic(dic)) == pytest.approx(f, abs=1e-3)


def test_saturation_dic():
    assert TitrationModel(temperature_c=25.0).saturation_dic() * 1e3 == pytest.approx(35.2, rel=0.01)
    assert TitrationModel(temperature_c=20.0).saturation_dic() * 1e3 == pytest.approx(40.3, rel=0.01)


def test_round_trip_is_vectorised_and_monotone():
    model = TitrationModel()
    f = np.array([[0.99, 0.9, 0.5], [0.2, 0.05, 0.01]])
    dic = model.dic_from_blue_fraction(f)
    assert dic.shape == f.shape
    assert np.all(np.diff(dic.ravel()) > 0)
    back = model.blue_fraction_from_dic(dic)
    assert np.allclose(back, f, atol=1e-4)


def test_mass_per_pixel():
    # 1 mM in a 1 mm x 1 mm x 1 cm pixel with porosity 0.45: 1 mol/m3 * 4.5e-9 m3 * 44.01 g/mol
    g = TitrationModel.mass_per_pixel_g(1e-3, 0.45, 0.01, 1e-6)
    assert float(g) == pytest.approx(1.0 * 4.5e-9 * 44.01, rel=1e-9)


def test_invalid_inputs_give_nan_and_policies_are_parameters():
    model = TitrationModel()
    ph = model.ph_from_dic(np.array([np.nan, -1e-3, 1e-3]))
    assert np.isnan(ph[0]) and np.isnan(ph[1]) and np.isfinite(ph[2])
    dic = model.dic_from_blue_fraction(np.array([np.nan, -0.1, 1.2, 1.0, 0.0]))
    assert np.isnan(dic[0]) and np.isnan(dic[1]) and np.isnan(dic[2])
    assert dic[3] == 0.0  # fully blue: clipped to zero by default
    assert np.isfinite(dic[4]) and dic[4] > 0.03  # fully yellow: finite through the floor
    raw = TitrationModel(clip_negative_dic=False)
    assert float(raw.dic_from_blue_fraction(1.0)) < 0.0
    with pytest.raises(ValueError):
        TitrationModel(blue_fraction_floor=0.6)


def test_mass_per_pixel_accepts_arrays():
    area = np.array([[1e-6, 2e-6]])
    g = TitrationModel.mass_per_pixel_g(np.array([[1e-3, 1e-3]]), 0.45, 0.01, area)
    assert g.shape == (1, 2) and g[0, 1] == pytest.approx(2 * g[0, 0])
