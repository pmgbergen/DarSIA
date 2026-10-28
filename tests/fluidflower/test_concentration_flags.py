"""Saturated and not-quantified flags."""
import numpy as np
import pytest

from darsia.fluidflower.concentration_flags import (
    dissolved_interval,
    flag_fractions,
    not_quantified_mask,
    saturated_mask,
)
from darsia.fluidflower.titration import TitrationModel


def test_saturated_mask_threshold_and_nan():
    f = np.array([0.9, 0.05, 0.049, 0.0, np.nan])
    assert saturated_mask(f).tolist() == [False, False, True, True, False]
    assert saturated_mask(f, f_min=0.1).tolist() == [False, True, True, True, False]


def test_dissolved_interval_bounds():
    model = TitrationModel(temperature_c=25.0)
    f = np.array([[0.9, 0.5], [0.01, np.nan]])
    iv = dissolved_interval(f, model)
    assert iv.saturated.tolist() == [[False, False], [True, False]]
    # unsaturated pixels: both bounds equal the closed-form value
    assert iv.lower[0, 0] == pytest.approx(float(model.dic_from_blue_fraction(0.9)))
    assert iv.upper[0, 0] == iv.lower[0, 0]
    # saturated pixel: lower bound at the threshold, upper bound at solubility
    assert iv.lower[1, 0] == pytest.approx(float(model.dic_from_blue_fraction(0.05)))
    assert iv.upper[1, 0] == pytest.approx(model.saturation_dic())
    assert iv.upper[1, 0] > 6 * iv.lower[1, 0]
    # not quantified stays NaN
    assert np.isnan(iv.lower[1, 1]) and np.isnan(iv.upper[1, 1])


def test_not_quantified_mask():
    labels = np.array([[1, 2], [3, 4]])  # 1 free water, 2 fine sand, 3 coarse sand, 4 excluded
    mask = not_quantified_mask(labels, quantified_labels=[2, 3])
    assert mask.tolist() == [[True, False], [False, True]]
    gas = np.array([[False, True], [False, False]])
    assert not_quantified_mask(labels, [2, 3], gas_mask=gas).tolist() == [[True, True], [False, True]]
    outside = np.array([[False, False], [True, False]])
    assert not_quantified_mask(labels, [2, 3], outside_mask=outside).tolist() == [[True, False], [True, True]]


def test_flag_fractions():
    sand = np.array([[True, True], [True, False]])
    sat = np.array([[True, False], [False, True]])
    nq = np.array([[False, True], [False, False]])
    out = flag_fractions(sat, nq, sand)
    assert out["sand_pixels"] == 3
    assert out["saturated_fraction"] == pytest.approx(1 / 3)
    assert out["not_quantified_fraction"] == pytest.approx(1 / 3)
    empty = flag_fractions(sat, nq, np.zeros_like(sand))
    assert empty["sand_pixels"] == 0 and np.isnan(empty["saturated_fraction"])


def test_interval_rejects_invalid_colour_and_bad_threshold():
    model = TitrationModel()
    f = np.array([0.5, -0.1, 1.2, 0.02])
    iv = dissolved_interval(f, model, valid_mask=np.array([True, True, True, False]))
    assert np.isnan(iv.lower[1]) and np.isnan(iv.lower[2]) and np.isnan(iv.upper[3])
    assert not iv.saturated[3]
    assert np.all(iv.upper[np.isfinite(iv.upper)] >= iv.lower[np.isfinite(iv.lower)])
    with pytest.raises(ValueError):
        dissolved_interval(f, model, f_min=0.0)
    with pytest.raises(ValueError):
        dissolved_interval(f, model, f_min=0.05, p_co2_atm=1e-4)  # solubility below the threshold
    labels = np.array([2, 2, 3])
    m = not_quantified_mask(labels, [2, 3], invalid_colour_mask=np.array([False, True, False]))
    assert m.tolist() == [False, True, False]


def test_interval_requires_finite_positive_pressure():
    model = TitrationModel()
    f = np.array([0.5, 0.02])
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            dissolved_interval(f, model, p_co2_atm=bad)
    iv = dissolved_interval(f, model, p_co2_atm=1.0)
    assert np.all(np.isfinite(iv.upper)) and iv.upper[1] > iv.lower[1]
