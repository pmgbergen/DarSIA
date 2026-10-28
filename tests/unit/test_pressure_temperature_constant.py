"""Only mathematically constant P/T protocols are independent of timezone axes."""

from datetime import datetime

import pandas as pd
import pytest

from darsia.experiment.protocols import PressureTemperatureProtocol


def protocol(tmp_path, *, varying=False, gradient_varying=False, aware=False, invalid=False):
    path = tmp_path / "pressure.csv"
    suffix = "+02:00" if aware else ""
    pd.DataFrame(
        {
            "datetime": [
                "2022-10-05T10:00:00" + suffix,
                "2022-10-05T12:00:00" + suffix,
            ],
            "pressure_bar": [
                0.998798,
                float("nan") if invalid else (1.1 if varying else 0.998798),
            ],
            "temperature_celsius": [23.0, 25.0 if varying else 23.0],
            "pressure_gradient_bar": [0.0, 0.2 if gradient_varying else 0.0],
            "temperature_gradient_celsius": [0.0, 0.4 if gradient_varying else 0.0],
        }
    ).to_csv(path, index=False)
    return path, PressureTemperatureProtocol(path)


def test_constant_naive_protocol_keeps_values_for_aware_query_without_rewriting(tmp_path):
    path, value = protocol(tmp_path)
    before = path.read_bytes()
    for date in (
        pd.Timestamp("2022-10-05T11:00:00+02:00"),
        pd.Timestamp("2030-01-01T00:00:00+00:00"),
        datetime(2022, 10, 5, 11),
    ):
        state = value.get_state(date)
        gradient = value.get_gradient(date)
        assert (state.pressure, state.temperature) == (0.998798, 23.0)
        assert (gradient.pressure, gradient.temperature) == (0.0, 0.0)
    # Callers receive independent state objects as before.
    state.pressure = 5.0
    assert value.get_state(date).pressure == 0.998798
    assert path.read_bytes() == before


@pytest.mark.parametrize("aware", [False, True])
def test_variable_protocol_keeps_ordinary_interpolation(tmp_path, aware):
    _, value = protocol(tmp_path, varying=True, gradient_varying=True, aware=aware)
    date = pd.Timestamp("2022-10-05T11:00:00" + ("+02:00" if aware else ""))
    state, gradient = value.get_state(date), value.get_gradient(date)
    assert state.pressure == pytest.approx((0.998798 + 1.1) / 2)
    assert state.temperature == 24.0
    assert (gradient.pressure, gradient.temperature) == pytest.approx((0.1, 0.2))


@pytest.mark.parametrize(
    "varying,gradient_varying,invalid",
    [(True, False, False), (False, True, False), (False, False, True)],
)
def test_nonconstant_or_invalid_protocol_cannot_bypass_ambiguous_timezone(
    tmp_path, varying, gradient_varying, invalid
):
    _, value = protocol(
        tmp_path,
        varying=varying,
        gradient_varying=gradient_varying,
        invalid=invalid,
    )
    date = pd.Timestamp("2022-10-05T11:00:00+02:00")
    with pytest.raises(TypeError):
        value.get_state(date)
    with pytest.raises(TypeError):
        value.get_gradient(date)
