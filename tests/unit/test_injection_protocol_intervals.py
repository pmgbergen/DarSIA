"""Repeated injections at one point retain their pauses on either time axis."""
from datetime import datetime, timedelta, timezone

import pytest

from darsia.experiment.protocols import InjectionProtocol


@pytest.fixture
def repeated_injections(tmp_path):
    path = tmp_path / "injection.csv"
    # Deliberately unsorted: elapsed time starts at the earliest injection,
    # rather than at the first row or separately at each pulse.
    path.write_text(
        "id,location_x,location_y,start,end,rate_kg/s\n"
        "1,0.4,0.1,2025-01-02T11:00:00+01:00,2025-01-02T12:00:00+01:00,0.002\n"
        "1,0.4,0.1,2025-01-02T09:00:00+01:00,2025-01-02T10:00:00+01:00,0.001\n",
        encoding="utf-8",
    )
    return InjectionProtocol(path)


@pytest.mark.parametrize(
    "hours,mass_kg",
    [(-1, 0), (0, 0), (0.5, 1.8), (1, 3.6), (1.5, 3.6),
     (2, 3.6), (2.5, 7.2), (3, 10.8), (4, 10.8)],
)
def test_absolute_and_elapsed_time_agree_across_repeated_injection_pause(
    repeated_injections, hours, mass_kg
):
    origin = datetime(2025, 1, 2, 8, tzinfo=timezone.utc)
    assert repeated_injections.injected_mass(
        date=origin + timedelta(hours=hours)
    ) == pytest.approx(mass_kg)
    assert repeated_injections.injected_mass(time=hours) == pytest.approx(mass_kg)
