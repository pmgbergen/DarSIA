from copy import deepcopy
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from darsia.fluidflower.round5_pressure import (
    bind_running_atmosphere, build_atmosphere_series, height_factor,
    merge_station_exports, station_rows, utc,
)

UTC = timezone.utc
ORIGIN = datetime(2023, 1, 5, 10, 0, tzinfo=UTC)


def table(missing=()):
    return {ORIGIN + timedelta(minutes=10 * k): {
        "station_pressure_hpa": None if k in missing else 1000.0 + k,
        "observations": [{"source": {"sha256": "source"}, "sheet_row": k + 20}],
    } for k in range(-6, 13)}


def build(station=None, **kwargs):
    return build_atmosphere_series(
        table() if station is None else station, series="FF_AC", run="ac23",
        origin_utc=ORIGIN, first_utc=ORIGIN - timedelta(minutes=13),
        last_utc=ORIGIN + timedelta(minutes=83), source_manifest={"sha256": "manifest"},
        **kwargs,
    )


def test_camera_offset_and_negative_baseline_have_one_utc_origin():
    assert utc("2023-01-05T12:00:00+02:00") == ORIGIN
    result = build()
    assert result["complete"] is True
    assert result["rows"][0]["time_hours"] == -0.5
    assert result["rows"][-1]["time_hours"] == 1.5
    row = next(row for row in result["rows"] if row["time_hours"] == 0)
    assert row["station_pressure_hpa"] == 1000
    assert row["pressure_bar"] == height_factor("FF_AC")
    assert row["temperature_c"] == 23
    assert result["height_term"]["rig_height_m"] == 14.9


@pytest.mark.parametrize("value", ["2023-01-05T12:00:00", datetime(2023, 1, 5)])
def test_naive_origin_is_not_host_local_time(value):
    with pytest.raises(ValueError, match="explicit timestamp offset"):
        utc(value)


def test_spring_first_duplicate_block_is_utc_two_without_dst_conversion():
    rows = [["Dato", "Tid", "Lufttrykk"]]
    for hour in [1, 3, 3, 4]:
        for minute in range(0, 60, 10):
            rows.append(["2023-03-26", f"{hour:02d}:{minute:02d}", 1000 + len(rows)])
    result, summary = station_rows(rows)
    assert len(result) == 24
    assert summary["spring_rows_relabelled"] == list(range(8, 14))
    assert result[utc("2023-03-26T02:00:00Z")]["station_pressure_hpa"] == 1007
    assert result[utc("2023-03-26T03:00:00Z")]["station_pressure_hpa"] == 1013


def test_other_duplicate_timestamp_is_refused():
    rows = [["Dato", "Tid", "Lufttrykk"], ["2023-01-05", "10:00", 1000]]
    with pytest.raises(ValueError, match="Unresolved duplicate"):
        station_rows(rows + rows[1:])


def test_extra_hand_entered_column_does_not_replace_station_sentinel():
    rows = [["Dato", "Tid", "Lufttrykk", None], ["2023-05-12", "20:20", 9999.99, 1020.7], [None] * 4]
    result, summary = station_rows(rows)
    assert list(result.values())[0]["station_pressure_hpa"] is None
    assert summary["missing_values"] == 1 and summary["blank_rows"] == 1


def test_yearly_overlap_requires_exact_agreement_and_keeps_both_sources():
    rows = [["Dato", "Tid", "Lufttrykk"], ["2023-01-01", "00:00", 1018]]
    merged, _ = merge_station_exports([({"sha256": "a"}, rows), ({"sha256": "b"}, rows)])
    assert len(merged) == 1 and len(next(iter(merged.values()))["observations"]) == 2
    other = deepcopy(rows); other[1][2] = 1018.1
    with pytest.raises(ValueError, match="Conflicting yearly overlap"):
        merge_station_exports([({}, rows), ({}, other)])


def test_two_missing_values_use_only_original_neighbours_and_stay_provisional():
    result = build(table(missing=(1, 2)), provisional_ac_short_gaps=True)
    assert result["complete"] is True and result["provisional_gap_default"] is True
    replacement = [row for row in result["rows"] if row["quality"] != "observed"]
    assert len(replacement) == 2
    assert replacement[0]["station_pressure_hpa"] == (998 + 999 + 1000 + 1003) / 4
    assert replacement[1]["station_pressure_hpa"] == (999 + 1000 + 1003 + 1004) / 4
    assert all(len(row["observations"]) == 4 for row in replacement)


def test_three_missing_values_are_not_bridged_by_a_wider_interpolation():
    result = build(table(missing=(1, 2, 3)), provisional_ac_short_gaps=True)
    assert result["complete"] is False
    assert result["gaps"][0]["missing_stamp_count"] == 3
    assert result["gaps"][0]["filled_count"] == 0


def test_without_gap_authority_missing_values_remain_incomplete():
    assert build(table(missing=(1,)))["complete"] is False
    with pytest.raises(ValueError, match="only for FF_AC"):
        build_atmosphere_series(table(), series="FF_BM", run="c3", origin_utc=ORIGIN,
                                first_utc=ORIGIN, last_utc=ORIGIN + timedelta(hours=1),
                                source_manifest={}, provisional_ac_short_gaps=True)


def test_missing_2026_export_cannot_look_complete():
    result = build_atmosphere_series(table(), series="FF_RE", run="e0_c1",
                                    origin_utc="2026-06-01T08:00:00Z",
                                    first_utc="2026-06-01T07:50:00Z", last_utc="2026-06-01T09:00:00Z",
                                    source_manifest={})
    assert result["complete"] is False and result["spans_all_captures"] is False
    assert result["rows"] == []


def test_offgrid_observation_keeps_its_actual_timestamp_and_value():
    station = table()
    station[ORIGIN + timedelta(minutes=29)] = station.pop(ORIGIN + timedelta(minutes=30))
    result = build(station)
    assert result["complete"] is True
    assert any(row["utc"] == "2023-01-05T10:29:00+00:00" for row in result["rows"])
    assert not any(row["utc"] == "2023-01-05T10:30:00+00:00" for row in result["rows"])


def test_data_binding_removes_all_constant_overrides_and_existing_consumer_interpolates():
    from darsia.fluidflower.mass.n1_pipeline import _atmosphere

    config = {"mass": {"template": {"untouched": 42, "atmosphere_series": {"old": True}}},
              "frames": [{"case_id": "baseline", "elapsed_h": -0.2, "mass_spec": {"atmosphere": {"pressure_bar": 1.01325}}},
                         {"case_id": "later", "elapsed_h": 0.25, "mass_spec": {"atmosphere": {"pressure_bar": 1.01325}}}]}
    before = deepcopy(config)
    result = bind_running_atmosphere(config, build())
    assert config == before and result["mass"]["template"]["untouched"] == 42
    assert all("atmosphere" not in row["mass_spec"] for row in result["frames"])
    for frame in result["frames"]:
        spec = {**frame["mass_spec"], "elapsed_h": frame["elapsed_h"]}
        value = _atmosphere(result["mass"]["template"], spec)
        assert value["pressure_bar"] == pytest.approx((1000 + 6 * spec["elapsed_h"]) / 1000 * height_factor("FF_AC"))
        assert value["temperature_c"] == 23
        assert value["pressure_gradient_bar_per_m"] == 0


def test_incomplete_or_uncovered_atmosphere_cannot_enter_a_plan():
    config = {"mass": {"template": {}}, "frames": [{"case_id": "late", "elapsed_h": 3, "mass_spec": {}}]}
    with pytest.raises(ValueError, match="incomplete"):
        bind_running_atmosphere(config, build(table(missing=(1,))))
    with pytest.raises(ValueError, match="does not span"):
        bind_running_atmosphere(config, build())


def test_inconsistent_repeated_frame_time_is_refused():
    config = {"mass": {"template": {}}, "frames": [{"case_id": "a", "elapsed_h": 0.25,
              "mass_spec": {"elapsed_h": 0.5}}]}
    with pytest.raises(ValueError, match="inconsistent frame and mass times"):
        bind_running_atmosphere(config, build())


def test_height_correction_matches_independent_hydrostatic_calculation():
    assert height_factor("FF_AC") == pytest.approx(np.exp(9.80665 * 30.1 / (287.05 * 296.15)), rel=1e-15)
    assert 1 < height_factor("FF_BM") < height_factor("FF_AC")
