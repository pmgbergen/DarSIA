"""Build staged pressure DATA from the bound GFI station observations.

This module is not imported by a worker. Station timestamps are UTC, including
the documented first duplicated 03:xx block on the spring clock-change day.
Original measurements and provisional gap values retain separate provenance.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
import math

UTC = timezone.utc
STEP = timedelta(minutes=10)
HEIGHTS_M = {"FF_AC": 14.9, "FF_RE": 14.9, "FF_BM": 19.92}
TEMPERATURES_C = {"FF_AC": 23.0, "FF_RE": 25.0, "FF_BM": 22.0}


def utc(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("An explicit timestamp offset is required")
    return value.astimezone(UTC)


def station_rows(rows):
    """Parse first-three-column GFI rows; retain missing values and row IDs.

    The header is checked. Additional spreadsheet columns cannot supply a
    missing station observation. The one documented spring export defect is
    corrected only when the complete, exact duplicate-block pattern exists.
    """
    iterator = iter(rows)
    header = next(iterator)
    if tuple(header[:3]) != ("Dato", "Tid", "Lufttrykk"):
        raise ValueError("Unexpected GFI column names")
    parsed = []
    blank = 0
    for row_number, row in enumerate(iterator, 2):
        values = tuple(row[:3])
        if not any(value is not None and str(value).strip() for value in values):
            blank += 1
            continue
        if len(values) < 3 or values[0] is None or values[1] is None:
            raise ValueError(f"Incomplete station timestamp at row {row_number}")
        day = date.fromisoformat(str(values[0]))
        clock = time.fromisoformat(str(values[1]))
        if clock.tzinfo is not None:
            raise ValueError("GFI sheet clocks must be unqualified UTC labels")
        stamp = datetime.combine(day, clock, UTC)
        pressure = None if values[2] is None else float(values[2])
        if pressure == 9999.99:
            pressure = None
        if pressure is not None and (not math.isfinite(pressure) or not 870 <= pressure <= 1100):
            raise ValueError(f"Invalid station pressure at row {row_number}")
        parsed.append({"utc": stamp, "station_pressure_hpa": pressure, "sheet_row": row_number})
    by_day = {}
    for record in parsed:
        by_day.setdefault(record["utc"].date(), []).append(record)
    corrected = []
    for day, records in by_day.items():
        if day.month != 3 or day.weekday() != 6 or (day + timedelta(days=7)).month == 3:
            continue
        block = [record for record in records if record["utc"].hour == 3]
        if len(block) == 12 and not any(record["utc"].hour == 2 for record in records):
            minutes = [record["utc"].minute for record in block]
            if minutes != list(range(0, 60, 10)) * 2:
                raise ValueError("Unexpected spring duplicate block")
            for record in block[:6]:
                record["utc"] -= timedelta(hours=1)
                corrected.append(record["sheet_row"])
    result = {}
    for record in parsed:
        stamp = record["utc"]
        if stamp in result:
            raise ValueError(f"Unresolved duplicate station stamp {stamp}")
        result[stamp] = record
    return result, {"data_rows": len(parsed), "blank_rows": blank,
                    "spring_rows_relabelled": corrected,
                    "missing_values": sum(row["station_pressure_hpa"] is None for row in parsed)}


def merge_station_exports(exports):
    """Merge yearly overlap only when observations agree exactly."""
    merged = {}
    summaries = []
    for source, rows in exports:
        samples, summary = station_rows(rows)
        summaries.append({"source": source, **summary})
        for stamp, row in samples.items():
            provenance = {"source": source, "sheet_row": row["sheet_row"]}
            if stamp in merged:
                if merged[stamp]["station_pressure_hpa"] != row["station_pressure_hpa"]:
                    raise ValueError(f"Conflicting yearly overlap at {stamp}")
                merged[stamp]["observations"].append(provenance)
            else:
                merged[stamp] = {"station_pressure_hpa": row["station_pressure_hpa"],
                                 "observations": [provenance]}
    return merged, summaries


def height_factor(series):
    """N198 height term, without anchoring to another station."""
    return math.exp(9.80665 * (45.0 - HEIGHTS_M[series]) / (287.05 * 296.15))


def build_atmosphere_series(station, *, series, run, origin_utc, first_utc,
                            last_utc, source_manifest, clock_notes=None,
                            provisional_ac_short_gaps=False):
    """Keep running observations and flag N274's reversible short-gap mean.

    Only FF_AC gaps of one or two ten-minute stamps may use the mean of at
    least four original valid observations in [t-30min,t+30min). No filled
    value feeds another mean. Other gaps leave the result incomplete.
    """
    origin, first, last = map(utc, (origin_utc, first_utc, last_utc))
    if first > last:
        raise ValueError("The imaging interval is reversed")
    if provisional_ac_short_gaps and series != "FF_AC":
        raise ValueError("The provisional short-gap rule is authorised only for FF_AC")
    factor = height_factor(series)
    start = first.replace(minute=first.minute // 10 * 10, second=0, microsecond=0) - STEP
    end = last.replace(minute=last.minute // 10 * 10, second=0, microsecond=0) + STEP
    valid = {stamp: row for stamp, row in station.items()
             if row["station_pressure_hpa"] is not None}
    selected = {stamp: {**row, "quality": "observed"}
                for stamp, row in valid.items() if start <= stamp <= end}
    missing = []
    stamp = start
    while stamp <= end:
        # Two original exports contain a one-minute off-grid timestamp. Keep
        # that observation at its original time, rather than resampling it.
        if not any(stamp + timedelta(seconds=offset) in valid for offset in (-60, 0, 60)):
            missing.append(stamp)
        stamp += STEP
    groups = []
    for stamp in missing:
        if not groups or stamp - groups[-1][-1] != STEP:
            groups.append([])
        groups[-1].append(stamp)
    gaps = []
    for group in groups:
        replacements = {}
        if provisional_ac_short_gaps and len(group) <= 2:
            for stamp in group:
                neighbours = [(moment, row) for moment, row in valid.items()
                              if stamp - timedelta(minutes=30) <= moment < stamp + timedelta(minutes=30)]
                if len(neighbours) >= 4 and any(t < stamp for t, _ in neighbours) and any(t > stamp for t, _ in neighbours):
                    replacements[stamp] = {
                        "station_pressure_hpa": math.fsum(row["station_pressure_hpa"] for _, row in neighbours) / len(neighbours),
                        "quality": "provisional_short_gap_mean_pm30min",
                        "observations": [{"utc": t.isoformat(), **row} for t, row in sorted(neighbours)],
                    }
        selected.update(replacements)
        gaps.append({"first_utc": group[0].isoformat(), "last_utc": group[-1].isoformat(),
                     "missing_stamp_count": len(group), "filled_count": len(replacements),
                     "resolved_provisionally": len(replacements) == len(group)})
    rows = []
    for stamp, row in sorted(selected.items()):
        rows.append({"utc": stamp.isoformat(), "time_hours": (stamp - origin).total_seconds() / 3600,
                     "pressure_bar": row["station_pressure_hpa"] * factor / 1000,
                     "temperature_c": TEMPERATURES_C[series],
                     "pressure_gradient_bar_per_m": 0.0, "temperature_gradient_c_per_m": 0.0,
                     **row})
    span = bool(rows) and utc(rows[0]["utc"]) <= first and utc(rows[-1]["utc"]) >= last
    complete = span and all(gap["resolved_provisionally"] for gap in gaps)
    return {"format": "darsia-running-gfi-pressure-v1", "series": series, "run": run,
            "origin_datetime": origin.isoformat(), "time_basis": "UTC",
            "imaging_first_utc": first.isoformat(), "imaging_last_utc": last.isoformat(),
            "rows": rows, "gaps": gaps, "spans_all_captures": span, "complete": complete,
            "provisional_gap_default": any(gap["filled_count"] for gap in gaps),
            "gap_default_authority": "N274; project-owner decision pending" if provisional_ac_short_gaps else None,
            "source_manifest": deepcopy(source_manifest), "clock_notes": deepcopy(clock_notes),
            "height_term": {"station_height_m": 45.0, "rig_height_m": HEIGHTS_M[series],
                            "g_m_s2": 9.80665, "R_d": 287.05, "T0_K": 296.15, "factor": factor},
            "extrapolation_allowed": False, "adopted": False}


def bind_running_atmosphere(configuration, atmosphere):
    """Stage R5G9 DATA and remove constants which override running rows."""
    if atmosphere.get("complete") is not True:
        raise ValueError("An incomplete atmosphere cannot be bound to a plan")
    rows = atmosphere["rows"]
    times = [float(row["time_hours"]) for row in rows]
    if len(times) < 2 or not all(math.isfinite(t) for t in times) or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("Atmosphere rows must be finite and strictly ordered")
    required = ("pressure_bar", "temperature_c", "pressure_gradient_bar_per_m", "temperature_gradient_c_per_m")
    if any(not math.isfinite(float(row[key])) for row in rows for key in required):
        raise ValueError("Atmosphere values must be finite")
    if any(float(row["pressure_bar"]) <= 0 for row in rows):
        raise ValueError("Atmospheric pressure must be positive")
    result = deepcopy(configuration)
    template = result["mass"]["template"]
    for frame in result["frames"]:
        spec = frame["mass_spec"]
        elapsed = float(frame["elapsed_h"])
        if "elapsed_h" in spec and float(spec["elapsed_h"]) != elapsed:
            raise ValueError(f"{frame['case_id']}: inconsistent frame and mass times")
        if not math.isfinite(elapsed) or not times[0] <= elapsed <= times[-1]:
            raise ValueError(f"{frame['case_id']}: atmosphere does not span the frame")
        spec.pop("atmosphere", None)
    template["atmosphere_series"] = deepcopy(atmosphere)
    return result
