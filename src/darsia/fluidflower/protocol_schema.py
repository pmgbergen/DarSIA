"""Universal tabular protocol schema for FluidFlower experimental series."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

PROTOCOL_WORKBOOK_FORMAT = "darsia-fluidflower-protocol-workbook-v2"
PROTOCOL_EVENT_COLUMNS = (
    "run_id",
    "event_type",
    "experiment_date",
    "timezone",
    "timing_basis",
    "datetime",
    "start_datetime",
    "end_datetime",
    "time_h",
    "start_h",
    "end_h",
    "location_id",
    "segment",
    "value",
    "start_value",
    "end_value",
    "unit",
    "first_image",
    "label",
    "path",
    "material",
    "measurement_location",
    "method",
    "uncertainty",
    "uncertainty_unit",
    "instrument_id",
    "temperature_c",
    "pressure_bar",
    "source",
    "notes",
)
PROTOCOL_RUN_COLUMNS = (
    "run_id",
    "label",
    "apparatus_type",
    "experiment_date",
    "experiment_start_datetime",
    "timezone",
    "operator",
    "camera_id",
    "colorchecker_id",
    "chemistry_batch",
    "indicator_system",
    "indicator_batch",
    "injection_fluid",
    "gas_composition",
    "overflow_mode",
    "indicator_initial_ph",
    "rig_initial_ph",
    "temperature_c",
    "pressure_bar",
    "closed_system",
    "indicator_state",
    "notes",
)
PROTOCOL_INJECTION_POINT_COLUMNS = (
    "run_id",
    "location_id",
    "label",
    "x_m",
    "y_m",
    "coordinate_system",
    "notes",
)
RESERVED_SHEETS = {
    "readme",
    "runs",
    "metadata",
    "event types",
    "injection points",
}


def _template_event_rows(run_id: str, experiment_date: date) -> list[dict[str, Any]]:
    date_text = experiment_date.isoformat()
    following_date = (experiment_date + timedelta(days=1)).isoformat()
    rows = [
        {
            "run_id": run_id,
            "event_type": "experiment_start",
            "experiment_date": date_text,
            "timezone": "Europe/Oslo",
            "timing_basis": "actual",
            "datetime": "09:00:00",
            "time_h": 0.0,
            "label": "Approved experiment time origin",
            "source": "protocol",
        },
        {
            "run_id": run_id,
            "event_type": "ph_measurement",
            "experiment_date": date_text,
            "timezone": "Europe/Oslo",
            "timing_basis": "actual",
            "datetime": "08:30:00",
            "time_h": -0.5,
            "value": 8.1,
            "unit": "pH",
            "material": "indicator solution",
            "measurement_location": "indicator reservoir",
            "method": "calibrated pH meter",
            "instrument_id": "ph_meter_1",
            "temperature_c": 20.0,
            "label": "Indicator-solution pH before filling",
            "source": "measurement",
        },
        {
            "run_id": run_id,
            "event_type": "ph_measurement",
            "experiment_date": date_text,
            "timezone": "Europe/Oslo",
            "timing_basis": "actual",
            "datetime": "08:50:00",
            "time_h": -1.0 / 6.0,
            "value": 8.0,
            "unit": "pH",
            "measurement_location": "rig before injection",
            "method": "calibrated pH meter",
            "instrument_id": "ph_meter_1",
            "temperature_c": 20.0,
            "label": "Rig pH immediately before injection",
            "source": "measurement",
        },
        {
            "run_id": run_id,
            "event_type": "baseline_image",
            "experiment_date": date_text,
            "timezone": "Europe/Oslo",
            "timing_basis": "actual",
            "datetime": "08:55:00",
            "time_h": -1.0 / 12.0,
            "first_image": "DSC00001.JPG",
            "path": "DSC00001.JPG",
            "label": "First image in series; pre-injection reference",
            "source": "images",
            "instrument_id": "camera_1",
        },
        {
            "run_id": run_id,
            "event_type": "camera_interval_start",
            "experiment_date": date_text,
            "timezone": "Europe/Oslo",
            "timing_basis": "actual",
            "datetime": "08:55:00",
            "time_h": -1.0 / 12.0,
            "value": 30.0,
            "unit": "s",
            "first_image": "DSC00001.JPG",
            "label": "Camera interval 1",
            "source": "camera",
            "instrument_id": "camera_1",
        },
        {
            "run_id": run_id,
            "event_type": "injection_ramp_up",
            "experiment_date": date_text,
            "timezone": "Europe/Oslo",
            "timing_basis": "actual",
            "start_datetime": "09:00:00",
            "end_datetime": "09:05:00",
            "start_h": 0.0,
            "end_h": 1.0 / 12.0,
            "location_id": "ip1",
            "segment": "ramp_up",
            "start_value": 0.1,
            "end_value": 2.0,
            "unit": "ml/min",
            "material": "CO2 gas",
            "label": "Injection 1 ramp-up",
            "source": "injection protocol",
            "instrument_id": "pump_1",
        },
        {
            "run_id": run_id,
            "event_type": "injection_steady",
            "experiment_date": date_text,
            "timezone": "Europe/Oslo",
            "timing_basis": "actual",
            "start_datetime": "09:05:00",
            "end_datetime": "09:50:00",
            "start_h": 1.0 / 12.0,
            "end_h": 5.0 / 6.0,
            "location_id": "ip1",
            "segment": "steady",
            "value": 2.0,
            "unit": "ml/min",
            "material": "CO2 gas",
            "label": "Injection 1 steady rate",
            "source": "injection protocol",
            "instrument_id": "pump_1",
        },
        {
            "run_id": run_id,
            "event_type": "injection_ramp_down",
            "experiment_date": date_text,
            "timezone": "Europe/Oslo",
            "timing_basis": "actual",
            "start_datetime": "09:50:00",
            "end_datetime": "09:55:00",
            "start_h": 5.0 / 6.0,
            "end_h": 11.0 / 12.0,
            "location_id": "ip1",
            "segment": "ramp_down",
            "start_value": 2.0,
            "end_value": 0.1,
            "unit": "ml/min",
            "material": "CO2 gas",
            "label": "Injection 1 ramp-down",
            "source": "injection protocol",
            "instrument_id": "pump_1",
        },
        {
            "run_id": run_id,
            "event_type": "camera_interval_start",
            "experiment_date": date_text,
            "timezone": "Europe/Oslo",
            "timing_basis": "actual",
            "datetime": "10:00:00",
            "time_h": 1.0,
            "value": 300.0,
            "unit": "s",
            "first_image": "DSC00121.JPG",
            "label": "Camera interval 2",
            "source": "camera",
            "instrument_id": "camera_1",
        },
        {
            "run_id": run_id,
            "event_type": "injection_steady",
            "experiment_date": date_text,
            "timezone": "Europe/Oslo",
            "timing_basis": "actual",
            "start_datetime": f"{following_date} 09:00:00",
            "end_datetime": f"{following_date} 10:00:00",
            "start_h": 24.0,
            "end_h": 25.0,
            "location_id": "ip2",
            "segment": "steady",
            "value": 5.0,
            "unit": "ml/min",
            "material": "CO2 gas",
            "label": "Injection 2 on the following day",
            "source": "injection protocol",
            "instrument_id": "pump_1",
        },
    ]
    for row in rows[1:]:
        row.pop("experiment_date", None)
        row.pop("timezone", None)
    return rows


EVENT_TYPE_DESCRIPTIONS = (
    ("experiment_start", "Approved t=0 for the run."),
    ("baseline_image", "Pre-injection reference and/or first image in the series."),
    (
        "camera_interval_start",
        "Start or change of image interval; value is the interval.",
    ),
    ("camera_pause", "Camera acquisition was paused or interrupted."),
    ("camera_resume", "Camera acquisition resumed; include the first new image."),
    (
        "camera_settings_change",
        "Exposure, white balance, lighting, or other image change.",
    ),
    ("injection_ramp_up", "Injection-rate ramp; use start_value and end_value."),
    ("injection_steady", "Constant-rate injection; use value."),
    ("injection_ramp_down", "Injection-rate ramp; use start_value and end_value."),
    ("ph_measurement", "Measured pH with material/location and method."),
    ("overflow_start", "Start of overflow addition or collection."),
    ("overflow_stop", "End of overflow addition or collection."),
    ("overflow_measurement", "Mass, volume, or rate measured for overflow."),
    ("flush_start", "Start of flushing or indicator replacement."),
    ("flush_stop", "End of flushing or indicator replacement."),
    ("temperature_measurement", "Measured fluid, rig, or room temperature."),
    ("pressure_measurement", "Measured pressure relevant to fluid properties."),
    ("water_level_measurement", "Measured free-water level or hydraulic head."),
    ("sample", "Physical sample collected from a stated location."),
    ("intervention", "Manual change, pause, sampling, camera restart, or anomaly."),
)


def write_protocol_workbook_template(
    path: Path,
    *,
    layout: str = "combined",
    run_ids: Sequence[str] = ("run01", "run02"),
) -> Path:
    """Write an editable XLSX example without treating any series as canonical."""

    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    if layout not in {"combined", "per_run_sheets", "single_run"}:
        raise ValueError("Protocol template layout is invalid")
    cleaned_runs = [
        str(value).strip().casefold() for value in run_ids if str(value).strip()
    ]
    if layout == "single_run":
        cleaned_runs = cleaned_runs[:1] or ["run01"]
    if not cleaned_runs:
        cleaned_runs = ["run01"]
    run_dates = {
        run_id: date(2026, 1, 15) + timedelta(days=index)
        for index, run_id in enumerate(cleaned_runs)
    }
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    readme = workbook.active
    readme.title = "README"
    readme_rows = [
        ("DarSIA universal FluidFlower timeline workbook",),
        ("Format", PROTOCOL_WORKBOOK_FORMAT),
        ("Purpose", "Align protocol, images, injection, overflow, and interventions."),
        ("Required", "Enter experiment_date for every run."),
        (
            "Optional",
            (
                "Set timezone once per run when needed for portable timestamps; "
                "otherwise DarSIA treats all timestamps as consistent local time."
            ),
        ),
        ("Required", "Add one experiment_start event defining t=0 for every run."),
        (
            "Time",
            (
                "Use ISO date/time (YYYY-MM-DD HH:MM:SS). A time without a date "
                "uses the run's experiment_date."
            ),
        ),
        (
            "Actual time",
            (
                "Enter observed times whenever possible and set "
                "timing_basis=actual. Use planned only when no observation exists."
            ),
        ),
        (
            "Multiple days",
            (
                "Include the date whenever an event occurs on a later day; "
                "relative time is then calculated across dates."
            ),
        ),
        (
            "Camera",
            "Record every camera_interval_start with interval, unit, and first_image.",
        ),
        (
            "Injection",
            (
                "Use location_id (ip1, ip2, ...) for the injection point. Split "
                "ramp-up, steady, and "
                "ramp-down into separate rows when relevant."
            ),
        ),
        (
            "Injection-point position",
            (
                "x_m and y_m are optional in this workbook. Leave them blank to "
                "place the injection point interactively on a baseline or early CO2 image "
                "in Geometry and facies. Keep location_id consistent between the "
                "Injection points and Events sheets."
            ),
        ),
        (
            "Chemistry",
            (
                "Record measured pH as events and state material/location, method, "
                "and temperature when available. pH events are authoritative; "
                "run-level pH is only a summary."
            ),
        ),
        (
            "Traceability",
            (
                "Record instruments, uncertainty, camera/lighting changes, "
                "overflow, flushing, samples, pauses, and manual interventions "
                "when relevant."
            ),
        ),
        ("Rule", "Use run_id in a combined events sheet or one sheet per run."),
        (
            "Rule",
            "Original files remain read-only; GUI adjustments are stored separately.",
        ),
        (
            "Required",
            (
                "run_id, event_type, and either "
                "datetime/start_datetime/end_datetime or the corresponding "
                "relative-hour fields."
            ),
        ),
    ]
    for row in readme_rows:
        readme.append(row)
    readme["A1"].font = Font(bold=True, size=14)
    readme.column_dimensions["A"].width = 18
    readme.column_dimensions["B"].width = 90

    runs = workbook.create_sheet("Runs")
    runs.append(PROTOCOL_RUN_COLUMNS)
    for run_id in cleaned_runs:
        values = {
            "run_id": run_id,
            "label": run_id.upper(),
            "apparatus_type": "",
            "experiment_date": run_dates[run_id].isoformat(),
            "experiment_start_datetime": "09:00:00",
            "timezone": "Europe/Oslo",
            "operator": "",
            "camera_id": "camera_1",
            "colorchecker_id": "",
            "chemistry_batch": "",
            "indicator_system": "",
            "indicator_batch": "",
            "injection_fluid": "CO2 gas",
            "gas_composition": "",
            "overflow_mode": "",
            "indicator_initial_ph": 8.1,
            "rig_initial_ph": 8.0,
            "temperature_c": 20.0,
            "pressure_bar": 1.0,
            "closed_system": True,
            "indicator_state": "indicator_filled",
            "notes": "Replace example values with measurements.",
        }
        runs.append(tuple(values.get(column, "") for column in PROTOCOL_RUN_COLUMNS))

    injection_points = workbook.create_sheet("Injection points")
    injection_points.append(PROTOCOL_INJECTION_POINT_COLUMNS)
    for run_id in cleaned_runs:
        for point_number in (1, 2):
            values = {
                "run_id": run_id,
                "location_id": f"ip{point_number}",
                "label": f"Injection Point {point_number}",
                "coordinate_system": "origin lower-left; x right; y up",
                "notes": (
                    "x_m and y_m are optional. Leave blank to place this injection point "
                    "on a rig or early CO2 image in the Geometry and facies GUI."
                ),
            }
            injection_points.append(
                tuple(
                    values.get(column, "")
                    for column in PROTOCOL_INJECTION_POINT_COLUMNS
                )
            )

    from openpyxl.comments import Comment

    injection_headers = {
        str(cell.value): cell for cell in injection_points[1] if cell.value is not None
    }
    for field, guidance in {
        "location_id": (
            "Required for every injection point used by an injection event. Use the same "
            "identifier in the Events sheet and in the DarSIA GUI."
        ),
        "x_m": (
            "Optional horizontal coordinate in metres from the left side of the "
            "registered rig. Leave blank to place the injection point in the GUI."
        ),
        "y_m": (
            "Optional vertical coordinate in metres above the bottom of the "
            "registered rig. Leave blank to place the injection point in the GUI."
        ),
        "coordinate_system": (
            "Coordinates use the registered 2D camera plane. No z-coordinate is "
            "required for injection-point placement."
        ),
    }.items():
        injection_headers[field].comment = Comment(guidance, "DarSIA")

    event_types = workbook.create_sheet("Event types")
    event_types.append(("event_type", "purpose"))
    for event_type, purpose in EVENT_TYPE_DESCRIPTIONS:
        event_types.append((event_type, purpose))

    if layout in {"combined", "single_run"}:
        events = workbook.create_sheet("Events")
        events.append(PROTOCOL_EVENT_COLUMNS)
        for run_id in cleaned_runs:
            for row in _template_event_rows(run_id, run_dates[run_id]):
                events.append(
                    tuple(row.get(column, "") for column in PROTOCOL_EVENT_COLUMNS)
                )
    else:
        for run_id in cleaned_runs:
            events = workbook.create_sheet(run_id[:31])
            events.append(PROTOCOL_EVENT_COLUMNS)
            for row in _template_event_rows(run_id, run_dates[run_id]):
                events.append(
                    tuple(row.get(column, "") for column in PROTOCOL_EVENT_COLUMNS)
                )

    header_fill = PatternFill("solid", fgColor="2F596D")
    for sheet in workbook.worksheets[1:]:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = header_fill
        for column in sheet.columns:
            letter = column[0].column_letter
            sheet.column_dimensions[letter].width = min(
                42,
                max(12, max(len(str(cell.value or "")) for cell in column) + 2),
            )
    workbook.save(destination)
    return destination


def write_protocol_file_set_template(
    directory: Path,
    *,
    run_ids: Sequence[str] = ("run01", "run02"),
) -> Path:
    """Write one editable Excel protocol workbook per run into an empty folder."""

    cleaned_runs = [
        str(value).strip().casefold() for value in run_ids if str(value).strip()
    ] or ["run01"]
    destination = Path(directory).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Protocol example folder is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for run_id in cleaned_runs:
        write_protocol_workbook_template(
            destination / f"{run_id}.xlsx",
            layout="single_run",
            run_ids=(run_id,),
        )
    (destination / "README.md").write_text(
        "# One Excel protocol workbook per run\n\n"
        "Rename each Excel workbook to the corresponding run ID and keep the "
        "`run_id` value in the workbook consistent with that name. Replace all "
        "example values. Record the run's "
        "`experiment_date` and one `experiment_start` defining t=0. `timezone` is "
        "optional and normally set once for the run. "
        "A clock time without a date uses `experiment_date`; events on later days "
        "must include their full date. Record actual camera intervals and their "
        "first images, injection ramp/steady/ramp segments, pH measurements, "
        "overflow, flushing, and interventions when relevant. Keep the column "
        "names and worksheet names unchanged. CSV and TSV remain supported only "
        "for importing legacy protocols.\n",
        encoding="utf-8",
    )
    return destination


def read_protocol_tables(path: Path) -> list[dict[str, Any]]:
    """Read one table, all workbook event sheets, or all tables in a folder."""

    source = Path(path).expanduser().resolve()
    if source.is_dir():
        rows: list[dict[str, Any]] = []
        for candidate in sorted(source.iterdir()):
            if candidate.suffix.casefold() not in {
                ".csv",
                ".tsv",
                ".txt",
                ".xls",
                ".xlsx",
            }:
                continue
            candidate_rows = read_protocol_tables(candidate)
            inferred_run = candidate.stem.strip().casefold()
            for row in candidate_rows:
                if not str(row.get("run_id") or row.get("run") or "").strip():
                    row["run_id"] = inferred_run
            rows.extend(candidate_rows)
        return rows
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.casefold()
    if suffix in {".csv", ".tsv", ".txt"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            sample = stream.read(4096)
            stream.seek(0)
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
            except csv.Error:
                pass
            return [dict(row) for row in csv.DictReader(stream, delimiter=delimiter)]
    if suffix not in {".xls", ".xlsx"}:
        raise ValueError("Protocol import supports CSV, TSV, XLSX, and XLS files")
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Excel import requires pandas and an Excel reader") from exc
    sheets = pd.read_excel(source, sheet_name=None)
    run_metadata: dict[str, dict[str, Any]] = {}
    for sheet_name, frame in sheets.items():
        if str(sheet_name).strip().casefold() != "runs":
            continue
        records = frame.where(frame.notna(), None).to_dict(orient="records")
        for record in records:
            metadata = {
                str(key).strip().casefold(): value for key, value in record.items()
            }
            run_id = (
                str(metadata.get("run_id") or metadata.get("run") or "")
                .strip()
                .casefold()
            )
            if run_id:
                run_metadata[run_id] = metadata

    rows = []
    for sheet_name, frame in sheets.items():
        normalized_sheet = str(sheet_name).strip().casefold()
        if normalized_sheet in RESERVED_SHEETS:
            continue
        records = frame.where(frame.notna(), None).to_dict(orient="records")
        for record in records:
            row = dict(record)
            if (
                normalized_sheet != "events"
                and not str(row.get("run_id") or row.get("run") or "").strip()
            ):
                row["run_id"] = normalized_sheet
            row_run = str(row.get("run_id") or row.get("run") or "").strip().casefold()
            for key, value in run_metadata.get(row_run, {}).items():
                current = row.get(key)
                if current is None or str(current).strip().casefold() in {
                    "",
                    "none",
                    "nan",
                    "nat",
                }:
                    row[key] = value
            row["__source_sheet"] = str(sheet_name)
            rows.append(row)

    origin_runs = {
        str(row.get("run_id") or row.get("run") or "").strip().casefold()
        for row in rows
        if str(row.get("event_type") or row.get("type") or "").strip().casefold()
        in {"experiment_start", "time_origin"}
    }
    for run_id, metadata in run_metadata.items():
        start = metadata.get("experiment_start_datetime")
        if run_id in origin_runs or start is None or not str(start).strip():
            continue
        rows.append(
            {
                **metadata,
                "run_id": run_id,
                "event_type": "experiment_start",
                "datetime": start,
                "time_h": 0.0,
                "timing_basis": "actual",
                "label": "Experiment time origin from Runs sheet",
                "source": "protocol",
                "__source_sheet": "Runs",
            }
        )
    return rows


def protocol_layout_summary(path: Path) -> dict[str, Any]:
    rows = read_protocol_tables(path)
    run_ids = sorted(
        {
            str(row.get("run_id") or row.get("run") or "").strip().casefold()
            for row in rows
            if str(row.get("run_id") or row.get("run") or "").strip()
        }
    )
    sheets = sorted(
        {
            str(row.get("__source_sheet") or "")
            for row in rows
            if str(row.get("__source_sheet") or "")
        }
    )
    return {"rows": len(rows), "run_ids": run_ids, "sheets": sheets}


__all__ = [
    "PROTOCOL_EVENT_COLUMNS",
    "PROTOCOL_INJECTION_POINT_COLUMNS",
    "PROTOCOL_RUN_COLUMNS",
    "PROTOCOL_WORKBOOK_FORMAT",
    "protocol_layout_summary",
    "read_protocol_tables",
    "write_protocol_file_set_template",
    "write_protocol_workbook_template",
]
