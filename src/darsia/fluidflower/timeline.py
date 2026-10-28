"""Versioned, non-destructive temporal alignment for FluidFlower projects."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import uuid
from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from darsia.fluidflower.protocol_schema import read_protocol_tables

if TYPE_CHECKING:
    from darsia.fluidflower.project import FluidFlowerProject


TIMELINE_EVENT_FORMAT = "darsia-fluidflower-timeline-event-v1"
TIMELINE_OVERRIDE_FORMAT = "darsia-fluidflower-timeline-override-v1"

_TIME_ALIASES = (
    "time_h",
    "elapsed_h",
    "elapsed_hours",
    "experiment_time_h",
    "hours",
    "time",
)
_DATETIME_ALIASES = (
    "datetime",
    "event_datetime",
    "timestamp",
    "image_time",
    "recorded_at",
)
_START_DATETIME_ALIASES = ("start_datetime", "start_timestamp", "start")
_END_DATETIME_ALIASES = ("end_datetime", "end_timestamp", "end")
_START_TIME_ALIASES = ("start_h", "start_time_h")
_END_TIME_ALIASES = ("end_h", "end_time_h")
_VALUE_ALIASES = (
    "value",
    "ph",
    "camera_interval_s",
    "image_interval_s",
    "interval_s",
    "injection_rate_ml_min",
    "weight_g",
    "mass_g",
    "rate_ml_min",
    "rate_sccm",
    "flow_percent",
)
_PATH_ALIASES = (
    "path",
    "first_image",
    "image_path",
    "file_path",
    "filename",
)
_ORIGIN_EVENT_TYPES = {"experiment_start", "time_origin"}
_EDITABLE_FIELDS = {
    "time_h",
    "value",
    "label",
    "note",
    "status",
    "run_id",
    "source",
    "event_type",
}


def _utc_now() -> str:
    from darsia.fluidflower.project import utc_now

    return utc_now()


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _append_jsonl as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON record in {path} at line {line_number}"
                ) from exc
            if not isinstance(payload, dict):
                raise TypeError(
                    f"Timeline record in {path} at line {line_number} is not an object"
                )
            records.append(payload)
    return records


def _first(row: Mapping[str, Any], aliases: Iterable[str]) -> tuple[str, Any]:
    normalized = {str(key).strip().casefold(): value for key, value in row.items()}
    for alias in aliases:
        if alias in normalized and not _is_blank(normalized[alias]):
            return alias, normalized[alias]
    return "", None


def _is_blank(value: Any) -> bool:
    return value is None or str(value).strip().casefold() in {
        "",
        "none",
        "nat",
        "nan",
    }


def _float_or_none(value: Any) -> float | None:
    if _is_blank(value):
        return None
    try:
        number = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _date_or_none(value: Any) -> date | None:
    if _is_blank(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _zone_or_none(value: Any) -> ZoneInfo | None:
    name = str(value or "").strip()
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (ValueError, ZoneInfoNotFoundError):
        return None


def _datetime_or_none(
    value: Any,
    *,
    fallback_date: date | None = None,
    timezone_name: str = "",
) -> datetime | None:
    if _is_blank(value):
        return None
    parsed: datetime | None = None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, time):
        if fallback_date is not None:
            parsed = datetime.combine(fallback_date, value)
    elif isinstance(value, date):
        parsed = datetime.combine(value, time())
    text = str(value).strip().replace("Z", "+00:00")
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            if fallback_date is None:
                return None
            try:
                parsed = datetime.combine(fallback_date, time.fromisoformat(text))
            except ValueError:
                return None
    zone = _zone_or_none(timezone_name)
    if parsed.tzinfo is None and zone is not None:
        parsed = parsed.replace(tzinfo=zone)
    return parsed


def _elapsed_hours(value: datetime, origin: datetime) -> float:
    if value.tzinfo is not None and origin.tzinfo is not None:
        delta = value.astimezone(timezone.utc) - origin.astimezone(timezone.utc)
    else:
        delta = value.replace(tzinfo=None) - origin.replace(tzinfo=None)
    return delta.total_seconds() / 3600.0


def _explicit_unit(row: Mapping[str, Any], value_column: str) -> str:
    unit = str(row.get("unit") or "").strip()
    return unit or _unit_for_column(value_column)


def _boundary_value(row: Mapping[str, Any], boundary: str) -> tuple[str, Any]:
    dedicated = f"{boundary}_value"
    if not _is_blank(row.get(dedicated)):
        return dedicated, row[dedicated]
    return _first(row, _VALUE_ALIASES)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date, time, Path)):
        return value.isoformat() if not isinstance(value, Path) else str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _unit_for_column(column: str) -> str:
    return {
        "weight_g": "g",
        "mass_g": "g",
        "ph": "pH",
        "camera_interval_s": "s",
        "image_interval_s": "s",
        "interval_s": "s",
        "injection_rate_ml_min": "ml/min",
        "rate_ml_min": "ml/min",
        "rate_sccm": "sccm",
        "flow_percent": "%",
    }.get(column, "")


def _load_table(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix in {".csv", ".tsv", ".txt"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            sample = stream.read(4096)
            stream.seek(0)
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
            except csv.Error:
                pass
            return [dict(row) for row in csv.DictReader(stream, delimiter=delimiter)]
    if suffix in {".xlsx", ".xls"}:
        return read_protocol_tables(path)
    raise ValueError("Timeline import supports CSV, TSV, XLSX, and XLS files")


def _source_digest(path: Path) -> str:
    from .project import _sha256_file_uncached
    return _sha256_file_uncached(path)


class TimelineStore:
    """Authoritative local timeline assembled from immutable raw events."""

    def __init__(self, project: FluidFlowerProject) -> None:
        self.project = project
        self.raw_path = project.root / "timeline" / "raw_events.jsonl"
        self.override_path = project.root / "timeline" / "overrides.jsonl"

    def raw_events(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.raw_path)

    def overrides(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.override_path)

    def add_event(
        self,
        *,
        run_id: str,
        source: str,
        event_type: str,
        time_h: float | None,
        label: str = "",
        value: float | None = None,
        unit: str = "",
        path: str = "",
        timestamp: str = "",
        actor: str = "",
        metadata: Mapping[str, Any] | None = None,
        import_id: str = "manual",
    ) -> dict[str, Any]:
        event = {
            "schema": 1,
            "format": TIMELINE_EVENT_FORMAT,
            "event_id": uuid.uuid4().hex,
            "project_id": self.project.project_id,
            "recorded_utc": _utc_now(),
            "recorded_by": str(actor).strip(),
            "import_id": import_id,
            "run_id": str(run_id).strip().casefold(),
            "source": str(source).strip().casefold() or "manual",
            "event_type": str(event_type).strip().casefold() or "event",
            "raw_time_h": _float_or_none(time_h),
            "raw_timestamp": str(timestamp).strip(),
            "raw_value": _float_or_none(value),
            "unit": str(unit).strip(),
            "label": str(label).strip(),
            "path": str(path).strip(),
            "metadata": _json_safe(deepcopy(dict(metadata or {}))),
        }
        _append_jsonl(self.raw_path, event)
        return deepcopy(event)

    def import_table(
        self,
        path: Path,
        *,
        source: str,
        run_id: str = "",
        actor: str = "",
    ) -> dict[str, Any]:
        source_path = Path(path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        rows = _load_table(source_path)
        digest = _source_digest(source_path)
        import_id = uuid.uuid4().hex
        existing_fingerprints = {
            str((event.get("metadata") or {}).get("source_fingerprint") or "")
            for event in self.raw_events()
        }

        normalized_rows: list[tuple[int, dict[str, Any], str, str]] = []
        for row_number, row in enumerate(rows, start=2):
            normalized = {
                str(key).strip().casefold(): value for key, value in row.items()
            }
            row_run = (
                str(normalized.get("run_id") or normalized.get("run") or run_id)
                .strip()
                .casefold()
            )
            source_sheet = str(normalized.get("__source_sheet") or "")
            normalized_rows.append((row_number, normalized, row_run, source_sheet))

        dates_by_run: dict[str, date] = {}
        timezones_by_run: dict[str, str] = {}
        for _, normalized, row_run, _ in normalized_rows:
            experiment_date = _date_or_none(normalized.get("experiment_date"))
            if experiment_date is not None:
                dates_by_run.setdefault(row_run, experiment_date)
            timezone_name = str(normalized.get("timezone") or "").strip()
            if timezone_name:
                timezones_by_run.setdefault(row_run, timezone_name)

        existing_origins: dict[str, datetime] = {}
        for event in self.raw_events():
            if str(event.get("event_type") or "").casefold() not in _ORIGIN_EVENT_TYPES:
                continue
            parsed = _datetime_or_none(event.get("raw_timestamp"))
            if parsed is not None:
                existing_origins.setdefault(
                    str(event.get("run_id") or "").casefold(), parsed
                )

        datetimes_by_run: dict[str, list[datetime]] = {}
        source_origins: dict[str, datetime] = {}
        for _, normalized, row_run, _ in normalized_rows:
            fallback_date = dates_by_run.get(row_run)
            timezone_name = timezones_by_run.get(row_run, "")
            _, single_value = _first(normalized, _DATETIME_ALIASES)
            single_timestamp = _datetime_or_none(
                single_value,
                fallback_date=fallback_date,
                timezone_name=timezone_name,
            )
            parsed_values = [single_timestamp]
            for aliases in (_START_DATETIME_ALIASES, _END_DATETIME_ALIASES):
                _, raw_value = _first(normalized, aliases)
                parsed_values.append(
                    _datetime_or_none(
                        raw_value,
                        fallback_date=fallback_date,
                        timezone_name=timezone_name,
                    )
                )
            datetimes_by_run.setdefault(row_run, []).extend(
                value for value in parsed_values if value is not None
            )
            event_type = (
                str(normalized.get("event_type") or normalized.get("type") or "")
                .strip()
                .casefold()
            )
            if event_type in _ORIGIN_EVENT_TYPES and single_timestamp is not None:
                source_origins[row_run] = single_timestamp

        origins: dict[str, datetime] = {}
        origin_policies: dict[str, str] = {}
        all_runs = set(datetimes_by_run) | set(existing_origins) | set(source_origins)
        for row_run in all_runs:
            if row_run in source_origins:
                origins[row_run] = source_origins[row_run]
                origin_policies[row_run] = "experiment_start_in_source"
            elif row_run in existing_origins:
                origins[row_run] = existing_origins[row_run]
                origin_policies[row_run] = "experiment_start_in_project"
            elif datetimes_by_run.get(row_run):
                origins[row_run] = min(
                    datetimes_by_run[row_run],
                    key=lambda value: value.replace(tzinfo=None),
                )
                origin_policies[row_run] = "first_timestamp_in_source"

        added = 0
        skipped = 0
        for row_number, normalized, row_run, source_sheet in normalized_rows:
            fallback_date = dates_by_run.get(row_run)
            timezone_name = timezones_by_run.get(row_run, "")
            origin = origins.get(row_run)
            origin_policy = origin_policies.get(row_run, "")
            start_datetime_column, raw_start_datetime = _first(
                normalized, _START_DATETIME_ALIASES
            )
            end_datetime_column, raw_end_datetime = _first(
                normalized, _END_DATETIME_ALIASES
            )
            start_time_column, raw_start_time = _first(normalized, _START_TIME_ALIASES)
            end_time_column, raw_end_time = _first(normalized, _END_TIME_ALIASES)
            has_timestamp_interval = bool(start_datetime_column and end_datetime_column)
            has_hour_interval = bool(start_time_column and end_time_column)
            if has_timestamp_interval or has_hour_interval:
                interval_id = str(normalized.get("id") or row_number)
                boundary_times: dict[
                    str,
                    tuple[
                        str, datetime | None, float | None, float | None, float | None
                    ],
                ] = {}
                for boundary in ("start", "end"):
                    datetime_column = (
                        start_datetime_column
                        if boundary == "start"
                        else end_datetime_column
                    )
                    raw_datetime = (
                        raw_start_datetime if boundary == "start" else raw_end_datetime
                    )
                    time_column = (
                        start_time_column if boundary == "start" else end_time_column
                    )
                    raw_time = raw_start_time if boundary == "start" else raw_end_time
                    timestamp = _datetime_or_none(
                        raw_datetime,
                        fallback_date=fallback_date,
                        timezone_name=timezone_name,
                    )
                    declared_time_h = _float_or_none(raw_time)
                    timestamp_time_h = (
                        _elapsed_hours(timestamp, origin)
                        if timestamp is not None and origin is not None
                        else None
                    )
                    if (
                        timestamp_time_h is not None
                        and origin_policy != "first_timestamp_in_source"
                    ):
                        time_h = timestamp_time_h
                        time_policy = origin_policy
                    elif declared_time_h is not None:
                        time_h = declared_time_h
                        time_policy = "explicit_hours"
                    else:
                        time_h = timestamp_time_h
                        time_policy = origin_policy
                    boundary_times[boundary] = (
                        datetime_column or time_column,
                        timestamp,
                        declared_time_h,
                        timestamp_time_h,
                        time_h,
                    )

                interval_duration_h = None
                if all(
                    boundary_times[boundary][4] is not None
                    for boundary in ("start", "end")
                ):
                    interval_duration_h = float(boundary_times["end"][4]) - float(
                        boundary_times["start"][4]
                    )

                for boundary in ("start", "end"):
                    (
                        time_column,
                        timestamp,
                        declared_time_h,
                        timestamp_time_h,
                        time_h,
                    ) = boundary_times[boundary]
                    base_event_type = (
                        str(
                            normalized.get("event_type")
                            or normalized.get("type")
                            or "injection"
                        )
                        .strip()
                        .casefold()
                    )
                    event_type = f"{base_event_type}_{boundary}"
                    label = str(
                        normalized.get("what") or normalized.get("label") or interval_id
                    )
                    fingerprint = (
                        f"{digest}:{source_path}:{source_sheet}:{row_number}:{boundary}"
                    )
                    if fingerprint in existing_fingerprints:
                        skipped += 1
                        continue
                    value_column, raw_value = _boundary_value(normalized, boundary)
                    path_column, raw_path = _first(normalized, _PATH_ALIASES)
                    self.add_event(
                        run_id=row_run,
                        source=source,
                        event_type=event_type,
                        time_h=time_h,
                        label=label,
                        value=_float_or_none(raw_value),
                        unit=_explicit_unit(normalized, value_column),
                        path=str(raw_path or "") if boundary == "start" else "",
                        timestamp=timestamp.isoformat() if timestamp else "",
                        actor=actor,
                        import_id=import_id,
                        metadata={
                            "source_file": str(source_path),
                            "source_sha256": digest,
                            "source_fingerprint": fingerprint,
                            "source_row": row_number,
                            "source_sheet": source_sheet,
                            "interval_id": interval_id,
                            "boundary": boundary,
                            "time_column": time_column,
                            "value_column": value_column,
                            "path_column": path_column if boundary == "start" else "",
                            "experiment_date": (
                                fallback_date.isoformat() if fallback_date else ""
                            ),
                            "timezone": timezone_name,
                            "timezone_recognized": bool(
                                not timezone_name or _zone_or_none(timezone_name)
                            ),
                            "declared_time_h": declared_time_h,
                            "timestamp_time_h": timestamp_time_h,
                            "relative_origin": origin.isoformat() if origin else "",
                            "relative_origin_policy": time_policy,
                            "interval_duration_h": interval_duration_h,
                            "raw_row": normalized,
                        },
                    )
                    added += 1
                continue

            time_column, raw_time = _first(normalized, _TIME_ALIASES)
            datetime_column, raw_datetime = _first(normalized, _DATETIME_ALIASES)
            timestamp = _datetime_or_none(
                raw_datetime,
                fallback_date=fallback_date,
                timezone_name=timezone_name,
            )
            declared_time_h = _float_or_none(raw_time)
            timestamp_time_h = (
                _elapsed_hours(timestamp, origin)
                if timestamp is not None and origin is not None
                else None
            )
            if (
                timestamp_time_h is not None
                and origin_policy != "first_timestamp_in_source"
            ):
                time_h = timestamp_time_h
                time_policy = origin_policy
            elif declared_time_h is not None:
                time_h = declared_time_h
                time_policy = "explicit_hours"
            else:
                time_h = timestamp_time_h
                time_policy = origin_policy
            value_column, raw_value = _first(normalized, _VALUE_ALIASES)
            path_column, raw_path = _first(normalized, _PATH_ALIASES)
            event_type = str(
                normalized.get("event_type") or normalized.get("type") or ""
            )
            if not event_type:
                event_type = (
                    "image_capture"
                    if "image_id" in normalized or raw_path
                    else "measurement"
                )
            label = str(
                normalized.get("label")
                or normalized.get("name")
                or normalized.get("image_id")
                or row_number
            )
            fingerprint = f"{digest}:{source_path}:{source_sheet}:{row_number}:event"
            if fingerprint in existing_fingerprints:
                skipped += 1
                continue
            self.add_event(
                run_id=row_run,
                source=source,
                event_type=event_type,
                time_h=time_h,
                label=label,
                value=_float_or_none(raw_value),
                unit=_explicit_unit(normalized, value_column),
                path=str(raw_path or ""),
                timestamp=timestamp.isoformat() if timestamp else "",
                actor=actor,
                import_id=import_id,
                metadata={
                    "source_file": str(source_path),
                    "source_sha256": digest,
                    "source_fingerprint": fingerprint,
                    "source_row": row_number,
                    "source_sheet": source_sheet,
                    "time_column": time_column or datetime_column,
                    "value_column": value_column,
                    "path_column": path_column,
                    "experiment_date": (
                        fallback_date.isoformat() if fallback_date else ""
                    ),
                    "timezone": timezone_name,
                    "timezone_recognized": bool(
                        not timezone_name or _zone_or_none(timezone_name)
                    ),
                    "declared_time_h": declared_time_h,
                    "timestamp_time_h": timestamp_time_h,
                    "relative_origin": origin.isoformat() if origin else "",
                    "relative_origin_policy": time_policy,
                    "raw_row": normalized,
                },
            )
            added += 1

        self.project._audit(
            "timeline_table_imported",
            actor,
            {
                "path": str(source_path),
                "source": source,
                "run_id": run_id,
                "import_id": import_id,
                "added": added,
                "skipped": skipped,
            },
        )
        return {"import_id": import_id, "added": added, "skipped": skipped}

    def import_folder(
        self,
        path: Path,
        *,
        source: str,
        actor: str = "",
    ) -> dict[str, Any]:
        """Import one timeline table per run from a selected folder."""

        folder = Path(path).expanduser().resolve()
        if not folder.is_dir():
            raise NotADirectoryError(folder)
        added = 0
        skipped = 0
        import_ids: list[str] = []
        supported = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
        for candidate in sorted(folder.iterdir()):
            if not candidate.is_file() or candidate.suffix.casefold() not in supported:
                continue
            result = self.import_table(
                candidate,
                source=source,
                run_id=candidate.stem.strip().casefold(),
                actor=actor,
            )
            added += int(result["added"])
            skipped += int(result["skipped"])
            import_ids.append(str(result["import_id"]))
        return {"import_ids": import_ids, "added": added, "skipped": skipped}

    def import_registered_protocols(self, *, actor: str = "") -> dict[str, Any]:
        """Import the protocol workbooks already registered during series setup.

        Shared workbooks are read once. Per-run workbooks use their registered run ID
        as a fallback when a sheet has no explicit ``run_id``. Existing source rows
        are retained without duplication through the normal import fingerprint.
        """

        sources: dict[str, set[str]] = {}
        unavailable: list[dict[str, str]] = []
        for raw_run in self.project.manifest.get("runs") or []:
            if not isinstance(raw_run, Mapping):
                continue
            run_id = str(raw_run.get("run_id") or "").strip().casefold()
            mapping = raw_run.get("source_mapping") or {}
            if not isinstance(mapping, Mapping):
                continue
            raw_paths = list(mapping.get("protocol_tables") or [])
            if mapping.get("protocol_table"):
                raw_paths.append(mapping.get("protocol_table"))
            for raw_path in raw_paths:
                text = str(raw_path or "").strip()
                if not text:
                    continue
                path = Path(text).expanduser()
                if not path.is_file():
                    unavailable.append({"run_id": run_id, "path": str(path)})
                    continue
                sources.setdefault(str(path.resolve()), set()).add(run_id)

        added = 0
        skipped = 0
        imported: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for source_path, run_ids in sorted(sources.items()):
            fallback_run = next(iter(run_ids)) if len(run_ids) == 1 else ""
            try:
                result = self.import_table(
                    Path(source_path),
                    source="registered_protocol",
                    run_id=fallback_run,
                    actor=actor,
                )
            except Exception as exc:  # noqa: BLE001 - continue with other experiments
                errors.append(
                    {
                        "path": source_path,
                        "run_ids": ",".join(sorted(run_ids)),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            added += int(result["added"])
            skipped += int(result["skipped"])
            imported.append(
                {
                    "path": source_path,
                    "run_ids": sorted(run_ids),
                    "added": int(result["added"]),
                    "skipped": int(result["skipped"]),
                }
            )
        return {
            "registered_source_count": len(sources),
            "imported": imported,
            "added": added,
            "skipped": skipped,
            "unavailable": unavailable,
            "errors": errors,
        }

    def set_override(
        self,
        event_id: str,
        changes: Mapping[str, Any],
        *,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        if event_id not in {str(event.get("event_id")) for event in self.raw_events()}:
            raise KeyError(f"Unknown timeline event: {event_id}")
        invalid = sorted(set(changes) - _EDITABLE_FIELDS)
        if invalid:
            raise ValueError(
                f"Unsupported timeline override fields: {', '.join(invalid)}"
            )
        normalized = deepcopy(dict(changes))
        if "time_h" in normalized:
            normalized["time_h"] = _float_or_none(normalized["time_h"])
        if "value" in normalized:
            normalized["value"] = _float_or_none(normalized["value"])
        record = {
            "schema": 1,
            "format": TIMELINE_OVERRIDE_FORMAT,
            "override_id": uuid.uuid4().hex,
            "event_id": event_id,
            "recorded_utc": _utc_now(),
            "recorded_by": str(actor).strip(),
            "reason": str(reason).strip(),
            "action": "set",
            "changes": normalized,
        }
        _append_jsonl(self.override_path, record)
        self.project._audit("timeline_event_overridden", actor, record)
        return deepcopy(record)

    def clear_overrides(
        self,
        event_id: str,
        fields: Iterable[str],
        *,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        selected = sorted(set(fields))
        invalid = sorted(set(selected) - _EDITABLE_FIELDS)
        if invalid:
            raise ValueError(
                f"Unsupported timeline override fields: {', '.join(invalid)}"
            )
        record = {
            "schema": 1,
            "format": TIMELINE_OVERRIDE_FORMAT,
            "override_id": uuid.uuid4().hex,
            "event_id": event_id,
            "recorded_utc": _utc_now(),
            "recorded_by": str(actor).strip(),
            "reason": str(reason).strip(),
            "action": "clear",
            "fields": selected,
        }
        _append_jsonl(self.override_path, record)
        self.project._audit("timeline_event_override_cleared", actor, record)
        return deepcopy(record)

    def effective_events(self) -> list[dict[str, Any]]:
        raw = self.raw_events()
        by_id = {str(event["event_id"]): deepcopy(event) for event in raw}
        active: dict[str, dict[str, Any]] = {event_id: {} for event_id in by_id}
        for record in self.overrides():
            event_id = str(record.get("event_id") or "")
            if event_id not in active:
                continue
            if record.get("action") == "clear":
                for field in record.get("fields") or []:
                    active[event_id].pop(str(field), None)
            elif record.get("action") == "set":
                active[event_id].update(dict(record.get("changes") or {}))
        for event_id, event in by_id.items():
            event["time_h"] = event.get("raw_time_h")
            event["value"] = event.get("raw_value")
            event["note"] = ""
            event["status"] = "active"
            event["overrides"] = deepcopy(active[event_id])
            event.update(active[event_id])
        return sorted(
            by_id.values(),
            key=lambda event: (
                str(event.get("run_id") or ""),
                float(event.get("time_h"))
                if event.get("time_h") is not None
                else math.inf,
                str(event.get("source") or ""),
                str(event.get("event_type") or ""),
            ),
        )

    def qc_issues(self) -> list[dict[str, Any]]:
        events = self.effective_events()
        issues: list[dict[str, Any]] = []
        duplicate_groups: dict[tuple[str, str, str, float], list[str]] = {}
        reported_relative_origins: set[str] = set()
        reported_missing_dates: set[tuple[str, str]] = set()
        reported_invalid_timezones: set[tuple[str, str]] = set()
        for event in events:
            event_id = str(event["event_id"])
            time_h = event.get("time_h")
            if time_h is None or not math.isfinite(float(time_h)):
                issues.append(
                    {
                        "event_id": event_id,
                        "severity": "error",
                        "code": "missing_time",
                        "message": "No effective experiment time is defined.",
                    }
                )
            else:
                key = (
                    str(event.get("run_id") or ""),
                    str(event.get("source") or ""),
                    str(event.get("event_type") or ""),
                    round(float(time_h), 9),
                )
                duplicate_groups.setdefault(key, []).append(event_id)
            path = str(event.get("path") or "")
            if path and not Path(path).expanduser().exists():
                issues.append(
                    {
                        "event_id": event_id,
                        "severity": "warning",
                        "code": "missing_linked_file",
                        "message": f"Linked source file is unavailable: {path}",
                    }
                )
            raw_time = event.get("raw_time_h")
            if (
                raw_time is not None
                and time_h is not None
                and abs(float(time_h) - float(raw_time)) > 1.0
            ):
                issues.append(
                    {
                        "event_id": event_id,
                        "severity": "info",
                        "code": "large_local_time_adjustment",
                        "message": "Local time differs from raw time by more than one hour.",
                    }
                )
            metadata = dict(event.get("metadata") or {})
            raw_row = dict(metadata.get("raw_row") or {})
            declared_time_h = _float_or_none(metadata.get("declared_time_h"))
            timestamp_time_h = _float_or_none(metadata.get("timestamp_time_h"))
            if (
                declared_time_h is not None
                and timestamp_time_h is not None
                and abs(declared_time_h - timestamp_time_h) > 1.0 / 3600.0
            ):
                issues.append(
                    {
                        "event_id": event_id,
                        "severity": "warning",
                        "code": "time_source_disagreement",
                        "message": (
                            "Declared relative time and the dated timestamp differ by "
                            "more than one second."
                        ),
                    }
                )
            experiment_date = str(metadata.get("experiment_date") or "")
            date_key = (
                str(event.get("import_id") or ""),
                str(event.get("run_id") or ""),
            )
            if (
                "experiment_date" in raw_row
                and not experiment_date
                and date_key not in reported_missing_dates
            ):
                reported_missing_dates.add(date_key)
                issues.append(
                    {
                        "event_id": event_id,
                        "severity": "warning",
                        "code": "missing_experiment_date",
                        "message": "The run has no experiment date.",
                    }
                )
            timezone_name = str(metadata.get("timezone") or "")
            timezone_key = (str(event.get("import_id") or ""), timezone_name)
            if (
                timezone_name
                and not bool(metadata.get("timezone_recognized", True))
                and timezone_key not in reported_invalid_timezones
            ):
                reported_invalid_timezones.add(timezone_key)
                issues.append(
                    {
                        "event_id": event_id,
                        "severity": "info",
                        "code": "unknown_timezone",
                        "message": (
                            f"Timezone {timezone_name!r} is not available locally; "
                            "timestamps are treated as consistent local time."
                        ),
                    }
                )
            timing_basis = str(raw_row.get("timing_basis") or "").strip().casefold()
            if timing_basis and timing_basis not in {"actual", "planned", "estimated"}:
                issues.append(
                    {
                        "event_id": event_id,
                        "severity": "warning",
                        "code": "invalid_timing_basis",
                        "message": "timing_basis must be actual, planned, or estimated.",
                    }
                )
            event_type = str(event.get("event_type") or "").casefold()
            boundary = str(metadata.get("boundary") or "")
            if boundary in {"", "start"}:
                interval_duration_h = _float_or_none(
                    metadata.get("interval_duration_h")
                )
                if interval_duration_h is not None and interval_duration_h < 0:
                    issues.append(
                        {
                            "event_id": event_id,
                            "severity": "error",
                            "code": "negative_interval_duration",
                            "message": "The interval ends before it starts.",
                        }
                    )
                if event_type.startswith("injection_"):
                    if _is_blank(raw_row.get("location_id")):
                        issues.append(
                            {
                                "event_id": event_id,
                                "severity": "warning",
                                "code": "missing_injection_location",
                                "message": "The injection segment has no injection-point ID.",
                            }
                        )
                    if event.get("raw_value") is None:
                        issues.append(
                            {
                                "event_id": event_id,
                                "severity": "warning",
                                "code": "missing_injection_rate",
                                "message": "The injection segment has no rate value.",
                            }
                        )
            if event_type == "camera_interval_start":
                if event.get("raw_value") is None:
                    issues.append(
                        {
                            "event_id": event_id,
                            "severity": "warning",
                            "code": "missing_camera_interval",
                            "message": "The camera interval is not specified.",
                        }
                    )
                if not str(event.get("path") or ""):
                    issues.append(
                        {
                            "event_id": event_id,
                            "severity": "warning",
                            "code": "missing_first_interval_image",
                            "message": "The first image for this camera interval is missing.",
                        }
                    )
            if event_type == "ph_measurement":
                ph = _float_or_none(event.get("raw_value"))
                if ph is None or not 0.0 <= ph <= 14.0:
                    issues.append(
                        {
                            "event_id": event_id,
                            "severity": "error",
                            "code": "invalid_ph_measurement",
                            "message": "pH must be a number between 0 and 14.",
                        }
                    )
                if _is_blank(raw_row.get("measurement_location")):
                    issues.append(
                        {
                            "event_id": event_id,
                            "severity": "warning",
                            "code": "missing_ph_location",
                            "message": "The pH measurement location is not specified.",
                        }
                    )
            import_id = str(event.get("import_id") or "")
            if (
                metadata.get("relative_origin_policy") == "first_timestamp_in_source"
                and import_id not in reported_relative_origins
            ):
                reported_relative_origins.add(import_id)
                issues.append(
                    {
                        "event_id": event_id,
                        "severity": "warning",
                        "code": "source_relative_origin",
                        "message": (
                            "This source is relative to its first timestamp and must be "
                            "aligned to the experiment anchor before approval."
                        ),
                    }
                )
        for event_ids in duplicate_groups.values():
            if len(event_ids) < 2:
                continue
            for event_id in event_ids:
                issues.append(
                    {
                        "event_id": event_id,
                        "severity": "warning",
                        "code": "duplicate_time",
                        "message": "Another event of the same type has the same effective time.",
                    }
                )
        return issues

    def summary(self) -> dict[str, int]:
        events = self.effective_events()
        issues = self.qc_issues()
        return {
            "events": len(events),
            "adjusted": sum(bool(event.get("overrides")) for event in events),
            "errors": sum(issue["severity"] == "error" for issue in issues),
            "warnings": sum(issue["severity"] == "warning" for issue in issues),
        }
