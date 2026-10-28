"""Versioned overflow-source selection and source validation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from darsia.fluidflower.modules import dependencies_satisfied

if TYPE_CHECKING:
    from darsia.fluidflower.project import FluidFlowerProject


OVERFLOW_CONFIG_FORMAT = "darsia-fluidflower-overflow-config-v1"
OVERFLOW_DRAFT_FORMAT = "darsia-fluidflower-overflow-draft-v1"
OVERFLOW_REVISION_FORMAT = "darsia-fluidflower-overflow-revision-v1"
OVERFLOW_SOURCE_MODES = ("none", "scale_images", "table")
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".xls", ".xlsx"}
_TIMESTAMP_PATTERN = re.compile(
    r"(?<!\d)(\d{4})[-_](\d{2})[-_](\d{2})[-_ T](\d{2})[-_:](\d{2})[-_:](\d{2})(?!\d)"
)
_TIME_COLUMNS = {
    "time",
    "time_h",
    "elapsed_h",
    "elapsed_hours",
    "experiment_time_h",
    "image_time",
    "timestamp",
    "datetime",
}
_WEIGHT_COLUMNS = {
    "weight",
    "weight_g",
    "mass_g",
    "scale_g",
    "scale_weight_g",
    "vekt",
    "vekt_g",
    "confirmed_value_g",
    "value_g",
}


def _utc_now() -> str:
    from darsia.fluidflower.project import utc_now

    return utc_now()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _atomic_json as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _append_jsonl as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(
                    f"Overflow revision at line {line_number} is not an object"
                )
            rows.append(payload)
    return rows


def _sha256_file(path: Path) -> str:
    from .project import _sha256_file_uncached
    return _sha256_file_uncached(path)


def _file_reference(path: Path) -> dict[str, Any]:
    selected = path.expanduser().resolve()
    stat = selected.stat()
    return {
        "path": str(selected),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(selected),
    }


def _normalized_columns(values: Sequence[Any]) -> list[str]:
    return [
        re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().casefold()).strip("_")
        for value in values
    ]


def overflow_run_source_digest(row: Mapping[str, Any]) -> str:
    payload = {
        "run_id": str(row.get("run_id") or "").strip().casefold(),
        "mode": str(row.get("mode") or "none").strip().casefold(),
        "image_folders": sorted(
            str(Path(value).expanduser().resolve())
            for value in row.get("image_folders") or []
            if str(value).strip()
        ),
        "table_path": (
            str(Path(str(row.get("table_path"))).expanduser().resolve())
            if str(row.get("table_path") or "").strip()
            else ""
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _discovered_overflow_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    run_id = str(raw.get("run_id") or "").strip().casefold()
    mapping = dict(raw.get("source_mapping") or {})
    image_folders = [
        str(Path(value).expanduser().resolve())
        for value in mapping.get("overflow_image_folders") or []
        if str(value).strip()
    ]
    tables = [
        str(Path(value).expanduser().resolve())
        for value in mapping.get("overflow_tables") or []
        if str(value).strip()
    ]
    if str(mapping.get("overflow_table") or "").strip():
        tables.insert(
            0, str(Path(str(mapping["overflow_table"])).expanduser().resolve())
        )
    tables = list(dict.fromkeys(tables))
    mode = "scale_images" if image_folders else "table" if tables else "none"
    return {
        "run_id": run_id,
        "mode": mode,
        "image_folders": image_folders,
        "table_path": tables[0] if tables else "",
        "discovered_image_folders": image_folders,
        "discovered_tables": tables,
        "source_origin": "setup_discovery",
        "scan": {},
    }


def default_overflow_config(project: FluidFlowerProject) -> dict[str, Any]:
    return {
        "schema": 1,
        "format": OVERFLOW_CONFIG_FORMAT,
        "source_policy": "select_per_run_after_data_integrity",
        "runs": [
            _discovered_overflow_row(row)
            for row in project.manifest.get("runs") or []
            if isinstance(row, Mapping) and str(row.get("run_id") or "").strip()
        ],
    }


def normalize_overflow_config(
    project: FluidFlowerProject, config: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Merge a saved selection with newly added experiments and discoveries."""

    defaults = default_overflow_config(project)
    existing = {
        str(row.get("run_id") or "").strip().casefold(): deepcopy(dict(row))
        for row in (config or {}).get("runs") or []
        if isinstance(row, Mapping) and str(row.get("run_id") or "").strip()
    }
    merged_rows: list[dict[str, Any]] = []
    for default_row in defaults["runs"]:
        run_id = str(default_row["run_id"])
        saved = existing.get(run_id)
        if saved is None:
            merged_rows.append(deepcopy(default_row))
            continue
        merged_row = deepcopy(default_row)
        merged_row.update(saved)
        # Discovery belongs to the current project setup, not to an older overflow
        # draft. Keep the active decision, but always expose newly registered sources.
        merged_row["discovered_image_folders"] = deepcopy(
            default_row["discovered_image_folders"]
        )
        merged_row["discovered_tables"] = deepcopy(default_row["discovered_tables"])
        merged_rows.append(merged_row)

    merged = deepcopy(dict(config or defaults))
    merged.update(
        {
            "schema": 1,
            "format": OVERFLOW_CONFIG_FORMAT,
            "source_policy": "select_per_run_after_data_integrity",
            "runs": merged_rows,
        }
    )
    return merged


def apply_registered_overflow_mode(
    config: Mapping[str, Any],
    *,
    mode: str,
    run_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one registered overflow-source policy for selected experiments.

    The operation only changes the working configuration. Saving an immutable
    overflow revision remains a separate, explicit step after the selected sources
    have been scanned.
    """

    selected_mode = str(mode).strip().casefold()
    if selected_mode not in OVERFLOW_SOURCE_MODES:
        raise ValueError(f"Unsupported overflow source mode: {selected_mode}")
    requested = tuple(
        dict.fromkeys(
            str(value).strip().casefold() for value in run_ids if str(value).strip()
        )
    )
    if not requested:
        raise ValueError("At least one experiment must be selected")

    candidate = deepcopy(dict(config))
    rows = [row for row in candidate.get("runs") or [] if isinstance(row, dict)]
    by_id = {
        str(row.get("run_id") or "").strip().casefold(): row
        for row in rows
        if str(row.get("run_id") or "").strip()
    }
    applied: list[str] = []
    changed: list[str] = []
    unchanged: list[str] = []
    skipped: dict[str, str] = {}

    for run_id in requested:
        row = by_id.get(run_id)
        if row is None:
            skipped[run_id] = "Experiment is not registered in this configuration."
            continue
        before = overflow_run_source_digest(row)
        if selected_mode == "table":
            tables = [
                str(value)
                for value in row.get("discovered_tables") or []
                if str(value).strip()
            ]
            if not tables:
                skipped[run_id] = "No Excel or CSV overflow file is registered."
                continue
            row["table_path"] = tables[0]
        elif selected_mode == "scale_images":
            image_folders = [
                str(value)
                for value in row.get("discovered_image_folders") or []
                if str(value).strip()
            ]
            if not image_folders:
                skipped[run_id] = "No scale-display image folder is registered."
                continue
            row["image_folders"] = image_folders

        row["mode"] = selected_mode
        row["source_origin"] = "global_registered_source"
        applied.append(run_id)
        if overflow_run_source_digest(row) != before:
            row["scan"] = {}
            changed.append(run_id)
        else:
            unchanged.append(run_id)

    summary = {
        "mode": selected_mode,
        "requested_run_ids": list(requested),
        "applied_run_ids": applied,
        "changed_run_ids": changed,
        "unchanged_run_ids": unchanged,
        "skipped": skipped,
    }
    if applied:
        candidate["last_bulk_assignment"] = {
            **deepcopy(summary),
            "recorded_utc": _utc_now(),
        }
    return candidate, summary


def _image_timestamp(path: Path) -> datetime | None:
    match = _TIMESTAMP_PATTERN.search(path.name)
    if match is None:
        return None
    try:
        # Camera filenames record local wall-clock time without a timezone.
        return datetime(*map(int, match.groups()))  # noqa: DTZ001
    except ValueError:
        return None


def _evenly_spaced(values: Sequence[Path], count: int) -> list[Path]:
    if len(values) <= count:
        return list(values)
    if count <= 1:
        return [values[len(values) // 2]]
    indices = {round(index * (len(values) - 1) / (count - 1)) for index in range(count)}
    return [values[index] for index in sorted(indices)]


def _scan_scale_images(row: Mapping[str, Any]) -> dict[str, Any]:
    folders = [
        Path(value).expanduser().resolve()
        for value in row.get("image_folders") or []
        if str(value).strip()
    ]
    errors: list[str] = []
    images: list[Path] = []
    for folder in folders:
        if not folder.is_dir():
            errors.append(f"Folder is unavailable: {folder}")
            continue
        try:
            images.extend(
                path
                for path in folder.rglob("*")
                if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
            )
        except OSError as exc:
            errors.append(f"Could not scan {folder}: {exc}")
    images = sorted(dict.fromkeys(images), key=lambda path: str(path).casefold())
    verified: list[dict[str, Any]] = []
    for path in _evenly_spaced(images, 5):
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            verified.append(
                {"path": str(path), "width": width, "height": height, "ok": True}
            )
        except (OSError, ValueError) as exc:
            errors.append(f"Unreadable image {path}: {exc}")
            verified.append({"path": str(path), "ok": False, "error": str(exc)})
    inventory_digest = hashlib.sha256()
    timestamps: list[datetime] = []
    total_bytes = 0
    for path in images:
        stat = path.stat()
        total_bytes += int(stat.st_size)
        inventory_digest.update(str(path).encode("utf-8", errors="surrogatepass"))
        inventory_digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}".encode())
        timestamp = _image_timestamp(path)
        if timestamp is not None:
            timestamps.append(timestamp)
    if not folders:
        errors.append("No scale-image folder is selected.")
    elif not images:
        errors.append("No supported scale-display images were found.")
    return {
        "kind": "scale_images",
        "status": "error" if errors else "ready",
        "source_digest": overflow_run_source_digest(row),
        "scanned_utc": _utc_now(),
        "folder_count": len(folders),
        "image_count": len(images),
        "timestamped_image_count": len(timestamps),
        "first_timestamp": min(timestamps).isoformat() if timestamps else None,
        "last_timestamp": max(timestamps).isoformat() if timestamps else None,
        "total_bytes": total_bytes,
        "inventory_sha256": inventory_digest.hexdigest(),
        "verified_samples": verified,
        "errors": errors,
        "warnings": (
            []
            if timestamps
            else ["No image timestamp could be parsed from the filenames."]
        ),
    }


def _csv_table_summary(path: Path) -> tuple[int, list[str], list[str]]:
    delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(4096)
        stream.seek(0)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
        except csv.Error:
            pass
        reader = csv.reader(stream, delimiter=delimiter)
        rows = list(reader)
    if not rows:
        return 0, [], [path.stem]
    columns = _normalized_columns(rows[0])
    count = sum(any(str(value).strip() for value in row) for row in rows[1:])
    return count, columns, [path.stem]


def _excel_table_summary(path: Path) -> tuple[int, list[str], list[str]]:
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None)
    columns: list[str] = []
    row_count = 0
    for frame in sheets.values():
        row_count += len(frame.dropna(how="all"))
        columns.extend(_normalized_columns(frame.columns.tolist()))
    return row_count, sorted(set(columns)), [str(value) for value in sheets]


def _scan_table(row: Mapping[str, Any]) -> dict[str, Any]:
    path_text = str(row.get("table_path") or "").strip()
    errors: list[str] = []
    warnings: list[str] = []
    columns: list[str] = []
    sheets: list[str] = []
    row_count = 0
    reference: dict[str, Any] = {}
    if not path_text:
        errors.append("No CSV or Excel file is selected.")
    else:
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            errors.append(f"Table is unavailable: {path}")
        elif path.suffix.casefold() not in TABLE_SUFFIXES:
            errors.append(f"Unsupported overflow table type: {path.suffix}")
        else:
            try:
                if path.suffix.casefold() in {".csv", ".tsv", ".txt"}:
                    row_count, columns, sheets = _csv_table_summary(path)
                else:
                    row_count, columns, sheets = _excel_table_summary(path)
                reference = _file_reference(path)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"Could not read {path.name}: {type(exc).__name__}: {exc}"
                )
    if not errors and row_count < 1:
        errors.append("The selected overflow table contains no data rows.")
    if not errors and not (_TIME_COLUMNS & set(columns)):
        warnings.append(
            "No standard time column was recognized; map it during synchronization."
        )
    if not errors and not (_WEIGHT_COLUMNS & set(columns)):
        warnings.append(
            "No standard weight column was recognized; map it during synchronization."
        )
    return {
        "kind": "table",
        "status": "error" if errors else "ready",
        "source_digest": overflow_run_source_digest(row),
        "scanned_utc": _utc_now(),
        "row_count": row_count,
        "columns": columns,
        "sheets": sheets,
        "file": reference,
        "errors": errors,
        "warnings": warnings,
    }


def scan_overflow_run(row: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(row.get("mode") or "none").strip().casefold()
    if mode == "none":
        return {
            "kind": "none",
            "status": "not_used",
            "source_digest": overflow_run_source_digest(row),
            "scanned_utc": _utc_now(),
            "errors": [],
            "warnings": [],
        }
    if mode == "scale_images":
        return _scan_scale_images(row)
    if mode == "table":
        return _scan_table(row)
    return {
        "kind": mode,
        "status": "error",
        "source_digest": overflow_run_source_digest(row),
        "scanned_utc": _utc_now(),
        "errors": [f"Unsupported overflow source mode: {mode}"],
        "warnings": [],
    }


def validate_overflow_config(
    config: Mapping[str, Any], *, expected_run_ids: Sequence[str] = ()
) -> dict[str, Any]:
    candidate = deepcopy(dict(config))
    issues: list[dict[str, Any]] = []
    if candidate.get("format") != OVERFLOW_CONFIG_FORMAT:
        issues.append(
            {
                "severity": "error",
                "code": "format",
                "message": "Unsupported overflow configuration.",
            }
        )
    rows = [
        dict(row) for row in candidate.get("runs") or [] if isinstance(row, Mapping)
    ]
    seen: set[str] = set()
    active = 0
    image_runs = 0
    table_runs = 0
    for row in rows:
        run_id = str(row.get("run_id") or "").strip().casefold()
        if not run_id or run_id in seen:
            issues.append(
                {
                    "severity": "error",
                    "code": "run_id",
                    "message": "Overflow run IDs must be non-empty and unique.",
                }
            )
            continue
        seen.add(run_id)
        mode = str(row.get("mode") or "none").strip().casefold()
        if mode not in OVERFLOW_SOURCE_MODES:
            issues.append(
                {
                    "severity": "error",
                    "code": "mode",
                    "run_id": run_id,
                    "message": f"Unsupported source mode for {run_id.upper()}.",
                }
            )
            continue
        if mode != "none":
            active += 1
            image_runs += int(mode == "scale_images")
            table_runs += int(mode == "table")
        scan = dict(row.get("scan") or {})
        if mode == "none" and not scan:
            continue
        if str(scan.get("source_digest") or "") != overflow_run_source_digest(row):
            issues.append(
                {
                    "severity": "error",
                    "code": "scan_outdated",
                    "run_id": run_id,
                    "message": f"Scan {run_id.upper()} after changing its overflow source.",
                }
            )
        elif str(scan.get("status") or "") not in {"ready", "not_used"}:
            issues.append(
                {
                    "severity": "error",
                    "code": "source_not_ready",
                    "run_id": run_id,
                    "message": f"The overflow source for {run_id.upper()} is not ready.",
                }
            )
        for warning in scan.get("warnings") or []:
            issues.append(
                {
                    "severity": "warning",
                    "code": "source_warning",
                    "run_id": run_id,
                    "message": str(warning),
                }
            )
    expected = {
        str(value).strip().casefold()
        for value in expected_run_ids
        if str(value).strip()
    }
    for run_id in sorted(expected - seen):
        issues.append(
            {
                "severity": "error",
                "code": "run_missing",
                "run_id": run_id,
                "message": f"No overflow decision exists for {run_id.upper()}.",
            }
        )
    return {
        "validated_utc": _utc_now(),
        "issues": issues,
        "summary": {
            "run_count": len(rows),
            "active_run_count": active,
            "scale_image_run_count": image_runs,
            "table_run_count": table_runs,
            "error_count": sum(row["severity"] == "error" for row in issues),
            "warning_count": sum(row["severity"] == "warning" for row in issues),
        },
    }


class OverflowRevisionStore:
    """Append-only overflow selections, scans, and source provenance."""

    def __init__(self, project: FluidFlowerProject) -> None:
        self.project = project
        self.override_root = project.root / "overrides" / "overflow"
        self.revisions_path = self.override_root / "revisions.jsonl"
        self.current_path = self.override_root / "current.json"
        self.draft_path = self.override_root / "working.json"
        self.artifact_root = project.root / "results" / "overflow" / "revisions"

    def revisions(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.revisions_path)

    def current(self) -> dict[str, Any]:
        if self.current_path.is_file():
            payload = json.loads(self.current_path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, dict):
                return payload
        revisions = self.revisions()
        if revisions:
            return deepcopy(revisions[-1])
        return default_overflow_config(self.project)

    def current_config(self) -> dict[str, Any]:
        if self.draft_path.is_file():
            payload = json.loads(self.draft_path.read_text(encoding="utf-8-sig"))
            if (
                isinstance(payload, dict)
                and payload.get("format") == OVERFLOW_DRAFT_FORMAT
            ):
                return deepcopy(dict(payload.get("configuration") or {}))
        current = self.current()
        if current.get("format") == OVERFLOW_REVISION_FORMAT:
            return deepcopy(dict(current.get("configuration") or {}))
        return deepcopy(current)

    def has_draft(self) -> bool:
        return self.draft_path.is_file()

    def save_draft(
        self,
        config: Mapping[str, Any],
        *,
        actor: str = "",
        reason: str = "Overflow working state updated.",
    ) -> dict[str, Any]:
        """Persist an incomplete working state without creating a revision."""

        payload = {
            "schema": 1,
            "format": OVERFLOW_DRAFT_FORMAT,
            "updated_utc": _utc_now(),
            "updated_by": str(actor).strip(),
            "reason": str(reason).strip(),
            "configuration": deepcopy(dict(config)),
        }
        _atomic_json(self.draft_path, payload)
        return deepcopy(payload)

    def discard_draft(self) -> None:
        self.draft_path.unlink(missing_ok=True)

    def save_revision(
        self,
        config: Mapping[str, Any],
        *,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        if not dependencies_satisfied("overflow", self.project.module_states):
            raise ValueError(
                "Data and integrity must be approved before an overflow revision "
                "can be saved."
            )
        candidate = deepcopy(dict(config))
        expected = [
            str(row.get("run_id") or "")
            for row in self.project.manifest.get("runs") or []
            if isinstance(row, Mapping)
        ]
        report = validate_overflow_config(candidate, expected_run_ids=expected)
        if int(report["summary"]["error_count"]) > 0:
            raise ValueError("Overflow configuration has blocking validation errors")
        previous = self.current()
        previous_rows = {
            str(row.get("run_id") or "").strip().casefold(): dict(row)
            for row in (previous.get("configuration") or {}).get("runs") or []
            if isinstance(row, Mapping) and str(row.get("run_id") or "").strip()
        }
        first_revision = previous.get("format") != OVERFLOW_REVISION_FORMAT
        changed_run_ids = [
            str(row.get("run_id") or "").strip().casefold()
            for row in candidate.get("runs") or []
            if isinstance(row, Mapping)
            and str(row.get("run_id") or "").strip()
            and (
                first_revision
                or previous_rows.get(
                    str(row.get("run_id") or "").strip().casefold()
                )
                != dict(row)
            )
        ]
        revision_id = uuid.uuid4().hex
        destination = self.artifact_root / revision_id
        destination.mkdir(parents=True, exist_ok=False)
        inventory_path = destination / "source_inventory.json"
        _atomic_json(
            inventory_path,
            {
                "schema": 1,
                "revision_id": revision_id,
                "runs": [
                    {
                        "run_id": row.get("run_id"),
                        "mode": row.get("mode"),
                        "scan": deepcopy(row.get("scan") or {}),
                    }
                    for row in candidate.get("runs") or []
                ],
            },
        )
        revision = {
            "schema": 1,
            "format": OVERFLOW_REVISION_FORMAT,
            "revision_id": revision_id,
            "recorded_utc": _utc_now(),
            "recorded_by": str(actor).strip(),
            "reason": str(reason).strip(),
            "configuration": candidate,
            "validation": report,
            "artifacts": {
                "source_inventory": _file_reference(inventory_path),
            },
        }
        _atomic_json(destination / "revision.json", revision)
        _append_jsonl(self.revisions_path, revision)
        _atomic_json(self.current_path, revision)
        self.discard_draft()
        active_rows = [
            dict(row)
            for row in candidate.get("runs") or []
            if str(row.get("mode") or "none") != "none"
        ]
        self.project.update_metadata(
            "overflow",
            {
                "mode": "per_run" if active_rows else "none",
                "revision_id": revision_id,
                "run_count": len(candidate.get("runs") or []),
                "active_run_count": len(active_rows),
                "source_modes_by_run": {
                    str(row.get("run_id") or ""): str(row.get("mode") or "none")
                    for row in candidate.get("runs") or []
                },
                "source_inventory_path": str(inventory_path),
            },
            updated_by=actor,
            reason=f"Overflow revision {revision_id}: {reason}".strip(),
        )
        current_status = str(
            self.project.module_states["overflow"].get("status") or "not_started"
        )
        next_status = (
            "review_required"
            if current_status in {"approved", "complete"}
            else "in_progress"
        )
        self.project.set_module_status(
            "overflow",
            next_status,
            updated_by=actor,
            reason="A validated overflow-source revision was saved.",
        )
        if changed_run_ids:
            self.project.set_run_module_status(
                changed_run_ids,
                "overflow",
                "review_required",
                updated_by=actor,
                reason=(
                    "A validated overflow-source revision was saved. This includes "
                    "experiments explicitly configured with no overflow measurements."
                ),
            )
        self.project._audit("overflow_revision_saved", actor, revision)
        return deepcopy(revision)


__all__ = [
    "IMAGE_SUFFIXES",
    "OVERFLOW_CONFIG_FORMAT",
    "OVERFLOW_DRAFT_FORMAT",
    "OVERFLOW_REVISION_FORMAT",
    "OVERFLOW_SOURCE_MODES",
    "OverflowRevisionStore",
    "apply_registered_overflow_mode",
    "default_overflow_config",
    "normalize_overflow_config",
    "overflow_run_source_digest",
    "scan_overflow_run",
    "validate_overflow_config",
]
