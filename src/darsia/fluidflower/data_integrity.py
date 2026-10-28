"""Versioned source and image integrity scans for FluidFlower projects."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable, Iterable, Mapping, TYPE_CHECKING
import uuid

from PIL import ExifTags, Image

from darsia.fluidflower.image_formats import (
    ALL_IMAGE_SUFFIXES,
    CAMERA_RAW_SUFFIXES,
    DEFAULT_IMAGE_FILE_POLICY,
    IMAGE_FILE_POLICY_ALL,
    IMAGE_FILE_POLICY_RAW,
    IMAGE_FILE_POLICY_RAW_ONLY,
    missing_image_family,
    normalize_image_file_policy,
    select_preferred_images,
)

if TYPE_CHECKING:
    from darsia.fluidflower.project import FluidFlowerProject


INTEGRITY_REPORT_FORMAT = "darsia-fluidflower-integrity-report-v1"
# Policies under which the scan reads camera RAW files (size, capture time).
RAW_READING_POLICIES = frozenset(
    {IMAGE_FILE_POLICY_RAW, IMAGE_FILE_POLICY_ALL, IMAGE_FILE_POLICY_RAW_ONLY}
)
INTEGRITY_WORKER_MEMORY_BUDGET_BYTES = 256 * 1024**2
IMAGE_SUFFIXES = set(ALL_IMAGE_SUFFIXES)
_SEQUENCE_PATTERN = re.compile(r"^(.*?)(\d+)$")
_IMAGE_COLLECTION_SOURCE_KINDS = {
    "alternate_color_reference_images",
    "alternate_color_reference_image_folder",
    "baseline_images",
    "clearwater_images",
    "clearwater_image_folder",
    "co2_image_folder",
    "colorchecker_images",
    "experiment_images",
    "indicator_reference_images",
    "indicator_reference_image_folder",
    "overflow_image_folder",
}
_CAMERA_SEQUENCE_SOURCE_KINDS = {"co2_image_folder", "experiment_images"}


class ScanCancelled(RuntimeError):
    """Raised when an integrity scan is cancelled by the caller."""


@dataclass(frozen=True)
class IntegrityScanOptions:
    mode: str = "fast"
    recursive: bool = True
    verify_images: bool = False
    read_exif: bool = True
    retry_count: int = 3
    retry_delay_s: float = 0.25
    quick_hash_bytes: int = 65536
    missing_sequence_max_gap: int = 1000
    parallel_reads: int = 8

    def validate(self) -> None:
        if self.mode not in {"fast", "verified"}:
            raise ValueError("Integrity mode must be 'fast' or 'verified'")
        if self.retry_count < 1:
            raise ValueError("Integrity retry count must be positive")
        if self.retry_delay_s < 0.0 or not math.isfinite(self.retry_delay_s):
            raise ValueError("Integrity retry delay must be finite and non-negative")
        if self.quick_hash_bytes < 1024:
            raise ValueError("Quick-hash sample must be at least 1024 bytes")
        if self.parallel_reads < 1:
            raise ValueError("Parallel reads must be positive")


def recommend_integrity_read_workers(
    host_resources: Mapping[str, Any],
) -> dict[str, int | None]:
    """Return hardware-derived start and safety limits for parallel file reads."""

    cpu = host_resources.get("cpu") or {}
    memory = host_resources.get("memory") or {}
    logical = max(1, int(cpu.get("logical_cores") or 1))
    raw_physical = cpu.get("physical_cores")
    physical = (
        max(1, int(raw_physical))
        if isinstance(raw_physical, (int, float)) and raw_physical
        else max(1, (logical + 1) // 2)
    )
    available_memory = memory.get("available_bytes")

    # File reads are I/O-bound and can benefit from more threads than CPU cores.
    # Keep half of currently available RAM outside the scan and budget generously
    # for image decoder buffers, network retries, and the rest of the application.
    cpu_capacity = max(1, logical * 2)
    memory_capacity = cpu_capacity
    if isinstance(available_memory, (int, float)) and available_memory > 0:
        memory_capacity = max(
            1,
            int(float(available_memory) * 0.5)
            // INTEGRITY_WORKER_MEMORY_BUDGET_BYTES,
        )
    maximum = max(1, min(cpu_capacity, memory_capacity))
    conservative_start = max(1, physical // 2)
    if logical >= 4:
        conservative_start = max(2, conservative_start)
    recommended = max(1, min(maximum, conservative_start))
    return {
        "recommended": recommended,
        "maximum": maximum,
        "logical_cores": logical,
        "physical_cores": physical,
        "available_memory_bytes": (
            int(available_memory)
            if isinstance(available_memory, (int, float)) and available_memory >= 0
            else None
        ),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _retry(
    action: Callable[[], Any],
    *,
    attempts: int,
    delay_s: float,
) -> tuple[Any, int]:
    last_error: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return action(), attempt
        except OSError as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(delay_s * attempt)
    assert last_error is not None
    raise last_error


def _quick_hash(path: Path, sample_bytes: int) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as stream:
        digest.update(stream.read(sample_bytes))
        if size > sample_bytes:
            stream.seek(max(0, size - sample_bytes))
            digest.update(stream.read(sample_bytes))
    return digest.hexdigest()


def _full_hash(path: Path) -> str:
    from .project import _sha256_file_uncached
    return _sha256_file_uncached(path)


def _parse_exif_datetime(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    for format_string in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, format_string).isoformat()
        except ValueError:
            continue
    return ""


def _raw_image_metadata(path: Path, *, read_exif: bool) -> dict[str, Any]:
    """Size and capture time of a camera RAW file (LibRaw and its EXIF)."""

    from darsia.fluidflower.photograph_io import raw_capture_time, raw_photo_size

    width, height = raw_photo_size(path)
    taken = raw_capture_time(path) if read_exif else None
    return {
        "width": int(width),
        "height": int(height),
        "mode": "RGB",
        "image_format": "RAW",
        "exif_datetime": taken.isoformat() if taken is not None else "",
    }


def _image_metadata(path: Path, *, verify: bool, read_exif: bool) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
        image_format = str(image.format or "")
        mode = str(image.mode or "")
        timestamp = ""
        if read_exif:
            exif = image.getexif()
            tagged = {
                ExifTags.TAGS.get(tag_id, str(tag_id)): value
                for tag_id, value in exif.items()
            }
            for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                timestamp = _parse_exif_datetime(tagged.get(key))
                if timestamp:
                    break
        if verify:
            image.verify()
    return {
        "width": int(width),
        "height": int(height),
        "mode": mode,
        "image_format": image_format,
        "exif_datetime": timestamp,
    }


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(path))
    )


class IntegrityScanner:
    """Read-only scanner for registered project sources."""

    def __init__(
        self,
        project: "FluidFlowerProject",
        options: IntegrityScanOptions | None = None,
    ) -> None:
        self.project = project
        self.options = options or IntegrityScanOptions()
        self.options.validate()

    def _enumerate_source(
        self, source: Mapping[str, Any]
    ) -> tuple[list[Path], int, list[tuple[Path, str]]]:
        """Files of one source, read attempts, and folders a strict policy rejects."""

        root = Path(str(source.get("path") or "")).expanduser()
        if root.is_file():
            return [root], 1, []
        if not root.is_dir():
            raise FileNotFoundError(root)
        missing: list[tuple[Path, str]] = []

        def collect() -> list[Path]:
            missing.clear()
            files: list[Path] = []
            if self.options.recursive:
                walk_errors: list[OSError] = []
                for folder, _directories, names in os.walk(
                    root,
                    onerror=walk_errors.append,
                ):
                    files.extend(Path(folder) / name for name in names)
                if walk_errors:
                    raise walk_errors[0]
            else:
                files.extend(path for path in root.iterdir() if path.is_file())
            if str(source.get("kind") or "") in _IMAGE_COLLECTION_SOURCE_KINDS:
                policy = normalize_image_file_policy(
                    source.get("image_file_policy") or DEFAULT_IMAGE_FILE_POLICY
                )
                images_by_folder: dict[Path, list[Path]] = defaultdict(list)
                for path in files:
                    if path.suffix.casefold() in ALL_IMAGE_SUFFIXES:
                        images_by_folder[path.parent].append(path)
                selected_images = {
                    path
                    for paths in images_by_folder.values()
                    for path in select_preferred_images(paths, policy)
                }
                for folder, paths in sorted(images_by_folder.items()):
                    reason = missing_image_family(paths, policy)
                    if reason:
                        missing.append((folder, reason))
                files = [
                    path
                    for path in files
                    if path.suffix.casefold() not in ALL_IMAGE_SUFFIXES
                    or path in selected_images
                ]
            return sorted(files, key=_natural_key)

        paths, attempts = _retry(
            collect,
            attempts=self.options.retry_count,
            delay_s=self.options.retry_delay_s,
        )
        return paths, attempts, list(missing)

    def _inspect_entry(
        self,
        source: Mapping[str, Any],
        path: Path,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        source_root = Path(str(source.get("path") or "")).expanduser()
        relative = path.name if source_root.is_file() else str(path.relative_to(source_root))
        # RAW files are read when the source's policy selects them.
        read_raw = normalize_image_file_policy(
            source.get("image_file_policy") or DEFAULT_IMAGE_FILE_POLICY
        ) in RAW_READING_POLICIES
        row: dict[str, Any] = {
            "source_id": str(source.get("source_id") or ""),
            "source_kind": str(source.get("kind") or ""),
            "run_id": str(source.get("run_id") or ""),
            "path": str(path),
            "relative_path": relative,
            "suffix": path.suffix.casefold(),
            "is_image": path.suffix.casefold() in IMAGE_SUFFIXES,
            "is_camera_raw": path.suffix.casefold() in CAMERA_RAW_SUFFIXES,
            "size_bytes": None,
            "mtime_utc": "",
            "quick_sha256": "",
            "sha256": "",
            "width": None,
            "height": None,
            "mode": "",
            "image_format": "",
            "exif_datetime": "",
            "read_attempts": 0,
            "read_error": "",
        }

        def inspect_file() -> dict[str, Any]:
            stat = path.stat()
            inspected: dict[str, Any] = {
                "size_bytes": int(stat.st_size),
                "mtime_utc": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat(),
            }
            if row["is_camera_raw"] and read_raw:
                inspected.update(
                    _raw_image_metadata(path, read_exif=self.options.read_exif)
                )
            elif row["is_image"] and not row["is_camera_raw"]:
                inspected.update(
                    _image_metadata(
                        path,
                        verify=(
                            self.options.verify_images
                            or self.options.mode == "verified"
                        ),
                        read_exif=self.options.read_exif,
                    )
                )
            if self.options.mode == "verified":
                inspected["sha256"] = _full_hash(path)
            return inspected

        try:
            inspected, attempts = _retry(
                inspect_file,
                attempts=self.options.retry_count,
                delay_s=self.options.retry_delay_s,
            )
            row.update(inspected)
            row["read_attempts"] = attempts
            return row, None
        except (OSError, ValueError, SyntaxError) as exc:
            row["read_attempts"] = self.options.retry_count
            row["read_error"] = f"{type(exc).__name__}: {exc}"
            return row, {
                "severity": "error",
                "code": "file_unreadable",
                "source_id": row["source_id"],
                "path": str(path),
                "message": row["read_error"],
            }

    def scan(
        self,
        *,
        source_ids: Iterable[str] | None = None,
        actor: str = "",
        progress: Callable[[int, int, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        max_parallel_reads: int | None = None,
        worker_limit: Callable[[], int] | None = None,
        worker_status: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        selected = set(str(value) for value in source_ids or [])
        sources = [
            dict(source)
            for source in self.project.current_source_records()
            if str(source.get("scan_policy") or "content") != "reference_only"
            and (not selected or str(source.get("source_id") or "") in selected)
        ]
        if not sources:
            raise ValueError("No registered sources were selected for the integrity scan")
        started = _utc_now()
        scan_id = uuid.uuid4().hex
        issues: list[dict[str, Any]] = []
        enumerated: list[tuple[dict[str, Any], Path]] = []
        source_retries: dict[str, int] = {}
        for source in sources:
            source_id = str(source.get("source_id") or "")
            try:
                paths, attempts, missing_family = self._enumerate_source(source)
            except OSError as exc:
                issues.append(
                    {
                        "severity": "error",
                        "code": "source_unavailable",
                        "source_id": source_id,
                        "path": str(source.get("path") or ""),
                        "message": f"Source remained unavailable after retries: {exc}",
                    }
                )
                continue
            source_retries[source_id] = attempts
            for folder, reason in missing_family:
                issues.append(
                    {
                        "severity": "error",
                        "code": "missing_image_family",
                        "source_id": source_id,
                        "path": str(folder),
                        "message": f"{folder}: {reason}.",
                    }
                )
            enumerated.extend((source, path) for path in paths)

        inventory_slots: list[dict[str, Any] | None] = [None] * len(enumerated)
        total = len(enumerated)
        initial_workers = max(1, int(self.options.parallel_reads))
        maximum_workers = max(
            initial_workers,
            int(max_parallel_reads or initial_workers),
        )
        worker_history: list[dict[str, int]] = []
        last_requested_workers: int | None = None

        def requested_workers(completed_files: int) -> int:
            nonlocal last_requested_workers
            raw_value = worker_limit() if worker_limit is not None else initial_workers
            value = max(1, min(int(raw_value), maximum_workers))
            if value != last_requested_workers:
                worker_history.append(
                    {
                        "completed_files": int(completed_files),
                        "requested_workers": value,
                    }
                )
                last_requested_workers = value
            return value

        futures: dict[Future[tuple[dict[str, Any], dict[str, Any] | None]], int] = {}
        completed = 0
        next_index = 0
        executor: ThreadPoolExecutor | None = None
        try:
            if cancelled is not None and cancelled():
                raise ScanCancelled("Integrity scan cancelled")
            if total:
                executor = ThreadPoolExecutor(
                    max_workers=maximum_workers,
                    thread_name_prefix="integrity",
                )

            def fill_queue() -> None:
                nonlocal next_index
                desired = requested_workers(completed)
                while (
                    executor is not None
                    and next_index < total
                    and len(futures) < desired
                ):
                    source, path = enumerated[next_index]
                    future = executor.submit(self._inspect_entry, source, path)
                    futures[future] = next_index
                    next_index += 1
                if worker_status is not None:
                    worker_status(len(futures), desired)

            fill_queue()
            while futures:
                if cancelled is not None and cancelled():
                    for pending in futures:
                        pending.cancel()
                    raise ScanCancelled("Integrity scan cancelled")
                finished, _pending = wait(
                    tuple(futures),
                    return_when=FIRST_COMPLETED,
                )
                for future in finished:
                    index = futures.pop(future)
                    row, issue = future.result()
                    inventory_slots[index] = row
                    if issue is not None:
                        issues.append(issue)
                    completed += 1
                    if progress is not None:
                        progress(completed, total, str(row["path"]))
                fill_queue()
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        if worker_status is not None:
            worker_status(0, requested_workers(completed))
        inventory = [row for row in inventory_slots if row is not None]
        if progress is not None:
            progress(total, total, "Finalizing integrity checks")

        camera_sequence_inventory = [
            row
            for row in inventory
            if str(row.get("source_kind") or "") in _CAMERA_SEQUENCE_SOURCE_KINDS
        ]
        self._complete_duplicate_hashes(
            camera_sequence_inventory, issues, cancelled
        )
        self._resolution_issues(inventory, issues)
        self._sequence_issues(camera_sequence_inventory, issues)
        self._timestamp_issues(camera_sequence_inventory, issues)
        source_summaries = self._source_summaries(sources, inventory, issues, source_retries)
        severity_counts = Counter(str(issue["severity"]) for issue in issues)
        image_count = sum(bool(row["is_image"]) for row in inventory)
        report = {
            "schema": 1,
            "format": INTEGRITY_REPORT_FORMAT,
            "scan_id": scan_id,
            "project_id": self.project.project_id,
            "started_utc": started,
            "completed_utc": _utc_now(),
            "created_by": str(actor).strip(),
            "options": asdict(self.options),
            "worker_scheduling": {
                "initial_worker_count": initial_workers,
                "maximum_worker_count": maximum_workers,
                "dynamic_worker_limit": worker_limit is not None,
                "history": worker_history,
            },
            "source_manifest_sha256": hashlib.sha256(
                self.project.source_manifest_path.read_bytes()
            ).hexdigest(),
            "selected_source_ids": [str(source.get("source_id") or "") for source in sources],
            "summary": {
                "source_count": len(sources),
                "available_source_count": sum(
                    summary["available"] for summary in source_summaries
                ),
                "file_count": len(inventory),
                "image_count": image_count,
                "error_count": int(severity_counts["error"]),
                "warning_count": int(severity_counts["warning"]),
                "info_count": int(severity_counts["info"]),
                "read_retry_count": sum(
                    max(0, int(row.get("read_attempts") or 0) - 1)
                    for row in inventory
                ),
            },
            "sources": source_summaries,
            "issues": issues,
            "inventory": inventory,
        }
        return report

    def _complete_duplicate_hashes(
        self,
        inventory: list[dict[str, Any]],
        issues: list[dict[str, Any]],
        cancelled: Callable[[], bool] | None,
    ) -> None:
        if self.options.mode == "fast":
            size_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in inventory:
                if row.get("read_error") or row.get("size_bytes") is None:
                    continue
                size_groups[int(row["size_bytes"])].append(row)
            for group in size_groups.values():
                if len(group) < 2:
                    continue
                for row in group:
                    if cancelled is not None and cancelled():
                        raise ScanCancelled("Integrity scan cancelled")
                    path = Path(str(row["path"]))
                    try:
                        digest, attempts = _retry(
                            lambda path=path: _quick_hash(
                                path, self.options.quick_hash_bytes
                            ),
                            attempts=self.options.retry_count,
                            delay_s=self.options.retry_delay_s,
                        )
                        row["quick_sha256"] = digest
                        row["read_attempts"] = max(int(row["read_attempts"]), attempts)
                    except OSError as exc:
                        issues.append(
                            {
                                "severity": "warning",
                                "code": "duplicate_sampling_failed",
                                "source_id": row["source_id"],
                                "path": row["path"],
                                "message": f"Could not sample possible duplicate: {exc}",
                            }
                        )
        quick_groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in inventory:
            if row.get("read_error") or not row.get("quick_sha256"):
                continue
            quick_groups[(int(row["size_bytes"]), str(row["quick_sha256"]))].append(row)
        candidates = [group for group in quick_groups.values() if len(group) > 1]
        for group in candidates:
            for row in group:
                if cancelled is not None and cancelled():
                    raise ScanCancelled("Integrity scan cancelled")
                if row.get("sha256"):
                    continue
                path = Path(str(row["path"]))
                try:
                    digest, attempts = _retry(
                        lambda path=path: _full_hash(path),
                        attempts=self.options.retry_count,
                        delay_s=self.options.retry_delay_s,
                    )
                    row["sha256"] = digest
                    row["read_attempts"] = max(int(row["read_attempts"]), attempts)
                except OSError as exc:
                    issues.append(
                        {
                            "severity": "warning",
                            "code": "duplicate_verification_failed",
                            "source_id": row["source_id"],
                            "path": row["path"],
                            "message": f"Could not verify likely duplicate: {exc}",
                        }
                    )
        full_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in inventory:
            if row.get("sha256"):
                full_groups[str(row["sha256"])].append(row)
        for digest, group in full_groups.items():
            if len(group) < 2:
                continue
            issues.append(
                {
                    "severity": "warning",
                    "code": "exact_duplicate_files",
                    "source_id": str(group[0]["source_id"]),
                    "path": str(group[0]["path"]),
                    "message": f"{len(group)} files have identical content.",
                    "details": {
                        "sha256": digest,
                        "paths": [str(row["path"]) for row in group],
                    },
                }
            )

    @staticmethod
    def _resolution_issues(
        inventory: list[dict[str, Any]], issues: list[dict[str, Any]]
    ) -> None:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in inventory:
            if row.get("is_image") and row.get("width") and row.get("height"):
                groups[(str(row["source_id"]), str(row.get("run_id") or ""))].append(row)
        for (source_id, run_id), rows in groups.items():
            counts = Counter((int(row["width"]), int(row["height"])) for row in rows)
            dominant, dominant_count = counts.most_common(1)[0]
            outliers = [
                row
                for row in rows
                if (int(row["width"]), int(row["height"])) != dominant
            ]
            if not outliers:
                continue
            issues.append(
                {
                    "severity": "warning",
                    "code": "resolution_outliers",
                    "source_id": source_id,
                    "path": str(outliers[0]["path"]),
                    "message": (
                        f"{len(outliers)} of {len(rows)} images differ from the dominant "
                        f"resolution {dominant[0]}×{dominant[1]}."
                    ),
                    "details": {
                        "run_id": run_id,
                        "dominant_resolution": list(dominant),
                        "dominant_count": dominant_count,
                        "resolutions": {
                            f"{width}x{height}": count
                            for (width, height), count in counts.items()
                        },
                        "sample_paths": [str(row["path"]) for row in outliers[:20]],
                    },
                }
            )

    def _sequence_issues(
        self, inventory: list[dict[str, Any]], issues: list[dict[str, Any]]
    ) -> None:
        groups: dict[tuple[str, str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for row in inventory:
            if not row.get("is_image"):
                continue
            path = Path(str(row["path"]))
            match = _SEQUENCE_PATTERN.match(path.stem)
            if match:
                groups[(str(row["source_id"]), str(path.parent), match.group(1))].append(
                    (int(match.group(2)), row)
                )
        for (source_id, folder, prefix), numbered in groups.items():
            ids = sorted(number for number, _row in numbered)
            duplicate_ids = sorted(number for number, count in Counter(ids).items() if count > 1)
            if duplicate_ids:
                issues.append(
                    {
                        "severity": "warning",
                        "code": "duplicate_sequence_numbers",
                        "source_id": source_id,
                        "path": folder,
                        "message": f"Duplicate image numbers: {duplicate_ids[:20]}",
                        "details": {"prefix": prefix, "numbers": duplicate_ids},
                    }
                )
            unique_ids = sorted(set(ids))
            missing_ranges: list[list[int]] = []
            discontinuities: list[list[int]] = []
            for left, right in zip(unique_ids, unique_ids[1:], strict=False):
                gap = right - left - 1
                if gap <= 0:
                    continue
                target = missing_ranges if gap <= self.options.missing_sequence_max_gap else discontinuities
                target.append([left + 1, right - 1])
            if missing_ranges:
                missing_count = sum(end - start + 1 for start, end in missing_ranges)
                issues.append(
                    {
                        "severity": "warning",
                        "code": "missing_sequence_numbers",
                        "source_id": source_id,
                        "path": folder,
                        "message": f"{missing_count} image numbers are missing within the sequence.",
                        "details": {"prefix": prefix, "ranges": missing_ranges[:100]},
                    }
                )
            if discontinuities:
                issues.append(
                    {
                        "severity": "info",
                        "code": "sequence_number_discontinuity",
                        "source_id": source_id,
                        "path": folder,
                        "message": "Large numbering discontinuity detected; this may be a camera-card reset.",
                        "details": {"prefix": prefix, "ranges": discontinuities[:100]},
                    }
                )

    @staticmethod
    def _timestamp_issues(
        inventory: list[dict[str, Any]], issues: list[dict[str, Any]]
    ) -> None:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in inventory:
            if row.get("is_image") and not row.get("read_error"):
                groups[str(row["source_id"])].append(row)
        for source_id, rows in groups.items():
            missing = [row for row in rows if not row.get("exif_datetime")]
            if missing:
                issues.append(
                    {
                        "severity": "warning",
                        "code": "missing_image_timestamps",
                        "source_id": source_id,
                        "path": str(missing[0]["path"]),
                        "message": f"{len(missing)} of {len(rows)} images have no readable EXIF timestamp.",
                    }
                )
            timestamp_rows = [row for row in rows if row.get("exif_datetime")]
            timestamps = [str(row["exif_datetime"]) for row in timestamp_rows]
            duplicate_count = sum(count - 1 for count in Counter(timestamps).values() if count > 1)
            if duplicate_count:
                issues.append(
                    {
                        "severity": "warning",
                        "code": "duplicate_image_timestamps",
                        "source_id": source_id,
                        "path": str(timestamp_rows[0]["path"]),
                        "message": f"{duplicate_count} image timestamps are duplicated.",
                    }
                )

    @staticmethod
    def _source_summaries(
        sources: list[dict[str, Any]],
        inventory: list[dict[str, Any]],
        issues: list[dict[str, Any]],
        source_retries: Mapping[str, int],
    ) -> list[dict[str, Any]]:
        files_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in inventory:
            files_by_source[str(row["source_id"])].append(row)
        issues_by_source = Counter(str(issue.get("source_id") or "") for issue in issues)
        summaries: list[dict[str, Any]] = []
        for source in sources:
            source_id = str(source.get("source_id") or "")
            rows = files_by_source[source_id]
            path = Path(str(source.get("path") or "")).expanduser()
            summaries.append(
                {
                    "source_id": source_id,
                    "kind": str(source.get("kind") or ""),
                    "run_id": str(source.get("run_id") or ""),
                    "path": str(path),
                    "available": path.exists(),
                    "enumeration_attempts": int(source_retries.get(source_id, 0)),
                    "file_count": len(rows),
                    "image_count": sum(bool(row["is_image"]) for row in rows),
                    "total_size_bytes": sum(int(row.get("size_bytes") or 0) for row in rows),
                    "issue_count": int(issues_by_source[source_id]),
                }
            )
        return summaries


class IntegrityReportStore:
    """Persist immutable scan reports and a small pointer to the current report."""

    def __init__(self, project: "FluidFlowerProject") -> None:
        self.project = project
        self.root = project.root / "results" / "integrity"
        self.current_path = self.root / "current.json"

    def save(self, report: Mapping[str, Any], *, actor: str = "") -> Path:
        payload = deepcopy(dict(report))
        if payload.get("format") != INTEGRITY_REPORT_FORMAT:
            raise ValueError("Unsupported integrity report format")
        scan_id = str(payload.get("scan_id") or "")
        if not scan_id:
            raise ValueError("Integrity report has no scan ID")
        inventory = list(payload.pop("inventory", []))
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / scan_id
        if destination.exists():
            raise FileExistsError(f"Integrity report already exists: {destination}")
        staging = self.root / f".{scan_id}.{uuid.uuid4().hex}.tmp"
        staging.mkdir(parents=True)
        try:
            inventory_name = "inventory.jsonl.gz"
            with gzip.open(staging / inventory_name, "wt", encoding="utf-8", newline="\n") as stream:
                for row in inventory:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            payload["inventory_file"] = inventory_name
            payload["inventory_record_count"] = len(inventory)
            (staging / "report.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        current = {
            "schema": 1,
            "format": INTEGRITY_REPORT_FORMAT,
            "scan_id": scan_id,
            "report_path": str(destination / "report.json"),
            "completed_utc": payload.get("completed_utc"),
            "summary": payload.get("summary"),
        }
        temporary = self.current_path.with_name(f".{self.current_path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(current, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.current_path)
        self.project._audit(
            "integrity_report_saved",
            actor,
            {
                "scan_id": scan_id,
                "summary": payload.get("summary"),
                "report_path": str(destination / "report.json"),
            },
        )
        return destination

    def current(self) -> dict[str, Any] | None:
        if not self.current_path.is_file():
            return None
        pointer = json.loads(self.current_path.read_text(encoding="utf-8-sig"))
        report_path = Path(str(pointer.get("report_path") or ""))
        if not report_path.is_file():
            return None
        payload = json.loads(report_path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else None

    def reports(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        reports: list[dict[str, Any]] = []
        for path in self.root.glob("*/report.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                continue
            if isinstance(payload, dict) and payload.get("format") == INTEGRITY_REPORT_FORMAT:
                payload["report_path"] = str(path)
                reports.append(payload)
        return sorted(
            reports,
            key=lambda row: str(row.get("completed_utc") or ""),
            reverse=True,
        )

    def inventory(self, report: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        selected = dict(report or self.current() or {})
        scan_id = str(selected.get("scan_id") or "")
        inventory_name = str(selected.get("inventory_file") or "inventory.jsonl.gz")
        path = self.root / scan_id / inventory_name
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with gzip.open(path, "rt", encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def camera_sequence_endpoints(
        self, report: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Return compact first/last-image records without loading the inventory.

        Integrity reports can contain hundreds of thousands of rows. Temporal
        synchronization only needs each camera sequence's endpoints, so the
        compressed inventory is streamed and only those records are retained.
        """

        selected = dict(report or self.current() or {})
        scan_id = str(selected.get("scan_id") or "")
        inventory_name = str(selected.get("inventory_file") or "inventory.jsonl.gz")
        path = self.root / scan_id / inventory_name
        if not path.is_file():
            return []

        source_rows = {
            str(row.get("source_id") or ""): dict(row)
            for row in selected.get("sources") or []
            if isinstance(row, Mapping)
            and str(row.get("kind") or "") in _CAMERA_SEQUENCE_SOURCE_KINDS
        }
        endpoints: dict[str, dict[str, Any]] = {}
        with gzip.open(path, "rt", encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping) or not row.get("is_image"):
                    continue
                source_id = str(row.get("source_id") or "")
                if source_id not in source_rows:
                    continue
                natural_key = _natural_key(
                    Path(str(row.get("relative_path") or row.get("path") or ""))
                )
                current = endpoints.setdefault(
                    source_id,
                    {
                        "source_id": source_id,
                        "source_kind": str(row.get("source_kind") or ""),
                        "run_id": str(row.get("run_id") or "").casefold(),
                        "source_path": str(source_rows[source_id].get("path") or ""),
                        "image_count": 0,
                        "first": None,
                        "last": None,
                        "first_key": None,
                        "last_key": None,
                        "first_timestamped": None,
                        "last_timestamped": None,
                        "first_timestamp_key": None,
                        "last_timestamp_key": None,
                        "timestamped_image_count": 0,
                    },
                )
                current["image_count"] = int(current["image_count"]) + 1
                compact = {
                    "path": str(row.get("path") or ""),
                    "relative_path": str(row.get("relative_path") or ""),
                    "exif_datetime": str(row.get("exif_datetime") or ""),
                    "mtime_utc": str(row.get("mtime_utc") or ""),
                }
                if current["first_key"] is None or natural_key < current["first_key"]:
                    current["first"] = compact
                    current["first_key"] = natural_key
                if current["last_key"] is None or natural_key > current["last_key"]:
                    current["last"] = compact
                    current["last_key"] = natural_key
                timestamp = compact["exif_datetime"]
                if timestamp:
                    current["timestamped_image_count"] = (
                        int(current["timestamped_image_count"]) + 1
                    )
                    if (
                        current["first_timestamp_key"] is None
                        or timestamp < current["first_timestamp_key"]
                    ):
                        current["first_timestamped"] = compact
                        current["first_timestamp_key"] = timestamp
                    if (
                        current["last_timestamp_key"] is None
                        or timestamp > current["last_timestamp_key"]
                    ):
                        current["last_timestamped"] = compact
                        current["last_timestamp_key"] = timestamp

        output: list[dict[str, Any]] = []
        for row in endpoints.values():
            if (
                int(row["timestamped_image_count"]) == int(row["image_count"])
                and row["first_timestamped"] is not None
                and row["last_timestamped"] is not None
            ):
                row["first"] = row["first_timestamped"]
                row["last"] = row["last_timestamped"]
                row["chronology"] = "exif_datetime"
            else:
                row["chronology"] = "natural_filename_order"
            row.pop("first_key", None)
            row.pop("last_key", None)
            row.pop("first_timestamped", None)
            row.pop("last_timestamped", None)
            row.pop("first_timestamp_key", None)
            row.pop("last_timestamp_key", None)
            output.append(row)
        return sorted(
            output,
            key=lambda row: (
                str(row.get("run_id") or ""),
                str(row.get("source_path") or ""),
            ),
        )
