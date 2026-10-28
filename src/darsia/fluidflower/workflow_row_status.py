"""Passive native-sidebar status from bounded, explicit saved metadata.

``done`` means the named saved artefact is present, never a new scientific gate
or whole-run completion claim. No image/NPZ content, fitting, history discovery,
project creation, output write, or recursive NAS glob occurs here.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import stat as stat_module
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
import tomllib
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

MAX_METADATA_BYTES = 1024 * 1024
# A built run model records per-pixel analysis metadata next to its baseline
# identity, so its meta_data.json can exceed the general bound.
MAX_RUN_MODEL_METADATA_BYTES = 16 * 1024 * 1024
MAX_LIST_ENTRIES = 128
MAX_OUTPUT_DIRS = 8
GEOMETRY_FORMAT = "darsia-fluidflower-geometry-facies-v1"
EXPORT_FORMAT = "darsia-fluidflower-native-setup-export-v1"
DELIVERY_FORMAT = "studio-mass-variant-delivery-v1"


def _map(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _same(first: Path | None, second: Path | None) -> bool:
    return bool(
        first is not None
        and second is not None
        and os.path.normcase(os.path.abspath(first))
        == os.path.normcase(os.path.abspath(second))
    )


def _path(value: Any, root: Path | None) -> Path | None:
    if isinstance(value, Mapping):
        value = value.get("path")
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        return None
    path = Path(value)
    from .storage_paths import resolve_storage_path
    joined = path if path.is_absolute() or root is None else root / path
    return resolve_storage_path(joined, series_roots=(root,) if root is not None else ())


def _reference(selection: Any, name: str) -> str:
    value = (
        selection.get(name, "")
        if isinstance(selection, Mapping)
        else getattr(selection, name, "")
    )
    return str(value or "")


def oslo_date(value: Any) -> str:
    """Display a stored date in Oslo; never guess a missing producer date."""
    try:
        if isinstance(value, dt.datetime):
            date = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            date = dt.datetime.fromtimestamp(value, dt.timezone.utc)
        elif isinstance(value, str) and value:
            date = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            return ""
        if date.tzinfo is None:
            return ""  # Unzoned producer time cannot be safely converted.
        return date.astimezone(ZoneInfo("Europe/Oslo")).strftime("%Y-%m-%d %H:%M %Z")
    except (ValueError, OverflowError, OSError, ZoneInfoNotFoundError):
        return ""


@dataclass(frozen=True)
class Artifact:
    path: str
    kind: str
    size_bytes: int
    modified: str = ""
    # What a step uses from the file, when that identifies it instead of its
    # size and time (the injection schedule of an injection protocol).
    content: str = ""


# Identifies an injection protocol by its schedule (injection_schedule).
SCHEDULE_FORMAT = "injection-schedule-v1"


def injection_schedule(path: Path) -> str:
    """The schedule of every worksheet of an injection protocol, without the positions.

    A shared workbook holds one worksheet per run, and a CSV file one run's
    schedule, so the answer is the same whichever run is loaded. Each
    worksheet is read by protocol_content.injection_schedule_digest (ids,
    start and end times and mass rates). "" when the file or one of its
    worksheets cannot be read that way; the file then keeps its size and time.
    """
    import hashlib

    from .protocol_content import injection_schedule_digest

    try:
        if Path(path).suffix.casefold() in {".xlsx", ".xlsm", ".xls"}:
            import pandas as pd

            with pd.ExcelFile(path) as book:
                sheets = list(book.sheet_names)
            rows = [[str(sheet), injection_schedule_digest(path, sheet)] for sheet in sheets]
        else:
            rows = [[None, injection_schedule_digest(path, None)]]
    except Exception:  # noqa: BLE001 - any unreadable file or worksheet keeps the byte identity
        return ""
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{SCHEDULE_FORMAT}:{digest}"


@dataclass(frozen=True)
class RowStatus:
    state: str  # done | missing | not_applicable | saved_choice | default_choice | choice_error
    summary: str
    date: str = ""
    artifacts: tuple[Artifact, ...] = ()
    missing: tuple[str, ...] = ()
    producer: str = ""
    next_step: str = ""
    date_source: str = ""
    selected_run_text: str = ""
    counts: Mapping = field(default_factory=dict)


@dataclass
class WorkflowStatusContext:
    series_root: Path | None
    run_id: str = ""
    config_path: Path | None = None
    config: Mapping = field(default_factory=dict)
    scope_context: Any = None
    method_selection: Any = None
    manifest: Mapping = field(default_factory=dict)
    run_bound: bool = False
    issues: tuple[str, ...] = ()
    # Only explicit selections supplied by the existing UI; never latest/history.
    selection_plan_reference: str = ""
    review_reference: str = ""
    result_receipt_reference: str = ""
    _metadata_cache: dict[str, Mapping] = field(default_factory=dict, repr=False)

    def document(self, path: Path, *, limit: int | None = None) -> Mapping:
        """Read one bounded JSON/TOML object once; ``limit`` defaults to MAX_METADATA_BYTES."""
        limit = MAX_METADATA_BYTES if limit is None else limit
        key = os.path.normcase(os.path.abspath(path))
        if key not in self._metadata_cache:
            with path.open("rb") as stream:
                raw = stream.read(limit + 1)
            if len(raw) > limit:
                raise ValueError(f"Metadata exceeds {limit} bytes: {path}")
            if path.suffix.casefold() == ".toml":
                value = tomllib.loads(raw.decode("utf-8-sig"))
            else:
                value = json.loads(raw.decode("utf-8-sig"))
            if not isinstance(value, Mapping):
                raise ValueError(f"Saved metadata is not an object: {path}")
            self._metadata_cache[key] = value
        return self._metadata_cache[key]


def load_status_context(
    series_root: str | Path | None,
    run_id: str = "",
    config_path: str | Path | None = None,
    loaded_config: Mapping | None = None,
    *,
    scope_context: Any = None,
    method_selection: Any = None,
    selection_plan_reference: str = "",
    review_reference: str = "",
    result_receipt_reference: str = "",
) -> WorkflowStatusContext:
    """Read one manifest and, if needed, the already selected single TOML.

    A foreign File > Open config cannot borrow the currently selected series/run
    identity. No merged configuration is generated and no run is automatically
    selected by this status function.
    """
    if scope_context is not None:
        series_root = (
            series_root
            if series_root is not None
            else getattr(scope_context, "series_root", None)
        )
        run_id = run_id or getattr(scope_context, "selected_run", "")
        config_path = config_path or getattr(scope_context, "loaded_path", None)
        loaded_config = (
            loaded_config
            if loaded_config is not None
            else getattr(scope_context, "loaded", None)
        )
    root = Path(series_root) if series_root is not None else None
    path = Path(config_path) if config_path else None
    context = WorkflowStatusContext(
        root,
        str(run_id),
        path,
        scope_context=scope_context,
        method_selection=method_selection,
        selection_plan_reference=selection_plan_reference,
        review_reference=review_reference,
        result_receipt_reference=result_receipt_reference,
    )
    issues = []
    if root is None:
        context.issues = ("No registered series is selected.",)
        return context
    try:
        manifest = context.document(root / "series.toml")
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported series.toml schema_version.")
        context.manifest = manifest
        runs, paths = _map(manifest.get("runs")), _map(manifest.get("paths"))
        ids = runs.get("ids")
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise ValueError("Series has no valid ordered run IDs.")
        wanted = next(
            (item for item in ids if item.casefold() == str(run_id).casefold()), ""
        )
        context.run_id = wanted or str(run_id)
        if wanted:
            run = _path(
                _map(runs.get("config_files")).get(wanted)
                or f"{paths.get('run_configs', 'config/runs')}/{wanted}.toml",
                root,
            )
            merged = _path(
                _map(runs.get("merged_files")).get(wanted)
                or f"{paths.get('merged_configs', 'config/merged')}/{wanted}.toml",
                root,
            )
            context.run_bound = bool(
                path is not None
                and path.is_file()
                and any(_same(path, item) for item in (run, merged))
            )
        if path is not None:
            context.config = (
                loaded_config
                if isinstance(loaded_config, Mapping)
                else context.document(path)
            )
            if not context.run_bound:
                issues.append(
                    "Open configuration is not bound to the selected registered run."
                )
        else:
            context.config = loaded_config if isinstance(loaded_config, Mapping) else {}
            issues.append("Choose and load a registered run configuration.")
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        issues.append(str(error))
    context.issues = tuple(issues)
    return context


def _artifact(path: Path | None, kind: str = "saved file") -> Artifact | None:
    if path is None:
        return None
    try:
        stat = path.stat()
        if not stat_module.S_ISREG(stat.st_mode) or stat.st_size == 0:
            return None
        return Artifact(str(path), kind, stat.st_size, oslo_date(stat.st_mtime))
    except OSError:
        return None


def _missing(
    summary: str, producer: str, *missing: str, next_step: str = ""
) -> RowStatus:
    return RowStatus(
        "missing",
        summary,
        missing=tuple(str(item) for item in missing if item),
        producer=producer,
        next_step=next_step or producer,
    )


def _saved(
    summary: str,
    artifacts: Sequence[Artifact | None],
    producer: str,
    document: Mapping | None = None,
) -> RowStatus:
    artifacts = tuple(artifacts)
    if not artifacts or any(item is None for item in artifacts):
        return _missing(
            "Saved output changed during inspection.",
            producer,
            "A required saved file disappeared or became unreadable while its metadata was read.",
            next_step="Refresh this row after checking the saved output; no calculation was started.",
        )
    document = document or {}
    for key in (
        "finished_utc",
        "completed_utc",
        "created_utc",
        "updated_utc",
        "saved_utc",
    ):
        date = oslo_date(document.get(key))
        if date:
            return RowStatus(
                "done", summary, date, artifacts, producer=producer, date_source=key
            )
    return RowStatus(
        "done",
        summary,
        artifacts[0].modified if artifacts else "",
        artifacts,
        producer=producer,
        date_source="file modification time" if artifacts else "",
    )


def _require_files(
    paths: Sequence[Path | None], producer: str, summary: str
) -> RowStatus:
    artifacts = [_artifact(path) for path in paths]
    missing = [
        str(path) if path is not None else "No output path is configured."
        for path, artifact in zip(paths, artifacts)
        if artifact is None
    ]
    if not paths:
        missing.append("No saved output paths are configured for this step.")
    if missing:
        return _missing("Saved output is missing.", producer, *missing)
    return _saved(summary, [item for item in artifacts if item is not None], producer)


def _package(context: WorkflowStatusContext) -> Path | None:
    if context.series_root is None:
        return None
    return _path(
        _map(context.manifest.get("paths")).get("geometry_facies", "geometry_facies"),
        context.series_root,
    )


def _geometry_revision(
    context: WorkflowStatusContext,
) -> tuple[Path, Mapping, Artifact]:
    package = _package(context)
    if package is None:
        raise ValueError("No series geometry package is configured.")
    current = package / "overrides/geometry_facies/current.json"
    value = context.document(current)
    revision = value.get("revision_id")
    if value.get("schema") != 1 or value.get("format") != GEOMETRY_FORMAT:
        raise ValueError("Current geometry metadata has an unsupported format.")
    if (
        not isinstance(revision, str)
        or len(revision) != 32
        or any(c not in "0123456789abcdef" for c in revision)
    ):
        raise ValueError("Current geometry revision ID is invalid.")
    artifact = _artifact(current, "current geometry revision")
    if artifact is None:
        raise ValueError("Current geometry revision metadata is missing.")
    return package, value, artifact


def _geometry(context: WorkflowStatusContext, item: str) -> RowStatus:
    producer = {
        "rig_frame": "Rig Geometry",
        "layers": "Layers and facies",
        "export": "Sand map in all runs",
    }.get(item, "Geometry and facies")
    package = _package(context)
    if package is None:
        return _missing(
            "No geometry workspace is saved.",
            producer,
            "No series geometry package is configured.",
        )
    try:
        if item == "rig_frame":
            path = package / "series.json"
            project = context.document(path)
            link = _map(_map(project.get("tools")).get("native_series"))
            reference = str(link.get("reference_run") or "")
            if not reference or reference not in _map(context.manifest.get("runs")).get(
                "ids", []
            ):
                raise ValueError(
                    "No saved registered reference run is bound to this geometry workspace."
                )
            frame = _map(_map(link.get("rig_frames")).get(reference))
            corners = frame.get("corners")
            # RigFrame.to_dict stores the actual named pixel points.
            names = ("top_left", "bottom_left", "bottom_right", "top_right")
            if not (
                isinstance(corners, Mapping)
                and all(
                    name in corners
                    and isinstance(corners[name], (list, tuple))
                    and len(corners[name]) == 2
                    and all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                        for value in corners[name]
                    )
                    for name in names
                )
            ):
                raise ValueError(
                    "Four saved rig corners are missing for the reference run."
                )
            if (
                frame.get("run_id")
                and str(frame["run_id"]).casefold() != reference.casefold()
            ):
                raise ValueError("Saved rig frame belongs to another reference run.")
            points = link.get("injection_points")

            def finite(value):
                return (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                )

            def located(point):
                if not isinstance(point, Mapping):
                    return False
                if all(finite(point.get(key)) for key in ("x_m", "y_m")):
                    return True
                pixel = _map(point.get("pixel"))
                return str(
                    pixel.get("run_id") or ""
                ).casefold() == reference.casefold() and all(
                    finite(pixel.get(key)) for key in ("row", "col")
                )

            if (
                not isinstance(points, list)
                or not points
                or not all(located(point) for point in points)
            ):
                raise ValueError(
                    "Saved injection points need finite rig coordinates or a finite pixel bound to the reference run."
                )
            artifact = _artifact(path, "saved geometry frame and injection points")
            return _saved(
                "Saved rig corners and injection points.", [artifact], producer, project
            )
        package, revision, artifact = _geometry_revision(context)
        if item == "export":
            destination = package / "exports/native_setup" / revision["revision_id"]
            path = destination / "export.json"
            try:
                receipt = context.document(path)
            except FileNotFoundError:
                return RowStatus(
                    state="missing",
                    summary="Apply has not run for the saved geometry revision yet.",
                    producer=producer,
                    next_step="Prepare the Apply and Build job, then start it in Jobs.",
                )
            if (
                receipt.get("schema") != 1
                or receipt.get("format") != EXPORT_FORMAT
                or receipt.get("revision_id") != revision["revision_id"]
            ):
                raise ValueError(
                    "Saved export belongs to another geometry revision or format."
                )
            names = (
                "colored_labels.png",
                "native_labels.npz",
                "facies_properties.csv",
                "native_setup.toml",
            )
            for name in names:
                descriptor = _map(_map(receipt.get("files")).get(name))
                digest = descriptor.get("sha256")
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                ):
                    raise ValueError(f"Saved export lacks its {name} descriptor.")
            status = _require_files(
                [destination / name for name in names],
                producer,
                "Saved export for the current geometry revision; importing it into run configurations is a separate step.",
            )
            if status.state == "done":
                return _saved(
                    status.summary,
                    [artifact, _artifact(path, "export receipt"), *status.artifacts],
                    producer,
                    receipt,
                )
            return status
        configuration, bound = _map(revision.get("configuration")), _map(
            revision.get("artifacts")
        )
        paths = [
            _path(configuration.get("labels_path"), package),
            _path(bound.get("facies_path"), package),
            _path(bound.get("facies_properties_path"), package),
        ]
        status = _require_files(
            paths,
            producer,
            "Saved layers and facies for the current geometry revision.",
        )
        return (
            _saved(status.summary, [artifact, *status.artifacts], producer, revision)
            if status.state == "done"
            else status
        )
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        return _missing(
            "Saved geometry output is not established.", producer, str(error)
        )


def _results(context: WorkflowStatusContext) -> Path | None:
    return _path(_map(context.config.get("data")).get("results"), context.series_root)


def _protocols(context: WorkflowStatusContext) -> RowStatus:
    """Attach extra metadata to every original configured-protocol outcome."""
    return _protocol_extra_summary(context, _configured_protocols(context))


def _configured_protocols(context: WorkflowStatusContext) -> RowStatus:
    table = _map(context.config.get("protocols"))
    imaging = table.get("imaging")
    data = _map(context.config.get("data"))
    plural = data.get("folders")
    singular = data.get("folder")
    if plural is None or plural == "":
        plural = []
    elif not isinstance(plural, list) or not plural:
        return _missing(
            "The imaging protocol cannot be bound to configured data folders.",
            "Protocols",
            "Configured data.folders must be a non-empty list.",
        )
    folders = (
        [] if singular is None or singular in ("", "No file chosen") else [singular]
    ) + plural
    if not isinstance(imaging, Mapping) or not imaging:
        return _missing(
            "No saved per-folder imaging protocol is established.",
            "Protocols",
            "Configure the imaging mapping for every data folder.",
        )
    if (
        not folders
        or len(folders) > MAX_LIST_ENTRIES
        or len(imaging) > MAX_LIST_ENTRIES
    ):
        return _missing(
            "The imaging protocol cannot be bound to configured data folders.",
            "Protocols",
            "Configure a bounded, non-empty data.folder or data.folders list.",
        )

    def identity(value):
        if not isinstance(value, (str, os.PathLike)):
            raise ValueError("Protocol data-folder identity must be a path.")
        path = _path(value, context.series_root)
        if path is None:
            raise ValueError("Protocol data-folder identity is empty or malformed.")
        return os.path.normcase(os.path.abspath(path))

    def destination(value):
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError("Protocol file/sheet entry is malformed.")
            value = value[0]
        if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
            raise ValueError(
                "An explicitly configured protocol file is empty or malformed."
            )
        return _path(value, context.series_root)

    try:
        expected = {identity(folder) for folder in folders}
        actual = {identity(folder) for folder in imaging}
        if len(actual) != len(imaging) or expected != actual:
            raise ValueError("The imaging mapping does not exactly cover data.folders.")
        paths = [destination(value) for value in imaging.values()]
        injection = None
        # The original producer makes injection/PT only when explicitly named.
        # Their absence is not invented as a failure of imaging-protocol setup.
        for key in ("injection", "pressure_temperature"):
            value = table.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            paths.append(destination(value))
            if key == "injection":
                injection = paths[-1]
        status = _require_files(
            paths,
            "Protocols",
            "Saved configured protocol files; file presence does not validate protocol contents.",
        )
        if injection is None or status.state != "done":
            return status
        # A corrected injection position keeps the step's confirmation; the
        # schedule identifies the injection protocol instead of its size and time.
        schedule = injection_schedule(injection)
        artifacts = tuple(replace(item, kind="injection protocol", content=schedule)
                          if schedule and _same(Path(item.path), injection) else item
                          for item in status.artifacts)
        return replace(status, artifacts=artifacts)
    except (OSError, ValueError, TypeError) as error:
        return _missing(
            "Saved configured protocols are not established.", "Protocols", str(error)
        )


def _protocol_extra_summary(context: WorkflowStatusContext, status: RowStatus) -> RowStatus:
    """Report one loaded run's extra metadata without changing completion."""
    if context.series_root is None or not context.run_id:
        return status
    folder = _path(_map(context.manifest.get("paths")).get("protocols", "protocols"),
                   context.series_root)
    if folder is None:
        return status
    import csv
    from .series_convert import inspect_protocol_extras
    try:
        extras = inspect_protocol_extras(folder / context.run_id)
    except (OSError, ValueError, TypeError, UnicodeError, csv.Error) as error:
        return replace(status, summary=status.summary +
            " Extra protocol metadata not inspected: " + str(error))
    if not extras:
        return status
    details = []
    for item in extras:
        name = Path(item.get("path") or item["file"]["path"]).name
        if item["role"] == "image_blacklist":
            name += " (" + str(item["row_count"]) + " image blacklist rows)"
        details.append(name)
    return replace(status, summary=status.summary +
        " Extra metadata ignored for completion: " + "; ".join(details) + ".")


def _depth(context: WorkflowStatusContext) -> RowStatus:
    config = _map(context.config.get("depth"))
    result = _results(context)
    path = _path(config.get("depth_map"), context.series_root) or (
        result / "setup/depth/depth_map.npz" if result else None
    )
    # CSV measurements alone are never a completed interpolated map.
    return _require_files(
        [path],
        "Compute depth map",
        "Saved configured depth map (contents and range are not loaded for status).",
    )


def _configured_rig_baseline(context: WorkflowStatusContext) -> Path:
    """Mirror selected data-folder order using path/stat metadata only."""
    data = _map(context.config.get("data"))
    value = data.get("baseline")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("No baseline is configured for the loaded run.")
    raw = Path(value)
    values = [data.get("folder"), *(data.get("folders") or [])]
    if len(values) > MAX_LIST_ENTRIES:
        raise ValueError("Too many configured image folders for passive status.")
    folders = []
    for value in values:
        folder = _path(value, context.series_root)
        if folder is not None and not any(_same(folder, old) for old in folders):
            folders.append(folder)
    if raw.is_absolute():
        baseline = _path(raw, context.series_root)
    elif folders:
        candidates = [folder / raw for folder in folders]
        baseline = next((path for path in candidates if path.is_file()), candidates[0])
    else:
        raise ValueError("Relative baseline has no explicit configured data folder.")
    if data.get("use_cache") is True:
        result = _results(context)
        if result is None:
            raise ValueError("Cached baseline has no configured results folder.")
        relative = Path(baseline.name)
        for folder in folders:
            try:
                relative = Path(os.path.abspath(baseline)).relative_to(
                    os.path.abspath(folder)
                )
                break
            except ValueError:
                continue
        baseline = result / "raw_cache" / relative.with_suffix(".npz")
    return baseline


def _rig(context: WorkflowStatusContext) -> RowStatus:
    result = _results(context)
    root = _path(_map(context.config.get("rig")).get("path"), context.series_root) or (
        result / "setup/rig" if result else None
    )
    producer = "Build run model"
    if root is None:
        return _missing(
            "No run-model folder is configured.",
            producer,
            "Set [rig].path or [data].results.",
        )
    try:
        meta_path = root / "meta_data.json"
        metadata = context.document(meta_path, limit=MAX_RUN_MODEL_METADATA_BYTES)
        baseline = _configured_rig_baseline(context)
        recorded = str(metadata.get("baseline_path") or "")
        # The configured baseline is read through the storage aliases; read the
        # recorded one the same way, so a relocated photograph stays the same.
        if (
            not recorded
            or not Path(recorded).is_absolute()
            or not _same(_path(recorded, context.series_root), baseline)
        ):
            raise ValueError(
                "Saved rig baseline identity does not match the loaded run configuration."
            )
        status = _require_files(
            [
                root / name
                for name in (
                    "baseline.npz",
                    "depth.npz",
                    "labels.npz",
                    "facies.npz",
                    "porosity.npz",
                )
            ],
            producer,
            "Saved configured run-model artefacts; this does not mark a new geometry workspace or export as complete.",
        )
        return (
            _saved(
                status.summary,
                [_artifact(meta_path, "rig baseline metadata"), *status.artifacts],
                producer,
                metadata,
            )
            if status.state == "done"
            else status
        )
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        return _missing("Saved run model is not established.", producer, str(error))


def _method_value(context: WorkflowStatusContext, name: str) -> str:
    supplied = _reference(context.method_selection, name)
    if supplied:
        return supplied
    method = _map(context.manifest.get("method"))
    if name in {"model_version", "mass_calibration"}:
        return str(_map(method.get("phase_model")).get(name) or "")
    return str(_map(method.get("colour_path")).get("calibration") or "")


def _phase_calibration(context: WorkflowStatusContext) -> RowStatus:
    producer = "Phase-model mass calibration"
    path = _path(_method_value(context, "mass_calibration"), context.series_root)
    if path is None:
        return _missing(
            "No saved phase-model mass calibration is referenced.",
            producer,
            "Choose an existing saved calibration reference under Calibration.",
        )
    try:
        value = context.document(path)
        if value.get("complete") is False or value.get("status") in {
            "failed",
            "stopped",
        }:
            raise ValueError("Referenced calibration is incomplete or failed.")
        parameters = (
            _map(value.get("parameters"))
            or _map(_map(value.get("calibration")).get("parameters"))
            or _map(_map(value.get("scalar_result")).get("parameters"))
        )
        if any(
            not isinstance(parameters.get(key), (int, float))
            or isinstance(parameters.get(key), bool)
            or not math.isfinite(parameters[key])
            for key in ("gain", "beta")
        ):
            raise ValueError("Saved calibration lacks finite recorded gain and beta.")
        return _saved(
            "Saved phase-model mass calibration; no refit or score is performed.",
            [_artifact(path, "saved mass calibration")],
            producer,
            value,
        )
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        return _missing(
            "Saved phase-model calibration is unavailable.", producer, str(error)
        )


def _colour_calibration(context: WorkflowStatusContext) -> RowStatus:
    producer = "Colour-path mass calibration"
    root = _path(_method_value(context, "colour_calibration"), context.series_root)
    if root is None:
        return _missing(
            "No saved colour-path calibration is referenced.",
            producer,
            "Choose its existing folder under Calibration.",
        )
    try:
        flash_path = root / "flash/flash.json"
        flash = context.document(flash_path)
        if any(
            key not in flash
            for key in ("min_value_aq", "max_value_aq", "min_value_g", "max_value_g")
        ):
            raise ValueError("Saved flash metadata lacks its four bounds.")
        signals = []
        with os.scandir(root / "signal_model") as listing:
            for count, entry in enumerate(listing, 1):
                if count > MAX_LIST_ENTRIES:
                    raise ValueError(
                        "Signal-model listing limit reached; select a narrower saved calibration folder."
                    )
                path = Path(entry.path)
                if (
                    path.name.startswith("signal_model_")
                    and path.suffix.casefold() == ".csv"
                ):
                    item = _artifact(path, "saved signal-model CSV")
                    if item:
                        signals.append(item)
        if not signals:
            raise ValueError(
                "No saved signal_model_*.csv is present in this explicit calibration folder."
            )
        return _saved(
            "Saved referenced colour-path calibration files; no new equivalence or mass-balance score is inferred.",
            [_artifact(flash_path, "saved flash bounds"), *signals],
            producer,
            flash,
        )
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        return _missing(
            "Saved colour-path calibration is unavailable.", producer, str(error)
        )


def _colour_path(context: WorkflowStatusContext) -> RowStatus:
    producer = "Colour Path"
    embedding = _map(_map(context.config.get("calibration")).get("color")).get(
        "embedding"
    )
    entries = context.config.get("color_path")
    entry = (
        next(
            (
                row
                for row in entries
                if isinstance(row, Mapping) and row.get("name") == embedding
            ),
            {},
        )
        if isinstance(entries, list)
        else {}
    )
    result = _results(context)
    root = _path(entry.get("root"), context.series_root) or (
        result / "color/color_path" / str(embedding) if embedding and result else None
    )
    if root is None:
        return _missing(
            "No colour-path embedding is configured.",
            producer,
            "Set the selected run's calibration.color.embedding and colour-path definition.",
        )
    try:
        meta_path = root / "metadata.json"
        metadata = context.document(meta_path)
        if not isinstance(metadata.get("label_ids"), list) or not metadata.get("basis"):
            raise ValueError("Saved colour-path metadata lacks basis/label IDs.")
        if entry.get("basis") and metadata.get("basis") != entry["basis"]:
            raise ValueError(
                "Saved colour-path basis differs from the selected configuration."
            )
        status = _require_files(
            [root / "color_paths.csv"],
            producer,
            "Saved configured colour paths and metadata.",
        )
        return (
            _saved(
                status.summary,
                [_artifact(meta_path, "colour-path metadata"), *status.artifacts],
                producer,
                metadata,
            )
            if status.state == "done"
            else status
        )
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        return _missing(
            "Saved colour-path calibration is not established.", producer, str(error)
        )


def _model(context: WorkflowStatusContext) -> RowStatus:
    producer = "Training"
    root = _path(_method_value(context, "model_version"), context.series_root)
    if root is None:
        return _missing(
            "No saved phase model is referenced.",
            producer,
            "Choose an existing model version under Calibration.",
        )
    try:
        path = root / "summary.json"
        summary = context.document(path)
        if not (summary.get("model_id") or summary.get("campaign_id")):
            raise ValueError("Saved model summary lacks its model identity.")
        if summary.get("complete") is False or summary.get("status") in {
            "failed",
            "stopped",
        }:
            raise ValueError(
                "Saved model summary records an incomplete/failed attempt."
            )
        publication = root / "publication_receipt.json"
        if (
            publication.is_file()
            and context.document(publication).get("qa_status") == "fail"
        ):
            raise ValueError(
                "Saved model publication records QA failure; it remains available for explicit review."
            )
        return _saved(
            "Saved referenced phase-model metadata; scientific readiness is verified by the existing model viewer on selection.",
            [_artifact(path, "saved model summary")],
            producer,
            summary,
        )
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        return _missing(
            "Saved phase model is unavailable or incomplete.", producer, str(error)
        )


def _identity(value: Any) -> str:
    return "".join(c for c in str(value).casefold() if c.isalnum())


def _selection_plan(context: WorkflowStatusContext) -> RowStatus:
    producer = "Image selection plan"
    path = _path(context.selection_plan_reference, context.series_root)
    if path is None:
        return _missing(
            "No saved image-selection plan is selected.",
            producer,
            "Open/select the saved plan under this series' config folder.",
        )
    try:
        if context.series_root is None or not Path(
            os.path.abspath(path)
        ).is_relative_to(Path(os.path.abspath(context.series_root / "config"))):
            raise ValueError(
                "Selected plan is not bound to this series' config folder."
            )
        document = context.document(path)
        purposes = _map(document.get("purposes"))
        if (
            path.suffix.casefold() != ".toml"
            or document.get("schema_version") != 1
            or not purposes
        ):
            raise ValueError(
                "Selected plan is not a version-1 image-selection TOML with saved purposes."
            )
        for name, section in purposes.items():
            mode = _map(section).get("mode")
            if name not in {"review", "calibration", "production"} or mode not in {
                "all",
                "target_hours",
            }:
                raise ValueError("Selected plan has an unsupported purpose or mode.")
            if mode == "target_hours":
                targets = _map(section).get("target_hours")
                if (
                    not isinstance(targets, list)
                    or not targets
                    or not all(
                        isinstance(x, (int, float))
                        and not isinstance(x, bool)
                        and math.isfinite(x)
                        for x in targets
                    )
                ):
                    raise ValueError(
                        "Selected target-hours plan has no finite saved targets."
                    )
        owner = document.get("series_id")
        if owner and _identity(owner) != _identity(context.manifest.get("series_id")):
            raise ValueError("Selected plan explicitly identifies another series.")
        return _saved(
            "Saved version-1 image-selection plan in this series' config folder; selection receipts remain separate.",
            [_artifact(path, "saved image-selection plan")],
            producer,
            document,
        )
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        return _missing(
            "Selected image-selection plan is unavailable or unbound.",
            producer,
            str(error),
        )


def _phase_review(context: WorkflowStatusContext) -> RowStatus:
    producer = "Phase review"
    path = _path(context.review_reference, context.series_root)
    root = _path(_method_value(context, "model_version"), context.series_root)
    if path is None or root is None:
        return _missing(
            "No prepared review metadata is bound to the selected model.",
            producer,
            "Choose a saved model and its prepared review collection explicitly.",
        )
    try:
        document = context.document(path)
        model = context.document(root / "summary.json")
        if document.get("format") != "fluidflower-phase-review-collection-v1":
            raise ValueError(
                "Referenced metadata is not a supported prepared phase-review collection."
            )
        model_id = model.get("model_id")
        if not model_id or document.get("source_model_id") != model_id:
            raise ValueError(
                "Prepared review metadata identifies another selected model."
            )
        jobs = document.get("jobs")
        wanted = _identity(context.manifest.get("series_id"))
        ids = {
            _identity(run) for run in _map(context.manifest.get("runs")).get("ids", [])
        }
        if not isinstance(jobs, list) or not any(
            _identity(_map(job).get("dataset")) == wanted
            and _identity(_map(job).get("run")) in ids
            for job in jobs
        ):
            raise ValueError(
                "Prepared review collection has no bound run for this series."
            )
        return _saved(
            "Saved prepared review collection for the selected model and series; review quality and map admission are verified on explicit Open.",
            [_artifact(path, "selected phase-review collection")],
            producer,
            document,
        )
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        return _missing(
            "Selected prepared review metadata is unavailable or unbound.",
            producer,
            str(error),
        )


def _analysis_output(context: WorkflowStatusContext, item: str) -> RowStatus:
    producer = f"Analysis: {item}"
    result = _results(context)
    folder = _path(
        _map(_map(context.config.get("analysis")).get(item)).get("folder"),
        context.series_root,
    ) or (result / item if result else None)
    if folder is None:
        return _missing(
            "No output folder is configured.",
            producer,
            "Set data.results or this Analysis row's folder.",
        )
    # Prefer the producer's fixed scalar table over directory contents.
    fixed = (
        result
        / "sparse_data"
        / ("integrated_mass.csv" if item == "mass" else "integrated_volume.csv")
        if result is not None and item in {"mass", "volume"}
        else None
    )
    fixed_artifact = _artifact(fixed, "saved scalar table")
    if fixed_artifact:
        return _saved(
            "Saved configured Analysis table; whole-run completion is not inferred from presence.",
            [fixed_artifact],
            producer,
        )
    # Bounded, shallow mode-folder listing for existing producer image exports.
    # A non-empty directory alone never counts as output. No rglob or history.
    pending = [(folder, 0)]
    visited = 0
    entries = 0
    try:
        while pending and visited < MAX_OUTPUT_DIRS:
            current, depth = pending.pop(0)
            visited += 1
            with os.scandir(current) as listing:
                for entry in listing:
                    entries += 1
                    if entries > MAX_LIST_ENTRIES:
                        return _missing(
                            "Saved output was not established within the bounded folder listing.",
                            producer,
                            str(folder),
                            next_step="Open the configured output folder explicitly.",
                        )
                    path = Path(entry.path)
                    if path.suffix.casefold() in {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".npz",
                        ".csv",
                    }:
                        artifact = _artifact(path, "saved Analysis output")
                        if artifact:
                            return _saved(
                                "Saved configured Analysis output; whole-run completion is not inferred from presence.",
                                [artifact],
                                producer,
                            )
                    elif depth < 2 and entry.is_dir(follow_symlinks=False):
                        pending.append((path, depth + 1))
    except OSError as error:
        return _missing(
            "No saved Analysis output is established.", producer, str(error)
        )
    return _missing("No saved Analysis output is established.", producer, str(folder))


def _delivery(context: WorkflowStatusContext) -> RowStatus:
    producer = "Frozen model mass analysis"
    path = _path(context.result_receipt_reference, context.series_root)
    if path is None:
        return _missing(
            "No completed saved delivery is selected.",
            producer,
            "Choose an existing delivery in the saved-results viewer; no result history is searched for row status.",
        )
    try:
        receipt = context.document(path)
        if receipt.get("format") != DELIVERY_FORMAT:
            raise ValueError("Selected delivery receipt has an unsupported format.")
        candidate = _map(receipt.get("candidate"))
        run = str(candidate.get("run") or "")
        series = str(candidate.get("dataset") or "").replace("_", "").casefold()
        current_series = (
            str(context.manifest.get("series_id") or "").replace("_", "").casefold()
        )
        registered_runs = {
            str(item).casefold()
            for item in _map(context.manifest.get("runs")).get("ids", [])
        }
        if series != current_series or run.casefold() not in registered_runs:
            raise ValueError(
                "Selected delivery belongs to another registered series/run."
            )
        digest = receipt.get("result_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Selected delivery lacks its recorded result SHA-256.")
        result_path = _path(receipt.get("result"), path.parent)
        artifact = _artifact(result_path, "receipt-bound saved result")
        if artifact is None:
            raise ValueError("Receipt-bound saved result file is missing.")
        # Do not read a potentially large per-frame result just to colour a dot.
        return _saved(
            "Saved delivery receipt and its bound result are present; the viewer verifies result SHA/content on explicit Open.",
            [_artifact(path, "delivery receipt"), artifact],
            producer,
            receipt,
        )
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        return _missing(
            "Completed saved delivery is not established.", producer, str(error)
        )


def _saved_row_status(context: WorkflowStatusContext, action: str, item_id: str) -> RowStatus:
    """Return one truthful passive row state without changing action routing."""
    if (action, item_id) == ("jobs", "overview"):
        return RowStatus(
            "not_applicable",
            "View and manage jobs; the selected job's progress is shown in the job list and panel.",
        )
    if action == "calibration" and item_id in {"reset", "delete"}:
        return RowStatus(
            "not_applicable",
            "Maintenance action; no saved artefact completion state.",
            producer="Explicit maintenance action",
            next_step="Available only when explicitly requested; this status does not disable it.",
        )
    if context.series_root is None or context.manifest.get("schema_version") != 1:
        return _missing(
            "Choose a registered series to establish saved output.",
            "Open a registered series",
            *context.issues or ("Series manifest is unavailable.",),
        )
    if (action, item_id) == ("calibration", "images"):
        from .calibration_image_selection import calibration_image_count

        try:
            count = calibration_image_count(context.manifest)
            choice = _map(_map(_map(context.manifest.get("calibration")).get("mass")).get("image_selection"))
            explicit = "max_images_per_run" in choice
            return RowStatus(
                "saved_choice" if explicit else "default_choice",
                f"Up to {count} calibration target images per run. "
                + ("The count is saved in this series. " if explicit else "Using the default; no explicit count is saved. ")
                + "Complete phase history and required references are retained. This is an input choice, not a completed calibration.",
                counts={"max_images_per_run": count, "explicit": explicit},
            )
        except (ValueError, TypeError, AttributeError) as error:
            return RowStatus("choice_error", "Calibration image choice is invalid: " + str(error))
    if (action, item_id) == ("image_selection", "plan"):
        from darsia.fluidflower.image_selection_status import saved_selection_summary

        try:
            info = saved_selection_summary(context.manifest)
            if not info["complete"]:
                return _missing(info["summary"], "Image selection",
                                "Choose and save images for: " + ", ".join(info["missing_runs"]),
                                next_step="Edit image selection; no calculation starts.")
            return _saved(info["summary"] + " " + info["note"],
                          [_artifact(context.series_root / "series.toml")],
                          "Image selection", {"saved_utc": info["saved_utc"]})
        except (OSError, ValueError, TypeError, KeyError) as error:
            return _missing("Saved image choices are unavailable.", "Image selection", str(error))
    if action == "geometry_facies":
        return _geometry(context, item_id)
    if action == "phase_model":
        if item_id == "training":
            return _model(context)
        if item_id == "mass_from_phase":
            return _phase_calibration(context)
        if item_id == "selection_plan":
            return _selection_plan(context)
    if action == "mass_analysis":
        if item_id == "frozen":
            return _delivery(context)
        if item_id == "review":
            return _phase_review(context)
    if (
        action in {"preprocessing", "setup", "calibration", "analysis"}
        and not context.run_bound
    ):
        return _missing(
            "Load a registered run configuration to establish this row's saved output.",
            "Choose and load the reference/run configuration",
            *context.issues,
        )
    if action == "preprocessing":
        if item_id == "protocols":
            return _protocols(context)
        if item_id == "depth_measurements":
            return _require_files(
                [
                    _path(
                        _map(context.config.get("depth")).get("measurements"),
                        context.series_root,
                    )
                ],
                "Depth measurements",
                "Saved configured depth measurements.",
            )
        if item_id == "crop":
            crop = _map(
                _map(_map(context.config.get("corrections")).get("curvature")).get(
                    "crop"
                )
            )
            if all(
                key in crop
                for key in ("top_left", "bottom_left", "bottom_right", "top_right")
            ):
                return _saved(
                    "Saved crop corners in the loaded run configuration; not a new geometry revision.",
                    [_artifact(context.config_path, "loaded run configuration")],
                    "Rig Geometry",
                )
            return _missing(
                "No crop corners are saved in the loaded configuration.",
                "Rig Geometry",
                "Save/apply the existing geometry through the explicit Setup action.",
            )
    if action == "setup":
        if item_id == "depth":
            return _depth(context)
        if item_id == "rig":
            return _rig(context)
        if item_id in {"segmentation", "facies"}:
            result = _results(context)
            section = "labeling" if item_id == "segmentation" else "facies"
            key = "labels" if item_id == "segmentation" else "path"
            default = (
                "setup/labels/labels.npz"
                if item_id == "segmentation"
                else "setup/facies/facies.npz"
            )
            path = _path(
                _map(context.config.get(section)).get(key), context.series_root
            ) or (result / default if result else None)
            return _require_files(
                [path],
                "Layers and facies / Sand map in all runs",
                "Saved configured labels/facies output; no new editable geometry revision is inferred.",
            )
        if item_id == "all":
            states = [_protocols(context), _depth(context), _rig(context)]
            if all(state.state == "done" for state in states):
                return _saved(
                    "Saved protocol, depth-map and run-model artefacts; no new execution or scientific gate is inferred.",
                    [a for state in states for a in state.artifacts],
                    "Full setup",
                )
            return _missing(
                "Some configured Setup outputs are missing.",
                "Full setup",
                *[
                    message
                    for state in states
                    if state.state != "done"
                    for message in state.missing
                ],
            )
    if action == "calibration":
        if item_id == "color":
            return _colour_path(context)
        if item_id in {"mass", "default_mass", "mass_optuna"}:
            return _colour_calibration(context)
    if action == "analysis" and item_id in {
        "cropping",
        "mass",
        "segmentation",
        "fingers",
        "thresholding",
        "volume",
    }:
        return _analysis_output(context, item_id)
    return _missing(
        "No saved-output status is defined for this row.",
        f"{action}/{item_id}",
        "Open the row's existing configuration/viewer explicitly; no applicability is inferred from absent settings.",
    )


def _active_method(context: WorkflowStatusContext) -> str:
    recorded = _map(context.manifest.get("method")).get("active")
    return str(recorded or _reference(context.method_selection, "active"))



def analysis_run_scope(context: WorkflowStatusContext) -> tuple[list[str], dict[str, str]]:
    """Registered analysis runs and explicit exclusions; never infer exclusions."""
    table = _map(context.manifest.get("runs"))
    ids = table.get("ids")
    if (not isinstance(ids, list) or len(ids) > MAX_LIST_ENTRIES
            or any(not isinstance(run, str) or not run for run in ids)
            or len({run.casefold() for run in ids}) != len(ids)):
        raise ValueError("Analysis requires bounded distinct registered run IDs.")
    names = {run.casefold(): run for run in ids}
    excluded = table.get("excluded", {})
    if not isinstance(excluded, Mapping) or len(excluded) > MAX_LIST_ENTRIES:
        raise ValueError("runs.excluded must be a bounded table of explicit reasons.")
    reasons = {}
    for run, reason in excluded.items():
        if (not isinstance(run, str) or run.casefold() not in names
                or not isinstance(reason, str) or not reason.strip()):
            raise ValueError("Every excluded run needs a registered ID and nonempty reason.")
        canonical = names[run.casefold()]
        if canonical in reasons:
            raise ValueError("Duplicate excluded run identity.")
        reasons[canonical] = reason
    return ids, reasons



def analysis_run_counts(context: WorkflowStatusContext, saved_runs: Sequence[str]) -> dict:
    """Counts only explicitly supplied saved run identities, not computed success."""
    ids, excluded = analysis_run_scope(context)
    names = {run.casefold(): run for run in ids}
    if len(saved_runs) > MAX_LIST_ENTRIES:
        raise ValueError("Too many saved run identities for passive counts.")
    saved = set()
    for run in saved_runs:
        if not isinstance(run, str) or run.casefold() not in names:
            raise ValueError("Saved output names an unregistered run.")
        canonical = names[run.casefold()]
        if canonical not in excluded:
            saved.add(canonical)
    eligible = [run for run in ids if run not in excluded]
    missing = [run for run in eligible if run not in saved]
    return {"run_count": len(ids), "eligible_count": len(eligible),
            "saved_run_count": len(saved), "excluded_count": len(excluded),
            "missing_runs": missing, "excluded_runs": dict(excluded)}



def _count_text(counts: Mapping) -> str:
    text = f"{counts['saved_run_count']} done · {counts['excluded_count']} excluded"
    if counts["missing_runs"]:
        text += f" · {len(counts['missing_runs'])} not done"
    return text



def _selected_exclusion(context: WorkflowStatusContext) -> str:
    _, reasons = analysis_run_scope(context)
    reason = next((value for run, value in reasons.items()
                   if run.casefold() == context.run_id.casefold()), "")
    return f"Selected {context.run_id.upper()}: excluded: {reason}" if reason else ""



def _production_grid_status(context: WorkflowStatusContext) -> RowStatus:
    """Saved when series.toml binds a production grid, the images mass analysis computes."""
    choice = _map(context.manifest.get("production_image_selection"))
    if not isinstance(choice.get("selection_manifest"), Mapping):
        return _missing("No production image selection is saved.", "Mass analysis > Image selection",
                        next_step="Choose the production images; mass analysis computes them in every run.")
    return _saved("A production image selection is saved.", [_artifact(context.series_root / "series.toml")],
                  "Mass analysis > Image selection")


def _image_selection_with_exclusions(context: WorkflowStatusContext) -> RowStatus:
    """Saved Setup choices use their active set; analysis has its own guards."""
    from darsia.fluidflower.image_selection_status import setup_selection_summary

    try:
        info = setup_selection_summary(context.manifest)
        counts = {
            key: info[key]
            for key in (
                "run_count", "eligible_count", "saved_run_count", "excluded_count",
                "missing_runs", "excluded_runs", "inactive_excluded_count",
                "inactive_excluded_runs",
            )
        }
        if not info["eligible_count"]:
            return RowStatus(
                "not_applicable", "No active run requires saved image choices.", counts=counts
            )
        if not info["complete"]:
            status = _missing(
                info["summary"], "Image selection",
                "Choose and save images for: " + ", ".join(info["missing_runs"]),
                next_step="Edit image selection; no calculation starts.",
            )
        else:
            status = _saved(
                info["summary"] + " " + info["note"],
                [_artifact(context.series_root / "series.toml")], "Image selection",
                {"saved_utc": info["saved_utc"]},
            )
        reason = next(
            (reason for run, reason in info["excluded_runs"].items()
             if run.casefold() == context.run_id.casefold()), ""
        )
        selected = f"Selected {context.run_id.upper()}: excluded: {reason}" if reason else ""
        return replace(status, counts=counts, selected_run_text=selected)
    except (OSError, ValueError, TypeError, KeyError) as error:
        return _missing("Saved image choices are unavailable.", "Image selection", str(error))



def row_status(context: WorkflowStatusContext, action: str, item_id: str) -> RowStatus:
    """Method-aware passive display; original routing and Setup criteria stay intact."""
    if context.series_root is None or context.manifest.get("schema_version") != 1:
        return _saved_row_status(context, action, item_id)
    active = _active_method(context)
    if (action, item_id) == ("method", "use_for_series"):
        recorded = _map(context.manifest.get("method")).get("active")
        from .series_method import MODEL_COMBINATIONS
        if recorded in MODEL_COMBINATIONS:
            return _saved("Saved calibration method: " + MODEL_COMBINATIONS[recorded]["name"] + ".",
                          [_artifact(context.series_root / "series.toml")], "Calibration Method")
        return _missing("No calibration method is saved for this series.", "Calibration Method",
                        "Choose a method and save it.")
    if active == "phase_model" and (action, item_id) == ("setup", "all"):
        return RowStatus("not_applicable", "Final Setup status is shown under Finish setup. Geometry Build alone does not complete image correction and colour paths.")
    colour_rows = {("calibration", item) for item in ("color", "mass", "default_mass", "mass_optuna")}
    colour_rows.update(("analysis", item) for item in ("cropping", "mass", "segmentation", "fingers", "thresholding", "volume"))
    phase_row = action == "phase_model" or (action == "mass_analysis" and item_id in {"frozen", "review"})
    if (active == "phase_model" and (action, item_id) in colour_rows) or (active == "colour_path" and phase_row):
        selected = ""
        if action in {"analysis", "mass_analysis"}:
            try:
                selected = _selected_exclusion(context)
            except (ValueError, TypeError, KeyError) as error:
                selected = "Analysis exclusion scope unavailable: " + str(error)
        return RowStatus("not_applicable", f"Not applicable for method {active}.", selected_run_text=selected)
    if (action, item_id) == ("image_selection", "plan"):
        return _image_selection_with_exclusions(context)
    if (action, item_id) == ("image_selection", "production"):
        return _production_grid_status(context)
    try:
        selected = _selected_exclusion(context) if (action in {"analysis", "mass_analysis"}
                     or (action, item_id) == ("phase_model", "selection_plan")) else ""
        if action == "analysis" and selected:
            return RowStatus("not_applicable", selected, selected_run_text=selected)
        status = _saved_row_status(context, action, item_id)
        if (action, item_id) == ("mass_analysis", "review") and status.state == "done":
            document = context.document(_path(context.review_reference, context.series_root))
            wanted = _identity(context.manifest.get("series_id"))
            reviewed = {_identity(_map(job).get("run")) for job in document.get("jobs", [])
                        if _identity(_map(job).get("dataset")) == wanted}
            runs = [run for run in analysis_run_scope(context)[0] if _identity(run) in reviewed]
            counts = analysis_run_counts(context, runs)
            if counts["saved_run_count"] == 0:
                return _missing("Review collection has no non-excluded run for this series.", "Phase review",
                                "Choose a collection with a bound, non-excluded registered run.")
            return replace(status, summary=status.summary + " " + _count_text(counts) + ".",
                           counts=counts, selected_run_text=selected)
        if (action, item_id) == ("mass_analysis", "frozen") and status.state == "done":
            _, excluded = analysis_run_scope(context)
            if excluded:
                receipt = context.document(_path(context.result_receipt_reference, context.series_root))
                run = str(_map(receipt.get("candidate")).get("run") or "")
                reason = next((reason for key, reason in excluded.items()
                               if key.casefold() == run.casefold()), "")
                if reason:
                    return RowStatus("not_applicable", f"Saved result run {run.upper()} is excluded: {reason}",
                                     selected_run_text=selected)
                counts = analysis_run_counts(context, [run])
                return replace(status, summary=status.summary + " Explicit selected run receipt: " + _count_text(counts) + ".",
                               counts=counts, selected_run_text=selected)
        return replace(status, selected_run_text=selected) if selected else status
    except (OSError, ValueError, TypeError, KeyError) as error:
        return _missing("Registered analysis scope is unavailable: " + str(error), f"{action}/{item_id}", str(error))
