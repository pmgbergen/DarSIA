"""Read-only UI scope and calibration configuration provenance.

Series/run identity comes from series.toml. Shared colour calibration scope is
shown only when its relevant calibration branch is demonstrably inherited from
common.toml by the currently loaded run configuration. This describes settings
scope, never a new all-run execution path. No calibration artifacts are opened.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_CONFIG_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ScopeContext:
    series_root: Path | None
    series_label: str
    run_ids: tuple[str, ...] = ()
    selected_run: str = ""
    common_path: Path | None = None
    run_path: Path | None = None
    merged_path: Path | None = None
    loaded_path: Path | None = None
    common: Mapping = field(default_factory=dict)
    run: Mapping = field(default_factory=dict)
    loaded: Mapping = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    run_count_known: bool = False


@dataclass(frozen=True)
class RowScope:
    scope: str  # series or run: the row's actual configuration/execution scope
    line: str
    note: str = ""

    @property
    def run_selector_enabled(self) -> bool:
        return self.scope == "run"


def _mapping(value) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _path(root: Path, value) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _same_path(first: Path | None, second: Path | None) -> bool:
    return bool(
        first is not None
        and second is not None
        and os.path.normcase(os.path.abspath(first))
        == os.path.normcase(os.path.abspath(second))
    )


def _read_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(MAX_CONFIG_BYTES + 1)
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError(f"Configuration exceeds the scope-display limit: {path}")
    return tomllib.loads(raw.decode("utf-8-sig"))


def load_scope_context(
    series_root: str | Path | None,
    series_id: str = "",
    run_id: str = "",
    selected_config_path: str | Path | None = None,
    loaded_config: Mapping | None = None,
    *,
    series_label: str | None = None,
) -> ScopeContext:
    """Read only manifest/common/selected-run TOML; use loaded UI values.

    The caller can retain this context for all row labels in one UI refresh.
    No other run files, calibration artifacts, images or outputs are scanned.
    The canonical explicitly selected run/merged TOML may also be read if no
    loaded_config was supplied. The existing loaded_config is used to detect stale/edited
    merged calibration values; its bytes are never modified or reloaded here.
    """
    selected_run = run_id
    loaded_path = selected_config_path
    label = str(series_label or series_id or "selected series")
    root = Path(series_root) if series_root is not None else None
    loaded_file = Path(loaded_path) if loaded_path else None
    if root is None:
        return ScopeContext(
            None,
            label,
            selected_run=str(selected_run or ""),
            loaded_path=loaded_file,
            notes=("No registered series folder is selected.",),
        )
    notes = []
    try:
        manifest = _read_toml(root / "series.toml")
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported series manifest schema.")
        paths = _mapping(manifest.get("paths"))
        runs = _mapping(manifest.get("runs"))
        raw_ids = runs.get("ids")
        if not isinstance(raw_ids, list) or not all(
            isinstance(run, str) for run in raw_ids
        ):
            raise ValueError("The series has no valid ordered run IDs.")
        run_ids = tuple(raw_ids)
        label = str(series_label or manifest.get("series_id") or series_id or label)
        wanted = str(selected_run or "")
        selected = next(
            (run for run in run_ids if run.casefold() == wanted.casefold()), wanted
        )
        common_path = _path(root, paths.get("common_config", "config/common.toml"))
        run_path = None
        merged_path = None
        if selected in run_ids:
            files = _mapping(runs.get("config_files"))
            default = f"{paths.get('run_configs', 'config/runs')}/{selected}.toml"
            run_path = _path(root, files.get(selected) or default)
            merged_path = (
                _path(root, paths.get("merged_configs", "config/merged"))
                / f"{selected}.toml"
            )
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        return ScopeContext(
            root,
            label,
            selected_run=str(selected_run or ""),
            loaded_path=loaded_file,
            loaded=_mapping(loaded_config),
            notes=(str(error),),
        )
    tables = []
    for path in (common_path, run_path):
        if path is None:
            tables.append({})
            continue
        try:
            tables.append(_read_toml(path))
        except (OSError, ValueError, TypeError, UnicodeError) as error:
            tables.append({})
            notes.append(str(error))
    loaded_values = _mapping(loaded_config)
    if (
        loaded_config is None
        and loaded_file is not None
        and any(
            _same_path(loaded_file, candidate) for candidate in (run_path, merged_path)
        )
    ):
        # Read the explicitly selected controller file only when the GUI has
        # not loaded it. This does not load a GUI config, open a result, or
        # discover another run. A run file already read above is reused.
        try:
            loaded_values = (
                tables[1]
                if _same_path(loaded_file, run_path)
                else _read_toml(loaded_file)
            )
        except (OSError, ValueError, TypeError, UnicodeError) as error:
            notes.append(str(error))
    return ScopeContext(
        root,
        label,
        run_ids,
        selected,
        common_path,
        run_path,
        merged_path,
        loaded_file,
        tables[0],
        tables[1],
        loaded_values,
        tuple(notes),
        True,
    )


def _branch(table: Mapping, name: str):
    return _mapping(table.get("calibration")).get(name)


def _has_leaves(value) -> bool:
    if isinstance(value, Mapping):
        return any(_has_leaves(child) for child in value.values())
    return True


def common_calibration_origin(context: ScopeContext, branch: str) -> tuple[bool, str]:
    """Prove common provenance for color or mass in the active configuration.

    Even a duplicate-valued explicit run override remains run provenance. An
    unbound external config or a stale merged branch cannot acquire common
    provenance merely because a common file contains calibration settings.
    """
    inherited = _branch(context.common, branch)
    override = _branch(context.run, branch)
    if not isinstance(inherited, Mapping) or not inherited:
        return (
            False,
            f"No [calibration.{branch}] definition is established in common.toml.",
        )
    if override is not None and _has_leaves(override):
        return False, f"The selected run explicitly overrides [calibration.{branch}]."
    if context.notes:
        return False, "Common/run provenance could not be established: " + " ".join(
            context.notes
        )
    if context.selected_run not in context.run_ids:
        return (
            False,
            "Choose a registered run before establishing its common calibration provenance.",
        )
    if not any(
        _same_path(context.loaded_path, path)
        for path in (context.run_path, context.merged_path, context.common_path)
    ):
        return (
            False,
            "The loaded configuration is not bound to this run's common/run/merged paths.",
        )
    effective = _branch(context.loaded, branch)
    if effective != inherited:
        return (
            False,
            f"Loaded [calibration.{branch}] differs from common.toml; it may be edited or a stale merged configuration.",
        )
    return (
        True,
        f"[calibration.{branch}] is inherited from {context.common_path}; no selected-run override is present.",
    )


def series_scope(context: ScopeContext, *, note: str = "") -> RowScope:
    count = len(context.run_ids)
    amount = (
        f" ({count} runs)" if context.run_count_known else " (run count unavailable)"
    )
    line = f"Applies to: whole series {context.series_label}{amount}"
    return RowScope(
        "series",
        line,
        note or "This step uses the model and settings for the whole series.",
    )


def scope_summary(context: ScopeContext, scope: RowScope | None) -> str:
    """The scope part of the header line: the whole series with its run count, or the run."""
    if scope is None:
        return ""
    if scope.scope == "series":
        count = (
            f" ({len(context.run_ids)} runs)"
            if context.run_count_known
            else " (run count unavailable)"
        )
        return "Whole series" + count
    text = scope.line.removeprefix("Applies to: ")
    return text[:1].upper() + text[1:]


def _run_scope(context: ScopeContext, note: str = "") -> RowScope:
    loaded_run_bound = context.loaded_path is None or any(
        _same_path(context.loaded_path, path)
        for path in (context.run_path, context.merged_path)
    )
    if (
        context.selected_run
        and context.selected_run in context.run_ids
        and loaded_run_bound
    ):
        line = f"Applies to: run {context.selected_run.upper()} only"
    elif context.selected_run:
        line = "Applies to: selected configuration (run not confirmed)"
    else:
        line = "Applies to: a selected run (choose a run)"
    return RowScope("run", line, note)


def row_scope(context: ScopeContext, action: str, item_id: str) -> RowScope | None:
    """Return supported rows' actual scope; unknown rows are never invented."""
    if (action, item_id) == ("geometry_controls", "curvature"):
        return _run_scope(context, "Optional original curvature settings; saving starts no calculation.")
    if action == "jobs" and item_id in {"overview", "sources"}:
        return series_scope(context, note="Jobs retain their own explicit whole-run selections; opening this list performs no calculation.")
    if action == "jobs" and item_id == "new_analysis":
        return series_scope(context, note="The adopted plan fixes the runs and images; opening this page performs no calculation.")
    if action == "jobs" and item_id == "wasserstein":
        return series_scope(context, note="The original Comparison config fixes its runs, times and ROIs; opening this page performs no calculation.")
    if action == "phase_model" and item_id == "checks":
        return series_scope(context, note="Check results cover the whole series; opening a file performs no calculation and writes nothing.")
    if action == "jobs" and item_id == "production_results":
        return series_scope(context, note="The page lists every run of the chosen finished job; opening it performs no calculation.")
    if (
        (
            action == "phase_model"
            and item_id
            in {"normalisation", "training", "overflow", "mass_from_phase", "calibration_review", "selection_plan", "colour_paths"}
        )
        or (action == "mass_analysis" and item_id in {"frozen", "review"})
        or (action == "method" and item_id == "use_for_series")
        or (action == "image_selection" and item_id in {"plan", "production"})
        or (action == "calibration" and item_id == "images")
        or (action == "setup" and item_id == "finish")
        or (action == "overflow_data" and item_id == "runs")
    ):
        return series_scope(context)
    if action == "analysis" and item_id in {
        "cropping",
        "mass",
        "segmentation",
        "fingers",
        "thresholding",
        "volume",
    }:
        note = "Uses the selected run."
        if item_id == "mass":
            shared, origin = common_calibration_origin(context, "mass")
            if shared:
                note += " Calibration defaults come from common.toml; this run has no override."
        return _run_scope(context, note)
    if action != "calibration":
        return None
    if item_id in {"reset", "delete"}:
        note = (
            "Uses the selected run's configured calibration folders and cache."
            if item_id == "delete"
            else "Uses the selected run."
        )
        return _run_scope(context, note)
    branch = (
        "color"
        if item_id == "color"
        else "mass" if item_id in {"mass", "default_mass", "mass_optuna"} else None
    )
    if branch is None:
        return None
    shared, origin = common_calibration_origin(context, branch)
    if shared:
        # The settings are the series' own, but the action runs on one run, so
        # the run selector stays enabled (N141).
        return RowScope(
            "run",
            f"Settings: whole series (common.toml) · Runs on: {context.selected_run.upper()}",
            "Shared calibration settings from common.toml. This run has no override; other runs may override these defaults. Existing actions use the selected run.",
        )
    return _run_scope(context, "Uses the selected run.")
