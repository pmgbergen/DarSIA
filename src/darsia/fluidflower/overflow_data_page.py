"""Setup > Overflow data: every run's overflow source, file and status (N141 spec 2.3).

The page is a read-only table over all runs of the open series. It reads what
the series records and writes nothing:

- ``series.toml`` lists the runs. It holds no overflow entry of its own; it
  reaches its overflow data through the Studio project it names in
  ``[legacy] studio_project``.
- That project's saved overflow revision (``overrides/overflow``) holds each
  run's source (none, scale images or a table), the file or folders, and the
  scan made when the revision was saved. Its per-run "Overflow measurements"
  step status and its scale-reader review files are shown beside it.

No source file is opened and no scan or discovery runs. What the records do
not say is shown as "—", with the reason in the tooltip.

There is no whole-series attach action. The existing "Attach overflow targets
to selected run" importer (Results and export, legacy comparison method) asks
for one file per run, supports a single series by its own check and keeps the
attachment inside that workspace, so it cannot be run once per run from here
without changing it.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Logical sidebar key of the Setup row; native_darsia_app routes it here.
OVERFLOW_DATA_ROW = ("overflow_data", "runs")
UNKNOWN = "—"
COLUMNS = ("Run", "Source", "File", "Status")
SOURCE_LABELS = {"none": "None", "scale_images": "Scale images", "table": "Table"}
SOURCE_NOTES = {
    "none": "No overflow measurements are used for this run.",
    "scale_images": "Read from scale-display images.",
    "table": "Read from a CSV or Excel table.",
}
ASSIGNMENT_LABELS = {
    "global_registered_source": "Global policy",
    "individual_override": "Individual override",
    "setup_discovery": "Initial setup",
}
SETUP_MODE_LABELS = {
    "none": "not relevant",
    "per_run": "per run",
    "table": "one table",
    "scale_images": "scale-display images",
}
SCAN_LABELS = {"ready": "Ready", "error": "Error", "not_used": "Not used"}
ATTACH_NOTE = (
    "Attaching reviewed overflow targets to one run for the legacy comparison "
    "method stays in Analysis › Results and export › Prepare calculations and "
    "manage saved results › Legacy comparison method and imported results. That "
    "importer asks for a file per selected run and supports a single series by "
    "its own check, so this page offers no attach for all runs."
)


@dataclass(frozen=True)
class OverflowCell:
    """One table cell: its text and the tooltip that explains it."""

    text: str
    tip: str = ""


@dataclass(frozen=True)
class OverflowRun:
    """One run of the series with its source, file and status cells."""

    run: str
    source: OverflowCell
    file: OverflowCell
    status: OverflowCell


@dataclass(frozen=True)
class OverflowData:
    """Everything the page shows; built by read_overflow_data, rendered without I/O."""

    runs: tuple[OverflowRun, ...] = ()
    summary: str = ""
    detail: str = ""


def _plural(count: int, word: str) -> str:
    """'1 run' or '1,749 rows'."""
    return f"{count:,} {word}{'' if count == 1 else 's'}"


def _int(value: Any) -> int | None:
    """An integer count from a record, or None for anything else (bool included)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _strings(values: Any) -> list[str]:
    """The non-empty strings of a recorded list; anything else gives an empty list."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return [str(value) for value in values if str(value).strip()]


def _utc_text(value: Any, *, date_only: bool = False) -> str:
    """A recorded ISO time as 'YYYY-MM-DD HH:MM UTC' (or its date); other text unchanged."""
    text = str(value or "").strip()
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    zone = ""
    if moment.tzinfo is not None:
        moment, zone = moment.astimezone(timezone.utc), " UTC"
    return moment.strftime("%Y-%m-%d") if date_only else moment.strftime("%Y-%m-%d %H:%M") + zone


def _file_name(path_text: str) -> str:
    """The last component of a recorded path, for Windows and POSIX spellings alike."""
    parts = [part for part in re.split(r"[\\/]+", path_text.strip()) if part]
    return parts[-1] if parts else path_text


def _run_key(value: Any) -> str:
    """The run identity the Studio project stores (its slug), for matching runs."""
    from darsia.fluidflower.project import slugify

    return slugify(str(value or ""), fallback="")


def _unknown_run(run: str, reason: str) -> OverflowRun:
    """A run the records say nothing about: '—' in every cell, the reason as tooltip."""
    cell = OverflowCell(UNKNOWN, reason)
    return OverflowRun(run, cell, cell, cell)


def _local_path_note(path_text: str, remap) -> str:
    """Where this machine reads a recorded path when a storage alias moves it."""
    moved = remap(path_text)
    return f"\nOn this machine: {moved}" if moved != path_text else ""


def _file_cell(mode: str, row: Mapping[str, Any], remap) -> OverflowCell:
    """The recorded table or scale-image folders of one run."""
    if mode == "table":
        path = str(row.get("table_path") or "").strip()
        if not path:
            return OverflowCell(UNKNOWN, "The saved revision records no table for this run.")
        return OverflowCell(_file_name(path), "Recorded: " + path + _local_path_note(path, remap))
    if mode == "scale_images":
        folders = _strings(row.get("image_folders"))
        if not folders:
            return OverflowCell(
                UNKNOWN, "The saved revision records no scale-image folder for this run."
            )
        more = f" (+{len(folders) - 1} more)" if len(folders) > 1 else ""
        lines = ["Recorded folders:"]
        for folder in folders:
            lines.append(folder + _local_path_note(folder, remap).replace("\n", "\n  "))
        return OverflowCell(_file_name(folders[0]) + more, "\n".join(lines))
    return OverflowCell(UNKNOWN, SOURCE_NOTES["none"])


def _review_parts(review: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Status text and tooltip lines for a run's scale-reader review file, if any."""
    if not review.get("available"):
        return [], []
    if review.get("error"):
        return ["scale review unreadable"], [
            "Scale-reader review could not be read: " + str(review["error"])
        ]
    reviewed = _int(review.get("reviewed_count")) or 0
    lines = [
        f"Scale-reader review: {_plural(reviewed, 'image')} reviewed manually",
        f"Exact readings (verified or corrected): {_int(review.get('exact_count')) or 0}",
        f"Uncertain readings: {_int(review.get('partial_count')) or 0}",
        f"Unreadable images: {_int(review.get('unreadable_count')) or 0}",
    ]
    first, last = review.get("first_time_h"), review.get("last_time_h")
    if isinstance(first, (int, float)) and isinstance(last, (int, float)):
        lines.append(f"Reviewed times: {float(first):.2f} to {float(last):.2f} h")
    if review.get("updated_at"):
        lines.append(f"Review last saved: {review['updated_at']}")
    return [f"{_plural(reviewed, 'scale image')} reviewed"], lines


def _status_cell(
    mode: str,
    row: Mapping[str, Any],
    gate: Mapping[str, Any],
    review: Mapping[str, Any],
    revision: str,
) -> OverflowCell:
    """Scan state, size, warnings, scale review and step status of one run."""
    parts: list[str] = []
    tips: list[str] = []
    scan = row.get("scan") if isinstance(row.get("scan"), Mapping) else {}
    if mode == "none":
        parts.append("Not used")
        tips.append("The saved revision records no overflow measurements for this run.")
    elif not scan:
        parts.append("Not scanned")
        tips.append("The saved revision holds no scan of this source.")
    else:
        state = str(scan.get("status") or "").strip()
        parts.append(SCAN_LABELS.get(state, state.replace("_", " ") or UNKNOWN))
        when = _utc_text(scan.get("scanned_utc"))
        tips.append(
            f"Scan when the revision was saved: {state or 'no status'}"
            + (f", {when}" if when else "")
        )
        count = _int(scan.get("row_count" if mode == "table" else "image_count"))
        if state == "ready" and count is not None:
            parts.append(_plural(count, "row" if mode == "table" else "image"))
        if mode == "table" and _strings(scan.get("sheets")):
            tips.append("Sheets: " + ", ".join(_strings(scan.get("sheets"))))
        errors, warnings = _strings(scan.get("errors")), _strings(scan.get("warnings"))
        tips.extend("Error: " + text for text in errors)
        tips.extend("Warning: " + text for text in warnings)
        if warnings:
            parts.append(_plural(len(warnings), "warning"))
    review_parts, review_tips = _review_parts(review)
    parts.extend(review_parts)
    tips.extend(review_tips)
    step = str(gate.get("status") or "").strip()
    if step:
        label = step.replace("_", " ")
        parts.append("step " + label)
        when = _utc_text(gate.get("updated_utc"), date_only=True)
        tips.append(
            f"Overflow measurements step in the Studio project: {label}"
            + (f" ({when})" if when else "")
        )
    tips.append("Recorded in " + revision + ".")
    return OverflowCell(" · ".join(parts), "\n".join(tips))


def studio_project_root(series_root: str | Path, manifest: Mapping[str, Any]) -> Path | None:
    """The Studio project folder that series.toml names in [legacy] studio_project.

    Only this explicit pointer is followed; nothing is discovered. It may name
    the project's series.json or its folder, relative to the series root, and
    the storage aliases apply. None when the manifest names no project.
    """
    legacy = manifest.get("legacy")
    pointer = legacy.get("studio_project") if isinstance(legacy, Mapping) else None
    if not isinstance(pointer, str) or not pointer.strip():
        return None
    from darsia.fluidflower.storage_paths import storage_path_remapper

    path = Path(storage_path_remapper()(pointer.strip()))
    path = path if path.is_absolute() else Path(series_root) / path
    if path.name.casefold() == "series.json":
        path = path.parent
    return path.resolve()


def _studio_overflow(series_root: Path, manifest: Mapping[str, Any]):
    """(project, saved revision or None, record path or None, draft exists), or None.

    Loads the named Studio project, checks that it belongs to this series and
    reads its saved overflow revision through the project's own store.
    """
    root = studio_project_root(series_root, manifest)
    if root is None:
        return None
    from darsia.fluidflower.overflow import OVERFLOW_REVISION_FORMAT, OverflowRevisionStore
    from darsia.fluidflower.project import FluidFlowerProject
    from darsia.fluidflower.series_paths import series_key

    project = FluidFlowerProject.load(root)
    if series_key(project.manifest.get("series_id", "")) != series_key(
        manifest.get("series_id", "")
    ):
        raise ValueError(f"the Studio project in {root} belongs to another series")
    store = OverflowRevisionStore(project)
    record = next(
        (path for path in (store.current_path, store.revisions_path) if path.is_file()),
        None,
    )
    saved = None
    if record is not None:
        # Without a saved file the store would fall back to setup discovery.
        current = store.current()
        if current.get("format") == OVERFLOW_REVISION_FORMAT:
            saved = current
    return project, saved, record, store.has_draft()


def _revision_rows(
    runs: Sequence[str], project, saved: Mapping[str, Any]
) -> tuple[list[OverflowRun], dict[str, int], list[str]]:
    """Table rows of all series runs from a saved revision, counts by source, extra runs."""
    from darsia.fluidflower.overflow_workflow import scale_reader_review_progress
    from darsia.fluidflower.storage_paths import storage_path_remapper

    remap = storage_path_remapper()
    configuration = saved.get("configuration")
    configuration = configuration if isinstance(configuration, Mapping) else {}
    decisions: dict[str, Mapping[str, Any]] = {}
    for row in configuration.get("runs") or []:
        if isinstance(row, Mapping) and _run_key(row.get("run_id")):
            decisions.setdefault(_run_key(row.get("run_id")), row)
    gates: dict[str, Mapping[str, Any]] = {}
    for row in project.manifest.get("runs") or []:
        workflow = row.get("workflow") if isinstance(row, Mapping) else None
        gate = workflow.get("overflow") if isinstance(workflow, Mapping) else None
        if isinstance(gate, Mapping):
            gates[_run_key(row.get("run_id"))] = gate
    identity = str(saved.get("revision_id") or "")
    when = _utc_text(saved.get("recorded_utc"), date_only=True)
    revision = f"overflow revision {identity[:8] or UNKNOWN}" + (f" saved {when}" if when else "")
    counts = {"table": 0, "scale_images": 0, "none": 0, "missing": 0}
    rows: list[OverflowRun] = []
    for run in runs:
        key = _run_key(run)
        row = decisions.get(key)
        if row is None:
            counts["missing"] += 1
            rows.append(_unknown_run(run, f"The saved {revision} has no decision for this run."))
            continue
        mode = str(row.get("mode") or "none").strip().casefold()
        if mode not in SOURCE_LABELS:
            counts["missing"] += 1
            rows.append(
                _unknown_run(run, f"The saved {revision} records an unsupported source '{mode}'.")
            )
            continue
        counts[mode] += 1
        assignment = ASSIGNMENT_LABELS.get(str(row.get("source_origin") or ""), "Saved selection")
        source = OverflowCell(
            SOURCE_LABELS[mode], f"{SOURCE_NOTES[mode]}\nAssignment: {assignment}"
        )
        review = scale_reader_review_progress(project.root, key)
        status = _status_cell(mode, row, gates.get(key, {}), review, revision)
        rows.append(OverflowRun(run, source, _file_cell(mode, row, remap), status))
    listed = {_run_key(run) for run in runs}
    extra = sorted(key for key in decisions if key not in listed)
    return rows, counts, extra


def read_overflow_data(series_root: str | Path | None) -> OverflowData:
    """Read every run's overflow source, file and status as the series records them.

    Only series.toml and the Studio project it names in [legacy]
    studio_project are read: that project's manifest, its saved overflow
    revision and its scale-reader review files. Nothing is written, no source
    file is opened, and no scan or discovery runs. A run the records say
    nothing about gets "—" with the reason in its tooltips.
    """
    if series_root is None:
        return OverflowData(summary="Choose a series on Start to see its overflow data.")
    root = Path(series_root)
    try:
        manifest = tomllib.loads((root / "series.toml").read_bytes().decode("utf-8-sig"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        return OverflowData(summary=f"The series manifest could not be read: {error}")
    table = manifest.get("runs")
    runs = _strings(table.get("ids")) if isinstance(table, Mapping) else []
    native = _configured_overflow(root, manifest, runs)
    if native is not None:
        return native
    try:
        found = _studio_overflow(root, manifest)
    except Exception as error:  # noqa: BLE001 - read-only display boundary
        reason = f"The Studio project that series.toml names could not be read: {error}"
        return OverflowData(tuple(_unknown_run(run, reason) for run in runs), reason)
    if found is None:
        entry = (
            "has an [overflow] entry in a form this page does not read"
            if "overflow" in manifest
            else "has no overflow entry"
        )
        reason = (
            "This series records no overflow data that DarSIA can read: series.toml "
            f"names no Studio project ([legacy] studio_project) and {entry}."
        )
        return OverflowData(tuple(_unknown_run(run, reason) for run in runs), reason)
    project, saved, record, draft = found
    detail = [f"Studio project: {project.root}"]
    setup = project.manifest.get("overflow")
    setup_mode = str(setup.get("mode") or "") if isinstance(setup, Mapping) else ""
    if setup_mode:
        detail.append(
            "Overflow mode chosen at setup: " + SETUP_MODE_LABELS.get(setup_mode, setup_mode)
        )
    step = str(project.module_states.get("overflow", {}).get("status") or "")
    if step:
        detail.append("Overflow measurements step: " + step.replace("_", " "))
    if draft:
        detail.append(
            "The Studio project also has an unsaved working draft; this table shows "
            "the saved revision only."
        )
    if saved is None:
        reason = "No overflow source is selected: the Studio project has no saved overflow revision."
        detail.append("This page reads these records only and writes nothing.")
        return OverflowData(
            tuple(_unknown_run(run, reason) for run in runs), reason, "\n".join(detail)
        )
    try:
        rows, counts, extra = _revision_rows(runs, project, saved)
    except Exception as error:  # noqa: BLE001 - read-only display boundary
        reason = f"The saved overflow revision could not be read: {error}"
        return OverflowData(
            tuple(_unknown_run(run, reason) for run in runs), reason, "\n".join(detail)
        )
    identity = str(saved.get("revision_id") or "")[:8] or UNKNOWN
    when = _utc_text(saved.get("recorded_utc"), date_only=True)
    groups = [
        (counts["table"], "with a table"),
        (counts["scale_images"], "with scale images"),
        (counts["none"], "with none"),
        (counts["missing"], "with no decision"),
    ]
    shown = [f"{_plural(count, 'run')} {text}" for count, text in groups if count]
    summary = (
        f"From the Studio project's saved overflow revision {identity}"
        + (f" ({when})" if when else "")
        + ": "
        + (", ".join(shown) if shown else "no runs")
        + "."
    )
    detail.insert(1, f"Record: {record}")
    if extra:
        detail.append(
            f"The revision also covers {_plural(len(extra), 'run')} that series.toml "
            "does not list: " + ", ".join(extra) + "."
        )
    detail.append("This page reads these records only and writes nothing.")
    return OverflowData(tuple(rows), summary, "\n".join(detail))


def _configured_overflow(root: Path, manifest: Mapping, runs: Sequence[str]):
    """Show direct protocol declarations before consulting older project records.

    This reads configuration only. A configured table is not described as a
    reviewed or calibrated gas target, and its measurement file is not opened.
    """
    from darsia.fluidflower.series_paths import series_folder_path
    from darsia.presets.workflows.config.protocols import OverflowProtocolConfig
    from darsia.presets.workflows.config.utils import _get_section_from_toml

    run_table = manifest.get("runs", {})
    names = run_table.get("config_files", {})
    common = series_folder_path(root, "common_config")
    directory = series_folder_path(root, "run_configs")
    rows, configured = [], 0
    for run in runs:
        path = Path(names[run]) if run in names else directory / f"{run}.toml"
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            rows.append(_unknown_run(run, "No saved run configuration is available."))
            continue
        try:
            files = [common, path] if common.is_file() else [path]
            protocol = OverflowProtocolConfig()
            protocol.load_overflow(_get_section_from_toml(files, "protocols"), files)
        except (OSError, ValueError, TypeError, KeyError) as error:
            rows.append(_unknown_run(run, f"Could not read the overflow declaration: {error}"))
            continue
        value = protocol.overflow
        if value is None:
            rows.append(OverflowRun(run, OverflowCell("None"), OverflowCell(UNKNOWN),
                                    OverflowCell("Not configured")))
            continue
        configured += 1
        source, sheet = value if isinstance(value, tuple) else (value, None)
        note = str(source) + (f"\nWorksheet: {sheet}" if sheet is not None else "")
        present = source.is_file()
        rows.append(OverflowRun(
            run, OverflowCell("Protocol table", "Selected in [protocols].overflow."),
            OverflowCell(source.name + (f" [{sheet}]" if sheet is not None else ""), note),
            OverflowCell("Configured" if present else "Unavailable",
                         "Measurement values are not read or assessed on this page.\n" + note),
        ))
    if not configured:
        return None
    return OverflowData(
        tuple(rows), f"{configured} of {len(runs)} runs have a directly configured overflow protocol.",
        "Files and worksheets come from the same [protocols].overflow declaration used by "
        "the experiment reader. Configured does not mean converted to gas volume.",
    )


class OverflowDataPage(QWidget):
    """Setup > Overflow data: all runs of the open series in one read-only table.

    The table holds the main content and takes the free height; the page has
    no scroll area of its own. The header line above names the whole series.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("nativeOverflowDataPage")
        self.series_root: Path | None = None
        self.data = OverflowData()
        layout = QVBoxLayout(self)
        self.title = QLabel("Overflow data")
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        layout.addWidget(self.title)
        self.summary = QLabel()
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)
        self.detail = QLabel()
        self.detail.setObjectName("secondaryText")
        self.detail.setTextFormat(Qt.TextFormat.PlainText)
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.detail)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)
        self.note = QLabel(ATTACH_NOTE)
        self.note.setObjectName("secondaryText")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        buttons = QHBoxLayout()
        self.reload_button = QPushButton("Reload")
        self.reload_button.setToolTip("Read the series records again; nothing is written.")
        self.reload_button.clicked.connect(self.reload)
        buttons.addWidget(self.reload_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.set_data(read_overflow_data(None))

    def show_series(self, series_root: str | Path | None) -> OverflowData:
        """Read the series' records and show them; returns what is shown."""
        self.series_root = Path(series_root) if series_root is not None else None
        data = read_overflow_data(self.series_root)
        self.set_data(data)
        return data

    def reload(self) -> OverflowData:
        """Read the shown series' records again."""
        return self.show_series(self.series_root)

    def set_data(self, data: OverflowData) -> None:
        """Render already read records; this method performs no I/O."""
        self.data = data
        self.summary.setText(data.summary)
        self.detail.setText(data.detail)
        self.detail.setVisible(bool(data.detail))
        self.reload_button.setEnabled(self.series_root is not None)
        self.table.setRowCount(len(data.runs))
        for index, run in enumerate(data.runs):
            cells = (OverflowCell(run.run, run.run), run.source, run.file, run.status)
            for column, cell in enumerate(cells):
                item = QTableWidgetItem(cell.text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(cell.tip or cell.text)
                self.table.setItem(index, column, item)
        self.table.resizeColumnsToContents()
