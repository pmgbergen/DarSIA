"""Calibration review's run list: every run of the series, its role and review status.

Roles come from what the series records: the referenced mass calibration's
``calibration_runs``, ``heldout_runs``, ``descriptive_runs`` and
``excluded_runs`` (fit, held-out, descriptive and excluded), and the series'
own ``[runs] excluded`` table in ``series.toml``. A run that neither names has
no role ("—"). The review status is the approval record that
``calibration_approvals`` keeps for the referenced calibration. Nothing here
writes; approving stays the page's explicit button.

``SliderPlayer`` steps an image slider with a Qt timer in the GUI thread, for
the viewer's Play toggle.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import NamedTuple

from PySide6.QtCore import QObject, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

NO_ROLE = "—"
FIT, HELD_OUT, DESCRIPTIVE, EXCLUDED = "fit", "held-out", "descriptive", "excluded"
#: Calibration document lists that name run roles, in the order they are read.
ROLE_LISTS = (
    ("calibration_runs", FIT),
    ("heldout_runs", HELD_OUT),
    ("descriptive_runs", DESCRIPTIVE),
    ("excluded_runs", EXCLUDED),
)
APPROVED, NOT_APPROVED, NO_REVIEW = "Approved", "Not approved", "—"
MAX_CALIBRATION_BYTES = 1024 * 1024
#: Milliseconds between two images while Play runs.
PLAY_INTERVAL_MS = 600
NO_RESULT = "No saved result of this run to view."
_DETAIL_ROLE = Qt.ItemDataRole.UserRole + 1


class ReviewRunRow(NamedTuple):
    """One run of the list, as display text."""

    run: str
    role: str
    review: str
    detail: str


class ReviewRuns(NamedTuple):
    """Every run of the series in ``series.toml`` order, and one summary note."""

    rows: tuple[ReviewRunRow, ...]
    note: str


def read_series_runs(series_root) -> tuple[tuple[str, ...], dict[str, str]]:
    """The ordered run ids of ``series.toml`` and its ``[runs] excluded`` runs.

    ``excluded`` maps each excluded run (as written) to its recorded reason; a
    list of runs gives empty reasons. Raises OSError or ValueError when the
    manifest cannot be read.
    """
    manifest = tomllib.loads((Path(series_root) / "series.toml").read_bytes().decode("utf-8-sig"))
    runs = manifest.get("runs") if isinstance(manifest.get("runs"), Mapping) else {}
    ids = runs.get("ids")
    if not isinstance(ids, list) or not all(isinstance(run, str) for run in ids):
        raise ValueError("series.toml has no list of run ids.")
    excluded = runs.get("excluded") or {}
    if isinstance(excluded, Mapping):
        excluded = {str(run): str(reason) for run, reason in excluded.items()}
    elif isinstance(excluded, list):
        excluded = {str(run): "" for run in excluded}
    else:
        raise ValueError("series.toml [runs] excluded is neither a table nor a list.")
    return tuple(ids), excluded


def calibration_roles(document: Mapping) -> dict[str, str]:
    """Run (case-folded) to role, from a calibration document's run lists."""
    roles: dict[str, str] = {}
    for key, role in ROLE_LISTS:
        values = document.get(key)
        if isinstance(values, list):
            for run in values:
                roles.setdefault(str(run).casefold(), role)
    return roles


def _calibration_document(series_root, reference) -> tuple[Path, dict]:
    path = Path(reference)
    if not path.is_absolute():
        path = Path(series_root) / path
    from .storage_paths import resolve_storage_path
    path = resolve_storage_path(path, series_roots=(Path(series_root),))
    with path.open("rb") as stream:
        raw = stream.read(MAX_CALIBRATION_BYTES + 1)
    if len(raw) > MAX_CALIBRATION_BYTES:
        raise ValueError(f"The calibration exceeds {MAX_CALIBRATION_BYTES} bytes: {path}")
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a saved JSON object: {path}")
    return path, document


def review_run_rows(series_root, *, roles_only=False) -> ReviewRuns:
    """Every run of the series with its role and review status, read only.

    The referenced calibration is the Phase-map method's mass calibration in
    ``series.toml``. Unreadable inputs are named in the note; the affected
    column then shows "—" rather than a guess.
    """
    from darsia.fluidflower.calibration_approvals import (
        calibration_identity,
        read_run_approvals,
        run_approval,
    )
    from darsia.fluidflower.series_method import read_series_method

    root = Path(series_root)
    try:
        runs, excluded = read_series_runs(root)
    except (OSError, ValueError, UnicodeError) as error:
        return ReviewRuns((), f"The series' runs could not be read: {error}")
    notes, roles, sources, sha, records = [], {}, [], None, []
    reference = read_series_method(root).mass_calibration
    historical_roles = False
    if not reference and roles_only:
        from .series_method import calibration_ancestry_reference
        try:
            manifest = tomllib.loads((root / "series.toml").read_text(encoding="utf-8-sig"))
            reference = calibration_ancestry_reference(manifest.get("method", {}).get("phase_model", {}))
            historical_roles = bool(reference)
        except (OSError, ValueError, UnicodeError) as error:
            notes.append(f"The retained calibration roles could not be read: {error}")
    if reference:
        try:
            path, document = _calibration_document(root, reference)
            roles = calibration_roles(document)
            if roles:
                sources.append(path.name)
            if not historical_roles:
                sha = calibration_identity(root, reference)["sha256"]
            else:
                notes.append("Run roles retained from the previous calibration; no active mass calibration is selected.")
        except (OSError, ValueError, TypeError, UnicodeError) as error:
            notes.append(f"The referenced calibration could not be read: {error}")
        if sha is not None:
            try:
                records = read_run_approvals(root)
            except (OSError, ValueError, TypeError, UnicodeError) as error:
                # A damaged record is never shown as "Not approved".
                sha = None
                notes.append(f"The approval records could not be read: {error}")
    else:
        notes.append("No mass calibration is referenced: only excluded runs have a role, "
                     "and runs cannot be approved.")
    if excluded:
        sources.append("series.toml [runs] excluded")
    excluded_folded = {run.casefold(): reason for run, reason in excluded.items()}
    rows = []
    for run in runs:
        role = roles.get(run.casefold())
        lines = []
        if run.casefold() in excluded_folded:
            reason = excluded_folded[run.casefold()]
            lines.append("series.toml excludes this run" + (f": {reason}" if reason else "."))
            role = role or EXCLUDED
        if sha is None:
            review = NO_REVIEW
        else:
            record = run_approval(records, run, sha)
            if record is None:
                review = NOT_APPROVED
                lines.append("Not approved for the referenced calibration.")
            else:
                review = APPROVED
                by = f" by {record['reviewer']}" if record.get("reviewer") else ""
                lines.append(f"Approved{by} at {record.get('recorded_utc')} "
                             "for the referenced calibration.")
        rows.append(ReviewRunRow(run, role or NO_ROLE, review, "\n".join(lines)))
    counts = [f"{sum(row.role == role for row in rows)} {role}"
              for role in (FIT, HELD_OUT, DESCRIPTIVE, EXCLUDED)
              if any(row.role == role for row in rows)]
    summary = " · ".join(counts or ["No roles recorded"])
    if sha is not None:
        summary += f" · {sum(row.review == APPROVED for row in rows)} of {len(rows)} approved"
    summary += "."
    if sources:
        summary += " Roles from " + " and ".join(sources) + "."
    return ReviewRuns(tuple(rows), " ".join([summary, *notes]))


class CalibrationReviewRunList(QWidget):
    """All runs of the series beside the viewer; choosing one asks to view it.

    Rows of runs without a saved result to view are shown but cannot be
    chosen. The list never writes and never loads a result itself.
    """

    run_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel("Runs")
        self.title.setWordWrap(True)
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        layout.addWidget(self.title)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(("Run", "Role", "Review"))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.table, 1)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.note)
        self.setMinimumWidth(200)
        self.setMaximumWidth(320)
        self._available: set[str] | None = None

    def set_rows(self, runs: ReviewRuns) -> None:
        """Show already read rows; keeps the chosen run. No I/O."""
        selected = self.selected_run()
        with QSignalBlocker(self.table):
            self.table.setRowCount(len(runs.rows))
            for index, row in enumerate(runs.rows):
                for column, text in enumerate((row.run.upper(), row.role, row.review)):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.ItemDataRole.UserRole, row.run)
                    item.setData(_DETAIL_ROLE, row.detail or text)
                    item.setToolTip(row.detail or text)
                    self.table.setItem(index, column, item)
            self.table.resizeColumnsToContents()
        self.title.setText(f"Runs — whole series ({len(runs.rows)})")
        self.note.setText(runs.note)
        self.set_available(self._available)
        self.select_run(selected)

    def runs(self) -> list[str]:
        """The listed run ids, in list order."""
        return [self.table.item(index, 0).data(Qt.ItemDataRole.UserRole)
                for index in range(self.table.rowCount())]

    def set_available(self, runs: Iterable[str] | None) -> None:
        """Runs that have a saved result to view; None makes every row choosable."""
        self._available = None if runs is None else {str(run).casefold() for run in runs}
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        with QSignalBlocker(self.table):
            for index in range(self.table.rowCount()):
                run = self.table.item(index, 0).data(Qt.ItemDataRole.UserRole)
                usable = self._available is None or str(run).casefold() in self._available
                for column in range(self.table.columnCount()):
                    item = self.table.item(index, column)
                    item.setFlags(flags if usable else Qt.ItemFlag.NoItemFlags)
                    detail = item.data(_DETAIL_ROLE)
                    item.setToolTip(detail if usable else NO_RESULT + "\n" + detail)

    def selected_run(self) -> str | None:
        """The chosen run id, or None."""
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        return self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)

    def select_run(self, run) -> None:
        """Mark ``run`` as chosen without asking the viewer to load it."""
        with QSignalBlocker(self.table):
            self.table.clearSelection()
            if run is None:
                return
            for index in range(self.table.rowCount()):
                item = self.table.item(index, 0)
                if (str(item.data(Qt.ItemDataRole.UserRole)).casefold() == str(run).casefold()
                        and item.flags() & Qt.ItemFlag.ItemIsSelectable):
                    self.table.selectRow(index)
                    self.table.scrollToItem(item)
                    return

    def _selection_changed(self) -> None:
        run = self.selected_run()
        if run is not None:
            self.run_selected.emit(run)


def _layout_holding(layout, widget):
    if layout is None:
        return None
    if layout.indexOf(widget) >= 0:
        return layout
    for index in range(layout.count()):
        found = _layout_holding(layout.itemAt(index).layout(), widget)
        if found is not None:
            return found
    return None


def insert_after(anchor: QWidget, widget: QWidget) -> bool:
    """Put ``widget`` right after ``anchor`` in the box layout that holds ``anchor``.

    Returns False, and changes nothing, when no layout of the anchor's parent
    holds it.
    """
    parent = anchor.parentWidget()
    layout = _layout_holding(parent.layout() if parent is not None else None, anchor)
    if layout is None or not hasattr(layout, "insertWidget"):
        return False
    layout.insertWidget(layout.indexOf(anchor) + 1, widget)
    return True


class SliderPlayer(QObject):
    """A checkable Play button that steps a slider with a QTimer, stopping at its end.

    No thread is used: one step per timer tick in the GUI thread. Playing
    from the last image starts again at the first. Hidden or disabled
    sliders stop the player.
    """

    def __init__(self, slider, button, *, interval_ms: int = PLAY_INTERVAL_MS, parent=None):
        super().__init__(parent)
        self.slider = slider
        self.button = button
        button.setCheckable(True)
        button.setText("Play")
        self.timer = QTimer(self)
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self._step)
        button.toggled.connect(self._toggled)

    def playing(self) -> bool:
        """Whether the timer is stepping the slider."""
        return self.timer.isActive()

    def stop(self) -> None:
        """Stop playing; the slider stays where it is."""
        self.button.setChecked(False)

    def _can_step(self) -> bool:
        slider = self.slider
        return slider.isEnabled() and slider.isVisible() and slider.maximum() > slider.minimum()

    def _toggled(self, checked: bool) -> None:
        if checked and not self._can_step():
            self.button.setChecked(False)
            return
        if checked:
            if self.slider.value() >= self.slider.maximum():
                self.slider.setValue(self.slider.minimum())
            self.button.setText("Pause")
            self.timer.start()
        else:
            self.timer.stop()
            self.button.setText("Play")

    def _step(self) -> None:
        if not self._can_step() or self.slider.value() >= self.slider.maximum():
            self.stop()
            return
        self.slider.setValue(self.slider.value() + 1)
        if self.slider.value() >= self.slider.maximum():
            self.stop()
