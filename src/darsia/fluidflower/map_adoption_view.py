"""Adopt a prepared painted map as a geometry revision, with its facies table.

An approved-session preparation derives one row per used painting facies,
with all extra current CSV cells carried by name and conflicts refused.
Legacy preparations retain their explicit label-to-series-facies choices.
"Check" builds the revision candidate and validates it as the Store will;
"Save revision…" asks once and then saves it with the existing geometry
revision store. Both read every run's files and run in a worker thread,
never on the GUI thread. The new revision is appended; the previous one
stays.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import json
import time
from typing import Any

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower import map_adoption as ma

_COLUMNS = ("Label", "Layer", "Facies in the map", "Role", "Facies id in the series' table")


def _check_messages(report):
    """Show every original issue with the run/source context that distinguishes it."""
    lines = []
    for row in (report or {}).get("issues") or []:
        label = str(row.get("severity") or "message").capitalize()
        code = f" [{row['code']}]" if row.get("code") else ""
        lines.append(f"{label}{code}: {row.get('message') or ''}")
        if row.get("details"):
            lines.append("  " + json.dumps(row["details"], ensure_ascii=False, sort_keys=True))
    return "\n".join(lines)


class _CheckWorker(QObject):
    """Build the revision candidate and validate it as the Store will."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, current: Mapping[str, Any], folder: Path, facies_ids: Mapping[int, int] | None,
                 *, facies_source_acquisition: Mapping[str, Any] | None = None,
                 facies_source_choice: Mapping[str, Any] | None = None,
                 display_preferences: Mapping[str, Any] | None = None) -> None:
        """Retain the current revision, preparation and optional explicit facies choice."""
        super().__init__()
        self.current, self.folder = deepcopy(dict(current)), Path(folder)
        self.facies_ids = None if facies_ids is None else dict(facies_ids)
        self._facies_source_acquisition = (None if facies_source_acquisition is None else deepcopy(dict(facies_source_acquisition)))
        self._facies_source_choice = deepcopy(facies_source_choice)
        self._display_preferences = deepcopy(display_preferences)

    @Slot()
    def run(self) -> None:
        """Build and validate a candidate off the GUI thread, reporting its result or error."""
        from .adoption_check_progress import observe_check_progress

        with observe_check_progress(self.progress.emit):
            self._run_check()

    def _run_check(self) -> None:
        try:
            from darsia.fluidflower.geometry_facies import validate_geometry_config
            from .adoption_check_progress import report_check_progress

            start = time.monotonic()
            options = ({} if self._facies_source_acquisition is None else
                       dict(facies_source_acquisition=self._facies_source_acquisition))
            if self._facies_source_choice is not None:
                report_check_progress("Verifying the selected facies-source acquisition bindings…")
                from .painted_analysis_exclusions import make_facies_source_acquisition
                action = deepcopy(self._facies_source_choice)
                state = action.pop("acquisition_state")
                options["facies_source_acquisition"] = make_facies_source_acquisition(self.folder, state, **action)
            candidate, summary = ma.adoption_candidate(self.current, self.folder, self.facies_ids, **options)
            check_options = {}
            preferences_error = None
            if self._display_preferences is not None:
                report_check_progress("Verifying display preferences against the saved preparation review…")
                try:
                    from .painted_adoption_table import _checked
                    from .painted_review_comparison import make_display_attachments, read_review
                    preparation = _checked(candidate['provenance']['preparation'], metadata=True)
                    review = read_review(preparation['gap_review']['path'])
                    if review['review_binding'] != preparation['gap_review']:
                        raise ValueError('Display preferences belong to another actual preparation review')
                    check_options['display_attachments'] = make_display_attachments(review, **self._display_preferences)
                except Exception as error:  # Preserve the genuine backend's authenticated original issues.
                    preferences_error = f'{type(error).__name__}: {error}'
            checked = None
            report_check_progress("Validating the complete geometry revision…")
            from .manual_map_approval import approved_candidate
            if ((candidate.get("provenance") or {}).get("local_map_reuse")
                    or approved_candidate(candidate)):
                from .local_geometry_check import check_geometry_candidate
                checked = check_geometry_candidate(candidate, **check_options)
                report = checked.report_for(candidate)
            else:
                report = validate_geometry_config(candidate, **check_options)
            if preferences_error is not None:
                report = deepcopy(report)
                report.setdefault('issues', []).append(dict(severity='error', code='painted_review_display_preferences_invalid',
                    message='The display preferences could not be checked; saving is refused. ' + preferences_error))
                report.setdefault('summary', {})['error_count'] = sum(issue.get('severity') == 'error' for issue in report['issues'])
            errors = [
                str(issue.get("message")) for issue in report.get("issues") or []
                if issue.get("severity") == "error"
            ]
            result = {
                "candidate": candidate,
                "local_checked": checked,
                "summary": summary,
                "errors": errors,
                "validation_report": report,
                "seconds": time.monotonic() - start,
                **({"facies_source_acquisition": deepcopy(options["facies_source_acquisition"])}
                   if self._facies_source_choice is not None else {}),
            }
        except Exception as error:  # noqa: BLE001 - the owner must retire this thread
            try:
                from .painted_analysis_exclusions import DisplayFindingsChanged
            except ImportError:
                DisplayFindingsChanged = ()
            if isinstance(error, DisplayFindingsChanged):
                issues = deepcopy(error.original_issues)
                issues.append(dict(severity='error', code='painted_review_display_context_changed',
                    message='The candidate was not created and geometry validation did not run; saving is refused. ' + str(error)))
                self.finished.emit(dict(candidate=None, summary=None,
                    validation_report=dict(issues=issues, validation_not_executed=True,
                        summary=dict(error_count=sum(issue.get('severity') == 'error' for issue in issues),
                                     warning_count=sum(issue.get('severity') == 'warning' for issue in issues))),
                    validation_not_executed=True,
                    errors=[str(issue.get('message')) for issue in issues if issue.get('severity') == 'error'],
                    seconds=time.monotonic() - start))
                return
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.finished.emit(result)


class _SaveRevisionWorker(QObject):
    """Save a geometry revision candidate with the existing Store."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, project: Any, candidate: Mapping[str, Any], actor: str, reason: str,
                 *, validation_report=None, review_resolutions=None, display_attachments=None,
                 local_checked=None) -> None:
        super().__init__()
        self.project, self.candidate, self.actor, self.reason = project, deepcopy(dict(candidate)), actor, reason
        self.validation_report, self.review_resolutions = deepcopy(validation_report), deepcopy(review_resolutions)
        self.display_attachments = deepcopy(display_attachments)
        self.local_checked = local_checked

    @Slot()
    def run(self) -> None:
        """Append the confirmed candidate through the existing geometry revision Store."""
        try:
            from darsia.fluidflower.geometry_facies import GeometryRevisionStore

            options = {"preserve_failed_artifacts": True}
            if self.local_checked is not None:
                options["checked_candidate"] = self.local_checked
            if self.review_resolutions is not None:
                from .painted_analysis_exclusions import seal_findings_resolutions
                options['review_resolutions'] = seal_findings_resolutions(
                    self.candidate, self.validation_report, self.review_resolutions,
                    **({} if self.display_attachments is None else dict(display_attachments=self.display_attachments)))
            revision = GeometryRevisionStore(self.project).save_revision(
                self.candidate, actor=self.actor, reason=self.reason, **options
            )
        except Exception as error:  # noqa: BLE001 - shown to the user
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.finished.emit(revision)


class AdoptionDialog(QDialog):
    """The facies table of a prepared map, its check and its explicit save."""

    operation_retired = Signal(str)
    confirmation_decided = Signal(bool)

    def __init__(
        self,
        project: Any,
        folder: Path | str,
        *,
        actor: str,
        parent: QWidget | None = None,
        automatic_only: bool = False,
        adoption_scope_note: str = "",
        facies_source_acquisition: Mapping[str, Any] | None = None,
        findings_managed: bool = False,
        preparation_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Show the preparation table, validation result and explicit revision-save controls."""
        super().__init__(parent)
        from darsia.fluidflower.geometry_facies import GeometryRevisionStore

        self.project = project
        self.folder = Path(folder)
        self.actor = str(actor)
        self.adoption_scope_note = str(adoption_scope_note)
        self._facies_source_acquisition = (None if facies_source_acquisition is None else deepcopy(dict(facies_source_acquisition)))
        self._facies_source_choice = None
        self.findings_managed = bool(findings_managed)
        self.review_resolutions = None
        self._display_preferences = None
        self.display_attachments = None
        self.checked_candidate = None
        # Metadata may come from the controller's authenticated saved index.
        # The owned Check still performs the full ordinary product admission.
        self.preparation = (ma.read_preparation(self.folder) if preparation_metadata is None
                            else deepcopy(dict(preparation_metadata)))
        self.current = GeometryRevisionStore(project).current()
        self.automatic = "painting_facies" in self.preparation
        if automatic_only and not self.automatic:
            raise ValueError("This action requires an approved-painting preparation; manual facies remapping is not a fallback")
        if self.automatic:
            binding = self.preparation["painting_facies"]
            snapshot = binding.get("facies_snapshot") if isinstance(binding, dict) else None
            if (not isinstance(snapshot, dict) or not isinstance(snapshot.get("layers"), list)
                    or not isinstance(snapshot.get("facies"), list)
                    or any(not isinstance(row, dict) or not {"layer_id", "name", "facies"} <= row.keys()
                           for row in snapshot["layers"])
                    or any(not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str)
                           or not isinstance(row[1], dict) or not {"role", "porosity", "permeability"} <= row[1].keys()
                           for row in snapshot["facies"])):
                raise ValueError("The automatic painting preparation has no readable facies snapshot; no manual fallback is offered")
        self.table_rows = [] if self.automatic else ma.facies_table(self.current.get("configuration") or {})
        self.candidate: dict[str, Any] | None = None
        self.validation_report: dict[str, Any] | None = None
        self.revision: dict[str, Any] | None = None
        self._thread: QThread | None = None
        self._worker: _SaveRevisionWorker | None = None
        self._operation = None
        self._waiting_for_workers = False
        self._embedded = False
        self._confirmation_pending = False
        self._check_progress_active = False
        self._check_progress_timer = QTimer(self)
        self._check_progress_timer.setInterval(1000)
        self._check_progress_timer.timeout.connect(self._refresh_check_progress)
        self.setWindowTitle("Adopt the prepared map")
        self.resize(900, 640)
        layout = QVBoxLayout(self)
        version = dict(self.preparation.get("map_version") or {})
        prepared = [row for row in self.preparation.get("runs") or [] if row.get("status") == "prepared"]
        instruction = (
            "Facies IDs come automatically from the approved saved painting, one row per used facies. "
            "Extra CSV cells come from the current table by exact name; conflicts refuse and are shown. "
            if self.automatic else
            "Give each label the facies id it has in the series' facies table; nothing is chosen for you. "
        )
        if self.automatic:
            from .painted_adoption_table import frame_basis_text
            instruction += frame_basis_text(self.preparation["painting_facies"]) + " Check verifies these bindings. "
        note = QLabel(self.adoption_scope_note + ("\n\n" if self.adoption_scope_note else "") +
            f"Map {version.get('name') or ''} (SHA-256 {str(version.get('sha256') or '')[:12]}…), "
            f"prepared on the label grid of {len(prepared)} of "
            f"{len(self.preparation.get('runs') or [])} runs. " + instruction + "'Check' builds the "
            "new geometry revision and validates it as the store will. 'Save revision…' then "
            f"appends it after revision {str(self.current.get('revision_id') or '')[:8]}, which "
            "stays. Finish setup then applies it to the runs and builds their run models. "
            f"Preparation: {self.folder}"
        )
        note.setWordWrap(True)
        if self.preparation.get("format") == ma.PREPARATION_FORMAT_V2:
            continued = sum(int((row.get("continuation") or {}).get("continued_pixels", 0)) for row in prepared)
            gaps = sum(int((row.get("continuation") or {}).get("inside_gap_pixels", 0)) for row in prepared)
            note.setText(note.text() + f" New photo-edge continuation contract: {continued:,} continued pixels, {gaps:,} true inside-frame gaps. "
                         "Original filled/support flags are retained; separate effective coverage includes only authorized continuation. "
                         "This preparation grants no adoption or optical support.")
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(note)
        self.values = ma.label_values(self.preparation)
        regions = {int(row["layer_id"]): row for row in self.preparation.get("regions") or []}
        auto_rows = []
        if self.automatic:
            snapshot = self.preparation["painting_facies"]["facies_snapshot"]
            used = {row["facies"] for row in snapshot["layers"] if row["layer_id"] in self.values}
            for name, entry in snapshot["facies"]:
                if name in used:
                    layers = ", ".join(str(row["layer_id"]) for row in snapshot["layers"]
                                       if row["layer_id"] in self.values and row["facies"] == name)
                    auto_rows.append((name, layers, str(entry["role"]), str(entry["porosity"]),
                                      str(entry["permeability"]), "Verified during Check"))
        headings = ["Facies", "Layers", "Role", "Porosity", "Permeability", "Extra properties"] if self.automatic else list(_COLUMNS)
        self.table = QTableWidget(len(auto_rows) if self.automatic else len(self.values), len(headings))
        self.table.setHorizontalHeaderLabels(headings)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.choices: dict[int, QComboBox] = {}
        for number, row in enumerate(auto_rows):
            for column, text in enumerate(row):
                self.table.setItem(number, column, QTableWidgetItem(text))
        for number, value in enumerate(self.values if not self.automatic else ()):
            region = regions.get(value) or {}
            cells = (
                str(value),
                str(region.get("name") or ("Outside the map" if value == 0 else "")),
                str(region.get("facies") or ""),
                str(region.get("role") or ("excluded" if value == 0 else "")),
            )
            for column, text in enumerate(cells):
                self.table.setItem(number, column, QTableWidgetItem(text))
            choice = QComboBox()
            choice.addItem("Choose…", None)
            for row in self.table_rows:
                choice.addItem(
                    f"{row['facies_id']}: {row['name']} (porosity {row['porosity']}, "
                    f"permeability {row['permeability']})",
                    int(row["facies_id"]),
                )
            choice.currentIndexChanged.connect(self._choice_changed)
            self.table.setCellWidget(number, len(cells), choice)
            self.choices[value] = choice
        layout.addWidget(self.table, 1)
        self.status = QLabel("Check the automatic table and its full saved-source bindings." if self.automatic
                             else "Choose a facies id for every label, then Check.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.check_messages = QPlainTextEdit()
        self.check_messages.setReadOnly(True)
        self.check_messages.setAccessibleName("Check messages")
        self.check_messages.setMaximumHeight(150)
        self.check_messages.hide()
        layout.addWidget(self.check_messages)
        self.confirmation = QFrame(self)
        confirmation_layout = QVBoxLayout(self.confirmation)
        self.confirmation_text = QLabel()
        self.confirmation_text.setWordWrap(True)
        confirmation_layout.addWidget(self.confirmation_text)
        confirmation_buttons = QDialogButtonBox()
        self.confirm_save = confirmation_buttons.addButton('Save revision', QDialogButtonBox.ButtonRole.AcceptRole)
        self.confirm_cancel = confirmation_buttons.addButton('Cancel', QDialogButtonBox.ButtonRole.RejectRole)
        self.confirm_save.clicked.connect(lambda: self._inline_confirmation_decided(True))
        self.confirm_cancel.clicked.connect(lambda: self._inline_confirmation_decided(False))
        confirmation_layout.addWidget(confirmation_buttons)
        self.confirmation.hide()
        layout.addWidget(self.confirmation)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.close_button = buttons.button(QDialogButtonBox.StandardButton.Close)
        self.check_button = buttons.addButton("Check", QDialogButtonBox.ButtonRole.ActionRole)
        self.check_button.clicked.connect(lambda: self.check())
        self.save_button = buttons.addButton("Save revision…", QDialogButtonBox.ButtonRole.ActionRole)
        self.save_button.clicked.connect(self._save_clicked)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._choice_changed()
        if self.automatic:
            from .painted_adoption_table import saved_adoption_revision
            self.revision = saved_adoption_revision(self.current, self.folder)
            if self.revision is not None:
                note.setText(
                    f"Map {version.get('name') or ''} is already adopted as the current geometry "
                    f"revision {self.revision['revision_id']}. Its saved facies table and "
                    f"{len(prepared)} prepared runs are retained. "
                    "Continue with Image correction, Colour paths and Finish setup. "
                    f"Preparation: {self.folder}")
                self.status.setText(
                    f"Already adopted as geometry revision {self.revision['revision_id']}. "
                    "Apply and Build are the next steps; no new Check or Save is needed.")
                self._update_buttons()

    def set_embedded(self, parent: QWidget) -> None:
        """Use the same owned dialog lifecycle inside the main workspace."""
        self._embedded = True
        self.setParent(parent, Qt.WindowType.Widget)
        self.close_button.setText('Back to map')
        self._update_buttons()

    def facies_ids(self) -> dict[int, int] | None:
        """The facies id chosen for each label so far."""

        if self.automatic:
            return None
        return {
            value: int(choice.currentData())
            for value, choice in self.choices.items()
            if choice.currentData() is not None
        }

    def _choice_changed(self, *_args: Any) -> None:
        self.candidate = self.checked_candidate = None
        self._local_checked = None
        self.review_resolutions = None
        self.validation_report = None
        self.check_messages.clear()
        self.check_messages.hide()
        self._update_buttons()

    def _update_buttons(self) -> None:
        """Check needs every facies id; Save needs a checked candidate; neither
        runs while a worker does or after the revision is saved."""

        idle = self._thread is None and self.revision is None
        self.close_button.setEnabled(self._thread is None)
        self.check_button.setEnabled(idle and not self._confirmation_pending and (self.automatic or len(self.facies_ids()) == len(self.values)))
        self.save_button.setEnabled(idle and not self._confirmation_pending and self.candidate is not None
            and (not self.findings_managed or self.review_resolutions is not None))

    def check(self) -> bool:
        """Start building and validating the candidate in a worker thread;
        False when it cannot start. The status says the outcome."""

        if self._thread is not None or self.revision is not None:
            return False
        if not self.automatic and len(self.facies_ids()) != len(self.values):
            self.status.setText("Choose a facies id for every label first.")
            return False
        self.candidate = self.checked_candidate = None
        self._local_checked = None
        self.review_resolutions = None
        self.validation_report = None
        self.check_messages.clear()
        self.check_messages.hide()
        options = ({} if self._facies_source_acquisition is None else
                   dict(facies_source_acquisition=self._facies_source_acquisition))
        if self._facies_source_choice is not None:
            options["facies_source_choice"] = deepcopy(self._facies_source_choice)
        if self._display_preferences is not None:
            options['display_preferences'] = deepcopy(self._display_preferences)
        worker = _CheckWorker(self.current, self.folder, self.facies_ids(), **options)
        worker.finished.connect(self._checked)
        worker.failed.connect(self._check_failed)
        worker.progress.connect(self._check_progress)
        self._start(worker, "Reading and verifying the prepared map…")
        return True

    @Slot(str)
    def _check_progress(self, message: str) -> None:
        """Accept observations only from the currently owned, unfinished Check."""
        if not self._check_progress_active or self.sender() is not self._worker:
            return
        self._check_progress_message = message
        self._check_stage_started = time.monotonic()
        self._refresh_check_progress()

    @Slot()
    def _refresh_check_progress(self) -> None:
        if self._check_progress_active:
            now = time.monotonic()
            self.status.setText(self._check_progress_message +
                f"\nElapsed: {now - self._check_started:.0f} s; current stage: {now - self._check_stage_started:.0f} s.")

    def _stop_check_progress(self) -> None:
        self._check_progress_active = False
        self._check_progress_timer.stop()

    def _checked(self, result: object) -> None:
        """Keep a candidate that validates, or say why it does not."""

        result = dict(result)
        self._stop_check_progress()
        self.validation_report = result.get("validation_report")
        self.display_attachments = deepcopy((self.validation_report or {}).get('display_attachments'))
        self.checked_candidate = deepcopy(result.get("candidate"))
        self._local_checked = result.get("local_checked")
        if "facies_source_acquisition" in result:
            self._facies_source_acquisition = deepcopy(result["facies_source_acquisition"])
        empty = "No validation issues." if self.validation_report is not None else "No validation report is available."
        self.check_messages.setPlainText(_check_messages(self.validation_report) or empty)
        self.check_messages.show()
        if result["errors"]:
            self.status.setText("The revision does not validate:\n" + "\n".join(result["errors"]))
            return
        self.candidate = result["candidate"]
        summary = result["summary"]
        if self.automatic:
            table = summary["facies_table"]
            self.table.setColumnCount(len(table["columns"]))
            self.table.setHorizontalHeaderLabels(table["columns"])
            self.table.setRowCount(len(table["rows"]))
            for number, row in enumerate(table["rows"]):
                for column, name in enumerate(table["columns"]):
                    self.table.setItem(number, column, QTableWidgetItem(str(row[name])))
        left = summary["not_prepared"]
        self.status.setText(
            f"Ready to save: {len(summary['runs'])} runs and {len(summary['regions'])} regions "
            f"(checked in {result['seconds']:.0f} s)."
            + (" " + summary["frame_basis"] if self.automatic else "")
            + (f" Not prepared, so not in the revision: {', '.join(run.upper() for run in left)}." if left else "")
        )

    def choose_facies_source_state(self, action):
        """Accept only an explicit GUI action, then rerun the original Check."""
        if self.busy() or self.revision is not None:
            return False
        self._facies_source_choice = deepcopy(action)
        self.review_resolutions = None
        return self.check()

    def set_review_resolutions(self, records):
        """Retain human choices; the Save worker and Store independently validate."""
        self.review_resolutions = deepcopy(records)
        self._update_buttons()

    def set_display_preferences(self, value):
        """Snapshot human display choices; a changed choice needs the existing Check."""
        if self.busy() or self.revision is not None:
            return False
        owned = deepcopy(value)
        if self._display_preferences != owned:
            self._display_preferences = owned
            self.candidate = self.checked_candidate = self.review_resolutions = self.validation_report = None
            self.display_attachments = None
            self.check_messages.clear()
            self.check_messages.hide()
            self._update_buttons()
        return True

    def _check_failed(self, message: str) -> None:
        self._stop_check_progress()
        self.validation_report = None
        self.check_messages.clear()
        self.check_messages.hide()
        self.status.setText(f"The revision cannot be built: {message}")

    def _start(self, worker: QObject, message: str) -> None:
        """Run one worker in its own thread (0008 rule: freed in the GUI thread)."""

        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Direct: quit from the worker thread, so that wait() in the GUI
        # thread cannot block a queued quit.
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._thread_finished)
        self._operation = "check" if isinstance(worker, _CheckWorker) else "save"
        self._thread, self._worker = thread, worker
        if self._operation == "check":
            self._check_started = self._check_stage_started = time.monotonic()
            self._check_progress_message = message
            self._check_progress_active = True
            self._check_progress_timer.start()
        self._update_buttons()
        self.status.setText(message)
        thread.start()

    def _save_clicked(self) -> None:
        """Ask before appending the checked revision with its retained frame explanation."""
        if self.busy() or self._confirmation_pending or self.candidate is None or self.findings_managed and self.review_resolutions is None:
            return
        frame_note = ""
        if self.automatic:
            from .painted_adoption_table import frame_basis_text
            frame_note = " " + frame_basis_text(self.preparation["painting_facies"])
        messages = _check_messages(getattr(self, "validation_report", None))
        check_note = "\n\nCheck messages:\n" + messages if messages else ""
        prompt = (
            self.adoption_scope_note + ("\n\n" if self.adoption_scope_note else "") +
            "Append a new geometry revision that adopts this map in every prepared run? The "
            "current revision stays in the history. Run configurations do not change until "
            "Finish setup." + frame_note)
        if self._embedded:
            self._confirmation_pending = True
            # The complete report stays in the bounded, scrollable messages
            # above; repeating it in a label would push Save out of view.
            self.confirmation_text.setText(prompt)
            self.confirmation.show()
            self._update_buttons()
            self.confirm_save.setFocus()
            return
        answer = QMessageBox.question(self, "Save revision", prompt + check_note)
        accepted = answer == QMessageBox.StandardButton.Yes
        self.confirmation_decided.emit(accepted)
        if accepted:
            self.save()

    def _inline_confirmation_decided(self, accepted: bool) -> None:
        """Emit the original confirmation gate once, before any Store write."""
        if not self._confirmation_pending or self.busy():
            return
        self._confirmation_pending = False
        self.confirmation.hide()
        self.confirmation_decided.emit(accepted)
        self._update_buttons()
        if accepted:
            self.save()
        else:
            self.reject()

    def save(self) -> bool:
        """Save the checked candidate in a worker thread; False when there is none."""

        if (self.candidate is None or self._thread is not None
                or self.findings_managed and self.review_resolutions is None):
            return False
        version = dict(self.preparation.get("map_version") or {})
        reason = (
            f"adopt the approved painted map {version.get('name') or ''} "
            f"(SHA-256 {str(version.get('sha256') or '')[:12]}) in every prepared run"
        )
        options = ({} if self.review_resolutions is None else dict(
            validation_report=deepcopy(self.validation_report), review_resolutions=deepcopy(self.review_resolutions)))
        if self.display_attachments is not None:
            options['display_attachments'] = deepcopy(self.display_attachments)
        if getattr(self, "_local_checked", None) is not None:
            options["local_checked"] = self._local_checked
        worker = _SaveRevisionWorker(self.project, self.candidate, self.actor, reason, **options)
        worker.finished.connect(self._saved)
        worker.failed.connect(self._failed)
        self._start(worker, "Saving the checked revision…")
        return True

    def busy(self) -> bool:
        """Whether a check or a save is running."""

        return self._thread is not None

    def _saved(self, revision: object) -> None:
        self.revision = dict(revision)
        self.status.setText(
            f"Saved revision {self.revision['revision_id']}. Finish setup applies it to the runs "
            "and builds their run models."
        )

    def _failed(self, message: str) -> None:
        self.status.setText(f"The revision was not saved: {message}")

    @Slot()
    def _thread_finished(self, expected=None) -> None:
        """Retire only the matching native thread before allowing a continuation."""
        if expected is None and isinstance(self.sender(), QThread):
            expected = self.sender()
        if self._thread is None or expected is not None and self._thread is not expected:
            return
        thread, worker, operation = self._thread, self._worker, self._operation
        thread.wait()
        if operation == "check":
            self._stop_check_progress()
        self._thread = self._worker = self._operation = None
        del worker
        thread.deleteLater()
        self._update_buttons()
        if not self._waiting_for_workers:
            self.operation_retired.emit(operation)

    def wait_for_workers(self) -> None:
        """Naturally retire an owned worker without opening another operation."""
        self._waiting_for_workers = True
        try:
            if self._thread is not None:
                thread = self._thread
                thread.wait()
                self._thread_finished(thread)
        finally:
            self._waiting_for_workers = False

    def reject(self) -> None:
        if self._thread is None:
            self._confirmation_pending = False
            self.confirmation.hide()
            super().reject()

    def closeEvent(self, event) -> None:
        """An embedded Back or close cannot retire an active worker early."""
        if self.busy():
            event.ignore()
            return
        super().closeEvent(event)


__all__ = ["AdoptionDialog"]
