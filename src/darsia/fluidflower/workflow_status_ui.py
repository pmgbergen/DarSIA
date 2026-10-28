"""Native-only display adapter for existing saved workflow artefacts.

Navigation, numerical actions and the original settings panes are unchanged.
This adapter never creates a workspace, starts a workflow or discovers history.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from darsia.fluidflower.workflow_row_status import RowStatus, load_status_context, oslo_date, row_status
from darsia.fluidflower.series_paths import series_key
from darsia.fluidflower.step_confirmations import (
    confirmation_state, confirmations_path, read_confirmations, record_step, step_basis, step_key)
from darsia.presets.workflows.results_folder import open_in_file_explorer


# Rows of the native pages (N141) with no saved-output status of their own.
# They get no status line rather than an invented "Not done".
NO_STATUS_ROWS = {
    ("setup", "finish"),
    ("geometry_controls", "curvature"),
    ("phase_model", "normalisation"),
    ("phase_model", "overflow"),
    ("phase_model", "calibration_review"),
    ("phase_model", "checks"),
    ("jobs", "new_analysis"),
    ("overflow_data", "runs"),
    ("jobs", "production_results"),
}

# Setup rows completed by one Finish setup job (image correction, colour paths
# and the job itself). Their mark follows that job's verified completion.
SETUP_COMPLETION_ROWS = (("phase_model", "normalisation"), ("calibration", "color"), ("setup", "finish"))
SETUP_COMPLETION_MAX_AGE_S = 60.0
SETUP_COMPLETION_SETTINGS_GROUP = "native_workflow_status/setup_completion"

# Geometry steps marked by their saved state: the saved rig frame and injection
# points, the current geometry revision and its export. Saving or adopting the
# map is their confirmation, so the mark follows that state.
SAVED_STATE_ROWS = {("geometry_facies", "rig_frame"), ("geometry_facies", "layers"), ("geometry_facies", "export")}
# Rows marked by a calculation's verified output or their saved state; they need no confirmation.
COMPUTED_ROWS = {*SETUP_COMPLETION_ROWS, ("setup", "rig"), ("setup", "all"), *SAVED_STATE_ROWS}
# Steps without a saved-output status that the user still confirms.
CONFIRM_ONLY_ROWS = {("overflow_data", "runs"), ("phase_model", "calibration_review"), ("phase_model", "checks")}
# States of a step whose output or choice is saved, so it can be confirmed.
SAVED_STATES = {"done", "saved_choice", "default_choice"}
# How often the series' choice files are checked for changes made elsewhere.
CHANGE_CHECK_INTERVAL_MS = 15000
# The rows' saved status is read in the background. An answer this recent is
# shown again at once; a load, a save or a change seen elsewhere reads anew.
STATUS_MAX_AGE_S = 30.0
CHECKING_TEXT = "Checking…"
UNAVAILABLE_TEXT = "Status unavailable"

STATE_LABELS = {
    "done": "Done",
    "missing": "Not done",
    "not_applicable": "— Not applicable",
    "saved_choice": "Saved choice",
    "default_choice": "Default choice",
    "choice_error": "Invalid choice",
}


def status_text(state) -> str:
    """Keep what was found and the date visible; paths stay in the detail pane."""
    line = STATE_LABELS[state.state]
    if state.date:
        line += " · " + state.date
    text = line + ("\n" + state.summary if state.summary else "")
    return text + "\n" + state.selected_run_text if state.selected_run_text else text


def state_style(state, *, compact=False) -> str:
    prefix = "font-size: 10px;" if compact else ""
    return prefix + ("color: #667085;" if state.state == "not_applicable" else "")


def confirmed_status_text(state, confirmation) -> str:
    """Row text of a confirmable step: its confirmation, then what its status found.

    ``state`` is the step's RowStatus, or None for a step without a saved-output
    status. ``confirmation`` is (``confirmed``/``changed``/``earlier``/``open``,
    event). Steps that are not saved yet keep their ordinary status text.
    """
    kind, event = confirmation
    when = oslo_date((event or {}).get("utc"))
    if state is None:
        return "Confirmed" + (" · " + when if when else "") if kind == "confirmed" else "Not confirmed"
    if state.state not in SAVED_STATES:
        return status_text(state)
    if kind == "confirmed":
        line = "Confirmed" + (" · " + when if when else "")
    elif kind == "changed":
        line = "Changed since confirmed" + (" · " + state.date if state.date else "")
    elif kind == "earlier":
        # Recorded before the files were compared by what the step uses; nothing tells
        # whether they changed since, so the user confirms once more.
        line = ("Confirmed" + (" · " + when if when else "")
                + " with the earlier file check · confirm once more")
    else:
        line = STATE_LABELS[state.state] + " · not confirmed" + (" · " + state.date if state.date else "")
    text = line + ("\n" + state.summary if state.summary else "")
    return text + "\n" + state.selected_run_text if state.selected_run_text else text


def setup_completion_display(result, *, checking=False) -> tuple[str, str, str]:
    """Return (completion state, row text, tooltip) for one Finish setup job reading.

    ``result`` is None before any reading, a dict with ``error`` after a failed
    reading, or the job status with ``complete``, ``message`` and the Oslo time
    it was ``checked``. ``restored`` marks an earlier session's reading, which
    is shown until this session's own reading returns.
    """
    if result is None:
        return "none", ("Checking the Finish setup job…" if checking else ""), ""
    if result.get("error"):
        return "error", "Status unavailable", "The Finish setup job could not be read: " + str(result["error"])
    message = str(result.get("message") or "")
    again = " · checking again…" if result.get("restored") and checking else ""
    if result.get("complete"):
        checked = str(result.get("checked") or "")
        return "done", "Done" + (" · checked " + checked if checked else "") + again, message
    return "none", "Not done" + again + ("\n" + message if message else ""), message


def restored_setup_completion(stored, series_bytes):
    """Return an earlier session's reading while series.toml still has the bytes it was checked with."""
    try:
        record = json.loads(str(stored or ""))
    except ValueError:
        return None
    if (not isinstance(record, dict) or not isinstance(series_bytes, bytes)
            or record.get("series_sha256") != hashlib.sha256(series_bytes).hexdigest()):
        return None
    return dict(complete=record.get("complete") is True, message=str(record.get("message") or ""),
                checked=str(record.get("checked") or ""), restored=True)


def _file_stat(path):
    """(size, modification time in ns) of a file, or None when it cannot be read."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


def setup_completion_store():
    """Per-user Qt settings that keep the last Finish setup reading of each series."""
    from PySide6.QtCore import QSettings
    return QSettings()


def _setup_completion_setting(key: str) -> str:
    """Name the settings value of one series by a digest of its root."""
    # Series roots contain separators that QSettings reads as groups.
    return SETUP_COMPLETION_SETTINGS_GROUP + "/" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class NativeWorkflowStatus:
    """Attach passive saved-file status to the native window's existing rows."""

    def __init__(self, window):
        self.window = window
        self.states = {}
        self.rows = {}
        # Setup rows whose mark follows the Finish setup job (SETUP_COMPLETION_ROWS).
        self.setup_rows = {}
        # Confirmed steps without a saved-output status (CONFIRM_ONLY_ROWS).
        self.confirm_rows = {}
        # Latest confirmation events per series key, and the watched files' stats.
        self.confirmations = {}
        self._watched = {}
        self.context = None
        # Latest background answer per status request: (read at, context, states).
        self._status_answers = {}
        self._states_series = None
        self._refreshing = False
        self.selected_references = {}
        self.panel = QWidget()
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(0, 2, 0, 2)
        line = QHBoxLayout()
        self.label = QLabel()
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        line.addWidget(self.label, 1)
        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(self._open_current)
        line.addWidget(self.open_button)
        self.start_button = QPushButton("Start")
        self.start_button.setToolTip(
            "Open this step's settings; no calculation starts."
        )
        self.start_button.clicked.connect(self._start_current)
        line.addWidget(self.start_button)
        self.confirm_button = QPushButton("Confirm step")
        self.confirm_button.setToolTip(
            "Confirm this step for the series; its checkmark stays until its saved choice changes."
        )
        self.confirm_button.clicked.connect(self._toggle_confirmation)
        line.addWidget(self.confirm_button)
        layout.addLayout(line)
        self.missing_label = QLabel()
        self.missing_label.setTextFormat(Qt.TextFormat.PlainText)
        self.missing_label.setWordWrap(True)
        self.missing_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.missing_label)
        right = window.content_stack.parentWidget().layout()
        right.insertWidget(right.indexOf(window.content_stack), self.panel)
        self.panel.hide()
        workspace = window.phase_model_workspace
        original_studio = workspace.ensure_studio

        def ensure_studio(*args, **kwargs):
            page = original_studio(*args, **kwargs)
            self.bind_studio_page(page)
            return page

        ensure_studio.__wrapped__ = original_studio
        workspace.ensure_studio = ensure_studio
        original_mass = window.ensure_mass_analysis

        def ensure_mass(*args, **kwargs):
            page = original_mass(*args, **kwargs)
            self.bind_mass_workspace(page)
            return page

        ensure_mass.__wrapped__ = original_mass
        window.ensure_mass_analysis = ensure_mass
        for category in ("setup", "calibration", "analysis", "comparison", "export"):
            section = window.sidebar._sections.get(category)
            if section is None:
                continue
            for row in section._rows.values():
                logical = (row.action, row.checkbox_id)
                if category == "comparison" and row.action != "analysis":
                    continue
                completion = category == "setup" and logical in SETUP_COMPLETION_ROWS
                confirm_only = logical in CONFIRM_ONLY_ROWS
                if logical in NO_STATUS_ROWS and not completion and not confirm_only:
                    continue
                controls = QWidget(row)
                controls.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                vertical = QVBoxLayout(controls)
                vertical.setContentsMargins(0, 0, 0, 0)
                vertical.setSpacing(1)
                row._workflow_original_label_style = row._text_label.styleSheet()
                row.layout().removeWidget(row._text_label)
                vertical.addWidget(row._text_label)
                text = QLabel()
                text.setTextFormat(Qt.TextFormat.PlainText)
                text.setWordWrap(True)
                text.setStyleSheet("font-size: 10px;")
                text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                vertical.addWidget(text)
                # Keep the original row label, selection and click behaviour.
                row.layout().insertWidget(1, controls, 1)
                if completion:
                    self.setup_rows[logical] = (row, text)
                if confirm_only:
                    self.confirm_rows[logical] = (row, text)
                if logical not in NO_STATUS_ROWS:
                    self.rows[logical] = (row, text)
        from PySide6.QtCore import QTimer
        self._change_timer = QTimer(self.panel)
        self._change_timer.setInterval(CHANGE_CHECK_INTERVAL_MS)
        self._change_timer.timeout.connect(self._check_for_changes)
        self._change_timer.start()

    def bind_studio_page(self, page) -> None:
        """Observe only an already requested Studio; never instantiate one."""
        if getattr(page, "_workflow_status_bound", False):
            return
        signal = getattr(page, "studio_loaded", None)
        if signal is None:
            return
        page._workflow_status_bound = True
        signal.connect(
            lambda: (
                self.bind_review_studio(page.studio)
                if getattr(page, "studio", None) is not None
                else None
            )
        )
        if getattr(page, "studio", None) is not None:
            self.bind_review_studio(page.studio)

    def bind_review_studio(self, studio) -> None:
        if getattr(studio, "_workflow_status_bound", False):
            return
        studio._workflow_status_bound = True
        studio.review_opened.connect(
            lambda binding: self._review_opened(studio, binding)
        )

    def _review_opened(self, studio, binding) -> None:
        workspace = self.window.phase_model_workspace
        if getattr(studio, "_active_model_series", None) != workspace.series_id:
            return
        selected = studio._selected_validation_model() or {}
        if selected.get("validation_binding") != binding:
            return
        root = binding.get("combined_root")
        if not isinstance(root, str) or not root.strip():
            return
        # This is the actual prepared collection manifest named by the already
        # selected binding, not its job-less receipt or a discovered history.
        self.set_selected_reference(
            "review", str(Path(root) / "postflight_manifest.json")
        )

    def bind_mass_workspace(self, page) -> None:
        saved = getattr(page, "saved_results", None)
        if saved is None or getattr(saved, "_workflow_status_bound", False):
            return
        saved._workflow_status_bound = True
        original = saved.load_result

        def load_result(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            current = getattr(saved, "result_path", None)
            if (
                current is not None
                and Path(current).absolute() == Path(path).absolute()
            ):
                self._result_loaded(saved, current)
            return result

        load_result.__wrapped__ = original
        saved.load_result = load_result

    def _result_loaded(self, saved, path) -> None:
        value = getattr(saved, "result", None) or {}
        candidate = value.get("selected_inputs", {}).get("candidate", {})
        series = self.window.phase_model_workspace.series_id
        try:
            if series_key(str(candidate.get("dataset") or "")) != series:
                return
        except ValueError:
            return
        wanted = str(Path(path).absolute()).casefold()
        entry = next(
            (
                entry
                for entry in saved.entries
                if str(Path(entry.get("path", "")).absolute()).casefold() == wanted
            ),
            None,
        )
        # A newly selected unregistered file must not inherit an earlier receipt.
        self.set_selected_reference(
            "result_receipt", str((entry or {}).get("receipt_path") or "")
        )

    def set_selected_reference(self, kind: str, path: str) -> None:
        """Called only by an existing explicit selection, never a history scan."""
        if kind not in {"selection_plan", "review", "result_receipt"}:
            raise ValueError(f"Unknown saved reference kind: {kind}")
        key = self._series_key()
        if not key:
            raise ValueError(
                "Open a registered series before selecting saved metadata."
            )
        self.selected_references.setdefault(key, {})[kind] = str(path or "")
        self.refresh()

    def _series_key(self):
        workspace = self.window.phase_model_workspace
        root = workspace.series_roots.get(workspace.series_id)
        return str(Path(root).absolute()).casefold() if root is not None else ""

    def _series_root(self):
        workspace = self.window.phase_model_workspace
        root = workspace.series_roots.get(workspace.series_id)
        return Path(root) if root is not None else None

    def _confirmable(self, logical) -> bool:
        """Whether a row is a step the user confirms, not a calculation's verified output."""
        return logical in self.confirm_rows or (logical in self.rows and logical not in COMPUTED_ROWS)

    def _confirmations_for(self, key):
        """Latest confirmation events of the open series: read once, then kept current by the change check."""
        if key not in self.confirmations:
            root = self._series_root()
            try:
                self.confirmations[key] = read_confirmations(confirmations_path(root)) if root else {}
            except (OSError, ValueError) as error:
                self.confirmations[key] = {}
                self.window.print_log(f"Step confirmations could not be read: {error}")
        return self.confirmations[key]

    def _confirmation(self, logical, state):
        """Return (``confirmed``/``changed``/``open``, event) of a confirmable step, else None."""
        key = self._series_key()
        if logical is None or not key or not self._confirmable(logical):
            return None
        event = self._confirmations_for(key).get(step_key(logical))
        return confirmation_state(event, step_basis(state)), event

    def _show_confirm_button(self, confirmation, state):
        """Offer Confirm step, or Undo confirmation, for the selected confirmable step."""
        button = self.confirm_button
        if confirmation is None or (state is not None and state.state == "not_applicable"):
            button.hide()
            return
        writer = self.__dict__.get("_confirm_writer")
        confirmed = confirmation[0] == "confirmed"
        saved = state is None or state.state in SAVED_STATES
        button.setText("Undo confirmation" if confirmed else "Confirm step")
        button.setEnabled(not (writer is not None and writer.busy) and (confirmed or saved))
        button.setToolTip(
            "Remove this step's checkmark; its saved choice stays unchanged." if confirmed else
            "Confirm this step for the series; its checkmark stays until its saved choice changes."
            if saved else "Save this step's choice first, then confirm it.")
        button.show()

    def _toggle_confirmation(self):
        """Record a confirmation, or its undoing, of the selected step in the series folder."""
        from .ui_metadata_preview import MetadataPreview
        selected = self.window.sidebar.get_selected()
        state = self.states.get(selected)
        confirmation = self._confirmation(selected, state) if selected is not None else None
        root = self._series_root()
        if confirmation is None or root is None:
            return
        writer = self.__dict__.get("_confirm_writer")
        if writer is None:
            # The adapter's one explicit write: the series' confirmation file.
            writer = self._confirm_writer = MetadataPreview(self.panel)
            writer.completed.connect(self._confirmation_written)
        elif writer.busy:
            return
        action = "reopen" if confirmation[0] == "confirmed" else "confirm"
        basis = step_basis(state) if action == "confirm" else ""
        key, path, step = self._series_key(), confirmations_path(root), step_key(selected)
        self.confirm_button.setEnabled(False)
        writer.start(lambda _cancelled: (key, record_step(path, step, action, basis)))

    def _confirmation_written(self, value, error):
        """Show the saved confirmation, or why it was not saved."""
        if error or value is None:
            # Read the file again; another computer may have changed it meanwhile.
            self.confirmations.pop(self._series_key(), None)
        else:
            key, latest = value
            self.confirmations[key] = latest
        self.refresh()
        if error or value is None:
            self.missing_label.setText("The confirmation was not saved: " + (error or "no result was returned."))
            self.missing_label.show()

    def _check_for_changes(self):
        """Compare the series' choice files in the background; refresh the marks when one changed."""
        from .ui_metadata_preview import MetadataPreview
        root = self._series_root()
        if root is None or not self.window.isVisible():
            return
        reader = self.__dict__.get("_change_reader")
        if reader is None:
            reader = self._change_reader = MetadataPreview(self.panel)
            reader.completed.connect(self._changes_read)
        elif reader.busy:
            return
        key = self._series_key()

        def read(_cancelled):
            from .series_paths import series_folder_path
            files = (root / "series.toml", root / "config/image_selection_plan.toml",
                     series_folder_path(root, "geometry_facies") / "overrides/geometry_facies/current.json")
            return key, tuple(_file_stat(path) for path in files), read_confirmations(confirmations_path(root))
        reader.start(read)

    def _changes_read(self, value, error):
        """Keep the files' state; a change of series.toml also makes the Setup reading stale."""
        if error or value is None:
            return  # The next check reads again.
        key, stats, latest = value
        previous = self._watched.get(key)
        self._watched[key] = stats
        confirmations_changed = key in self.confirmations and self.confirmations[key] != latest
        self.confirmations[key] = latest
        if previous is not None and previous[0] != stats[0]:
            self.__dict__.get("_setup_completion", {}).pop(key, None)
        if key == self._series_key() and (confirmations_changed or previous not in (None, stats)):
            self.window.refresh_sidebar_progress()

    @property
    def checking(self):
        """True while the rows' saved status is read in the background, until its answer shows."""
        reader = self.__dict__.get("_status_reader")
        return (reader is not None and reader.busy) or self.__dict__.get("_status_unshown", False)

    def invalidate(self):
        """Read the rows' saved status again at the next refresh; a choice or output may have changed."""
        self._status_answers.clear()

    def _status_request(self, root):
        """What one background reading of the rows needs, captured on the GUI thread."""
        window = self.window
        references = self.selected_references.get(self._series_key(), {})
        # The plan dialog has one documented default path. No latest/history
        # search and no pixel/NPZ read is needed to inspect that named plan.
        plan = references.get("selection_plan", "")
        if not plan and root is not None:
            plan = str(Path(root) / "config/image_selection_plan.toml")
        scope = window.scope_context
        effective = tuple(dict.fromkeys(window._effective_workflow_selection(logical) for logical in self.rows))
        request = dict(root=root, scope=scope, method=window._saved_method, plan=plan,
                       review=references.get("review", ""), receipt=references.get("result_receipt", ""),
                       effective=effective, series=self._series_key())
        request["key"] = (request["series"], str(getattr(scope, "selected_run", "") or ""),
                          str(getattr(scope, "loaded_path", "") or ""), repr(request["method"]),
                          plan, request["review"], request["receipt"], effective,
                          self._status_files_stamp(root, scope))
        return request

    @staticmethod
    def _status_files_stamp(root, scope):
        """Size and time of the files most rows read, so that a save there reads anew at once.

        Four stat calls instead of the rows' own reads. Outputs written by jobs
        are read again when the answer is older than STATUS_MAX_AGE_S.
        """
        if root is None:
            return ()
        from .series_paths import series_folder_path
        files = [Path(root) / "series.toml", Path(root) / "config/image_selection_plan.toml"]
        try:
            files.append(series_folder_path(root, "geometry_facies") / "overrides/geometry_facies/current.json")
        except (OSError, ValueError, KeyError):
            pass
        loaded = getattr(scope, "loaded_path", None)
        if loaded:
            files.append(Path(loaded))
        return tuple(_file_stat(path) for path in files)

    def _status_answer(self, request):
        """The last answer for ``request``; a background reading starts when it is missing or old."""
        import time
        answer = self._status_answers.get(request["key"])
        if answer is None or time.monotonic() - answer[0] >= STATUS_MAX_AGE_S:
            self._start_status_read(request)
        return (answer[1], answer[2]) if answer is not None else (None, None)

    def _start_status_read(self, request):
        """Read every row's status in the background, unless the same reading runs."""
        from .ui_metadata_preview import MetadataPreview
        reader = self.__dict__.get("_status_reader")
        if reader is None:
            reader = self._status_reader = MetadataPreview(self.panel)
            reader.completed.connect(self._status_read)
        elif reader.busy and self.__dict__.get("_status_reading") == request["key"]:
            return
        self._status_reading = request["key"]

        def read(_cancelled):
            context = load_status_context(
                request["root"], scope_context=request["scope"], method_selection=request["method"],
                selection_plan_reference=request["plan"], review_reference=request["review"],
                result_receipt_reference=request["receipt"])
            states = {}
            for effective in request["effective"]:
                try:
                    states[effective] = row_status(context, *effective)
                except Exception as error:  # noqa: BLE001 - one unreadable row must not hide the others
                    states[effective] = RowStatus("missing", f"Status could not be read: {error}")
            try:
                root = request["root"]
                confirmations = read_confirmations(confirmations_path(Path(root))) if root is not None else {}
            except (OSError, ValueError):
                confirmations = None  # Read on the GUI thread, which reports the error.
            return request["key"], context, states, confirmations

        reader.start(read)

    def _status_read(self, value, error):
        """Show the answer; a failed reading keeps the rows' last status and is retried later."""
        import time
        reading = self.__dict__.pop("_status_reading", None)
        if error or value is None:
            self.window.print_log("Step status could not be read: " + (error or "no answer was returned"))
            answer = self._status_answers.get(reading)
            self._status_answers[reading] = (time.monotonic(), *(answer[1:] if answer else (None, {})))
        else:
            key, context, states, confirmations = value
            self._status_answers[key] = (time.monotonic(), context, states)
            if confirmations is not None:
                self.confirmations.setdefault(key[0], confirmations)
        self._status_unshown = True
        if self._refreshing:
            # The answer arrived inside a refresh; show it right after that one.
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, self.refresh)
        else:
            self.refresh()

    def refresh(self, *, context=None, states=None) -> None:
        if self._refreshing:
            return
        provider = getattr(self, 'prepared_context_provider', None)
        if context is None and provider is not None:
            prepared = provider()
            if prepared is None:
                if getattr(self.window, '_series_loading', False):
                    return
            else:
                context, states = prepared
        self._refreshing = True
        try:
            window = self.window
            workspace = window.phase_model_workspace
            root = workspace.series_roots.get(workspace.series_id)
            request = self._status_request(root)
            if context is not None and states is not None:
                # Given states (the loaded series' snapshot) show as they are;
                # a fresh reading follows them.
                self._status_answers[request["key"]] = (-math.inf, context, states)
            context, states = self._status_answer(request)
            self._status_unshown = False
            self.context = context
            # Until a reading answers, each row keeps its last status of this series.
            previous = self.states if self._states_series == request["series"] else {}
            self._states_series, self.states = request["series"], {}
            for logical, (row, text) in self.rows.items():
                if logical == ("calibration", "color") and window._saved_method.active == "phase_model":
                    # This is the P1 colour-path workflow, not the original
                    # colour-calibration folder. Its own page reports that job.
                    text.clear()
                    row.set_completion_state("none")
                    row._text_label.setStyleSheet(row._workflow_original_label_style)
                    continue
                effective = window._effective_workflow_selection(logical)
                from .native_common_workflow import availability
                applicable, _reason = availability(logical, window._saved_method.active)
                if not row.isEnabled() and not applicable:
                    # This route cannot be selected for the saved model. Its
                    # exclusions/output checks belong to another calculation.
                    state = RowStatus("not_applicable", "")
                else:
                    state = states.get(effective) if states is not None else None
                    if state is None:
                        state = previous.get(logical)
                    if state is None:
                        text.setText(CHECKING_TEXT if self.checking else UNAVAILABLE_TEXT)
                        text.setToolTip("")
                        continue
                self.states[logical] = state
                confirmation = self._confirmation(logical, state)
                text.setText(confirmed_status_text(state, confirmation) if confirmation else status_text(state))
                text.setStyleSheet(state_style(state, compact=True))
                row._text_label.setStyleSheet(row._workflow_original_label_style + state_style(state))
                details = self._details(state)
                text.setToolTip(details)
                done = (confirmation[0] == "confirmed" and state.state in SAVED_STATES if confirmation
                        else state.state in {"done", "saved_choice"})
                row.set_completion_state("done" if done else "none")
            for logical, (row, text) in self.confirm_rows.items():
                confirmation = self._confirmation(logical, None)
                text.setText(confirmed_status_text(None, confirmation) if confirmation else "")
                row.set_completion_state("done" if confirmation and confirmation[0] == "confirmed" else "none")
            self._refresh_setup_completion(root)
            if root is not None and self._series_key() not in self._watched:
                self._check_for_changes()  # The first check only notes the files' state.
            selected = window.sidebar.get_selected()
            state = self.states.get(selected)
            confirmation = self._confirmation(selected, state) if selected is not None else None
            self._show_confirm_button(confirmation, state)
            self.panel.setVisible(state is not None or (selected in self.confirm_rows and confirmation is not None))
            checking = state is None and selected in self.rows and self.checking
            if checking:
                # Confirming needs the step's status; it shows once the reading answers.
                self.panel.show()
                self.confirm_button.hide()
            if state is None:
                self.label.setText("Checking this step's saved status…" if checking and not confirmation
                                   else confirmed_status_text(None, confirmation) if confirmation else "")
                self.label.setStyleSheet("")
                self.label.setToolTip("")
                self.open_button.hide()
                self.start_button.hide()
                self.missing_label.hide()
                return
            self.label.setText((confirmed_status_text(state, confirmation) if confirmation else status_text(state))
                               + ("\nChecking again…" if self.checking else ""))
            self.label.setStyleSheet(state_style(state))
            self.label.setToolTip(self._details(state))
            # Phase-model panels own their navigation; the saved-model status
            # above them is informational. Other pages retain their file action.
            self.open_button.setVisible(
                selected != ("phase_model", "training")
                and state.state == "done" and bool(state.artifacts)
            )
            self.start_button.setVisible(state.state == "missing")
            row = self.rows[selected][0]
            self.start_button.setEnabled(row.isEnabled())
            self.missing_label.setText(self._details(state))
            self.missing_label.setVisible(state.state == "missing")
        finally:
            self._refreshing = False

    def _apply_setup_completion(self, result, *, checking=False):
        """Show one Finish setup reading (see setup_completion_display) on its Setup rows."""
        state, line, tooltip = setup_completion_display(result, checking=checking)
        for row, text in self.setup_rows.values():
            row.set_completion_state(state)
            text.setText(line)
            text.setToolTip(tooltip)
            if tooltip:
                row.setToolTip(tooltip)

    def _clear_setup_completion(self):
        """Unmark the job-completed Setup rows that have no saved-output status of their own."""
        for logical, (row, text) in self.setup_rows.items():
            if logical not in self.rows:
                row.set_completion_state("none")
                text.clear()

    def _refresh_setup_completion(self, root):
        """Mark the Setup rows from the registered Finish setup job, read in the background.

        The last reading of the series shows at once, also after a restart while
        series.toml keeps the bytes it was checked with, and it is marked as being
        checked again until this session's reading returns. A failed reading
        shows as unavailable instead of as not done.
        """
        import time
        if root is None or self.window._saved_method.active != "phase_model":
            self._clear_setup_completion()
            return
        key = self._series_key()
        cache = self.__dict__.setdefault("_setup_completion", {})
        cached = cache.get(key)
        restored_keys = self.__dict__.setdefault("_setup_completion_restored", set())
        if (cached is None or cached[1] is None) and key not in restored_keys:
            # Once per session: a later series.toml change must be read again.
            restored = self._restored_setup_completion(key)
            if restored is not None:
                restored_keys.add(key)
                cached = cache[key] = (cached[0] if cached else -math.inf, restored)
        if not cached or time.monotonic() - cached[0] >= SETUP_COMPLETION_MAX_AGE_S:
            self._start_setup_completion_read(root, key, cached)
        reading = self.__dict__.get("_setup_completion_reading") == key
        self._apply_setup_completion(cached[1] if cached else None, checking=reading)

    def _start_setup_completion_read(self, root, key, cached):
        """Start one background reading of the series' Finish setup job unless one runs."""
        import time
        from .ui_metadata_preview import MetadataPreview
        reader = self.__dict__.get("_setup_completion_reader")
        if reader is None:
            reader = self._setup_completion_reader = MetadataPreview(self.panel)
            reader.completed.connect(self._setup_completion_read)
        elif reader.busy:
            return
        ensure_jobs = getattr(self.window, "ensure_jobs", None)
        if ensure_jobs is None:
            return
        jobs = ensure_jobs()
        registry, series = jobs.registry, jobs.series_id
        self._setup_completion[key] = (time.monotonic(), cached[1] if cached else None)
        self._setup_completion_reading = key

        def read(_cancelled):
            from .native_finish_setup import finish_setup_status
            return key, finish_setup_status(root, registry.jobs(root, series))
        reader.start(read)

    def _setup_completion_read(self, value, error):
        """Keep, store and show a returned reading; a failure is kept as unavailable."""
        import time
        key = self.__dict__.pop("_setup_completion_reading", None)
        if error or value is None:
            if key is None:
                return
            result = dict(error=error or "No status was returned.")
        else:
            key, status = value
            result = dict(status, checked=oslo_date(dt.datetime.now(dt.timezone.utc)))
            self._store_setup_completion(key, result)
            # This session's own reading replaces any stored one from now on.
            self.__dict__.setdefault("_setup_completion_restored", set()).add(key)
        self._setup_completion[key] = (time.monotonic(), result)
        if key == self._series_key():
            self._apply_setup_completion(result)

    def _restored_setup_completion(self, key):
        """Read the last stored reading against the series.toml bytes of the loaded series."""
        snapshot = getattr(self.window, "_series_snapshot", None)
        workspace = self.window.phase_model_workspace
        if not snapshot or snapshot[0] != workspace.series_id:
            return None
        return restored_setup_completion(
            setup_completion_store().value(_setup_completion_setting(key), ""),
            snapshot[1].get("raw"))

    def _store_setup_completion(self, key, result):
        """Remember a reading with the digest of the series.toml bytes it was checked with."""
        raw = result.get("raw")
        if not isinstance(raw, bytes):
            return
        record = dict(series_sha256=hashlib.sha256(raw).hexdigest(), complete=bool(result.get("complete")),
                      message=str(result.get("message") or ""), checked=str(result.get("checked") or ""))
        setup_completion_store().setValue(_setup_completion_setting(key), json.dumps(record))

    @staticmethod
    def _details(state) -> str:
        lines = []
        if state.selected_run_text:
            lines.append(state.selected_run_text)
        if state.date_source:
            lines.append("Date source: " + state.date_source)
        lines.extend(f"Saved: {item.kind}: {item.path}" for item in state.artifacts)
        lines.extend("Missing: " + item for item in state.missing)
        if state.producer:
            lines.append("Produced by: " + state.producer)
        if state.next_step:
            lines.append("Next: " + state.next_step)
        return "\n".join(lines)

    def activate(self, logical) -> None:
        state = self.states.get(logical)
        if logical == ("image_selection", "plan") and state is not None and state.state == "done":
            self.window._select_row(logical)
            self.window.image_selection_workflow.open()
            return
        if state is not None and state.state == "done" and state.artifacts:
            try:
                open_in_file_explorer(Path(state.artifacts[0].path))
            except (OSError, RuntimeError) as error:
                self.window.print_log(f"Could not open saved output: {error}")
            return
        # An explicit Start is navigation only, including when output is absent.
        self.window._select_row(logical)

    def _open_current(self):
        self.activate(self.window.sidebar.get_selected())

    def _start_current(self):
        self.window._select_row(self.window.sidebar.get_selected())
