"""Explicit local store Settings and passive Jobs availability presentation.

Opening or changing series performs no filesystem/availability query. Load,
Check and Save are explicit finite metadata operations off the UI event thread.
No job, queue, worker, probe, input hash or result publication is started here.
"""
from __future__ import annotations

from pathlib import Path
import json
import traceback

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout,
)

from .store_map import (
    LIMIT, default_store_map_path, parse_store_map, propose_manifest_roots,
    read_store_map, root_availability, save_store_map_text,
)


class _MetadataCall(QThread):
    def __init__(self, callback, parent):
        super().__init__(parent)
        self.callback, self.value, self.error = callback, None, ""

    def run(self):
        try:
            self.value = self.callback()
        except Exception:
            self.error = traceback.format_exc()


def _label(text=""):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _small_bytes(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise ValueError("Settings/manifest document exceeds 1 MiB")
    return raw


class StoreSettingsDialog(QDialog):
    saved = Signal(str)
    series_locations_saved = Signal(str)
    jobs_folder_saved = Signal(str)

    def __init__(self, series_root=None, parent=None, *, path=None,
                 reader=read_store_map, writer=save_store_map_text):
        super().__init__(parent)
        self.setWindowTitle("Machine data stores — Settings")
        self.setModal(True)
        self.resize(850, 650)
        self.path = Path(path) if path is not None else default_store_map_path()
        self.series_root = Path(series_root) if series_root else None
        self.reader, self.writer = reader, writer
        self._thread = None
        self._expected = None
        self._loaded = False
        self._displayed_text = ""
        self._locations = None
        self._results_migration_handler = self._preview_results_migration
        self._results_migration_plan = self._results_migration_fields = None
        self._location_browse = []
        layout = QVBoxLayout(self)
        layout.addWidget(_label("This machine's store map: " + str(self.path)))
        layout.addWidget(_label("Shared series locations are saved in series.toml. The machine store map separately binds this computer's locations, working folder and cache folder."))
        shared = self.shared_series_locations = QGroupBox("Shared series locations")
        shared_form = QFormLayout(shared)
        self.setup_location = _label("Choose a series, then load its locations.")
        shared_form.addRow("DarSIA setup folder", self.setup_location)
        self.project_location, self.results_location, self.jobs_location = QLineEdit(), QLineEdit(), QLineEdit()
        for label, edit in (("Project folder", self.project_location), ("Results folder", self.results_location),
                            ("Shared Jobs / workers folder", self.jobs_location)):
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            browse = QPushButton("Browse…")
            self._location_browse.append(browse)
            browse.clicked.connect(lambda _checked=False, widget=edit, title=label: self._browse_location(widget, title))
            row.addWidget(browse)
            shared_form.addRow(label, row)
            edit.textChanged.connect(self._controls)
        self.replace_run_results = QCheckBox("Also replace individually configured run result folders")
        self.replace_run_results.toggled.connect(self._controls)
        shared_form.addRow(self.replace_run_results)
        self.load_locations_button = QPushButton("Load series locations")
        self.save_locations_button = QPushButton("Save future output locations")
        self.use_locations_button = QPushButton("Use series locations in this machine's store map")
        self.move_results_button = QPushButton("Move existing Results…")
        self.move_results_button.setToolTip("Preview a move of existing results before changing their location.")
        self.move_writers_closed = QCheckBox("Jobs and other writers for this series are closed")
        self.move_writers_closed.toggled.connect(self._controls)
        shared_form.addRow(self.move_writers_closed)
        self.apply_move_button = QPushButton("Move reviewed Results")
        self.apply_move_button.hide()
        for button in (self.load_locations_button, self.save_locations_button, self.use_locations_button,
                       self.move_results_button, self.apply_move_button):
            shared_form.addRow(button)
        shared_form.addRow(_label("Saving output locations moves no files. Working and cache folders can be local or shared NAS folders, selected for each computer below."))
        layout.addWidget(shared)
        actions = QHBoxLayout()
        self.load_button = QPushButton("Load existing stores.toml")
        self.new_button = QPushButton("New local store map")
        self.save_button = QPushButton("Save local settings")
        self.roots_button = QPushButton("Preview series root migration…")
        for button in (self.load_button, self.new_button, self.save_button, self.roots_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        working_form = QFormLayout()
        working_row = QHBoxLayout()
        self.working_location = QLineEdit()
        self.working_location.setAccessibleName("Working folder on this computer")
        self.working_location.setToolTip("Stored for this series on this computer. Choose a local folder or a shared NAS folder for calculation outputs.")
        self.working_browse = QPushButton("Browse…")
        working_row.addWidget(self.working_location, 1)
        working_row.addWidget(self.working_browse)
        working_form.addRow("Working folder (this computer)", working_row)
        cache_row = QHBoxLayout()
        self.cache_location = QLineEdit()
        self.cache_location.setAccessibleName("Cache folder on this computer")
        self.cache_location.setToolTip("Temporary files for this series. Choose a local folder or a shared NAS folder; each machine uses its own subfolder.")
        self.cache_browse = QPushButton("Browse…")
        cache_row.addWidget(self.cache_location, 1)
        cache_row.addWidget(self.cache_browse)
        working_form.addRow("Cache folder (this computer)", cache_row)
        layout.addLayout(working_form)
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText('Load the existing map, or choose New and enter [machine] and [stores.<series>] with actual locations.')
        layout.addWidget(self.editor, 1)
        self.validation_label = _label()
        self.validation_label.hide()
        layout.addWidget(self.validation_label)
        self.status = _label("Nothing loaded. No filesystem check or write has run.")
        layout.addWidget(self.status)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.hide()
        layout.addWidget(self.preview)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        layout.addWidget(close)
        self.load_button.clicked.connect(self.load)
        self.new_button.clicked.connect(self.new)
        self.save_button.clicked.connect(self.save)
        self.roots_button.clicked.connect(self.preview_roots)
        self.load_locations_button.clicked.connect(self.load_locations)
        self.save_locations_button.clicked.connect(self.save_locations)
        self.use_locations_button.clicked.connect(self.use_series_locations)
        self.move_results_button.clicked.connect(self.request_results_migration)
        self.apply_move_button.clicked.connect(self.apply_results_migration)
        self.editor.textChanged.connect(self._controls)
        self.working_location.editingFinished.connect(self._working_edited)
        self.working_browse.clicked.connect(self._browse_working)
        self.cache_location.editingFinished.connect(self._cache_edited)
        self.cache_browse.clicked.connect(self._browse_cache)
        self._controls()

    def _controls(self):
        idle = self._thread is None
        if hasattr(self, "load_locations_button"):
            same_series = self._locations is not None and self._locations["series_root"] == self.series_root
            self.load_locations_button.setEnabled(idle and self.series_root is not None)
            self.save_locations_button.setEnabled(idle and same_series and bool(
                self.project_location.text().strip() and self.results_location.text().strip()))
            self.use_locations_button.setEnabled(idle and self._loaded and self.series_root is not None
                                                 and not self.has_location_edits())
            self.move_results_button.setEnabled(idle and same_series and self._results_migration_handler is not None)
            self.apply_move_button.setEnabled(idle and same_series and self._results_migration_plan is not None
                and self._results_migration_fields == self._location_fields() and self.move_writers_closed.isChecked())
            self.project_location.setReadOnly(not idle)
            self.results_location.setReadOnly(not idle)
            self.jobs_location.setReadOnly(not idle)
            self.replace_run_results.setEnabled(idle)
            self.move_writers_closed.setEnabled(idle)
            for button in self._location_browse:
                button.setEnabled(idle)
        self.load_button.setEnabled(idle)
        self.new_button.setEnabled(idle)
        self.roots_button.setEnabled(idle and self.series_root is not None)
        self.editor.setReadOnly(not idle)
        valid = False
        validation = ""
        if self._loaded:
            try:
                parse_store_map(self.editor.toPlainText().encode("utf-8"))
                valid = True
            except (ValueError, TypeError, KeyError) as error:
                validation = "Cannot save local settings: " + str(error)
        self.validation_label.setText(validation)
        self.validation_label.setVisible(bool(validation))
        self.save_button.setEnabled(idle and valid)
        if not idle:
            reason = "Wait for the current local metadata operation to finish before saving."
        elif not self._loaded:
            reason = "Load existing stores.toml or choose New local store map before saving."
        else:
            reason = validation or "Save the validated local settings."
        self.save_button.setToolTip(reason)
        key = self._sync_working_field()
        self.working_location.setReadOnly(not idle or not key or not self._loaded)
        self.working_browse.setEnabled(idle and bool(key) and self._loaded)
        self.cache_location.setReadOnly(not idle or not key or not self._loaded)
        self.cache_browse.setEnabled(idle and bool(key) and self._loaded)

    def _working_store(self, document):
        """Use the selected series' existing machine-store entry without probing it."""
        stores = document.get("stores") or {}
        if self._locations and self._locations["series_root"] == self.series_root:
            from .series_paths import series_key
            key = next((key for key in stores if series_key(key) == series_key(self._locations["series_id"])), None)
            if key is not None:
                return key
        if self.series_root is not None:
            from .storage_paths import same_storage_path
            return next((key for key, row in stores.items() if isinstance(row, dict)
                and row.get("series") and same_storage_path(row["series"], self.series_root)), None)
        return None

    def _sync_working_field(self):
        """Mirror the current local editor, including existing source-cache defaults."""
        import tomllib
        try:
            document = tomllib.loads(self.editor.toPlainText())
            key = self._working_store(document)
            if key is not None:
                row = document["stores"][key]
                self.working_location.setText(str(row.get("working", row.get("results", ""))))
                self.cache_location.setText(str(row.get("cache", "")))
            else:
                self.working_location.clear()
                self.cache_location.clear()
            return key
        except (ValueError, KeyError, TypeError):
            return None

    def _working_edited(self):
        """Put an explicit folder choice in the same local settings draft."""
        if self._thread is not None or not self._loaded:
            return
        import toml
        import tomllib
        text = self.working_location.text().strip()
        try:
            document = tomllib.loads(self.editor.toPlainText())
            key = self._working_store(document)
            if key is None:
                raise ValueError("Load this series' machine store before choosing its working folder")
            from .store_map import _path
            if text:
                document["stores"][key]["working"] = str(_path(text))
            else:
                document["stores"][key].pop("working", None)
            self.editor.setPlainText(toml.dumps(document))
            self.status.setText("Working folder updated in local settings. Save local settings to apply it.")
        except (ValueError, KeyError, TypeError) as error:
            self.status.setText(str(error))

    def _browse_working(self):
        chosen = QFileDialog.getExistingDirectory(self, "Choose working folder", self.working_location.text())
        if chosen:
            self.working_location.setText(chosen)
            self._working_edited()

    def _cache_edited(self):
        """Keep a user-selected ordinary cache beside the other machine paths."""
        if self._thread is not None or not self._loaded:
            return
        import toml
        import tomllib
        try:
            document = tomllib.loads(self.editor.toPlainText())
            key = self._working_store(document)
            if key is None:
                raise ValueError("Load this series' machine store before choosing its cache folder")
            from .store_map import _path
            text = self.cache_location.text().strip()
            if text:
                document["stores"][key]["cache"] = str(_path(text))
            else:
                document["stores"][key].pop("cache", None)
            self.editor.setPlainText(toml.dumps(document))
            self.status.setText("Cache folder updated in local settings. Save local settings to apply it.")
        except (ValueError, KeyError, TypeError) as error:
            self.status.setText(str(error))

    def _browse_cache(self):
        chosen = QFileDialog.getExistingDirectory(self, "Choose cache folder", self.cache_location.text())
        if chosen:
            self.cache_location.setText(chosen)
            self._cache_edited()

    def _start(self, callback, done):
        if self._thread is not None:
            return
        task = _MetadataCall(callback, self)
        self._thread = task
        def finished():
            self._thread = None
            if task.error:
                self.status.setText(task.error.strip().splitlines()[-1])
                self.status.setToolTip(task.error)
            else:
                done(task.value)
            task.deleteLater()
            self._controls()
        task.finished.connect(finished)
        self.status.setText("Reading local settings metadata…")
        self._controls()
        task.start()

    def load(self):
        def read():
            stores = self.reader(self.path, forbidden_roots=([self.series_root] if self.series_root else []))
            # Reuse exactly the admitted file, then recheck its descriptor.
            raw = _small_bytes(stores.source["path"])
            check = parse_store_map(raw)
            if check.source["sha256"] != stores.source["sha256"]:
                raise ValueError("Store map changed during Load; reload it")
            return stores, raw
        def done(value):
            stores, raw = value
            self.path = Path(stores.source["path"])
            self._expected, self._loaded = raw, True
            self.editor.setPlainText(raw.decode("utf-8-sig"))
            self._displayed_text = self.editor.toPlainText()
            self.status.setText("Loaded " + str(self.path) + "\nSHA-256: " + stores.source["sha256"])
        self._start(read, done)

    def new(self):
        self._expected, self._loaded = None, True
        self._displayed_text = '# Enter the real machine ID and actual paths; no value is inferred.\n[machine]\nid = ""\n\n[stores]\n'
        self.editor.setPlainText(self._displayed_text)
        self.status.setText("New map requested. Save refuses to replace an existing file that was not loaded.")
        self._controls()

    def save(self):
        if not self.save_button.isEnabled():
            return
        text = self.editor.toPlainText()
        raw = self._expected if self._expected is not None and text == self._displayed_text else text.encode("utf-8")
        expected = self._expected
        def write():
            return self.writer(self.path, raw, expected_bytes=expected,
                               forbidden_roots=([self.series_root] if self.series_root else []))
        def done(receipt):
            self.path = Path(receipt["path"])
            self._expected = raw
            self._displayed_text = self.editor.toPlainText()
            self.status.setText("Saved local settings. SHA-256: " + receipt["sha256"] + "\nPrevious content preserved: " + str(receipt["previous_copy"] or "new file"))
            self.saved.emit(str(self.path))
        self._start(write, done)

    def preview_roots(self):
        root = self.series_root
        if root is None:
            return
        def read():
            return propose_manifest_roots(root, _small_bytes(root / "series.toml"))
        def done(proposal):
            self.preview.setPlainText(json.dumps(proposal.receipt, ensure_ascii=False, indent=2))
            self.preview.show()
            self.status.setText("Proposal only. No series manifest, backup or migration receipt has been written.")
        self._start(read, done)

    def _browse_location(self, edit, title):
        chosen = QFileDialog.getExistingDirectory(self, title, edit.text())
        if chosen:
            edit.setText(chosen)

    def load_locations(self):
        from .series_locations import read_series_locations
        root = self.series_root
        if root is None:
            return
        self._start(lambda: read_series_locations(root), self._show_locations)

    def _show_locations(self, locations):
        self._locations = locations
        self._results_migration_plan = self._results_migration_fields = None
        self.apply_move_button.hide()
        self.move_writers_closed.setChecked(False)
        self.setup_location.setText(str(locations["series_root"]))
        self.project_location.setText(str(locations["project_folder"]))
        self.results_location.setText(str(locations["results_folder"]))
        self.jobs_location.setText(str(locations["jobs_folder"]))
        self.status.setText("Shared locations loaded. Choose Save for future output, or Move existing Results for a reviewed relocation.")
        self._controls()

    def has_location_edits(self):
        return self._locations is not None and (
            self.project_location.text() != str(self._locations["project_folder"])
            or self.results_location.text() != str(self._locations["results_folder"])
            or self.jobs_location.text() != str(self._locations["jobs_folder"]))

    def save_locations(self):
        from .series_locations import save_series_locations
        if not self.save_locations_button.isEnabled():
            return
        locations = self._locations
        project, results = self.project_location.text().strip(), self.results_location.text().strip()
        jobs = self.jobs_location.text().strip()
        replace = self.replace_run_results.isChecked()
        def save():
            if (project == str(locations["project_folder"]) and results == str(locations["results_folder"])
                    and not replace):
                from .series_locations import save_series_jobs_folder
                value = save_series_jobs_folder(locations["series_root"], jobs,
                    expected_bytes=locations["manifest_bytes"])
                return dict(value, retained_runs=[], updated_runs=[], jobs_only=True)
            return save_series_locations(locations["series_root"], project, results,
                jobs_folder=jobs, expected_bytes=locations["manifest_bytes"], replace_custom_results=replace)
        def done(value):
            self._show_locations(value)
            retained = value["retained_runs"]
            self.status.setText("Saved future output locations. No files moved." +
                (" Individually configured runs retained: " + ", ".join(retained) if retained else ""))
            signal = self.jobs_folder_saved if value.get("jobs_only") else self.series_locations_saved
            signal.emit(str(value["series_root"]))
        self._start(save, done)

    def use_series_locations(self):
        from .series_locations import store_map_with_series_locations
        root, raw = self.series_root, self.editor.toPlainText().encode("utf-8")
        if root is None or not self._loaded:
            return
        def done(text):
            self.editor.setPlainText(text)
            self.status.setText("Series locations placed in the machine-store editor. Review and Save local settings to apply them.")
        self._start(lambda: store_map_with_series_locations(raw, root), done)

    def request_results_migration(self):
        if self._locations is not None and self._results_migration_handler is not None:
            self._results_migration_handler(dict(self._locations,
                project_folder=Path(self.project_location.text()), results_folder=Path(self.results_location.text()),
                replace_custom_results=self.replace_run_results.isChecked()))

    def set_results_migration_handler(self, callback):
        """Bind the ordinary reviewed relocation workflow; saving defaults is separate."""
        self._results_migration_handler = callback
        self._controls()

    def _location_fields(self):
        return (self.series_root, self.project_location.text().strip(), self.results_location.text().strip(),
                self.replace_run_results.isChecked())

    def _preview_results_migration(self, locations):
        from .series_storage_migration import preview_results_migration
        fields = self._location_fields()
        def preview():
            return preview_results_migration(locations["series_root"], locations["project_folder"],
                locations["results_folder"], expected_bytes=locations["manifest_bytes"],
                replace_custom_results=locations["replace_custom_results"])
        def done(plan):
            self._results_migration_plan, self._results_migration_fields = plan, fields
            self.preview.setPlainText("Current Results: " + str(plan.get("old_results_folder", "")) +
                "\nNew Results: " + str(plan.get("new_results_folder", locations["results_folder"])) +
                "\nRename on the same shared storage. Existing products are moved without copying their payloads.")
            self.preview.show()
            self.apply_move_button.show()
            self.status.setText("Review the Results move, close other writers, then choose Move reviewed Results.")
        self._start(preview, done)

    def apply_results_migration(self):
        from .series_storage_migration import apply_results_migration
        from .series_locations import read_series_locations
        if not self.apply_move_button.isEnabled():
            return
        plan, root = self._results_migration_plan, self.series_root
        def apply():
            result = apply_results_migration(plan, writers_quiescent=True)
            return result, read_series_locations(root)
        def done(value):
            result, locations = value
            self._results_migration_plan = self._results_migration_fields = None
            self.apply_move_button.hide()
            self._show_locations(locations)
            self.status.setText("Results moved. Shared locations and compatibility paths saved.")
            self.series_locations_saved.emit(str(root))
        self._start(apply, done)

    def reject(self):
        if self._thread is not None:
            self.status.setText("Wait for the current local metadata operation to finish before closing.")
            return
        if self.editor.toPlainText() != self._displayed_text or self.has_location_edits():
            answer = QMessageBox.question(self, "Discard unsaved settings?", "The store-map or shared-location edits have not been saved.", QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Discard:
                return
        super().reject()

    def closeEvent(self, event):
        self.reject()
        event.setAccepted(not self.isVisible())


class StoreAvailabilityPanel(QGroupBox):
    """Compact Jobs report; explicit checks only, no passive filesystem query."""

    def __init__(self, parent=None, *, path=None, reader=read_store_map,
                 availability=root_availability, dialog_factory=StoreSettingsDialog):
        super().__init__("Machine data stores", parent)
        self.path = Path(path) if path is not None else default_store_map_path()
        self.reader, self.availability, self.dialog_factory = reader, availability, dialog_factory
        self.context = None
        self.generation = 0
        self._thread = self._dialog = None
        self.report = None
        layout = QVBoxLayout(self)
        buttons = QHBoxLayout()
        self.summary = _label("Choose a series. Availability is not checked by opening Jobs.")
        buttons.addWidget(self.summary, 1)
        self.settings_button = QPushButton("Machine stores…")
        self.check_button = QPushButton("Check availability")
        buttons.addWidget(self.settings_button)
        buttons.addWidget(self.check_button)
        layout.addLayout(buttons)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(72)
        self.details.hide()
        layout.addWidget(self.details)
        self.settings_button.clicked.connect(self.open_settings)
        self.check_button.clicked.connect(self.check)
        self.check_button.setEnabled(False)

    def set_context(self, root, series_id):
        context = (str(root or ""), str(series_id or ""))
        if context == self.context:
            return
        self.context = context
        self.generation += 1
        self.report = None
        self.details.clear(); self.details.hide()
        self.summary.setText("Availability not checked for " + context[1] + ". Click Check availability; opening the page starts no work.")
        self.check_button.setEnabled(bool(root and series_id) and self._thread is None)

    def check(self):
        if self._thread is not None or not self.context or not all(self.context):
            return
        root, series = self.context
        generation = self.generation
        def read():
            stores = self.reader(self.path, forbidden_roots=[root])
            return stores, self.availability(stores, series)
        from .ui_metadata_preview import MetadataPreview
        task = MetadataPreview(self)
        self._thread = task
        self.check_button.setEnabled(False)
        self.summary.setText("Checking explicitly configured directories…")
        def finished(value, error):
            self._thread = None
            if generation == self.generation:
                self.details.show()
                if error:
                    self.report = None
                    self.summary.setText("Machine store availability unavailable.")
                    self.details.setPlainText(error)
                else:
                    stores, report = value
                    self.report = report
                    self.summary.setText(stores.machine_id + " has " + series + ": " + " · ".join(row["alias"] + (" ✓" if row["available"] else " unavailable") for row in report["roots"]))
                    self.details.setPlainText("\n".join(row["alias"] + ": " + row["path"] + "\n" + ("Directory present" if row["available"] else row["reason"]) for row in report["roots"]) + "\n\nDirectory presence only. Input hashes, local output admission, capacity, worker eligibility and publication are not verified here. No job was started.")
            self.check_button.setEnabled(bool(self.context and all(self.context)))
            task.deleteLater()
        task.completed.connect(finished)
        task.start(lambda cancelled: read())

    def open_settings(self):
        if self._dialog is not None and self._dialog.isVisible():
            self._dialog.raise_()
            return
        root = self.context[0] if self.context and self.context[0] else None
        dialog = self.dialog_factory(root, self, path=self.path, reader=self.reader)
        self._dialog = dialog
        def changed(path):
            self.path = Path(path)
            self.generation += 1
            self.report = None
            self.details.clear(); self.details.hide()
            self.summary.setText("Local settings changed. Click Check availability to read the new settings.")
        dialog.saved.connect(changed)
        dialog.show()

    def can_close(self):
        return self._thread is None and (self._dialog is None or not self._dialog.isVisible())

    def prepare_close(self):
        """Availability is a disposable read; settings editing/saving is not."""
        self.generation += 1
        if self._thread is not None:
            self._thread.cancel()
            self._thread.deleteLater()
            self._thread = None
