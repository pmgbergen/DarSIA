"""Explain native protocol sources without changing their configured values."""

import ast
from pathlib import Path

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QVBoxLayout
from shiboken6 import isValid

from darsia.gui.ui.file_dialog import NO_FILE_CHOSEN
from darsia.gui.ui.settings import unwrap_composite_widget

from .ui_metadata_preview import MetadataPreview


_SOURCE_PAGES = {
    "protocols.imaging": ("imaging", "Image timestamps", False),
    "protocols.blacklist": ("exclusions", "Excluded images", True),
    "protocols.injection": ("injection", "Injection", False),
    "protocols.pressure_temperature": ("pressure_temperature", "Pressure / temperature", False),
    "protocols.overflow": ("overflow", "Overflow", True),
}

_NOTES = {
    "data": (
        "Image folders: every folder listed here supplies experiment photographs. "
        "A second folder is another part of the image sequence, not automatically a washing-image folder. "
        "No separate experiment/washing role is configured in this list. "
        "Baseline image selects the photograph used as the reference."
    ),
    "imaging": (
        "The imaging protocol gives each photograph its timestamp. Each row links an image folder "
        "to a file and, for workbooks, a worksheet. Image exclusions are optional and have their own section."
    ),
    "exclusions": (
        "Optional. Select a blacklist only when individual photographs must be excluded. "
        "Left column: image folder. Right column: exclusion protocol (optional). "
        "An empty entry means no blacklist for that image folder; a path alone does not mean the file exists."
    ),
    "injection": (
        "The injection protocol supplies the injection periods, positions and rates. "
        "Use an existing protocol to retain multiple injection periods. The optional generation "
        "settings create CSV templates; workbook protocols use their saved worksheet data."
    ),
    "pressure_temperature": (
        "The pressure / temperature protocol supplies the measured conditions over time. "
        "The optional generation settings create CSV templates. "
        "Workbook protocols use the pressure and temperature rows saved in the selected worksheet."
    ),
    "overflow": (
        "Optional overflow measurements belong to this protocol. The overview reads the same "
        "configured sources; selecting a file does not convert its measurements or change their meaning."
    ),
}

_TEMPLATE_KEYS = (
    "imaging_mode", "imaging_interval_seconds", "start_reference", "start_reference_fixed", "start_reference_image",
    "injection_mode", "injection_rate_unit", "injection_rate", "injection_density", "injection_coordinates",
    "pressure_temperature_mode", "pressure_bar", "temperature_celsius",
)


def _form_row(widget):
    """Find the original scalar field row, keeping its label with the input."""
    while widget is not None and widget.parentWidget() is not None:
        parent = widget.parentWidget()
        layout = parent.layout()
        if isinstance(layout, QFormLayout) and layout.getWidgetPosition(widget)[0] >= 0:
            return layout, widget
        widget = parent
    return None


def _insert_note(page, widget, *, first=False):
    layout = page.layout()
    if isinstance(layout, QFormLayout):
        layout.insertRow(0 if first else layout.rowCount(), widget)
    elif first and hasattr(layout, "insertWidget"):
        layout.insertWidget(0, widget)
    else:
        layout.addWidget(widget)


def _source_reference(text):
    """Read the original editor's plain path or [file, worksheet] spelling."""
    text = str(text).strip()
    if not text or text == NO_FILE_CHOSEN:
        return None
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        value = text
    if isinstance(value, (tuple, list)):
        if len(value) != 2 or not all(isinstance(part, str) for part in value):
            raise ValueError("Use a file path or [file, worksheet].")
        if not value[0].strip() or value[0] == NO_FILE_CHOSEN:
            return None
        return value[0], value[1]
    if not isinstance(value, str):
        raise ValueError("Use a file path or [file, worksheet].")
    return value, None


def _check_sources(entries, config_file, cancelled):
    """Only check configured file metadata, once per unique resolved file."""
    from darsia.fluidflower.storage_paths import storage_path_remapper
    from darsia.presets.workflows.config.protocols import _parse_protocol_value

    remap = None
    results, files = {}, {}
    for token, text in entries:
        if cancelled():
            return {}
        try:
            reference = _source_reference(text)
            if reference is None:
                results[token] = "Not selected"
                continue
            raw_path, sheet = reference
            if remap is None:
                remap = storage_path_remapper()
            path = _parse_protocol_value(remap(raw_path), Path(config_file) if config_file else None)
            key = str(path)
            if key not in files:
                files[key] = "File found" if path.is_file() else "File missing"
            suffix = f" · worksheet: {sheet} (not checked)" if sheet else ""
            results[token] = files[key] + suffix
        except (OSError, ValueError, TypeError) as error:
            results[token] = f"Unavailable: {error}"
    return results


class ProtocolFieldNotes(QObject):
    """Metadata-only companion of the existing settings inputs."""

    def __init__(self, window, pages):
        super().__init__(next(iter(pages.values())))
        self.window, self.pages = window, pages
        self.inputs = window.settings_inputs
        self.reader = MetadataPreview(self)
        self.reader.completed.connect(self._loaded)
        self.rows = {}
        self.groups = {}
        self.connected = set()
        self.folder_labels = {}
        self.notes = {}
        self.template_rows = []
        self._using_workbook = False
        self.config_file = window.config_file
        for key, text in _NOTES.items():
            page = pages.get(key)
            if page is not None:
                note = QLabel(text)
                note.setWordWrap(True)
                note.setObjectName("protocolSourceExplanation")
                _insert_note(page, note, first=True)
                self.notes[key] = note
        # The original form creates later folder rows on the next event-loop turn.
        QTimer.singleShot(0, self.refresh)

    def _group(self, page_key):
        if page_key in self.groups:
            return self.groups[page_key]
        page = self.pages.get(page_key, self.pages.get("protocols"))
        if page is None:
            return None
        group = QGroupBox("Configured source files")
        layout = QVBoxLayout(group)
        note = QLabel("File presence is checked here; table contents and worksheet names are not validated.")
        note.setWordWrap(True)
        layout.addWidget(note)
        button = QPushButton("Check configured files")
        button.clicked.connect(self.refresh)
        layout.addWidget(button)
        _insert_note(page, group)
        self.groups[page_key] = group
        return group

    def _edits(self, key):
        value = self.inputs.get(key)
        if isinstance(value, dict) and value.get("path_map"):
            return [(f"Image folder {index}", edit) for index, (_, edit) in enumerate(value["rows"], 1)]
        value = unwrap_composite_widget(value)
        return [("", value)] if isinstance(value, QLineEdit) else []

    def _changed(self):
        self.reader.cancel()
        self._update_templates()
        for _, (_, label) in self.rows.items():
            label.setText(label.property("source_caption") + " · Not checked after editing")

    def refresh(self):
        folders = self.inputs.get("data.folders")
        if isinstance(folders, list):
            for index, edit in enumerate(folders, 1):
                if not isinstance(edit, QLineEdit) or not isValid(edit):
                    continue
                if id(edit) not in self.folder_labels:
                    label = QLabel(f"Experiment images {index}")
                    parent_layout = edit.parentWidget().layout()
                    if hasattr(parent_layout, "insertWidget"):
                        parent_layout.insertWidget(0, label)
                        self.folder_labels[id(edit)] = label
                edit.setToolTip("Experiment photographs from this folder enter the image sequence. No washing-image role is assigned here.")
        for key, (page_key, name, optional) in _SOURCE_PAGES.items():
            for index, (folder, edit) in enumerate(self._edits(key)):
                token = (key, index)
                if token not in self.rows:
                    group = self._group(page_key)
                    if group is None:
                        continue
                    label = QLabel()
                    label.setWordWrap(True)
                    label.setObjectName("protocolSourceStatus")
                    caption = name + (f" · {folder}" if folder else "") + (" (optional)" if optional else "")
                    label.setProperty("source_caption", caption)
                    group.layout().insertWidget(group.layout().count() - 1, label)
                    self.rows[token] = edit, label
                if id(edit) not in self.connected:
                    self.connected.add(id(edit))
                    edit.textChanged.connect(self._changed)
                    edit.editingFinished.connect(self.refresh)
                    if key == "protocols.blacklist":
                        edit.setPlaceholderText("No exclusion protocol selected (optional)")
                        edit.setToolTip("Exclusion protocol (optional). Leave empty for no blacklist in this image folder.")
                _, label = self.rows[token]
                value = edit.text().strip() and edit.text().strip() != NO_FILE_CHOSEN
                label.setText(label.property("source_caption") + (" · Checking file…" if value else " · Not selected"))
        blacklist = self.inputs.get("protocols.blacklist")
        if isinstance(blacklist, dict):
            for folder, _ in blacklist.get("rows", ()):
                folder.setToolTip("Image folder. This is not the exclusion-protocol file.")
        self._update_templates()
        entries = [(token, edit.text()) for token, (edit, _) in self.rows.items() if isValid(edit)]
        self.reader.start(lambda cancelled: _check_sources(entries, self.config_file, cancelled))

    def _update_templates(self):
        using_workbook = False
        for key in ("protocols.imaging", "protocols.injection", "protocols.pressure_temperature"):
            for _, edit in self._edits(key):
                try:
                    reference = _source_reference(edit.text())
                except ValueError:
                    continue
                if reference and Path(reference[0]).suffix.casefold() in {".xls", ".xlsx"}:
                    using_workbook = True
        if using_workbook and not self._using_workbook:
            self.template_rows = []
            for name in _TEMPLATE_KEYS:
                value = self.inputs.get("protocols." + name)
                widgets = ([row[-1] if isinstance(row, tuple) else row for row in value.get("rows", ())]
                           if isinstance(value, dict) else [value])
                seen = set()
                for item in widgets:
                    row = _form_row(item) if item is not None else None
                    if row is not None:
                        layout, widget = row
                        if id(widget) not in seen:
                            seen.add(id(widget))
                            self.template_rows.append((layout, widget, layout.isRowVisible(widget)))
                            layout.setRowVisible(widget, False)
        elif self._using_workbook and not using_workbook:
            for layout, widget, visible in self.template_rows:
                if isValid(widget):
                    layout.setRowVisible(widget, visible)
            self.template_rows = []
        self._using_workbook = using_workbook
        for key in ("imaging", "injection", "pressure_temperature"):
            if key in self.notes:
                self.notes[key].setText(_NOTES[key] + (
                    " Workbook-backed protocol set: saved table data are used; CSV template settings are inactive."
                    if using_workbook else ""))

    def _loaded(self, values, error):
        for token, (_, label) in self.rows.items():
            if isValid(label):
                label.setText(label.property("source_caption") + " · " +
                              ("Unavailable: " + error if error else values.get(token, "Not selected")))


def decorate_protocol_fields(window, pages):
    """Add explanatory context after building the unchanged native form fields."""
    if not pages:
        return None
    return ProtocolFieldNotes(window, pages)
