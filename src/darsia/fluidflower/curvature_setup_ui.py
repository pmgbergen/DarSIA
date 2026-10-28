"""Embedded optional stages using DarSIA's original curvature form and engine."""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QCheckBox, QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from darsia.fluidflower.crop_wizard_ui import _install_sync_policy, protect_curvature_form


class NativeCurvatureWorkflow:
    def __init__(self, window):
        self.window = window
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.master = QCheckBox("Curvature correction")
        self.master.setToolTip("Enable initial bulge, bulge and stretch. The saved crop stays independent.")
        layout.addWidget(self.master)
        self.context_note = QLabel("Choose a run configuration.")
        self.context_note.setWordWrap(True)
        layout.addWidget(self.context_note)
        self.note = QLabel(
            "Optional. Off disables only the three curvature stages; their values and "
            "stage choices are retained. Reference points and crop stay separate. "
            "Save stores the settings; calculations run through the existing Setup/Jobs workflow."
        )
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.options = QWidget()
        self.options_layout = QVBoxLayout(self.options)
        self.options_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.options)
        self.save_button = QPushButton("Save curvature settings")
        layout.addWidget(self.save_button)
        self.master.toggled.connect(self._toggle)
        self.save_button.clicked.connect(self.save)
        self.inputs = {}
        self._active = False
        self._opening = False
        self._saving = False
        self._config_path = None
        self._context_key = None
        _install_sync_policy(window)
        original_sync = window.settings_factory._sync_settings_inputs_to_config_dict

        def sync(*args, **kwargs):
            if not self._saving:
                return original_sync(*args, **kwargs)
            with self._own_inputs():
                return original_sync(*args, **kwargs)

        sync.__wrapped__ = original_sync
        window.settings_factory._sync_settings_inputs_to_config_dict = sync
        self.set_active(False)

    @contextmanager
    def _own_inputs(self):
        window = self.window
        other = window.settings_inputs
        window.settings_inputs = self.inputs
        try:
            yield
        finally:
            self.inputs = window.settings_inputs
            window.settings_inputs = other

    def _key(self):
        window = self.window
        return (window.phase_model_workspace.series_id, window.config_file,
                id(window.config_dict), getattr(window, "_config_stamp", None))

    def set_active(self, active):
        self._active = bool(active)
        self.master.setEnabled(self._active and self._config_path is not None)
        self.save_button.setEnabled(self._active and self._config_path is not None)
        if self._active:
            self.refresh()

    def refresh(self):
        if not self._active or self._opening:
            return
        self._opening = True
        try:
            self._refresh()
        finally:
            self._opening = False

    def _refresh(self):
        from darsia.fluidflower.native_darsia_app import _same_file
        from darsia.gui.ui.preprocessing_wizard import _corrections_fields

        window = self.window
        window._ensure_run_loaded()
        selected = window.run_combo.currentData()
        if selected and not _same_file(selected, window.config_file):
            if not window._load_run_config(selected):
                self._config_path = None
                self.set_active(False)
                return
        if not selected or not window.config_file or window._config_series != window.phase_model_workspace.series_id:
            self._config_path = None
            self.context_note.setText("Choose and load a run configuration.")
            self.master.setEnabled(False)
            self.options.setEnabled(False)
            self.save_button.setEnabled(False)
            return
        if self._context_key == self._key():
            self.master.setEnabled(True)
            self.save_button.setEnabled(True)
            return
        # Settle the existing editor's pending values, without moving its widgets.
        protect_curvature_form(window)
        window.settings_factory._sync_settings_inputs_to_config_dict()
        for index in reversed(range(self.options_layout.count())):
            item = self.options_layout.takeAt(index)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.inputs = {}
        holder = QWidget()
        form = QFormLayout(holder)
        self.options_layout.addWidget(holder)
        fields = deepcopy(_corrections_fields("curvature"))
        for field in fields:
            field.pop("active_list_key", None)
            field["name"] = "Curvature stages"
            field["fields"] = [value for value in field["fields"]
                               if value["key"] != "corrections.curvature.optional_stages_active"]
        corrections = window.config_dict.get("corrections", {})
        configured = corrections.get("curvature", {})
        outer_on = bool(configured) and (corrections.get("active") is None or "curvature" in corrections["active"])
        stage_on = any(configured.get(name) and (configured.get("active") is None or name in configured["active"])
                       for name in ("init", "bulge", "stretch"))
        with QSignalBlocker(self.master):
            self.master.setChecked(bool(outer_on and stage_on and configured.get("optional_stages_active", True)))
        with self._own_inputs():
            window.settings_factory.build_tab_form(form, fields)
            self.inputs["corrections.curvature"]["optional_master"] = self.master
            self.inputs["corrections.curvature.optional_stages_active"] = self.master
            protect_curvature_form(window, stages_only=True)
        self._config_path = window.config_file
        self._context_key = self._key()
        self.context_note.setText(f"Run {window.run_combo.currentText().upper()} · {Path(window.config_file).name}")
        self.master.setEnabled(True)
        self.save_button.setEnabled(True)
        self.options.setEnabled(self.master.isChecked())

    def _toggle(self, enabled):
        self.options.setEnabled(enabled)
        if enabled and self.inputs:
            controls = [self.inputs[f"corrections.curvature.{name}"]["checkbox"]
                        for name in ("init", "bulge", "stretch")]
            if not any(control.isChecked() for control in controls):
                controls[0].setChecked(True)

    def save(self):
        from darsia.fluidflower.native_darsia_app import _same_file

        window = self.window
        if not (self._active and self._config_path
                and window._config_series == window.phase_model_workspace.series_id
                and _same_file(window.config_file, self._config_path)
                and _same_file(window.run_combo.currentData(), self._config_path)):
            window.print_log("Choose and load this run again before saving curvature settings.")
            return
        self._saving = True
        try:
            window.settings_factory.save_settings()
        finally:
            self._saving = False
        self._context_key = self._key()
