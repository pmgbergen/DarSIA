"""Inline presentation of the original production preparation controls."""

from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QBoxLayout, QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from . import analysis_plan_sources as plan_sources
from .analysis_preparation_ui import AnalysisPreparationDialog, adopt_prepared_plan
from .production_preparation_ui import ProductionPreparationDialog
from .workflow_section import WorkflowSection, WorkflowWorkspace


class _InlinePreparation:
    """Retain the original producer/worker; collect approval on the same page."""

    def __init__(self, *args, retired, **kwargs):
        super().__init__(*args, **kwargs)
        self._retired_callback = retired
        self._active_ui_context = self._reviewed_plan = self._reviewed_ui_context = None
        self.setWindowFlags(Qt.WindowType.Widget)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            if button.text() == "Close":
                button.hide()
        approval = QWidget()
        form = QFormLayout(approval)
        self.approval_binding = QLabel("No plan prepared.")
        self.approval_binding.setWordWrap(True)
        self.approval_binding.setTextFormat(Qt.TextFormat.PlainText)
        form.addRow(self.approval_binding)
        self.approval_actor = QLineEdit()
        self.approval_reason = QLineEdit()
        form.addRow("Approved by", self.approval_actor)
        form.addRow("Reason for adopting these exact inputs", self.approval_reason)
        self.approval_confirmed = QCheckBox("Adopt exactly this plan")
        form.addRow(self.approval_confirmed)
        self.confirm_adoption = QPushButton("Adopt prepared plan")
        self.confirm_adoption.setAutoDefault(False)
        self.confirm_adoption.clicked.connect(self._confirm_adoption)
        form.addRow(self.confirm_adoption)
        self.approval_section = WorkflowSection(None, "Review and adopt prepared plan…", approval)
        self.layout().addWidget(self.approval_section)
        for field in (self.approval_actor, self.approval_reason):
            field.textChanged.connect(self._approval_controls)
        self.approval_confirmed.toggled.connect(self._approval_controls)
        from .series_geometry import default_actor
        self.approval_actor.setText(default_actor())
        self._approval_controls()
        self._add_series_inputs()

    def _add_series_inputs(self):
        """Find the inputs where the series keeps them; one button prepares from them.

        The original file choices stay available behind "Choose the input files
        myself". Reading the inputs is a disposable read when the section first
        shows, and again on Check again or another stage.
        """
        from .ui_metadata_preview import MetadataPreview
        self._series_sources = None
        self._sources_preview = MetadataPreview(self)
        self._sources_preview.completed.connect(self._sources_returned)
        block = QWidget()
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        self.series_sources = QLabel("The series' analysis inputs are read when this section opens.")
        self.series_sources.setWordWrap(True)
        self.series_sources.setTextFormat(Qt.TextFormat.PlainText)
        self.series_sources.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.series_sources)
        row = QHBoxLayout()
        self.prepare_from_series_button = QPushButton("Prepare plan from the series")
        self.prepare_from_series_button.setToolTip(
            "Prepares a plan for review from the inputs listed above; no job or worker starts.")
        self.prepare_from_series_button.clicked.connect(self.prepare_from_series)
        self.check_sources_button = QPushButton("Check again")
        self.check_sources_button.clicked.connect(self.check_series_sources)
        for button in (self.prepare_from_series_button, self.check_sources_button):
            button.setAutoDefault(False)
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        self.choose_files = QCheckBox("Choose the input files myself")
        self.choose_files.toggled.connect(self._show_file_rows)
        layout.addWidget(self.choose_files)
        self.layout().insertWidget(1, block)
        self.stage.currentIndexChanged.connect(self.check_series_sources)
        self._show_file_rows(False)
        self._series_controls()

    def _show_file_rows(self, shown):
        """Show the original file choices only when the user chooses files."""
        form = next(form for form in self.findChildren(QFormLayout) if form.indexOf(self.stage) >= 0)
        for field in (self.inputs, self.geometry, self.pressure, self.calibration, self.phase1_jobs,
                      self.collect_button, self.collect_note, self.name, self.folder):
            row = _form_row(form, field)
            if row >= 0:
                form.setRowVisible(row, shown)
        self.prepare_button.setVisible(shown)

    def showEvent(self, event):
        super().showEvent(event)
        if self._series_sources is None and not self._sources_preview.busy:
            self.check_series_sources()

    def check_series_sources(self, *_args):
        """Read the current stage's inputs where the series keeps them."""
        self._series_sources = None
        if not self.series_root:
            self.series_sources.setText("Choose a series first.")
            self._series_controls()
            return
        series_root, stage = self.series_root, self.stage.currentData()
        calibration = deepcopy(getattr(self, "_production_calibration", None))

        def find(_cancelled):
            if stage == "phase1":
                return stage, plan_sources.phase_one_sources(series_root)
            return stage, plan_sources.phase_two_sources(series_root, calibration)

        self.series_sources.setText("Reading the series' analysis inputs…")
        self._sources_preview.start(find)
        self._series_controls()

    @Slot(object, str)
    def _sources_returned(self, value, error):
        if error:
            self.series_sources.setText("The series' analysis inputs could not be read: " + error)
        elif value[0] == self.stage.currentData():
            self._series_sources = value[1]
            self.series_sources.setText(plan_sources.sources_text(value[1]))
        self._series_controls()

    def _series_controls(self):
        found = self._series_sources is not None and not any(s.missing for s in self._series_sources)
        self.prepare_from_series_button.setEnabled(
            found and self._thread is None and not self._sources_preview.busy)
        self.check_sources_button.setEnabled(not self._sources_preview.busy)

    def prepare_from_series(self):
        """Prepare a plan for review from the found inputs, in a fresh folder of the series."""
        if not self.prepare_from_series_button.isEnabled():
            return
        found = list(self._series_sources)
        if self.stage.currentData() != "phase1":
            folder, _manifest = plan_sources.fresh_plan_paths(self.series_root, "mass")
            self.inputs.setText(found[0].bound["path"])
            self.name.setText("mass")
            self.folder.setText(str(folder))
            self.prepare()
            return
        folder, manifest = plan_sources.fresh_plan_paths(self.series_root, "phase-maps")
        _runs, geometry, pressure = (source.bound for source in found)
        self.geometry.setText(geometry["path"])
        self.pressure.setText(pressure["path"])
        self.inputs.setText(str(manifest))
        self.name.setText("phase-maps")
        self.folder.setText(str(folder))
        series_root, probability = self.series_root, self.probability.isChecked()
        repository = Path(__file__).resolve().parents[3]
        self.prepared_result = None
        self._run(lambda: plan_sources.prepare_phase_one_plan(
            series_root, found, folder=folder, manifest=manifest, gas_probability=probability,
            repository=repository, job_id="phase-maps"))

    def _ui_context(self):
        values = [str(self.series_root), str(self._source_panel.series_root), self._source_panel.series_id]
        for name in ("stage", "inputs", "geometry", "pressure", "calibration", "phase1_jobs",
                     "name", "folder", "probability", "profile", "stored_resolution"):
            control = getattr(self, name, None)
            if isinstance(control, QLineEdit):
                values.append(control.text())
            elif isinstance(control, QComboBox):
                values.append(control.currentData())
            elif isinstance(control, QCheckBox):
                values.append(control.isChecked())
        return tuple(values)

    def _run(self, call, **kwargs):
        if self._thread is not None:
            return
        self._active_ui_context = self._ui_context()
        self.confirm_adoption.setEnabled(False)
        self.prepare_from_series_button.setEnabled(False)
        return super()._run(call, **kwargs)

    def _done(self, result):
        if self._ui_context() != self._active_ui_context:
            self.prepared_result = self._reviewed_plan = None
            self.status.setText("The series or input choices changed. Earlier files are preserved; prepare and review the current choices.")
            return
        super()._done(result)

    def _closed(self):
        super()._closed()
        self._active_ui_context = None
        self._approval_controls()
        self._series_controls()
        self._retired_callback()

    def adopt(self):
        if (not self.prepared_result or self._thread is not None
                or self.prepared_result.get("adopted_for_execution")):
            return
        if (isinstance(self, ProductionPreparationDialog)
                and self._prepared_context != self._preparation_context()):
            self._profile_changed()
            return
        self._reviewed_plan = dict(self.prepared_result["plan"])
        self._reviewed_ui_context = self._ui_context()
        self.approval_binding.setText(
            self._reviewed_plan["path"] + "\nSHA-256: " + self._reviewed_plan["sha256"]
            + "\nAdoption creates no job and starts no worker.")
        self.approval_confirmed.setChecked(False)
        self.approval_section.set_expanded(True)
        self.approval_actor.setFocus()
        self._approval_controls()

    def _approval_controls(self, *_args):
        self.confirm_adoption.setEnabled(
            self._thread is None and self._reviewed_plan is not None
            and not (self.prepared_result or {}).get("adopted_for_execution")
            and bool(self.approval_actor.text().strip())
            and bool(self.approval_reason.text().strip())
            and self.approval_confirmed.isChecked())

    def _confirm_adoption(self):
        if not self.confirm_adoption.isEnabled():
            return
        bound = (self.prepared_result or {}).get("plan")
        if (bound != self._reviewed_plan or self._reviewed_ui_context != self._ui_context()
                or (self.prepared_result or {}).get("adopted_for_execution")):
            self._reviewed_plan = None
            self._approval_controls()
            self.status.setText("The reviewed plan or choices changed. Review the current prepared plan before adopting it.")
            return
        bound = dict(bound)
        actor, reason = self.approval_actor.text().strip(), self.approval_reason.text().strip()
        source = Path(bound["path"])
        self._run(lambda: adopt_prepared_plan(
            source, source.with_name("execution.adopted.toml"),
            expected_sha256=bound["sha256"], actor=actor, reason=reason), adopting=True)

    def reject(self):
        if not self.can_close():
            self.status.setText("Plan preparation or adoption is still finishing; wait for it to return.")


def _form_row(form, widget):
    """The row of ``widget`` in ``form``, also inside a row's own layout; -1 if absent."""
    row, _role = form.getWidgetPosition(widget)
    if row >= 0:
        return row
    for row in range(form.rowCount()):
        for role in (QFormLayout.ItemRole.FieldRole, QFormLayout.ItemRole.SpanningRole):
            item = form.itemAt(row, role)
            if item is not None and item.layout() is not None and item.layout().indexOf(widget) >= 0:
                return row
    return -1


class _InlineAnalysisPreparation(_InlinePreparation, AnalysisPreparationDialog):
    pass


class _InlineProductionPreparation(_InlinePreparation, ProductionPreparationDialog):
    pass


class NativeProductionSections:
    """Keep all existing choices and the selected plan editor on the page."""

    def __init__(self, page):
        self.page, self._key = page, None
        self.sections = []
        layout = page.scroll.widget().layout()
        for number, group in enumerate((page.what, page.files, page.where, page.actions), 1):
            index, title = layout.indexOf(group), group.title()
            layout.removeWidget(group)
            group.setTitle("")
            group.setFlat(True)
            section = WorkflowSection(number, title, group)
            layout.insertWidget(index, section)
            self.sections.append(section)
        outer = page.layout()
        outer.removeWidget(page.scroll)
        self.editor_scroll = QScrollArea()
        self.editor_scroll.setWidgetResizable(True)
        self.editor_body = QWidget()
        self.editor_layout = QVBoxLayout(self.editor_body)
        self.editor_layout.setContentsMargins(0, 0, 0, 0)
        self.editor_scroll.setWidget(self.editor_body)
        self.editor_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.splitter = WorkflowWorkspace(workspace=self.editor_scroll)
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                self.splitter.controls_layout.addWidget(item.widget())
        page.scroll.deleteLater()
        page.scroll = self.splitter.controls_scroll
        self.results = QFormLayout()
        self.results.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.editor_layout.addLayout(self.results)
        for group, widgets in (
            (page.what, (page.method_label, page.form.binding_label, page.selection_label,
                         page.pipeline_label, page.pressure_label, page.probability_label)),
            (page.files, (page.form.estimate_label, page.precision_label)),
        ):
            form = group.layout()
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            for widget in widgets:
                row = form.takeRow(widget)
                if row.labelItem is not None:
                    self.results.addRow(row.labelItem.widget(), widget)
                else:
                    self.results.addRow(widget)
        self.editor_layout.addWidget(page.action_label)
        page.target_layout.setDirection(QBoxLayout.Direction.TopToBottom)
        outer.addWidget(self.splitter)
        page.prepare_plan_button.clicked.disconnect()
        page.prepare_plan_button.clicked.connect(self.open)
        page.prepare_plan = self.open
        self.sync()

    def sync(self):
        page = self.page
        choice = page.calibration_combo.currentData()
        if (page.facts or {}).get("pipeline") == "native-phase-statistics":
            choice = None
        key = (str(page.panel.series_root), page.panel.series_id, deepcopy(choice))
        previous = page._preparation_dialog
        if previous is not None and key == self._key:
            if previous.can_close():
                previous.setEnabled(bool(page.panel.series_root))
            return
        if previous is not None and not previous.can_close():
            previous.setEnabled(False)
            previous.status.setText("The selected series changed. Waiting for the earlier preparation to finish before showing the new choices.")
            return
        if previous is not None:
            self.splitter.controls_layout.removeWidget(self.preparation_section)
            self.splitter.controls_layout.removeWidget(previous.approval_section)
            self.editor_layout.removeWidget(self.preparation_results)
            self.preparation_section.hide()
            previous.approval_section.hide()
            self.preparation_results.hide()
            self.preparation_section.deleteLater()
            previous.approval_section.deleteLater()
            self.preparation_results.deleteLater()
        kwargs = dict(parent=self.splitter.controls_widget, retired=self.sync)
        if choice is None:
            dialog = _InlineAnalysisPreparation(page.panel, **kwargs)
        else:
            dialog = _InlineProductionPreparation(page.panel, choice,
                storage_profile=page.form.profile_combo.currentData(), **kwargs)
        page._preparation_dialog = dialog
        self._key = key
        dialog.adopted.connect(page._prepared_adopted)
        dialog.setEnabled(bool(page.panel.series_root))
        self.preparation_section = WorkflowSection(None, "Prepare a new analysis plan…", dialog)
        self.splitter.controls_layout.addWidget(self.preparation_section)
        self.splitter.controls_layout.addWidget(dialog.approval_section)
        self.preparation_results = QWidget()
        result_layout = QVBoxLayout(self.preparation_results)
        # The found and missing inputs name long paths; they read beside the status.
        result_layout.addWidget(dialog.series_sources)
        result_layout.addWidget(dialog.status)
        result_layout.addWidget(dialog.approval_binding)
        self.editor_layout.addWidget(self.preparation_results)
        for form in dialog.findChildren(QFormLayout):
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        # File selectors and actions must remain usable in the narrow column.
        for row in dialog.findChildren(QBoxLayout):
            row.setDirection(QBoxLayout.Direction.TopToBottom)
        dialog.approval_section.content.layout().setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapLongRows)

    def open(self):
        self.sync()
        self.preparation_section.set_expanded(True)
        self.splitter.controls_scroll.ensureWidgetVisible(self.preparation_section.toggle)
        self.page._preparation_dialog.inputs.setFocus()
