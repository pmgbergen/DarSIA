"""Native-only layout of the original mass calibration widgets."""

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QTextDocument, QTextOption
from PySide6.QtWidgets import QFormLayout, QLabel, QSizePolicy, QSplitter, QStackedWidget, QVBoxLayout, QWidget

from .mass_calibration_display import MassCalibrationDisplay
from .native_normalisation_layout import _WrappedTextHeight
from .workflow_section import WorkflowPanelColumn, WorkflowWorkspace


class _MassWrappedTextHeight(_WrappedTextHeight):
    def eventFilter(self, watched, event):
        # QLabel can recreate its document layout during paint after a resize.
        # Apply wrapping after that lazy layout, using the same queued refresh.
        if isinstance(watched, QLabel) and event.type() == QEvent.Type.Paint:
            if not self._pending:
                self._pending = True
                QTimer.singleShot(0, self, self._refresh)
        return super().eventFilter(watched, event)

    def _refresh(self):
        # QLabel's ordinary word wrap leaves a long path component unbroken.
        # Its selectable text document can wrap within that component without
        # adding invisible characters to the displayed or copied source path.
        form_fields = []
        for parent, labels in self._by_parent.items():
            for label in labels:
                if isinstance(parent.layout(), QFormLayout):
                    form_fields.append((parent, label, label.minimumHeight()))
                for document in label.findChildren(QTextDocument):
                    option = document.defaultTextOption()
                    wrap = QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere
                    if option.wrapMode() != wrap:
                        option.setWrapMode(wrap)
                        document.setDefaultTextOption(option)
        super()._refresh()
        for parent, label, previous_height in form_fields:
            if label.minimumHeight() != previous_height:
                # QFormLayout also caches the old item's one-line height.
                # Renew only that layout item, retaining the original widget.
                layout = parent.layout()
                row, role = layout.getWidgetPosition(label)
                layout.removeWidget(label)
                layout.setWidget(row, role, label)
                layout.invalidate()
                parent.updateGeometry()


def _empty(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.layout() is not None:
            _empty(item.layout())


def _column(*widgets):
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    for widget in widgets:
        layout.addWidget(widget)
    return panel


def _form(rows, names):
    panel = QWidget()
    layout = QFormLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
    for name in names:
        label, field = rows[name]
        layout.addRow(label, field)
    return panel


def _take_form_rows(form):
    """Retain labelled and spanning widgets when adapting the original form."""
    rows, spanning = {}, []
    while form.rowCount():
        row = form.takeRow(0)
        label = row.labelItem.widget() if row.labelItem is not None else None
        field = row.fieldItem.widget() if row.fieldItem is not None else None
        if field is None and row.fieldItem is not None and row.fieldItem.layout() is not None:
            field = QWidget()
            field.setLayout(row.fieldItem.layout())
        if label is not None and field is not None:
            rows[label.text()] = label, field
        elif field is not None:
            spanning.append(field)
        elif label is not None:
            spanning.append(label)
    return rows, spanning


class NativeMassCalibrationDisplay(MassCalibrationDisplay):
    """Keep the original owners and callbacks while arranging their controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = self.layout()
        title = outer.itemAt(0).widget()
        modes = outer.itemAt(2).layout()
        mode_label = modes.itemAt(0).widget()
        saved = self.columns.widget(0)
        panel = self.refit_panel
        form = panel.layout().itemAt(1).layout()
        rows, spanning = _take_form_rows(form)
        source_locations = (panel.source_folder_button, panel.source_folder_panel)
        other_spanning = [widget for widget in spanning if widget not in source_locations]
        _empty(outer)
        _empty(panel.layout())
        self.refit_area.takeWidget()
        panel.setParent(self)
        self.columns.hide()

        self.workflow = WorkflowWorkspace()
        outer.setContentsMargins(0, 0, 0, 0)
        # Keep the original owner visible with the page: its show/hide events
        # own the saved-job refresh timer. Folding a section must not stop it.
        job_title = QLabel(panel.title())
        job_title.setWordWrap(True)
        panel.setTitle("")
        panel.setFlat(True)
        panel.layout().setContentsMargins(0, 0, 0, 0)
        panel.layout().addWidget(self.workflow)
        outer.addWidget(panel)
        self.panels = WorkflowPanelColumn()
        self.workflow.controls_layout.addWidget(self.panels)
        self.sections = {}

        saved_controls = _column(title, self.source, mode_label,
                                 self.manual_button, self.automatic_button, self.mode_notice)
        title.setText("Saved calibration")
        # Execution choices have their own active controls. These original
        # read-only radio indicators must not resemble a second mode selector.
        for widget in (mode_label, self.manual_button, self.automatic_button):
            widget.hide()
        parameters = _form(rows, ("Chemistry model", "Calibration", "Fitted parameters and dilution", "Measured inputs",
                                 "Images used by the fit", "Run roles", "Starting parameters", "Rules", "Phase-one job",
                                 "Folds", "Free CO₂-gas target", "Overflow sources"))
        self.phase_parameters = _column(parameters, *other_spanning, *source_locations)
        self.overflow_policy = QLabel(
            "Use each run's own reviewed overflow measurement directly for the free/dissolved CO₂ phase split. "
            "Missing measurements use an explicitly labelled estimate from comparable fit runs at the nearest pressure, "
            "with injected-amount and pressure/temperature conversion. Outside all reference coverage the phase model "
            "is retained and labelled as having no overflow reference. Total mass is calibrated against injected mass.")
        self.overflow_policy.setWordWrap(True)
        self.phase_parameters.layout().addWidget(self.overflow_policy)
        from .shared_calibration_cohort import SharedCalibrationCohort
        self.shared_cohort = SharedCalibrationCohort(panel)
        self.shared_cohort.saved.connect(self._shared_cohort_saved)
        self.phase_parameters.layout().insertWidget(0, self.shared_cohort)
        job = _column(job_title, panel.notice, _form(rows, ("Jobs folder", "Calibration name")),
                      panel.create_button, panel.start_button, panel.start_reason,
                      panel.open_button, panel.refresh_button, panel.job_label)
        self.job_controls = job
        # The operative overflow choice belongs to the parameter form above.
        # Retain the original diagnostic owner for compatibility, without a
        # second visible mode selector or a separate navigation panel.
        self.overflow_choice.hide()
        from .common_calibration_actions import OriginalCalibrationControls, PhaseColourPathPage
        self.original_controls = OriginalCalibrationControls()
        self.phase_colour_paths = PhaseColourPathPage()
        self.original_model = QLabel(
            "Colour-path signal with simple flash. Automatic calibration fits "
            "the signal values to injected mass and retains the saved flash split. "
            "Use the automatic fit or open the interactive controls in Calibration job.")
        self.original_model.setWordWrap(True)
        self.original_inputs = QLabel()
        self.original_inputs.setWordWrap(True)
        self.model_context = QStackedWidget()
        self.model_context.addWidget(self.phase_parameters)
        self.model_context.addWidget(_column(self.original_model, self.original_inputs,
                                            self.original_controls.parameters))
        self.job_context = QStackedWidget()
        self.job_context.addWidget(job)
        self.job_context.addWidget(self.original_controls)
        for key, name, content in (
            ("chemistry", "Model parameters", self.model_context),
            ("job", "Calibration job", self.job_context),
        ):
            self.panels.addTab(content, name)
            self.sections[key] = self.panels.sections[-1]

        self.saved_calibration_page = _column(saved_controls, saved)
        self.saved_calibration_page.setObjectName("nativeSavedCalibrationPage")
        self.saved_calibration_page.layout().setStretch(1, 1)
        self.saved_calibration_page.setParent(self)
        self.saved_calibration_page.hide()
        self.results = QSplitter(Qt.Orientation.Vertical)
        self.results.setChildrenCollapsible(False)
        recipe_label, recipe_table = rows["Measured recipes — read only"]
        self.job_results = _column(recipe_label, recipe_table, panel.tasks, panel.result_label)
        self.job_results.layout().setStretch(2, 1)
        self.results.addWidget(self.job_results)
        self.results.setStretchFactor(0, 1)
        workspace = QVBoxLayout(self.workflow.workspace)
        workspace.setContentsMargins(8, 0, 0, 0)
        workspace.addWidget(self.results)
        for label in self.findChildren(QLabel):
            if label.wordWrap():
                # A full path can be one unbroken word. Its natural width must
                # not set the scroll-column width; retain the exact label text.
                policy = label.sizePolicy()
                policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
                label.setSizePolicy(policy)
                label.setMinimumWidth(0)
                label.setTextInteractionFlags(
                    label.textInteractionFlags() | Qt.TextInteractionFlag.TextSelectableByMouse)
        self._wrapped_text_height = _MassWrappedTextHeight(self)
        self._common_method = ""
        self._common_root = None
        self._show_refit_controls(False)

    def install_preparation(self, window):
        from .calibration_preparation import CalibrationPreparation
        self.preparation = CalibrationPreparation(window, self.refit_panel, self)
        self.job_controls.layout().insertWidget(2, self.preparation)

    def _show_refit_controls(self, offered):
        for name in ("chemistry", "job"):
            self.sections[name].show()
            self.sections[name].setEnabled(offered)
        self.job_results.setVisible(offered)

    def set_refit_series(self, series_root):
        super().set_refit_series(series_root)
        self.shared_cohort.set_series(series_root)
        self.overflow_choice.hide()
        self._show_refit_controls(series_root is not None)

    def _shared_cohort_saved(self, root):
        """Refresh the operative roles after the owned save worker retires."""
        panel = self.refit_panel
        if panel.series_root != root:
            return
        if panel._thread is not None:
            QTimer.singleShot(25, self, lambda: self._shared_cohort_saved(root))
            return
        panel._scope_changed()

    def install_common_actions(self, window):
        from .common_calibration_actions import start_original_automatic
        self._common_window = window
        self.original_controls.start_requested.connect(
            lambda options: start_original_automatic(window, self.original_controls, options))
        self.original_controls.inputs_requested.connect(
            lambda: window._show_original_mass_calibration_controls("mass"))
        self.original_controls.interactive_requested.connect(
            lambda: window._show_original_mass_calibration_controls("mass"))
        self.phase_colour_paths.images_requested.connect(window.open_image_selection_plan)
        self.phase_colour_paths.preparation_requested.connect(self._open_phase_preparation)
        self.phase_colour_paths.calibration_requested.connect(
            lambda: window._select_row(("calibration", "mass")))

    def _open_phase_preparation(self):
        self._common_window._select_row(("setup", "finish"))

    def show_setup_colour_paths(self, root):
        """Show optical choices; model preparation stays in Calibration."""
        self.phase_colour_paths.set_series(root)

    def restore_calibration_preparation(self):
        if self.preparation.parentWidget() is not self.job_controls:
            self.job_controls.layout().insertWidget(2, self.preparation)
        self.preparation.set_preparation_only_view(False)
        self.preparation.show()

    def set_calibration_context(self, method, series_root):
        """Show the existing model's controls in one stable shared page."""
        self._common_method, self._common_root = method, series_root
        self.phase_colour_paths.set_series(series_root if method == "phase_model" else None)
        ready = series_root is not None and method in ("colour_path", "phase_model")
        phase = method == "phase_model"
        for name in ("chemistry", "job"):
            self.sections[name].show()
            self.sections[name].setEnabled(ready)
        self.overflow_choice.hide()
        for stack in (self.model_context, self.job_context):
            stack.setCurrentIndex(0 if phase else 1)
        self.job_results.setVisible(ready and phase)
        self.original_controls.setEnabled(ready and not phase)
        self.mode_notice.setText(
            "Read-only saved calibration. Its source and parameters are shown below; "
            "new calibration choices do not change this saved result."
            if ready else "Choose an experimental series and its calibration method.")
        window = getattr(self, "_common_window", None)
        selected = window.run_combo.currentData() if window is not None else None
        self.original_inputs.setText(
            "Uses the selected run's saved calibration.mass image selection and regions, "
            "colour paths, injection protocol, pressure, temperature and geometry. "
            "Signal functions may be shared by facies; that is different from fitting "
            "one calibration jointly across runs.\nSaved run configuration: "
            + (str(selected) if selected else "Choose a run."))
