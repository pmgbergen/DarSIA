"""Production controls on the original Jobs page and its worker lifecycle."""
from __future__ import annotations

from copy import deepcopy
from PySide6.QtCore import QSignalBlocker, Signal, Slot
from PySide6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton

from .jobs_panel import NewAnalysisPage
from .storage_profile import DEFAULT_STORAGE_PROFILE
from .production_selection import (
    CHOICE_KEY, STORES_KEY, production_choice_blocker, read_named_calibration, read_production_facts,
)


class ProductionPage(NewAnalysisPage):
    """Reuse Create, plan reading, launches and machine widgets from Jobs.

    The native owner supplies this page through its page factory. Calibration
    browsing is read-only, finite and owned by the same existing call runner.
    """

    jobs_requested = Signal()

    def __init__(self, panel, parent=None, *, facts_reader=read_production_facts):
        self._calibration_thread = self._calibration_worker = None
        self._calibration_generation = 0
        self._production_context_generation = 0
        self._active_facts_context = None
        self._facts_storage_profile = DEFAULT_STORAGE_PROFILE
        self._choice_error = ""
        self._creation_choice = None
        super().__init__(panel, parent, facts_reader=facts_reader)
        self.calibration_combo = QComboBox()
        self.calibration_combo.setPlaceholderText("Choose a completed mass-calibration job")
        self.calibration_combo.currentIndexChanged.connect(self._calibration_choice_changed)
        self.open_calibration_button = QPushButton("Choose mass calibration…")
        self.open_calibration_button.clicked.connect(self.choose_calibration)
        row = QHBoxLayout()
        row.addWidget(self.calibration_combo, 1)
        row.addWidget(self.open_calibration_button)
        self.what.layout().insertRow(1, "Mass calibration and chemistry", row)
        self.calibration_note = QLabel()
        self.calibration_note.setWordWrap(True)
        self.what.layout().insertRow(2, self.calibration_note)
        self.jobs_button = QPushButton("Open Jobs")
        self.jobs_button.clicked.connect(self.jobs_requested)
        self.where.layout().addWidget(self.jobs_button)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Run analysis on this computer", "local")
        self.mode_combo.addItem("Distribute over several machines", "shared")
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.where.layout().insertWidget(0, self.mode_combo)
        self.execution_note = QLabel()
        self.execution_note.setWordWrap(True)
        self.where.layout().insertWidget(1, self.execution_note)
        panel.target_combo.currentIndexChanged.connect(self.sync_actions)
        self.form.name_edit.textChanged.connect(self.sync_actions)
        self._show_facts()
        self._mode_changed()
        from .native_production_sections import NativeProductionSections

        self.inline_layout = NativeProductionSections(self)

    @Slot()
    def choose_calibration(self):
        if self._calibration_thread is not None:
            return
        path = QFileDialog.getExistingDirectory(self, "Choose a completed named mass-calibration job")
        if path:
            self.load_calibration(path)

    def load_calibration(self, path):
        """Read only the explicitly chosen job; no folder or library discovery."""
        if self._calibration_thread is not None:
            return
        self._choice_error = "Reading the chosen mass calibration…"
        self.open_calibration_button.setEnabled(False)
        self._calibration_generation += 1
        generation = self._calibration_generation
        series = (self.panel.series_root, self.panel.series_id)

        def read():
            try:
                return dict(generation=generation, series=series, choice=read_named_calibration(path))
            except Exception as error:
                return dict(generation=generation, series=series, error=f"{type(error).__name__}: {error}")

        self._calibration_thread, self._calibration_worker = self.panel._start_call(
            read, self._calibration_ready, self._calibration_closed)
        self.sync_actions()

    @Slot(object)
    def _calibration_ready(self, result):
        if (result["generation"] != self._calibration_generation
                or result["series"] != (self.panel.series_root, self.panel.series_id)):
            return
        self._choice_error = result.get("error", "")
        choice = result.get("choice")
        if choice and choice["series_id"] != self.panel.series_id:
            self._choice_error = "The selected mass-calibration job belongs to another series."
        elif choice:
            existing = next((index for index in range(self.calibration_combo.count())
                             if self.calibration_combo.itemData(index) == choice), -1)
            if existing < 0:
                self.calibration_combo.addItem(
                    f"{choice['name']} · {choice['chemistry_label']} · {choice['calibration']['sha256'][:12]}",
                    deepcopy(choice))
                existing = self.calibration_combo.count() - 1
            self.calibration_combo.setCurrentIndex(existing)
        self.sync_actions()

    @Slot()
    def _calibration_closed(self):
        if self._calibration_thread is not None:
            self._calibration_thread.wait()
        self._calibration_thread = self._calibration_worker = None
        self.open_calibration_button.setEnabled(True)
        self.sync_actions()

    def reset(self):
        super().reset()
        self._creation_choice = None
        self._calibration_generation += 1
        self._production_context_generation += 1
        self.facts = self.facts_error = self._facts_key = None
        self._facts_pending = self._facts_thread is not None
        self._choice_error = ""
        if hasattr(self, "calibration_combo"):
            self.calibration_combo.clear()
        self.sync_actions()

    def _calibration_choice_changed(self, *_):
        if self._calibration_thread is None:
            self._choice_error = ""
        self.sync_actions()

    def prepare_plan(self):
        # Retain the pre-calibration phase-one route while the shared navigation
        # is integrated. A production preparation uses the chosen frozen bytes.
        if (self.facts or {}).get("pipeline") == "native-phase-statistics":
            return super().prepare_plan()
        choice = self.calibration_combo.currentData()
        if choice is None:
            # A new series has no mass calibration yet. Its original phase-one
            # preparation must remain reachable before any plan is registered.
            return super().prepare_plan()
        if self._preparation_dialog is not None:
            self._preparation_dialog.raise_()
            self._preparation_dialog.activateWindow()
            return
        from .production_preparation_ui import ProductionPreparationDialog
        dialog = ProductionPreparationDialog(self.panel, choice, self,
            storage_profile=self.form.profile_combo.currentData())
        self._preparation_dialog = dialog
        try:
            dialog.adopted.connect(self._prepared_adopted)
            dialog.exec()
        finally:
            self._preparation_dialog = None
            dialog.deleteLater()

    def _choices_changed(self):
        # Capacity and provider admission belong to the requested profile. An
        # old Full result must not enable Create after the user selects Lean.
        if self.form.profile_combo.currentData() != self._facts_storage_profile:
            self._read_facts()
            self.sync_actions()
            return
        super()._choices_changed()

    def _inspect_resources(self, plan, *, series_root, stores_path, storage_profile):
        from .production_resources import inspect_production_resources
        return inspect_production_resources(plan, series_root=series_root,
            stores_path=stores_path, storage_profile=storage_profile)

    def _read_facts(self):
        """Keep the existing call lifecycle, with captured series read identity."""
        self._facts_storage_profile = self.form.profile_combo.currentData()
        context = (self.panel.series_root, self.panel.series_id, self._production_context_generation)
        reader = self.facts_reader
        if self._facts_thread is None:
            self._active_facts_context = context

        def contextual_reader(plan):
            try:
                facts = (reader(plan, series_root=context[0], series_id=context[1])
                         if reader is read_production_facts else reader(plan))
                return dict(facts, _production_context=context)
            except Exception as error:
                return dict(_production_context=context, _production_error=f"{type(error).__name__}: {error}")

        self.facts_reader = contextual_reader
        try:
            super()._read_facts()
        finally:
            self.facts_reader = reader

    @Slot(object)
    def _facts_ready(self, result):
        facts = dict(result.get("facts") or {})
        if facts.pop("_production_context", self._active_facts_context) != (
                self.panel.series_root, self.panel.series_id, self._production_context_generation):
            return
        error = facts.pop("_production_error", result.get("error"))
        super()._facts_ready(dict(result, facts=None if error else facts, error=error))

    def _show_facts(self, *_):
        profile = self.form.profile_combo.currentData()
        phase2 = (self.facts or {}).get("pipeline") == "n1"
        # The ordinary page pins staged routes to Full. Production admits its
        # selected profile through the original capacity and provider readers.
        with QSignalBlocker(self.form.profile_combo):
            super()._show_facts()
            if phase2:
                self.form.profile_combo.setCurrentIndex(self.form.profile_combo.findData(profile))
                self.form.profile_combo.setEnabled(True)
                self.form.profile_combo.setToolTip(
                    "The selected profile must match the saved plan and its qualified provider. "
                    "A different sealed setting needs a newly prepared and adopted plan.")
        with QSignalBlocker(self.form):
            self.form._update_estimate()
        if not hasattr(self, "calibration_combo"):
            return
        self.selection_label.setText(
            self.selection_label.text() + "\nLocked to Setup > Image selection, including "
            "blacklists and saved rules. Creation rechecks the exact saved sources, times and order.")
        self.precision_label.setText(
            self.precision_label.text() + "\nHigher precision cannot be selected here. "
            "Float64 requires eligible phase maps computed again at that precision and a new admitted plan.")
        self.stored_combo.setToolTip(
            "The adopted plan fixes these stored sizes. Use Prepare a new analysis plan to choose "
            "another supported size, then review and adopt that new plan.")
        admission = (self.facts or {}).get("resource_admission", {})
        product_bytes = admission.get("products_expected_bytes")
        if phase2 and admission.get("ready") is True and type(product_bytes) is int:
            self.form.estimate_label.setText(
                f"Reserved product capacity: {product_bytes:,} bytes. Start checks available space again.")
        if self.form.profile_combo.currentData() != self._facts_storage_profile:
            # Switching from a Lean mass plan to phase one selects Full and
            # needs a fresh admission for that choice, through the same reader.
            self._choices_changed()
        self.sync_actions()

    def _mode_changed(self, *_):
        if self.mode_combo.currentData() == "local":
            index = next((i for i in range(self.panel.target_combo.count())
                          if (self.panel.target_combo.itemData(i) or {}).get("local")), -1)
            if index >= 0:
                self.panel.target_combo.setCurrentIndex(index)
            self.execution_note.setText(
                "This creates a real job in Jobs. Choose the worker count below, then start. "
                "The shared job folder and the ordinary queue guards are required on this computer too.")
        else:
            self.execution_note.setText(
                "This creates one real job in Jobs. Choose shared folders, register machines and "
                "choose workers per machine there. Start each registered machine against this same job.")
        self.sync_actions()

    def sync_actions(self, *_):
        super().sync_actions()
        if not hasattr(self, "calibration_combo"):
            return
        choice = self.calibration_combo.currentData()
        blocker = self._choice_error or production_choice_blocker(self.facts, choice)
        if blocker:
            self.create_button.setEnabled(False)
            self.create_button.setToolTip(blocker)
        self.calibration_note.setText(blocker or (
            "The exact frozen calibration matches this plan. Create rechecks the locked series choice. "
            "Choosing here never changes the series approval or the adopted plan."
            if choice else "Phase maps and statistics precede the mass calibration."))
        if not hasattr(self, "mode_combo"):
            return
        target = self.panel.target_combo.currentData() or {}
        local = self.mode_combo.currentData() == "local"
        if local and not target.get("local"):
            self.start_button.setEnabled(False)
            self.start_button.setToolTip("Choose this computer as the target for local production.")
        if self.created_root and self._creation_choice != self._current_choice():
            self.start_button.setEnabled(False)
            self.start_button.setToolTip(
                "These choices differ from the created job. Open that job in Jobs, "
                "or create a new job from the current choices.")
        self.start_button.setText("Run analysis on this computer" if local else "Start selected machine for this job")
        if hasattr(self, "inline_layout"):
            self.inline_layout.sync()

    def _current_choice(self):
        return dict(self.form.choices(), calibration=deepcopy(self.calibration_combo.currentData()))

    def create(self):
        if not self.create_button.isEnabled():
            return
        self._creation_choice = self._current_choice()
        choices = self.form.choices()
        # Capture the selected machine's store settings on the UI thread. The
        # creator runs later and must not query a possibly switched workspace.
        provider = self.panel.local_stores_path_provider
        if provider is not None:
            choices["plan"][STORES_KEY] = provider()
        if (self.facts or {}).get("pipeline") != "native-phase-statistics":
            choices["plan"][CHOICE_KEY] = deepcopy(self.calibration_combo.currentData())
        self._pending_action = "create"
        self.action_label.setText("Creating job files… No worker is started.")
        self.panel.create_job(**choices)
        self.sync_actions()

    def start(self):
        # Re-evaluate the actual target immediately before the existing action.
        self.sync_actions()
        super().start()

    def can_close(self):
        return self._calibration_thread is None and super().can_close()
