"""Persistent native Jobs forms backed by the original metadata controls."""

from pathlib import Path
import traceback

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QBoxLayout, QButtonGroup, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton,
    QLayout, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from . import jobs_panel
from .store_map_ui import StoreAvailabilityPanel, StoreSettingsDialog
from .workflow_section import WorkflowSection


def _embed(dialog):
    dialog.setWindowFlags(Qt.WindowType.Widget)
    dialog.setModal(False)
    for button in dialog.findChildren(QPushButton):
        button.setAutoDefault(False)


class _MachineRegistration(jobs_panel.RegisterMachineDialog):
    def __init__(self, owner):
        self.owner = owner
        self.bound_root = None
        super().__init__(owner, local_machine=owner.panel.local_machine)
        _embed(self)
        self.target = QLabel("Choose a job, then Add machine… to register its machine facts.")
        self.target.setWordWrap(True)
        self.target.setTextFormat(Qt.TextFormat.PlainText)
        self.layout().insertWidget(0, self.target)
        for box in self.findChildren(QDialogButtonBox):
            cancel = box.button(QDialogButtonBox.StandardButton.Cancel)
            if cancel is not None:
                cancel.hide()

    def _current_root(self):
        current = self.owner.panel.current or {}
        return str(current.get("root") or "") if current.get("job_facts") else ""

    def bind(self):
        self.bound_root = self._current_root() or None
        self.target.setText(
            "Register machine facts for: " + self.bound_root
            if self.bound_root else "Choose a job before registering machine facts."
        )
        self._validate()

    def _validate(self, *_):
        super()._validate()
        self.ok_button.setEnabled(
            self.ok_button.isEnabled()
            and bool(self.bound_root)
            and self.bound_root == self._current_root()
        )

    def sync(self):
        self._validate()
        if self.bound_root and self.bound_root != self._current_root():
            self.target.setText(
                "The selected job changed. Click Add machine… to review the new registration target."
            )

    def accept(self):
        self._validate()
        if not self.ok_button.isEnabled():
            return
        panel = self.owner.panel
        try:
            result = jobs_panel.register_machine_facts(
                self.bound_root, self.facts(), local_machine=panel.local_machine,
                new_version=self.new_version.isChecked(),
            )
        except Exception:
            panel._error(traceback.format_exc())
            return
        panel.machine_registered.emit(result)
        panel.control_note.setText(jobs_panel.REMOTE_GAP)
        self.target.setText("Machine facts registered for: " + self.bound_root)
        panel.refresh()

    def reject(self):
        # Escape must not remove the persistent form from its section.
        pass


class _StoreSettings(StoreSettingsDialog):
    def __init__(self, stores):
        self._next_root = None
        super().__init__(parent=stores, path=stores.path, reader=stores.reader)
        _embed(self)
        self.shared_series_locations.hide()
        note = QLabel("Set shared series folders in Start > Series folders. This page contains this computer's path mappings, working folder and cache.")
        note.setWordWrap(True)
        self.layout().insertWidget(0, note)
        for box in self.findChildren(QDialogButtonBox):
            box.hide()
        self.discard_button = QPushButton("Discard local settings edits")
        self.discard_button.setAutoDefault(False)
        self.discard_button.clicked.connect(self.discard)
        self.layout().addWidget(self.discard_button)
        self._controls()

    def set_series_root(self, root):
        self._next_root = Path(root) if root else None
        self._controls()

    def _controls(self):
        # The original callbacks read series_root. Keep their admitted context
        # stable until the metadata thread has retired, including on navigation.
        if self._thread is None:
            self.series_root = self._next_root
        super()._controls()
        if hasattr(self, "discard_button"):
            self.discard_button.setEnabled(self._thread is None and self.has_edits())

    def has_edits(self):
        return self.editor.toPlainText() != self._displayed_text or self.has_location_edits()

    def discard(self):
        if self._thread is not None:
            return
        self.editor.setPlainText(self._displayed_text)
        if self._locations is not None:
            self._show_locations(self._locations)
        self.status.setText("Local settings edits discarded. No file was changed.")

    def reject(self):
        pass


class NativeJobsSections:
    """Use the left column to select one persistent Jobs workspace at a time."""

    _LIST = 1000

    def __init__(self, owner):
        self.owner = owner
        self._selecting = False
        self._sections = []
        self._nested = []
        self._parents = {}
        panel = owner.panel
        panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        panel.layout().setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        panel.setMinimumSize(0, 0)
        layout = owner.jobs_view.controls_layout
        owner.jobs_view.setSizes([360, 920])
        controls = QWidget()
        self._controls_layout = QVBoxLayout(controls)
        self._controls_layout.setContentsMargins(0, 0, 0, 0)
        self.jobs_section = WorkflowSection(None, "Jobs", controls)
        self.jobs_section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        layout.addWidget(self.jobs_section)
        self._sections.append(self.jobs_section)
        self.result_buttons = []
        self._result_group = QButtonGroup(controls)
        self._result_group.setExclusive(True)
        self.list_button = QPushButton("Job list")
        self.list_button.setCheckable(True)
        self._style_choice(self.list_button)
        self._result_group.addButton(self.list_button, self._LIST)
        self._controls_layout.addWidget(self.list_button)
        panel.detail_tabs.tabBar().hide()
        self.sync_result_buttons()
        self._result_group.idClicked.connect(self._result_clicked)
        panel.detail_tabs.currentChanged.connect(self._result_changed)

        # The original table/actions retain their owner and callbacks, but no
        # longer occupy half of every selected job's Progress/Resources page.
        listing = panel.splitter.widget(0)
        listing.setParent(None)
        self.list_results = listing
        owner.result_views.addWidget(listing)
        for index, widget in enumerate((panel.context_label, panel.refresh_state_label)):
            listing.layout().insertWidget(index, widget)
        self.selected_job = QLabel("No active job")
        self.selected_job.setWordWrap(True)
        self.selected_job.setTextFormat(Qt.TextFormat.PlainText)
        self.selected_job.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        owner.jobs_view.workspace.layout().insertWidget(0, self.selected_job)
        progress = panel.detail_tabs.widget(0)
        for index, widget in enumerate((panel.master_label, panel.progress_label, panel.execution_label)):
            progress.layout().insertWidget(index, widget)
        self.operations = QScrollArea()
        self.operations.setWidgetResizable(True)
        self.operations.setWidget(panel.operation_label)
        self.operations.setVisible(not panel.operation_label.isHidden())
        progress.layout().insertWidget(3, self.operations, 1)
        files = panel.detail_tabs.widget(3)
        files.layout().insertWidget(0, panel.job_folder_label)
        files.layout().insertWidget(1, panel.profile_label)
        panel.layout().removeWidget(panel.splitter)
        panel.splitter.hide()
        # Keep the reader/action owner alive without carrying the former
        # splitter's cached minimum into the new single-page workspace.
        owner.result_views.removeWidget(panel)
        panel.setParent(owner)
        panel.hide()
        owner.job_results = QWidget()
        result_layout = QVBoxLayout(owner.job_results)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.addWidget(panel.detail_tabs)
        owner.result_views.addWidget(owner.job_results)

        # These are the same borrowable target controls used by Production.
        panel._controls.parent().removeItem(panel._controls)
        panel._controls.setParent(None)
        panel._controls.setDirection(QBoxLayout.Direction.TopToBottom)
        self.control_results = QWidget()
        control_layout = QVBoxLayout(self.control_results)
        control_layout.setContentsMargins(0, 0, 0, 0)
        manual_launch = QWidget()
        manual_layout = QVBoxLayout(manual_launch)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.addLayout(panel._controls)
        self.master_controls = QWidget()
        master_layout = QVBoxLayout(self.master_controls)
        master_layout.setContentsMargins(0, 0, 0, 0)
        for widgets in ((panel.pause_button, panel.resume_button, panel.stop_button),
                        (panel.reset_button, panel.restart_button)):
            row = QHBoxLayout()
            for widget in widgets:
                row.addWidget(widget)
            master_layout.addLayout(row)
        control_layout.addWidget(self.master_controls)
        control_layout.addWidget(panel.control_state_label)
        for widget in (panel.target_label, panel.start_reason_label, panel.launch_state_label,
                       panel.control_note, panel.declare_stopped_button):
            manual_layout.addWidget(widget)
        self.manual_launch_section = WorkflowSection(
            None, "Manual machine launch", manual_launch, expanded=False)
        control_layout.addWidget(self.manual_launch_section)
        control_layout.addStretch()
        self.control_section = self._section("Job control", self.control_results)
        # Preserve the existing native role guard's owner without a second
        # launch heading or another set of actions.
        self.legacy_section = self.manual_launch_section

        self.machine_navigation = QWidget()
        machines_layout = QVBoxLayout(self.machine_navigation)
        machines_layout.setContentsMargins(0, 0, 0, 0)
        self.machines_section = WorkflowSection(
            None, "Machines — advanced", self.machine_navigation, expanded=False)
        self.machines_section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        layout.addWidget(self.machines_section)
        self._sections.append(self.machines_section)
        self.machine = _MachineRegistration(owner)
        self.machine_section = self._section(
            "Register an existing machine", self.machine, parent_section=self.machines_section)
        panel.add_machine_button.clicked.disconnect(panel.open_machine_dialog)
        panel.open_machine_dialog = self.open_machine
        panel.add_machine_button.clicked.connect(self.open_machine)

        self.store_editor = None
        stores = owner.stores
        if isinstance(stores, StoreAvailabilityPanel):
            self.store_editor = _StoreSettings(stores)
            store_body = QWidget()
            store_layout = QVBoxLayout(store_body)
            store_layout.addWidget(stores)
            store_layout.addWidget(self.store_editor)
            self.store_editor.show()
            self.stores_section = self._section(
                "Machine path mappings", store_body,
                parent_section=self.machines_section)
            self.store_results = self._pages[self.stores_section]
            stores._dialog = self.store_editor
            stores.settings_button.clicked.disconnect(stores.open_settings)
            stores.open_settings = self.open_stores
            stores.settings_button.clicked.connect(self.open_stores)
            original_context = stores.set_context

            def set_context(root, series_id):
                original_context(root, series_id)
                self.store_editor.set_series_root(root)

            stores.set_context = set_context
            stores.can_close = lambda: (
                stores._thread is None and self.store_editor._thread is None
                and not self.store_editor.has_edits())
            self.store_editor.saved.connect(self._stores_saved)
            self.store_editor.series_locations_saved.connect(owner.series_locations_changed)

            def jobs_folder_saved(root):
                from .series_paths import series_folder_path
                if owner.series_root == Path(root):
                    owner.shared_jobs.set_series(root, series_folder_path(root, "jobs"))

            self.store_editor.jobs_folder_saved.connect(jobs_folder_saved)
            self.store_editor.set_series_root(stores.context[0] if stores.context else None)

        self._bind_section(owner.sources, owner.sources_body)
        self._wire_group(self.jobs_section, self._show_jobs_choice)
        self._wire_group(self.machines_section, self.open_machine)
        original_control_state = panel._show_control_state

        def show_control_state():
            original_control_state()
            self.sync_job_controls()

        panel._show_control_state = show_control_state
        original_snapshot = panel._show_snapshot

        def show_snapshot(data):
            original_snapshot(data)
            self.machine.sync()
            self.operations.setVisible(not panel.operation_label.isHidden())
            self.selected_job.setText(
                f"{data.get('name', 'Selected job')} · {data.get('state', '')}" if data else "No active job")

        panel._show_snapshot = show_snapshot
        show_snapshot(panel.current)
        self.show_job_list()

    def sync_job_controls(self, *, worker_only=None):
        panel = self.owner.panel
        if worker_only is None:
            service = getattr(getattr(self.owner, "shared_jobs", None), "_service", None)
            worker_only = service is not None and service.role == "worker"
        selected = bool((panel.current or {}).get("job_facts")) and not worker_only
        buttons = (panel.pause_button, panel.resume_button, panel.stop_button,
                   panel.reset_button, panel.restart_button)
        for button in buttons:
            button.setVisible(selected and button.isEnabled())
        self.master_controls.setVisible(selected and any(not button.isHidden() for button in buttons))
        panel.control_state_label.setVisible(selected)
        self.manual_launch_section.setVisible(selected)

    def _page(self, content):
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setWidget(content)
        content.show()
        self.owner.result_views.addWidget(page)
        return page

    def _wire_group(self, section, select):
        original = section.set_expanded

        def expanded(value):
            original(value)
            if value and not self._selecting:
                select()

        section.set_expanded = expanded
        section.toggle.toggled.connect(lambda value: select() if value else None)

    def _show_jobs_choice(self):
        self._result_clicked(self._result_group.checkedId())

    def _section(self, title, content, *, parent_section=None):
        placeholder = QWidget()
        section = WorkflowSection(None, title, placeholder, expanded=False)
        if parent_section is None:
            self.owner.jobs_view.controls_layout.addWidget(section)
            self._sections.append(section)
        else:
            parent_section.content.layout().addWidget(section)
            self._nested.append(section)
        self._bind_section(section, content, parent_section=parent_section, existing_body=False)
        return section

    def _bind_section(self, section, content, *, parent_section=None, existing_body=True):
        section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        if not hasattr(self, "_pages"):
            self._pages = {}
        if existing_body:
            section.layout().removeWidget(content)
            section.content = QWidget()
            section.layout().addWidget(section.content)
        self._pages[section] = self._page(content)
        self._parents[section] = parent_section
        if section not in self._sections and parent_section is None:
            self._sections.append(section)
        original = section.set_expanded

        def expanded(value):
            original(value)
            if value and not self._selecting:
                self.show_section(section)

        section.set_expanded = expanded
        section.toggle.toggled.connect(
            lambda value: self.show_section(section) if value else None)

    def _choose(self, page, section, *, parent_section=None):
        self._selecting = True
        try:
            selected = {section}
            ancestor = parent_section or self._parents.get(section)
            while ancestor is not None:
                selected.add(ancestor)
                ancestor = self._parents.get(ancestor)
            for other in self._sections:
                other.set_expanded(other in selected)
            for other in self._nested:
                other.set_expanded(other in selected)
            self.owner.jobs_view.controls_layout.invalidate()
            self.owner.jobs_view.controls_layout.activate()
            self.owner.result_views.setCurrentWidget(page)
        finally:
            self._selecting = False

    def show_section(self, section, *, parent_section=None):
        if section is getattr(self, "machine_section", None):
            self.machine.bind()
        self._choose(self._pages[section], section, parent_section=parent_section)

    def attach_shared_jobs(self, section, content):
        self._bind_section(section, content)
        # One folder menu owns its advanced import and path-mapping pages.
        # The existing forms/results are reparented, never recreated.
        sources = self.owner.sources
        self.owner.jobs_view.controls_layout.removeWidget(sources)
        self._sections.remove(sources)
        QVBoxLayout(section.content).addWidget(sources)
        self._parents[sources] = section
        self._nested.append(sources)
        if self.store_editor is not None:
            self.machines_section.content.layout().removeWidget(self.stores_section)
            QVBoxLayout(sources.content).addWidget(self.stores_section)
            self._parents[self.stores_section] = sources
        section.set_expanded(False)
        # One set of capacity and participation widgets owns both automatic
        # CPU/GPU tuning and manual CPU+GPU counts. Folder selection does not
        # replace it with the advanced single-backend machine launcher.
        self.control_results.layout().insertWidget(0, content.execution)
        content.execution.show()
        self.open_job_control = QPushButton("Open job control")
        self.open_job_control.clicked.connect(lambda: self.show_section(self.control_section))
        content.layout().addWidget(self.open_job_control)
        self.back_to_job_control = QPushButton("Back to job control")
        self.back_to_job_control.clicked.connect(lambda: self.show_section(self.control_section))
        content.results.layout().insertWidget(0, self.back_to_job_control)
        # The status button already emits the owner's existing view signal.
        self.owner.shared_jobs.show_status_requested.connect(
            lambda: self._choose(self.owner.shared_jobs.results, self.control_section))

    @staticmethod
    def _style_choice(button):
        button.setStyleSheet("QPushButton:checked {background-color: palette(highlight); "
                            "color: palette(highlighted-text); font-weight: bold;}")

    def sync_result_buttons(self):
        tabs = self.owner.panel.detail_tabs
        for index in range(len(self.result_buttons), tabs.count()):
            button = QPushButton(tabs.tabText(index))
            button.setCheckable(True)
            self._style_choice(button)
            self._result_group.addButton(button, index)
            self._controls_layout.addWidget(button)
            self.result_buttons.append(button)

    def _result_clicked(self, index):
        if index == self._LIST:
            self.show_job_list()
        else:
            self.owner.panel.detail_tabs.setCurrentIndex(index)
            self._result_changed(index)

    def _result_changed(self, index):
        if 0 <= index < len(self.result_buttons):
            self.result_buttons[index].setChecked(True)
            self._choose(self.owner.job_results, self.jobs_section)

    def show_job_list(self):
        self.list_button.setChecked(True)
        self._choose(self.list_results, self.jobs_section)

    def open_machine(self):
        self.machine.bind()
        self.show_section(self.machine_section, parent_section=self.machines_section)
        self.machine.machine_edit.setFocus()

    def open_stores(self):
        self.show_section(self.stores_section)
        self.store_editor.editor.setFocus()

    def _stores_saved(self, path):
        stores = self.owner.stores
        stores.path = Path(path)
        stores.generation += 1
        stores.report = None
        stores.details.clear()
        stores.details.hide()
        stores.summary.setText(
            "Local settings changed. Click Check availability to read the new settings.")
