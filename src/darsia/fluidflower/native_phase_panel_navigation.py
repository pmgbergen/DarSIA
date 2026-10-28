"""User-driven navigation from phase panels to the original inline review."""

from PySide6.QtCore import QEvent, QObject, QTimer

from .workflow_section import WorkflowPanelColumn


class NativePhasePanelColumn(WorkflowPanelColumn):
    """Select one phase panel without changing the working area's owner."""

    def _collapse_others(self, index):
        for other, section in enumerate(self.sections):
            if other != index:
                section.set_expanded(False)

    def _selected(self, index):
        self._collapse_others(index)
        super()._selected(index)

    def setCurrentIndex(self, index):
        self._collapse_others(index)
        super().setCurrentIndex(index)


class NativePhasePanelNavigation(QObject):
    """Keep the selected working area open as its prerequisites arrive."""

    def __init__(self, steps, panel):
        super().__init__(steps)
        self.steps, self.panel = steps, panel
        self._pending = None
        self._retry_queued = False
        self.sections = {number: next(section for section in steps.phase_tabs.sections
                                      if section.content is steps.groups[number])
                         for number in (2, 3, 4)}
        for number, mode in ((2, "open_correction"), (3, "blind_region_labeling")):
            self.sections[number].toggle.clicked.connect(
                lambda _checked, mode=mode: self.open(mode))
        self.sections[4].toggle.clicked.connect(self.show_training)
        for section in steps.phase_tabs.sections:
            if section not in self.sections.values():
                section.toggle.clicked.connect(self.cancel)
        self._gates = tuple(widget for widget in (
            getattr(steps.studio, "open_model_correction", None),
            getattr(steps.studio, "open_blind_labeling", None),
        ) if widget is not None)
        for widget in (*self._gates, steps, steps.phase_workspace):
            widget.installEventFilter(self)
        steps.studio.validation_model.currentIndexChanged.connect(self._queue_retry)
        steps.review_readiness_changed.connect(self._queue_retry)
        steps.workspace_stack.currentChanged.connect(self._workspace_changed)

    def _context(self):
        return str(self.steps.series_root or ""), self.steps.series_id

    def cancel(self, *_args):
        self._pending = None

    def open(self, mode):
        if self.steps.model_selection_only():
            self.cancel()
            return
        section = self.sections[3 if mode == "blind_region_labeling" else 2]
        self.steps.phase_tabs.setCurrentWidget(section.content)
        self._pending = (mode, self._context())
        self._try_open()

    def show_training(self, *_args):
        self.cancel()
        if self.steps.model_selection_only():
            return
        self.steps.phase_tabs.setCurrentWidget(self.sections[4].content)
        self.steps.workspace_stack.setCurrentWidget(self.steps.workspace_overview)

    def _metadata_loading(self):
        studio = self.steps.studio
        getter = getattr(studio, "_selected_validation_model", None)
        model = (getter() or {}) if callable(getter) else {}
        return bool(model.get("validation_binding_loading")
                    or getattr(studio, "_model_catalog_inflight", False)
                    or getattr(studio, "_model_catalog_pending", False))

    def _try_open(self):
        if self._pending is None:
            return
        if self.steps.model_selection_only():
            self.cancel()
            return
        mode, context = self._pending
        if context != self._context() or self.steps.section != "phase_model":
            self.cancel()
            return
        self.panel.requested_role = "blind" if mode == "blind_region_labeling" else "label"
        if self.panel.process is not None:
            self.cancel()
            self.panel.open_role(mode)
            return
        role = self.panel.requested_role
        reason = self.steps.review_launch_reason(role)
        self.panel.show_requirement(reason or "Opening phase-model review…")
        if reason:
            # A prerequisite can finish after catalogue loading (for example
            # the selected saved binding). Keep this panel's intent until the
            # user navigates away; a state change then opens it without a button.
            return
        # The original callback owns admission and session creation. Clear the
        # wish first, so its synchronous control updates cannot launch twice.
        self.cancel()
        self.steps._launch_review(mode)
        if self.panel.process is not None:
            self.panel.open_role(mode)
            return
        reason = self.steps.review_launch_reason(role) or self.steps.studio.review_status.text()
        self.panel.show_requirement(reason or "The phase-model review could not be opened.")
        if self._metadata_loading():
            self._pending = (mode, context)

    def _workspace_changed(self, _index):
        if self.steps.workspace_stack.currentWidget() is not self.panel:
            self.cancel()

    def _queue_retry(self, *_args):
        if self._pending is not None and not self._retry_queued:
            self._retry_queued = True
            QTimer.singleShot(0, self._retry)

    def _retry(self):
        self._retry_queued = False
        if not self.steps.isVisible() or not self.steps.phase_workspace.isVisible():
            self.cancel()
            return
        self._try_open()

    def eventFilter(self, watched, event):
        if watched in self._gates and event.type() in (
                QEvent.Type.EnabledChange, QEvent.Type.ToolTipChange):
            self._queue_retry()
        elif watched in (self.steps, self.steps.phase_workspace) and event.type() == QEvent.Type.Hide:
            self.cancel()
        return super().eventFilter(watched, event)
