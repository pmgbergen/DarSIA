"""Inline presentation of the existing native model-management controls."""

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QDialog, QPushButton, QScrollArea

from .ui.phase_model_dependency_review import ModelDependencyDialog
from .workflow_section import WorkflowSection


class InlineDependencyReview(ModelDependencyDialog):
    """Retain the original reader, cancellation and retirement callbacks."""

    closed = Signal()

    def event(self, event):
        result = super().event(event)
        if event.type() == QEvent.Type.Close and event.isAccepted():
            self.closed.emit()
        return result


class NativePhaseManagementSections:
    def __init__(self, steps):
        self.steps = steps
        self.sections = {}
        self.contexts = {}

    def context(self):
        return str(self.steps.series_id), str(self.steps.series_root)

    def present(self, key, title, content, group):
        previous = self.sections.get(key)
        if previous is None or previous.content is not content:
            if previous is not None:
                self.steps.layouts[group].removeWidget(previous)
                previous.deleteLater()
            content.setParent(self.steps.groups[group], Qt.WindowType.Widget)
            if isinstance(content, QDialog):
                content.setSizeGripEnabled(False)
            section = WorkflowSection(None, title, content)
            close = QPushButton("Close")
            close.clicked.connect(content.close)
            section.layout().itemAt(0).layout().addWidget(close)
            if isinstance(content, InlineDependencyReview):
                content.closed.connect(section.hide)
            elif isinstance(content, QDialog):
                content.finished.connect(section.hide)
            self.steps.layouts[group].addWidget(section)
            self.sections[key] = section
            self.contexts[key] = self.context()
        self.show(key)

    def show(self, key):
        section = self.sections[key]
        section.show()
        section.set_expanded(True)
        ancestor = section.parentWidget()
        while ancestor is not None:
            if isinstance(ancestor, QScrollArea):
                ancestor.ensureWidgetVisible(section)
            ancestor = ancestor.parentWidget()

    def sync_context(self):
        for key, section in self.sections.items():
            current = self.contexts[key] == self.context()
            section.setEnabled(current)
            if not current:
                section.hide()
