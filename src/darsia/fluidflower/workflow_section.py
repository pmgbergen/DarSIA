"""Foldable, persistent sections for the native workflow pages."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QScrollArea, QSplitter, QToolButton, QVBoxLayout, QWidget,
)


class WorkflowWorkspace(QSplitter):
    """A persistent column of controls beside the page's working area."""

    def __init__(self, parent=None, *, workspace=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setChildrenCollapsible(False)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setMinimumWidth(360)
        self.controls_scroll.setMaximumWidth(720)
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.controls_layout.setContentsMargins(4, 4, 4, 4)
        self.controls_layout.setSpacing(8)
        self.controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.controls_scroll.setWidget(self.controls_widget)
        self.workspace = workspace if workspace is not None else QWidget()
        self.addWidget(self.controls_scroll)
        self.addWidget(self.workspace)
        self.setStretchFactor(0, 0)
        self.setStretchFactor(1, 1)
        self.setSizes([580, 1020])

    def setSizes(self, sizes):
        # QSplitter can reserve a requested size beyond a child's maximum,
        # leaving a blank gap even though the child itself is clamped.
        sizes = list(sizes)
        if len(sizes) == 2 and sizes[0] > self.controls_scroll.maximumWidth():
            excess = sizes[0] - self.controls_scroll.maximumWidth()
            sizes = [sizes[0] - excess, sizes[1] + excess]
        super().setSizes(sizes)


class WorkflowSection(QFrame):
    """Keep a section's controls on the page when its header is folded."""

    def __init__(self, number, title, content, *, expanded=True, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)
        header = QHBoxLayout()
        self.toggle = QToolButton()
        self.toggle.setCheckable(True)
        self.toggle.setAutoRaise(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setText(title)
        font = self.toggle.font()
        font.setBold(True)
        self.toggle.setFont(font)
        self.toggle.toggled.connect(self.set_expanded)
        header.addWidget(self.toggle)
        header.addStretch(1)
        layout.addLayout(header)
        self.content = content
        layout.addWidget(content)
        self.set_expanded(expanded)

    def set_expanded(self, expanded):
        self.toggle.blockSignals(True)
        self.toggle.setChecked(bool(expanded))
        self.toggle.blockSignals(False)
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.content.setVisible(bool(expanded))

    def is_expanded(self):
        return self.toggle.isChecked()


class WorkflowPanelColumn(QWidget):
    """Keep the existing page-selection API while showing vertical panels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sections = []
        self._current = 0
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    def addTab(self, content, title):
        index = len(self.sections)
        section = WorkflowSection(index + 1, title, content, expanded=index == 0)
        section.toggle.toggled.connect(
            lambda expanded, index=index: self._selected(index) if expanded else None)
        self.sections.append(section)
        self._layout.addWidget(section)
        return index

    def _selected(self, index):
        self._current = index

    def count(self):
        return len(self.sections)

    def widget(self, index):
        return self.sections[index].content

    def tabText(self, index):
        return self.sections[index].toggle.text()

    def setTabText(self, index, title):
        self.sections[index].toggle.setText(title)

    def currentIndex(self):
        return self._current

    def currentWidget(self):
        return self.widget(self._current)

    def setCurrentWidget(self, widget):
        for index, section in enumerate(self.sections):
            if section.content is widget:
                self.setCurrentIndex(index)
                return

    def setCurrentIndex(self, index):
        self._current = index
        section = self.sections[index]
        section.set_expanded(True)
        self._layout.activate()
        ancestor = self.parentWidget()
        while ancestor is not None:
            if isinstance(ancestor, QScrollArea):
                ancestor.ensureWidgetVisible(section.toggle)
                break
            ancestor = ancestor.parentWidget()
