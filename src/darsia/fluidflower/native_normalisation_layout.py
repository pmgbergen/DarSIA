"""Current image preparation, with earlier results grouped separately."""

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import QBoxLayout, QLabel, QVBoxLayout, QWidget

from .workflow_section import WorkflowSection, WorkflowWorkspace


class _WrappedTextHeight(QObject):
    """Keep asynchronously replaced text readable inside the split workspace."""

    def __init__(self, step):
        super().__init__(step)
        self._by_parent = {}
        self._pending = False
        for label in step.findChildren(QLabel):
            if label.wordWrap():
                label.installEventFilter(self)
                parent = label.parentWidget()
                self._by_parent.setdefault(parent, []).append(label)
                parent.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.LayoutRequest):
            if not self._pending:
                self._pending = True
                QTimer.singleShot(0, self, self._refresh)
        return False

    def _refresh(self):
        self._pending = False
        for parent, labels in self._by_parent.items():
            changed = False
            for label in labels:
                height = label.heightForWidth(label.width())
                if height >= 0 and label.minimumHeight() != height:
                    label.setMinimumHeight(height)
                    # Refresh the box-layout item as well as its widget. Qt can
                    # otherwise retain the old one-line item after async text
                    # replacement, placing the next control over the new text.
                    layout = parent.layout()
                    index = layout.indexOf(label)
                    if isinstance(layout, QBoxLayout) and index >= 0:
                        alignment = layout.itemAt(index).alignment()
                        stretch = layout.stretch(index)
                        layout.removeWidget(label)
                        layout.insertWidget(index, label, stretch, alignment)
                    changed = True
            if changed:
                parent.layout().invalidate()
                parent.updateGeometry()


def _clear_layout(layout):
    """Remove layout items while retaining their original widgets and signals."""
    while layout.count():
        item = layout.takeAt(0)
        child_layout = item.layout()
        if child_layout is not None:
            _clear_layout(child_layout)
            child_layout.deleteLater()


def _stack_controls(widget):
    layout = widget.layout()
    layout.setContentsMargins(0, 0, 0, 0)
    for index in range(layout.count()):
        row = layout.itemAt(index).layout()
        if isinstance(row, QBoxLayout):
            row.setDirection(QBoxLayout.Direction.TopToBottom)
            for item_index in range(row.count()):
                row.setStretch(item_index, 0)


def _panel(*widgets):
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    for widget in widgets:
        layout.addWidget(widget)
    return panel


def install_normalisation_layout(step):
    """Reparent only; the step and its child widgets retain all worker ownership."""
    outer = step.layout()
    _clear_layout(outer)
    outer.setContentsMargins(0, 0, 0, 0)
    _stack_controls(step.method_choice)
    _stack_controls(step.reproduction)

    step.workflow = WorkflowWorkspace()
    outer.addWidget(step.workflow)
    step.sections = {}
    from .image_preparation_preview import ImagePreparationPreview
    from .preparation_crop_summary import PreparationCropSummary
    step.image_preview = ImagePreparationPreview(step.method_choice)
    step.crop_summary = PreparationCropSummary(step.method_choice, step.image_preview)
    history_controls = _panel()
    history_sections = {}
    for key, title, content in (
        ("evidence", "Saved lighting history", _panel(
            step.status, step.source_label, step.choose_source, step.reload_button)),
        ("reproduction", "Verify an earlier RAW result", step.reproduction),
        ("reference", "Historical reference", _panel(
            step.override, step.reference, step.save_button)),
    ):
        section = WorkflowSection(None, title, content, expanded=False)
        history_sections[key] = section
        history_controls.layout().addWidget(section)
    history_note = QLabel(
        "Earlier results are optional history. They are not required for the current "
        "crop, lighting or ColorChecker choices and do not start new image processing."
    )
    history_note.setWordWrap(True)
    history_controls.layout().insertWidget(0, history_note)
    panels = (
        ("crop", "Crop and alignment", step.crop_summary),
        ("method", "Image correction", step.method_choice),
        ("history", "Earlier results (optional)", history_controls),
    )
    for number, (key, title, content) in enumerate(panels, 1):
        advanced = key == "history"
        section = WorkflowSection(None if advanced else number, title, content,
                                  expanded=not advanced)
        step.sections[key] = section
        step.workflow.controls_layout.addWidget(section)
    step.sections.update(history_sections)

    workspace = QVBoxLayout(step.workflow.workspace)
    workspace.setContentsMargins(8, 0, 0, 0)
    step.preparation_note = QLabel(
        "Image correction before Finish setup\n\n"
        "Crop and alignment use the approved reference points and geometry. "
        "Lighting / intensity normalisation and ColorChecker colour correction "
        "are independent choices for the new phase-map image correction, both before crop. "
        "The direct colour method retains its existing Rig correction settings.\n\n"
        "Saving a choice starts no calculation and does not change an existing job. "
        "Prepared images, colour paths and trained models must use the same image basis.\n\n"
        "Reference proposals use baseline "
        "images, the ColorChecker and stable areas, not CO2 or chemistry ranking. "
        "Your saved reference remains in use until you explicitly choose and save another one."
    )
    step.preparation_note.setWordWrap(True)
    from .image_preparation_status import ImagePreparationStatus
    step.execution_status = ImagePreparationStatus(step.method_choice)
    step.execution_status.completion_changed.connect(step.prepared_completion)
    step.current_preparation = _panel(step.preparation_note, step.execution_status, step.image_preview)
    step.current_preparation.layout().setStretch(2, 1)
    workspace.addWidget(step.current_preparation, 1)
    step.history_panel = _panel(step.ranking_note, step.imported_fields,
                                step.table, step.provenance)
    workspace.addWidget(step.history_panel, 1)

    def show_history(_checked=False):
        visible = step.sections["history"].toggle.isChecked() and any(
            step.sections[key].toggle.isChecked() for key in ("evidence", "reference"))
        step.history_panel.setVisible(visible)
        step.current_preparation.setVisible(not visible)

    for key in ("history", "evidence", "reference"):
        step.sections[key].toggle.toggled.connect(show_history)
    show_history()
    step._wrapped_text_height = _WrappedTextHeight(step)
