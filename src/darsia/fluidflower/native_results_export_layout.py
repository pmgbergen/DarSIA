"""Native Results and export layout around the original saved-result browser."""

from PySide6.QtWidgets import QBoxLayout, QComboBox, QLayout, QVBoxLayout, QWidget

from .workflow_section import WorkflowSection, WorkflowWorkspace


def _take(layout):
    parts = []
    while layout.count():
        stretch = layout.stretch(0)
        item = layout.takeAt(0)
        value = item.widget() or item.layout()
        if item.layout() is not None:
            item.layout().setParent(None)
        parts.append((value, stretch, item.alignment()))
    return parts


def _add(layout, part):
    value, stretch, alignment = part
    if isinstance(value, QLayout):
        parent = value.parent()
        if isinstance(parent, QLayout):
            parent.removeItem(value)
        value.setParent(None)
        layout.addLayout(value, stretch)
    elif value is not None:
        layout.addWidget(value, stretch, alignment)


def _column(parts):
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    for part in parts:
        _add(layout, part)
    return widget


class NativeResultsExportLayout:
    """Move existing controls only in Results; restore Calibration review intact."""

    def __init__(self, page):
        self.page = page
        self.workflow = None
        self.sections = {}
        self._folds = {}
        self._connections = []
        self._directions = []
        self._widths = []
        show_view = page.show_view

        def show(view):
            # Validate before moving anything, matching the original API.
            if view not in page.VIEW_TITLES:
                return show_view(view)
            self._restore()
            result = show_view(view)
            if view == "results":
                self._arrange()
            return result

        show.__wrapped__ = show_view
        page.show_view = show
        if page.view == "results":
            self._arrange()

    def _section(self, key, title, parts, *, expanded=False, toggle=None):
        content = _column(parts)
        expanded = self._folds.get(key, toggle.isChecked() if toggle is not None else expanded)
        section = WorkflowSection(len(self.sections) + 1, title, content, expanded=expanded)
        section.toggle.toggled.connect(lambda checked: self._folds.__setitem__(key, checked))
        if toggle is not None:
            toggle.setParent(content)
            toggle.hide()
            toggle.setChecked(expanded)
            # Keep the existing toggle/body API and all its original callbacks.
            section.toggle.toggled.connect(toggle.setChecked)
            toggle.toggled.connect(section.set_expanded)
            self._connections.append((toggle.toggled, section.set_expanded))
        self.sections[key] = section
        self.workflow.controls_layout.addWidget(section)
        return section

    def _arrange(self):
        page, saved = self.page, self.page.saved_results
        self.sections = {}
        self._snapshots = [
            (page.layout(), _take(page.layout())),
            (page.scroll.widget().layout(), _take(page.scroll.widget().layout())),
            (saved.layout(), _take(saved.layout())),
        ]
        outer, page_parts, saved_parts = [parts for _, parts in self._snapshots]
        self.workflow = WorkflowWorkspace(workspace=saved)
        page.layout().addWidget(self.workflow, 1)
        page.scroll.hide()
        saved.show()
        by_widget = {part[0]: part for part in page_parts + saved_parts}
        rows = saved._legacy_review_rows
        excluded = {saved, page.publish_button, saved.setup_toggle, saved.setup_body,
                    saved.export_toggle, saved.export_body, saved.details_toggle, saved.details_body,
                    saved.views, saved.review_panels, saved.frame_summary, saved.model_assumption,
                    saved.status, *rows}
        _add(self.workflow.controls_layout, by_widget[page.title])
        main = [part for part in page_parts if part[0] not in excluded and part[0] is not page.title]
        self._section("model", "Phase model", [*main, by_widget[rows[0]]])
        self._section("results", "Current models", [by_widget[saved.status],
                      by_widget[rows[1]], by_widget[saved.model_assumption]], expanded=True)
        self._section("filters", "Light correction", [by_widget[rows[2]]])
        self._section("map", "Map", [by_widget[rows[3]]])
        prepare = [by_widget[saved.setup_body]]
        # This optional native button is installed by Jobs before this adapter.
        # It retains its original callback and moves out of the full-width row.
        start = getattr(page, "native_start_analysis_button", None)
        if start is not None:
            prepare.extend(part for part in outer if part[0] is start)
        self._section("prepare", saved.setup_toggle.text(), prepare, toggle=saved.setup_toggle)
        self._section("details", saved.details_toggle.text(), [by_widget[saved.details_body]],
                      toggle=saved.details_toggle)
        self._section("export", saved.export_toggle.text(),
                      [by_widget[saved.export_body], by_widget[page.publish_button]], toggle=saved.export_toggle)
        # Keep the original tab indices, figures, slider, Play and frame reader.
        for widget in (saved.views, saved.review_panels):
            _add(saved.layout(), by_widget[widget])
        for part in saved_parts:
            if part[0] not in excluded:
                _add(saved.layout(), part)
        _add(saved.layout(), by_widget[saved.frame_summary])
        self._view_minimum = saved.views.minimumHeight()
        saved.views.setMinimumHeight(240)
        for box in self.workflow.controls_widget.findChildren(QBoxLayout):
            if box.direction() in (QBoxLayout.Direction.LeftToRight, QBoxLayout.Direction.RightToLeft):
                self._directions.append((box, box.direction(), [box.stretch(i) for i in range(box.count())]))
                box.setDirection(QBoxLayout.Direction.TopToBottom)
                for index in range(box.count()):
                    box.setStretch(index, 0)
        for combo in self.workflow.controls_widget.findChildren(QComboBox):
            self._widths.append((combo, combo.minimumWidth()))
            combo.setMinimumWidth(0)

    def _restore(self):
        if self.workflow is None:
            return
        page, saved = self.page, self.page.saved_results
        for signal, slot in self._connections:
            signal.disconnect(slot)
        self._connections.clear()
        for key, section in self.sections.items():
            self._folds[key] = section.is_expanded()
        for box, direction, stretches in self._directions:
            box.setDirection(direction)
            for index, stretch in enumerate(stretches):
                box.setStretch(index, stretch)
        self._directions.clear()
        for combo, width in self._widths:
            combo.setMinimumWidth(width)
        self._widths.clear()
        saved.views.setMinimumHeight(self._view_minimum)
        # A layout item is never cloned: all original owners and order return.
        for layout, parts in reversed(self._snapshots):
            _take(layout)
            for part in parts:
                _add(layout, part)
        for toggle in (saved.setup_toggle, saved.details_toggle, saved.export_toggle):
            toggle.show()
        page.scroll.show()
        self.workflow.hide()
        self.workflow.deleteLater()
        self.workflow = None
        self.sections = {}
