"""Native header: one status line above every workflow page (N141, layout rule 4).

The line names the series, the method with its model and calibration, and the
page's scope, for example "FF_AC · Phase-map method · model … · calibration … ·
Whole series (39 runs)". The former lines (method badge, analysis method badge,
frozen reference, scope and rig) keep their objects and wording in an
expandable details section and in the line's tooltip. The run selector shows
only on run pages (rule 5). Nothing is moved into a scroll area.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from darsia.fluidflower.row_scope import scope_summary

# The sidebar category whose page a logical row action shows.
ROW_CATEGORIES = {
    "geometry_controls": "setup",
    "phase_model": "calibration",
    "method": "calibration",
    "mass_analysis": "analysis",
    "jobs": "analysis",
    "image_selection": "setup",
    "overflow_data": "setup",
}


def row_category(action):
    """The sidebar category of a logical row action; other actions name their own."""
    return ROW_CATEGORIES.get(action, action)


def status_line(series, method_parts, scope):
    """Join one header line, series · method parts · scope, skipping empty parts."""
    return " · ".join(part for part in (series, *method_parts, scope) if part)


class NativeStatusHeader:
    """One status line with expandable details, on every page but Start."""

    def __init__(self, window):
        self.window = window
        right = window.content_stack.parentWidget().layout()
        self.bar = QWidget()
        self.bar.setObjectName("nativeStatusHeader")
        row = QHBoxLayout(self.bar)
        row.setContentsMargins(0, 0, 0, 0)
        self.line = QLabel()
        self.line.setObjectName("nativeStatusLine")
        self.line.setTextFormat(Qt.TextFormat.PlainText)
        self.line.setWordWrap(True)
        self.line.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.line, 1)
        self.details_button = QToolButton()
        self.details_button.setText("Details")
        self.details_button.setCheckable(True)
        self.details_button.setToolTip(
            "Show the method, frozen reference, scope and rig lines"
        )
        row.addWidget(self.details_button)
        right.insertWidget(0, self.bar)
        self.details = QWidget()
        self.details.setObjectName("nativeStatusDetails")
        layout = QVBoxLayout(self.details)
        layout.setContentsMargins(0, 0, 0, 0)
        self.detail_widgets = (
            window.method_badge,
            window.analysis_method_badge,
            window.frozen_reference_line,
            window.scope_label,
            window.rig_label,
        )
        for widget in self.detail_widgets:
            hidden = widget.isHidden()
            right.removeWidget(widget)
            layout.addWidget(widget)
            widget.setHidden(hidden)
        right.insertWidget(1, self.details)
        self.details.hide()
        self.details_button.toggled.connect(lambda _checked: self.refresh())
        window.content_stack.currentChanged.connect(self.refresh)
        self.refresh()

    def hide(self):
        """Hide the line and its details, as for the full configuration view."""
        self.bar.hide()
        self.details.hide()

    def refresh(self, *_args):
        """Show the line for the current page and the run selector on run pages."""
        w = self.window
        active = row_category(w._current_category) != "start"
        self.bar.setVisible(active)
        self.details.setVisible(active and self.details_button.isChecked())
        if not active:
            return
        series = (
            w.phase_model_workspace.series_selector.currentText() or "No series selected"
        )
        scope, context = w.current_row_scope, w.scope_context
        self.line.setText(
            status_line(
                series,
                w._method_line_parts,
                scope_summary(context, scope) if context is not None else "",
            )
        )
        self.line.setToolTip(
            "\n".join(
                widget.text()
                for widget in self.detail_widgets
                if not widget.isHidden() and widget.text()
            )
        )
        # Painting has its own run chooser. The native configuration selector
        # does not control that page; keep the other pages' scope unchanged.
        painting = getattr(w, "_last_sidebar_selection", None) == ("geometry_facies", "layers")
        run_page = (scope is None or scope.run_selector_enabled) and not painting
        w.setup_series_label.hide()
        w._set_run_controls_visible(run_page)
        w.series_header.setVisible(run_page)
