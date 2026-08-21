"""Two-level accordion sidebar navigation for the DarSIA GUI."""

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .help import HelpPopup
from .icons import qta_icon


class SidebarRow(QWidget):
    """A single selectable item row in the sidebar: icon + label + hover help."""

    clicked = Signal()

    def __init__(
        self, action: str, checkbox_id: str, label: str, icon_name: str, help_text: str
    ):
        super().__init__()
        self.action = action
        self.checkbox_id = checkbox_id
        self.label = label
        self.icon_name = icon_name
        self.help_text = help_text
        self._is_selected = False
        self._help_timer = None
        self._help_popup = None

        # Layout: icon + label
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Icon
        icon = qta_icon(icon_name, scale_factor=1.0)
        icon_label = QLabel()
        icon_label.setPixmap(icon.pixmap(16, 16))
        layout.addWidget(icon_label)

        # Text label
        text_label = QLabel(label)
        text_label.setStyleSheet("color: inherit; font-size: 12px;")
        layout.addWidget(text_label)

        layout.addStretch()
        self.setLayout(layout)

        # Styling
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(32)
        self.update_selection_style()

    def mousePressEvent(self, event):
        """Emit clicked signal on mouse press."""
        self.clicked.emit()

    def enterEvent(self, event):
        """Show help popup on mouse enter after 1s delay."""
        if not self._help_timer:
            self._help_timer = QTimer()
            self._help_timer.timeout.connect(self._show_help_popup)
        self._help_timer.start(1000)

    def leaveEvent(self, event):
        """Hide help popup on mouse leave."""
        if self._help_timer:
            self._help_timer.stop()
        if self._help_popup:
            self._help_popup.hide()

    def _show_help_popup(self):
        """Display the help popup near the cursor."""
        if not self._help_popup:
            self._help_popup = HelpPopup(self.help_text, parent=self)
        self._help_popup.show_at_cursor()

    def set_selected(self, selected: bool):
        """Update selection state and visual highlight."""
        self._is_selected = selected
        self.update_selection_style()

    def update_selection_style(self):
        """Apply or remove selection styling."""
        if self._is_selected:
            stylesheet = (
                "QWidget { background-color: rgba(59, 89, 152, 0.3); "
                "border-left: 3px solid #3b5998; }"
            )
        else:
            stylesheet = "QWidget { background-color: transparent; }"
        self.setStyleSheet(stylesheet)


class GroupHeaderLabel(QLabel):
    """A group header label (e.g. 'Workflow steps') — bold, muted color."""

    def __init__(self, text: str):
        super().__init__(text)
        self.setStyleSheet(
            "QLabel { color: #888; font-weight: bold; font-size: 11px; "
            "margin-top: 8px; margin-bottom: 4px; }"
        )


class CategorySection(QWidget):
    """One accordion section for a category (e.g. Setup, Calibration, etc.)."""

    selection_changed = Signal(str, str)  # (action, checkbox_id)

    def __init__(
        self, action: str, category_label: str, category_icon: str, groups: list
    ):
        """
        Args:
            action: category identifier (e.g. "setup", "calibration")
            category_label: display name (e.g. "Setup")
            category_icon: qtawesome icon name (e.g. "fa5s.cogs")
            groups: list of (group_label, items) tuples, where items is list of
                    (label, checkbox_id, icon_name, help_text) tuples
        """
        super().__init__()
        self.action = action
        self.category_label = category_label
        self.category_icon = category_icon
        self.groups = groups
        self._is_expanded = False
        self._rows = {}  # checkbox_id -> SidebarRow

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header (clickable to expand/collapse)
        self.header_button = QPushButton()
        self.header_button.setFlat(True)
        self.header_button.setCursor(Qt.PointingHandCursor)
        self.header_button.clicked.connect(self._toggle_expand)
        self._update_header()
        layout.addWidget(self.header_button)

        # Items container (initially collapsed)
        self.items_container = QWidget()
        self.items_layout = QVBoxLayout(self.items_container)
        self.items_layout.setContentsMargins(0, 0, 0, 0)
        self.items_layout.setSpacing(0)

        # Populate items from groups
        for group_label, items in groups:
            self.items_layout.addWidget(GroupHeaderLabel(group_label))
            for label, checkbox_id, icon_name, help_text in items:
                row = SidebarRow(action, checkbox_id, label, icon_name, help_text)
                row.clicked.connect(lambda cid=checkbox_id: self._on_row_clicked(cid))
                self.items_layout.addWidget(row)
                self._rows[checkbox_id] = row

        self.items_layout.addStretch()
        self.items_container.setVisible(False)
        layout.addWidget(self.items_container)

    def _toggle_expand(self):
        """Toggle accordion expand/collapse."""
        self._is_expanded = not self._is_expanded
        self.items_container.setVisible(self._is_expanded)
        self._update_header()

    def _update_header(self):
        """Update header button text and chevron icon."""
        chevron = "fa5s.chevron-down" if self._is_expanded else "fa5s.chevron-right"
        icon = qta_icon(self.category_icon, scale_factor=1.0)
        chevron_icon = qta_icon(chevron, scale_factor=1.0)

        self.header_button.setIcon(icon)
        self.header_button.setText(self.category_label)
        self.header_button.setIconSize(
            __import__("PySide6.QtCore", fromlist=["QSize"]).QSize(16, 16)
        )
        self.header_button.setMinimumHeight(40)
        self.header_button.setStyleSheet(
            "QPushButton { text-align: left; padding-left: 8px; "
            "background-color: #f0f0f0; border-bottom: 1px solid #ddd; }"
        )

    def _on_row_clicked(self, checkbox_id: str):
        """Emit selection change when a row is clicked."""
        self.selection_changed.emit(self.action, checkbox_id)

    def get_row(self, checkbox_id: str) -> SidebarRow | None:
        """Get a specific row by checkbox_id."""
        return self._rows.get(checkbox_id)

    def deselect_all(self):
        """Clear selection from all rows in this category."""
        for row in self._rows.values():
            row.set_selected(False)


class Sidebar(QWidget):
    """Main sidebar widget: accordion of categories, single-select items."""

    selection_changed = Signal(str, str)  # (action, checkbox_id)

    def __init__(self, categories_data: dict):
        """
        Args:
            categories_data: dict mapping action -> (category_label, category_icon, groups)
                            where groups is list of (group_label, items) tuples
        """
        super().__init__()
        self.categories_data = categories_data
        self._sections = {}  # action -> CategorySection
        self._selected_row = None  # SidebarRow
        self._selected_action = None  # str
        self._selected_checkbox_id = None  # str

        # Main layout with scroll area
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Build category sections
        for action, (category_label, category_icon, groups) in categories_data.items():
            section = CategorySection(action, category_label, category_icon, groups)
            section.selection_changed.connect(self._on_selection_changed)
            container_layout.addWidget(section)
            self._sections[action] = section

        container_layout.addStretch()
        scroll_area.setWidget(container)
        layout.addWidget(scroll_area)

    def _on_selection_changed(self, action: str, checkbox_id: str):
        """Handle selection change: deselect previous, highlight new, emit signal."""
        # Deselect previous
        if self._selected_row is not None:
            self._selected_row.set_selected(False)

        # Select new
        row = self._sections[action].get_row(checkbox_id)
        if row:
            row.set_selected(True)
            self._selected_row = row
            self._selected_action = action
            self._selected_checkbox_id = checkbox_id

            # Emit signal for external wiring
            self.selection_changed.emit(action, checkbox_id)

    def select(self, action: str, checkbox_id: str):
        """Public API: programmatically select a row."""
        if action in self._sections:
            self._on_selection_changed(action, checkbox_id)

    def get_selected(self) -> tuple[str, str] | None:
        """Get current selection as (action, checkbox_id), or None if nothing selected."""
        if self._selected_action and self._selected_checkbox_id:
            return (self._selected_action, self._selected_checkbox_id)
        return None
