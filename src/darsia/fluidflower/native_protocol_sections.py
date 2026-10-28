"""Contextual protocol navigation using the original settings and readers."""
from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
    QScrollArea, QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)

from .overflow_data_page import OverflowDataPage, read_overflow_data
from .ui_metadata_preview import MetadataPreview


class ProtocolOverflowPage(OverflowDataPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.reader = MetadataPreview(self)
        self.reader.completed.connect(self._loaded)
        self._loaded_root = None

    def show_series(self, series_root):
        self.series_root = Path(series_root) if series_root is not None else None
        root = self.series_root
        self.summary.setText("Reading the configured overflow protocols…")
        self.reload_button.setEnabled(False)
        self.reader.start(lambda cancelled: read_overflow_data(root))

    def _loaded(self, data, error):
        if error:
            self.summary.setText("Overflow overview unavailable: " + error)
            self.reload_button.setEnabled(self.series_root is not None)
        else:
            self._loaded_root = self.series_root
            self.set_data(data)


class NativeProtocolSections:
    TITLES = {
        "data": "Images and results",
        "imaging": "Image times",
        "injection": "Injection",
        "pressure_temperature": "Pressure and temperature",
        "exclusions": "Excluded images (optional)",
        "overflow": "Overflow (optional)",
        "workbooks": "Protocol workbooks",
    }
    # The original [options.preprocessing] "Show plots" is left out: preparing
    # the protocols ignores it, so it would be a switch without an effect.
    LEFT_OUT = ("options",)
    PRESSURE_DATASETS_TEXT = (
        "Each run's pressure from a pressure dataset index (*.json), with its imaging "
        "window and recorded gaps. Read only: nothing is fitted, adopted or written.")
    WORKBOOKS_TEXT = (
        "Shared protocol workbooks hold the protocol tables of all runs: one workbook per "
        "protocol (image times, injection, pressure and temperature, overflow) with one "
        "worksheet per run. The conversion previews the saved tables and creates the "
        "workbooks; the run configurations use them only after Activate workbooks.")

    def __init__(self, window):
        self.window = window
        self.selected_key = "data"
        self.navigation = None
        self.pages = {}
        self.overflow_page = None
        render = window.settings_factory._render_settings_tabs

        def render_settings(settings_by_section):
            view = getattr(window, "_last_settings_view", None)
            protocol_view = (view is not None and tuple(view[:2]) == ("action", "preprocessing")
                             and len(view) == 3 and list(view[2]) == ["protocols"])
            if protocol_view:
                settings_by_section = self._sections(settings_by_section)
            result = render(settings_by_section)
            if protocol_view:
                self._build_navigation(settings_by_section)
            return result

        render_settings.__wrapped__ = render
        window.settings_factory._render_settings_tabs = render_settings
        self.update_method(getattr(getattr(window, "_saved_method", None), "active", ""))

    @classmethod
    def _sections(cls, original):
        """Regroup fields without changing their keys, values or dependencies."""
        sections = {key: [] for key in cls.TITLES}
        for section, fields in original.items():
            if section in cls.LEFT_OUT:
                continue
            for source in fields:
                field = deepcopy(source)
                key = field["key"].removeprefix("protocols.")
                if section == "protocols":
                    if key == "blacklist":
                        destination = "exclusions"
                    elif key == "overflow":
                        destination = "overflow"
                    elif key.startswith("injection"):
                        destination = "injection"
                    elif key.startswith(("pressure", "temperature")):
                        destination = "pressure_temperature"
                    else:
                        destination = "imaging"
                    # The navigation already names the source. A second generic
                    # "Operating conditions" group obscures those identities.
                    field.pop("group_name", None)
                    field.pop("group", None)
                else:
                    destination = section
                sections.setdefault(destination, []).append(field)
        return {key: values for key, values in sections.items() if values}

    def update_method(self, method):
        rows = self.window.sidebar._sections["setup"]
        rows.get_row("overflow_data:runs").hide()
        # Setup retains the reference and colour-path image choices. Calibration
        # and production have their own selections; neither replaces this row.
        rows.get_row("image_selection:plan").show()
        # The phase route builds through its shared Apply/Build job. These are
        # the original per-run executors, retained for the original method.
        for key in ("setup:rig", "setup:all"):
            rows.get_row(key).setVisible(method != "phase_model")

    def _build_navigation(self, sections):
        layout = self.window.settings_layout
        tabs = next((layout.itemAt(i).widget() for i in range(layout.count())
                     if isinstance(layout.itemAt(i).widget(), QTabWidget)), None)
        if tabs is None:
            return
        host = QWidget()
        host.setObjectName("nativeProtocolSections")
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        navigation = QListWidget()
        navigation.setAccessibleName("Protocol sections")
        navigation.setMinimumWidth(180)
        navigation.setMaximumWidth(260)
        row.addWidget(navigation)
        stack = QStackedWidget()
        row.addWidget(stack, 1)
        self.navigation, self.stack = navigation, stack
        self.pages = {}
        for key in sections:
            page = tabs.widget(0)
            tabs.removeTab(0)
            self.pages[key] = page
            scroll = QScrollArea()
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            stack.addWidget(scroll)
            item = QListWidgetItem(self.TITLES.get(key, key.replace("_", " ").title()))
            item.setData(Qt.ItemDataRole.UserRole, key)
            navigation.addItem(item)
        self.overflow_page = None
        if "overflow" in self.pages:
            self.overflow_page = ProtocolOverflowPage(self.pages["overflow"])
            self.pages["overflow"].layout().addRow(self.overflow_page)
        # Decorations describe the existing sources; they never generate a path
        # or save a configuration just because this page was opened.
        from .native_protocol_fields import decorate_protocol_fields
        self.field_notes = decorate_protocol_fields(self.window, self.pages)
        self._add_pressure_datasets()
        self._add_workbooks_page(navigation, stack)
        index = layout.indexOf(tabs)
        layout.removeWidget(tabs)
        tabs.deleteLater()
        layout.insertWidget(index, host, 1)
        navigation.currentRowChanged.connect(self._selected)
        wanted = next((i for i in range(navigation.count())
                       if navigation.item(i).data(Qt.ItemDataRole.UserRole) == self.selected_key), 0)
        navigation.setCurrentRow(wanted)

    def _add_pressure_datasets(self):
        """The read-only pressure dataset view, on the page of the pressure protocol."""
        page = self.pages.get("pressure_temperature")
        running_pressure = getattr(self.window, "running_pressure", None)
        if page is None or running_pressure is None:
            return
        group = QGroupBox("Pressure datasets")
        layout = QVBoxLayout(group)
        text = QLabel(self.PRESSURE_DATASETS_TEXT)
        text.setWordWrap(True)
        layout.addWidget(text)
        button = QPushButton("Show pressure datasets…")
        button.clicked.connect(lambda: running_pressure.open())
        row = QHBoxLayout()
        row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        page.layout().addRow(group)
        self.pressure_datasets_button = button

    def _add_workbooks_page(self, navigation, stack):
        """Converting the series' protocol tables to shared workbooks, as one list page."""
        if getattr(self.window, "protocol_workbooks_panel", None) is None:
            return
        from .protocol_workbooks_ui import open_protocol_workbooks

        page = QWidget()
        layout = QVBoxLayout(page)
        text = QLabel(self.WORKBOOKS_TEXT)
        text.setWordWrap(True)
        layout.addWidget(text)
        button = QPushButton("Open workbook conversion…")
        button.clicked.connect(lambda: open_protocol_workbooks(self.window))
        row = QHBoxLayout()
        row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(1)
        self.pages["workbooks"] = page
        self.workbooks_button = button
        stack.addWidget(page)
        item = QListWidgetItem(self.TITLES["workbooks"])
        item.setData(Qt.ItemDataRole.UserRole, "workbooks")
        navigation.addItem(item)

    def _selected(self, index):
        if index < 0:
            return
        self.stack.setCurrentIndex(index)
        self.selected_key = self.navigation.item(index).data(Qt.ItemDataRole.UserRole)
        if self.selected_key == "overflow" and self.overflow_page is not None:
            root = self.window._method_series_root()
            if self.overflow_page._loaded_root != root:
                self.overflow_page.show_series(root)
