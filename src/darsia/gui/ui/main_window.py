import ast
import re
from pathlib import Path

import psutil
import toml
from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .about_dialog import AboutDialog
from .analysis import AnalysisTab
from .calibration import CalibrationTab
from .comparison import ComparisonTab
from .helper import HelperTab
from .menu import MenuBuilder
from .recent_files import add_recent_config, remove_recent_config
from .settings import SettingsFactory
from .setup import SetupTab
from .theme import apply_theme
from .theme import set_theme as save_theme
from .toolbar import ToolbarBuilder
from .utils_tab import UtilsTab


def _unwrap_composite_widget(value):
    """Unwrap a composite field widget to extract the real editable control.

    Composite field widgets returned by create_simple_input(), create_bool_input(),
    create_dropdown_input(), and create_file_chooser() are QWidget wrappers
    containing [control + type_label + help_button] in a QHBoxLayout.

    This helper finds and returns the actual editable control (QLineEdit, QComboBox,
    or QCheckBox) embedded inside such a wrapper. If value is not a composite wrapper,
    or is already a bare control widget, it is returned unchanged.

    NOTE: Necessary for making saving settings work.

    Args:
        value: A widget or other value from settings_inputs.

    Returns:
        The unwrapped editable control if value is a composite wrapper; value itself
        otherwise.
    """
    if (
        type(value) is QWidget
    ):  # Exact type, not isinstance (to avoid unwrapping subclasses)
        for widget_type in (QLineEdit, QComboBox, QCheckBox):
            found = value.findChild(widget_type)
            if found is not None:
                return found
    return value


class MainWindow(QMainWindow):
    """The main class containing the window and the relevant methods for the visualization."""

    log_message = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DarSIA")

        # Connect log_message signal to append_log slot for thread-safe logging
        self.log_message.connect(self._append_log)

        # Set window icon
        logo_path = (
            Path(__file__).parent
            / "../../presets/workflows/interface/DarSIA_Horisontal_Positiv_part.png"
        )
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))

        # Set up the menu bar
        self.menu_builder = MenuBuilder(self)
        self.menu_builder.build()

        # Set up the toolbar
        self.toolbar_builder = ToolbarBuilder(self, self.menu_builder)
        self.toolbar_builder.build()

        # Setting up the three upper layouts
        upper_container = QWidget()
        upper_layout = QVBoxLayout(upper_container)
        upper_container.setFixedHeight(100)

        upper_mid_container = QWidget()
        upper_mid_layout = QVBoxLayout(upper_mid_container)

        upper_right_container = QWidget()
        upper_right_layout = QVBoxLayout(upper_right_container)

        # Setting up the left upper layout
        upper_layout.addWidget(QLabel("Loaded config:"))
        self.config_path_label = QLabel("No config loaded.")
        self.config_path_label.setWordWrap(True)
        upper_layout.addWidget(self.config_path_label)

        # Storage for file/folder chooser widgets used by settings.py's
        # FileDialogHelper (e.g. depth.measurements, facies.props).
        self.chosen_files = {}
        self.baseline_images = []
        self.baseline_container = QWidget()
        self.baseline_layout = QVBoxLayout(self.baseline_container)
        self.baseline_layout.setContentsMargins(0, 0, 0, 0)

        # Add stretch to push the config label to the top
        upper_layout.addStretch()

        # Setting up the middle upper layout with tabs
        tabs = QTabWidget()

        # Initialize tab managers
        self.setup_tab = SetupTab(self)
        self.calibration_tab = CalibrationTab(self)
        self.analysis_tab = AnalysisTab(self)
        self.helper_tab = HelperTab(self)
        self.comparison_tab = ComparisonTab(self)
        self.utils_tab = UtilsTab(self)

        # Add tabs
        tabs.addTab(self.setup_tab.create_tab(), "Setup")
        tabs.addTab(self.calibration_tab.create_tab(), "Calibration")
        tabs.addTab(self.analysis_tab.create_tab(), "Analysis")
        tabs.addTab(self.helper_tab.create_tab(), "Helper")
        tabs.addTab(self.comparison_tab.create_tab(), "Comparison")
        tabs.addTab(self.utils_tab.create_tab(), "Utils")

        upper_mid_layout.addWidget(tabs)

        # Initialize settings factory
        self.settings_factory = SettingsFactory(self)

        # Setting up the right upper layout
        # Create settings container with scroll area
        self.settings_container = QWidget()
        self.settings_layout = QVBoxLayout(self.settings_container)
        self.settings_layout.setContentsMargins(0, 0, 0, 0)

        # Add scroll area for settings
        self.settings_scroll_area = QScrollArea()
        self.settings_scroll_area.setWidget(self.settings_container)
        self.settings_scroll_area.setWidgetResizable(True)
        upper_right_layout.addWidget(self.settings_scroll_area)

        # Store config
        self.config_file = ""
        self.config_dict = {}
        self.settings_inputs = {}  # Store setting input widgets
        self._last_settings_view = (
            None  # Track the last displayed settings view for refresh after save
        )

        # Create logging container with its own scroll area
        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_label = QLabel("Logging:")
        log_layout.addWidget(log_label)

        # Add a text edit for logging output
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        log_scroll_area = QScrollArea()
        log_scroll_area.setWidget(log_container)
        log_scroll_area.setWidgetResizable(True)

        # Splitter for the upper half of the GUI
        upper_splitter = QSplitter(Qt.Horizontal)
        upper_splitter.addWidget(upper_mid_container)
        upper_splitter.addWidget(upper_right_container)
        upper_splitter.setStretchFactor(0, 1)  # left panel: 1/7 of space
        upper_splitter.setStretchFactor(1, 7)  # right panel: 6/7 of space

        # Vertical splitter between the log-window and the rest of the GUI
        content_splitter = QSplitter(Qt.Vertical)
        content_splitter.addWidget(upper_splitter)
        content_splitter.addWidget(log_scroll_area)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 1)

        # Create central widget with all components
        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        self.setCentralWidget(main_container)
        main_layout.addWidget(upper_container)
        main_layout.addWidget(content_splitter)

        self.showMaximized()

        # Display welcome message
        self.print_log("No config loaded. Use:")
        self.print_log("- <b><i>File > New</i></b> to start a new config, ")
        self.print_log(
            "- <b><i>File > Open Config...</i></b> to open an existing one, or "
        )
        self.print_log(
            "- <b><i>File > Open Recent</i></b> to open a recently-used config."
        )

    def _sync_settings_inputs_to_config_dict(self):
        """Flush all live settings widget values into self.config_dict.

        Shared by save_settings() (before writing to disk) and
        _render_settings_tabs() (before it deletes the current widgets and
        resets settings_inputs), so in-progress edits are never silently
        discarded by navigating between tabs before saving.
        """
        # First pass: collect group checkbox states and determine which sub-inputs to skip
        group_active_names: dict[str, set[str]] = {}
        skip_keys: set[str] = set()

        for key, value in self.settings_inputs.items():
            if isinstance(value, dict) and "checkbox" in value:
                # Handle section-level boolean toggle (active_bool_key)
                if "bool_key" in value:
                    self.settings_factory.set_value(
                        self.config_dict,
                        value["bool_key"],
                        value["checkbox"].isChecked(),
                    )
                    if not value["checkbox"].isChecked():
                        skip_keys.update(value["sub_inputs"].keys())
                    continue

                # Handle field-level toggles within a section (active_list_key)
                # This is a group dict; record its active state
                active_list_key = value["active_list_key"]
                group_active_names.setdefault(active_list_key, set())
                if value["checkbox"].isChecked():
                    group_active_names[active_list_key].add(value["name"])
                else:
                    # Group is unchecked; skip saving any of its sub-inputs
                    skip_keys.update(value["sub_inputs"].keys())

        # Second pass: save all regular values (non-group dicts),
        # skipping unchecked group sub-inputs.
        for key, value in self.settings_inputs.items():
            # Skip group result dicts (marked with is_group_result)
            if isinstance(value, dict) and value.get("is_group_result"):
                continue
            # Skip group dicts with checkboxes (already handled above)
            if isinstance(value, dict) and "checkbox" in value:
                continue
            # Skip path_map, int_group_list, int_list_map, time_interval_map, time_window_map,
            # time_data_map, path_data_map, registry_key_list, format_map dicts (handled below)
            if isinstance(value, dict) and (
                "path_map" in value
                or "int_group_list" in value
                or "int_list_map" in value
                or "time_interval_map" in value
                or "time_window_map" in value
                or "time_data_map" in value
                or "path_data_map" in value
                or "registry_key_list" in value
                or "format_map" in value
                or "format_key_list" in value
            ):
                continue
            # Skip sub-inputs of unchecked groups
            if key in skip_keys:
                continue

            # Unwrap composite field widgets (wrapper widget containing type label + help button)
            # to extract the actual editable control (QLineEdit, QComboBox, QCheckBox)
            value = _unwrap_composite_widget(value)

            try:
                if isinstance(value, QLineEdit):
                    self.settings_factory.set_value(
                        self.config_dict, key, ast.literal_eval(value.text())
                    )
                elif isinstance(value, QComboBox):
                    self.settings_factory.set_value(
                        self.config_dict, key, value.currentText()
                    )
                elif isinstance(value, QCheckBox):
                    self.settings_factory.set_value(
                        self.config_dict, key, value.isChecked()
                    )
                elif isinstance(value, list):
                    if len(value) > 0:
                        if isinstance(value[0], QCheckBox):
                            self.settings_factory.set_value(
                                self.config_dict,
                                key,
                                [item.text() for item in value if item.isChecked()],
                            )
                        elif isinstance(value[0], QLineEdit):
                            self.settings_factory.set_value(
                                self.config_dict,
                                key,
                                [item.text() for item in value if item.text().strip()],
                            )
            except (ValueError, SyntaxError):
                if hasattr(value, "text"):
                    self.settings_factory.set_value(self.config_dict, key, value.text())

        # Third pass: save path_map dicts (key -> value mappings)
        for key, value in self.settings_inputs.items():
            if isinstance(value, dict) and "path_map" in value:
                rows = value["rows"]
                result = {
                    k.text(): v.text()
                    for k, v in rows
                    if k.text().strip() and v.text().strip()
                }
                self.settings_factory.set_value(self.config_dict, key, result)

        # Fourth pass: parse int_group_list rows into list[list[int]]
        for key, value in self.settings_inputs.items():
            if isinstance(value, dict) and "int_group_list" in value:
                groups = []
                for edit in value["rows"]:
                    text = edit.text().strip()
                    if not text:
                        continue
                    tokens = [t for t in re.split(r"[,\s]+", text) if t]
                    try:
                        groups.append([int(t) for t in tokens])
                    except ValueError:
                        self.print_log(
                            f"Skipping invalid group '{text}' for {key}: not all-integer."
                        )
                self.settings_factory.set_value(self.config_dict, key, groups)

        # Fifth pass: parse int_list_map rows into dict[int, list[int]]
        for key, value in self.settings_inputs.items():
            if isinstance(value, dict) and "int_list_map" in value:
                result = {}
                for key_edit, value_edit in value["rows"]:
                    key_text = key_edit.text().strip()
                    value_text = value_edit.text().strip()
                    if not key_text or not value_text:
                        continue
                    tokens = [t for t in re.split(r"[,\s]+", value_text) if t]
                    try:
                        result[int(key_text)] = [int(t) for t in tokens]
                    except ValueError:
                        self.print_log(
                            f"Skipping invalid row '{key_text}: {value_text}' for {key}: "
                            "key and values must all be integers."
                        )
                if value.get("flatten_in_section"):
                    section = value["section"]
                    section_dict = self.config_dict.setdefault(section, {})
                    # Drop id sub-tables removed in the GUI (has "labels", not in new result)
                    stale_ids = [
                        k
                        for k, v in section_dict.items()
                        if isinstance(v, dict)
                        and "labels" in v
                        and k not in {str(i) for i in result}
                    ]
                    for k in stale_ids:
                        del section_dict[k]
                    for facies_id, labels in result.items():
                        section_dict[str(facies_id)] = {"labels": labels}
                else:
                    self.settings_factory.set_value(self.config_dict, key, result)

        # Sixth pass: parse time_interval_map rows into dict[str, {start/end/num/tol}]
        for key, value in self.settings_inputs.items():
            if isinstance(value, dict) and "time_interval_map" in value:
                result = {}
                for name_edit, start_edit, end_edit, num_edit, tol_edit in value[
                    "rows"
                ]:
                    name_text = name_edit.text().strip()
                    if not name_text:
                        continue
                    result[name_text] = {
                        "start": start_edit.text().strip() or "00:00:00",
                        "end": end_edit.text().strip() or "00:00:00",
                        "num": (
                            int(num_edit.text().strip())
                            if num_edit.text().strip().isdigit()
                            else 0
                        ),
                        "tol": tol_edit.text().strip() or "00:00:00",
                    }
                self.settings_factory.set_value(self.config_dict, key, result)

        # Seventh pass: parse time_window_map rows into dict[str, {start/end}]
        for key, value in self.settings_inputs.items():
            if isinstance(value, dict) and "time_window_map" in value:
                result = {}
                for name_edit, start_edit, end_edit in value["rows"]:
                    name_text = name_edit.text().strip()
                    if not name_text:
                        continue
                    result[name_text] = {
                        "start": start_edit.text().strip() or "00:00:00",
                        "end": end_edit.text().strip() or "00:00:00",
                    }
                self.settings_factory.set_value(self.config_dict, key, result)

        # Eighth pass: parse time_data_map rows into dict[str, {times}]
        for key, value in self.settings_inputs.items():
            if isinstance(value, dict) and "time_data_map" in value:
                result = {}
                for name_edit, times_edit in value["rows"]:
                    name_text = name_edit.text().strip()
                    if not name_text:
                        continue
                    times_text = times_edit.text().strip()
                    if times_text:
                        # Parse comma-separated times
                        times_list = [
                            t.strip() for t in times_text.split(",") if t.strip()
                        ]
                        result[name_text] = {"times": times_list}
                    else:
                        result[name_text] = {"times": []}
                self.settings_factory.set_value(self.config_dict, key, result)

        # Ninth pass: parse path_data_map rows into dict[str, {paths}]
        for key, value in self.settings_inputs.items():
            if isinstance(value, dict) and "path_data_map" in value:
                result = {}
                for name_edit, paths_edit in value["rows"]:
                    name_text = name_edit.text().strip()
                    if not name_text:
                        continue
                    paths_text = paths_edit.text().strip()
                    if paths_text:
                        # Parse comma-separated paths
                        paths_list = [
                            p.strip() for p in paths_text.split(",") if p.strip()
                        ]
                        result[name_text] = {"paths": paths_list}
                    else:
                        result[name_text] = {"paths": []}
                self.settings_factory.set_value(self.config_dict, key, result)

        # Tenth pass: parse registry_key_list rows into list[str]
        for key, value in self.settings_inputs.items():
            if isinstance(value, dict) and "registry_key_list" in value:
                result = []
                seen = set()
                for combo in value["rows"]:
                    text = combo.currentText().strip()
                    if text and text not in seen:
                        result.append(text)
                        seen.add(text)
                # Write as list[str] (empty list if no selections, never None)
                if result:
                    self.settings_factory.set_value(self.config_dict, key, result)
                else:
                    # Empty selection: delete the key (or set to empty list)
                    # For now, set to empty list to match the field's Optional nature
                    self.settings_factory.set_value(self.config_dict, key, None)

        # Eleventh pass: parse format_map rows into list[dict] for [[format]] TOML shape
        for key, value in self.settings_inputs.items():
            if isinstance(value, dict) and "format_map" in value:
                result = []
                for (
                    name_edit,
                    type_combo,
                    filename_pattern_combo,
                    resolution_edit,
                    keep_ratio_check,
                    dpi_edit,
                    cmap_edit,
                    quality_edit,
                    compression_edit,
                    dtype_edit,
                    delimiter_edit,
                    header_edit,
                    float_format_edit,
                ) in value["rows"]:
                    name_text = name_edit.text().strip()
                    if not name_text:
                        continue

                    # Always-required fields
                    entry = {
                        "type": type_combo.currentText().strip(),
                        "name": name_text,
                        "filename_pattern": filename_pattern_combo.currentText().strip(),
                    }

                    # Optional fields: empty text means None/omit
                    # resolution: comma-separated rows,cols
                    res_text = resolution_edit.text().strip()
                    if res_text:
                        parts = [p.strip() for p in res_text.split(",")]
                        if len(parts) == 2:
                            try:
                                entry["resolution"] = [int(parts[0]), int(parts[1])]
                            except ValueError:
                                pass

                    # keep_ratio (bool)
                    if keep_ratio_check.isChecked():
                        entry["keep_ratio"] = True

                    # Type-specific optional fields
                    dpi_text = dpi_edit.text().strip()
                    if dpi_text:
                        try:
                            entry["dpi"] = int(dpi_text)
                        except ValueError:
                            pass

                    cmap_text = cmap_edit.text().strip()
                    if cmap_text:
                        entry["cmap"] = cmap_text

                    quality_text = quality_edit.text().strip()
                    if quality_text:
                        try:
                            entry["quality"] = int(quality_text)
                        except ValueError:
                            pass

                    compression_text = compression_edit.text().strip()
                    if compression_text:
                        try:
                            entry["compression"] = int(compression_text)
                        except ValueError:
                            pass

                    dtype_text = dtype_edit.text().strip()
                    if dtype_text:
                        entry["dtype"] = dtype_text

                    # CSV-specific fields
                    delimiter_text = delimiter_edit.text().strip()
                    if delimiter_text and delimiter_text != ",":
                        entry["delimiter"] = delimiter_text

                    header_text = header_edit.text().strip()
                    if header_text:
                        entry["header"] = header_text

                    float_format_text = float_format_edit.text().strip()
                    if float_format_text and float_format_text != "{:.2e}":
                        entry["float_format"] = float_format_text

                    result.append(entry)

                # Write as list[dict] directly to config_dict["format"]
                self.config_dict["format"] = result

        # Twelfth pass: parse format_key_list rows into list[str] (or single str)
        for key, value in self.settings_inputs.items():
            if isinstance(value, dict) and "format_key_list" in value:
                result = []
                seen = set()
                for combo in value["rows"]:
                    text = combo.currentText().strip()
                    if text and text not in seen:
                        result.append(text)
                        seen.add(text)
                if value.get("max_rows") == 1:
                    self.settings_factory.set_value(
                        self.config_dict, key, result[0] if result else None
                    )
                else:
                    self.settings_factory.set_value(
                        self.config_dict, key, result if result else None
                    )

        # Thirteenth pass: write all active lists
        for active_list_key, names in group_active_names.items():
            self.settings_factory.set_value(
                self.config_dict, active_list_key, sorted(names)
            )

    def save_settings(self):
        """Save the current settings to the loaded config file."""
        self._sync_settings_inputs_to_config_dict()

        if self.config_file != "":
            with open(self.config_file, "w") as f:
                toml.dump(self.config_dict, f)
            self.print_log(f"Settings saved to {self.config_file}")

            # Refresh the currently-displayed settings panel if one is open, so
            # newly added registry entries immediately appear in dropdowns etc.
            if self._last_settings_view:
                # Capture the currently active tab index
                current_tab_index = None
                if self.settings_layout.count() > 0:
                    widget = self.settings_layout.itemAt(0).widget()
                    if isinstance(widget, QTabWidget):
                        current_tab_index = widget.currentIndex()

                # Replay the last view to refresh
                if self._last_settings_view[0] == "full":
                    self.display_full_settings()
                elif self._last_settings_view[0] == "action":
                    _, action, checked_ids = self._last_settings_view
                    self.display_settings(action, checked_ids)

                # Restore the tab index if possible
                if current_tab_index is not None:
                    widget = self.settings_layout.itemAt(0).widget()
                    if isinstance(widget, QTabWidget):
                        if 0 <= current_tab_index < widget.count():
                            widget.setCurrentIndex(current_tab_index)
        else:
            self.print_log("Settings not saved, please choose a config file")

    def add_baseline_image(self):
        """Add a new baseline image chooser row."""
        baseline_index = len(self.baseline_images)

        chooser_container = QWidget()
        chooser_layout = QHBoxLayout(chooser_container)
        chooser_layout.setContentsMargins(0, 5, 0, 5)

        # Browse button
        browse_button = QPushButton(f"Browse Baseline {baseline_index + 1}")
        browse_button.setMinimumWidth(200)

        # Path label to display selected path
        path_label = QLabel("No file chosen")
        path_label.setStyleSheet("color: white;")

        # Remove button (only show if 2+ baselines)
        remove_button = QPushButton("Remove")
        remove_button.setMaximumWidth(80)
        remove_button.setVisible(len(self.baseline_images) >= 1)

        # Store baseline info
        baseline_info = {
            "index": baseline_index,
            "path": "",
            "label": path_label,
            "browse_button": browse_button,
            "remove_button": remove_button,
            "container": chooser_container,
        }
        self.baseline_images.append(baseline_info)

        # Connect buttons
        browse_button.clicked.connect(
            lambda: self.browse_baseline_image(baseline_index)
        )
        remove_button.clicked.connect(
            lambda: self.remove_baseline_image(baseline_index)
        )

        chooser_layout.addWidget(browse_button)
        chooser_layout.addWidget(path_label)
        chooser_layout.addWidget(remove_button)
        chooser_layout.addStretch()

        self.baseline_layout.addWidget(chooser_container)

        # Update all remove buttons visibility
        self.update_baseline_remove_buttons()

    def remove_baseline_image(self, index):
        """Remove a baseline image by index."""
        if index < len(self.baseline_images):
            baseline_info = self.baseline_images[index]
            baseline_info["container"].deleteLater()
            self.baseline_images.pop(index)

            # Reindex remaining baselines
            for i, info in enumerate(self.baseline_images):
                info["index"] = i
                info["browse_button"].setText(f"Browse Baseline {i + 1}")

            # Update remove buttons visibility
            self.update_baseline_remove_buttons()
            self.print_log(f"Removed baseline image {index + 1}")

    def update_baseline_remove_buttons(self):
        """Update visibility of remove buttons based on number of baselines."""
        should_show = len(self.baseline_images) > 1
        for baseline_info in self.baseline_images:
            baseline_info["remove_button"].setVisible(should_show)

    def browse_baseline_image(self, index):
        """Open file dialog for baseline image and store selected path."""
        file_filter = "Image Files (*.jpg *.jpeg *.png);;All Files (*)"
        selected_path, _ = QFileDialog.getOpenFileName(
            self, f"Select Baseline Image {index + 1}", "", file_filter
        )

        if selected_path and index < len(self.baseline_images):
            baseline_info = self.baseline_images[index]
            baseline_info["path"] = selected_path
            baseline_info["label"].setText(selected_path)
            self.print_log(f"Selected baseline image {index + 1}: {selected_path}")

    def browse_file(self, key):
        """Open file/folder dialog and store selected path."""
        file_info = self.chosen_files[key]
        is_directory = file_info["is_directory"]
        file_filter = file_info["filter"]

        if is_directory:
            selected_path = QFileDialog.getExistingDirectory(
                self, f"Select {key.replace('_', ' ')}"
            )
        else:
            selected_path, _ = QFileDialog.getOpenFileName(
                self, f"Select {key.replace('_', ' ')}", "", file_filter
            )

        if selected_path:
            self.chosen_files[key]["path"] = selected_path
            file_info["label"].setText(selected_path)
            file_info["label"].setStyleSheet("color: white;")
            self.print_log(f"Selected {key}: {selected_path}")

    def get_checked_checkbox_ids(self, checkboxes):
        """
        Function to get the ids of checked checkboxes from a list of (id, checkbox) tuples.
        """
        checked_ids = []
        for checkbox_id, checkbox in checkboxes:
            if checkbox.isChecked():
                checked_ids.append(checkbox_id)
        return checked_ids

    def _render_settings_tabs(self, settings_by_section):
        """Render settings_by_section dict into tabbed layout.

        Clears settings_layout and builds QTabWidget with one tab per section,
        populating form inputs via settings_factory.build_tab_form.
        Shared with both display_settings and display_full_settings.
        """
        # Flush any pending edits into config_dict before the widgets holding
        # them are destroyed below, so switching tabs never silently drops
        # unsaved changes (e.g. newly added registry rows).
        if self.settings_inputs:
            self._sync_settings_inputs_to_config_dict()

        while self.settings_layout.count():
            child = self.settings_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.settings_inputs = {}

        # If no sections, add a message and return
        if not settings_by_section:
            self.settings_layout.addWidget(QLabel("No settings available"))
            self.settings_layout.addStretch()
            return

        # Create a tabbed interface with one tab per section
        tabs = QTabWidget()

        # Iterate through sections in order (dict preserves insertion order in Python 3.7+)
        for section, settings_list in settings_by_section.items():
            # Create a scroll area and container for this section's fields
            tab_container = QWidget()
            tab_form = QFormLayout(tab_container)
            tab_form.setContentsMargins(8, 8, 8, 8)  # Padding on all sides

            # Create form_context for multi_file/path_map dynamic row insertion
            form_context = {"form": tab_form}

            # Build rows in the form, handling grouping via group_name metadata
            self.settings_factory.build_tab_form(
                tab_form, settings_list, form_context=form_context
            )

            # Add stretch at the end of the section (QFormLayout doesn't auto-stretch rows)
            tab_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
            # Push content to top by adding a stretch row at the end
            tab_form.addItem(QVBoxLayout())

            # Add this section as a tab (capitalize section name for display)
            tabs.addTab(tab_container, section.capitalize())

        self.settings_layout.addWidget(tabs)
        self.settings_layout.addStretch()

    def display_settings(self, action, checked_ids):
        """Display the relevant settings based on the action being used."""
        self._last_settings_view = ("action", action, checked_ids)
        relevant_settings = self.settings_factory.get_relevant_settings(
            action, checked_ids
        )
        self._render_settings_tabs(relevant_settings)

    def display_full_settings(self):
        """Display all fixed-schema sections in the settings panel."""
        self._last_settings_view = ("full",)
        all_settings = self.settings_factory.get_all_settings()
        self._render_settings_tabs(all_settings)

    def set_theme(self, mode: str):
        """Set the application theme (System/Light/Dark).

        Parameters
        ----------
        mode : str
            One of "System", "Light", or "Dark".
        """
        from PySide6.QtWidgets import QApplication

        apply_theme(QApplication.instance(), mode)
        save_theme(mode)
        self.print_log(f"Theme set to {mode}")

    def show_about_dialog(self):
        """Show the About dialog."""
        AboutDialog(self).exec()

    def new_config(self):
        """Create a new empty config file at a chosen path and open it."""
        file, _ = QFileDialog.getSaveFileName(
            self, "New Config File", "", "TOML Files (*.toml);;All Files (*)"
        )
        if not file:
            return
        try:
            with open(file, "w") as f:
                toml.dump({}, f)
        except Exception as e:
            self.print_log(f"Error creating config file: {e}")
            return
        self.config_file = file
        self.config_path_label.setText(file)
        self.config_dict = {}
        add_recent_config(file)
        self.print_log(f"New config created and opened: {file}")

    def open_config(self):
        """Open a config file via dialog and load it immediately."""
        file, _ = QFileDialog.getOpenFileName(
            self, "Open Config File", "", "TOML Files (*.toml);;All Files (*)"
        )
        if not file:
            return
        self.config_path_label.setText(file)
        self.load_config()

    def save_config_as(self):
        """Save current settings to a new config file chosen via dialog."""
        file, _ = QFileDialog.getSaveFileName(
            self, "Save Config As", "", "TOML Files (*.toml);;All Files (*)"
        )
        if not file:
            return
        self.config_file = file
        self.config_path_label.setText(file)
        add_recent_config(file)
        self.save_settings()

    def open_recent_config(self, path):
        """Open a config file from the recent-files list."""
        if not Path(path).exists():
            self.print_log(f"Recent config file no longer exists: {path}")
            remove_recent_config(path)
            return
        self.config_path_label.setText(path)
        self.load_config()

    def load_config(self):
        """Method that loads the config file chosen in the GUI."""
        file = self.config_path_label.text()
        if not file:
            self.print_log(
                """No config file selected. """
                """Use <b><i>File > Open Config</i></b> to select a config file."""
            )
            return
        try:
            with open(file, "r") as f:
                self.config_dict = toml.load(f)
        except Exception as e:
            self.print_log(f"Error loading config file: {e}")
            return
        self.config_file = file
        add_recent_config(file)
        self.print_log("Config loaded")

    def print_log(self, text):
        """Emit log_message signal to append text to log window (thread-safe via Qt signal)."""
        self.log_message.emit(text)

    def _append_log(self, text):
        """Slot that appends text to log window and prints to console."""
        self.log_text.append(text)
        print(text)

    def start_workflow_process(self, argv, run_button, abort_button, cwd=None):
        """Launch argv as a QProcess, streaming merged stdout/stderr to the log.
        Disables run_button and shows/enables abort_button while running; restores
        button state and logs completion/abort/error when the process finishes.
        Returns the QProcess (caller must keep a reference alive, e.g. on the tab,
        and call abort_workflow_process(process) to abort it).
        """
        process = QProcess(self)
        process.setProgram(argv[0])
        process.setArguments(argv[1:])
        if cwd:
            process.setWorkingDirectory(str(cwd))
        process.setProcessChannelMode(QProcess.MergedChannels)

        def handle_output():
            data = bytes(process.readAllStandardOutput()).decode(errors="replace")
            for line in data.splitlines():
                if line:
                    self.print_log(line)

        def handle_finished(exit_code, exit_status):
            run_button.setEnabled(True)
            abort_button.setVisible(False)
            abort_button.setEnabled(False)
            if exit_status == QProcess.CrashExit:
                self.print_log("Process aborted.")
            elif exit_code != 0:
                self.print_log(f"Process exited with code {exit_code}.")
            else:
                self.print_log("Completed successfully!")

        process.readyReadStandardOutput.connect(handle_output)
        process.finished.connect(handle_finished)
        run_button.setEnabled(False)
        abort_button.setVisible(True)
        abort_button.setEnabled(True)
        process.start()
        return process

    def abort_workflow_process(self, process):
        """Abort a process started via start_workflow_process, killing its whole tree."""
        if process is None or process.state() == QProcess.NotRunning:
            return
        self._kill_process_tree(process.processId())

    @staticmethod
    def _kill_process_tree(pid):
        """Best-effort kill of a process and all its descendants (children, grandchildren)."""
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
