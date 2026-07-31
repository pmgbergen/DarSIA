import ast
from pathlib import Path

import toml
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
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

from .analysis import AnalysisTab
from .calibration import CalibrationTab
from .help import HelpButton
from .setup import SetupTab


class MainWindow(QMainWindow):
    """The main class containing the window and the relevant methods for the visualization."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DarSIA")

        # Set window icon
        logo_path = (
            Path(__file__).parent
            / "../../presets/workflows/interface/DarSIA_Horisontal_Positiv_part.png"
        )
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))

        # Setting up the three upper layouts
        upper_container = QWidget()
        upper_layout = QVBoxLayout(upper_container)
        upper_container.setFixedHeight(100)

        upper_mid_container = QWidget()
        upper_mid_layout = QVBoxLayout(upper_mid_container)

        upper_right_container = QWidget()
        upper_right_layout = QVBoxLayout(upper_right_container)

        # Setting up the left upper layout
        upper_layout.addWidget(QLabel("Choose experiment files"))

        # Initialize file chooser storage
        self.chosen_files = {}
        self.baseline_images = []
        self.baseline_container = QWidget()
        self.baseline_layout = QVBoxLayout(self.baseline_container)
        self.baseline_layout.setContentsMargins(0, 0, 0, 0)

        # Add config file chooser first
        config_file_chooser_container, self.config_path_label = (
            self.create_file_chooser(
                "Config File", "TOML Files (*.toml);;All Files (*)", False
            )
        )
        upper_layout.addWidget(config_file_chooser_container)
        load_config_button = QPushButton("Load Config")
        load_config_button.clicked.connect(self.load_config)
        upper_layout.addWidget(load_config_button)

        # Add stretch to push file choosers to top
        upper_layout.addStretch()

        # Setting up the middle upper layout with tabs
        tabs = QTabWidget()

        # Initialize tab managers
        self.setup_tab = SetupTab(self)
        self.calibration_tab = CalibrationTab(self)
        self.analysis_tab = AnalysisTab(self)

        # Add tabs
        tabs.addTab(self.setup_tab.create_tab(), "Setup")
        tabs.addTab(self.calibration_tab.create_tab(), "Calibration")
        tabs.addTab(self.analysis_tab.create_tab(), "Analysis")

        upper_mid_layout.addWidget(tabs)

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

        # Save Settings button
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        upper_right_layout.addWidget(save_button)

        # Store config and load settings mapping
        self.config_file = ""
        self.config_dict = {}
        self.settings_mapping = self.load_settings_mapping()
        self.settings_inputs = {}  # Store setting input widgets

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

    def load_settings_mapping(self):
        """Load the settings_mapping.toml file."""
        mapping_file = Path(__file__).parent / "../config" / "settings.toml"
        if mapping_file.exists():
            return toml.load(mapping_file)
        return {}

    def save_settings(self):
        """Method to save the settings set in the gui to the settings.toml file."""
        for key, value in self.settings_inputs.items():
            try:
                if isinstance(value, QLineEdit):
                    self.set_value(
                        self.config_dict, key, ast.literal_eval(value.text())
                    )
                elif isinstance(value, QComboBox):
                    self.set_value(self.config_dict, key, value.currentText())
                elif isinstance(value, QCheckBox):
                    self.set_value(self.config_dict, key, value.isChecked())
                elif isinstance(value, list):
                    if len(value) > 0:
                        if isinstance(value[0], QCheckBox):
                            self.set_value(
                                self.config_dict,
                                key,
                                [item.text() for item in value if item.isChecked()],
                            )
                        elif isinstance(value[0], QLineEdit):
                            self.set_value(
                                self.config_dict,
                                key,
                                [item.text() for item in value if item.text().strip()],
                            )
            except (ValueError, SyntaxError):
                if hasattr(value, "text"):
                    self.set_value(self.config_dict, key, value.text())
        if self.config_file != "":
            with open(self.config_file, "w") as f:
                toml.dump(self.config_dict, f)
            self.print_log(f"Settings saved to {self.config_file}")
        else:
            self.print_log("Settings not saved, please choose a config file")

    def create_file_chooser(self, display_name, file_filter, is_directory):
        """Create a file/folder chooser UI element (button + path label)."""
        if not file_filter:
            file_filter = "All Files (*)"

        chooser_container = QWidget()
        chooser_layout = QHBoxLayout(chooser_container)
        chooser_layout.setContentsMargins(0, 5, 0, 5)

        # Browse button
        browse_button = QPushButton(f"Browse {display_name}")
        browse_button.setMinimumWidth(200)

        # Path label to display selected path
        path_label = QLineEdit("No file chosen")
        path_label.setStyleSheet("color: white;")

        # Store label reference for updating
        key = display_name.lower().replace(" ", "_")
        self.chosen_files[key] = {
            "path": "",
            "label": path_label,
            "is_directory": is_directory,
            "filter": file_filter,
        }

        # Connect button to file dialog
        browse_button.clicked.connect(lambda: self.browse_file(key))

        chooser_layout.addWidget(browse_button)
        chooser_layout.addWidget(path_label)
        chooser_layout.addStretch()
        return chooser_container, path_label
        # parent_layout.addWidget(chooser_container)

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

    def get_relevant_settings(self, action, checked_ids):
        """
        Method that gets the relevant settings based on the action being used,
        and the checkboxes that are checked.

        Parameters
        ----------
        action : str
        checked_ids : list

        Returns
        -------
        list
            List of lists, where each list contains both the name and the type of the setting.
        """
        settings = []
        for checked_id in checked_ids:
            try:
                settings += self.settings_mapping["settings"][action][checked_id][
                    "content"
                ]
            except KeyError:
                self.print_log(f"No settings found for {action} with id {checked_id}")
        return settings

    def wrap_setting_with_help(self, setting_container, setting_dict):
        """Wrap a setting container with a dedicated help button column."""
        help_text = setting_dict.get("help")
        link_url = setting_dict.get("link")

        wrapper = QWidget()
        wrapper_layout = QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(8)

        # Left: setting container with stretch
        wrapper_layout.addWidget(setting_container, stretch=1)

        # Right: fixed-width column for help button (or empty space)
        right_column = QWidget()
        right_layout = QHBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)

        if help_text:
            help_button = HelpButton(help_text, link_url)
            right_layout.addWidget(help_button)
        else:
            right_layout.addStretch()

        right_column.setFixedWidth(40)
        wrapper_layout.addWidget(right_column)

        return wrapper

    def display_settings(self, action, checked_ids):
        """Method that displays the relevant settings based on the action being used."""
        while self.settings_layout.count():
            child = self.settings_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add a label describing what settings is being shown
        label = QLabel(f"{action.capitalize()}: Checked specifics - {checked_ids}")
        self.settings_layout.addWidget(label)

        relevant_settings = self.get_relevant_settings(action, checked_ids)
        self.settings_inputs = {}

        for setting in relevant_settings:
            setting_container, setting_edit = self.create_setting_edit(setting)
            wrapped_container = self.wrap_setting_with_help(setting_container, setting)

            self.settings_layout.addWidget(wrapped_container)
            self.settings_inputs[setting["key"]] = setting_edit

        self.settings_layout.addStretch()

    def load_config(self):
        """Method that loads the config file chosen in the GUI."""
        # file = self.chosen_files["config_file"]["path"]
        file = self.config_path_label.text()
        if not file:
            self.print_log("No config file selected.")
            return
        try:
            with open(file, "r") as f:
                self.config_dict = toml.load(f)
        except Exception as e:
            self.print_log(f"Error loading config file: {e}")
            return
        self.config_file = file
        self.print_log("Config loaded")

    def print_log(self, text):
        """Method that prints to the console and to the log window."""
        self.log_text.append(text)
        print(text)

    def get_value(self, dictionary, key_path):
        """Get a value from nested dict using dot notation (e.g., 'a.b.c')"""
        keys = key_path.split(".")
        value = dictionary
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def set_value(self, dictionary, key_path, value):
        """Set a value in nested dict using dot notation, creating keys as needed"""
        keys = key_path.split(".")
        current = dictionary
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def create_setting_edit(self, setting_dict):
        """Create a new setting edit based on the setting type."""
        setting_type = setting_dict["type"]

        if setting_type == "int":
            return self.create_simple_input(setting_dict)
        elif setting_type == "float":
            return self.create_simple_input(setting_dict)
        elif setting_type == "string":
            return self.create_string_input(setting_dict)
        elif setting_type == "bool":
            return self.create_bool_input(setting_dict)
        elif setting_type == "fixed_list" and setting_dict["list_type"] == "string":
            return self.create_fixed_list_string_input(setting_dict)
        elif setting_type == "file":
            return self.create_file_chooser(setting_dict["key"], None, False)
        elif setting_type == "folder":
            return self.create_file_chooser(setting_dict["key"], None, True)
        elif setting_type == "multi_file":
            return self.create_multi_file_input(setting_dict)

        else:
            self.print_log(
                f"Setting type {setting_type} not supported yet, using simple input"
            )
            return self.create_simple_input(setting_dict)

    def create_simple_input(self, setting_dict):
        """Creates a new input as a line edit."""
        setting = setting_dict["key"]

        value = self.get_value(self.config_dict, setting)
        setting_container = QWidget()
        setting_layout = QHBoxLayout(setting_container)
        setting_label = QLabel(setting)
        setting_edit = QLineEdit()
        if value is not None:
            setting_edit.setText(str(value))
        if setting_dict["type"] == "list":
            type_label = QLabel(
                f"({setting_dict['type']}, {setting_dict['list_type']})"
            )
        else:
            type_label = QLabel(f"({setting_dict['type']})")
        setting_layout.addWidget(setting_label)
        setting_layout.addWidget(setting_edit)
        setting_layout.addWidget(type_label)
        return setting_container, setting_edit

    def create_bool_input(self, setting_dict):
        """Creates a new input as a checkbox."""
        setting = setting_dict["key"]
        value = self.get_value(self.config_dict, setting)
        setting_container = QWidget()
        setting_layout = QHBoxLayout(setting_container)
        setting_label = QLabel(setting)
        setting_checkbox = QCheckBox()
        if value is not None:
            setting_checkbox.setChecked(bool(value))
        setting_layout.addWidget(setting_label)
        setting_layout.addWidget(setting_checkbox)
        setting_layout.addWidget(QLabel("(bool)"))
        return setting_container, setting_checkbox

    def create_string_input(self, setting_dict):
        """Creates a new input as a combobox with predefined options."""
        setting = setting_dict["key"]
        value = self.get_value(self.config_dict, setting)
        setting_container = QWidget()
        setting_layout = QHBoxLayout(setting_container)
        setting_label = QLabel(setting)
        options = setting_dict["options"]
        setting_combo = QComboBox()
        setting_combo.addItems([str(option) for option in options])

        if value is not None:
            value = str(value)
            index = setting_combo.findText(value)
            if index >= 0:
                setting_combo.setCurrentIndex(index)

        setting_layout.addWidget(setting_label)
        setting_layout.addWidget(setting_combo)

        return setting_container, setting_combo

    def create_fixed_list_string_input(self, setting_dict):
        """Creates a new input as checkboxes with predefined options."""
        setting = setting_dict["key"]
        values = self.get_value(self.config_dict, setting)
        setting_container = QWidget()
        setting_layout = QHBoxLayout(setting_container)
        setting_label = QLabel(setting)
        setting_layout.addWidget(setting_label)
        options = setting_dict["options"]
        check_boxes = []
        for option in options:
            check_box = QCheckBox(option)
            check_boxes.append(check_box)
            if values is not None:
                if option in values:
                    check_box.setChecked(True)
            setting_layout.addWidget(check_box)
        return setting_container, check_boxes

    def create_multi_file_input(self, setting_dict):
        """Creates a new input as a variable number of files."""
        setting = setting_dict["key"]
        values = self.get_value(self.config_dict, setting)

        setting_container = QWidget()
        setting_layout = QVBoxLayout(setting_container)
        setting_layout.setContentsMargins(0, 0, 0, 0)

        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        setting_label = QLabel(setting)
        add_button = QPushButton("Add file")
        header_layout.addWidget(setting_label)
        header_layout.addStretch()
        header_layout.addWidget(add_button)
        setting_layout.addWidget(header_container)

        rows_container = QWidget()
        rows_layout = QVBoxLayout(rows_container)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        setting_layout.addWidget(rows_container)

        file_edits = []
        file_rows = []

        def refresh_remove_buttons():
            show_remove = len(file_rows) > 1
            for row in file_rows:
                row["remove_button"].setVisible(show_remove)

        def add_row(initial_value=""):
            row_container = QWidget()
            row_layout = QHBoxLayout(row_container)
            row_layout.setContentsMargins(0, 0, 0, 0)

            browse_button = QPushButton("Browse")
            browse_button.setMinimumWidth(100)
            path_edit = QLineEdit()
            path_edit.setPlaceholderText("Select a file or type a path")
            if initial_value:
                path_edit.setText(str(initial_value))
            remove_button = QPushButton("Remove")
            remove_button.setMaximumWidth(80)

            def browse():
                selected_path, _ = QFileDialog.getOpenFileName(
                    self,
                    f"Select file for {setting}",
                    path_edit.text() if path_edit.text() else "",
                    "All Files (*)",
                )
                if selected_path:
                    path_edit.setText(selected_path)

            def remove():
                row_container.deleteLater()
                if row_data in file_rows:
                    file_rows.remove(row_data)
                if path_edit in file_edits:
                    file_edits.remove(path_edit)
                refresh_remove_buttons()

            browse_button.clicked.connect(browse)
            remove_button.clicked.connect(remove)

            row_layout.addWidget(browse_button)
            row_layout.addWidget(path_edit)
            row_layout.addWidget(remove_button)

            rows_layout.addWidget(row_container)

            row_data = {
                "container": row_container,
                "remove_button": remove_button,
            }
            file_rows.append(row_data)
            file_edits.append(path_edit)
            refresh_remove_buttons()

        add_button.clicked.connect(lambda: add_row())

        if isinstance(values, list) and values:
            for value in values:
                add_row(value)
        else:
            add_row("")

        return setting_container, file_edits


def set_dict_value(inp_dict, path, value):
    """Function that sets the value of a dict where path is a list of keys."""
    local_dict = inp_dict
    for key in path[:-1]:
        local_dict = local_dict[key]
    local_dict[path[-1]] = value
