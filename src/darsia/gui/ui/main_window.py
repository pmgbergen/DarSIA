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
from .menu import MenuBuilder
from .recent_files import add_recent_config, remove_recent_config
from .settings import SettingsFactory
from .setup import SetupTab
from .toolbar import ToolbarBuilder


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

        # Add tabs
        tabs.addTab(self.setup_tab.create_tab(), "Setup")
        tabs.addTab(self.calibration_tab.create_tab(), "Calibration")
        tabs.addTab(self.analysis_tab.create_tab(), "Analysis")

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

    def save_settings(self):
        """Save the current settings to the loaded config file."""
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
            # Skip path_map, int_group_list, and int_list_map dicts (handled below)
            if isinstance(value, dict) and (
                "path_map" in value
                or "int_group_list" in value
                or "int_list_map" in value
            ):
                continue
            # Skip sub-inputs of unchecked groups
            if key in skip_keys:
                continue

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

        # Sixth pass: write all active lists
        for active_list_key, names in group_active_names.items():
            self.settings_factory.set_value(
                self.config_dict, active_list_key, sorted(names)
            )

        if self.config_file != "":
            with open(self.config_file, "w") as f:
                toml.dump(self.config_dict, f)
            self.print_log(f"Settings saved to {self.config_file}")
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

    def display_settings(self, action, checked_ids):
        """Method that displays the relevant settings based on the action being used."""
        while self.settings_layout.count():
            child = self.settings_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        relevant_settings = self.settings_factory.get_relevant_settings(
            action, checked_ids
        )
        self.settings_inputs = {}

        # If no sections, add a message and return
        if not relevant_settings:
            self.settings_layout.addWidget(QLabel("No settings available"))
            self.settings_layout.addStretch()
            return

        # Create a tabbed interface with one tab per section
        tabs = QTabWidget()

        # Iterate through sections in order (dict preserves insertion order in Python 3.7+)
        for section, settings_list in relevant_settings.items():
            # Create a scroll area and container for this section's fields
            tab_container = QWidget()
            tab_form = QFormLayout(tab_container)
            tab_form.setContentsMargins(0, 0, 0, 0)

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
