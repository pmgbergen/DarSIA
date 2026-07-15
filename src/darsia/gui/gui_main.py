from PySide6.QtWidgets import (QScrollArea, QMainWindow, QApplication, QLabel, QWidget, QVBoxLayout, QPushButton,
                               QLineEdit, QTextEdit, QSlider, QProgressBar, QComboBox, QListWidget, QRadioButton,
                               QHBoxLayout, QTabWidget, QSplitter, QCheckBox, QFileDialog)
from PySide6.QtCore import Qt
import toml
import ast
import os


class MainWindow(QMainWindow):
    def __init__(self, config_dict={}):
        super().__init__()
        self.setWindowTitle("Darsia")
        self.showMaximized()

        # Setting up the three upper layouts
        upper_left_container = QWidget()
        upper_left_layout = QVBoxLayout(upper_left_container)

        upper_mid_container = QWidget()
        upper_mid_layout = QVBoxLayout(upper_mid_container)

        upper_right_container = QWidget()
        upper_right_layout = QVBoxLayout(upper_right_container)

        # Setting up the left upper layout
        upper_left_layout.addWidget(QLabel("Choose experiment files"))
        
        # Initialize file chooser storage
        self.chosen_files = {}
        self.baseline_images = []
        self.baseline_container = QWidget()
        self.baseline_layout = QVBoxLayout(self.baseline_container)
        self.baseline_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add config file chooser first
        self.create_file_chooser(upper_left_layout, "Config File", "TOML Files (*.toml);;All Files (*)", False)
        upper_left_layout.addSpacing(10)
        
        # Define file chooser items: (display_name, file_type, is_directory)
        # Note: Baseline Image is handled separately for multiple selections
        file_chooser_items = [
            ("Segmented Image", "Image Files (*.jpg *.jpeg *.png);;All Files (*)", False),
            ("Depth Measurements", "CSV Files (*.csv);;All Files (*)", False),
            ("Facies Properties", "CSV Files (*.csv);;All Files (*)", False),
            ("Results Folder", "", True),
            ("Data Images Folder", "", True),
        ]
        
        # Create baseline images section (with add/remove capability)
        baseline_title = QLabel("Baseline Image(s)")
        baseline_title.setStyleSheet("font-weight: bold;")
        upper_left_layout.addWidget(baseline_title)
        upper_left_layout.addWidget(self.baseline_container)
        
        # Add initial baseline image
        self.add_baseline_image()
        
        # Add baseline image button
        add_baseline_button = QPushButton("+ Add Baseline Image")
        add_baseline_button.clicked.connect(self.add_baseline_image)
        upper_left_layout.addWidget(add_baseline_button)
        
        upper_left_layout.addSpacing(10)
        
        # Create file chooser UI for each item
        for display_name, file_filter, is_directory in file_chooser_items:
            self.create_file_chooser(upper_left_layout, display_name, file_filter, is_directory)
        
        # Add stretch to push file choosers to top
        upper_left_layout.addStretch()


        # Setting up the middle upper layout
        # Creating tabs
        tabs = QTabWidget()
        setup_container = QWidget()
        setup_layout = QVBoxLayout(setup_container)
        calibration_container = QWidget()
        calibration_layout = QVBoxLayout(calibration_container)
        analysis_container = QWidget()
        analysis_layout = QVBoxLayout(analysis_container)
        tabs.addTab(setup_container, "Setup")
        tabs.addTab(calibration_container, "Calibration")
        tabs.addTab(analysis_container, "Analysis")
        upper_mid_layout.addWidget(tabs)

        # Initialize checkbox storage with IDs for settings mapping
        self.action_checkboxes = {
            "setup": [],
            "calibration": [],
            "analysis": []
        }

        # Creating the setup-tab
        setup_items = [
            ("All", "all"),
            ("Depth", "depth"),
            ("Segmentation", "segmentation"),
            ("Facies", "facies"),
            ("Protocol", "protocol"),
            ("Rig", "rig"),
            ("Show plots", "show_plots")
        ]
        
        self.setup_checkboxes = []
        for label, checkbox_id in setup_items:
            checkbox = QCheckBox(label)
            self.setup_checkboxes.append((checkbox_id, checkbox))
            setup_layout.addWidget(checkbox)

        setup_button = QPushButton("Setup")
        setup_button.clicked.connect(self.setup)
        setup_layout.addWidget(setup_button)
        setup_layout.addStretch()

        # Creating the calibration-tab
        calibration_items = [
            ("Color Path", "color"),
            ("Mass", "mass")
        ]
        
        self.calibration_checkboxes = []
        for label, checkbox_id in calibration_items:
            checkbox = QCheckBox(label)
            self.calibration_checkboxes.append((checkbox_id, checkbox))
            calibration_layout.addWidget(checkbox)
        
        calibration_button = QPushButton("Calibrate")
        calibration_button.clicked.connect(self.calibrate)
        calibration_layout.addWidget(calibration_button)
        calibration_layout.addStretch()

        # Creating the analysis-tab
        analysis_items = [
            ("Fingers", "fingers"),
            ("Mass", "mass"),
            ("Segmentation", "segmentation")
        ]
        
        self.analysis_checkboxes = []
        for label, checkbox_id in analysis_items:
            checkbox = QCheckBox(label)
            self.analysis_checkboxes.append((checkbox_id, checkbox))
            analysis_layout.addWidget(checkbox)
        
        analysis_button = QPushButton("Analysis")
        analysis_button.clicked.connect(self.analysis)
        analysis_layout.addWidget(analysis_button)
        analysis_layout.addStretch()


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
        self.config_dict = config_dict
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
        upper_splitter.addWidget(upper_left_container)
        upper_splitter.addWidget(upper_mid_container)
        upper_splitter.addWidget(upper_right_container)

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
        main_layout.addWidget(content_splitter)

    def load_settings_mapping(self):
        """Load the settings_mapping.toml file."""
        mapping_file = os.path.join(os.path.dirname(__file__), "settings_mapping.toml")
        if os.path.exists(mapping_file):
            return toml.load(mapping_file)
        return {}


    def save_settings(self):
        for key, value in self.settings_inputs.items():
            try:
                self.set_value(self.config_dict, key, ast.literal_eval(value.text()))
            except (ValueError, SyntaxError):
                self.set_value(self.config_dict, key, value.text())
        with open("test.toml", "w") as f:
            toml.dump(self.config_dict, f)
        self.print_log("Settings saved")


    def create_file_chooser(self, parent_layout, display_name, file_filter, is_directory):
        """Create a file/folder chooser UI element (button + path label)."""
        chooser_container = QWidget()
        chooser_layout = QHBoxLayout(chooser_container)
        chooser_layout.setContentsMargins(0, 5, 0, 5)
        
        # Browse button
        browse_button = QPushButton(f"Browse {display_name}")
        browse_button.setMinimumWidth(200)
        
        # Path label to display selected path
        path_label = QLabel("No file chosen")
        path_label.setStyleSheet("color: white;")
        
        # Store label reference for updating
        key = display_name.lower().replace(" ", "_")
        self.chosen_files[key] = {"path": "", "label": path_label, "is_directory": is_directory, "filter": file_filter}
        
        # Connect button to file dialog
        browse_button.clicked.connect(lambda: self.browse_file(key))
        
        chooser_layout.addWidget(browse_button)
        chooser_layout.addWidget(path_label)
        chooser_layout.addStretch()
        
        parent_layout.addWidget(chooser_container)

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
            "container": chooser_container
        }
        self.baseline_images.append(baseline_info)
        
        # Connect buttons
        browse_button.clicked.connect(lambda: self.browse_baseline_image(baseline_index))
        remove_button.clicked.connect(lambda: self.remove_baseline_image(baseline_index))
        
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
        selected_path, _ = QFileDialog.getOpenFileName(self, f"Select Baseline Image {index + 1}", "", file_filter)
        
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
            selected_path = QFileDialog.getExistingDirectory(self, f"Select {key.replace('_', ' ')}")
        else:
            selected_path, _ = QFileDialog.getOpenFileName(self, f"Select {key.replace('_', ' ')}", "", file_filter)
        
        if selected_path:
            self.chosen_files[key]["path"] = selected_path
            file_info["label"].setText(selected_path)
            file_info["label"].setStyleSheet("color: white;")
            self.print_log(f"Selected {key}: {selected_path}")

    def saveSettings(self):
        total_dict = {}
        """
        for key, value in self.subsections.items():
            local_dict = {}
            for loc_key, line_edit in value.items():
                local_dict[loc_key] = int(line_edit.text())
            total_dict[key] = local_dict"""
        for key, value in self.subsections.items():
            #total_dict[key] = {}
            self.get_save_dict(key, value, [], total_dict)


        with open("test.toml", "w") as f:
            toml.dump(total_dict, f)
        self.print_log("Settings saved")

    def get_checked_checkbox_ids(self, checkboxes):
        checked_ids = []
        for checkbox_id, checkbox in checkboxes:
            if checkbox.isChecked():
                checked_ids.append(checkbox_id)
        return checked_ids

    def get_relevant_settings(self, action, checked_ids):
        settings = []
        for checked_id in checked_ids:
            try:
                settings += self.settings_mapping['settings'][action][checked_id]["content"]
            except KeyError:
                self.print_log(f"No settings found for {action} with id {checked_id}")
        return settings


    def display_settings(self, action, checked_ids):
        # Clearing the settings panel
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
            # Add each setting to the layout
            value = self.get_value(self.config_dict, setting)
            setting_container = QWidget()
            setting_layout = QHBoxLayout(setting_container)
            setting_label = QLabel(setting)
            setting_edit = QLineEdit()
            if value is not None:
                setting_edit.setText(str(value))
            setting_layout.addWidget(setting_label)
            setting_layout.addWidget(setting_edit)
            self.settings_layout.addWidget(setting_container)

            self.settings_inputs[setting] = setting_edit

        self.settings_layout.addStretch()


    def setup(self):
        """Setup action: get checked specifics and display relevant settings."""
        checked_ids = self.get_checked_checkbox_ids(self.setup_checkboxes)
        self.print_log(f"Setup: Checked specifics - {checked_ids}")
        self.print_log(f"Settings mapping loaded: {bool(self.settings_mapping)}")
        self.display_settings("setup", checked_ids)

    def calibrate(self):
        """Calibration action: get checked specifics and display relevant settings."""
        checked_ids = self.get_checked_checkbox_ids(self.calibration_checkboxes)
        self.display_settings("calibration", checked_ids)
        self.print_log(f"Calibration: Checked specifics - {checked_ids}")

    def analysis(self):
        """Analysis action: get checked specifics and display relevant settings."""
        checked_ids = self.get_checked_checkbox_ids(self.analysis_checkboxes)
        self.display_settings("analysis", checked_ids)
        self.print_log(f"Analysis: Checked specifics - {checked_ids}")


    def get_save_dict(self, inp_key, inp_val, path, total_dict):
        if type(inp_val) == dict:
            local_dict = total_dict
            for key in path:
                local_dict = local_dict[key]
            local_dict[inp_key] = {}
            for key, value in inp_val.items():
                self.get_save_dict(key, value, path + [inp_key], total_dict)
        else:
            try:
                set_dict_value(total_dict, path+[inp_key], ast.literal_eval(inp_val.text()))
            except (ValueError, SyntaxError):
                set_dict_value(total_dict, path+[inp_key], inp_val.text())




    def unravel_settings(self, inp_key, inp_val, path, parent_layout):
        if type(inp_val) == dict:
            local_container = QWidget()
            local_layout = QVBoxLayout(local_container)
            label = QLabel(inp_key)
            local_layout.addWidget(label)
            parent_layout.addWidget(local_container)
            loc_dict = self.subsections
            for key in path:
                loc_dict = loc_dict[key]
            loc_dict[inp_key] = {}

            for key, value in inp_val.items():
                self.unravel_settings(key, value, path + [inp_key], local_layout)
        else:
            setting_container = QWidget()
            setting_layout = QHBoxLayout(setting_container)
            input_label = QLabel(inp_key)
            line_edit = QLineEdit()
            line_edit.setText(str(inp_val))

            loc_dict = self.subsections
            for key in path:
                loc_dict = loc_dict[key]
            loc_dict[inp_key] = line_edit
            setting_layout.addWidget(input_label)
            setting_layout.addWidget(line_edit)
            parent_layout.addWidget(setting_container)

    def print_log(self, text):
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


def set_dict_value(inp_dict, path, value):
    local_dict = inp_dict
    for key in path[:-1]:
        local_dict = local_dict[key]
    local_dict[path[-1]] = value

if __name__ == '__main__':

    with open("test.toml", "r") as f:
        config = toml.load(f)


    app = QApplication()
    window = MainWindow(config_dict=config)
    window.show()
    app.exec()
