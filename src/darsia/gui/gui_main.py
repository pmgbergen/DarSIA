from PySide6.QtWidgets import (QScrollArea,QMainWindow, QApplication, QLabel, QWidget, QVBoxLayout, QPushButton,
                               QLineEdit, QTextEdit, QSlider, QProgressBar, QComboBox, QListWidget, QRadioButton,
                               QHBoxLayout, QTabWidget)
from PySide6.QtCore import Qt
import toml
import ast


class MainWindow(QMainWindow):
    def __init__(self, config_dict={}):
        super().__init__()
        self.setWindowTitle("Darsia")

        # Create main container that will hold settings
        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        tabs = QTabWidget()
        setup_container = QWidget()
        calibration_container = QWidget()
        analysis_container = QWidget()
        tabs.addTab(setup_container, "Setup")
        tabs.addTab(calibration_container, "Calibration")
        tabs.addTab(analysis_container, "Analysis")


        first_layout = QVBoxLayout(setup_container)
        label = QLabel("Settings:")
        first_layout.addWidget(label)
        self.subsections = {}

        for key, value in config_dict.items():
            self.subsections[key] = {}
            self.unravel_settings(key, value, [], first_layout)


        # Add scroll area for settings
        scroll_area = QScrollArea()
        scroll_area.setWidget(tabs)
        scroll_area.setWidgetResizable(True)

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
        log_scroll_area.setMaximumHeight(150)

        button = QPushButton("Save Settings")
        button.clicked.connect(self.saveSettings)

        # Create central widget with all components
        self.setCentralWidget(main_container)
        main_layout.addWidget(scroll_area)  # Settings scroll area
        main_layout.addWidget(button)     # Save button
        main_layout.addWidget(log_scroll_area)  # Logging scroll area (separate)



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


def set_dict_value(inp_dict, path, value):
    local_dict = inp_dict
    for key in path[:-1]:
        local_dict = local_dict[key]
    local_dict[path[-1]] = value

if __name__ == '__main__':

    with open("common.toml", "r") as f:
        config = toml.load(f)


    app = QApplication()
    window = MainWindow(config_dict=config)
    window.show()
    app.exec()
