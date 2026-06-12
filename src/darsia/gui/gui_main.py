from PySide6.QtWidgets import QMainWindow, QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QLineEdit, QTextEdit, QSlider, QProgressBar, QComboBox, QListWidget, QRadioButton, QHBoxLayout
from PySide6.QtCore import Qt
import toml


class MainWindow(QMainWindow):
    def __init__(self, config_dict={}):
        super().__init__()
        self.setWindowTitle("Darsia")

        container = QWidget()
        self.setCentralWidget(container)

        layout = QVBoxLayout(container)

        label = QLabel("Settings:")
        layout.addWidget(label)

        settings_dict = config_dict["slider_settings"]
        settings_container = QWidget()
        h_layout = QHBoxLayout(settings_container)
        self.sliders = {}
        for key, value in settings_dict.items():
            slider_label = QLabel(key)
            slider_value_label = QLabel(str(value))
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(10)
            slider.setValue(value)
            self.sliders[key] = slider
            h_layout.addWidget(slider_label)
            h_layout.addWidget(slider)
            h_layout.addWidget(slider_value_label)

            slider.valueChanged.connect(lambda v, lbl=slider_value_label: lbl.setText(str(v)))
        layout.addWidget(settings_container)

        input_settings_dict = config_dict["input_settings"]
        input_settings_container = QWidget()
        v_layout = QVBoxLayout(input_settings_container)
        self.input_settings = {}
        for key, value in input_settings_dict.items():
            local_container = QWidget()
            h_layout = QHBoxLayout(local_container)
            input_label = QLabel(key)
            line_edit = QLineEdit()
            line_edit.setText(str(value))
            self.input_settings[key] = line_edit
            h_layout.addWidget(input_label)
            h_layout.addWidget(line_edit)
            v_layout.addWidget(local_container)
        layout.addWidget(input_settings_container)

        menu_container = QWidget()
        menu_layout_vertical = QVBoxLayout(menu_container)
        self.menu_settings = {}
        menu_settings_dict = config_dict["menu_settings"]
        options = ["Option 1", "Option 2", "Option 3"]
        for key, value in menu_settings_dict.items():
            combo_box = QComboBox()
            combo_box.addItems(options)
            self.menu_settings[key] = combo_box
            for i, option in enumerate(options):
                if option == value:
                    combo_box.setCurrentIndex(i)
            menu_layout_vertical.addWidget(QLabel(key))
            menu_layout_vertical.addWidget(combo_box)
        layout.addWidget(menu_container)

        button = QPushButton("Save Settings")
        button.clicked.connect(self.saveSettings)

        layout.addWidget(button)

    def saveSettings(self):
        setting_dict = {}
        for key, slider in self.sliders.items():
            setting_dict[key] = slider.value()

        line_edit_dict = {}
        for key, line_edit in self.input_settings.items():
            line_edit_dict[key] = int(line_edit.text())

        menu_edit_dict = {}
        for key, menu in self.menu_settings.items():
            menu_edit_dict[key] = menu.currentText()

        total_dict = {"slider_settings": setting_dict, "input_settings": line_edit_dict, "menu_settings": menu_edit_dict}
        with open("test.toml", "w") as f:
            toml.dump(total_dict, f)

if __name__ == '__main__':

    with open("test.toml", "r") as f:
        config = toml.load(f)


    app = QApplication()
    window = MainWindow(config_dict=config)
    window.show()
    app.exec()
