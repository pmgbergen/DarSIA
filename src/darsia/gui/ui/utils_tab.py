"""Utils workflow tab for DarSIA GUI."""

from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class UtilsTab:
    """Manages the utils tab UI and workflow execution."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.utils_checkboxes = []
        self.process = None
        self.calibration_bundle_input = None
        self.calibration_target_input = None

    def create_tab(self):
        """Create and return the utils tab widget."""
        container = QWidget()
        layout = QVBoxLayout(container)

        utils_items = [
            ("Build protocol-time media (MP4/GIF)", "build_media"),
            ("Download/cache data", "download_data"),
            ("Export calibration", "export_calibration"),
            ("Import calibration", "import_calibration"),
        ]

        self.utils_checkboxes = []
        for label, checkbox_id in utils_items:
            checkbox = QCheckBox(label)
            self.utils_checkboxes.append((checkbox_id, checkbox))
            layout.addWidget(checkbox)

        layout.addWidget(QLabel("Calibration bundle path:"))
        bundle_container = QWidget()
        bundle_layout = QHBoxLayout(bundle_container)
        bundle_layout.setContentsMargins(0, 0, 0, 0)
        self.calibration_bundle_input = QLineEdit()
        self.calibration_bundle_input.setPlaceholderText(
            "Select calibration bundle file..."
        )
        bundle_button = QPushButton("Browse")
        bundle_button.clicked.connect(self.on_browse_calibration_bundle)
        bundle_layout.addWidget(self.calibration_bundle_input)
        bundle_layout.addWidget(bundle_button)
        layout.addWidget(bundle_container)

        layout.addWidget(QLabel("Calibration target path:"))
        target_container = QWidget()
        target_layout = QHBoxLayout(target_container)
        target_layout.setContentsMargins(0, 0, 0, 0)
        self.calibration_target_input = QLineEdit()
        self.calibration_target_input.setPlaceholderText(
            "Select calibration target folder..."
        )
        target_button = QPushButton("Browse")
        target_button.clicked.connect(self.on_browse_calibration_target)
        target_layout.addWidget(self.calibration_target_input)
        target_layout.addWidget(target_button)
        layout.addWidget(target_container)

        settings_button = QPushButton("Open Utils settings")
        settings_button.clicked.connect(self.on_settings_clicked)
        layout.addWidget(settings_button)

        self.run_button = QPushButton("Run Utils")
        self.run_button.clicked.connect(self.on_run_clicked)
        layout.addWidget(self.run_button)

        self.abort_button = QPushButton("Abort Utils")
        self.abort_button.setVisible(False)
        self.abort_button.setEnabled(False)
        self.abort_button.clicked.connect(self.on_abort_clicked)
        layout.addWidget(self.abort_button)

        layout.addStretch()
        return container

    def on_browse_calibration_bundle(self):
        """Browse for calibration bundle file."""
        file, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Select Calibration Bundle",
            "",
            "ZIP Files (*.zip);;All Files (*)",
        )
        if file:
            self.calibration_bundle_input.setText(file)
            self.main_window.print_log(f"Selected calibration bundle: {file}")

    def on_browse_calibration_target(self):
        """Browse for calibration target directory."""
        folder = QFileDialog.getExistingDirectory(
            self.main_window, "Select Calibration Target Folder"
        )
        if folder:
            self.calibration_target_input.setText(folder)
            self.main_window.print_log(f"Selected calibration target: {folder}")

    def on_settings_clicked(self):
        """Handle settings button click."""
        self.main_window.print_log("Utils settings: not yet implemented")

    def on_run_clicked(self):
        """Handle run button click."""
        self.main_window.print_log("Utils run: not yet implemented")

    def on_abort_clicked(self):
        """Handle abort button click."""
        if self.process is not None:
            self.main_window.abort_workflow_process(self.process)
