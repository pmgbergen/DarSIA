"""Calibration workflow tab for DarSIA GUI."""

import sys
from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QPushButton, QVBoxLayout, QWidget


class CalibrationTab:
    """Manages the calibration tab UI and workflow execution."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.calibration_checkboxes = []
        self.process = None

    def create_tab(self):
        """Create and return the calibration tab widget."""
        container = QWidget()
        layout = QVBoxLayout(container)

        calibration_items = [
            ("Color Path", "color"),
            ("Mass", "mass"),
            ("Default mass", "default_mass"),
            ("Delete all calibrations", "delete"),
            ("Reset mass calibration (option)", "reset"),
            ("Show plots (option)", "show"),
        ]

        self.calibration_checkboxes = []
        for label, checkbox_id in calibration_items:
            checkbox = QCheckBox(label)
            self.calibration_checkboxes.append((checkbox_id, checkbox))
            layout.addWidget(checkbox)

        settings_button = QPushButton("Open Calibration settings")
        settings_button.clicked.connect(self.on_settings_clicked)
        layout.addWidget(settings_button)

        self.run_button = QPushButton("Run Calibration")
        self.run_button.clicked.connect(self.on_run_clicked)
        layout.addWidget(self.run_button)

        self.abort_button = QPushButton("Abort Calibration")
        self.abort_button.setVisible(False)
        self.abort_button.setEnabled(False)
        self.abort_button.clicked.connect(self.on_abort_clicked)
        layout.addWidget(self.abort_button)

        layout.addStretch()
        return container

    def on_settings_clicked(self):
        """Handle settings button click."""
        checked_ids = self.main_window.get_checked_checkbox_ids(
            self.calibration_checkboxes
        )
        self.main_window.display_settings("calibration", checked_ids)

    def on_run_clicked(self):
        """Handle run button click."""
        self.run_calibration()

    def on_abort_clicked(self):
        """Handle abort button click."""
        if self.process is not None:
            self.main_window.abort_workflow_process(self.process)

    def run_calibration(self):
        """Run calibration workflow based on checked checkboxes."""
        config_file = self.main_window.config_path_label.text()
        if not config_file or config_file == "No file chosen":
            self.main_window.print_log("Please select a config file first.")
            return

        checked_ids = self.main_window.get_checked_checkbox_ids(
            self.calibration_checkboxes
        )
        if not checked_ids:
            self.main_window.print_log("Please select at least one calibration option.")
            return

        # Build options dictionary matching the CLI interface
        options = {
            "color": "color" in checked_ids,
            "mass": "mass" in checked_ids,
            "default_mass": "default_mass" in checked_ids,
            "delete": "delete" in checked_ids,
            "reset": "reset" in checked_ids,
            "show": "show" in checked_ids,
        }

        self.main_window.print_log(
            f"Starting calibration with options: {[k for k, v in options.items() if v]}"
        )

        # Build command-line arguments for subprocess
        argv = [
            sys.executable,
            "-m",
            "darsia.presets.workflows.user_interface_calibration",
            "--config",
            str(Path(config_file).resolve()),
        ]
        if options["color"]:
            argv.append("--color-embedding")
        if options["mass"]:
            argv.append("--mass")
        if options["default_mass"]:
            argv.append("--default-mass")
        if options["delete"]:
            argv.append("--delete")
        if options["reset"]:
            argv.append("--reset")
        if options["show"]:
            argv.append("--show")

        # Launch workflow in a separate process
        self.process = self.main_window.start_workflow_process(
            argv, self.run_button, self.abort_button, cwd=Path.cwd()
        )
