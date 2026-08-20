"""Analysis workflow tab for DarSIA GUI."""

import sys
from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QPushButton, QVBoxLayout, QWidget


class AnalysisTab:
    """Manages the analysis tab UI and workflow execution."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.analysis_checkboxes = []
        self.process = None

    def create_tab(self):
        """Create and return the analysis tab widget."""
        container = QWidget()
        layout = QVBoxLayout(container)

        analysis_items = [
            ("Cropping", "cropping"),
            ("Segmentation", "segmentation"),
            ("Fingers", "fingers"),
            ("Mass", "mass"),
            ("Volume", "volume"),
            ("Thresholding", "thresholding"),
            ("All images (option)", "all"),
            ("Show plots (option)", "show"),
        ]

        self.analysis_checkboxes = []
        for label, checkbox_id in analysis_items:
            checkbox = QCheckBox(label)
            self.analysis_checkboxes.append((checkbox_id, checkbox))
            layout.addWidget(checkbox)

        settings_button = QPushButton("Open Analysis settings")
        settings_button.clicked.connect(self.on_settings_clicked)
        layout.addWidget(settings_button)

        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.on_run_clicked)
        layout.addWidget(self.run_button)

        self.abort_button = QPushButton("Abort Analysis")
        self.abort_button.setVisible(False)
        self.abort_button.setEnabled(False)
        self.abort_button.clicked.connect(self.on_abort_clicked)
        layout.addWidget(self.abort_button)

        layout.addStretch()
        return container

    def on_settings_clicked(self):
        """Handle settings button click."""
        checked_ids = self.main_window.get_checked_checkbox_ids(
            self.analysis_checkboxes
        )
        self.main_window.settings_factory.display_settings("analysis", checked_ids)

    def on_run_clicked(self):
        """Handle run button click."""
        self.run_analysis()

    def on_abort_clicked(self):
        """Handle abort button click."""
        if self.process is not None:
            self.main_window.process_runner.abort_workflow_process(self.process)

    def run_analysis(self):
        """Run analysis workflow based on checked checkboxes."""
        config_file = self.main_window.config_path_label.text()
        if not config_file or config_file == "No file chosen":
            self.main_window.print_log("Please select a config file first.")
            return

        checked_ids = self.main_window.get_checked_checkbox_ids(
            self.analysis_checkboxes
        )
        if not checked_ids:
            self.main_window.print_log("Please select at least one analysis option.")
            return

        # Build options dictionary matching the CLI interface
        options = {
            "cropping": "cropping" in checked_ids,
            "segmentation": "segmentation" in checked_ids,
            "fingers": "fingers" in checked_ids,
            "mass": "mass" in checked_ids,
            "volume": "volume" in checked_ids,
            "thresholding": "thresholding" in checked_ids,
            "all": "all" in checked_ids,
            "show": "show" in checked_ids,
        }

        self.main_window.print_log(
            f"Starting analysis with options: {[k for k, v in options.items() if v]}"
        )

        # Build command-line arguments for subprocess
        argv = [
            sys.executable,
            "-m",
            "darsia.presets.workflows.user_interface_analysis",
            "--config",
            str(Path(config_file).resolve()),
        ]
        if options["cropping"]:
            argv.append("--cropping")
        if options["segmentation"]:
            argv.append("--segmentation")
        if options["fingers"]:
            argv.append("--fingers")
        if options["mass"]:
            argv.append("--mass")
        if options["volume"]:
            argv.append("--volume")
        if options["thresholding"]:
            argv.append("--thresholding")
        if options["all"]:
            argv.append("--all")
        if options["show"]:
            argv.append("--show")

        # Launch workflow in a separate process
        self.process = self.main_window.process_runner.start_workflow_process(
            argv, self.run_button, self.abort_button, cwd=Path.cwd()
        )
