"""Calibration workflow tab for DarSIA GUI."""

import threading
from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QPushButton, QVBoxLayout, QWidget


class CalibrationTab:
    """Manages the calibration tab UI and workflow execution."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.calibration_checkboxes = []

    def create_tab(self):
        """Create and return the calibration tab widget."""
        container = QWidget()
        layout = QVBoxLayout(container)

        calibration_items = [("Color Path", "color"), ("Mass", "mass")]

        self.calibration_checkboxes = []
        for label, checkbox_id in calibration_items:
            checkbox = QCheckBox(label)
            self.calibration_checkboxes.append((checkbox_id, checkbox))
            layout.addWidget(checkbox)

        settings_button = QPushButton("Open Calibration settings")
        settings_button.clicked.connect(self.on_settings_clicked)
        layout.addWidget(settings_button)

        run_button = QPushButton("Run Calibration")
        run_button.clicked.connect(self.on_run_clicked)
        layout.addWidget(run_button)

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
        }

        self.main_window.print_log(
            f"Starting calibration with options: {[k for k, v in options.items() if v]}"
        )

        # Run workflow in a separate thread to avoid blocking the GUI
        def run_workflow():
            try:
                from darsia.presets.workflows.analysis.analysis_context import (
                    prepare_analysis_context,
                )
                from darsia.presets.workflows.calibration import (
                    calibration_color_to_mass_analysis as c2m_analysis_module,
                )
                from darsia.presets.workflows.calibration.calibration_color_paths import (
                    calibration_color_paths_from_context,
                )
                from darsia.presets.workflows.rig import Rig

                config_paths = [Path(config_file)]

                # Prepare shared context once for all analyses
                ctx = prepare_analysis_context(
                    cls=Rig,
                    path=config_paths,
                    all=False,
                    require_color_to_mass=False,
                    section="calibration",
                    require_results=False,
                )

                if options["color"]:
                    self.main_window.print_log("Running color embedding calibration...")
                    calibration_color_paths_from_context(ctx, False)
                if options["mass"]:
                    self.main_window.print_log("Running mass calibration...")
                    c2m_analysis_module.calibration_color_to_mass_analysis_from_context(
                        ctx,
                        reset=False,
                        show=False,
                        default=False,
                    )

                self.main_window.print_log("Calibration completed successfully!")
            except Exception as e:
                self.main_window.print_log(f"Error during calibration: {str(e)}")
                import traceback

                self.main_window.print_log(traceback.format_exc())

        thread = threading.Thread(target=run_workflow, daemon=True)
        thread.start()
