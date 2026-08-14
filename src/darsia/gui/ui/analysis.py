"""Analysis workflow tab for DarSIA GUI."""

import argparse
import threading
from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QPushButton, QVBoxLayout, QWidget


class AnalysisTab:
    """Manages the analysis tab UI and workflow execution."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.analysis_checkboxes = []

    def create_tab(self):
        """Create and return the analysis tab widget."""
        container = QWidget()
        layout = QVBoxLayout(container)

        analysis_items = [
            ("Fingers", "fingers"),
            ("Mass", "mass"),
            ("Segmentation", "segmentation"),
        ]

        self.analysis_checkboxes = []
        for label, checkbox_id in analysis_items:
            checkbox = QCheckBox(label)
            self.analysis_checkboxes.append((checkbox_id, checkbox))
            layout.addWidget(checkbox)

        settings_button = QPushButton("Open Analysis settings")
        settings_button.clicked.connect(self.on_settings_clicked)
        layout.addWidget(settings_button)

        run_button = QPushButton("Run Analysis")
        run_button.clicked.connect(self.on_run_clicked)
        layout.addWidget(run_button)

        layout.addStretch()
        return container

    def on_settings_clicked(self):
        """Handle settings button click."""
        checked_ids = self.main_window.get_checked_checkbox_ids(
            self.analysis_checkboxes
        )
        self.main_window.display_settings("analysis", checked_ids)

    def on_run_clicked(self):
        """Handle run button click."""
        self.run_analysis()

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
        available_options = self.get_available_options()
        options = {
            "all": "all" in checked_ids,
            "cropping": (
                "cropping" in checked_ids if "cropping" in available_options else False
            ),
            "segmentation": "segmentation" in checked_ids,
            "fingers": "fingers" in checked_ids,
            "mass": "mass" in checked_ids,
            "show": False,
        }

        self.main_window.print_log(
            f"Starting analysis with options: {[k for k, v in options.items() if v]}"
        )

        # Run workflow in a separate thread to avoid blocking the GUI
        def run_workflow():
            try:
                from darsia.presets.workflows.rig import Rig
                from darsia.presets.workflows.user_interface_analysis import (
                    run_analysis,
                )

                config_paths = [Path(config_file)]

                args = argparse.Namespace(
                    config=config_paths,
                    all=options["all"],
                    cropping=options.get("cropping", False),
                    segmentation=options["segmentation"],
                    fingers=options["fingers"],
                    mass=options["mass"],
                    volume=False,
                    thresholding=False,
                    show=options["show"],
                    info=False,
                )
                run_analysis(Rig, args)

                self.main_window.print_log("Analysis completed successfully!")
            except Exception as e:
                self.main_window.print_log(f"Error during analysis: {str(e)}")
                import traceback

                self.main_window.print_log(traceback.format_exc())

        thread = threading.Thread(target=run_workflow, daemon=True)
        thread.start()

    def get_available_options(self):
        """Get list of available analysis checkbox IDs."""
        return [checkbox_id for checkbox_id, _ in self.analysis_checkboxes]
