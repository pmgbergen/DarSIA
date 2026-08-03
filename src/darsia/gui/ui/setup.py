"""Setup workflow tab for DarSIA GUI."""

import threading
from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QPushButton, QVBoxLayout, QWidget


class SetupTab:
    """Manages the setup tab UI and workflow execution."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.setup_checkboxes = []

    def create_tab(self):
        """Create and return the setup tab widget."""
        container = QWidget()
        layout = QVBoxLayout(container)

        setup_items = [
            ("All", "all"),
            ("Depth", "depth"),
            ("Segmentation", "segmentation"),
            ("Facies", "facies"),
            ("Protocols", "protocols"),
            ("Rig", "rig"),
            ("Show plots", "show_plots"),
        ]

        self.setup_checkboxes = []
        for label, checkbox_id in setup_items:
            checkbox = QCheckBox(label)
            self.setup_checkboxes.append((checkbox_id, checkbox))
            layout.addWidget(checkbox)

        settings_button = QPushButton("Open Setup settings")
        settings_button.clicked.connect(self.on_settings_clicked)
        layout.addWidget(settings_button)

        run_button = QPushButton("Run Setup")
        run_button.clicked.connect(self.on_run_clicked)
        layout.addWidget(run_button)

        layout.addStretch()
        return container

    def on_settings_clicked(self):
        """Handle settings button click."""
        checked_ids = self.main_window.get_checked_checkbox_ids(self.setup_checkboxes)
        self.main_window.print_log(f"Setup: Checked specifics - {checked_ids}")
        self.main_window.display_settings("setup", checked_ids)

    def on_run_clicked(self):
        """Handle run button click."""
        self.run_setup()

    def run_setup(self):
        """Run setup workflow based on checked checkboxes."""
        config_file = self.main_window.config_path_label.text()
        if not config_file or config_file == "No file chosen":
            self.main_window.print_log("Please select a config file first.")
            return

        checked_ids = self.main_window.get_checked_checkbox_ids(self.setup_checkboxes)
        if not checked_ids:
            self.main_window.print_log("Please select at least one setup option.")
            return

        # Build options dictionary matching the CLI interface
        options = {
            "all": "all" in checked_ids,
            "depth": "depth" in checked_ids,
            "segmentation": "segmentation" in checked_ids,
            "facies": "facies" in checked_ids,
            "protocols": "protocols" in checked_ids,
            "rig": "rig" in checked_ids,
            "show": "show_plots" in checked_ids,
            "force": False,
        }

        self.main_window.print_log(
            """Starting setup with options: """
            f"""{[k for k, v in options.items() if v and k != "force"]}"""
        )

        # Run workflow in a separate thread to avoid blocking the GUI
        def run_workflow():
            try:
                from darsia.presets.workflows.rig import Rig
                from darsia.presets.workflows.setup.setup_depth import setup_depth_map
                from darsia.presets.workflows.setup.setup_facies import setup_facies
                from darsia.presets.workflows.setup.setup_labeling import (
                    segment_colored_image,
                )
                from darsia.presets.workflows.setup.setup_protocols import (
                    setup_imaging_protocol,
                )
                from darsia.presets.workflows.setup.setup_rig import setup_rig

                config_paths = [Path(config_file)]
                show = options["show"]

                if options["all"] or options["depth"]:
                    self.main_window.print_log("Running depth map setup...")
                    setup_depth_map(config_paths, key="depth", show=show)
                if options["all"] or options["segmentation"]:
                    self.main_window.print_log("Running segmentation setup...")
                    segment_colored_image(config_paths, show=show)
                if options["all"] or options["facies"]:
                    self.main_window.print_log("Running facies setup...")
                    setup_facies(Rig, config_paths, show=show)
                if options["all"] or options["rig"]:
                    self.main_window.print_log("Running rig setup...")
                    setup_rig(Rig, config_paths, show=show)
                if options["protocols"]:
                    self.main_window.print_log("Running protocol setup...")
                    setup_imaging_protocol(
                        config_paths, force=options["force"], show=show
                    )

                self.main_window.print_log("Setup completed successfully!")
            except Exception as e:
                self.main_window.print_log(f"Error during setup: {str(e)}")
                import traceback

                self.main_window.print_log(traceback.format_exc())

        thread = threading.Thread(target=run_workflow, daemon=True)
        thread.start()
