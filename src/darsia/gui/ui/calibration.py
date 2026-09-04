"""Calibration workflow tab for DarSIA GUI."""

import sys
from pathlib import Path


class CalibrationTab:
    """Manages the calibration tab UI and workflow execution."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.process = None

    def on_run_clicked(self):
        """Handle run button click."""
        self.run_calibration()

    def on_abort_clicked(self):
        """Handle abort button click."""
        if self.process is not None:
            self.main_window.process_runner.abort_workflow_process(self.process)

    def run_calibration(self):
        """Run calibration workflow based on selected sidebar item."""
        config_file = self.main_window.config_path_label.text()
        if not config_file or config_file == "No file chosen":
            self.main_window.print_log("Please select a config file first.")
            return

        selected_id = self.main_window.selected_checkbox_id
        if not selected_id:
            self.main_window.print_log("Please select an option in the sidebar.")
            return

        # Sync GUI widgets to config_dict before reading live values
        self.main_window.settings_factory._sync_settings_inputs_to_config_dict()
        show_plots = bool(
            self.main_window.settings_factory.get_value(
                self.main_window.config_dict, "options.calibration.show_plots"
            )
        )

        # Build options dictionary matching the CLI interface
        options = {
            "color": selected_id == "color",
            "mass": selected_id == "mass",
            "default_mass": selected_id == "default_mass",
            "delete": selected_id == "delete",
            "reset": selected_id == "reset",
            "show": show_plots,
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

        # Map sidebar ids to the human labels the results-folder inference
        # (suggested_workflow_results_folder) matches against.
        action_label = {
            "color": "color embedding",
            "mass": "mass",
            "default_mass": "default mass",
            "delete": "delete",
            "reset": "reset",
        }.get(selected_id, selected_id)

        # Launch workflow in a separate process
        play_action = self.main_window.toolbar_builder.play_action
        stop_action = self.main_window.toolbar_builder.stop_action
        self.process = self.main_window.process_runner.start_workflow_process(
            argv,
            play_action,
            stop_action,
            cwd=Path.cwd(),
            workflow="calibration",
            actions=[action_label],
            config_paths=[Path(config_file)],
        )

    def sidebar_items(self):
        """Return sidebar data structure for Calibration category."""
        from .help_text import get_help_text

        return [
            (
                "Actions",
                [
                    (
                        "Color Path",
                        "color",
                        "fa5s.circle",
                        get_help_text("calibration", "color", "Color Path"),
                    ),
                    (
                        "Mass",
                        "mass",
                        "fa5s.circle",
                        get_help_text("calibration", "mass", "Mass"),
                    ),
                    (
                        "Default mass",
                        "default_mass",
                        "fa5s.circle",
                        get_help_text("calibration", "default_mass", "Default mass"),
                    ),
                ],
            ),
            (
                "Danger zone",
                [
                    (
                        "Reset mass calibration",
                        "reset",
                        "fa5s.circle",
                        get_help_text("calibration", "reset", "Reset mass calibration"),
                    ),
                    (
                        "Delete all calibrations",
                        "delete",
                        "fa5s.circle",
                        get_help_text(
                            "calibration", "delete", "Delete all calibrations"
                        ),
                    ),
                ],
            ),
        ]
