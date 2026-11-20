"""Calibration workflow tab for DarSIA GUI."""

import sys
from pathlib import Path

from PySide6.QtWidgets import QMessageBox


def _confirm_resolution_level(main_window, config_file: str) -> bool:
    """Warn if the requested resolution level of the mass calibration is unavailable.

    Returns
    -------
        True if the calibration may start, False if the user cancelled.

    """
    try:
        from darsia.presets.workflows.calibration.consistency import (
            unavailable_resolution_level,
        )

        reason = unavailable_resolution_level(Path(config_file).resolve())
    except Exception as error:  # a check must never block the calibration
        main_window.print_log(f"Skipped the resolution level check: {error}")
        return True
    if reason is None:
        return True

    main_window.print_log(reason)
    box = QMessageBox(main_window)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle("Resolution level not available")
    box.setText(reason)
    box.setInformativeText(
        "The calibration runs at full resolution (1/1), without the level slider: "
        "each Update takes longer and needs more memory. To calibrate at a reduced "
        "level, remove the restoration method; set resolution_level = 1 to start at "
        "full resolution without this warning.\n\nContinue at full resolution?"
    )
    continue_button = box.addButton("Continue", QMessageBox.AcceptRole)
    box.addButton("Cancel", QMessageBox.RejectRole)
    box.setDefaultButton(continue_button)
    box.exec()
    if box.clickedButton() is not continue_button:
        main_window.print_log("Calibration cancelled: resolution level not available.")
        return False
    return True


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

        if selected_id == "mass" and not _confirm_resolution_level(
            self.main_window, config_file
        ):
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
            config_path=Path(config_file),
        )

    def sidebar_items(self):
        """Return sidebar data structure for Calibration category.

        Names are fluid-neutral: the second step calibrates the interpretation of the
        color path (color to signal); only its anchor depends on the fluid (injected mass
        for CO2, injected volume for tracer). The ids are kept for the CLI flags.
        """
        from .help_text import get_help_text

        def item(checkbox_id, label):
            return (
                label,
                checkbox_id,
                "fa5s.circle",
                get_help_text("calibration", checkbox_id, label),
            )

        return [
            (
                "Actions",
                [
                    item("color", "Color Path"),
                    item("mass", "Interpretation"),
                    item("default_mass", "Default interpretation"),
                ],
            ),
            (
                "Danger zone",
                [
                    item("reset", "Reset interpretation"),
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
