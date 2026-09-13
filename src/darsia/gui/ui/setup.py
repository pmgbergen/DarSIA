"""Setup workflow tab for DarSIA GUI: depth map, segmentation, facies, rig.

Preprocessing (protocols, depth measurements, crop correction) is a separate
tab/category — see :mod:`darsia.gui.ui.preprocessing`.
"""

import sys
from pathlib import Path


class SetupTab:
    """Manages the setup tab UI and workflow execution."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.process = None

    def on_run_clicked(self):
        """Handle run button click."""
        self.run_setup()

    def on_abort_clicked(self):
        """Handle abort button click."""
        if self.process is not None:
            self.main_window.process_runner.abort_workflow_process(self.process)

    def run_setup(self):
        """Run setup workflow based on selected sidebar item."""
        config_file = self.main_window.config_path_label.text()
        if not config_file or config_file == "No file chosen":
            self.main_window.print_log("Please select a config file first.")
            return

        selected_id = self.main_window.selected_checkbox_id
        if not selected_id:
            self.main_window.print_log("Please select an option in the sidebar.")
            return

        # Sync GUI widgets to config_dict to read current show_plots setting
        self.main_window.settings_factory._sync_settings_inputs_to_config_dict()
        show_plots = bool(
            self.main_window.settings_factory.get_value(
                self.main_window.config_dict, "options.setup.show_plots"
            )
        )

        # Build options dictionary matching the CLI interface
        options = {
            "all": selected_id == "all",
            "depth": selected_id == "depth",
            "segmentation": selected_id == "segmentation",
            "facies": selected_id == "facies",
            "rig": selected_id == "rig",
            "show": show_plots,
            "force": False,
        }

        self.main_window.print_log(
            """Starting setup with options: """
            f"""{[k for k, v in options.items() if v and k != "force"]}"""
        )

        # Build command-line arguments for subprocess
        argv = [
            sys.executable,
            "-m",
            "darsia.presets.workflows.user_interface_setup",
            "--config",
            str(Path(config_file).resolve()),
        ]
        if options["all"]:
            argv.append("--all")
        if options["depth"]:
            argv.append("--depth")
        if options["segmentation"]:
            argv.append("--segmentation")
        if options["facies"]:
            argv.append("--facies")
        if options["rig"]:
            argv.append("--rig")
        if options["force"]:
            argv.append("--force")
        if options["show"]:
            argv.append("--show")

        # Launch workflow in a separate process
        play_action = self.main_window.toolbar_builder.play_action
        stop_action = self.main_window.toolbar_builder.stop_action

        self.process = self.main_window.process_runner.start_workflow_process(
            argv,
            play_action,
            stop_action,
            cwd=Path.cwd(),
            workflow="setup",
            actions=[selected_id],
            config_path=Path(config_file),
        )

    def sidebar_items(self):
        """Return sidebar data structure for Setup category."""
        from .help_text import get_help_text

        return [
            (
                "Full setup",
                [
                    ("All", "all", "fa5s.circle", get_help_text("setup", "all", "All")),
                ],
            ),
            (
                "Setup steps",
                [
                    (
                        "Depth",
                        "depth",
                        "fa5s.circle",
                        get_help_text("setup", "depth", "Depth"),
                    ),
                    (
                        "Segmentation",
                        "segmentation",
                        "fa5s.circle",
                        get_help_text("setup", "segmentation", "Segmentation"),
                    ),
                    (
                        "Facies",
                        "facies",
                        "fa5s.circle",
                        get_help_text("setup", "facies", "Facies"),
                    ),
                    ("Rig", "rig", "fa5s.circle", get_help_text("setup", "rig", "Rig")),
                ],
            ),
        ]
