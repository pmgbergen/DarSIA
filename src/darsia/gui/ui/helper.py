"""Helper workflow tab for DarSIA GUI."""

import sys
from pathlib import Path


class HelperTab:
    """Manages the helper tab UI and workflow execution."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.process = None

    def on_run_clicked(self):
        """Handle run button click."""
        self.run_helper()

    def on_abort_clicked(self):
        """Handle abort button click."""
        if self.process is not None:
            self.main_window.process_runner.abort_workflow_process(self.process)

    def run_helper(self):
        """Run helper workflow based on selected sidebar item."""
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
                self.main_window.config_dict, "options.helper.show_plots"
            )
        )

        # Build options dictionary matching the CLI interface
        options = {
            "color": selected_id == "color",
            "roi": selected_id == "roi",
            "roi_viewer": selected_id == "roi_viewer",
            "results": selected_id == "results",
            "show": show_plots,
        }

        self.main_window.print_log(
            f"Starting helper with options: {[k for k, v in options.items() if v]}"
        )

        # Build command-line arguments for subprocess
        argv = [
            sys.executable,
            "-m",
            "darsia.presets.workflows.user_interface_helper",
            "--config",
            str(Path(config_file).resolve()),
        ]
        if options["color"]:
            argv.append("--color")
        if options["roi"]:
            argv.append("--roi")
        if options["roi_viewer"]:
            argv.append("--roi-viewer")
        if options["results"]:
            argv.append("--results")
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
            workflow="helper",
            actions=[selected_id],
            config_path=Path(config_file),
        )

    def sidebar_items(self):
        """Return sidebar data structure for Helper category."""
        from .help_text import get_help_text

        return [
            (
                "Actions",
                [
                    (
                        "Color Embedding",
                        "color",
                        "fa5s.circle",
                        get_help_text("helper", "color", "Color Embedding"),
                    ),
                    (
                        "ROI",
                        "roi",
                        "fa5s.circle",
                        get_help_text("helper", "roi", "ROI"),
                    ),
                    (
                        "ROI Viewer",
                        "roi_viewer",
                        "fa5s.circle",
                        get_help_text("helper", "roi_viewer", "ROI Viewer"),
                    ),
                    (
                        "ResultReader",
                        "results",
                        "fa5s.circle",
                        get_help_text("helper", "results", "ResultReader"),
                    ),
                ],
            ),
        ]
