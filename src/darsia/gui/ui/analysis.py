"""Analysis workflow tab for DarSIA GUI."""

import sys
from pathlib import Path


class AnalysisTab:
    """Manages the analysis tab UI and workflow execution."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.process = None

    def on_run_clicked(self):
        """Handle run button click."""
        self.run_analysis()

    def on_abort_clicked(self):
        """Handle abort button click."""
        if self.process is not None:
            self.main_window.process_runner.abort_workflow_process(self.process)

    def run_analysis(self):
        """Run analysis workflow based on selected sidebar item."""
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
                self.main_window.config_dict, "options.analysis.show_plots"
            )
        )

        # Build options dictionary matching the CLI interface
        options = {
            "cropping": selected_id == "cropping",
            "segmentation": selected_id == "segmentation",
            "fingers": selected_id == "fingers",
            "mass": selected_id == "mass",
            "volume": selected_id == "volume",
            "thresholding": selected_id == "thresholding",
            "show": show_plots,
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
        if options["show"]:
            argv.append("--show")

        # Launch workflow in a separate process
        play_action = self.main_window.toolbar_builder.play_action
        stop_action = self.main_window.toolbar_builder.stop_action
        self.process = self.main_window.process_runner.start_workflow_process(
            argv, play_action, stop_action, cwd=Path.cwd()
        )

    def sidebar_items(self):
        """Return sidebar data structure for Analysis category."""
        from .help_text import get_help_text

        return [
            (
                "Analysis steps",
                [
                    (
                        "Cropping",
                        "cropping",
                        "fa5s.circle",
                        get_help_text("analysis", "cropping", "Cropping"),
                    ),
                    (
                        "Mass",
                        "mass",
                        "fa5s.circle",
                        get_help_text("analysis", "mass", "Mass"),
                    ),
                    (
                        "Segmentation",
                        "segmentation",
                        "fa5s.circle",
                        get_help_text("analysis", "segmentation", "Segmentation"),
                    ),
                    (
                        "Fingers",
                        "fingers",
                        "fa5s.circle",
                        get_help_text("analysis", "fingers", "Fingers"),
                    ),
                    (
                        "Thresholding",
                        "thresholding",
                        "fa5s.circle",
                        get_help_text("analysis", "thresholding", "Thresholding"),
                    ),
                    (
                        "Volume",
                        "volume",
                        "fa5s.circle",
                        get_help_text("analysis", "volume", "Volume"),
                    ),
                ],
            ),
        ]
