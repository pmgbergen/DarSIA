"""Comparison workflow tab for DarSIA GUI."""

import sys
from pathlib import Path


class ComparisonTab:
    """Manages the comparison tab UI and workflow execution."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.process = None

    def on_run_clicked(self):
        """Handle run button click."""
        self.run_comparison()

    def on_abort_clicked(self):
        """Handle abort button click."""
        if self.process is not None:
            self.main_window.process_runner.abort_workflow_process(self.process)

    def run_comparison(self):
        """Run comparison workflow based on selected sidebar item."""
        config_file = self.main_window.config_path_label.text()
        if not config_file or config_file == "No file chosen":
            self.main_window.print_log("Please select a config file first.")
            return

        selected_id = self.main_window.selected_checkbox_id
        if not selected_id:
            self.main_window.print_log("Please select an option in the sidebar.")
            return

        # Build command-line arguments for subprocess. The sidebar is
        # single-selection, so exactly one of --events/--wasserstein-compute/
        # --wasserstein-assemble is ever passed, satisfying the CLI's
        # "exactly one" requirement by construction.
        argv = [
            sys.executable,
            "-m",
            "darsia.presets.workflows.user_interface_comparison",
            "--config",
            str(Path(config_file).resolve()),
        ]
        if selected_id == "events":
            argv.append("--events")
        elif selected_id == "wasserstein_compute":
            argv.append("--wasserstein-compute")
        elif selected_id == "wasserstein_assemble":
            argv.append("--wasserstein-assemble")

        self.main_window.print_log(f"Starting comparison: {selected_id}")

        # Launch workflow in a separate process
        play_action = self.main_window.toolbar_builder.play_action
        stop_action = self.main_window.toolbar_builder.stop_action
        self.process = self.main_window.process_runner.start_workflow_process(
            argv,
            play_action,
            stop_action,
            cwd=Path.cwd(),
            workflow="comparison",
            actions=[selected_id],
            config_paths=[Path(config_file)],
        )

    def sidebar_items(self):
        """Return sidebar data structure for Comparison category."""
        from .help_text import get_help_text

        return [
            (
                "Actions",
                [
                    (
                        "Events",
                        "events",
                        "fa5s.circle",
                        get_help_text("comparison", "events", "Events"),
                    ),
                    (
                        "Wasserstein compute",
                        "wasserstein_compute",
                        "fa5s.circle",
                        get_help_text(
                            "comparison", "wasserstein_compute", "Wasserstein compute"
                        ),
                    ),
                    (
                        "Wasserstein assemble",
                        "wasserstein_assemble",
                        "fa5s.circle",
                        get_help_text(
                            "comparison",
                            "wasserstein_assemble",
                            "Wasserstein assemble",
                        ),
                    ),
                ],
            ),
        ]
