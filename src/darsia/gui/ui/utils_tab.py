"""Utils workflow tab for DarSIA GUI."""


class UtilsTab:
    """Manages the utils tab UI and workflow execution."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.process = None

    def on_run_clicked(self):
        """Handle run button click."""
        self.main_window.print_log("Utils run: not yet implemented")

    def on_abort_clicked(self):
        """Handle abort button click."""
        if self.process is not None:
            self.main_window.process_runner.abort_workflow_process(self.process)

    def sidebar_items(self):
        """Return sidebar data structure for Utils category."""
        from .help_text import get_help_text

        return [
            (
                "Actions",
                [
                    (
                        "Build protocol-time media (MP4/GIF)",
                        "build_media",
                        "fa5s.circle",
                        get_help_text(
                            "build_media", "Build protocol-time media (MP4/GIF)"
                        ),
                    ),
                    (
                        "Download/cache data",
                        "download_data",
                        "fa5s.circle",
                        get_help_text("utils", "download_data", "Download/cache data"),
                    ),
                    (
                        "Export calibration",
                        "export_calibration",
                        "fa5s.circle",
                        get_help_text(
                            "utils", "export_calibration", "Export calibration"
                        ),
                    ),
                    (
                        "Import calibration",
                        "import_calibration",
                        "fa5s.circle",
                        get_help_text(
                            "utils", "import_calibration", "Import calibration"
                        ),
                    ),
                ],
            ),
        ]
