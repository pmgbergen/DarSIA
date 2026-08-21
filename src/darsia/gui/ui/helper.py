"""Helper workflow tab for DarSIA GUI."""


class HelperTab:
    """Manages the helper tab UI and workflow execution."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.process = None

    def on_run_clicked(self):
        """Handle run button click."""
        self.main_window.print_log("Helper run: not yet implemented")

    def on_abort_clicked(self):
        """Handle abort button click."""
        if self.process is not None:
            self.main_window.process_runner.abort_workflow_process(self.process)

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
            (
                "Options",
                [
                    (
                        "Show plots",
                        "show",
                        "fa5s.circle",
                        get_help_text("helper", "show", "Show plots"),
                    ),
                ],
            ),
        ]
