"""Helper workflow tab for DarSIA GUI."""

from PySide6.QtWidgets import QCheckBox, QPushButton, QVBoxLayout, QWidget


class HelperTab:
    """Manages the helper tab UI and workflow execution."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.helper_checkboxes = []
        self.process = None

    def create_tab(self):
        """Create and return the helper tab widget."""
        container = QWidget()
        layout = QVBoxLayout(container)

        helper_items = [
            ("Color Embedding", "color"),
            ("ROI", "roi"),
            ("ROI Viewer", "roi_viewer"),
            ("ResultReader", "results"),
            ("Show plots", "show"),
        ]

        self.helper_checkboxes = []
        for label, checkbox_id in helper_items:
            checkbox = QCheckBox(label)
            self.helper_checkboxes.append((checkbox_id, checkbox))
            layout.addWidget(checkbox)

        settings_button = QPushButton("Open Helper settings")
        settings_button.clicked.connect(self.on_settings_clicked)
        layout.addWidget(settings_button)

        self.run_button = QPushButton("Run Helper")
        self.run_button.clicked.connect(self.on_run_clicked)
        layout.addWidget(self.run_button)

        self.abort_button = QPushButton("Abort Helper")
        self.abort_button.setVisible(False)
        self.abort_button.setEnabled(False)
        self.abort_button.clicked.connect(self.on_abort_clicked)
        layout.addWidget(self.abort_button)

        layout.addStretch()
        return container

    def on_settings_clicked(self):
        """Handle settings button click."""
        self.main_window.print_log("Helper settings: not yet implemented")

    def on_run_clicked(self):
        """Handle run button click."""
        self.main_window.print_log("Helper run: not yet implemented")

    def on_abort_clicked(self):
        """Handle abort button click."""
        if self.process is not None:
            self.main_window.abort_workflow_process(self.process)
