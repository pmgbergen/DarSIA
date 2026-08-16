"""Comparison workflow tab for DarSIA GUI."""

from PySide6.QtWidgets import QCheckBox, QPushButton, QVBoxLayout, QWidget


class ComparisonTab:
    """Manages the comparison tab UI and workflow execution."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.comparison_checkboxes = []
        self.process = None

    def create_tab(self):
        """Create and return the comparison tab widget."""
        container = QWidget()
        layout = QVBoxLayout(container)

        comparison_items = [
            ("Events", "events"),
            ("Wasserstein compute", "wasserstein_compute"),
            ("Wasserstein assemble", "wasserstein_assemble"),
        ]

        self.comparison_checkboxes = []
        for label, checkbox_id in comparison_items:
            checkbox = QCheckBox(label)
            self.comparison_checkboxes.append((checkbox_id, checkbox))
            layout.addWidget(checkbox)

        settings_button = QPushButton("Open Comparison settings")
        settings_button.clicked.connect(self.on_settings_clicked)
        layout.addWidget(settings_button)

        self.run_button = QPushButton("Run Comparison")
        self.run_button.clicked.connect(self.on_run_clicked)
        layout.addWidget(self.run_button)

        self.abort_button = QPushButton("Abort Comparison")
        self.abort_button.setVisible(False)
        self.abort_button.setEnabled(False)
        self.abort_button.clicked.connect(self.on_abort_clicked)
        layout.addWidget(self.abort_button)

        layout.addStretch()
        return container

    def on_settings_clicked(self):
        """Handle settings button click."""
        self.main_window.print_log("Comparison settings: not yet implemented")

    def on_run_clicked(self):
        """Handle run button click."""
        self.main_window.print_log("Comparison run: not yet implemented")

    def on_abort_clicked(self):
        """Handle abort button click."""
        if self.process is not None:
            self.main_window.abort_workflow_process(self.process)
