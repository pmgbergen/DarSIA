"""Compatibility slot for the retired experimental optical-inference preview.

The review desktop still paints and inspects the admitted saved model layers.
Threshold/head recomputation, project-profile writes and their worker/timer
scaffolding are intentionally absent from this saved-review surface.
"""
from PySide6.QtWidgets import QWidget


class OpticalCleanup(QWidget):
    """Keep the desktop's widget slot while exposing no inference action."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setVisible(False)

    @property
    def active(self):
        """Saved layers remain the active, unchanged review source."""
        return False

    def decorate(self):
        """Leave the ordinary saved-layer legend and paint controls unchanged."""

    def asset(self, deferred_type, index=None):
        """Report the retired action without loading models or creating maps."""
        raise ValueError(
            "Experimental optical inference preview is retired; open the saved model layer."
        )
