"""Streaming preview panel for the DarSIA GUI.

Displays the low-resolution preview images an Analysis run publishes while
"Stream preview" is enabled (options.analysis.stream_preview). The workflow
subprocess writes each preview frame to a per-run cache directory (one PNG
file per stream key, overwritten in place) and prints a tiny stdout
notification line naming which keys were just (re)written; this panel reacts
to that line by loading the corresponding file. See
darsia.presets.workflows.analysis.streaming for the producer side.
"""

import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from darsia.presets.workflows.analysis.streaming import try_decode_stream_notify_line


class StreamingPanel(QWidget):
    """Live viewer for streamed Analysis preview images, keyed by image type."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._cache_dir: Path | None = None
        self._keys: list[str] = []

        layout = QVBoxLayout(self)

        self.status_label = QLabel("Streaming disabled.")
        layout.addWidget(self.status_label)

        self.key_selector = QComboBox()
        self.key_selector.currentTextChanged.connect(self._on_key_changed)
        layout.addWidget(self.key_selector)

        self.image_label = QLabel("No streamed image.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(200, 200)
        layout.addWidget(self.image_label, stretch=1)

    def prepare_cache_dir(self) -> Path:
        """Free any previous run's cache dir and create a fresh one."""
        if self._cache_dir is not None:
            shutil.rmtree(self._cache_dir, ignore_errors=True)
        self._cache_dir = Path(tempfile.mkdtemp(prefix="darsia_stream_"))
        return self._cache_dir

    def reset_for_run(self, enabled: bool, cache_dir: Path | None) -> None:
        """Called when an Analysis run starts, whether streaming is on or not."""
        self._cache_dir = cache_dir
        self._keys = []
        self.key_selector.clear()
        if enabled:
            self._show_message("Streaming enabled. Waiting for data...")
        else:
            self._show_message("Streaming disabled.")

    def handle_stream_line(self, line: str) -> None:
        """Handle one stdout notification line from the workflow subprocess."""
        try:
            is_stream_line, decoded = try_decode_stream_notify_line(line)
        except Exception:
            self._show_message("Stream error.")
            return
        if not is_stream_line:
            return
        keys = decoded.get("keys") if decoded else None
        if not keys:
            self._show_message("Nothing is streamed.")
            return

        current_key = self.key_selector.currentText()
        self._keys = keys
        self.key_selector.blockSignals(True)
        self.key_selector.clear()
        self.key_selector.addItems(keys)
        selected_key = current_key if current_key in keys else keys[0]
        self.key_selector.setCurrentText(selected_key)
        self.key_selector.blockSignals(False)
        self._render_key(selected_key)

    def _on_key_changed(self, key: str) -> None:
        if key:
            self._render_key(key)

    def _render_key(self, key: str) -> None:
        if self._cache_dir is None:
            self._show_message("Nothing is streamed.")
            return
        path = self._cache_dir / f"{key}.png"
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._show_message("Stream error.")
            return
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        self.status_label.setText(f"Showing stream key: {key}")

    def _show_message(self, message: str) -> None:
        self.image_label.setText(message)
        self.image_label.setPixmap(QPixmap())
        self.status_label.setText(message)

    def cleanup(self) -> None:
        """Best-effort removal of the last cache dir (e.g. on GUI close)."""
        if self._cache_dir is not None:
            shutil.rmtree(self._cache_dir, ignore_errors=True)
            self._cache_dir = None
