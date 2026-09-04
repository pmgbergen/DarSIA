"""Streaming preview panel for the DarSIA GUI.

Displays the low-resolution preview images an Analysis run publishes while
"Stream preview" is enabled (options.analysis.stream_preview), as a
scrubbable timeline: one entry per processed image, sorted by its real
imaging-protocol timestamp (not by arrival order, since
options.analysis.random_traverse can process images out of chronological
order).

The workflow subprocess writes each preview frame to a per-run cache
directory (one PNG file per (stream key, sequence number), never
overwritten) and prints two tiny stdout notification lines per image: a
stream line naming which keys were just written, tagged with a sequence
number, and a progress line carrying that same sequence number alongside the
image's real index/datetime. This panel correlates the two by that shared
sequence number. See darsia.presets.workflows.analysis.streaming and
darsia.presets.workflows.analysis.progress for the producer side.
"""

import bisect
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from darsia.presets.workflows.analysis.progress import try_decode_progress_line
from darsia.presets.workflows.analysis.streaming import try_decode_stream_notify_line


class StreamingPanel(QWidget):
    """Live/scrubbable viewer for streamed Analysis preview images."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._cache_dir: Path | None = None
        self._pending_seq_keys: dict[int, list[str]] = {}
        self._timeline: list[dict] = []  # sorted by "datetime" (None sorts last)
        self._auto_follow = True

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

        self.timeline_label = QLabel("")
        layout.addWidget(self.timeline_label)

        slider_row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(self._on_user_scrub_start)
        self.slider.valueChanged.connect(self._on_slider_moved)
        slider_row.addWidget(self.slider, stretch=1)

        self.jump_to_latest_button = QPushButton("Jump to latest")
        self.jump_to_latest_button.setEnabled(False)
        self.jump_to_latest_button.clicked.connect(self._jump_to_latest)
        slider_row.addWidget(self.jump_to_latest_button)
        layout.addLayout(slider_row)

    def prepare_cache_dir(self) -> Path:
        """Free any previous run's cache dir and create a fresh one."""
        if self._cache_dir is not None:
            shutil.rmtree(self._cache_dir, ignore_errors=True)
        self._cache_dir = Path(tempfile.mkdtemp(prefix="darsia_stream_"))
        return self._cache_dir

    def reset_for_run(self, enabled: bool, cache_dir: Path | None) -> None:
        """Called when an Analysis run starts, whether streaming is on or not."""
        self._cache_dir = cache_dir
        self._pending_seq_keys = {}
        self._timeline = []
        self._auto_follow = True
        self.key_selector.clear()
        self.slider.blockSignals(True)
        self.slider.setRange(0, 0)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.blockSignals(False)
        self.jump_to_latest_button.setEnabled(False)
        self.timeline_label.setText("")
        if enabled:
            self._show_message("Streaming enabled. Waiting for data...")
        else:
            self._show_message("Streaming disabled.")

    def handle_stream_line(self, line: str) -> None:
        """Handle one stream-notify stdout line (names the keys just written)."""
        try:
            is_stream_line, decoded = try_decode_stream_notify_line(line)
        except Exception:
            self._show_message("Stream error.")
            return
        if not is_stream_line:
            return
        keys = decoded.get("keys") if decoded else None
        seq = decoded.get("seq") if decoded else None
        if not keys or seq is None:
            # payload=None ("clear", from an encoding error upstream): the
            # paired progress event (still sent unconditionally) will find
            # nothing pending for this seq and skip it. Nothing to do here.
            return
        self._pending_seq_keys[seq] = keys

    def handle_progress_line(self, line: str) -> None:
        """Handle one progress-notify stdout line (carries index/datetime),
        pairing it with the matching pending stream keys by shared seq."""
        try:
            is_progress_line, event = try_decode_progress_line(line)
        except Exception:
            return
        if not is_progress_line or event is None:
            return
        if event.get("event") != "image_progress":
            return
        seq = event.get("seq")
        if seq is None:
            return
        keys = self._pending_seq_keys.pop(seq, None)
        if keys is None:
            return

        raw_datetime = event.get("image_datetime")
        entry_datetime = None
        if raw_datetime:
            try:
                entry_datetime = datetime.fromisoformat(raw_datetime)
            except ValueError:
                entry_datetime = None

        entry = {
            "seq": seq,
            "image_index": event.get("image_index"),
            "image_path": event.get("image_path"),
            "datetime": entry_datetime,
            "keys": keys,
        }
        self._insert_entry(entry)

    def _insert_entry(self, entry: dict) -> None:
        sort_key = entry["datetime"] or datetime.max
        existing_keys = [e["datetime"] or datetime.max for e in self._timeline]
        position = bisect.bisect_left(existing_keys, sort_key)
        self._timeline.insert(position, entry)

        # Refresh the key dropdown with the union of keys seen so far,
        # preserving the current selection if still valid.
        current_key = self.key_selector.currentText()
        all_keys = sorted({k for e in self._timeline for k in e["keys"]})
        shown_keys = [
            self.key_selector.itemText(i) for i in range(self.key_selector.count())
        ]
        if all_keys != shown_keys:
            self.key_selector.blockSignals(True)
            self.key_selector.clear()
            self.key_selector.addItems(all_keys)
            if current_key in all_keys:
                self.key_selector.setCurrentText(current_key)
            self.key_selector.blockSignals(False)

        self.slider.blockSignals(True)
        self.slider.setRange(0, len(self._timeline) - 1)
        self.slider.setEnabled(True)
        self.slider.blockSignals(False)

        if self._auto_follow:
            self.jump_to_latest_button.setEnabled(False)
            self._set_slider_position(position)
        else:
            self.jump_to_latest_button.setEnabled(True)

    def _on_user_scrub_start(self) -> None:
        self._auto_follow = False
        self.jump_to_latest_button.setEnabled(True)

    def _on_slider_moved(self, value: int) -> None:
        self._render_position(value)

    def _jump_to_latest(self) -> None:
        self._auto_follow = True
        self.jump_to_latest_button.setEnabled(False)
        if self._timeline:
            self._set_slider_position(len(self._timeline) - 1)

    def _set_slider_position(self, index: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self._render_position(index)

    def _on_key_changed(self, key: str) -> None:
        if key:
            self._render_position(self.slider.value())

    def _render_position(self, index: int) -> None:
        if not self._timeline or not (0 <= index < len(self._timeline)):
            self._show_message("Nothing is streamed.")
            return
        entry = self._timeline[index]
        key = self.key_selector.currentText()
        if not key or self._cache_dir is None:
            self._show_message("Nothing is streamed.")
            return
        path = self._cache_dir / key / f"{entry['seq']}.png"
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
        when = (
            entry["datetime"].strftime("%Y-%m-%d %H:%M:%S")
            if entry["datetime"]
            else "unknown time"
        )
        self.timeline_label.setText(f"{index + 1} / {len(self._timeline)} — {when}")
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
