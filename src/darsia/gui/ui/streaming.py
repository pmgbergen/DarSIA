"""View panel for the DarSIA GUI: streaming preview and results browsing.

Two modes share one image display/slider/status trio, switched via
mode_selector:

- Streaming mode displays the low-resolution preview images an Analysis run
  publishes while "Stream preview" is enabled (options.analysis.stream_preview),
  as a scrubbable timeline: one entry per processed image, sorted by its real
  imaging-protocol timestamp. The workflow subprocess writes each preview
  frame to a per-run cache directory (one PNG file per (stream key, sequence
  number), never overwritten) and prints two tiny stdout notification lines
  per image: a stream line naming which keys were just written, tagged with
  a sequence number, and a progress line carrying that same sequence number
  alongside the image's real index/datetime. This panel correlates the two
  by that shared sequence number. See darsia.presets.workflows.analysis.streaming
  and darsia.presets.workflows.analysis.progress for the producer side.
  Entering this mode is triggered automatically when a run starts with
  Stream preview enabled (reset_for_run); the user can still switch back to
  Results mode at any time, including mid-run.

- Results mode browses every on-disk output image for whichever sidebar
  step is currently selected (results_folder.list_workflow_output_images),
  refreshed whenever the sidebar selection changes or a run completes (see
  refresh_results, called from main_window.py).
"""

import bisect
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
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
from darsia.presets.workflows.results_folder import list_workflow_output_images

from .image_canvas import ImageCanvas

_NO_SELECTION_TEXT = "Select a step to see its results."
_NO_IMAGE_TEXT = "No image output yet for this step."
_NO_CONFIG_TEXT = "Load a config to see step output."


class StreamingPanel(QWidget):
    """View panel: live/scrubbable streamed preview, or on-disk results browsing."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._mode = "results"

        # Streaming-mode state.
        self._cache_dir: Path | None = None
        self._pending_seq_keys: dict[int, list[str]] = {}
        self._timeline: list[dict] = []  # sorted by "datetime" (None sorts last)
        self._auto_follow = True
        self._streaming_index = 0

        # Results-mode state.
        self._results_images: list[Path] = []
        self._results_index = 0

        layout = QVBoxLayout(self)

        # Built and defaulted to "Results" before the signal is connected
        # (below, once the rest of the widgets exist): its handler calls
        # refresh_results(), which reads main_window state not yet set up
        # this early in MainWindow.__init__.
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Streaming", "Results"])
        self.mode_selector.blockSignals(True)
        self.mode_selector.setCurrentText("Results")
        self.mode_selector.blockSignals(False)
        layout.addWidget(self.mode_selector)

        self.status_label = QLabel("Streaming disabled.")
        layout.addWidget(self.status_label)

        self.key_selector = QComboBox()
        self.key_selector.currentTextChanged.connect(self._on_key_changed)
        layout.addWidget(self.key_selector)

        self.image_canvas = ImageCanvas(placeholder_text=_NO_SELECTION_TEXT)
        layout.addWidget(self.image_canvas, stretch=1)

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

        self.key_selector.setVisible(False)
        self._show_message(_NO_SELECTION_TEXT)

        # Connected only now that every widget _apply_mode/refresh_results
        # touches exists (see the comment by mode_selector's construction).
        self.mode_selector.currentTextChanged.connect(
            lambda text: self._apply_mode(text.lower())
        )

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def _apply_mode(self, mode: str) -> None:
        self._mode = mode
        self.key_selector.setVisible(mode == "streaming")
        if mode == "streaming":
            self.slider.blockSignals(True)
            self.slider.setRange(0, max(0, len(self._timeline) - 1))
            index = (
                len(self._timeline) - 1 if self._auto_follow else self._streaming_index
            )
            index = max(0, index)
            self.slider.setValue(index)
            self.slider.setEnabled(bool(self._timeline))
            self.slider.blockSignals(False)
            self.jump_to_latest_button.setEnabled(
                bool(self._timeline) and not self._auto_follow
            )
            if self._timeline:
                self._render_streaming_position(index)
            else:
                self._show_message("Streaming disabled.")
        else:
            self.refresh_results()

    # ------------------------------------------------------------------
    # Results mode
    # ------------------------------------------------------------------

    def refresh_results(self) -> None:
        """Re-resolve on-disk output images for the current sidebar
        selection. Call on sidebar selection change and after
        refresh_sidebar_progress() (covers initial load, config reload, and
        post-run completion, since all three already call it). Rescans
        disk unconditionally, but only touches the shared display widgets
        when Results mode is currently active."""
        mw = self.main_window
        action, checkbox_id = mw.selected_action, mw.selected_checkbox_id
        if action is None or checkbox_id is None:
            self._results_images = []
            if self._mode == "results":
                self._show_message(_NO_SELECTION_TEXT)
            return
        if not mw.config_file or not Path(mw.config_file).exists():
            self._results_images = []
            if self._mode == "results":
                self._show_message(_NO_CONFIG_TEXT)
            return
        action_label = mw.action_label_for(action, checkbox_id)
        try:
            self._results_images = list_workflow_output_images(
                action, Path(mw.config_file), [action_label]
            )
        except Exception:
            self._results_images = []
        self._results_index = max(0, len(self._results_images) - 1)
        if self._mode != "results":
            return
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(self._results_images) - 1))
        self.slider.setValue(self._results_index)
        self.slider.setEnabled(bool(self._results_images))
        self.slider.blockSignals(False)
        self.jump_to_latest_button.setEnabled(False)
        if self._results_images:
            self._render_results_position(self._results_index)
        else:
            self._show_message(_NO_IMAGE_TEXT)

    def _render_results_position(self, index: int) -> None:
        if not self._results_images or not (0 <= index < len(self._results_images)):
            self._show_message(_NO_IMAGE_TEXT)
            return
        path = self._results_images[index]
        self.image_canvas.set_image_path(path)
        self.timeline_label.setText(f"{index + 1} / {len(self._results_images)}")
        self.timeline_label.setToolTip(str(path))
        self.status_label.setText(f"Showing results for: {path.name}")

    # ------------------------------------------------------------------
    # Streaming mode
    # ------------------------------------------------------------------

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
        self._streaming_index = 0
        self.key_selector.clear()
        self.slider.blockSignals(True)
        self.slider.setRange(0, 0)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.blockSignals(False)
        self.jump_to_latest_button.setEnabled(False)
        self.timeline_label.setText("")
        if enabled:
            self.mode_selector.blockSignals(True)
            self.mode_selector.setCurrentText("Streaming")
            self.mode_selector.blockSignals(False)
            self._apply_mode("streaming")
            self._show_message("Streaming enabled. Waiting for data...")
        elif self._mode == "streaming":
            self._show_message("Streaming disabled.")

    def handle_stream_line(self, line: str) -> None:
        """Handle one stream-notify stdout line (names the keys just written)."""
        try:
            is_stream_line, decoded = try_decode_stream_notify_line(line)
        except Exception:
            if self._mode == "streaming":
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
        # preserving the current selection if still valid. Kept up to date
        # unconditionally (cheap, and hidden while not in Streaming mode)
        # so switching back to Streaming mode later shows current data.
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

        if self._mode != "streaming":
            return

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
        if self._mode != "streaming":
            return
        self._auto_follow = False
        self.jump_to_latest_button.setEnabled(True)

    def _on_slider_moved(self, value: int) -> None:
        if self._mode == "streaming":
            self._streaming_index = value
            self._render_streaming_position(value)
        else:
            self._results_index = value
            self._render_results_position(value)

    def _jump_to_latest(self) -> None:
        if self._mode == "streaming":
            self._auto_follow = True
            self.jump_to_latest_button.setEnabled(False)
            if self._timeline:
                self._set_slider_position(len(self._timeline) - 1)
        else:
            if self._results_images:
                self._results_index = len(self._results_images) - 1
                self._set_slider_position(self._results_index)

    def _set_slider_position(self, index: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        if self._mode == "streaming":
            self._render_streaming_position(index)
        else:
            self._render_results_position(index)

    def _on_key_changed(self, key: str) -> None:
        if key and self._mode == "streaming":
            self._render_streaming_position(self.slider.value())

    def _render_streaming_position(self, index: int) -> None:
        if not self._timeline or not (0 <= index < len(self._timeline)):
            self._show_message("Nothing is streamed.")
            return
        entry = self._timeline[index]
        key = self.key_selector.currentText()
        if not key or self._cache_dir is None:
            self._show_message("Nothing is streamed.")
            return
        path = self._cache_dir / key / f"{entry['seq']}.png"
        self.image_canvas.set_image_path(path)
        when = (
            entry["datetime"].strftime("%Y-%m-%d %H:%M:%S")
            if entry["datetime"]
            else "unknown time"
        )
        self.timeline_label.setText(f"{index + 1} / {len(self._timeline)} — {when}")
        self.timeline_label.setToolTip(str(path))
        self.status_label.setText(f"Showing stream key: {key}")

    def _show_message(self, message: str) -> None:
        self.image_canvas.set_message(message)
        self.status_label.setText(message)

    def cleanup(self) -> None:
        """Best-effort removal of the last cache dir (e.g. on GUI close)."""
        if self._cache_dir is not None:
            shutil.rmtree(self._cache_dir, ignore_errors=True)
            self._cache_dir = None
