"""Playback of existing saved review frames, paced by completed display reads."""
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget


class CalibrationReviewPlayback(QWidget):
    def __init__(self, slider, panels, parent=None):
        super().__init__(parent)
        self.slider, self.panels = slider, panels
        self._playing = self._advancing = False
        self._ready_generation = None
        self._context = None
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._advance)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.play = QPushButton("Play")
        self.play.setEnabled(False)
        self.play.clicked.connect(self.toggle)
        layout.addWidget(self.play)
        layout.addWidget(QLabel("Playback"))
        self.speed = QComboBox()
        for rate in (0.5, 1, 2, 5):
            self.speed.addItem(f"{rate:g} images/s", rate)
        self.speed.setCurrentIndex(self.speed.findData(2))
        self.speed.setToolTip("Maximum playback rate. Every image is shown; slow saved-file reads reduce the actual rate.")
        self.speed.currentIndexChanged.connect(self._pace)
        layout.addWidget(self.speed)
        self.position = QLabel("Choose a saved result.")
        layout.addWidget(self.position, 1)
        panels.display_requested.connect(self._requested)
        panels.display_presented.connect(self._presented)
        slider.valueChanged.connect(self._slider_changed)
        slider.sliderPressed.connect(self.pause)

    def _requested(self, generation, index):
        self.timer.stop()
        context = (id(self.panels.result), str(self.panels.path)) if self.panels.result is not None else None
        if context != self._context or not self.panels._active:
            self.pause()
        self._context, self._ready_generation = context, None
        available = self.panels._active and context is not None
        self.play.setEnabled(available and self.slider.maximum() > self.slider.minimum())
        self._position(index, loading=available)

    def _position(self, index, *, loading=False, error=False):
        if self._context is None:
            self.position.setText("Choose a saved result.")
            return
        result = self.panels.result
        if isinstance(result, dict):
            frames = result.get("frames", [])
            elapsed = frames[index].get("elapsed_h") if 0 <= index < len(frames) else None
        else:
            elapsed = result.frame_metadata(index).get("elapsed_h")
        time = f" · {elapsed:.3f} h" if isinstance(elapsed, (int, float)) else ""
        state = " · Loading saved images…" if loading else " · Saved image unavailable" if error else ""
        self.position.setText(f"Image {index + 1}/{self.slider.maximum() + 1}{time}{state}")

    def _presented(self, generation, index, available):
        if generation != self.panels._generation or index != self.slider.value():
            return
        self._ready_generation = generation if available else None
        self._position(index, error=not available)
        if not available:
            self.pause()
        else:
            self._pace()

    def _slider_changed(self, _index):
        if not self._advancing:
            self.pause()

    def toggle(self):
        if self._playing:
            self.pause()
            return
        if not self.play.isEnabled() or not self.isVisible():
            return
        self._playing = True
        self.play.setText("Pause")
        if self.slider.value() == self.slider.maximum():
            self._set_index(self.slider.minimum())
        elif self._ready_generation == self.panels._generation:
            self._pace()
        elif self.panels._worker is None and not self.panels._pending:
            self.panels.refresh()

    def _pace(self, *_):
        self.timer.stop()
        if (self._playing and self.isVisible()
                and self._ready_generation == self.panels._generation):
            if self.slider.value() >= self.slider.maximum():
                self.pause()
            else:
                self.timer.start(round(1000 / float(self.speed.currentData())))

    def _set_index(self, index):
        self._advancing = True
        try:
            self.slider.setValue(index)
        finally:
            self._advancing = False

    def _advance(self):
        if (not self._playing or not self.isVisible()
                or self._ready_generation != self.panels._generation):
            return
        self._set_index(min(self.slider.maximum(), self.slider.value() + 1))

    def pause(self, *_):
        self.timer.stop()
        self._playing = False
        self.play.setText("Play")

    def hideEvent(self, event):
        self.pause()
        super().hideEvent(event)

    def closeEvent(self, event):
        self.pause()
        super().closeEvent(event)
