"""Playback follows completed saved-panel reads, never the wall-clock alone."""
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSlider, QVBoxLayout, QWidget

from darsia.fluidflower import calibration_review_panels as panels_module
from darsia.fluidflower.calibration_review_playback import CalibrationReviewPlayback
from .test_calibration_review_panels import _ManualRead, _image, _result


@pytest.fixture
def player(qtbot, monkeypatch, tmp_path):
    reads = []
    monkeypatch.setattr(panels_module, "_PanelRead",
        lambda *args: reads.append(_ManualRead(*args)) or reads[-1])
    root = QWidget()
    layout = QVBoxLayout(root)
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, 2)
    panels = panels_module.CalibrationReviewPanels(root)
    panels.layout_choice.setCurrentIndex(panels.layout_choice.findData("side_by_side"))
    playback = CalibrationReviewPlayback(slider, panels, root)
    layout.addWidget(slider)
    layout.addWidget(playback)
    layout.addWidget(panels)
    result, path = _result(), tmp_path / "result.json"
    slider.valueChanged.connect(lambda index: panels.set_result(result, path, index, []))
    panels.set_result(result, path, 0, [])
    panels.set_active(True)
    qtbot.addWidget(root)
    root.show()
    playback.speed.setCurrentIndex(playback.speed.findData(5))
    yield playback, slider, panels, reads, root
    playback.pause()
    panels.stop()


def _complete(read, *, error=""):
    read.ready.emit(read.generation, [(_image(17), error)] * 2)
    read.finished.emit()


def test_playback_waits_for_every_frame_and_keeps_side_by_side_in_sync(player, qtbot):
    playback, slider, panels, reads, _ = player
    playback.play.click()
    qtbot.wait(230)
    assert slider.value() == 0 and len(reads) == 1
    _complete(reads[-1])
    assert playback.timer.isActive()
    playback.timer.timeout.emit()
    assert slider.value() == 1 and len(reads) == 2
    assert reads[-1].index == 1 and len(reads[-1].requests) == 2
    qtbot.wait(230)
    assert slider.value() == 1 and len(reads) == 2  # A slow reader cannot skip ahead.
    _complete(reads[-1])
    assert "Image 2/3 · 0.500 h" in playback.position.text()
    playback.timer.timeout.emit()
    assert slider.value() == 2 and [read.index for read in reads] == [0, 1, 2]
    _complete(reads[-1])
    assert not playback._playing and not playback.timer.isActive()
    assert playback.play.text() == "Play"
    assert panels.index == slider.value() == 2


def test_manual_slider_pause_and_replay_from_end(player):
    playback, slider, panels, reads, _ = player
    _complete(reads[-1])
    playback.play.click()
    slider.setValue(2)
    assert not playback._playing and not playback.timer.isActive()
    _complete(reads[-1])
    playback.play.click()
    assert playback._playing and slider.value() == panels.index == 0
    assert reads[-1].index == 0


def test_pause_on_saved_image_failure_or_new_result(player, tmp_path):
    playback, slider, panels, reads, _ = player
    playback.play.click()
    _complete(reads[-1], error="Saved photograph is unavailable")
    assert not playback._playing and not playback.timer.isActive()
    assert "unavailable" in playback.position.text()
    playback.play.click()
    assert playback._playing
    panels.set_result(_result(), tmp_path / "another-run.json", slider.value(), [])
    assert not playback._playing and not playback.timer.isActive()


def test_hide_and_panel_deactivation_stop_playback(player):
    playback, _, panels, reads, root = player
    _complete(reads[-1])
    playback.play.click()
    root.hide()
    assert not playback._playing and not playback.timer.isActive()
    root.show()
    playback.play.click()
    panels.set_active(False)
    assert not playback._playing and not playback.play.isEnabled()


def test_rate_change_only_changes_display_delay(player):
    playback, _, _, reads, _ = player
    _complete(reads[-1])
    playback.play.click()
    assert playback.timer.interval() == 200
    playback.speed.setCurrentIndex(playback.speed.findData(.5))
    assert playback.timer.interval() == 2000
    assert len(reads) == 1
