from pathlib import Path
import importlib.util
import sys

from darsia.fluidflower.series_method import MethodSelection, SeriesMethodPanel

from .test_native_phase_steps import FakeStudio

from darsia.fluidflower import native_phase_steps as ui


def test_real_a5_colour_use_bypasses_only_phase_gate(qtbot):
    studio = FakeStudio()
    method = SeriesMethodPanel()
    calls = []
    original_use = calls.append
    method.use_requested.connect(original_use)
    steps = ui.NativePhaseSteps(studio, method_panel=method, method_use=original_use)
    qtbot.addWidget(studio)
    qtbot.addWidget(steps)
    colour = MethodSelection(
        active="colour_path", colour_calibration="saved-colour-calibration"
    )
    method.set_selection(colour, available=True)
    steps.refresh()
    assert not steps._ready
    assert method.use_button.isEnabled()
    method.use_button.click()
    assert calls == [colour]
    phase = MethodSelection(
        active="phase_model", model_version="saved-model", mass_calibration="saved-mass"
    )
    method.set_selection(phase, available=True)
    steps.refresh()
    assert not method.use_button.isEnabled()
    method.use_requested.emit(phase)
    assert calls == [colour]
    steps.set_normalisation_ready(True)
    assert method.use_button.isEnabled()
    method.use_button.click()
    assert calls == [colour, phase]
