"""Slow metadata never blocks selection, and owners retire before close."""

from pathlib import Path
from threading import Event

from PySide6.QtCore import QCoreApplication, QThread

from darsia.fluidflower import image_normalisation as method
from darsia.fluidflower.image_normalisation_step import ImageNormalisationChoice
from darsia.fluidflower.normalisation_step import NormalisationStep, read_series_context


def manifest(root, series):
    root.mkdir()
    raw = f'series_id = "{series}"\n[runs]\nids = ["{series.lower()}1"]\n'.encode()
    (root / "series.toml").write_bytes(raw)
    return raw


def release_and_retire(qtbot, panel, release):
    """Assertions must not strand a blocked native owner during test teardown."""
    release.set()
    try:
        qtbot.waitUntil(lambda: panel.can_close() and not getattr(panel, "metadata_busy", lambda: False)())
    finally:
        if not panel.can_close():
            panel.wait_for_workers()


def test_slow_choice_read_is_offthread_and_old_selection_is_discarded(qtbot, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    manifest(first, "A")
    second_raw = manifest(second, "B")
    entered, release = Event(), Event()
    calls = []
    def reader(root, series):
        calls.append((str(root), series, QThread.currentThread()))
        if series == "A":
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release metadata read")
        return read_series_context(root, series)
    panel = ImageNormalisationChoice(reader=reader)
    qtbot.addWidget(panel)
    try:
        panel.set_context(first, "A")
        qtbot.waitUntil(entered.is_set)
        assert not panel.can_close()
        panel.set_context(second, "B")
        assert panel.original is None and panel.reference.count() == 0
        # This GUI operation runs while the original metadata reader is blocked.
        panel.status.setText("GUI remains responsive")
        assert panel.status.text() == "GUI remains responsive"
    finally:
        release_and_retire(qtbot, panel, release)
    assert panel.series_id == "B" and panel.original == second_raw and panel.runs == ("b1",)
    assert all(thread is not QCoreApplication.instance().thread() for _root, _series, thread in calls)


def test_close_retains_slow_choice_owner_and_discards_late_packet(qtbot, tmp_path):
    manifest(tmp_path / "a", "A")
    entered, release = Event(), Event()
    def reader(root, series):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release metadata read")
        return read_series_context(root, series)
    panel = ImageNormalisationChoice(reader=reader)
    qtbot.addWidget(panel)
    panel.show()
    try:
        panel.set_context(tmp_path / "a", "A")
        qtbot.waitUntil(entered.is_set)
        owner = panel.thread
        panel.close()
        assert panel.thread is owner and not panel.can_close()
        assert panel.original is None
    finally:
        release_and_retire(qtbot, panel, release)
    assert panel.thread is None and panel.worker is None and panel.original is None


def test_save_owns_original_root_choice_and_manifest_when_context_switches(qtbot, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first_raw, second_raw = manifest(first, "A"), manifest(second, "B")
    entered, release = Event(), Event()
    calls = []
    def writer(root, series, runs, selected, reference, *, color_method, stable_band_mm, expected_bytes):
        calls.append((root, series, runs, selected, reference, expected_bytes, QThread.currentThread()))
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release metadata save")
        return method.save_choice(root, series, runs, selected, reference,
                                  color_method=color_method, stable_band_mm=stable_band_mm,
                                  expected_bytes=expected_bytes)
    panel = ImageNormalisationChoice(writer=writer)
    qtbot.addWidget(panel)
    panel.set_context(first, "A")
    qtbot.waitUntil(lambda: panel.can_close() and not getattr(panel, "metadata_busy", lambda: False)())
    panel.method.setCurrentIndex(panel.method.findData(method.OFF))
    try:
        panel.save_button.click()
        qtbot.waitUntil(entered.is_set)
        panel.set_context(second, "B")
        assert not panel.save_button.isEnabled()
    finally:
        release_and_retire(qtbot, panel, release)
    root, series, runs, selected, reference, expected, worker_thread = calls[0]
    assert Path(root) == first and series == "A" and runs == ("a1",)
    assert selected == method.OFF and reference is None and expected == first_raw
    assert worker_thread is not QCoreApplication.instance().thread()
    assert (second / "series.toml").read_bytes() == second_raw
    assert panel.series_id == "B" and panel.original == second_raw


def test_parent_reads_manifest_once_offthread_and_shares_snapshot(qtbot, tmp_path, monkeypatch):
    from darsia.fluidflower import normalisation_step as normalisation
    root = tmp_path / "a"
    raw = manifest(root, "A")
    entered, release = Event(), Event()
    calls = []
    def reader(selected_root, selected_series):
        calls.append((str(selected_root), selected_series, QThread.currentThread()))
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release parent metadata read")
        return read_series_context(selected_root, selected_series)
    monkeypatch.setattr(normalisation, "read_series_context", reader)
    panel = NormalisationStep()
    qtbot.addWidget(panel)
    try:
        panel.set_context(root, "A")
        qtbot.waitUntil(entered.is_set)
        assert panel.can_close() and panel.metadata_busy() and not panel.method_choice.save_button.isEnabled()
        panel.status.setText("GUI remains responsive")
    finally:
        release_and_retire(qtbot, panel, release)
    assert len(calls) == 1 and calls[0][2] is not QCoreApplication.instance().thread()
    assert panel.method_choice.original == raw
    assert not panel.ready


def test_shutdown_wait_retires_native_thread_without_consuming_packet(qtbot, tmp_path):
    root = tmp_path / "a"
    manifest(root, "A")
    panel = ImageNormalisationChoice()
    qtbot.addWidget(panel)
    panel.set_context(root, "A")
    # The shutdown barrier may wait natively; interactive Close never does.
    panel.wait_for_workers()
    assert panel.can_close() and panel.thread is None and panel.worker is None
    assert panel.original is None and panel._closing


def test_native_phase_close_does_not_wait_on_disposable_normalisation_metadata(qtbot, tmp_path, monkeypatch):
    from darsia.fluidflower import normalisation_step as normalisation
    from darsia.fluidflower.native_phase_steps import NativePhaseSteps
    from .test_native_phase_steps import FakeStudio
    root = tmp_path / "a"
    manifest(root, "A")
    entered, release = Event(), Event()
    def reader(selected_root, selected_series):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release native close guard")
        return read_series_context(selected_root, selected_series)
    monkeypatch.setattr(normalisation, "read_series_context", reader)
    panel = NormalisationStep()
    qtbot.addWidget(panel)
    studio = FakeStudio()
    qtbot.addWidget(studio)
    steps = NativePhaseSteps(studio, normalisation=panel)
    qtbot.addWidget(steps)
    try:
        panel.set_context(root, "A")
        qtbot.waitUntil(entered.is_set)
        assert steps.can_close() and panel.metadata_busy()
        steps.cancel_metadata_previews()
        assert not panel.metadata_busy() and steps.can_close()
    finally:
        release_and_retire(qtbot, panel, release)
    assert steps.can_close()
