"""The native Training page exposes the original asynchronous saved-review controls."""
from threading import Event, get_ident

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QScrollArea

from darsia.fluidflower.native_phase_steps import NativePhaseSteps
from .test_native_phase_steps import FakeStudio
from .test_review_session_refresh import studio  # noqa: F401


def test_native_training_shows_saved_review_refresh_loading_error_and_retry(
    studio, qtbot, monkeypatch, tmp_path, record_property,
):
    page, initial = studio
    button, status = page.refresh_review_sessions_button, page.saved_sessions_status
    steps = NativePhaseSteps(page, actor_provider=lambda: "reviewer")
    qtbot.addWidget(steps)
    steps.show_section("phase_model")
    steps.phase_tabs.setCurrentWidget(steps.groups[4])
    steps.resize(1780, 950)
    steps.show()
    steps.scroll.ensureWidgetVisible(status)
    qtbot.waitUntil(lambda: button.isVisible() and not button.visibleRegion().isEmpty())
    assert page.isHidden()
    assert page.saved_sessions.parentWidget() is steps.groups[4]
    assert button.parentWidget() is status.parentWidget() is steps.groups[4]
    assert [area for area in steps.findChildren(QScrollArea) if area.isVisible()] == [steps.scroll]
    assert steps.scroll is steps.phase_workspace.controls_scroll
    assert steps.width() == 1780
    assert steps.scroll.horizontalScrollBar().maximum() == 0
    entered, release = Event(), Event()
    calls = []
    main_thread = get_ident()

    def slow_discovery(_roots):
        calls.append(get_ident())
        entered.set()
        assert release.wait(5)
        return [initial]

    monkeypatch.setattr(page, "_discover_review_session_entries", slow_discovery)
    try:
        qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(entered.is_set)
        assert page._review_sessions_loading and not button.isEnabled()
        assert status.isVisible() and not status.visibleRegion().isEmpty()
        assert "Loading saved review sessions" in status.text()
        assert page._selected_session() == initial
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        qtbot.waitUntil(lambda: bool(ticks))
        screenshot = tmp_path / "native_training_saved_reviews_loading.png"
        assert steps.grab().save(str(screenshot))
        record_property("private_gui_screenshot", str(screenshot))
        assert len(calls) == 1 and calls[0] != main_thread
    finally:
        release.set()
        qtbot.waitUntil(lambda: not page._review_sessions_loading)
    assert button.isEnabled()

    def unavailable(_roots):
        calls.append(get_ident())
        raise OSError("Saved review storage is unavailable")

    monkeypatch.setattr(page, "_discover_review_session_entries", unavailable)
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not page._review_sessions_loading)
    assert len(calls) == 2
    assert status.isVisible() and not status.visibleRegion().isEmpty()
    assert "OSError: Saved review storage is unavailable" in status.text()
    assert "Previously loaded sessions are still available" in status.text()
    assert page._selected_session() == initial
    assert button.isEnabled()

    def recovered(_roots):
        calls.append(get_ident())
        return [initial]

    monkeypatch.setattr(page, "_discover_review_session_entries", recovered)
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not page._review_sessions_loading)
    assert len(calls) == 3
    assert button.isEnabled() and status.text() == "Saved review sessions are up to date."
    assert page._selected_session() == initial
    assert page.training_process is page.validation_process is None


def test_lightweight_studio_without_optional_refresh_controls_still_builds(qtbot):
    page = FakeStudio()
    qtbot.addWidget(page)
    steps = NativePhaseSteps(page, actor_provider=lambda: "reviewer")
    qtbot.addWidget(steps)
    assert not hasattr(page, "refresh_review_sessions_button")
    assert not hasattr(page, "saved_sessions_status")
    assert page.saved_sessions.parentWidget() is steps.groups[4]
