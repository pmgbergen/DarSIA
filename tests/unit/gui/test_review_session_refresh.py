"""Saved-session refresh keeps the actual Qt controls responsive on slow storage."""
from threading import Event, get_ident

import pytest
from PySide6.QtCore import QSettings, QTimer, Qt

from darsia.fluidflower.ui import model_training as ui


@pytest.fixture
def studio(qtbot, monkeypatch, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(ui, "QSettings", lambda *_args: settings)
    monkeypatch.setattr(ui, "configured_series", lambda: ())
    monkeypatch.setattr(ui, "default_phase_review_roots", lambda: (tmp_path / "reviews",))
    monkeypatch.setattr(ui, "default_project_root", lambda: tmp_path / "project")
    monkeypatch.setenv("DARSIA_FLUIDFLOWER_PHASE_MODEL_LIBRARY", str(tmp_path / "models"))
    monkeypatch.setattr(ui, "local_cache_root", lambda *_args: tmp_path / "cache")
    monkeypatch.setattr(ui, "default_execution_cache_root", lambda: tmp_path / "execution")
    monkeypatch.setattr(ui.SharedPhaseModelTrainingPage, "_request_async_model_refresh", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ui.SharedPhaseModelTrainingPage, "_load_pending_evidence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ui.SharedPhaseModelTrainingPage, "_load_contributions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ui.SharedPhaseModelTrainingPage, "_session_matches_selected_model", lambda *_args: True)
    initial = dict(session_path=str(tmp_path / "saved-session.json"),
                   review_session_id="saved-review", manual_regions=1, modified_timestamp=1,
                   reviewer="reviewer", target_model_id="saved-model")
    monkeypatch.setattr(ui.SharedPhaseModelTrainingPage, "_discover_review_session_entries",
                        staticmethod(lambda _roots: [initial]))
    page = ui.SharedPhaseModelTrainingPage()
    qtbot.addWidget(page)
    for timer in page.findChildren(QTimer):
        timer.stop()
    qtbot.waitUntil(lambda: not page._review_sessions_loading)
    assert page.saved_sessions.count() == 1
    return page, initial


def test_refresh_button_returns_while_storage_waits_and_keeps_selection(
    studio, qtbot, monkeypatch, tmp_path, record_property,
):
    page, initial = studio
    entered, release = Event(), Event()
    calls = []
    next_session = dict(initial, session_path=initial["session_path"] + ".new",
                        review_session_id="new-review", modified_timestamp=2)

    def slow_discovery(roots):
        calls.append((get_ident(), roots))
        entered.set()
        assert release.wait(5)
        return [next_session, initial]

    monkeypatch.setattr(page, "_discover_review_session_entries", slow_discovery)
    page._transfer_review_roots = ["private-review-folder"]
    page.resize(1780, 950)
    page.tabs.setCurrentWidget(page.evidence_tab)
    page.show()
    page.evidence_scroll.ensureWidgetVisible(page.refresh_review_sessions_button)
    main_thread = get_ident()
    try:
        assert page.refresh_review_sessions_button.text() == "Refresh saved review sessions"
        qtbot.mouseClick(page.refresh_review_sessions_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(entered.is_set)
        assert page._review_sessions_loading
        assert page.width() == 1780
        assert not page.refresh_review_sessions_button.isEnabled()
        assert "Loading saved review sessions" in page.saved_sessions_status.text()
        assert page._selected_session() == initial
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        qtbot.waitUntil(lambda: bool(ticks))
        screenshot = tmp_path / "saved_review_sessions_loading.png"
        assert page.grab().save(str(screenshot))
        record_property("private_gui_screenshot", str(screenshot))
        page._refresh_review_sessions()
        assert len(calls) == 1
    finally:
        release.set()
        qtbot.waitUntil(lambda: not page._review_sessions_loading)
    assert len(calls) == 1
    assert calls[0][1] == ("private-review-folder",)
    assert calls[0][0] != main_thread
    assert page.refresh_review_sessions_button.isEnabled()
    assert page.review_sessions == [next_session, initial]
    assert page._selected_session() == initial
    assert page.saved_sessions_status.text() == "Saved review sessions are up to date."


def test_storage_error_is_visible_retains_sessions_and_allows_retry(studio, qtbot, monkeypatch):
    page, initial = studio

    def unavailable(_roots):
        raise OSError("Saved review folder is unavailable")

    monkeypatch.setattr(page, "_discover_review_session_entries", unavailable)
    page.catalog_loading_status.hide()
    page.refresh_review_sessions_button.click()
    qtbot.waitUntil(lambda: not page._review_sessions_loading)
    assert page.saved_sessions.count() == 1
    assert page._selected_session() == initial
    assert "OSError: Saved review folder is unavailable" in page.saved_sessions_status.text()
    assert "Previously loaded sessions are still available" in page.saved_sessions_status.text()
    assert not page.saved_sessions_status.isHidden()
    assert page.refresh_review_sessions_button.isEnabled()
    monkeypatch.setattr(page, "_discover_review_session_entries", lambda _roots: [initial])
    page.refresh_review_sessions_button.click()
    qtbot.waitUntil(lambda: not page._review_sessions_loading)
    assert page._selected_session() == initial
    assert page.saved_sessions_status.text() == "Saved review sessions are up to date."
