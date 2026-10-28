"""The sidebar rows' saved status is read in the background and shown from the last answer."""
import threading

import pytest

from darsia.fluidflower import workflow_status_ui as status_ui
from darsia.fluidflower.workflow_row_status import RowStatus
from .test_native_sidebar import native_window  # noqa: F401

METHOD = ("method", "use_for_series")


@pytest.mark.parametrize("native_window", [("series-a",)], indirect=True)
def test_rows_are_read_off_the_gui_thread_and_kept_between_clicks(native_window, qtbot, monkeypatch):
    window = native_window
    adapter = window.workflow_status
    calls, release = [], threading.Event()
    original = status_ui.row_status

    def row_status(context, action, item_id):
        calls.append(threading.current_thread() is threading.main_thread())
        assert release.wait(10)
        return original(context, action, item_id)

    monkeypatch.setattr(status_ui, "row_status", row_status)
    window._select_row(METHOD)
    adapter.invalidate()
    adapter.refresh()  # Returns at once; the reading waits for release.
    assert adapter.checking
    row, text = adapter.rows[METHOD]
    assert text.text() in {status_ui.CHECKING_TEXT} or adapter.states.get(METHOD) is not None
    release.set()
    qtbot.waitUntil(lambda: not adapter.checking, timeout=10000)
    assert calls and not any(calls)  # Every row was read off the GUI thread.
    assert adapter.states[METHOD].state in {"done", "missing", "saved_choice", "default_choice"}
    count = len(calls)
    for _ in range(3):  # Clicks show the kept answer and read nothing.
        window._select_row(METHOD)
        adapter.refresh()
    assert len(calls) == count and not adapter.checking
    window.refresh_sidebar_progress()  # A load, a save or a change seen elsewhere reads anew.
    assert adapter.checking
    qtbot.waitUntil(lambda: not adapter.checking, timeout=10000)
    assert len(calls) > count


@pytest.mark.parametrize("native_window", [("series-a",)], indirect=True)
def test_an_unreadable_row_says_so_and_the_others_still_show(native_window, qtbot, monkeypatch):
    window = native_window
    adapter = window.workflow_status
    original = status_ui.row_status

    def row_status(context, action, item_id):
        if (action, item_id) == METHOD:
            raise OSError("share unavailable")
        return original(context, action, item_id)

    monkeypatch.setattr(status_ui, "row_status", row_status)
    window._select_row(METHOD)
    adapter.invalidate()
    adapter.refresh()
    qtbot.waitUntil(lambda: not adapter.checking, timeout=10000)
    state = adapter.states[METHOD]
    assert isinstance(state, RowStatus) and state.state == "missing"
    assert "Status could not be read: share unavailable" in state.summary
    assert len(adapter.states) > 1
