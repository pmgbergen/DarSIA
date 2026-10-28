"""Setup rows completed by the Finish setup job: checking, done, unavailable and restored marks."""

import threading

import pytest

from darsia.fluidflower import native_finish_setup
from darsia.fluidflower import workflow_status_ui as status_ui
from darsia.fluidflower.series_method import MethodSelection
from darsia.fluidflower.workflow_status_ui import restored_setup_completion, setup_completion_display
from .test_native_sidebar import native_window  # noqa: F401


def test_display_names_checking_done_unavailable_and_not_done():
    assert setup_completion_display(None) == ("none", "", "")
    assert setup_completion_display(None, checking=True) == ("none", "Checking the Finish setup job…", "")
    state, line, tooltip = setup_completion_display(dict(error="OSError: share unavailable"))
    assert (state, line) == ("error", "Status unavailable") and "share unavailable" in tooltip
    done = dict(complete=True, message="Setup complete.", checked="2026-10-27 13:05 CET")
    assert setup_completion_display(done) == ("done", "Done · checked 2026-10-27 13:05 CET", "Setup complete.")
    restored = dict(done, restored=True)
    assert setup_completion_display(restored, checking=True)[1].endswith(" · checking again…")
    state, line, _tooltip = setup_completion_display(dict(complete=False, message="Geometry changed."))
    assert state == "none" and line == "Not done\nGeometry changed."


def test_restored_reading_requires_the_same_series_bytes():
    import hashlib
    import json
    stored = json.dumps(dict(series_sha256=hashlib.sha256(b"series").hexdigest(), complete=True,
                             message="Setup complete.", checked="2026-10-27 13:05 CET"))
    restored = restored_setup_completion(stored, b"series")
    assert restored == dict(complete=True, message="Setup complete.", checked="2026-10-27 13:05 CET", restored=True)
    assert restored_setup_completion(stored, b"changed series") is None
    assert restored_setup_completion("not json", b"series") is None
    assert restored_setup_completion("", b"series") is None


class _Store:
    def __init__(self):
        self.values = {}

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value


def _completion_rows(window):
    rows = window.workflow_status.setup_rows
    assert set(rows) == set(status_ui.SETUP_COMPLETION_ROWS)
    return rows.values()


@pytest.mark.parametrize("native_window", [("series-a",)], indirect=True)
def test_rows_show_checking_then_done_and_restore_after_restart(native_window, qtbot, monkeypatch):
    window = native_window
    store = _Store()
    monkeypatch.setattr(status_ui, "setup_completion_store", lambda: store)
    window._saved_method = MethodSelection(active="phase_model")
    raw = window._series_snapshot[1]["raw"]
    release = threading.Event()

    def finish_setup_status(root, registrations):
        assert release.wait(10)
        return dict(raw=raw, complete=True, message="Setup complete for all runs.")

    monkeypatch.setattr(native_finish_setup, "finish_setup_status", finish_setup_status)
    jobs = type("Jobs", (), dict(registry=type("Registry", (), dict(jobs=lambda self, root, series: []))(),
                                 series_id="series-a"))()
    monkeypatch.setattr(window, "ensure_jobs", lambda: jobs)
    window.workflow_status.refresh()
    for row, text in _completion_rows(window):
        assert row._completion_state == "none"
        assert text.text() == "Checking the Finish setup job…"
    release.set()
    for row, text in _completion_rows(window):
        qtbot.waitUntil(lambda row=row: row._completion_state == "done", timeout=10000)
        assert text.text().startswith("Done · checked ")
        assert text.toolTip() == "Setup complete for all runs."
    assert len(store.values) == 1
    # A restart starts without this session's reading: the stored one shows at
    # once while the same series.toml bytes are checked again.
    window.workflow_status._setup_completion.clear()
    window.workflow_status._setup_completion_restored.clear()
    release.clear()
    window.workflow_status.refresh()
    for row, text in _completion_rows(window):
        assert row._completion_state == "done"
        assert text.text().endswith(" · checking again…")
    release.set()
    for row, text in _completion_rows(window):
        qtbot.waitUntil(lambda text=text: not text.text().endswith("…"), timeout=10000)


@pytest.mark.parametrize("native_window", [("series-a",)], indirect=True)
def test_failed_reading_is_unavailable_not_undone(native_window, qtbot, monkeypatch):
    window = native_window
    monkeypatch.setattr(status_ui, "setup_completion_store", _Store)
    window._saved_method = MethodSelection(active="phase_model")

    def finish_setup_status(root, registrations):
        raise OSError("share unavailable")

    monkeypatch.setattr(native_finish_setup, "finish_setup_status", finish_setup_status)
    jobs = type("Jobs", (), dict(registry=type("Registry", (), dict(jobs=lambda self, root, series: []))(),
                                 series_id="series-a"))()
    monkeypatch.setattr(window, "ensure_jobs", lambda: jobs)
    window.workflow_status.refresh()
    for row, text in _completion_rows(window):
        qtbot.waitUntil(lambda row=row: row._completion_state == "error", timeout=10000)
        assert text.text() == "Status unavailable"
        assert "share unavailable" in text.toolTip()
