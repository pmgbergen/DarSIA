"""Confirm step: a saved step gets its checkmark only from an explicit confirmation."""

import json

import pytest

from darsia.fluidflower import workflow_status_ui as status_ui
from darsia.fluidflower.series_method import MethodSelection
from darsia.fluidflower.step_confirmations import FILE_NAME
from darsia.fluidflower.workflow_row_status import RowStatus
from darsia.fluidflower.workflow_status_ui import confirmed_status_text
from .test_native_sidebar import native_window  # noqa: F401

METHOD = ("method", "use_for_series")


def test_row_text_names_confirmed_changed_and_unconfirmed_steps():
    saved = RowStatus("done", "Saved choice.", "2026-10-27 12:00 CET")
    event = dict(action="confirm", utc="2026-10-27T11:30:00Z")
    assert confirmed_status_text(saved, ("confirmed", event)).startswith("Confirmed · 2026-10-27 12:30")
    assert confirmed_status_text(saved, ("changed", event)).startswith("Changed since confirmed")
    assert confirmed_status_text(saved, ("earlier", event)).startswith(
        "Confirmed · 2026-10-27 12:30 CET with the earlier file check · confirm once more\nSaved choice.")
    assert confirmed_status_text(saved, ("open", None)) == "Done · not confirmed · 2026-10-27 12:00 CET\nSaved choice."
    missing = RowStatus("missing", "Nothing saved.")
    assert confirmed_status_text(missing, ("open", None)) == status_ui.status_text(missing)
    assert confirmed_status_text(None, ("open", None)) == "Not confirmed"


def _method(root, active):
    text = (root / "series.toml").read_text(encoding="utf-8").split("[method]")[0]
    (root / "series.toml").write_text(text + f'[method]\nactive = "{active}"\n', encoding="utf-8")


@pytest.mark.parametrize("native_window", [("series-a",)], indirect=True)
def test_confirm_marks_the_step_until_its_saved_choice_changes(native_window, qtbot, monkeypatch):
    from darsia.fluidflower import native_finish_setup
    from .test_setup_completion_marks import _Store
    window = native_window
    root = window.phase_model_workspace.series_roots["series-a"]
    # The Setup completion reading of the phase-model method stays local.
    monkeypatch.setattr(status_ui, "setup_completion_store", _Store)
    monkeypatch.setattr(native_finish_setup, "finish_setup_status",
                        lambda root, registrations: dict(complete=False, message="No Finish setup job."))
    jobs = type("Jobs", (), dict(registry=type("Registry", (), dict(jobs=lambda self, root, series: []))(),
                                 series_id="series-a"))()
    monkeypatch.setattr(window, "ensure_jobs", lambda: jobs)
    _method(root, "phase_model")
    window._saved_method = MethodSelection(active="phase_model")
    window._select_row(METHOD)
    adapter = window.workflow_status
    adapter.refresh()
    qtbot.waitUntil(lambda: not adapter.checking, timeout=10000)
    row, text = adapter.rows[METHOD]
    assert row._completion_state == "none"
    assert text.text().startswith("Done · not confirmed")
    assert adapter.confirm_button.isVisible() and adapter.confirm_button.text() == "Confirm step"
    adapter.confirm_button.click()
    qtbot.waitUntil(lambda: row._completion_state == "done", timeout=10000)
    assert text.text().startswith("Confirmed · ")
    assert adapter.confirm_button.text() == "Undo confirmation"
    events = json.loads((root / "config" / FILE_NAME).read_text(encoding="utf-8"))["events"]
    assert [(event["step"], event["action"]) for event in events] == [("method:use_for_series", "confirm")]
    # Another saved method is not what was confirmed.
    _method(root, "colour_path")
    adapter.refresh()
    qtbot.waitUntil(lambda: not adapter.checking, timeout=10000)
    assert row._completion_state == "none" and text.text().startswith("Changed since confirmed")
    _method(root, "phase_model")
    adapter.refresh()
    qtbot.waitUntil(lambda: not adapter.checking, timeout=10000)
    assert row._completion_state == "done"
    adapter.confirm_button.click()
    qtbot.waitUntil(lambda: row._completion_state == "none", timeout=10000)
    assert text.text().startswith("Done · not confirmed")
    assert len(json.loads((root / "config" / FILE_NAME).read_text(encoding="utf-8"))["events"]) == 2


GEOMETRY = (("geometry_facies", "rig_frame"), ("geometry_facies", "layers"), ("geometry_facies", "export"))


@pytest.mark.parametrize("native_window", [("series-a",)], indirect=True)
def test_saved_geometry_steps_are_marked_by_their_saved_state(native_window, qtbot, monkeypatch):
    window = native_window
    root = window.phase_model_workspace.series_roots["series-a"]
    original = status_ui.row_status
    saved = {logical: RowStatus("done", "Saved geometry.", "2026-10-28 09:00 CET") for logical in GEOMETRY}
    monkeypatch.setattr(status_ui, "row_status",
                        lambda context, *logical: saved.get(logical) or original(context, *logical))
    adapter = window.workflow_status
    adapter.invalidate()
    adapter.refresh()
    qtbot.waitUntil(lambda: not adapter.checking, timeout=10000)
    for logical in GEOMETRY:
        row, text = adapter.rows[logical]
        assert row._completion_state == "done"
        assert text.text() == "Done · 2026-10-28 09:00 CET\nSaved geometry."
        assert adapter._confirmation(logical, saved[logical]) is None
    saved[GEOMETRY[2]] = RowStatus("missing", "Apply has not run for the saved geometry revision yet.")
    adapter.invalidate()
    adapter.refresh()
    qtbot.waitUntil(lambda: not adapter.checking, timeout=10000)
    assert adapter.rows[GEOMETRY[2]][0]._completion_state == "none"
    assert adapter.rows[GEOMETRY[0]][0]._completion_state == "done"
    assert not (root / "config" / FILE_NAME).exists()


@pytest.mark.parametrize("native_window", [("series-a",)], indirect=True)
def test_a_changed_series_file_refreshes_marks_and_rereads_setup(native_window, qtbot, monkeypatch):
    window = native_window
    adapter = window.workflow_status
    key = adapter._series_key()
    # Let the window's own first check finish, then start from no noted state.
    reader = adapter.__dict__.get("_change_reader")
    if reader is not None:
        qtbot.waitUntil(lambda: not reader.busy, timeout=10000)
    adapter._watched.clear()
    calls = []
    monkeypatch.setattr(window, "refresh_sidebar_progress", lambda: calls.append(True))
    adapter._setup_completion = {key: (0.0, dict(complete=True))}
    adapter._changes_read((key, ((1, 1), None, None), {}), "")
    assert not calls and key in adapter._setup_completion
    adapter._changes_read((key, ((1, 2), None, None), {}), "")
    assert calls and key not in adapter._setup_completion
    calls.clear()
    adapter._changes_read((key, ((1, 2), None, None), {"a:b": dict(action="confirm")}), "")
    assert calls
