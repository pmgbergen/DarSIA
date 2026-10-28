"""The version library and the reference point check are read off the GUI thread.

On a series share both take seconds. A quick reading answers at once; a slow
one answers later while the window stays usable. A synthetic project in a
temporary folder; no real series is used.
"""
import threading
import time
from types import SimpleNamespace

from darsia.fluidflower import background_read
from darsia.fluidflower import layer_painting_version_archive as archive
from darsia.fluidflower import rig_point_review
from darsia.fluidflower import series_layer_view as view_module
from darsia.fluidflower.background_read import BackgroundRead

from .test_painting_corrections import _proposal, _show
from .test_painting_photo_saturation import make_view, settings, view  # noqa: F401


def _slow(monkeypatch, module, name):
    """Make module.name wait for the returned event, and the window not wait for it."""
    release, calls = threading.Event(), []
    original = getattr(module, name)

    def wait_then_answer(*args, **kwargs):
        calls.append(args)
        assert release.wait(20)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, wait_then_answer)
    monkeypatch.setattr(background_read, "QUICK_ANSWER_S", 0.0)
    return release, calls


def test_a_quick_reading_answers_at_once_and_a_slow_one_later(qtbot):
    reader = BackgroundRead()
    answers = []
    reader.answered.connect(lambda key, value, error: answers.append((key, value, error)))
    assert reader.read("quick", lambda: 41 + 1, wait_s=10) == (True, 42, "")
    assert reader.read("fails", lambda: 1 / 0, wait_s=10) == (True, None, "ZeroDivisionError: division by zero")
    release = threading.Event()
    assert reader.read("first", lambda: release.wait(20) and "old", wait_s=0) == (False, None, "")
    assert reader.busy() and reader.key == "first"
    assert reader.read("second", lambda: release.wait(20) and "new", wait_s=0) == (False, None, "")
    release.set()
    qtbot.waitUntil(lambda: not reader.busy(), timeout=10000)
    qtbot.wait(50)
    # Only the latest reading answers.
    assert answers == [("second", "new", "")]


def test_a_slow_version_reading_keeps_the_rows_and_answers_later(view, qtbot, monkeypatch):
    _show(view, _proposal())
    first = view.save_version("Proposal only")
    view.versions_table.selectRow(0)
    assert view._selected_version()["folder"] == first["folder"]
    release, _calls = _slow(monkeypatch, archive, "list_version_metadata")
    started = time.monotonic()
    rows = view._read_versions(view.session.run_id)
    assert time.monotonic() - started < 2.0
    assert rows == view.versions and view.versions_reading()
    assert view.versions_note.text() == "Reading the saved versions…"
    assert view.adoption_controller._chart_readers_busy()
    # Restoring waits: whether the current map is saved is known only after reading.
    assert not view.restore_version(first)
    release.set()
    qtbot.waitUntil(lambda: not view.versions_reading(), timeout=10000)
    assert [row["folder"] for row in view.versions] == [first["folder"]]
    assert view._selected_version()["folder"] == first["folder"]
    assert view.versions_note.text().startswith("Selected version · Proposal only")


def test_the_remembered_version_survives_a_slow_first_reading(view, qtbot, monkeypatch):
    _show(view, _proposal())
    first = view.save_version("Proposal only")
    view.versions_table.selectRow(0)
    key = view._version_selection_key()
    release, _calls = _slow(monkeypatch, archive, "list_version_metadata")
    view.versions = []
    view._versions_shown = None
    view._read_versions(view.session.run_id)
    view._refresh_versions()
    # No row is shown yet, and the remembered choice is not forgotten.
    assert view.versions_table.rowCount() == 0 and view._selected_version() is None
    assert view_module.QSettings(*view_module._BACKGROUND_SETTINGS).value(key)
    release.set()
    qtbot.waitUntil(lambda: not view.versions_reading(), timeout=10000)
    assert view._selected_version()["folder"] == first["folder"]


def test_arriving_versions_keep_the_transfer_review_of_the_series_map_route(view, monkeypatch):
    _show(view, _proposal())
    view.save_version("Proposal only")
    calls = []
    monkeypatch.setattr(view, "show_mark_layers", lambda *args: calls.append(True))
    view._workflow_scope = "apply"
    view._versions_arriving = True
    view._version_preview_identity = None
    view.versions_table.selectRow(0)
    view._version_selected()
    assert calls == []
    view._versions_arriving = False
    view._version_preview_identity = None
    view._version_selected()
    assert calls == [True]


def test_a_waiting_automatic_chart_review_is_renewed_for_the_same_run(view, monkeypatch):
    controller = view.adoption_controller
    queued = []
    monkeypatch.setattr(controller, "_queue_automatic_chart_review", lambda *args: queued.append(True))
    controller._automatic_chart_request = None
    controller.background_reading_answered()
    assert queued == []
    current = controller._automatic_chart_identity()
    controller._automatic_chart_request = (*current[:3], "a version not yet selected")
    controller.background_reading_answered()
    assert controller._automatic_chart_request == current and queued == [True]
    controller._automatic_chart_request = (current[0], "another project", *current[2:])
    controller.background_reading_answered()
    assert queued == [True]


def test_the_reference_point_check_answers_later_and_is_shown_again(view, qtbot, monkeypatch):
    document = {"runs": {"ac52": {"photo": {"path": "photo.png"}}}}
    frame = SimpleNamespace(to_dict=lambda: {"run_id": "ac52"})
    gates, answers, calls = [threading.Event(), threading.Event()], ["ac52: photograph changed", None], []

    def review_blocker(*args):
        index = len(calls)
        calls.append(args)
        assert gates[index].wait(20)
        return answers[index]

    monkeypatch.setattr(rig_point_review, "review_blocker", review_blocker)
    monkeypatch.setattr(background_read, "QUICK_ANSWER_S", 0.0)
    refreshed = []
    monkeypatch.setattr(view, "_refresh_versions", lambda: refreshed.append(True))
    assert view._point_review_problem(document, frame, ["ac52"]) == "Checking the reference point review…"
    # The same check is not started twice.
    assert view._point_review_problem(document, frame, ["ac52"]) == "Checking the reference point review…"
    gates[0].set()
    qtbot.waitUntil(lambda: bool(refreshed), timeout=10000)
    assert view._point_review_problem(document, frame, ["ac52"]) == "ac52: photograph changed"
    assert len(calls) == 1
    # After a while the answer is shown while it is checked again.
    key, _when, problem, slow = view._point_review_last
    view._point_review_last = (key, time.monotonic() - 60, problem, slow)
    assert view._point_review_problem(document, frame, ["ac52"]) == "ac52: photograph changed"
    gates[1].set()
    qtbot.waitUntil(lambda: len(refreshed) == 2, timeout=10000)
    assert len(calls) == 2 and view._point_review_problem(document, frame, ["ac52"]) is None


def test_a_quick_reference_point_check_is_read_again_each_time(view, monkeypatch):
    document = {"runs": {"ac52": {"photo": {"path": "photo.png"}}}}
    frame = SimpleNamespace(to_dict=lambda: {"run_id": "ac52"})
    calls = []
    monkeypatch.setattr(rig_point_review, "review_blocker", lambda *args: calls.append(args) or "not approved")
    assert view._point_review_problem(document, frame, ["ac52"]) == "not approved"
    assert view._point_review_problem(document, frame, ["ac52"]) == "not approved"
    assert len(calls) == 2
