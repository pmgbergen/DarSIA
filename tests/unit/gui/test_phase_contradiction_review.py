"""Unrun source-phase tests of the original asynchronous save boundary."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
from darsia.fluidflower.ui.phase_contradiction_review import PendingCorrection, CalibrationContradictionDock


def pending():
    return PendingCorrection("ac01", 4, dict(id="human-revision", category="confirmed_no_co2",
        frame_index=3, shape="rectangle", points=[], x=.1, y=.2, width=.3, height=.4,
        evidence_audit=dict(original_region_id="ac01:manual:original", queue_sha256="queue", verdict="reclassified")))


def test_only_matching_durable_async_result_completes_correction():
    state = pending()
    record = dict(run="ac01", annotations=[deepcopy(state.annotation)])
    assert not state.matches(3, "ac01", record, "")
    assert not state.matches(4, "ac02", record, "")
    assert not state.matches(4, "ac01", record, "disk full")
    assert not state.matches(4, "ac01", dict(run="ac01", annotations=[]), "")
    assert state.matches(4, "ac01", record, "")
    assert state.matches(5, "ac01", record, "")  # Original recovery retry.


@pytest.mark.parametrize("field,value", [("id", "other"), ("category", "confirmed_only_free_gas"),
    ("x", .2), ("frame_index", 4), ("evidence_audit", {"queue_sha256": "different"})])
def test_changed_saved_payload_cannot_complete_old_correction(field, value):
    state = pending()
    changed = dict(state.annotation, **{field: value})
    assert not state.matches(4, "ac01", dict(run="ac01", annotations=[changed]), "")


def test_close_waits_for_actual_worker_and_pending_correction():
    state = SimpleNamespace(task=None, pending=None, pool=SimpleNamespace(activeThreadCount=lambda: 0))
    assert CalibrationContradictionDock.can_close(state)
    state.pending = pending()
    assert not CalibrationContradictionDock.can_close(state)
    state.pending, state.task = None, object()
    assert not CalibrationContradictionDock.can_close(state)
    state.task = None
    state.pool.activeThreadCount = lambda: 1
    assert not CalibrationContradictionDock.can_close(state)


def test_close_cancels_pending_launch_and_can_reopen_on_save_failure():
    enabled = []
    window = SimpleNamespace(_workflow_sealing=True, _workflow_closing=False,
        centralWidget=lambda: SimpleNamespace(setEnabled=enabled.append))
    state = SimpleNamespace(closing=False, awaiting_save=True, pending=None, task=None,
        seal_invalid=False, _controls=lambda: None, _refresh_peer_controls=lambda: None,
        window=window, _central_was_enabled=True)
    state._unlock = lambda: CalibrationContradictionDock._unlock(state)
    CalibrationContradictionDock.begin_close(state)
    assert state.closing and not state.awaiting_save
    assert window._workflow_sealing is False
    CalibrationContradictionDock.cancel_close(state)
    assert not state.closing
    assert enabled == [True]


def test_cancel_close_keeps_live_seal_worker_and_invalid_marker_locked():
    enabled = []
    window = SimpleNamespace(_workflow_sealing=True,
        centralWidget=lambda: SimpleNamespace(setEnabled=enabled.append))
    state = SimpleNamespace(closing=False, awaiting_save=False, pending=None, task=object(),
        seal_invalid=False, _controls=lambda: None, window=window)
    CalibrationContradictionDock.begin_close(state)
    CalibrationContradictionDock.cancel_close(state)
    assert window._workflow_sealing is True and enabled == [False]
    state.task, state.seal_invalid = None, True
    CalibrationContradictionDock.begin_close(state)
    CalibrationContradictionDock.cancel_close(state)
    assert window._workflow_sealing is True and enabled == [False, False]


@pytest.mark.parametrize("inflight,queued,dirty,error", [(True, {}, False, False),
    (False, {"ac01": {}}, False, False), (False, {}, True, False), (False, {}, False, True)])
def test_evaluation_does_not_launch_before_original_save_barrier_drains(inflight, queued, dirty, error, monkeypatch):
    from darsia.fluidflower.ui import phase_contradiction_review as module
    def forbidden(**kwargs):
        raise AssertionError("Evaluation was started before durable save completion")
    monkeypatch.setattr(module, "_EvaluationTask", forbidden)
    state = SimpleNamespace(awaiting_save=True, closing=False, window=SimpleNamespace(
        save_inflight=inflight, pending_save_requests=queued, save_dirty=dirty, sync_error_pending=error))
    CalibrationContradictionDock.saved_complete(state)
    assert state.awaiting_save


def test_install_restores_reveal_guard_before_editable_window(tmp_path, monkeypatch):
    from darsia.fluidflower.ui import phase_contradiction_review as module
    class Window:
        workspace = SimpleNamespace(output_dir=tmp_path)
    calls = []
    monkeypatch.setattr(module, "read", lambda path: dict(blindness_status="revealed"))
    monkeypatch.setattr(module, "read_reveal_marker", lambda root: {"valid": "marker"})
    monkeypatch.setattr(module, "CalibrationContradictionDock", lambda w, d: SimpleNamespace(_sealed=calls.append))
    module.install(Window(), SimpleNamespace(CalibrationReviewWindow=Window))
    assert calls == [{"valid": "marker"}]


def test_install_refuses_invalid_or_missing_declared_reveal_before_constructing_dock(tmp_path, monkeypatch):
    from darsia.fluidflower.ui import phase_contradiction_review as module
    class Window:
        workspace = SimpleNamespace(output_dir=tmp_path)
    monkeypatch.setattr(module, "read", lambda path: dict(blindness_status="revealed"))
    def invalid(root):
        raise ValueError("sealed snapshot changed")
    def forbidden(*args):
        raise AssertionError("Editable dock created for an invalid reveal")
    monkeypatch.setattr(module, "read_reveal_marker", invalid)
    monkeypatch.setattr(module, "CalibrationContradictionDock", forbidden)
    with pytest.raises(ValueError, match="snapshot changed"):
        module.install(Window(), SimpleNamespace(CalibrationReviewWindow=Window))
