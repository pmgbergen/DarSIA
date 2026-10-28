"""Prediction disclosure uses the actual review writer and first blind seal."""
import json
from pathlib import Path
from threading import Event

import pytest

from darsia.fluidflower.ui import phase_prediction_disclosure as disclosure
from .test_native_inline_phase_review import review, opened  # noqa: F401


def active(review, qtbot):
    process, window = opened(review, qtbot)
    adapter = getattr(window, "_workflow_prediction_disclosure", None)
    if adapter is None:
        adapter = disclosure.install(window, review.desktop)
        window._workflow_adapters.append(adapter)
    timeline = review.root / "disclosure-timeline.csv"
    timeline.write_text("frame_index,elapsed_h\n0,0.0\n", encoding="utf-8")
    candidate = review.desktop.review.Candidate(
        run="ac01", role="selected_model", family="saved", variant="saved",
        candidate_id="candidate-a", grid_order=0, params_sha256="a" * 64,
        normalized_mae=None, mass_plot=review.root / "unused.png",
        contour_movie=review.root / "unused.mp4", per_frame_csv=timeline,
        summary_path=review.root / "unused.json", plot_seconds=0, image_seconds=0,
        fps=1, frame_count=1, dataset="ffac")
    window.store.update_pairs([review.desktop.review.ReviewPair(
        "ac01", candidate, candidate, "single_candidate")])
    window.current_run = "ac01"
    return window, adapter


def test_original_durable_save_precedes_first_reveal_and_later_requests_reuse_snapshot(review, qtbot, monkeypatch):
    window, adapter = active(review, qtbot)
    window.notes.setPlainText("Last blind observation")
    window.annotations = [dict(id="blind-region", candidate="A", frame_index=0,
        category="confirmed_no_co2", shape="rectangle", x=.1, y=.2, width=.3,
        height=.2, points=[], confidence="high", severity="major")]
    save = window.store.save
    entered, release = Event(), Event()
    def slow_save(request):
        entered.set()
        assert release.wait(5)
        return save(request)
    monkeypatch.setattr(window.store, "save", slow_save)
    visible = []
    assert adapter.request(lambda: visible.append("obsolete-request"))
    try:
        qtbot.waitUntil(entered.is_set)
        assert not visible and not adapter.can_close()
        assert not (window.workspace.output_dir / "blind_reveal.json").exists()
        assert not window.centralWidget().isEnabled()
        assert adapter.request(lambda: visible.append(window._workflow_revealed))
    finally:
        release.set()
    qtbot.waitUntil(lambda: bool(visible), timeout=10000)
    assert visible == [True]
    qtbot.waitUntil(adapter.can_close)
    marker = disclosure.read_reveal_marker(window.workspace.output_dir)
    snapshot_path = Path(marker["frozen_blind_snapshot"])
    frozen = snapshot_path.read_bytes()
    snapshot = json.loads(frozen)
    assert snapshot["decisions"]["ac01"]["notes"] == "Last blind observation"
    assert snapshot["decisions"]["ac01"]["annotations"][0]["id"] == "blind-region"
    session = json.loads((window.workspace.output_dir / "review_session.json").read_text())
    assert session["blindness_status"] == "revealed"
    assert session["post_reveal_edits_are_blind"] is False
    monkeypatch.setattr(window.store, "save", save)
    window.notes.setPlainText("Later open correction")
    window._save_now()
    qtbot.waitUntil(lambda: not window.save_inflight and not window.save_dirty)
    assert adapter.request(lambda: visible.append("immediate"))
    assert visible[-1] == "immediate"
    restored = disclosure.install(window, review.desktop)
    assert restored.request(lambda: visible.append("reopened"))
    assert visible[-1] == "reopened" and snapshot_path.read_bytes() == frozen


def test_failed_original_save_never_seals_or_reveals_and_can_retry(review, qtbot, monkeypatch):
    window, adapter = active(review, qtbot)
    save = window.store.save
    def fail(request):
        raise OSError("fixture disk unavailable")
    monkeypatch.setattr(window.store, "save", fail)
    visible = []
    adapter.request(lambda: visible.append(True))
    qtbot.waitUntil(lambda: window.sync_error_pending)
    assert not visible and not adapter.awaiting_save
    assert not (window.workspace.output_dir / "blind_reveal.json").exists()
    assert "Phase map remains hidden" in window.statusBar().currentMessage()
    monkeypatch.setattr(window.store, "save", save)
    window._save_now()
    qtbot.waitUntil(lambda: not window.save_inflight and not window.sync_error_pending)
    adapter.request(lambda: visible.append(True))
    qtbot.waitUntil(lambda: bool(visible), timeout=10000)


def test_close_during_seal_waits_for_worker_and_suppresses_reveal(review, qtbot, monkeypatch):
    window, adapter = active(review, qtbot)
    seal = disclosure.seal_blind_evidence
    entered, release = Event(), Event()
    def slow_seal(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return seal(*args, **kwargs)
    monkeypatch.setattr(disclosure, "seal_blind_evidence", slow_seal)
    visible = []
    adapter.request(lambda: visible.append(True))
    try:
        qtbot.waitUntil(entered.is_set)
        adapter.begin_close()
        assert not adapter.can_close()
    finally:
        release.set()
    qtbot.waitUntil(adapter.can_close, timeout=10000)
    assert not visible and window._workflow_revealed
    adapter.cancel_close()
    assert window.centralWidget().isEnabled()


def test_seal_failure_is_visible_without_reveal(review, qtbot, monkeypatch):
    window, adapter = active(review, qtbot)
    def fail(*args, **kwargs):
        raise OSError("fixture snapshot unavailable")
    monkeypatch.setattr(disclosure, "seal_blind_evidence", fail)
    visible = []
    adapter.request(lambda: visible.append(True))
    qtbot.waitUntil(lambda: "snapshot unavailable" in window.statusBar().currentMessage())
    qtbot.waitUntil(adapter.can_close)
    assert not visible and not window._workflow_revealed
    assert window.centralWidget().isEnabled()


def test_invalid_existing_snapshot_blocks_new_disclosure(review, qtbot):
    window, adapter = active(review, qtbot)
    visible = []
    adapter.request(lambda: visible.append(True))
    qtbot.waitUntil(lambda: bool(visible), timeout=10000)
    marker = disclosure.read_reveal_marker(window.workspace.output_dir)
    Path(marker["frozen_blind_snapshot"]).write_text("changed", encoding="utf-8")
    restored = disclosure.install(window, review.desktop)
    assert restored.seal_invalid
    assert not restored.request(lambda: visible.append(False))
    assert visible == [True]
    assert "snapshot is missing or changed" in window.statusBar().currentMessage()


def test_metadata_failure_after_sealing_keeps_blind_answers_locked_until_retry(review, qtbot, monkeypatch):
    window, adapter = active(review, qtbot)
    writer = review.desktop.write_review_session_manifest
    def fail(*args, **kwargs):
        raise OSError("fixture metadata write unavailable")
    monkeypatch.setattr(review.desktop, "write_review_session_manifest", fail)
    visible = []
    adapter.request(lambda: visible.append(True))
    qtbot.waitUntil(lambda: "metadata write unavailable" in window.statusBar().currentMessage())
    qtbot.waitUntil(adapter.can_close)
    marker = disclosure.read_reveal_marker(window.workspace.output_dir)
    assert window._workflow_revealed and not visible and adapter.marker is None
    monkeypatch.setattr(review.desktop, "write_review_session_manifest", writer)
    adapter.request(lambda: visible.append(True))
    qtbot.waitUntil(lambda: bool(visible), timeout=10000)
    assert disclosure.read_reveal_marker(window.workspace.output_dir) == marker
