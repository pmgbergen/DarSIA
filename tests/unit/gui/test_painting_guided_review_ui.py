"""Guided presentation around the original reader and human-event boundaries.

The finite draft spy records UI strokes only.  It never creates an approval,
proof, saved target, or successful typed-validator result.  Those contracts
are qualified by the original backend and controller integration cases.
"""
from copy import deepcopy
import threading

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QTimer

from darsia.fluidflower.painting_deviation_step_ui import DeviationPaintingStep
from darsia.fluidflower.painting_guided_review_ui import GuidedGapReviewDialog

from .test_painting_gap_review_ui import _Backend, _review
from .test_painting_deviation_step_ui import CHOICES, DraftSpy


class GuidedBackend(_Backend):
    def __init__(self):
        super().__init__()
        self.drafts = []
        self.comparisons = []
        for packet in self.packets.values():
            packet.update(image_role="current", display_grid=dict(
                format="synthetic-bound-native-grid", shape_hw=list(packet["gap_mask"].shape),
                native_to_photo_h=packet["native_to_photo_h"].tolist()),
                image_choices=[dict(image_id="clearwater", available=True)])

    def draft(self, review, packet):
        assert review["review_binding"]["sha256"] == "f" * 64
        draft = DraftSpy(packet["gap_mask"])
        self.drafts.append(draft)
        return draft

    def comparison(self, review, run, component, role):
        self.comparisons.append((run, component, role,
                                 threading.current_thread() is threading.main_thread()))
        return dict(self.packets[run], component_id=component, image_role=role,
                    image_rgb=np.full((64, 64, 3), 80, np.uint8),
                    valid_support=np.ones((64, 64), bool))

    def approve(self, *_args, **_options):
        raise AssertionError("The guided partial-region path must never call whole-area approval")


def _bound_review(*, gaps=True, source_water=True):
    review = _review()
    review["review_binding"] = dict(path="C:/synthetic/gap_review.json",
        sha256="f" * 64, size_bytes=1234)
    if not gaps:
        for run in review["runs"]:
            run["components"] = []
            run["gap"]["pixels"] = 0
    if not source_water:
        review["source"].pop("painting_facies")
    return review


def _settle(qtbot, widget):
    qtbot.waitUntil(lambda: not widget.busy() and widget._pending is None,
                    timeout=5000)


def _dialog(qtbot, *, backend=None, review=None, editor=True, actor="Synthetic reviewer",
            draft_factory=None,
            **options):
    backend = backend or GuidedBackend()
    wiring = dict(deviation_editor_factory=DeviationPaintingStep,
                  region_draft_factory=draft_factory or backend.draft,
                  class_choices=CHOICES, sand_class="sand") if editor else {}
    widget = GuidedGapReviewDialog(review if review is not None else _bound_review(),
        actor=actor, review_reader=backend.read, preview_reader=backend.preview,
        comparison_reader=backend.comparison, approval_builder=backend.approve,
        **wiring, **options)
    qtbot.addWidget(widget, before_close_func=lambda owned: owned.wait_for_workers())
    widget.resize(1100, 850)
    widget.show()
    _settle(qtbot, widget)
    return widget, backend


def _stage(qtbot, widget, stage, occurrence=0):
    index = [i for i, (kind, _) in enumerate(widget._questions) if kind == stage][occurrence]
    widget._show_question(index)
    _settle(qtbot, widget)


def _mark(widget, class_name="air"):
    draft = widget._deviation_draft
    row, column = np.argwhere(draft.editable_mask)[0]
    assert draft.paint(kind="brush", points_xy_native=[[int(column), int(row)]],
                       class_name=class_name, radius_px=1)
    widget.deviation_editor._buttons()
    return draft, [int(column), int(row)]


def test_owned_catalogue_result_reaches_original_consumer_after_retirement(qtbot):
    received = []
    widget, _ = _dialog(qtbot, additional_result_consumer=lambda key, result:
        received.append((key, result, threading.current_thread() is threading.main_thread(),
                         widget.busy())) or True)
    # Retained controls existed both directly in the old root layout and in
    # nested layouts. They must survive the old wrappers' actual deferred delete.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    widget.reason.setText("The original controls survived the layout change")
    widget._update_coverage()
    widget._buttons()
    assert widget.reason.text() == "The original controls survived the layout change"
    assert widget.coverage_note.text() == "" and widget.guided_stage == "deviation"
    key, result = ("findings-catalogue", 7), {"generation": 7, "synthetic_payload": "reader result only"}
    widget._start(key, lambda: result)
    _settle(qtbot, widget)
    assert received == [(key, result, True, False)]
    assert widget.approval is None and widget._candidate is None


def test_changed_display_choices_keep_original_refresh_and_check_lock(qtbot):
    widget, _ = _dialog(qtbot, review=_bound_review(gaps=False, source_water=False))
    _stage(qtbot, widget, "check")
    qtbot.wait(1)  # Deliver layout teardown's deleteLater events.
    assert widget.recheck.isEnabled()
    starts, refreshes = [], []
    widget.recheck_requested.connect(starts.append)
    widget.refresh_findings_requested.connect(lambda: refreshes.append(True))
    widget._display_preferences_dirty = True
    widget._buttons()
    assert widget.refresh_findings.isVisible() and widget.refresh_findings.isEnabled()
    assert not widget.recheck.isEnabled()
    widget._recheck()
    assert starts == []
    widget.refresh_findings.click()
    assert refreshes == [True] and widget.operation_waiting
    widget.catalogue_refresh_failed("synthetic reader failure")
    widget._buttons()
    assert widget.coverage_note.text() == "" and widget.refresh_findings.isEnabled()
    assert not widget.can_recheck() and starts == []


@pytest.mark.parametrize("editor", [False, True])
def test_construction_navigation_and_missing_protocol_never_make_a_choice(qtbot, editor):
    widget, backend = _dialog(qtbot, editor=editor)
    assert [kind for kind, _ in widget._questions] == [
        "deviation", "deviation", "source_water", "check", "adopt"]
    assert widget.guided_stage == "deviation" and "AC18" in widget.question_title.text()
    assert widget.reason.isVisible() and widget.other_text.isVisible()
    assert not widget.technical_details.isVisible()
    assert not widget.runs.isVisible() and not widget.approve_one.isVisible()
    assert not widget.approve_all.isVisible() and not widget.paint.isVisible()
    assert "No frame or position" in widget.skipped_steps_note.text()
    assert "No other warnings" in widget.skipped_steps_note.text()
    assert "sha256" not in widget.question_explanation.text()
    signals = []
    widget.region_confirmation_requested.connect(signals.append)
    widget.recheck_requested.connect(signals.append)
    widget.adopt_requested.connect(signals.append)
    for _ in range(4):
        widget.next_button.click()
        _settle(qtbot, widget)
    assert widget.guided_stage == "adopt" and not widget.adopt.isVisible()
    assert signals == [] and widget.approval is None
    assert widget._acquisition_actions == [] and widget._warning_actions == {}
    assert backend.approvals == [] and widget._candidate is None
    if not editor:
        _stage(qtbot, widget, "deviation")
        assert "not connected" in widget.region_unavailable.text()
        assert widget.region_unavailable.isVisible() and widget.deviation_editor is None


def test_reasons_and_other_description_belong_to_the_actual_question(qtbot):
    widget, _ = _dialog(qtbot)
    widget.reason.setText("first uncovered edge")
    widget.other_text.setText("first visible cable")
    _stage(qtbot, widget, "deviation", 1)
    assert widget.reason.text() != "first uncovered edge" and widget.other_text.text() == ""
    widget.reason.setText("second thin edge")
    widget.other_text.setText("second corner")
    _stage(qtbot, widget, "source_water")
    assert "water state" in widget.reason.text() and widget.source_state.currentData() is None
    assert widget.source_state.isVisible() and widget.set_state.isVisible()
    widget.reason.setText("source sand inspected")
    _stage(qtbot, widget, "deviation")
    assert widget.reason.text() == "first uncovered edge"
    assert widget.other_text.text() == "first visible cable"
    _stage(qtbot, widget, "deviation", 1)
    assert widget.reason.text() == "second thin edge" and widget.other_text.text() == "second corner"
    _stage(qtbot, widget, "source_water")
    assert widget.reason.text() == "source sand inspected"
    assert widget.approval is None and widget._acquisition_actions == []


def test_only_real_continue_freezes_exact_human_request_and_still_does_not_approve(qtbot):
    collected = []
    def collect(snapshot, *, review):
        collected.append((deepcopy(snapshot), deepcopy(review)))
        return snapshot  # UI seam only; this is not a typed proof or approval.
    widget, _ = _dialog(qtbot, region_snapshot_collector=collect)
    received = []
    widget.region_confirmation_requested.connect(received.append)
    draft, _ = _mark(widget)
    widget.reason.setText("  inspected this uncovered edge  ")
    widget.other_text.setText("  cable beside the frame  ")
    assert received == [] and widget.approval is None
    widget.deviation_editor.continue_button.click()
    assert len(received) == len(collected) == 1
    request = received[0]
    assert request["review_binding"] == widget.review["review_binding"]
    assert (request["run_id"], request["component_id"], request["review_row_sha256"]) == (
        "ac18", "c" * 64, "c" * 64)
    assert request["actor"] == "Synthetic reviewer" and request["approved_utc"]
    assert request["reason"] == "inspected this uncovered edge"
    assert request["other_text"] == "cable beside the frame" and request["decision_ref"]
    assert request["previous"] is None
    assert collected[0][1] == widget.review
    request["snapshot"]["class_masks"]["air"][:] = False
    request["reason"] = "mutable receiver"
    assert draft.class_masks["air"].any()
    assert widget._region_submission[1]["snapshot"]["class_masks"]["air"].any()
    assert widget._region_submission[1]["reason"] == "inspected this uncovered edge"
    assert widget.approval is None and widget._candidate is None
    assert not widget._resolved(widget._current_row()) and not widget.recheck.isEnabled()


@pytest.mark.parametrize("missing", ["actor", "reason", "collector"])
def test_missing_human_identity_or_rejected_snapshot_cannot_request_region_save(qtbot, missing):
    def refuse(snapshot, *, review):
        raise ValueError("changed bound row")
    options = dict(region_snapshot_collector=refuse) if missing == "collector" else {}
    widget, _ = _dialog(qtbot, **options)
    requests = []
    widget.region_confirmation_requested.connect(requests.append)
    _mark(widget)
    if missing == "actor":
        widget.actor = ""
    elif missing == "reason":
        widget.reason.setText(" ")
    widget.deviation_editor.continue_button.click()
    assert requests == [] and widget._region_submission is None
    assert widget.approval is None and not widget.can_adopt()


def test_sand_requests_exact_original_map_fix_once_and_never_region_exclusion(qtbot):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Sand must not be submitted as an exclusion snapshot")
    widget, _ = _dialog(qtbot, region_snapshot_collector=forbidden)
    fixes, painted, region = [], [], []
    widget.map_fix_requested.connect(fixes.append)
    widget.paint_requested.connect(painted.append)
    widget.region_confirmation_requested.connect(region.append)
    _, native_point = _mark(widget, "sand")
    widget.deviation_editor.continue_button.click()
    assert len(fixes) == len(painted) == 1 and region == []
    assert painted[0]["review"] == widget.review
    assert painted[0]["run_id"] == "ac18" and painted[0]["component_id"] == "c" * 64
    assert painted[0]["point_native_xy"] == native_point
    assert widget.approval is None and widget._region_submission is None
    assert not widget._resolved(widget._current_row()) and not widget.recheck.isEnabled()


@pytest.mark.parametrize("failure", ["no_request", "stale_row", "missing_validator", "validator", "review"])
def test_async_region_result_never_publishes_without_current_request_and_original_validation(qtbot, failure):
    checked = []
    def refuse(*_args):
        checked.append(True)
        raise ValueError("original typed validator refused the action")
    options = dict(region_result_validator=refuse) if failure in {"validator", "review"} else {}
    widget, _ = _dialog(qtbot, **options)
    published = []
    widget.approval_changed.connect(published.append)
    key = widget.region_context_key()
    if failure != "no_request":
        _mark(widget)
        widget.deviation_editor.continue_button.click()
    if failure == "stale_row":
        _stage(qtbot, widget, "deviation", 1)
    if failure == "review":
        widget._review_validator = refuse
    assert widget._consume_additional_result(("region-approval", key), {"untrusted": "result"})
    assert widget.approval is None and published == [] and not widget.can_adopt()
    assert bool(checked) == (failure in {"validator", "review"})
    assert widget._consume_additional_result(("foreign-kind", key), {}) is False


def test_check_lock_covers_direct_slot_pending_targets_errors_and_unanswered_source(qtbot):
    widget, _ = _dialog(qtbot)
    starts = []
    widget.recheck_requested.connect(starts.append)
    _stage(qtbot, widget, "check")
    widget._preparation_changed = True
    widget.pending_corrections = [dict(run_id="ac18", component_id="c" * 64)]
    widget._buttons()
    assert not widget.recheck.isEnabled() and widget.check_explanation.text() == (
        "Check is locked. A finding still needs your explicit answer or correction.")
    widget._recheck()  # A programmatic slot call cannot bypass the visible lock.
    assert starts == [] and "locked" in widget.guided_status.text()
    assert widget.approval is None and not widget.can_adopt()

    source_only, _ = _dialog(qtbot, review=_bound_review(gaps=False))
    source_starts = []
    source_only.recheck_requested.connect(source_starts.append)
    _stage(qtbot, source_only, "check")
    assert not source_only.recheck.isEnabled()
    source_only._recheck()
    assert source_starts == []
    _stage(qtbot, source_only, "source_water")
    source_only.source_state.setCurrentIndex(1)
    source_only.set_state.click()  # Original human action; no fabricated resolved record.
    source_only._preparation_changed = True
    _stage(qtbot, source_only, "check")
    assert source_only.recheck.isEnabled() and source_only.recheck.isVisible()
    source_only.recheck.click()
    assert source_starts == [None] and source_only._candidate is None
    source_only.set_check_result(None, {"issues": [dict(
        severity="error", code="invalid-source", message="Fix this original source.")]})
    _stage(qtbot, source_only, "check")
    source_only._recheck()
    assert source_starts == [None] and not source_only.recheck.isEnabled()


def test_empty_review_requires_explicit_check_and_unconfirmed_water_still_blocks(qtbot):
    widget, _ = _dialog(qtbot, review=_bound_review(gaps=False, source_water=False),
                        preparation_dirty=lambda: True)
    starts = []
    widget.recheck_requested.connect(starts.append)
    assert widget.guided_stage == "check" and widget.recheck.isEnabled()
    widget.next_button.click()
    assert starts == [] and not widget.adopt.isVisible()
    widget.back_button.click()
    widget._unconfirmed_water_edits = {"ac18": {"unconfirmed": "line"}}
    widget._buttons()
    widget._recheck()
    assert starts == [] and not widget.recheck.isEnabled()
    widget._unconfirmed_water_edits = {}
    widget._buttons()
    widget.recheck.click()
    assert starts == [None] and widget.approval is None and widget._candidate is None


def test_all_issue_messages_remain_visible_and_warnings_need_their_own_reason(qtbot):
    widget, _ = _dialog(qtbot, review=_bound_review(gaps=False, source_water=False))
    notice = dict(severity="notice", code="for-information", message="Original notice remains visible.")
    warning = dict(severity="warning", code="placement-warning", message="Inspect the original placement.")
    widget.set_check_result(None, {"issues": [notice, warning, deepcopy(warning)]})
    assert widget._report["issues"][0] == notice
    assert "notice: for-information" in widget.information_details.toPlainText()
    assert notice["message"] in widget.information_details.toPlainText()
    widget.details_toggle.click()
    assert widget.information_details.isVisible()
    _stage(qtbot, widget, "warnings")
    assert widget.reason.text() == "" and widget.approve_warning.isVisible()
    assert not widget.approve_warning.isEnabled() and not widget.recheck.isEnabled()
    widget.reason.setText("inspected this exact warning")
    widget.approve_warning.click()
    assert len(widget._warning_actions) == 1 and not widget.recheck.isEnabled()
    _stage(qtbot, widget, "warnings", 1)
    assert widget.reason.text() == ""  # Duplicate text is a separate original occurrence.
    widget.reason.setText("inspected the second occurrence")
    widget.approve_warning.click()
    assert len(widget._warning_actions) == 2
    assert widget._report["issues"][0] == notice and widget._candidate is None


@pytest.mark.parametrize("wrong_role", [False, True])
def test_photo_bridge_preserves_draft_panzoom_records_and_refuses_wrong_role(qtbot, wrong_role):
    backend = GuidedBackend()
    original_reader = backend.comparison
    if wrong_role:
        def wrong(review, run, component, role):
            return dict(original_reader(review, run, component, role), image_role="previous-run")
        backend.comparison = wrong
    widget, _ = _dialog(qtbot, backend=backend)
    draft, _ = _mark(widget)
    before = deepcopy(draft.snapshot())
    source_rgb = backend.packets["ac18"]["image_rgb"].copy()
    editor = widget.deviation_editor
    transform = editor.canvas.transform()
    center = editor.canvas.mapToScene(editor.canvas.viewport().rect().center())
    assert not widget.image_shortcut.isEnabled() and editor.photo_shortcut.isEnabled()
    editor.photo_toggle.click()
    _settle(qtbot, widget)
    assert backend.comparisons == [("ac18", "c" * 64, "clearwater", False)]
    assert not editor._photo_pending
    if wrong_role:
        assert editor._photo_role == "current" and widget.images.currentData() == "current"
        assert editor.canvas.pixmap_item.pixmap().toImage().pixelColor(8, 7).red() == 30
        backend.comparison = original_reader
        widget._comparison_reader = original_reader
        editor.photo_toggle.click()  # The failed role cannot leave a stuck same-index retry.
        _settle(qtbot, widget)
    assert editor._photo_role == "clearwater"
    assert editor.canvas.transform() == transform
    after = editor.canvas.mapToScene(editor.canvas.viewport().rect().center())
    assert abs(after.x() - center.x()) < 1 and abs(after.y() - center.y()) < 1
    np.testing.assert_array_equal(draft.class_masks["air"], before["class_masks"]["air"])
    assert draft.actions == before["actions"]
    np.testing.assert_array_equal(backend.packets["ac18"]["image_rgb"], source_rgb)
    assert widget.approval is None and widget._acquisition_actions == [] and widget._warning_actions == {}
    editor.photo_toggle.click()
    assert editor._photo_role == "current" and not editor._photo_pending
    _stage(qtbot, widget, "source_water")
    assert widget.image_shortcut.isEnabled() and not editor.photo_shortcut.isEnabled()
    assert widget.photo_toolbar.isVisible()


def test_close_during_inherited_read_keeps_gui_responsive_and_discards_late_result(qtbot):
    backend = GuidedBackend()
    entered, release = threading.Event(), threading.Event()
    original = backend.preview
    def delayed(review, run, component_id=None):
        if run == "ac61":
            entered.set()
            assert release.wait(5)
        return original(review, run, component_id)
    backend.preview = delayed
    widget, _ = _dialog(qtbot, backend=backend)
    approvals, requests = [], []
    widget.approval_changed.connect(approvals.append)
    widget.region_confirmation_requested.connect(requests.append)
    try:
        index = next(i for i, item in enumerate(widget._questions)
                     if item == ("deviation", ("gap", "ac61", "d" * 64)))
        widget._show_question(index)
        qtbot.waitUntil(entered.is_set, timeout=5000)
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        qtbot.waitUntil(lambda: bool(ticks), timeout=1000)
        widget.close()
        assert widget._closing and widget.busy() and widget._pending is None
        assert not widget.next_button.isEnabled()
        assert not widget.deviation_editor.photo_shortcut.isEnabled()
        release.set()
        qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
        assert widget.thread is None and widget.worker is None
        assert approvals == requests == [] and widget.approval is None
        assert widget._region_submission is None
    finally:
        release.set()
        widget.wait_for_workers()


class OpaqueDraft(DraftSpy):
    """A transferred backend draft must never be copied into a photo packet."""
    def __deepcopy__(self, memo):
        raise AssertionError("The backend draft is opaque to display-packet copying")


def _suggested_backend():
    backend = GuidedBackend()
    draft = OpaqueDraft(backend.packets["ac18"]["gap_mask"])
    row, column = np.argwhere(draft.editable_mask)[0]
    draft.class_masks["air"][row, column] = True
    original_preview = backend.preview
    def preview(review, run, component_id=None):
        packet = original_preview(review, run, component_id)
        if run == "ac18" and component_id == "c" * 64:
            packet.update(region_draft=draft,
                region_transfer_note="Suggestion from AC17: 1 matching pixel; 8 pixels still need classification. Confirm this area yourself.")
        return packet
    backend.preview = preview
    def factory(review, packet):
        return packet["region_draft"] if "region_draft" in packet else backend.draft(review, packet)
    return backend, draft, factory


def test_transferred_draft_is_displayed_as_suggestion_without_copying_or_confirmation(qtbot):
    backend, draft, factory = _suggested_backend()
    widget, _ = _dialog(qtbot, backend=backend, draft_factory=factory)
    assert widget._deviation_draft is draft and widget._base_packet["region_draft"] is draft
    assert "region_draft" not in widget.deviation_editor.packet
    assert widget.region_transfer_note.isVisible()
    assert "Suggestion from AC17" in widget.region_transfer_note.text()
    assert "8 pixels" in widget.region_transfer_note.text()
    assert widget.region_preview_reset.isVisible() and widget.region_preview_reset.isEnabled()
    assert widget.confirmed_region_request() is None and widget.approval is None
    requests = []
    widget.region_confirmation_requested.connect(requests.append)
    widget.deviation_editor.photo_toggle.click()
    _settle(qtbot, widget)
    assert "region_draft" not in widget.deviation_editor.packet
    assert draft.class_masks["air"].sum() == 1 and requests == []
    widget.deviation_editor.photo_toggle.click()
    _stage(qtbot, widget, "deviation", 1)
    _stage(qtbot, widget, "deviation")
    assert requests == [] and widget.confirmed_region_request() is None
    widget.deviation_editor.continue_button.click()
    assert len(requests) == 1  # This destination still needs its own real human action.
    assert widget.confirmed_region_request() is None and widget.approval is None


def test_explicit_reset_replaces_only_current_draft_after_original_packet_admission(qtbot):
    backend, suggested, factory = _suggested_backend()
    widget, _ = _dialog(qtbot, backend=backend, draft_factory=factory)
    _stage(qtbot, widget, "deviation", 1)
    other, _ = _mark(widget, "free_water")
    other_key = widget.region_context_key()
    _stage(qtbot, widget, "deviation")
    key = widget.region_context_key()
    resets, confirmations = [], []
    widget.region_preview_reset_requested.connect(resets.append)
    widget.region_confirmation_requested.connect(confirmations.append)
    assert resets == [] and widget._deviation_draft is suggested
    widget.region_preview_reset.click()
    assert resets == [dict(review_binding=widget.review["review_binding"], run_id="ac18",
        component_id="c" * 64, review_row_sha256="c" * 64)]
    assert widget._deviation_draft is suggested  # A request alone cannot discard the old draft.
    resets[0]["review_binding"]["sha256"] = "mutable receiver"
    packet = dict(backend.packets["ac18"], component_id="c" * 64)
    assert widget._consume_additional_result(("region-preview-reset", key), packet)
    assert widget._deviation_draft is not suggested
    assert not any(mask.any() for mask in widget._deviation_draft.class_masks.values())
    assert suggested.class_masks["air"].sum() == 1
    assert widget._drafts[other_key] is other and other.class_masks["free_water"].any()
    assert not widget.region_preview_reset.isVisible() and confirmations == []
    assert widget.approval is None and widget.confirmed_region_request() is None


@pytest.mark.parametrize("state", ["unsolicited", "stale", "closing"])
def test_reset_result_without_current_explicit_request_cannot_replace_a_draft(qtbot, state):
    backend, suggested, factory = _suggested_backend()
    widget, _ = _dialog(qtbot, backend=backend, draft_factory=factory)
    key = widget.region_context_key()
    if state != "unsolicited":
        widget.region_preview_reset.click()
    if state == "stale":
        _stage(qtbot, widget, "deviation", 1)
    elif state == "closing":
        widget.close()
    before = widget._deviation_draft
    packet = dict(backend.packets["ac18"], component_id="c" * 64)
    assert widget._consume_additional_result(("region-preview-reset", key), packet)
    assert widget._deviation_draft is before and suggested.class_masks["air"].sum() == 1
    assert widget.approval is None and widget.confirmed_region_request() is None


def test_validated_callback_result_is_retained_once_on_close_without_new_reader(qtbot):
    """Isolate dispatch lifetime; the empty sentinel approves no actual gap.

    Real C acceptance/proof equality is a separate backend integration gate.
    This callback spy tests ordering and exactly-once publication only.
    """
    validated = []
    empty_ui_result = {"runs": [], "ui_lifetime_test": "not a scientific approval"}
    def validator(result, review, request):
        assert widget._closing and widget.confirmed_region_request() is None
        validated.append((deepcopy(result), deepcopy(review), deepcopy(request)))
        return empty_ui_result
    widget, backend = _dialog(qtbot, region_result_validator=validator)
    _mark(widget)
    widget.deviation_editor.continue_button.click()
    assert widget.confirmed_region_request() is None
    key = widget.region_context_key()
    actual_request = deepcopy(widget._region_submission[1])
    published = []
    widget.approval_changed.connect(published.append)
    reads_before = list(backend.reads)
    widget.close()
    def suspended_live_view(_review):
        raise AssertionError("A suspended live view cannot decide retention of a completed typed write")
    widget._review_validator = suspended_live_view
    refresh, buttons = widget._refresh_rows, widget._buttons
    def forbidden():
        raise AssertionError("Closing publication must not refresh rows, start reads, or update controls")
    widget._refresh_rows = widget._buttons = forbidden
    try:
        assert widget._consume_additional_result(("region-approval", key), {"worker": "saved result"})
        assert len(validated) == len(published) == 1
        assert validated[0][2]["reason"] == actual_request["reason"]
        assert widget.confirmed_region_request()["other_text"] == actual_request["other_text"]
        confirmed = widget.confirmed_region_request()
        confirmed["reason"] = "mutable caller"
        confirmed["snapshot"]["class_masks"]["air"][:] = False
        assert widget.confirmed_region_request()["reason"] == actual_request["reason"]
        assert widget.confirmed_region_request()["snapshot"]["class_masks"]["air"].any()
        assert widget._consume_additional_result(("region-approval", key), {"duplicate": "delivery"})
        assert len(validated) == len(published) == 1 and backend.reads == reads_before
        assert widget.approval == empty_ui_result and not widget._resolved(widget._current_row())
        widget.review["review_binding"]["sha256"] = "changed review"
        assert widget.confirmed_region_request() is None
    finally:
        widget._refresh_rows, widget._buttons = refresh, buttons


def test_refused_result_while_closing_never_retains_a_confirmation(qtbot):
    def refuse(*_args):
        raise ValueError("The original typed proof does not match this submission")
    widget, _ = _dialog(qtbot, region_result_validator=refuse)
    _mark(widget)
    widget.deviation_editor.continue_button.click()
    key = widget.region_context_key()
    published = []
    widget.approval_changed.connect(published.append)
    widget.close()
    assert widget._consume_additional_result(("region-approval", key), {"untrusted": "result"})
    assert widget.confirmed_region_request() is None and widget.approval is None and published == []


@pytest.mark.parametrize("change", ["paint", "reset"])
def test_changed_confirmed_draft_locks_ui_until_same_context_is_confirmed_again(qtbot, change):
    """Exercise UI predicates with an explicitly injected callback result.

    The row/candidate below are finite UI fixtures, not scientific proof. The
    production factory must qualify actual C proof/snapshot equality separately.
    """
    backend, _, factory = _suggested_backend()
    review = _bound_review(source_water=False)
    review["runs"][1]["components"] = []
    review["runs"][1]["gap"]["pixels"] = 0
    accepted_requests = []
    def validator(result, review, request):
        accepted_requests.append(deepcopy(request))
        return dict(runs=[{key: deepcopy(request[key]) for key in (
            "run_id", "component_id", "review_row_sha256", "actor", "approved_utc",
            "reason", "decision_ref")}], ui_test_only="no scientific proof")
    widget, _ = _dialog(qtbot, backend=backend, review=review, draft_factory=factory,
                        region_result_validator=validator)
    events = []
    widget.draft_choices_changed.connect(events.append)
    widget.deviation_editor.continue_button.click()
    key = widget.region_context_key()
    widget._consume_additional_result(("region-approval", key), {"ui": "first result"})
    assert accepted_requests[0]["previous"] is None and events == []
    saved = deepcopy(widget.approval)
    assert widget._resolved(widget._current_row()) and widget.pending_draft_choices() == []
    widget.set_check_result({"ui_test_candidate": True}, {"issues": []})
    assert widget.can_adopt()

    if change == "reset":
        widget.region_preview_reset.click()
        fresh = dict(backend.packets["ac18"], component_id="c" * 64)
        widget._consume_additional_result(("region-preview-reset", key), fresh)
        assert not any(mask.any() for mask in widget._deviation_draft.class_masks.values())
    else:
        _mark(widget, "free_water")
        widget.deviation_editor.preview_changed.emit(widget._deviation_draft.snapshot())
    assert widget.approval == saved  # Pending strokes do not rewrite the old literal proof.
    assert widget.confirmed_region_request() is None and not widget.can_adopt()
    assert widget.pending_draft_choices() == [dict(review_binding=review["review_binding"],
        run_id="ac18", component_id="c" * 64, review_row_sha256="c" * 64, pending=True)]
    assert [event["pending"] for event in events] == [True]
    widget.deviation_editor.preview_changed.emit(widget._deviation_draft.snapshot())
    assert len(events) == 1

    # Neither navigation nor a later displayed report clears this new human
    # decision requirement; all original predicates would otherwise be ready.
    widget.set_check_result({"ui_test_candidate": True}, {"issues": []})
    _stage(qtbot, widget, "check")
    starts = []
    widget.recheck_requested.connect(starts.append)
    widget._recheck()
    assert starts == [] and not widget.recheck.isEnabled() and not widget.can_adopt()
    _stage(qtbot, widget, "deviation")
    widget.deviation_editor.photo_toggle.click()
    _settle(qtbot, widget)
    widget.deviation_editor.photo_toggle.click()
    assert widget.pending_draft_choices() and len(events) == 1
    if change == "reset":
        _mark(widget, "free_water")
        widget.deviation_editor.preview_changed.emit(widget._deviation_draft.snapshot())
    widget.reason.setText("Confirmed these changed classifications separately")
    widget.deviation_editor.continue_button.click()
    assert widget._region_submission[1]["previous"] == saved
    widget._consume_additional_result(("region-approval", key), {"ui": "second result"})
    assert accepted_requests[1]["previous"] == saved
    assert widget.pending_draft_choices() == []
    assert [event["pending"] for event in events] == [True, False]
    assert widget.confirmed_region_request()["reason"] == "Confirmed these changed classifications separately"
    assert widget._first_precheck_unresolved() is None
