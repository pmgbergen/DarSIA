"""Grouped original tiny deviations, with genuine separate typed confirmations.

Only host selection and native display-grid decoration use finite GUI seams.
Original source/product replay, carrier, drafts, C save/result validation and
owned reader retirement remain active. Injected suggestion/read faults grant
no scientific authority and must return the original row to its own question.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import painted_analysis_exclusions as gaps
from darsia.fluidflower import painted_gap_regions as regions
from darsia.fluidflower.painted_gap_review_groups import deviation_groups_from_replay
from darsia.fluidflower.painted_transfer_replay import replay_transfer_review
from darsia.fluidflower.painting_guided_review_bindings import RegionPreviewSession, guided_gap_dialog

from .test_painting_gap_review_ui import _controller
from .test_painting_guided_review_ui import _settle, _stage
from ..test_painted_gap_regions_integration import (  # noqa: F401
    _fill, both_gap_case, imported, registered_setup, synthetic_payload_admission,
)


class SuggestionSession(RegionPreviewSession):
    """Post-original finite DATA faults, never a scientific admission waiver."""
    def __init__(self, mode):
        super().__init__()
        self.mode, self.failed = mode, False
        self.partial_injections = 0
        self.original_seed_contexts = []

    def reader(self, original_reader):
        verified = super().reader(original_reader)
        def read(review, run, component=None):
            packet = verified(review, run, component)
            draft = packet.get("region_draft")
            origin = None if draft is None else draft.preview_source_context
            if origin is not None:
                self.original_seed_contexts.append(deepcopy(origin))
            if origin is not None and self.mode == "read_error" and not self.failed:
                self.failed = True
                raise ValueError("Finite suggestion-read fault after the original preview")
            if origin is not None and self.mode == "partial":
                self.partial_injections += 1
                # The real typed draft allows incomplete descriptive seeds;
                # it never allows them to become a complete C approval.
                masks = {name: mask.copy() for name, mask in draft.class_masks.items()}
                y, x = np.argwhere(draft.editable_mask)[0]
                for mask in masks.values():
                    mask[y, x] = False
                partial = regions.region_draft_factory(review, packet)
                partial.set_preview_class_masks(masks, source_context=origin)
                packet = dict(packet, region_draft=partial)
            return packet
        return read


def _grouped_dialog(qtbot, monkeypatch, case, mode):
    host, _unused_backend, controller = _controller(qtbot, monkeypatch)
    host.project, host.record = case.data.project, deepcopy(case.review["map_version"])
    host.approval = deepcopy(host.record["approval"])
    controller.context = controller._findings_context_snapshot()
    controller._findings_context = controller.context
    controller._findings_review_binding = deepcopy(case.review["review_binding"])
    controller.catalogue_refresher = lambda request: pytest.fail("This test must not launch a cohort collector")
    replay = replay_transfer_review(case.review,
        source_context=deepcopy(case.review["preparation_context"]))
    grouping = deviation_groups_from_replay(replay)
    assert grouping["summary"]["deviation_count"] == 2
    assert grouping["summary"]["classification_groups"] == 1

    def display_preview(review, run, component=None):
        packet = gaps.read_gap_preview(review, run, component)
        packet.update(image_role="current", display_grid=dict(
            format="synthetic-native-display-only", shape_hw=list(packet["gap_mask"].shape),
            native_to_photo_h=packet["native_to_photo_h"]))
        return packet

    widget = guided_gap_dialog(case.review, actor=host._reviewer(), parent=host,
        preview_session=SuggestionSession(mode), preview_reader=display_preview,
        review_reader=gaps.read_gap_review, review_validator=gaps.read_gap_review)
    controller.gap_dialog = widget
    qtbot.addWidget(widget, before_close_func=lambda owned: owned.wait_for_workers())
    requests, accepted = [], []
    widget.region_confirmation_requested.connect(requests.append)
    widget.region_confirmation_requested.connect(controller._confirm_gap_regions)
    widget.region_preview_reset_requested.connect(controller._reset_region_preview)
    widget.approval_changed.connect(controller._areas_changed)
    widget.approval_changed.connect(lambda value: accepted.append(deepcopy(value)))
    widget.draft_choices_changed.connect(lambda payload: controller.set_findings_draft_pending(widget, payload))
    widget.resize(1100, 850)
    widget.show()
    _settle(qtbot, widget)
    initial_cache, initial_key = widget._deviation_draft, widget._context_key()
    if mode == "undo_cache":
        # Real original stroke and undo history are human draft state even
        # when the last visible state is empty. It must never be reseeded.
        _fill(initial_cache, "free_water")
        widget.deviation_editor.preview_changed.emit(initial_cache.snapshot())
        assert initial_cache.undo()
        widget.deviation_editor.preview_changed.emit(initial_cache.snapshot())
        widget.deviation_editor._buttons()
    widget.set_catalogue_result(dict(issues=deepcopy(case.review.get("issues", [])),
        deviation_groups=deepcopy(grouping)))
    _settle(qtbot, widget)
    return widget, controller, requests, accepted, grouping, initial_cache, initial_key


@pytest.mark.parametrize("mode", ["full", "partial", "read_error", "bad_group", "undo_cache", "reset_cache"])
def test_group_classifies_once_but_each_destination_needs_its_own_original_confirm(
        qtbot, monkeypatch, both_gap_case, mode):
    case = both_gap_case
    widget, controller, requests, accepted, grouping, initial_cache, initial_key = _grouped_dialog(
        qtbot, monkeypatch, case, mode)
    before = Path(case.review["review_binding"]["path"]).read_bytes()
    rows = deepcopy(widget._rows)
    assert len([stage for stage, _ in widget._questions if stage == "deviation"]) == 1
    assert len([row for row in widget._rows if row["kind"] == "gap"]) == 2
    assert "Groups to classify: 1" in widget.deviation_group_summary.text()
    assert "Transferred areas: 1" in widget.deviation_group_summary.text()
    assert widget.approval is widget._candidate is None and accepted == requests == []
    if mode == "bad_group":
        wrong = deepcopy(grouping)
        wrong["groups"][0]["members"][1]["review_row_sha256"] = "0"*64
        widget.set_catalogue_result(dict(issues=deepcopy(case.review.get("issues", [])), deviation_groups=wrong))
        _settle(qtbot, widget)
        assert len([stage for stage, _ in widget._questions if stage == "deviation"]) == 2
        assert widget._rows == rows and "unavailable" in widget.deviation_group_summary.text()
        assert accepted == requests == [] and not widget.can_recheck()
        return

    _stage(qtbot, widget, "deviation")
    assert not widget.deviation_group_choice.model().item(1).isEnabled()
    representative = grouping["groups"][0]["representative"]
    assert widget._current_row()["run_id"] == representative["run_id"]
    if mode in {"full", "undo_cache"}:
        # The ordinary initial review already displayed the smaller first row
        # before grouping selected the larger representative. Its pristine
        # cached draft must not suppress a later verified transfer suggestion.
        destination = grouping["groups"][0]["members"][1]
        destination_row = next(row for row in widget._rows if row.get("run_id") == destination["run_id"]
            and row.get("component_id") == destination["component_id"])
        pristine = widget._drafts[widget._row_context_key(destination_row)]
        assert pristine is initial_cache and pristine.preview_source_context is None
        assert not any(mask.any() for mask in pristine.class_masks.values())
        if mode == "undo_cache":
            history = deepcopy(pristine.events)
            assert history and pristine.can_redo and initial_key in widget._pending_region_contexts
        else:
            assert not pristine.events
    _fill(widget._deviation_draft, "air")  # Original brush/polygon kernel on the real component.
    widget.deviation_editor._buttons()
    widget.deviation_editor.continue_button.click()
    qtbot.waitUntil(lambda: len(accepted) == 1 and not widget.busy(), timeout=10000)
    assert len(requests) == 1 and len(accepted[0]["runs"]) == 1
    assert widget.deviation_group_choice.model().item(1).isEnabled()
    assert "1 need individual confirmation" in widget.deviation_group_summary.text()
    assert not widget.can_recheck()
    widget.next_button.click()  # Next inspects the group's destination before the next work stage.
    _settle(qtbot, widget)
    destination = grouping["groups"][0]["members"][1]
    assert widget._current_row()["run_id"] == destination["run_id"]
    assert len(accepted) == len(requests) == 1 and not widget._resolved(widget._current_row())
    assert widget._rows == rows and not widget.can_recheck()
    session = widget.region_preview_session
    assert session.original_seed_contexts
    if mode in {"partial", "read_error", "undo_cache"}:
        assert len([stage for stage, _ in widget._questions if stage == "deviation"]) == 2
        assert widget._current_deviation_group() == [widget._row_key(widget._current_row())]
        assert controller._area_approval == accepted[0] and widget.approval == accepted[0]
        if mode == "partial":
            assert session.partial_injections > 0
            assert widget._deviation_draft.preview_source_context is not None
            assert not np.array_equal(widget._deviation_draft.class_masks["air"], widget._deviation_draft.editable_mask)
        elif mode == "read_error":
            assert session.failed
        else:
            assert widget._deviation_draft is pristine and pristine.events == history and pristine.can_redo
            assert pristine.preview_source_context is None and initial_key in widget._pending_region_contexts
            assert widget._base_packet["region_draft"].preview_source_context is not None
        assert Path(case.review["review_binding"]["path"]).read_bytes() == before
        return

    draft = widget._deviation_draft
    assert draft.preview_source_context == dict(review=case.review["review_binding"], **representative)
    assert draft is widget._base_packet["region_draft"] and draft is not initial_cache
    assert not draft.events  # No second drawing or fabricated confirmation.
    np.testing.assert_array_equal(draft.class_masks["air"], draft.editable_mask)
    assert "Confirm these classifications" in widget.deviation_editor.continue_button.text()
    if mode == "reset_cache":
        assert widget.region_preview_reset.isEnabled()
        widget.region_preview_reset.click()
        _settle(qtbot, widget)
        reset = widget._deviation_draft
        assert reset.preview_source_context is None and not reset.events
        assert not any(mask.any() for mask in reset.class_masks.values())
        assert widget._context_key() in widget._pending_region_contexts
        assert widget._current_deviation_group() == [widget._row_key(widget._current_row())]
        widget._load(widget._preview_key)  # Original owned reread must preserve the explicit suppression.
        _settle(qtbot, widget)
        assert widget._base_packet["region_draft"].preview_source_context is None
        assert widget._deviation_draft is reset and widget.approval == accepted[0]
        assert len(accepted) == len(requests) == 1 and not widget.can_recheck()
        assert Path(case.review["review_binding"]["path"]).read_bytes() == before
        return
    widget.deviation_editor.continue_button.click()
    qtbot.waitUntil(lambda: len(accepted) == 2 and not widget.busy(), timeout=10000)
    assert requests[1]["previous"] == accepted[0]
    # The original approval publishes canonical review order, not click order.
    assert len(accepted[-1]["runs"]) == 2
    prior = accepted[0]["runs"][0]
    assert next(row for row in accepted[-1]["runs"]
                if (row["run_id"], row["component_id"]) ==
                (prior["run_id"], prior["component_id"])) == prior
    assert {(row["run_id"], row["component_id"]) for row in accepted[-1]["runs"]} == {
        (representative["run_id"], representative["component_id"]),
        (destination["run_id"], destination["component_id"])}
    regions.validate_gap_region_approval(accepted[-1], case.review)
    assert "0 need individual confirmation" in widget.deviation_group_summary.text()
    assert widget._rows == rows and widget._candidate is None and controller.preparation is controller.dialog is None
    assert Path(case.review["review_binding"]["path"]).read_bytes() == before
