"""First valid grouping focuses once; subsequent redraws preserve local work.

Finite original GUI fixtures supply two independent descriptive groups. No
carrier result, scientific source acceptance or successful approval is faked.
"""
from copy import deepcopy

import numpy as np

from darsia.fluidflower.painted_gap_review_groups import FORMAT

from .test_painting_guided_review_ui import _bound_review, _dialog, _mark, _settle, _stage


def test_first_valid_group_catalogue_starts_at_first_group_once_and_preserves_later_selection_and_draft(qtbot):
    review = _bound_review()
    widget, backend = _dialog(qtbot, review=review)
    assert widget._current_row()["run_id"] == "ac18"
    assert not widget._first_group_catalogue_installed
    groups = []
    for run in ("ac61", "ac18"):
        row = next(row for row in review["runs"] if row["run_id"] == run)
        component = row["components"][0]
        item = dict(run_id=run, component_id=component["component_id"],
                    review_row_sha256=component["review_row_sha256"])
        groups.append(dict(representative=deepcopy(item), members=[deepcopy(item)]))
    data = dict(format=FORMAT, review=deepcopy(review["review_binding"]), groups=groups,
        summary=dict(deviation_count=2, classification_groups=2, transferred_confirmations=0, failed_pairs=0))
    rows = deepcopy(widget._rows)
    requests = []
    for signal in (widget.region_confirmation_requested, widget.recheck_requested,
                   widget.adopt_requested, widget.state_requested):
        signal.connect(lambda *args: requests.append(args))

    wrong = deepcopy(data)
    wrong["review"]["sha256"] = "a"*64
    widget.set_catalogue_result(dict(issues=[], deviation_groups=wrong))
    _settle(qtbot, widget)
    assert not widget._first_group_catalogue_installed and widget._current_row()["run_id"] == "ac18"

    widget.set_catalogue_result(dict(issues=[], deviation_groups=deepcopy(data)))
    _settle(qtbot, widget)
    assert widget._first_group_catalogue_installed and widget._question_index == 0
    assert widget._current_row()["run_id"] == "ac61"
    assert widget.question_progress.text().startswith("Step 1 of")

    _stage(qtbot, widget, "deviation", 1)
    assert widget._current_row()["run_id"] == "ac18" and widget._question_index == 1
    draft, _ = _mark(widget, "air")
    widget.deviation_editor.preview_changed.emit(draft.snapshot())
    widget.reason.setText("Keep this local inspection while the list refreshes")
    key = widget._context_key()
    masks = {name: mask.copy() for name, mask in draft.class_masks.items()}
    history = deepcopy(draft.actions)
    assert widget._report is None and key in widget._pending_region_contexts
    widget.refresh_guidance()  # Dirty _report=None is not a new first install.
    _settle(qtbot, widget)
    assert widget._current_row()["run_id"] == "ac18" and widget._question_index == 1
    assert widget._deviation_draft is draft and draft.actions == history

    widget.set_catalogue_result(dict(issues=[], deviation_groups=deepcopy(data)))
    _settle(qtbot, widget)
    assert widget._first_group_catalogue_installed and widget._question_index == 1
    assert widget._current_row()["run_id"] == "ac18" and widget._deviation_draft is draft
    assert widget.reason.text() == "Keep this local inspection while the list refreshes"
    assert draft.actions == history and key in widget._pending_region_contexts
    for name, mask in masks.items():
        np.testing.assert_array_equal(draft.class_masks[name], mask)
    assert widget._rows == rows and requests == [] and backend.approvals == []
    assert widget.approval is widget._candidate is None and not widget.can_recheck()
