"""Check copy stays truthful when original rows and grouped questions differ.

The existing finite GUI reader supplies two independent descriptive groups.
Navigation records no answers and grants no scientific or approval authority.
"""
from copy import deepcopy

from darsia.fluidflower.painted_gap_review_groups import FORMAT

from .test_painting_guided_review_ui import _bound_review, _dialog, _settle, _stage


def test_locked_check_has_no_numeric_pointer_and_next_uses_actual_group_order(qtbot):
    review = _bound_review(source_water=False)
    widget, backend = _dialog(qtbot, review=review)
    groups = []
    for run in ("ac61", "ac18"):
        row = next(row for row in review["runs"] if row["run_id"] == run)
        component = row["components"][0]
        item = dict(run_id=run, component_id=component["component_id"],
                    review_row_sha256=component["review_row_sha256"])
        groups.append(dict(representative=deepcopy(item), members=[deepcopy(item)]))
    data = dict(format=FORMAT, review=deepcopy(review["review_binding"]), groups=groups,
        summary=dict(deviation_count=2, classification_groups=2,
                     transferred_confirmations=0, failed_pairs=0))
    requests = []
    for signal in (widget.region_confirmation_requested, widget.recheck_requested,
                   widget.adopt_requested, widget.state_requested, widget.approval_changed):
        signal.connect(lambda *args: requests.append(args))
    rows = deepcopy(widget._rows)
    assert [row["run_id"] for row in rows] == ["ac18", "ac61"]
    widget.set_catalogue_result(dict(issues=[], deviation_groups=data))
    _settle(qtbot, widget)
    question_rows = {widget._row_key(row): row for row in widget._rows}
    assert [question_rows[key]["run_id"] for kind, key in widget._questions
            if kind == "deviation"] == ["ac61", "ac18"]
    assert widget._first_precheck_unresolved()["run_id"] == "ac18"

    _stage(qtbot, widget, "check")
    assert not widget.recheck.isEnabled() and widget.next_button.isEnabled()
    assert widget.check_explanation.text() == (
        "Check is locked. A finding still needs your explicit answer or correction.")
    assert not any(character.isdigit() for character in widget.check_explanation.text())
    widget._recheck()  # The original direct-slot guard still refuses to Check.
    assert widget.guided_status.text() == widget.check_explanation.text()
    widget.next_button.click()
    _settle(qtbot, widget)
    assert widget.guided_stage == "deviation" and widget._question_index == 0
    assert widget._current_row()["run_id"] == "ac61"
    assert widget.question_progress.text().startswith("Step 1 of")
    assert widget._rows == rows and widget._first_precheck_unresolved()["run_id"] == "ac18"
    assert widget.pending_draft_choices() == [] and not widget.can_recheck()
    assert widget.approval is widget._candidate is None
    assert widget._acquisition_actions == [] and widget._warning_actions == {}
    assert requests == [] and backend.approvals == []
