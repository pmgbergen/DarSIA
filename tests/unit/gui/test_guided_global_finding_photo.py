"""Finite presentation cases; global findings create no photograph or approval."""
from copy import deepcopy

import pytest

from .test_painting_guided_review_ui import _bound_review, _dialog, _stage


@pytest.mark.parametrize("severity,code", [
    ("error", "unmapped_labels"),
    ("warning", "synthetic_series_warning"),
])
def test_global_finding_without_run_has_truthful_photo_state_and_original_controls(qtbot, severity, code):
    widget, backend = _dialog(qtbot, editor=False,
        review=_bound_review(gaps=False, source_water=False))
    # Report DATA is a finite GUI fixture, not scientific acceptance. The real
    # original row builder/select path must decide there is no photo context.
    issue = dict(severity=severity, code=code,
        message="Some image labels have no layer definition." if severity == "error"
                else "Inspect this series-wide warning.")
    report = dict(config_sha256="f" * 64, issues=[deepcopy(issue)])
    widget.set_catalogue_result(report)
    _stage(qtbot, widget, "warnings")
    row = widget._current_row()
    assert row["run_id"] == "" and row["issue"] == issue
    assert widget._preview_key is None and widget.packet is None and widget._pending is None
    assert not widget.busy() and backend.reads == [("review", False)]
    assert widget.image_stack.currentWidget() is widget.photo_placeholder
    assert widget.photo_placeholder.text() == (
        "This finding has no photograph. Use the existing controls to resolve its cause.")
    assert "loading" not in widget.photo_placeholder.text()
    assert not widget.photo_toolbar.isVisible() and widget.photo_availability.text() == ""
    assert "fix or resolve its cause" in widget.status.text()
    assert widget.reason.isVisible() and widget.reason.isEnabled()
    assert widget.fix.isVisible() and widget.fix.isEnabled()
    assert widget.next_button.isEnabled() and not widget.recheck.isEnabled()
    assert not widget._resolved(row) and not widget.approve_warning.isEnabled()
    widget.reason.setText("I inspected the original finding; no choice has been recorded.")
    assert widget.approve_warning.isVisible() is (severity == "warning")
    assert widget.approve_warning.isEnabled() is (severity == "warning")
    fixed = []
    widget.fix_requested.connect(fixed.append)
    widget.fix.click()
    assert fixed == [issue] and not widget._resolved(row)

    # A subsequent real per-run selection still uses the original owned reader;
    # neither its photo nor its toolbar leaks back onto the global finding.
    per_run = dict(severity="warning", code="synthetic_run_warning",
                   message="Inspect this run's photograph.", details=dict(run_id="ac18"))
    widget.set_catalogue_result(dict(report, issues=[deepcopy(issue), per_run]))
    _stage(qtbot, widget, "warnings", 1)
    assert widget._preview_key == ("ac18", None) and widget.packet["run_id"] == "ac18"
    assert widget.image_stack.currentWidget() is widget.photo and widget.photo_toolbar.isVisible()
    assert "has no photograph" not in widget.photo_placeholder.text()
    _stage(qtbot, widget, "warnings", 0)
    assert widget._preview_key is None and widget.packet is None
    assert widget.image_stack.currentWidget() is widget.photo_placeholder
    assert "has no photograph" in widget.photo_placeholder.text()
    assert not widget.photo_toolbar.isVisible()
    assert backend.reads == [("review", False), ("ac18", False)]
    assert widget.approval is None and widget._candidate is None
    assert widget._warning_actions == {} and widget._acquisition_actions == []
    assert backend.approvals == [] and not widget.recheck.isEnabled()
