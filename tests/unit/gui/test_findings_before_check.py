"""Human-resolution readiness and catalogue coverage with finite synthetic readers."""
from copy import deepcopy

import pytest

from darsia.fluidflower import painting_gap_review_ui as gaps

from .test_painting_gap_review_ui import _Backend, _dialog, _review


def _open(qtbot, issues=()):
    source = _review()
    source["issues"] = deepcopy(list(issues))
    return _dialog(qtbot, source=source)


def _select(qtbot, widget, kind):
    index = next(index for index, row in enumerate(widget._rows) if row["kind"] == kind)
    widget.runs.setCurrentCell(index, 0)
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    widget.reason.setText("The displayed evidence was reviewed explicitly")


def _area_rows(widget):
    return [{key: row[key] for key in ("run_id", "component_id", "review_row_sha256")}
            for row in widget._rows if row["kind"] == "gap"]


def _resolved_inputs(widget):
    widget.approval = dict(runs=[dict(row, classification="air", reason="Explicit area choice")
                                 for row in _area_rows(widget)])
    widget._acquisition_actions = [dict(acquisition_state="clearwater", reason="Explicit source choice")]
    widget._refresh_rows()


def _waiting(qtbot):
    backend = _Backend()
    widget = gaps.GapReviewDialog(None, actor="Synthetic reviewer", review_reader=backend.read,
        preview_reader=backend.preview, approval_builder=backend.approve)
    qtbot.addWidget(widget, before_close_func=lambda owned: owned.wait_for_workers())
    widget.show()
    return widget


def test_check_requires_every_area_source_choice_and_warning_even_for_direct_calls(qtbot):
    warning = dict(severity="warning", code="review_warning", message="Review this warning", run_id="ac18")
    widget, backend = _open(qtbot, [warning])
    rechecks, changes = [], []
    widget.recheck_requested.connect(rechecks.append)
    widget.readiness_changed.connect(lambda: changes.append(widget.unresolved_findings_reason()))
    widget.classification.setCurrentIndex(widget.classification.findData("air"))
    widget.reason.setText("The selected area was reviewed")
    widget.approve_one.click()
    assert backend.approvals[-1][2]["classification"] == "air"
    assert not widget.can_recheck() and not widget.recheck.isEnabled()
    assert "3 remaining findings" in widget.recheck.toolTip()
    widget._recheck()
    assert rechecks == [] and "3 remaining findings" in widget.status.text()
    widget.approve_all.click()
    assert not widget.can_recheck()
    _select(qtbot, widget, "source_state")
    widget.source_state.setCurrentIndex(widget.source_state.findData("clearwater"))
    widget.set_state.click()
    assert not widget.can_recheck()
    _select(qtbot, widget, "warning")
    widget.approve_warning.click()
    assert widget.can_recheck() and widget.recheck.isEnabled() and changes[-1] == ""
    widget.recheck.click()
    assert rechecks == [widget.approval]
    assert not widget.adopt.isEnabled()  # Human choices are not a checked candidate.


def test_pending_target_is_insufficient_and_verified_rows_bind_exact_current_components(qtbot):
    widget, _ = _open(qtbot)
    widget._acquisition_actions = [dict(acquisition_state="clearwater", reason="Explicit source choice")]
    widget.pending_corrections = [{"saved": "target receipt"}]
    widget._buttons()
    assert not widget.can_recheck()
    rows = _area_rows(widget)
    wrong = deepcopy(rows)
    wrong[0]["review_row_sha256"] = "f" * 64
    widget.set_resolved_target_rows(wrong)
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    assert not widget.can_recheck()
    widget.set_resolved_target_rows(rows)
    rows[0]["review_row_sha256"] = "0" * 64
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    assert widget.can_recheck() and not widget.adopt.isEnabled()
    widget.review["runs"][0]["components"][0]["review_row_sha256"] = "e" * 64
    widget._refresh_rows()
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    assert not widget.can_recheck()


def test_other_writes_english_key_only_after_explicit_description(qtbot):
    widget, backend = _open(qtbot)
    assert tuple(widget.classification.itemData(index) for index in range(1, 5)) == (
        "water_above_sand", "air", "rig_or_frame", "other")
    widget.classification.setCurrentIndex(widget.classification.findData("other"))
    widget.reason.setText("The selected non-sand fixture was reviewed")
    widget._approve(False)
    assert backend.approvals == []
    widget.other_text.setText("The visible fixture")
    widget.approve_one.click()
    assert backend.approvals[-1][2]["classification"] == "other"
    assert backend.approvals[-1][2]["other_text"] == "The visible fixture"


@pytest.mark.parametrize("stored,label", [("Luft", "Air"), ("Annet", "Other")])
def test_legacy_class_is_readable_without_rewriting_saved_action(qtbot, stored, label):
    widget, _ = _open(qtbot)
    action = dict(_area_rows(widget)[0], classification=stored, reason="Original explicit reason",
                  other_text="Original fixture description")
    widget.approval = dict(runs=[deepcopy(action)])
    widget._refresh_rows()
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    assert widget.runs.item(0, 4).text() == label + ": Original explicit reason"
    assert widget.approval["runs"] == [action]


@pytest.mark.parametrize("severity", ["info", "error"])
def test_new_check_type_reports_list_defect_and_preserves_original_issue_semantics(qtbot, severity):
    known = dict(severity="info", code="already_listed", message="Initial information")
    widget, _ = _open(qtbot, [known])
    _resolved_inputs(widget)
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    new = dict(severity=severity, code="new_check_type", message="Original Check message")
    report = dict(issues=[known, new])
    original = deepcopy(report)
    widget.set_check_result({"synthetic": "checked candidate"}, report)
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    assert widget.coverage_defects == [new] and widget.coverage_note.isVisible()
    assert "Findings list coverage defect" in widget.coverage_note.text()
    assert "new_check_type" in widget.coverage_note.text()
    assert [row["issue"] for row in widget._issues] == original["issues"]
    assert report == original
    assert widget.can_recheck() is (severity == "info")
    assert widget.can_adopt() is (severity == "info")


def test_same_check_type_with_changed_message_is_not_reported_as_a_missing_type(qtbot):
    known = dict(severity="info", code="already_listed", message="Initial information")
    widget, _ = _open(qtbot, [known])
    widget.set_check_result({"synthetic": "checked candidate"}, dict(issues=[dict(known, message="Fresh evidence")]))
    assert widget.coverage_defects == [] and not widget.coverage_note.isVisible()
    assert widget._issues[0]["issue"]["message"] == "Fresh evidence"


def test_check_result_received_before_catalogue_still_reports_missing_type(qtbot):
    widget = _waiting(qtbot)
    issue = dict(severity="info", code="late_check_type", message="Original Check message")
    widget.set_check_result({"synthetic": "checked candidate"}, dict(issues=[issue]))
    assert widget.coverage_defects == []
    source = _review()
    source["issues"] = []
    widget._catalogue(source)
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    assert widget.coverage_defects == [issue] and widget.coverage_note.isVisible()
    assert [row["issue"] for row in widget._issues] == [issue]


def test_full_catalogue_received_before_review_remains_baseline_for_its_bound_review(qtbot):
    widget = _waiting(qtbot)
    original = dict(severity="info", code="original", message="Original review information")
    additional = dict(severity="info", code="full_input_diagnostic", message="Original input diagnostic")
    report = dict(issues=[original, additional])
    widget.set_catalogue_result(report)
    assert not widget._check_report_installed and widget.coverage_defects == []
    source = _review()
    source.update(issues=[original], review_binding=dict(path="C:/synthetic/review.json", sha256="a" * 64, size_bytes=101))
    widget._catalogue(source)
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    assert widget._initial_issue_types == {("info", "original"), ("info", "full_input_diagnostic")}
    assert not widget._check_report_installed
    widget.set_check_result({"synthetic": "checked candidate"}, report)
    widget._catalogue(deepcopy(source))
    assert widget.coverage_defects == [] and not widget.coverage_note.isVisible()
    changed = deepcopy(source)
    changed["review_binding"]["sha256"] = "b" * 64
    widget._catalogue(changed)
    assert widget._initial_issue_types == {("info", "original")}
    assert widget.coverage_defects == [additional]


def test_real_failed_check_reports_missing_error_type_without_a_store_candidate(qtbot):
    original = dict(severity="info", code="original", message="Original review information")
    widget, _ = _open(qtbot, [original])
    assert not widget._check_report_installed and widget.coverage_defects == []
    error = dict(severity="error", code="painted_review_display_context_changed",
        message="The candidate was not created and geometry validation did not run")
    report = dict(issues=[original, error], validation_not_executed=True, summary=dict(error_count=1))
    widget.set_check_result(None, report)
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    assert widget._candidate is None and widget._report == report
    assert widget._check_report_installed and widget.coverage_defects == [error]
    assert widget.coverage_note.isVisible() and error["code"] in widget.coverage_note.text()
    assert [row["issue"] for row in widget._issues] == report["issues"]
    assert not widget.can_adopt() and not widget.can_recheck()


def test_unloaded_list_and_unconfirmed_water_edit_cannot_be_checked(qtbot):
    widget = _waiting(qtbot)
    calls = []
    widget.recheck_requested.connect(calls.append)
    widget._recheck()
    assert not widget.can_recheck() and "finish loading" in widget.status.text() and calls == []
    widget._catalogue(_review())
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    _resolved_inputs(widget)
    widget._unconfirmed_water_edits.add("ac18")
    widget._buttons()
    widget._recheck()
    assert "Confirm the edited water line" in widget.status.text() and calls == []
    widget._unconfirmed_water_edits.clear()
    widget.operation_waiting = True
    widget._buttons()
    assert not widget.can_recheck() and not widget.recheck.isEnabled()


def test_dirty_display_warning_requires_fresh_catalogue_before_resolution_and_check(qtbot):
    old = dict(severity="warning", code="position_warning", message="Previous display finding", run_id="ac18")
    widget, _ = _open(qtbot, [old])
    _resolved_inputs(widget)
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    _select(qtbot, widget, "warning")
    widget.approve_warning.click()
    assert widget.can_recheck()
    source_actions = deepcopy(widget._acquisition_actions)
    area_actions = deepcopy(widget.approval)
    widget._preparation_changed = True
    widget._invalidate_display_report()
    assert widget._warning_actions == {} and not widget.approve_warning.isEnabled()
    assert not widget.can_recheck() and "Refresh findings" in widget.unresolved_findings_reason()
    refreshes, checks = [], []
    widget.refresh_findings_requested.connect(lambda: refreshes.append(True))
    widget.recheck_requested.connect(checks.append)
    widget.refresh_findings.click()
    widget._refresh_findings()
    widget._recheck()
    assert refreshes == [True] and checks == []
    assert widget._display_preferences_dirty and "Refreshing findings" in widget.unresolved_findings_reason()
    widget.catalogue_refresh_failed("The bound input is unavailable")
    assert widget._display_preferences_dirty and widget.refresh_findings.isEnabled()
    assert not widget.can_recheck() and "unavailable" in widget.status.text()
    assert widget._issues[0]["issue"] == old and widget._warning_actions == {}
    widget.refresh_findings.click()
    fresh = dict(old, message="Fresh display finding", measurement={"shift_mm": 5.0})
    report = dict(issues=[fresh])
    widget.set_catalogue_result(report)
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    assert not widget._display_preferences_dirty and not widget.refresh_findings.isVisible()
    assert widget._preparation_changed and widget._candidate is None
    assert widget.approval == area_actions and widget._acquisition_actions == source_actions
    assert widget._initial_issue_types == {("warning", "position_warning")}
    assert widget._issues[0]["issue"] == fresh and widget._warning_actions == {}
    assert not widget.can_recheck() and not widget.adopt.isEnabled()
    _select(qtbot, widget, "warning")  # Changed evidence creates a new question identity.
    widget.reason.setText("The fresh displayed evidence was reviewed")
    widget.approve_warning.click()
    assert widget.can_recheck() and not widget.adopt.isEnabled()
    widget.recheck.click()
    assert checks == [area_actions] and refreshes == [True, True]


def _literal_review():
    """Structural UI fixture only; source replay remains the worker's contract."""
    from darsia.fluidflower import painted_analysis_exclusions as core
    review = _review()
    review.update(meaning=core.MEANING, issues=[],
        review_binding=dict(path="C:/synthetic/review.json", sha256="a" * 64, size_bytes=101))
    for row in review["runs"]:
        row["findings"] = []
        context = core._digest({key: value for key, value in row.items()
                               if key not in {"components", "review_row_sha256"}})
        for component in row["components"]:
            component["review_row_sha256"] = core._digest(dict(context=context,
                component={key: value for key, value in component.items() if key != "review_row_sha256"}))
        row["review_row_sha256"] = core._row_digest(row)
    return review


def test_worker_catalogue_install_is_metadata_only_and_rejects_bad_binding_atomically(qtbot):
    source = _literal_review()
    widget, _ = _dialog(qtbot, source=source)
    widget._preparation_changed = True
    widget._invalidate_display_report()
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    widget.set_resolved_target_rows(_area_rows(widget))
    before = deepcopy(widget.review), deepcopy(widget._report), widget._display_preferences_dirty
    fresh = deepcopy(source)
    fresh["review_binding"]["sha256"] = "b" * 64
    report = dict(issues=[dict(severity="info", code="fresh_input", message="Original input diagnostic")],
        input_bindings=dict(review=deepcopy(source["review_binding"])))
    wrong = deepcopy(report)
    wrong["input_bindings"]["review"]["sha256"] = "c" * 64
    assert widget.install_catalogue(fresh, wrong) is False
    assert (widget.review, widget._report, widget._display_preferences_dirty) == before
    assert widget._resolved_target_rows
    widget._review_validator = lambda *_: pytest.fail("Catalogue installation attempted owner/source validation")

    assert widget.install_catalogue(fresh, report) is True
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)

    assert widget.review == fresh and widget._report == report
    assert widget._candidate is None and widget._preparation_changed
    assert not widget._display_preferences_dirty and widget._resolved_target_rows == set()
    assert [row["issue"] for row in widget._issues] == report["issues"]
    assert not widget.can_adopt() and widget.unresolved_findings_reason()
