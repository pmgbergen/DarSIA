"""Original tiny bound metadata plus finite display/lifetime seams.

No campaign result, physical decision or source acceptance is inferred from a
finite display callback. The original metadata reader is tested separately.
"""
from copy import deepcopy
import json
from pathlib import Path
import threading

import pytest

from darsia.fluidflower import painting_recorded_review_ui as subject
from darsia.fluidflower import painted_analysis_exclusions as exclusions
from darsia.fluidflower.pipeline.job_queue import descriptor
from tests.unit.test_painted_analysis_exclusions import (
    gap_case, registered_setup, imported, synthetic_payload_admission,  # noqa: F401
)
from .test_painting_guided_review_ui import GuidedBackend, _bound_review, _settle

ACTION = dict(actor="finite human", approved_utc="2026-01-01T00:00:00Z",
              reason="Inspect this finite recorded choice", decision_ref="Human review")


def _write(path, document):
    path.write_bytes((json.dumps(document, indent=2) + "\n").encode())
    return descriptor(path)


def _snapshot(case, tmp_path):
    keys = [dict(run_id=row["run_id"], component_id=c["component_id"])
            for row in case.review["runs"] for c in row["components"]]
    approval = exclusions.make_gap_approval(case.review, keys, classification="air", **ACTION)
    issues = deepcopy(case.review.get("issues", []))
    issues.append(dict(severity="warning", code="finite_review_note", message="Inspect recorded note"))
    report = _write(tmp_path / "recorded-report.json", dict(issues=issues))
    choices = dict(analysis_exclusions=approval,
                   source_choices=[dict(acquisition_state="indicator", **ACTION)],
                   warnings=[dict(issue_index=len(issues)-1,
                                  issue_sha256=exclusions.issue_identity(issues[-1]), **ACTION)])
    path = tmp_path / "recorded-decisions.json"
    _write(path, dict(format=subject.FORMAT, review=case.review["review_binding"],
                     report=report, decisions=choices))
    return path, choices


def test_original_bound_reader_retains_actions_and_changes_no_source(gap_case, tmp_path):
    path, choices = _snapshot(gap_case, tmp_path)
    sources = [path, Path(gap_case.review["review_binding"]["path"]),
               tmp_path / "recorded-report.json"]
    before = {p: p.read_bytes() for p in sources}
    result = subject.read_recorded_review(path)
    assert result["decisions"] == choices
    assert result["approval"] == choices["analysis_exclusions"]
    assert result["review"]["review_binding"] == gap_case.review["review_binding"]
    assert before == {p: p.read_bytes() for p in sources}


@pytest.mark.parametrize("fault", ["report_bytes", "warning_identity"])
def test_changed_bound_report_or_warning_is_never_displayed_as_recorded(gap_case, tmp_path, fault):
    path, _choices = _snapshot(gap_case, tmp_path)
    if fault == "report_bytes":
        report = tmp_path / "recorded-report.json"
        report.write_bytes(report.read_bytes() + b" ")
    else:
        doc = json.loads(path.read_bytes())
        doc["decisions"]["warnings"][0]["issue_sha256"] = "0" * 64
        _write(path, doc)
    with pytest.raises(ValueError):
        subject.read_recorded_review(path)


def _display_payload():
    review = _bound_review()
    approval = dict(runs=[dict(run_id="ac18", component_id="c"*64,
        review_row_sha256="c"*64, classification="air", other_text="", **ACTION)])
    return dict(snapshot=dict(path="C:/finite/recorded.json", sha256="1"*64, size_bytes=1),
                review=review, report=dict(issues=[]), approval=approval,
                decisions=dict(analysis_exclusions=approval,
                               source_choices=[dict(acquisition_state="indicator", **ACTION)], warnings=[]))


def test_guided_inspection_retains_choices_and_every_mutating_callback_is_inert(qtbot, monkeypatch):
    payload, backend = _display_payload(), GuidedBackend()
    monkeypatch.setattr(subject, "read_recorded_review", lambda _source: deepcopy(payload))
    widget = subject.RecordedGuidedReviewDialog(
        "finite-read-seam", preview_reader=backend.preview, comparison_reader=backend.comparison)
    qtbot.addWidget(widget, before_close_func=lambda owned: owned.wait_for_workers())
    widget.show()
    _settle(qtbot, widget)
    before = deepcopy((widget.approval, widget._acquisition_actions, widget._warning_actions))
    actions = []
    widget.recheck_requested.connect(actions.append)
    widget.adopt_requested.connect(actions.append)
    widget.state_requested.connect(actions.append)
    for name in ("_approve", "_approve_warning", "_set_state", "_fix", "_adopt",
                 "_paint", "_recheck", "_tolerances", "_refresh_findings",
                 "_water_tolerances", "_water_pick_requested", "_water_point_clicked",
                 "_water_confirmed", "_water_line_edited", "_clear_water_confirmations",
                 "_point_selected", "_region_requested", "_sand_requested",
                 "_reset_region_preview", "_region_draft_changed", "_prepare_water_source_review",
                 "_editor_photo_requested"):
        assert getattr(widget, name)() is False
    for _ in range(len(widget._questions)-1):
        widget.next_button.click()
        _settle(qtbot, widget)
    assert widget.guided_stage == "adopt"
    assert not widget.can_recheck() and not widget.can_adopt()
    assert not widget.recheck.isEnabled() and not widget.adopt.isEnabled()
    assert widget.deviation_editor is None and widget._candidate is None
    assert before == (widget.approval, widget._acquisition_actions, widget._warning_actions)
    assert actions == [] and backend.approvals == []
    assert "read only" in widget.guided_status.text()
    assert all(not main for _run, main in backend.reads)


def test_initial_recorded_read_remains_owned_until_natural_retirement(qtbot, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    observed = []
    def blocked(_source):
        observed.append(threading.current_thread() is threading.main_thread())
        entered.set()
        if not release.wait(5):
            raise RuntimeError("Finite read release missing")
        return _display_payload()
    monkeypatch.setattr(subject, "read_recorded_review", blocked)
    backend = GuidedBackend()
    widget = subject.RecordedGuidedReviewDialog("finite-read-seam", preview_reader=backend.preview)
    qtbot.addWidget(widget, before_close_func=lambda owned: owned.wait_for_workers())
    widget.show()
    try:
        qtbot.waitUntil(entered.is_set)
        thread = widget.thread
        assert widget.busy() and thread is not None
        widget.close()
        assert widget.thread is thread
        assert not widget.can_adopt() and not widget.can_recheck()
        release.set()
        qtbot.waitUntil(lambda: widget.thread is None and not widget.isVisible())
        assert widget.worker is None and widget.approval is None
        assert observed == [False]
    finally:
        release.set()
        widget.wait_for_workers()


def test_existing_controller_keeps_recorded_window_separate_from_adoption(qtbot, monkeypatch):
    from PySide6.QtWidgets import QWidget, QLabel
    from darsia.fluidflower.painting_series_adoption_ui import SeriesAdoptionController
    host = QWidget()
    host.busy = lambda: False
    host.use_map_progress = QLabel(host)
    qtbot.addWidget(host)
    controller = SeriesAdoptionController(host)
    context, preparation = dict(original_context=True), dict(original_preparation=True)
    controller.context, controller.preparation = context, preparation
    controller._source_choices = [dict(retain_original_choice=True)]
    saved_sources = deepcopy(controller._source_choices)
    backend = GuidedBackend()
    actual_dialog = subject.RecordedGuidedReviewDialog
    monkeypatch.setattr(subject, "read_recorded_review", lambda _source: _display_payload())
    def display_only(source, *, parent):
        return actual_dialog(source, parent=parent, preview_reader=backend.preview,
                             comparison_reader=backend.comparison)
    monkeypatch.setattr(subject, "RecordedGuidedReviewDialog", display_only)
    widget = controller.open_recorded_review("finite-read-seam")
    try:
        _settle(qtbot, widget)
        assert widget is controller.recorded_dialog is host._recorded_review_dialog
        assert controller.context is context and controller.preparation is preparation
        assert controller.gap_dialog is None and controller.dialog is None
        assert controller._source_choices == saved_sources
        assert not widget.can_adopt() and not widget.can_recheck()
    finally:
        widget.wait_for_workers()
        widget.close()
