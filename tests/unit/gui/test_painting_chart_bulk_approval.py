"""Bulk contour confirmation remains atomic, source-bound and responsive."""
from copy import deepcopy
import threading
import weakref

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox

from darsia.fluidflower import painting_chart_review_ui as ui
from darsia.fluidflower import painting_chart_transfer as chart
from .test_painting_chart_review_ui import Session, preview, approve_current


RUNS = chart.RUNS[:3]


class ThreeRunSession(Session):
    def __init__(self, source):
        super().__init__(source)
        self.review['runs'] = [dict(run_id=run) for run in RUNS]


def open_review(qtbot, session_factory=ThreeRunSession, validator=None):
    reads = []

    def photo(review, run):
        reads.append(run)
        return preview(review, run)

    panel = ui.ChartReviewDialog('saved review', actor='Reviewer', session_factory=session_factory,
        preview_reader=photo, review_validator=validator)
    qtbot.addWidget(panel)
    panel.show()
    qtbot.waitUntil(lambda: not panel.busy() and panel.proposal is not None)
    return panel, reads


def confirm(monkeypatch, answer=QMessageBox.StandardButton.Yes):
    messages = []

    def question(parent, title, text, choices, default):
        messages.append((title, text, choices, default))
        return answer

    monkeypatch.setattr(QMessageBox, 'question', question)
    return messages


def test_bulk_approval_preserves_individual_records_and_current_adjustments(qtbot, monkeypatch):
    panel, reads = open_review(qtbot)
    try:
        panel.right.setValue(2.5)
        qtbot.waitUntil(lambda: not panel.busy() and panel.proposal is not None)
        approve_current(qtbot, panel)
        individual = deepcopy(panel.approvals[RUNS[0]])
        panel.runs.setCurrentIndex(1)
        qtbot.waitUntil(lambda: not panel.busy() and panel.proposal is not None)
        panel.up.setValue(1.5)
        qtbot.waitUntil(lambda: not panel.busy() and panel.proposal is not None)
        changes = []
        panel.approvals_changed.connect(lambda value: changes.append(deepcopy(value)))
        read_count = len(reads)
        messages = confirm(monkeypatch)
        panel.approve_all.click()
        qtbot.waitUntil(lambda: not panel.busy())
        assert len(messages) == 1
        assert 'all 3 runs' in messages[0][1]
        assert 'sand-layer map outlines and ColorChecker outline' in messages[0][1]
        assert 'photographs are not opened individually' in messages[0][1]
        assert messages[0][3] == QMessageBox.StandardButton.No
        assert set(panel.approvals) == set(RUNS)
        assert panel.approvals[RUNS[0]] == individual
        assert panel.approvals[RUNS[1]]['correction_mm'] == [0., 1.5]
        assert panel.approvals[RUNS[2]]['correction_mm'] == [0., 0.]
        assert len(reads) == read_count
        assert len(changes) == 1 and set(changes[0]) == set(RUNS)
        for run in RUNS:
            record = panel.approvals[run]
            assert record['actor'] == 'Reviewer'
            if run != RUNS[0]:
                assert record['reason'].startswith('Bulk approval')
            chart.apply_chart_transfer(panel.session.proposal(run,
                correction_mm=record['correction_mm']), record)
        assert panel.sand_valid.isChecked() and panel.chart_valid.isChecked()
        assert panel.reuse.isEnabled() and not panel.approve_all.isEnabled()
        panel.right.setValue(3.)
        assert set(panel.approvals) == {RUNS[0], RUNS[2]}
        assert panel.approvals[RUNS[0]] == individual
    finally:
        panel.close()


def test_cancelled_bulk_confirmation_does_not_read_or_approve(qtbot, monkeypatch):
    panel, reads = open_review(qtbot)
    try:
        before = list(reads)
        confirm(monkeypatch, QMessageBox.StandardButton.No)
        panel.approve_all.click()
        assert not panel.busy() and not panel.approvals
        assert reads == before
    finally:
        panel.close()


def test_bulk_failure_does_not_keep_partial_approvals(qtbot, monkeypatch):
    class FailingSession(ThreeRunSession):
        def proposal(self, run, *, correction_mm=None):
            if run == RUNS[2]:
                raise ValueError('Bound product changed')
            return super().proposal(run, correction_mm=correction_mm)

    panel, _ = open_review(qtbot, FailingSession)
    try:
        approve_current(qtbot, panel)
        before = deepcopy(panel.approvals)
        changes = []
        panel.approvals_changed.connect(changes.append)
        confirm(monkeypatch)
        panel.approve_all.click()
        qtbot.waitUntil(lambda: not panel.busy())
        assert panel.approvals == before and not changes
        assert panel.proposal is not None and panel.sand_valid.isChecked()
        assert 'No additional runs were approved' in panel.status.text()
        assert 'Bound product changed' in panel.status.text()
        assert panel.approve_all.isEnabled() and not panel.reuse.isEnabled()
    finally:
        panel.close()


@pytest.mark.parametrize('change', ['navigate', 'source', 'approval_context'])
def test_bulk_results_are_discarded_after_navigation_or_source_change(qtbot, monkeypatch, change):
    entered, release = threading.Event(), threading.Event()
    binding_current = [True]

    class SlowSession(ThreeRunSession):
        def proposal(self, run, *, correction_mm=None):
            if run == RUNS[2]:
                entered.set()
                assert release.wait(5)
            return super().proposal(run, correction_mm=correction_mm)

    def validate(review):
        if not binding_current[0]:
            raise ValueError('The selected saved map changed')

    panel, _ = open_review(qtbot, SlowSession, validate)
    try:
        approve_current(qtbot, panel)
        before = deepcopy(panel.approvals)
        confirm(monkeypatch)
        panel.approve_all.click()
        qtbot.waitUntil(entered.is_set)
        assert panel.bulk_approval_active
        assert not panel.runs.isEnabled() and not panel.right.isEnabled()
        assert not panel.next.isEnabled() and not panel.approve_all.isEnabled()
        changes = []
        panel.approvals_changed.connect(changes.append)
        if change == 'navigate':
            panel.runs.setCurrentIndex(1)  # An external host selection must also invalidate the batch.
        elif change == 'source':
            panel.session.review['fixture'] = 'another source'
        else:
            binding_current[0] = False
        release.set()
        qtbot.waitUntil(lambda: not panel.busy())
        assert not panel.bulk_approval_active
        assert panel.approvals == before and not changes
        assert not panel.reuse.isEnabled()
    finally:
        release.set()
        panel.close()


def test_bulk_worker_is_responsive_and_releases_each_large_proposal(qtbot, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    proposal_refs, worker_threads = [], []

    class SerialSession(ThreeRunSession):
        def proposal(self, run, *, correction_mm=None):
            if run != RUNS[0]:
                worker_threads.append(threading.get_ident())
                assert sum(reference() is not None for reference in proposal_refs) <= 1
                if run == RUNS[1]:
                    entered.set()
                    assert release.wait(5)
            result = super().proposal(run, correction_mm=correction_mm)
            proposal_refs.append(weakref.ref(result))
            return result

    panel, reads = open_review(qtbot, SerialSession)
    ticks = []
    timer = QTimer(panel)
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(True))
    try:
        confirm(monkeypatch)
        panel.approve_all.click()
        qtbot.waitUntil(entered.is_set)
        timer.start()
        qtbot.waitUntil(lambda: len(ticks) >= 3)
        assert panel.busy() and panel.bulk_approval_active
        assert not panel.approvals
        release.set()
        qtbot.waitUntil(lambda: not panel.busy())
        assert worker_threads and all(value != threading.get_ident() for value in worker_threads)
        assert sum(reference() is not None for reference in proposal_refs) == 1
        assert reads == [RUNS[0]] and set(panel.approvals) == set(RUNS)
    finally:
        timer.stop()
        release.set()
        panel.close()


def test_changed_selection_while_confirmation_is_open_cannot_approve(qtbot, monkeypatch):
    panel, _ = open_review(qtbot)
    try:
        def changed(parent, title, text, choices, default):
            panel.session.review['fixture'] = 'replaced while asking'
            return QMessageBox.StandardButton.Yes

        monkeypatch.setattr(QMessageBox, 'question', changed)
        panel.approve_all.click()
        assert not panel.busy() and not panel.approvals
        assert 'The review changed' in panel.status.text()
    finally:
        panel.close()
