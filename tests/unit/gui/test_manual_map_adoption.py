"""Already approved map pixels enter the real Check/Save without another review."""
from copy import deepcopy

import pytest
from PySide6.QtWidgets import QMessageBox

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower import painted_adoption_table as table

from ..test_chart_transfer_findings import _current_chart_case
from ..test_painted_adoption_table import (  # noqa: F401
    imported, registered_setup, synthetic_payload_admission,
)
from .test_painting_series_adoption_controller import _Host


@pytest.mark.parametrize('change_after_check', [False, True])
def test_saved_manual_approval_uses_existing_preparation_and_real_check_save(
        qtbot, registered_setup, tmp_path, monkeypatch, change_after_check):
    _, _, prepared, _ = _current_chart_case(registered_setup, tmp_path, monkeypatch)
    session, _, frame = table.verify_version(prepared['painting_facies'])
    record = dict(prepared['painting_facies']['record'],
                  folder=prepared['map_version']['folder'])
    widget = _Host(registered_setup, prepared, record, session, frame)
    widget.series_use_status.result = dict(preparation=deepcopy(prepared))
    qtbot.addWidget(widget, before_close_func=lambda owned: owned.wait_for_workers())
    widget.show()
    controller = widget.adoption_controller
    before = geometry.GeometryRevisionStore(registered_setup.project).current()['revision_id']
    questions = []
    validations = []
    original_validation = geometry.validate_geometry_config

    def validate(*args, **kwargs):
        validations.append(True)
        return original_validation(*args, **kwargs)

    def confirm(*args):
        questions.append(args[1])
        if change_after_check:
            controller.dialog.candidate['width_m'] *= 2
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(geometry, 'validate_geometry_config', validate)
    monkeypatch.setattr(QMessageBox, 'question', confirm)
    monkeypatch.setattr(widget, 'prepare_adoption_of',
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError('No new preparation')))
    monkeypatch.setattr(controller, 'open_gap_review',
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError('No new findings reader')))
    assert controller.start(widget.record, widget.selection, prepared), widget.use_map_progress.text()
    qtbot.waitUntil(lambda: not widget.busy(), timeout=10000)
    assert controller.context is None, (controller.dialog.status.text(), controller.dialog.validation_report)
    assert len(questions) == 1
    assert validations == [True], 'Save must reuse the genuine Check instead of validating again'
    if change_after_check:
        assert controller.dialog.revision is None
        assert 'not saved' in controller.dialog.status.text()
        assert geometry.GeometryRevisionStore(registered_setup.project).current()['revision_id'] == before
    else:
        assert controller.dialog.revision is not None, controller.dialog.status.text()
        assert geometry.GeometryRevisionStore(registered_setup.project).current()['revision_id'] != before
    assert controller.preparation['folder'] == prepared['folder']
    assert controller.dialog.validation_report['summary']['error_count'] == 0
    assert any(issue['severity'] == 'warning'
               for issue in controller.dialog.validation_report['issues'])


def test_manual_check_capture_keeps_structural_replay_errors(
        qtbot, registered_setup, tmp_path, monkeypatch):
    from pathlib import Path
    from darsia.fluidflower.map_adoption_view import _CheckWorker
    from darsia.fluidflower.local_geometry_check import CheckedGeometryCandidate
    _, _, prepared, _ = _current_chart_case(registered_setup, tmp_path, monkeypatch)
    original_candidate = adoption.adoption_candidate

    def incomplete_provenance(*args, **kwargs):
        candidate, summary = original_candidate(*args, **kwargs)
        # This takes the new manual-approval branch. The actual validator must
        # still reject a chart product with its required reuse provenance lost.
        candidate['provenance'].pop('local_map_reuse', None)
        return candidate, summary

    monkeypatch.setattr(adoption, 'adoption_candidate', incomplete_provenance)
    result = []
    worker = _CheckWorker(registered_setup.revision, Path(prepared['folder']), None)
    worker.finished.connect(result.append)
    worker.run()
    assert len(result) == 1
    assert isinstance(result[0]['local_checked'], CheckedGeometryCandidate)
    assert result[0]['errors']
    assert result[0]['validation_report']['summary']['error_count'] > 0
