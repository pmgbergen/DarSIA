"""Shrinking active inputs filters the review without rebuilding archived rows."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from darsia.fluidflower import rig_point_review as review
from darsia.fluidflower import rig_point_review_ui as ui
from darsia.fluidflower.rig_reference_quadrilateral import reference_frame_geometry
from darsia.fluidflower.series_geometry import RigFrame
from .test_rig_full_raw_editor import packet


EXCLUDED = {'ac20', 'ac24', 'ac27', 'ac51'}
PHOTO = dict(top_left=[5., 5.], top_right=[5., 70.], bottom_right=[50., 70.], bottom_left=[50., 5.])


@pytest.fixture
def context(tmp_path, qtbot, monkeypatch):
    inputs = []
    for number in range(11, 54):
        run = f'ac{number}'
        path = tmp_path / f'{run}.ARW'
        path.write_bytes(f'private {run} photograph'.encode())
        inputs.append(SimpleNamespace(run_id=run, baseline_image=path))
    source = next(row for row in inputs if row.run_id == 'ac17')
    frame = RigFrame(**reference_frame_geometry(PHOTO, measurements_m=dict(width=.9, height=.49)),
        run_id='ac17', image=str(source.baseline_image), image_size=(80, 60))
    document = review.prepare_review(frame, [dict(run_id=row.run_id, path=str(row.baseline_image),
        image_size=(80, 60)) for row in inputs])
    for run in ('ac17', 'ac18', 'ac19'):
        if document['runs'][run]['current_rc'] is None:
            document = review.update_run(document, run, PHOTO)
        document = review.approve_run(document, run, viewed_identity=document['runs'][run]['photo']['identity'],
                                      actor='Reviewer')
    active = [row for row in inputs if row.run_id not in EXCLUDED]
    document['required_runs'] = [row.run_id for row in active]
    calls = []
    project = SimpleNamespace(root=tmp_path, manifest={})
    def metadata(section, values, **kwargs):
        calls.append((section, deepcopy(values), kwargs))
        project.manifest.setdefault(section, {}).update(deepcopy(values))
    project.update_metadata = metadata
    review.save_review(project, document)
    view = QWidget()
    qtbot.addWidget(view)
    layout = QVBoxLayout(view)
    view.point_review_host = QWidget(view)
    QVBoxLayout(view.point_review_host)
    layout.addWidget(view.point_review_host)
    view.project = project
    view.current_frame = lambda: frame
    view.static_reference_runs = lambda: active
    monkeypatch.setattr(ui, '_read_photo', lambda path, profile: packet(path))
    def forbidden_rebuild(*args, **kwargs):
        pytest.fail('A matching 43-row draft must not be rebuilt for 39 active inputs')
    monkeypatch.setattr(ui, '_Prepare', forbidden_rebuild)
    panel = ui.RigPointReviewPanel(view)
    layout.addWidget(panel)
    view.resize(900, 1000)
    view.show()
    yield SimpleNamespace(view=view, panel=panel, document=document, active=active, calls=calls, frame=frame)
    panel.wait_for_workers()


def listed(panel):
    return [panel.runs.item(index, 0).data(Qt.ItemDataRole.UserRole) for index in range(panel.runs.rowCount())]


def test_43_row_saved_draft_resumes_39_active_runs_without_changing_three_approvals(context, qtbot):
    panel = context.panel
    original = json.dumps(review.load_review(context.view.project), sort_keys=True)
    calls = len(context.calls)
    panel.prepare()
    qtbot.waitUntil(lambda: panel._viewed_identity is not None)
    assert len(panel.document['runs']) == 43 and panel.runs.rowCount() == 39
    assert not EXCLUDED.intersection(listed(panel))
    assert panel.summary.text() == '3 of 39 approved'
    assert panel.scope_note.isHidden() and not panel.scope_note.text()
    assert panel.current_run() == 'ac17'
    assert all(panel.document['runs'][run] == context.document['runs'][run] for run in ('ac17', 'ac18', 'ac19'))
    assert panel.document == context.document
    assert json.dumps(review.load_review(context.view.project), sort_keys=True) == original
    assert len(context.calls) == calls
    assert not panel.use_points.isEnabled()


def test_restore_populate_filters_current_inputs_without_mutating_frozen_cohort(context, qtbot):
    panel = context.panel
    # Match the host's restore path: it assigns a document and active input map.
    panel.document = review.load_review(context.view.project)
    panel._runs = {row.run_id: row for row in context.active}
    before = deepcopy(panel.document)
    panel._populate()
    qtbot.waitUntil(lambda: panel._viewed_identity is not None)
    assert len(listed(panel)) == 39 and not EXCLUDED.intersection(listed(panel))
    assert panel._required_active_runs() == before['required_runs']
    assert panel.summary.text() == '3 of 39 approved'
    assert panel.document == before
    assert review.load_review(context.view.project) == before
