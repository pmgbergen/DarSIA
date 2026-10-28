"""Human chart review retains exact per-run choices and invalidates edited ones."""
from copy import deepcopy
import numpy as np
import pytest
from PySide6.QtCore import QRectF, Qt

from darsia.fluidflower.painting_chart_review_ui import ChartReviewDialog
from darsia.fluidflower import painting_chart_transfer as chart
from tests.unit.test_painting_chart_transfer import inputs
from tests.unit.test_painting_chart_approval_store import bound_session


class Session:
    def __init__(self, source):
        self.review = {'fixture': source, 'runs': [dict(run_id=run) for run in chart.RUNS]}
        self.roles = {2: 'free_water', 3: 'porous_medium', 17: 'colorchecker', 18: 'excluded'}
        self.manifest_binding = {'fixture': 'detections'}
    def proposal(self, run, *, correction_mm=None, proposal_format=chart.FORMAT):
        data = inputs()
        data['base_labels'][40:45, 40:45] = 18
        data['source_material'][40:45, 40:45] = 18
        return chart.propose_chart_transfer(run, **data, correction_mm=correction_mm,
                                            proposal_format=proposal_format)


def preview(review, run):
    return dict(run_id=run, image_rgb=np.full((60, 80, 3), 90, np.uint8),
        image_encoding='rgb_uint8', native_to_photo_h=np.eye(3).tolist(),
        continued_mask=np.zeros((60, 80), bool), own_frame_native_xy=[[0, 0], [79, 0], [79, 59], [0, 59]])


def open_dialog(qtbot):
    dialog = ChartReviewDialog('saved review', actor='Reviewer', session_factory=Session, preview_reader=preview)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    return dialog


def test_chart_overlay_shows_whole_card_outside_analysis_domain(qtbot):
    class ClippedSession(Session):
        def proposal(self, run, *, correction_mm=None):
            data = inputs()
            data['inside'][:10] = False
            data['base_labels'][~data['inside']] = 0
            data['original_chart_mask'] = data['base_labels'] == data['chart_id']
            return chart.propose_chart_transfer(run, **data, correction_mm=correction_mm)
    dialog = ChartReviewDialog('saved review', actor='Reviewer',
                              session_factory=ClippedSession, preview_reader=preview)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    cyan = next(item for item in dialog._overlays if item.pen().color().name() == '#00ffff')
    assert cyan.path().boundingRect().top() == pytest.approx(7.5)
    assert np.where(dialog.proposal.result['chart_mask'])[0].min() == 10
    dialog.overlays['chart'].setChecked(False)
    assert not any(item.pen().color().name() == '#00ffff' for item in dialog._overlays)


@pytest.mark.parametrize('complete', [True, False])
@pytest.mark.parametrize('route', ['new_transfer', 'current_review'])
def test_fresh_transfer_cannot_silently_use_the_old_cropped_display(qtbot, complete, route):
    class FreshSession(Session):
        def __init__(self, source):
            super().__init__(source)
            self.review.update(runs=[dict(run_id='sample')],
                               transfer_policy={'material_display': 'full-source-to-full-camera-photo-v1'})
            if route == 'current_review':
                from darsia.fluidflower.painting_chart_transfer import DETECTED_FORMAT
                self.review.pop('transfer_policy')
                self.decision_proposal_format = DETECTED_FORMAT
            self.restored = {'material_labels': inputs()['source_material']}
    def full_preview(review, run):
        return dict(preview(review, run), image_rgb=np.full((110, 140, 3), 90, np.uint8),
                    full_camera_photograph=complete,
                    native_to_photo_h=[[1., 0., 20.], [0., 1., 20.], [0., 0., 1.]],
                    own_frame_native_xy=[[-10., -10.], [100., -10.], [100., 80.], [-10., 80.]])
    dialog = ChartReviewDialog('new transfer', actor='Reviewer',
                              session_factory=FreshSession, preview_reader=full_preview)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: not dialog.busy() and (
        dialog.proposal is not None or 'legacy cropped preview' in dialog.status.text()))
    if complete:
        assert dialog.packet['material_display'].shape == (110, 140)
        assert dialog.packet['material_display'][75, 115] == 3
        assert not dialog.packet['material_display'][:, :10].any()
    else:
        assert dialog.proposal is None and not dialog.approve.isEnabled()


def approve_current(qtbot, dialog):
    dialog.sand_valid.setChecked(True)
    assert not dialog.approve.isEnabled()
    dialog.chart_valid.setChecked(True)
    assert dialog.approve.isEnabled()
    qtbot.mouseClick(dialog.approve, Qt.MouseButton.LeftButton)


def test_navigation_keeps_legacy_approved_motion_until_manual_edit(qtbot):
    dialog = open_dialog(qtbot)
    run = dialog.runs.currentData()
    legacy = dialog.session.proposal(run, proposal_format=chart.LEGACY_FORMAT)
    approval = chart.confirm_chart_transfer(legacy, actor='Reviewer', reason='Reviewed prior motion')
    dialog.approvals[run] = approval
    dialog.confirmations[run] = [True, True]
    dialog._load_run()
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    assert dialog.proposal.payload == legacy.payload
    assert dialog.approvals[run] == approval
    dialog.right.setValue(1.)
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    assert dialog.proposal.metadata()['format'] == chart.FORMAT
    assert run not in dialog.approvals


def test_navigation_is_passive_and_adjustment_clears_only_that_run(qtbot):
    dialog = open_dialog(qtbot)
    assert dialog.overlays['chart'].text() == 'ColorChecker'
    assert dialog.hatching.text() == 'Hatch excluded ColorChecker'
    assert not hasattr(dialog, 'reason')
    assert dialog.sand_valid.text() == 'The sand-layer map outlines are valid in this photograph'
    assert dialog.chart_valid.text() == 'The ColorChecker outline is correct in this photograph'
    assert dialog.windowTitle() == 'Review the transfer — ColorChecker masks'
    assert not dialog.hatching.isChecked()
    assert not hasattr(dialog, 'sand')
    assert not dialog.approvals and not dialog.reuse.isEnabled()
    assert not dialog.photo.item.pixmap().isNull()
    first = dialog.runs.currentData()
    assert not dialog.approve.isEnabled()
    approve_current(qtbot, dialog)
    approval = deepcopy(dialog.approvals[first])
    assert approval['actor'] == 'Reviewer'
    assert approval['reason'] == dialog.sand_valid.text() + '. ' + dialog.chart_valid.text() + '.'
    qtbot.mouseClick(dialog.next, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    second = dialog.runs.currentData()
    assert set(dialog.approvals) == {first}
    assert not dialog.sand_valid.isChecked() and not dialog.chart_valid.isChecked()
    approve_current(qtbot, dialog)
    dialog.overlays['chart'].setChecked(False)
    dialog.right.setValue(3.)
    assert set(dialog.approvals) == {first}
    assert dialog.approvals[first] == approval
    assert not dialog.approve.isEnabled()
    assert dialog.overlays['chart'].isChecked()
    assert not dialog.sand_valid.isChecked() and not dialog.chart_valid.isChecked()
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    assert dialog.proposal.metadata()['correction_mm'] == [3., 0.]
    assert dialog.proposal.metadata()['margin_mm'] == 5.
    approve_current(qtbot, dialog)
    assert dialog.approvals[second]['correction_mm'] == [3., 0.]
    dialog.close()
    assert not dialog.busy()


def test_reuse_requires_all_39_explicit_approvals(qtbot):
    dialog = open_dialog(qtbot)
    requests = []
    dialog.reuse_requested.connect(requests.append)
    for index in range(dialog.runs.count()):
        dialog.runs.setCurrentIndex(index)
        qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
        approve_current(qtbot, dialog)
        assert dialog.reuse.isEnabled() == (index == 38)
    qtbot.mouseClick(dialog.reuse, Qt.MouseButton.LeftButton)
    assert len(requests) == 1
    assert set(requests[0]['approvals']) == set(chart.RUNS)
    assert dialog.counter.text() == '39 of 39 approved'
    dialog.up.setValue(1.)
    assert not dialog.reuse.isEnabled()
    dialog.close()


def test_39_saved_approvals_reopen_and_an_edit_withdraws_only_that_run(qtbot, monkeypatch, bound_session):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **kw: QMessageBox.StandardButton.Yes)
    def opened():
        dialog = ChartReviewDialog('finite bound review', actor='Reviewer',
            session_factory=bound_session.factory, preview_reader=preview)
        qtbot.addWidget(dialog)
        dialog.show()
        qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
        return dialog
    first = opened()
    first._approve_all()
    qtbot.waitUntil(lambda: not first.busy() and len(first.approvals) == 39, timeout=20000)
    records = deepcopy(first.approvals)
    first.close()
    second = opened()
    assert second.approvals == records and second.reuse.isEnabled()
    assert second.counter.text() == '39 of 39 approved'
    selected = second.runs.currentData()
    second.right.setValue(1.)
    assert set(second.approvals) == set(records) - {selected}
    second.close()
    third = opened()
    assert set(third.approvals) == set(records) - {selected}
    assert all(third.approvals[run] == records[run] for run in third.approvals)
    assert third.right.value() == 1. and not third.reuse.isEnabled()
    assert not third.sand_valid.isChecked() and not third.chart_valid.isChecked()
    third.close()


def test_compact_chart_controls_remain_reachable_in_800_by_500_panel(qtbot):
    dialog = ChartReviewDialog('finite review', actor='Reviewer', session_factory=Session, preview_reader=preview)
    qtbot.addWidget(dialog)
    dialog.resize(800, 500)
    dialog.show()
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    assert dialog.width() <= 800 and dialog.height() <= 500
    scroll = dialog.controls_scroll
    assert dialog.rect().contains(scroll.geometry()) and scroll.height() <= 260
    assert dialog.photo.height() >= 80
    for control in (dialog.right, dialog.up, dialog.sand_valid, dialog.chart_valid,
                    dialog.approve, dialog.approve_all, dialog.reuse):
        scroll.ensureWidgetVisible(control, 0, 0)
        qtbot.wait(1)
        centre = scroll.viewport().mapFromGlobal(control.mapToGlobal(control.rect().center()))
        assert scroll.viewport().rect().contains(centre)
    dialog.close()


def test_all_overlays_are_independent_and_preserved_per_run(qtbot):
    dialog = open_dialog(qtbot)
    assert set(dialog.overlays) == {'sand', 'chart', 'continued', 'rig_frame', 'silicone', 'gaps'}
    before = {key: control.isChecked() for key, control in dialog.overlays.items()}
    dialog.photo.scale(2., 2.)
    zoomed = dialog.photo.transform()
    for key, control in dialog.overlays.items():
        control.setChecked(not before[key])
        assert all(other.isChecked() == before[name] for name, other in dialog.overlays.items() if name != key)
        control.setChecked(before[key])
        assert dialog.photo.transform() == zoomed
    dialog.overlays['sand'].setChecked(False)
    dialog.overlays['rig_frame'].setChecked(False)
    dialog.overlays['silicone'].setChecked(False)
    dialog.hatching.setChecked(True)
    qtbot.mouseClick(dialog.next, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    assert dialog.overlays['sand'].isChecked() and not dialog.hatching.isChecked()
    assert dialog.overlays['rig_frame'].isChecked() and dialog.overlays['silicone'].isChecked()
    qtbot.mouseClick(dialog.back, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    assert not dialog.overlays['sand'].isChecked() and dialog.hatching.isChecked()
    assert not dialog.overlays['rig_frame'].isChecked() and not dialog.overlays['silicone'].isChecked()
    assert dialog.right.isEnabled() and dialog.up.isEnabled()
    dialog.close()


def test_reactive_adjustment_applies_only_latest_request_and_keeps_photo_view(qtbot):
    import threading
    entered, release = threading.Event(), threading.Event()
    calls, reads, accepted = [], [], []
    class SlowSession(Session):
        def proposal(self, run, *, correction_mm=None):
            correction = list(correction_mm or [0., 0.])
            calls.append(correction)
            if correction == [1., 0.]:
                entered.set()
                assert release.wait(5)
            return super().proposal(run, correction_mm=correction)
    def photo(review, run):
        reads.append(run)
        return preview(review, run)
    dialog = ChartReviewDialog('saved review', actor='Reviewer', session_factory=SlowSession, preview_reader=photo)
    qtbot.addWidget(dialog)
    dialog.show()
    try:
        qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
        approve_current(qtbot, dialog)
        dialog.proposal_ready.connect(lambda metadata: accepted.append(metadata['correction_mm']))
        dialog.photo.scale(3., 3.)
        transform = dialog.photo.transform()
        centre = dialog.photo.mapToScene(dialog.photo.viewport().rect().center())
        dialog.overlays['chart'].setChecked(False)
        dialog.right.setValue(1.)
        qtbot.waitUntil(entered.is_set)
        assert dialog.right.isEnabled() and dialog.up.isEnabled()
        dialog.right.setValue(2.)
        dialog.up.setValue(3.)
        dialog.right.setValue(4.)
        qtbot.wait(250)
        release.set()
        qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
        assert dialog.proposal.metadata()['correction_mm'] == [4., 3.]
        assert accepted == [[4., 3.]]
        assert calls == [[0., 0.], [1., 0.], [4., 3.]]
        assert reads == [dialog.runs.currentData()]
        assert dialog.overlays['chart'].isChecked()
        assert dialog.photo.transform() == transform
        assert dialog.photo.mapToScene(dialog.photo.viewport().rect().center()) == centre
        assert not dialog.approvals and not dialog.approve.isEnabled()
        assert not dialog.sand_valid.isChecked() and not dialog.chart_valid.isChecked()
    finally:
        release.set()
        dialog.close()


def test_run_change_covers_old_photo_and_only_shows_the_latest_selected_run(qtbot):
    import threading
    entered, release = threading.Event(), threading.Event()
    slow_run = chart.RUNS[1]
    def read_photo(review, run):
        if run == slow_run:
            entered.set()
            assert release.wait(5)
        packet = preview(review, run)
        packet['image_rgb'].fill(chart.RUNS.index(run) + 20)
        return packet
    dialog = ChartReviewDialog('saved review', actor='Reviewer', session_factory=Session, preview_reader=read_photo)
    qtbot.addWidget(dialog)
    dialog.show()
    try:
        qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
        dialog.runs.setCurrentIndex(1)
        qtbot.waitUntil(entered.is_set)
        assert dialog.photo_loading.isVisible()
        assert slow_run.upper() in dialog.photo_loading_text.text()
        assert 'Loading photograph' in dialog.run_title.text()
        assert dialog.packet is None and dialog.proposal is None
        assert not dialog.approve.isEnabled() and not dialog.sand_valid.isEnabled()
        dialog.runs.setCurrentIndex(2)
        latest = chart.RUNS[2]
        assert latest.upper() in dialog.photo_loading_text.text()
        release.set()
        qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
        assert dialog.packet['run_id'] == latest and dialog._photo_run == latest
        assert dialog.photo.item.pixmap().toImage().pixelColor(20, 20).red() == 22
        assert dialog.photo_loading.isHidden() and not dialog._photo_load_timer.isActive()
    finally:
        release.set()
        dialog.close()


def test_failed_run_load_keeps_previous_photo_covered_and_cannot_be_approved(qtbot):
    def read_photo(review, run):
        if run == chart.RUNS[1]:
            raise ValueError('Photograph unavailable')
        return preview(review, run)
    dialog = ChartReviewDialog('saved review', actor='Reviewer', session_factory=Session, preview_reader=read_photo)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    dialog.next.click()
    qtbot.waitUntil(lambda: not dialog.busy())
    assert dialog.photo_loading.isVisible()
    assert 'Could not load' in dialog.photo_loading_text.text()
    assert not dialog._photo_load_timer.isActive()
    assert not dialog.approve.isEnabled() and dialog.proposal is None and dialog.packet is None
    dialog.close()


def test_unchecking_a_confirmation_withdraws_only_current_run_approval(qtbot):
    dialog = open_dialog(qtbot)
    approve_current(qtbot, dialog)
    first = dialog.runs.currentData()
    dialog.next.click()
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    approve_current(qtbot, dialog)
    second = dialog.runs.currentData()
    dialog.sand_valid.setChecked(False)
    assert set(dialog.approvals) == {first} and second not in dialog.approvals
    assert not dialog.approve.isEnabled()
    dialog.back.click()
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    assert dialog.sand_valid.isChecked() and dialog.chart_valid.isChecked()
    dialog.close()


def test_default_chart_preview_uses_full_camera_display_and_saved_mask(monkeypatch):
    from darsia.fluidflower.painting_chart_review_ui import read_chart_preview
    from darsia.fluidflower import painted_camera_jpg_display, painted_analysis_exclusions
    packet = preview({}, 'run_b')
    packet.update(image_rgb=np.full((90, 110, 3), 90, np.uint8),
                  native_to_photo_h=[[1., 0., 4.], [0., 1., 5.], [0., 0., 1.]])
    mask = np.ones((60, 80), bool)
    seen = []
    monkeypatch.setattr(painted_camera_jpg_display, 'read_transfer_preview', lambda review, run: packet)
    def arrays(bound):
        seen.append(bound)
        return {'continued': mask}
    monkeypatch.setattr(painted_analysis_exclusions, '_arrays', arrays)
    result = read_chart_preview({'runs': [dict(run_id='run_a', product='a'),
                                         dict(run_id='run_b', product='b')]}, 'run_b')
    assert result is packet and result['run_id'] == 'run_b'
    assert result['continued_mask'] is mask and seen == ['b']
    assert result['image_rgb'].shape == (90, 110, 3)
    np.testing.assert_array_equal(result['native_to_photo_h'], [[1., 0., 4.], [0., 1., 5.], [0., 0., 1.]])


@pytest.mark.parametrize('camera_u_grid', [False, True])
def test_common_full_photo_plane_preserves_zoom_and_pan_between_runs(qtbot, monkeypatch, camera_u_grid):
    from darsia.fluidflower.painting_chart_review_ui import _Photo
    refits = []
    original = _Photo.show_rgb
    def show_rgb(photo, image, *, refit=True):
        refits.append(refit)
        return original(photo, image, refit=refit)
    monkeypatch.setattr(_Photo, 'show_rgb', show_rgb)
    def full_preview(review, run):
        packet = preview(review, run)
        matrix = [[1., 0., 4.], [0., 1., 5.], [0., 0., 1.]]
        if run == chart.RUNS[2]:
            matrix[0][2] += 2.  # Same dimensions, but a different display plane.
        native = deepcopy(matrix)
        if camera_u_grid:
            # Different measured native placements share the same U display.
            native[0][2] += 9. * chart.RUNS.index(run)
        grid = dict(shape_hw=[120, 180], native_to_photo_h=native, runs=list(chart.RUNS))
        if camera_u_grid:
            grid['camera_u_to_photo_h'] = matrix
        packet.update(image_rgb=np.full((120, 180, 3), 30 + chart.RUNS.index(run), np.uint8),
            native_to_photo_h=native, full_camera_photograph=True, display_grid=grid)
        return packet
    dialog = ChartReviewDialog('saved review', actor='Reviewer',
        session_factory=Session, preview_reader=full_preview)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    assert refits == [True]
    dialog.photo.scale(3., 3.)
    dialog.photo.centerOn(12., 27.)
    transform = dialog.photo.transform()
    centre = dialog.photo.mapToScene(dialog.photo.viewport().rect().center())
    dialog.next.click()
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    assert refits == [True, False]
    assert dialog.photo.transform() == transform
    assert dialog.photo.mapToScene(dialog.photo.viewport().rect().center()) == centre
    assert dialog.photo.item.pixmap().toImage().pixelColor(20, 20).red() == 31
    dialog.next.click()
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    assert refits == [True, False, True]
    assert dialog.photo.transform() != transform
    assert not dialog.approvals
    dialog.close()


def test_rig_area_is_a_fixed_display_choice_across_runs_and_full_photo_returns(qtbot):
    crop = [20, 15, 100, 70]
    def full_preview(review, run):
        packet = preview(review, run)
        matrix = [[1., 0., 30. + chart.RUNS.index(run)], [0., 1., 20.], [0., 0., 1.]]
        packet.update(image_rgb=np.full((120, 180, 3), 80, np.uint8),
            native_to_photo_h=matrix, full_camera_photograph=True, review_crop_xywh=list(crop),
            display_grid=dict(shape_hw=[120, 180], camera_u_to_photo_h=np.eye(3).tolist(),
                              native_to_photo_h=matrix, runs=list(chart.RUNS), review_crop_xywh=list(crop)))
        return packet
    dialog = ChartReviewDialog('saved review', actor='Reviewer',
        session_factory=Session, preview_reader=full_preview)
    qtbot.addWidget(dialog)
    dialog.resize(1600, 950)
    dialog.show()
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    saved_review = deepcopy(dialog.session.review)
    assert dialog.photo_view.currentData() == 'full' and dialog.photo.view_rect is None
    dialog.photo_view.setCurrentIndex(dialog.photo_view.findData('rig'))
    assert dialog.photo.view_rect == QRectF(*crop)
    dialog.photo.scale(1.5, 1.5)
    dialog.photo.centerOn(70., 50.)
    transform = dialog.photo.transform()
    centre = dialog.photo.mapToScene(dialog.photo.viewport().rect().center())
    dialog.next.click()
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    assert dialog.photo_view.currentData() == 'rig' and dialog.photo.view_rect == QRectF(*crop)
    assert dialog.photo.transform() == transform
    after = dialog.photo.mapToScene(dialog.photo.viewport().rect().center())
    assert abs(after.x() - centre.x()) * abs(transform.m11()) <= 2.
    assert abs(after.y() - centre.y()) * abs(transform.m22()) <= 2.
    assert dialog.photo.item.pixmap().size().toTuple() == (180, 120)
    dialog.photo_view.setCurrentIndex(dialog.photo_view.findData('full'))
    assert dialog.photo.view_rect is None
    whole = dialog.photo.mapFromScene(dialog.photo.item.boundingRect()).boundingRect()
    assert dialog.photo.viewport().rect().adjusted(-2, -2, 2, 2).contains(whole)
    assert not dialog.approvals and dialog.session.review == saved_review
    dialog.close()


def test_rig_frame_and_silicone_are_independent_display_only_outlines(qtbot):
    dialog = open_dialog(qtbot)
    assert dialog.overlays['rig_frame'].text() == 'Rig frame'
    assert dialog.overlays['rig_frame'].isChecked()
    assert 'coverage and unmapped-gap checks' in dialog.overlays['rig_frame'].toolTip()
    assert 'does not trace the sand layers' in dialog.overlays['rig_frame'].toolTip()
    assert dialog.overlays['silicone'].text() == 'Silicone'
    assert dialog.overlays['silicone'].isChecked()
    before = {key: value.copy() for key, value in dialog.proposal.result.items()
              if isinstance(value, np.ndarray)}
    for control in dialog.overlays.values():
        control.setChecked(False)
    assert not dialog._overlays
    dialog.overlays['rig_frame'].setChecked(True)
    assert len(dialog._overlays) == 1
    frame = dialog._overlays[0].path().boundingRect()
    assert (frame.left(), frame.top(), frame.right(), frame.bottom()) == (0., 0., 79., 59.)
    dialog.overlays['rig_frame'].setChecked(False)
    dialog.overlays['silicone'].setChecked(True)
    assert len(dialog._overlays) == 1
    assert dialog._overlays[0].path().elementCount() > 0
    assert dialog._overlays[0].path().boundingRect() != frame
    dialog.overlays['rig_frame'].setChecked(True)
    assert len(dialog._overlays) == 2
    assert all(item.path().elementCount() > 0 for item in dialog._overlays)
    for key, original in before.items():
        np.testing.assert_array_equal(dialog.proposal.result[key], original)
    assert not dialog.approvals and not dialog.sand_valid.isChecked()
    dialog.close()


def test_excluded_cohort_reasons_are_shown_without_approval_rows(qtbot):
    excluded = [dict(run_id=run, reason='Saved reason' if run == 'ac20'
        else 'Outside the selected adoption cohort') for run in ('ac20', 'ac24', 'ac27', 'ac51')]
    dialog = ChartReviewDialog('saved review', actor='Reviewer', session_factory=Session,
        preview_reader=preview, excluded_runs=excluded)
    qtbot.addWidget(dialog)
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    assert 'AC20: Saved reason' in dialog.excluded.text()
    assert all(run.upper() in dialog.excluded.text() for run in ('ac24', 'ac27', 'ac51'))
    assert dialog.runs.count() == 39 and set(dialog.approvals) == set()
    dialog.close()


def test_review_reads_excluded_reasons_from_its_bound_selected_index(qtbot, tmp_path):
    import json
    from darsia.fluidflower.map_placement import GEOMETRY_FORMAT
    from darsia.fluidflower.painted_adoption_table import _bound
    path = tmp_path / 'index.json'
    path.write_text(json.dumps(dict(format=GEOMETRY_FORMAT, run_order=list(chart.RUNS),
        run_count=39, runs={run: {} for run in chart.RUNS}, reference_models_complete=True,
        excluded_runs={'ac20': 'Saved exclusion reason'})), encoding='utf8')
    def factory(source):
        session = Session(source)
        session.review.update(geometry_index=_bound(path), source={'reference_run': 'ac17'})
        return session
    dialog = ChartReviewDialog('saved review', actor='Reviewer', session_factory=factory,
        preview_reader=preview, registered_runs=[*chart.RUNS, 'ac20', 'ac24', 'ac27', 'ac51'])
    qtbot.addWidget(dialog)
    qtbot.waitUntil(lambda: not dialog.busy() and dialog.proposal is not None)
    assert 'AC20: Saved exclusion reason' in dialog.excluded.text()
    assert 'AC51: Outside the selected adoption cohort' in dialog.excluded.text()
    dialog.close()


def test_foreign_run_photograph_cannot_be_approved(qtbot):
    def wrong_preview(review, run):
        return dict(preview(review, run), run_id='wrong')
    dialog = ChartReviewDialog('saved review', actor='Reviewer', session_factory=Session,
        preview_reader=wrong_preview)
    qtbot.addWidget(dialog)
    qtbot.waitUntil(lambda: not dialog.busy() and 'another selected run' in dialog.status.text())
    assert not dialog.approve.isEnabled()
    dialog.close()


def test_reuse_worker_does_not_enter_raw_preparation(tmp_path, monkeypatch):
    from darsia.fluidflower import local_map_adoption, raw_shape_basis_reader
    from darsia.fluidflower.series_layer_view import _AdoptionWorker
    calls = []
    def prepare(review, **kwargs):
        calls.append((review, kwargs))
        return {'folder': str(kwargs['destination'])}
    monkeypatch.setattr(local_map_adoption, 'prepare_local_adoption', prepare)
    def forbidden(*args, **kwargs):
        raise AssertionError('Local reuse must not start full RAW preparation')
    monkeypatch.setattr(raw_shape_basis_reader, 'require_basis_index', forbidden)
    request = dict(review=dict(review_binding={'fixture': 'saved'}, preparation_context={'fixture': 'context'}),
        approvals={'fixture': 'approved'}, detection_manifest={'fixture': 'manifest'})
    worker = _AdoptionWorker({}, None, dict(destination=tmp_path, local_reuse=request))
    results, failures = [], []
    worker.finished.connect(results.append)
    worker.failed.connect(failures.append)
    worker.run()
    assert not failures and len(results) == 1
    assert calls[0][1]['chart_transfers'] == request['approvals']
    assert results[0]['gui_preparation_actions'] == dict(analysis_exclusions=None, target_corrections=[])


def test_gui_preserves_local_reuse_when_photo_continuation_is_selected(qtbot, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QWidget, QLabel
    from types import SimpleNamespace
    from darsia.fluidflower import series_layer_view as editor
    from darsia.fluidflower import painting_frame_models, layer_painting
    view = QWidget()
    qtbot.addWidget(view)
    view.session = SimpleNamespace(run_id='reference')
    view.project = SimpleNamespace(manifest={})
    view.versions_note = QLabel()
    view._read_approvals = lambda run: None
    view._placeable = lambda record: True
    view.current_frame = lambda: None
    view._approval_for_version = lambda record: {'approved': True}
    view._capture_review_display_context = lambda: None
    view._series_runs = lambda: ['reference']
    view._current_label_fields = lambda: {}
    monkeypatch.setattr(painting_frame_models, 'capture_frame_model_selection', lambda *a, **k: object())
    monkeypatch.setattr(layer_painting, 'painting_folder', lambda project: tmp_path / 'painting')
    captured = []
    class Captured(Exception):
        pass
    def worker(record, frame, options):
        captured.append(options)
        raise Captured()
    monkeypatch.setattr(editor, '_AdoptionWorker', worker)
    request = {'review': 'saved', 'approvals': {'reference': 'approved'}}
    import pytest
    with pytest.raises(Captured):
        editor.LayerPainterView.prepare_adoption_of(view, {'run_id': 'reference'}, 'index',
            continue_photo_edges=True, local_reuse=request)
    assert captured[0]['local_reuse'] == request


def test_chart_approval_edit_invalidates_controller_save_context(qtbot, tmp_path):
    from PySide6.QtWidgets import QWidget, QLabel
    from types import SimpleNamespace
    from darsia.fluidflower.painting_series_adoption_ui import SeriesAdoptionController
    view = QWidget()
    qtbot.addWidget(view)
    record = {'run_id': 'reference', 'name': 'saved map'}
    approval = {'saved': True}
    selection = SimpleNamespace(index_path=str(tmp_path / 'index.json'))
    view.project = SimpleNamespace(root=tmp_path)
    view.use_map_progress = QLabel()
    view._read_approvals = lambda run: None
    view._approval_for_version = lambda version: approval
    view._selected_version = lambda: record
    view.current_frame = lambda: SimpleNamespace(to_dict=lambda: {'saved': 'frame'})
    view.run_preview = SimpleNamespace(geometry_selection=lambda: selection)
    view.busy = lambda: False
    view._adoption_dialog = view._adoption_thread = None
    captured = []
    view.prepare_adoption_of = lambda record, path, **options: captured.append(options) or True
    controller = SeriesAdoptionController(view)
    controller._allow_findings_check = lambda: True
    request = {'approvals': {'reference': {'approved': True}}, 'review': 'saved review'}
    controller._chart_approvals = deepcopy(request['approvals'])
    assert controller.start(record, selection, local_reuse=request)
    assert captured[0]['local_reuse'] == request and controller._current()
    controller._chart_approvals.clear()
    assert not controller._current()
