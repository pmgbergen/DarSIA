"""Named reference edits are explicit, one-shot and independent of injectors."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt

from darsia.fluidflower import rig_point_review as review
from darsia.fluidflower import series_rig_frame_panel as panel_ui
from darsia.fluidflower.rig_reference_quadrilateral import reference_frame_geometry
from darsia.fluidflower.series_geometry import RigFrame
from .test_rig_full_raw_editor import click_u, loaded, packet
from .test_painting_raw_reference_on_u import raw_view, raw_file, profile, renders, settings  # noqa: F401


PHOTO = dict(top_left=[5., 5.], top_right=[5., 70.], bottom_right=[50., 70.], bottom_left=[40., 5.])


@pytest.fixture
def context(raw_view, raw_file, qtbot, monkeypatch):
    view = raw_view
    view._rig_photo_loader.reader = lambda path, profile: packet(path)
    view.set_workflow_scope('rig_frame')
    qtbot.waitUntil(lambda: loaded(view))
    panel = view.rig_panel
    frame = RigFrame(**reference_frame_geometry(PHOTO, measurements_m=dict(width=.9, height=.49)),
        run_id=view.reference().run_id, image=str(raw_file), image_size=(80, 60))
    panel.frame = frame
    panel._loading = True
    panel._show_frame()
    panel.points.setRowCount(0)
    for location, x, y in (('ip1', .45, .02), ('ip2', .72, .18)):
        panel._append_point(dict(location_id=location, name=location, x_m=x, y_m=y, pixel=None, source='typed'))
    panel._refresh_point_pixels()
    panel._loading = False
    saved = []
    monkeypatch.setattr(panel_ui, 'save_rig_frame', lambda project, frame, **kwargs: saved.append((deepcopy(frame), kwargs)))
    monkeypatch.setattr(panel_ui, 'save_injection_points', lambda *args, **kwargs:
        pytest.fail('Changing reference points must never save injection metadata'))
    return SimpleNamespace(view=view, panel=panel, frame=frame, saved=saved, raw=raw_file)


def test_selected_named_point_uses_exactly_one_click_then_exits_without_injector_edits(context, qtbot):
    view, panel = context.view, context.panel
    injectors = deepcopy(panel.current_points())
    panel.corners.selectRow(0)
    assert panel.place_reference_point.isEnabled()
    panel.place_reference_point.click()
    assert panel._mode == 'reference_point' and view.canvas.place_mode
    assert 'Top left' in panel.status.text() and 'Top left' in panel.reference_target.text()
    assert not panel.place_point.isChecked() and not panel.mark_corners.isChecked()
    click_u(qtbot, view, 7., 7.)
    assert panel._mode == '' and not view.canvas.place_mode and not panel.place_reference_point.isChecked()
    assert len(context.saved) == 1
    points = panel.reference_photo_corners()
    np.testing.assert_allclose(points['top_left'], [7., 7.], atol=.3)
    for name in PHOTO.keys() - {'top_left'}:
        assert points[name] == tuple(PHOTO[name])
    click_u(qtbot, view, 20., 30.)
    assert panel.reference_photo_corners() == points and len(context.saved) == 1
    assert panel.current_points() == injectors


def test_valid_named_row_edit_never_sorts_other_points_by_screen_height(context):
    panel = context.panel
    before = deepcopy(panel.reference_photo_corners())
    # A valid convex quad whose TR is below BL on screen: sorting by row would
    # silently exchange identities, although only the named TR was edited.
    panel.corners.item(1, 1).setText('45; 90')
    assert not panel.reference_error.text()
    assert panel.reference_photo_corners()['top_right'] == (45., 90.)
    for name in before.keys() - {'top_right'}:
        assert panel.reference_photo_corners()[name] == before[name]
    restored = RigFrame.from_dict(context.saved[-1][0].to_dict())
    assert restored.reference_geometry['photo_corners_rc']['top_right'] == [45., 90.]
    assert restored.reference_geometry['photo_corners_rc']['bottom_left'] == PHOTO['bottom_left']


@pytest.mark.parametrize('value', ['5; 5', '60; 0', 'nan; 10'])
def test_invalid_duplicate_crossed_or_nonfinite_edit_cannot_replace_saved_valid_frame(context, value):
    panel = context.panel
    before = deepcopy(panel.frame.to_dict())
    injectors = deepcopy(panel.current_points())
    panel.corners.item(1, 1).setText(value)
    assert panel.reference_error.text() and not panel.copy_crop.isEnabled()
    assert not context.saved and panel.frame.to_dict() == before
    assert panel.current_points() == injectors


def test_selection_manual_edit_escape_and_ip_mode_cancel_reference_placement(context, qtbot):
    view, panel = context.view, context.panel
    panel.corners.selectRow(0)
    panel.place_reference_point.click()
    panel.corners.selectRow(1)
    assert not panel._mode and not panel.place_reference_point.isChecked()
    panel.place_reference_point.click()
    panel.corners.item(1, 1).setText('6; 71')
    assert not panel._mode and not panel.place_reference_point.isChecked()
    panel.place_reference_point.click()
    view.canvas.setFocus()
    qtbot.keyClick(view.canvas, Qt.Key.Key_Escape)
    assert not panel._mode and not view.canvas.place_mode
    panel.place_reference_point.click()
    panel.points.selectRow(0)
    assert not panel._mode
    panel.place_point.click()
    assert panel._mode == 'point' and not panel.place_reference_point.isChecked()
    panel.corners.selectRow(2)
    panel.place_reference_point.click()
    assert panel._mode == 'reference_point' and not panel.place_point.isChecked()
    panel.corners.clearSelection()
    assert not panel.place_reference_point.isEnabled() and not panel._mode
    panel.corners.selectRow(0)
    panel.use_host_photo((80, 60), view.canvas.preview)
    assert panel.place_reference_point.isEnabled()
    view.canvas.clear_photo()
    panel._update_reference_placement_controls()
    assert not panel.place_reference_point.isEnabled()


def install_approval(context):
    document = review.prepare_review(context.frame, [dict(run_id=context.frame.run_id,
        path=str(context.raw), image_size=(80, 60))])
    document = review.approve_run(document, context.frame.run_id,
        viewed_identity=document['runs'][context.frame.run_id]['photo']['identity'], actor='Reviewer')
    context.view.project.manifest.setdefault('tools', {}).setdefault('native_series', {})['reference_point_review'] = deepcopy(document)
    return document


def test_restore_approved_reference_recovers_geometry_without_touching_injection_data(context, monkeypatch):
    panel = context.panel
    monkeypatch.setattr(panel_ui, 'default_actor', lambda: 'reviewer')
    document = install_approval(context)
    injectors = deepcopy(panel.current_points())
    panel.corners.item(0, 1).setText('7; 7')
    assert panel.restore_reference.isEnabled()
    assert panel.reference_photo_corners()['top_left'] == (7., 7.)
    panel.restore_reference.click()
    assert panel.frame.to_dict() == review.reviewed_frame(document, context.frame.run_id).to_dict()
    assert panel.reference_photo_corners() == {name: tuple(point) for name, point in PHOTO.items()}
    assert panel.current_points() == injectors
    assert context.saved[-1][1]['actor'] == 'reviewer'
    assert review.load_review(context.view.project) == document


@pytest.mark.parametrize('changed', ['run', 'photo', 'stamp'])
def test_restore_rejects_noncurrent_run_wrong_photo_or_changed_photo(context, tmp_path, monkeypatch, changed):
    panel = context.panel
    install_approval(context)
    reference = panel.reference()
    if changed == 'run':
        monkeypatch.setattr(panel, '_reference', lambda: replace(reference, run_id='another'))
    elif changed == 'photo':
        other = tmp_path / 'other.ARW'
        other.write_bytes(b'other photograph')
        monkeypatch.setattr(panel, '_reference', lambda: replace(reference, baseline_image=other))
    else:
        context.raw.write_bytes(context.raw.read_bytes() + b'changed')
    before = deepcopy(panel.frame.to_dict())
    panel._restore_approved_reference()
    assert not context.saved and panel.frame.to_dict() == before
    assert not panel.restore_reference.isEnabled()
    assert 'No current approved reference' in panel.reference_error.text()
