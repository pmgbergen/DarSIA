"""Rig steps share their live controllers and leave painting routes unchanged."""
from copy import deepcopy

from darsia.fluidflower import series_rig_frame_panel as rig_ui
from .test_rig_full_raw_editor import loaded, packet
from .test_painting_raw_reference_on_u import raw_view, raw_file, profile, renders, settings  # noqa: F401
from .test_rig_point_review_ui import context as review_context  # noqa: F401


def rig_sections(view):
    return (view.step_frame, view.step_injection, view.step_analysis_boxes, view.step_point_review)


def test_rig_route_keeps_analysis_boxes_separate_and_restores_painting_layout(raw_view, qtbot, monkeypatch):
    view = raw_view
    view._rig_photo_loader.reader = lambda path, profile: packet(path)
    before = deepcopy(view.session.to_dict())
    view.set_workflow_scope('rig_frame')
    qtbot.waitUntil(lambda: loaded(view))
    assert [section.toggle.text() for section in rig_sections(view)] == [
        'Reference quad', 'Injection points', 'Analysis boxes',
        'Review reference points on all runs']
    assert all(not section.isHidden() for section in (view.step_frame, view.step_injection, view.step_point_review))
    assert view.step_analysis_boxes.isHidden() and view.analysis_boxes.widget.isHidden()
    assert view.step_injection.isAncestorOf(view.rig_panel.injection_widget)
    assert not view.rig_panel.isAncestorOf(view.rig_panel.injection_widget)
    assert view.step_analysis_boxes.isAncestorOf(view.analysis_boxes.widget)
    assert view.step_point_review.isAncestorOf(view.point_review_host)
    assert view.point_review_host.layout() is not None
    for widget in (view.measured_edges.widget, view.other_frames.widget,
                   view.measured_geometry_button, view.rig_panel.review_points):
        assert widget.isHidden()
    assert all(section.isHidden() for section in (view.step_paint, view.step_facies, view.step_propose, view.step_use_map))

    view.set_workflow_scope('layers')
    assert view.rig_panel.isAncestorOf(view.rig_panel.injection_widget)
    assert view.analysis_boxes_host.isAncestorOf(view.analysis_boxes.widget)
    assert view.analysis_boxes.widget.isHidden()
    assert view.rig_panel.injection_widget.title() == 'Injection points'
    assert [view.step_paint.toggle.text(), view.step_facies.toggle.text(), view.step_propose.toggle.text()] == [
        'Mark layers', 'Facies', 'Versions']
    assert all(section.isHidden() for section in rig_sections(view)[1:])
    assert not view.measured_edges.widget.isHidden() and not view.other_frames.widget.isHidden()
    assert view.session.to_dict() == before
    monkeypatch.setattr(view.adoption_controller, 'open_chart_review', lambda **kwargs: None)
    view.set_workflow_scope('apply')
    assert view.step_use_map.is_expanded() and not view.step_use_map.isHidden()
    assert all(section.isHidden() for section in rig_sections(view))
    assert all(section.isHidden() for section in (view.step_paint, view.step_facies, view.step_propose))


def test_reparented_injection_controls_keep_controller_enable_and_save_signals(raw_view, qtbot, monkeypatch):
    view = raw_view
    view._rig_photo_loader.reader = lambda path, profile: packet(path)
    view.set_workflow_scope('rig_frame')
    qtbot.waitUntil(lambda: loaded(view))
    panel = view.rig_panel
    saved = []
    monkeypatch.setattr(rig_ui, 'save_injection_points', lambda project, points, **kwargs: saved.append(points))
    view.step_injection.set_expanded(True)
    panel.setEnabled(False)
    assert not panel.injection_widget.isEnabled() and not panel.points.isEnabled()
    assert not panel.place_point.isEnabled()
    panel.setEnabled(True)
    assert panel.points.isEnabled() and panel.place_point.isEnabled()
    previous = panel.points.rowCount()
    panel._add_point()
    assert panel.points.rowCount() == previous + 1
    panel.points.item(previous, 1).setText('New injection port')
    assert saved[-1][-1]['name'] == 'New injection port'
    view.set_workflow_scope('layers')
    assert panel.points.rowCount() == previous + 1
    assert panel.points.item(previous, 1).text() == 'New injection port'


def test_review_section_hosts_run_list_and_disables_reparented_injection_controls(review_context, qtbot):
    view, panel, _ = review_context
    view.resize(1600, 1200)
    qtbot.waitUntil(lambda: panel.runs.height() > view.steps_scroll.viewport().height() / 2)
    before = deepcopy(panel.document)
    assert view.step_point_review.is_expanded()
    assert all(not section.is_expanded() for section in rig_sections(view)[:3])
    assert view.point_review_host.isAncestorOf(panel.runs)
    assert not panel.isAncestorOf(panel.runs)
    assert not view.rig_panel.injection_widget.isEnabled()
    assert not view.rig_panel.points.isEnabled()
    assert view.run_preview.stack.currentWidget() is panel
    view.step_frame.toggle.click()
    qtbot.waitUntil(lambda: not view._point_review_active and loaded(view))
    assert not view.step_point_review.is_expanded()
    assert view.step_frame.is_expanded()
    assert view._steps_layout.stretch(view._steps_layout.indexOf(view.step_point_review)) == 0
    assert view._steps_layout.stretch(view._steps_bottom_spacer) == 1
    assert view.rig_panel.injection_widget.isEnabled()
    assert panel.document == before
    view.step_point_review.toggle.click()
    qtbot.waitUntil(lambda: view._point_review_active and panel._viewed_identity is not None)
    assert view.run_preview.stack.currentWidget() is panel
    assert view.point_review_host.isAncestorOf(panel.runs)
    assert panel.document == before


def test_opening_review_without_complete_frame_reveals_reference_error(raw_view, monkeypatch):
    view = raw_view
    monkeypatch.setattr(view, 'current_frame', lambda: None)
    view.step_frame.set_expanded(False)
    view.step_point_review.set_expanded(True)
    view.open_reference_point_review()
    assert view.step_frame.is_expanded() and not view.step_point_review.is_expanded()
    assert 'Define four visible reference points' in view.rig_panel.reference_error.text()
    assert view._steps_layout.stretch(view._steps_bottom_spacer) == 1
