"""Sidebar geometry routes share one editor and retain its review decisions."""
from copy import deepcopy

from PySide6.QtWidgets import QApplication

from darsia.fluidflower.painting_chart_review_ui import ChartReviewDialog

from .test_native_sidebar import native_window  # noqa: F401
from .test_painting_corrections import view  # noqa: F401
from .test_painting_chart_review_ui import Session, preview, approve_current
from .test_painting_manual_step4_route import _saved_manual


def _install_shared_view(window, view):
    workspace = window.geometry_workspace
    workspace.simple_view = view
    workspace.view_stack.insertWidget(0, view)
    workspace.view_stack.setCurrentWidget(view)
    workspace.pages.setCurrentWidget(workspace.page_host)
    return workspace


def test_layers_and_apply_menu_scopes_share_photo_and_preserve_review_state(native_window, view, qtbot, monkeypatch):
    window = native_window
    workspace = _install_shared_view(window, view)
    record = _saved_manual(view, qtbot)
    approval = view.approve_version(record)
    assert approval is not None
    panel = ChartReviewDialog('fixture', actor='Reviewer', parent=view.run_preview.stack,
        session_factory=Session, preview_reader=preview)
    view.adoption_controller.chart_dialog = panel
    retained_context = {'fixture': 'same selected saved map'}
    view.adoption_controller._chart_context = retained_context
    monkeypatch.setattr(view.adoption_controller, '_findings_context_snapshot', lambda: retained_context)
    view.show_transfer_review(panel)
    qtbot.waitUntil(lambda: not panel.busy() and panel.proposal is not None)
    panel.right.setValue(2.5)
    qtbot.waitUntil(lambda: not panel.busy() and panel.proposal is not None)
    approve_current(qtbot, panel)
    decisions = deepcopy(panel.approvals)
    windows = {widget for widget in QApplication.topLevelWidgets() if widget.isVisible()}
    def no_legacy_export():
        raise AssertionError('Menu selection must not open a separate legacy export workflow.')
    monkeypatch.setattr(window.crop_workflow, 'open_current_export', no_legacy_export)
    window._select_row(('geometry_facies', 'layers'))
    assert workspace.simple_view is view and window.content_stack.currentWidget() is workspace
    assert all(section.isVisible() for section in (view.step_frame, view.step_paint, view.step_facies, view.step_propose))
    assert view.step_use_map.isHidden() and view.import_map_start.isVisible()
    assert not view._transfer_review_active
    window._select_row(('geometry_facies', 'export'))
    assert workspace.simple_view is view and view.step_use_map.isVisible()
    assert all(section.isHidden() for section in (view.step_frame, view.step_paint, view.step_facies, view.step_propose))
    assert view.import_map_start.isHidden() and view.workflow_intro.isHidden()
    assert view.run_preview.stack.currentWidget() is panel and not panel.isWindow()
    assert view._selected_version()['folder'] == record['folder']
    assert view._approval_for_version(record) == approval
    assert panel.approvals == decisions and panel.right.value() == 2.5
    assert panel.sand_valid.isChecked() and panel.chart_valid.isChecked()
    assert {widget for widget in QApplication.topLevelWidgets() if widget.isVisible()} == windows - {view}
    window._select_row(('geometry_facies', 'layers'))
    assert workspace.simple_view is view and panel.approvals == decisions
    assert view.step_use_map.isHidden() and view.step_propose.isVisible()
    window._select_row(('geometry_facies', 'export'))
    assert view.run_preview.stack.currentWidget() is panel and panel.approvals == decisions


def test_rig_menu_shows_only_rig_and_step5_link_routes_to_apply_menu(native_window, view):
    window = native_window
    _install_shared_view(window, view)
    window._select_row(('geometry_facies', 'rig_frame'))
    assert view.step_frame.isVisible()
    # The crop choice sits with the reference points; by default Setup crops to the reference quad.
    crop = view.rig_panel
    assert view.step_frame.content.isAncestorOf(crop.crop_same)
    assert crop.crop_same.isChecked() and crop.crop_same.isEnabled()
    assert crop.mark_crop.isHidden() and "reference quadrilateral" in crop.crop_status.text()
    assert all(section.isHidden() for section in (view.step_paint, view.step_facies,
                                                 view.step_propose, view.step_use_map))
    window._select_row(('setup', 'depth'))
    assert window.sidebar.get_selected() == ('geometry_facies', 'rig_frame')
    assert view.step_depth.is_expanded()
    assert view.rig_depth_host.isAncestorOf(window.depth_workflow.panel)
    window._select_row(('geometry_facies', 'layers'))
    view._open_round5_step()
    assert window.sidebar.get_selected() == ('geometry_facies', 'export')
    assert view.step_use_map.isVisible() and view.step_frame.isHidden()
