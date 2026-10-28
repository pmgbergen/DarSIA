"""Rig-only photo controls alter graphics, never reference or painting data."""
from copy import deepcopy

from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsPolygonItem, QGraphicsSimpleTextItem

from darsia.fluidflower.rig_reference_quadrilateral import reference_frame_geometry
from darsia.fluidflower.series_geometry import RigFrame
from .test_rig_full_raw_editor import loaded, packet
from .test_painting_raw_reference_on_u import raw_view, raw_file, profile, renders, settings  # noqa: F401


PHOTO = dict(top_left=[5., 5.], top_right=[5., 70.], bottom_right=[50., 70.], bottom_left=[50., 5.])


def open_rig(view, qtbot):
    view._rig_photo_loader.reader = lambda path, profile: packet(path)
    view.set_workflow_scope('rig_frame')
    qtbot.waitUntil(lambda: loaded(view))
    frame = RigFrame(**reference_frame_geometry(PHOTO, measurements_m=dict(width=.9, height=.49)),
                     run_id=view.reference().run_id)
    view.rig_panel.frame = frame
    view.rig_panel._loading = True
    view.rig_panel._show_frame()
    view.rig_panel._loading = False
    view.rig_panel._draw_overlay()
    return frame


def graphics(panel):
    return (sum(isinstance(item, QGraphicsPolygonItem) for item in panel._overlay),
            sum(isinstance(item, QGraphicsEllipseItem) and item.zValue() == 21 for item in panel._overlay),
            sum(isinstance(item, QGraphicsSimpleTextItem) for item in panel._overlay))


def test_reference_points_numbers_and_lines_toggle_independently_without_data_changes(raw_view, qtbot):
    view = raw_view
    frame = open_rig(view, qtbot)
    before = deepcopy((frame.to_dict(), view.session.to_dict(), view.project.manifest))
    assert view.rig_photo_toolbar.isVisible()
    assert view.rig_show_points.text() == 'Points' and view.rig_show_lines.text() == 'Lines'
    assert graphics(view.rig_panel) == (1, 4, 4)
    view.rig_show_points.setChecked(False)
    assert graphics(view.rig_panel) == (1, 0, 0)
    view.rig_show_lines.setChecked(False)
    assert view.rig_panel._overlay == []
    view.rig_show_points.setChecked(True)
    assert graphics(view.rig_panel) == (0, 4, 4)
    view.rig_show_lines.setChecked(True)
    assert graphics(view.rig_panel) == (1, 4, 4)
    assert (frame.to_dict(), view.session.to_dict(), view.project.manifest) == before


def test_rig_hides_painting_and_generic_selectors_then_restores_original_visibility(raw_view, qtbot, monkeypatch):
    view = raw_view
    controls = [*view._painting_photo_groups, view.photo_return_button,
        view.run_preview.runs, view.run_preview.previous, view.run_preview.next,
        view.run_preview.reference_button, view.run_preview.geometry_button,
        view.run_preview.sources, view.run_preview.refresh_button]
    before = [(control, control.isHidden()) for control in controls]
    open_rig(view, qtbot)
    irrelevant = [*view.tool_buttons.values(), view.brush_size, view.join_edge_button,
        view.show_corrections, view.fill_region_tool.button, view.fill_region_tool.cut_button,
        view.map_shown, view.color_mode, view.background, view.opacity, view.show_strokes,
        view.show_proposal, view.show_uncertain, view.show_differences,
        view.run_preview.runs, view.run_preview.sources, view.run_preview.geometry_button]
    assert all(not control.isVisible() for control in irrelevant)
    assert view.rig_fit.isVisible() and view.rig_enhance.isVisible()
    assert view.rig_enhance.isChecked() == view.enhance.isChecked()
    # Normal refreshes may alter child visibility, but the hidden group wins.
    view._refresh_manual_status()
    assert not view.manual_status.isVisible() and not view.canvas_hint.isVisible()
    view.set_workflow_scope('layers')
    assert view.rig_photo_toolbar.isHidden()
    assert all(control.isHidden() == hidden for control, hidden in before)
    assert not hasattr(view, '_rig_photo_visibility')
    open_rig(view, qtbot)
    restored = []
    monkeypatch.setattr(view.adoption_controller, 'open_chart_review', lambda **kwargs:
        restored.append(not hasattr(view, '_rig_photo_visibility') and view.rig_photo_toolbar.isHidden()))
    view.set_workflow_scope('apply')
    assert restored == [True]
