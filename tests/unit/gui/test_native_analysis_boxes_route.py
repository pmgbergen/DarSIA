"""Analysis boxes have their own route and never open a Setup action."""

from .test_native_sidebar import native_window  # noqa: F401
from .test_painting_corrections import view  # noqa: F401
from .test_native_geometry_workflow_scope import _install_shared_view


def test_analysis_boxes_route_reuses_canvas_outside_setup(native_window, view, monkeypatch):
    window = native_window
    workspace = _install_shared_view(window, view)
    monkeypatch.setattr(workspace, "ensure_page", lambda: None)
    scopes = []
    monkeypatch.setattr(view, "set_workflow_scope", scopes.append)

    window._select_row(("analysis_boxes", "edit"))

    assert window.sidebar.display_selection("analysis_boxes", "edit") == (
        "comparison", "analysis_boxes:edit")
    assert window.sidebar.get_selected() == ("analysis_boxes", "edit")
    assert window.content_stack.currentWidget() is workspace
    assert workspace.simple_view is view
    assert scopes == ["analysis_boxes"]

