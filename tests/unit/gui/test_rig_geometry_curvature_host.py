"""Curvature shares Rig Geometry without losing its editor on view changes."""

from types import SimpleNamespace

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QLineEdit, QVBoxLayout, QWidget
from shiboken6 import isValid

from .test_native_sidebar import native_window  # noqa: F401
from .test_painting_raw_reference_on_u import raw_view, raw_file, profile, renders, settings  # noqa: F401
from .test_rig_full_raw_editor import loaded, packet


def test_shared_curvature_editor_survives_geometry_view_replacement(native_window, qtbot):
    window = native_window
    draft = QLineEdit("retained draft")
    calls = []
    window.curvature_workflow = SimpleNamespace(widget=draft, set_active=calls.append)

    def view():
        panel = QWidget()
        panel.rig_curvature_host = QWidget(panel)
        QVBoxLayout(panel.rig_curvature_host)
        panel._native_workflow_scope_connected = True
        panel.set_workflow_scope = lambda scope: None
        panel.wait_for_workers = lambda: None
        window.geometry_workspace.view_stack.addWidget(panel)
        window.geometry_workspace.simple_view = panel
        return panel

    first = view()
    window._set_geometry_workflow_scope("rig_frame")
    assert first.rig_curvature_host.isAncestorOf(draft)
    assert calls[-1] is True
    window.geometry_workspace._discard_page()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(first)
    assert isValid(draft) and draft.parentWidget() is window
    assert draft.text() == "retained draft"

    second = view()
    window._set_geometry_workflow_scope("rig_frame")
    assert second.rig_curvature_host.isAncestorOf(draft)
    assert draft.text() == "retained draft"
    window._set_geometry_workflow_scope("layers")
    assert second.rig_curvature_host.isHidden() and draft.isHidden()
    assert calls[-1] is False
    window.geometry_workspace._discard_page()


def test_curvature_host_is_only_visible_in_rig_geometry(raw_view, qtbot):
    view = raw_view
    view._rig_photo_loader.reader = lambda path, profile: packet(path)
    form = QLineEdit("geometry settings")
    view.rig_curvature_host.layout().addWidget(form)
    view.set_workflow_scope("rig_frame")
    qtbot.waitUntil(lambda: loaded(view))
    assert not view.rig_curvature_host.isHidden()
    assert not view.step_depth.isHidden() and not view.step_depth.is_expanded()
    view.set_workflow_scope("layers")
    assert view.rig_curvature_host.isHidden()
    assert view.step_depth.isHidden()
    assert not view.step_paint.isHidden()
    view.set_workflow_scope("rig_frame")
    assert not view.rig_curvature_host.isHidden()
    assert form.text() == "geometry settings"
