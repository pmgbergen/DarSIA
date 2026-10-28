"""Application-wide guards against accidental Qt input changes."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QApplication, QAbstractSpinBox, QComboBox, QWidget
from shiboken6 import isValid


class ComboBoxWheelGuard(QObject):
    """Prevent passive page scrolling from changing editable controls."""

    @staticmethod
    def _combo_box(widget: Any) -> QComboBox | None:
        current = widget if isinstance(widget, QWidget) else None
        while current is not None:
            if isinstance(current, QComboBox):
                return current
            current = current.parentWidget()
        return None

    @staticmethod
    def _spin_box(widget: Any) -> QAbstractSpinBox | None:
        current = widget if isinstance(widget, QWidget) else None
        while current is not None:
            if isinstance(current, QAbstractSpinBox):
                return current
            current = current.parentWidget()
        return None

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        # Reentrant widget teardown can invalidate either Python wrapper. Qt
        # must not dispatch further to a deleted receiver/event.
        if not isValid(watched) or not isValid(event):
            return True
        event_type = event.type()
        if event_type in (QEvent.Type.Polish, QEvent.Type.Show):
            self.prepare_control(watched)
            # Setting focus policy can itself deliver events and delete a
            # control before the outer event filter resumes.
            if not isValid(watched) or not isValid(event):
                return True
        if event_type != QEvent.Type.Wheel:
            return False
        combo = self._combo_box(watched)
        if combo is not None:
            popup = combo.view()
            if (
                popup.isVisible()
                and isinstance(watched, QWidget)
                and (watched is popup or popup.isAncestorOf(watched))
            ):
                return False
            if not self._has_control_focus(combo):
                event.ignore()
                return True
        spin_box = self._spin_box(watched)
        if spin_box is not None and not self._has_control_focus(spin_box):
            event.ignore()
            return True
        return False

    @staticmethod
    def _has_control_focus(control: QWidget) -> bool:
        focus = QApplication.focusWidget()
        return control.hasFocus() or (focus is not None and control.isAncestorOf(focus))

    @staticmethod
    def prepare_control(widget: QObject) -> None:
        if isinstance(widget, (QComboBox, QAbstractSpinBox)):
            # WheelFocus would let the very scroll being guarded acquire focus.
            # Click and keyboard focus, including an editable child, remain valid.
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


def install_interaction_safety(app: QApplication | None = None) -> ComboBoxWheelGuard:
    """Install and retain the global input guard for one application instance."""

    if app is None:
        app = QApplication.instance()
    if app is None:
        raise RuntimeError("A QApplication is required before installing input guards")
    existing = getattr(app, "_fluidflower_combo_wheel_guard", None)
    if isinstance(existing, ComboBoxWheelGuard):
        guard = existing
    else:
        guard = ComboBoxWheelGuard(app)
        app.installEventFilter(guard)
        app._fluidflower_combo_wheel_guard = guard  # type: ignore[attr-defined]
    for widget in app.allWidgets():
        guard.prepare_control(widget)
    return guard


__all__ = ["ComboBoxWheelGuard", "install_interaction_safety"]
