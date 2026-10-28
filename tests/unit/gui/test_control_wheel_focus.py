"""Real window-system wheel delivery through controls into a scrollable page."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QDate, QEvent, QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDateEdit, QDoubleSpinBox, QLineEdit,
    QLabel, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from shiboken6 import isValid

from darsia.fluidflower.ui.interaction_safety import ComboBoxWheelGuard, install_interaction_safety


def make_control(kind):
    if kind in ('combo', 'editable_combo'):
        control = QComboBox()
        control.addItems([str(number) for number in range(20)])
        control.setCurrentIndex(8)
        control.setEditable(kind == 'editable_combo')
        value = control.currentIndex
    elif kind == 'date':
        control = QDateEdit(QDate(2020, 6, 15))
        control.setCurrentSection(QDateEdit.Section.DaySection)
        value = lambda: control.date().toJulianDay()
    else:
        control = QDoubleSpinBox() if kind == 'double_spin' else QSpinBox()
        control.setRange(0, 100)
        control.setValue(50)
        value = control.value
    return control, value


def page(qtbot, kind):
    install_interaction_safety()
    window = QWidget()
    qtbot.addWidget(window)
    layout = QVBoxLayout(window)
    sink = QLineEdit('Page focus stays here until a control is selected')
    layout.addWidget(sink)
    area = QScrollArea()
    area.setWidgetResizable(True)
    layout.addWidget(area)
    content = QWidget()
    controls = QVBoxLayout(content)
    control, value = make_control(kind)
    controls.addWidget(control)
    filler = QWidget()
    filler.setMinimumHeight(1500)
    controls.addWidget(filler)
    area.setWidget(content)
    window.resize(480, 300)
    window.show()
    window.activateWindow()
    sink.setFocus(Qt.FocusReason.OtherFocusReason)
    qtbot.waitUntil(sink.hasFocus)
    assert control.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert area.verticalScrollBar().maximum() > 0
    return SimpleNamespace(window=window, sink=sink, area=area, content=content,
                           layout=controls, control=control, value=value)


def wheel(control):
    window = control.window()
    position = control.mapTo(window, control.rect().center())
    # QTest's window-system API produces a real routed wheel event. A direct
    # sendEvent(QWheelEvent) is synthetic and Qt deliberately does not propagate it.
    QTest.wheelEvent(window.windowHandle(), QPointF(position), QPoint(0, -120))


@pytest.mark.parametrize('kind', ['combo', 'editable_combo', 'spin', 'double_spin', 'date'])
def test_unfocused_wheel_preserves_values_and_scrolls_the_page(qtbot, kind):
    ui = page(qtbot, kind)
    before = ui.value()
    wheel(ui.control)
    qtbot.waitUntil(lambda: ui.area.verticalScrollBar().value() > 0)
    assert ui.value() == before
    assert ui.sink.hasFocus()


@pytest.mark.parametrize('kind', ['combo', 'editable_combo', 'spin', 'double_spin', 'date'])
def test_focused_control_or_editor_keeps_its_normal_wheel_editing(qtbot, kind):
    ui = page(qtbot, kind)
    target = ui.control.lineEdit() if kind != 'combo' else ui.control
    target.setFocus(Qt.FocusReason.TabFocusReason)
    qtbot.waitUntil(lambda: ui.control.hasFocus() or
                   (QApplication.focusWidget() is not None and
                    ui.control.isAncestorOf(QApplication.focusWidget())))
    focused = QApplication.focusWidget()
    before = ui.value()
    wheel(ui.control)
    qtbot.waitUntil(lambda: ui.value() != before)
    assert ui.area.verticalScrollBar().value() == 0
    assert QApplication.focusWidget() is focused


def test_later_created_controls_are_guarded_and_installation_is_idempotent(qtbot):
    ui = page(qtbot, 'combo')
    guard = install_interaction_safety()
    assert install_interaction_safety() is guard
    later, value = make_control('spin')
    ui.layout.insertWidget(1, later)
    later.show()
    qtbot.waitUntil(lambda: later.focusPolicy() == Qt.FocusPolicy.StrongFocus)
    ui.sink.setFocus(Qt.FocusReason.OtherFocusReason)
    before = value()
    wheel(later)
    qtbot.waitUntil(lambda: ui.area.verticalScrollBar().value() > 0)
    assert value() == before and ui.sink.hasFocus()


def test_existing_controls_receive_strong_focus_and_keyboard_popup_selection_stays_normal(qtbot):
    ui = page(qtbot, 'combo')
    ui.control.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
    guard = install_interaction_safety()
    assert install_interaction_safety() is guard
    assert ui.control.focusPolicy() == Qt.FocusPolicy.StrongFocus
    ui.control.setFocus(Qt.FocusReason.TabFocusReason)
    before = ui.control.currentIndex()
    qtbot.keyClick(ui.control, Qt.Key.Key_Down)
    assert ui.control.currentIndex() == before + 1
    ui.control.showPopup()
    view = ui.control.view()
    qtbot.waitUntil(view.isVisible)
    index = ui.control.model().index(3, 0)
    view.scrollTo(index)
    qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton,
                     pos=view.visualRect(index).center())
    assert ui.control.currentIndex() == 3


@pytest.mark.parametrize("expired", ["receiver", "event"])
def test_expired_hide_delivery_is_consumed_without_native_wrapper_conversion(qapp, expired):
    guard = ComboBoxWheelGuard()
    watched = QLabel("Private lifetime regression")
    event = QEvent(QEvent.Type.Hide)
    try:
        if expired == "receiver":
            watched.deleteLater()
            QCoreApplication.sendPostedEvents(watched, QEvent.Type.DeferredDelete)
            assert not isValid(watched) and isValid(event)
        else:
            # Qt owns and destroys a posted event after delivery. Retain its
            # Python wrapper to exercise the expired-event boundary too.
            QCoreApplication.postEvent(watched, event)
            QCoreApplication.sendPostedEvents(watched, QEvent.Type.Hide)
            assert isValid(watched) and not isValid(event)
        assert guard.eventFilter(watched, event) is True
    finally:
        if isValid(watched):
            watched.deleteLater()
            QCoreApplication.sendPostedEvents(watched, QEvent.Type.DeferredDelete)


def test_valid_unhandled_hide_remains_unfiltered(qapp):
    guard = ComboBoxWheelGuard()
    watched = QLabel("Private live receiver")
    event = QEvent(QEvent.Type.Hide)
    try:
        assert guard.eventFilter(watched, event) is False
        assert isValid(watched) and isValid(event)
    finally:
        if isValid(watched):
            watched.deleteLater()
            QCoreApplication.sendPostedEvents(watched, QEvent.Type.DeferredDelete)


def test_reentrant_focus_preparation_consumes_a_receiver_deleted_during_the_callback(qapp):
    class SelfDeletingCombo(QComboBox):
        def setFocusPolicy(self, policy):
            super().setFocusPolicy(policy)
            if getattr(self, "expire_on_policy", False):
                self.deleteLater()
                QCoreApplication.sendPostedEvents(self, QEvent.Type.DeferredDelete)

    guard = ComboBoxWheelGuard()
    watched = SelfDeletingCombo()
    watched.expire_on_policy = True
    event = QEvent(QEvent.Type.Show)
    try:
        assert isValid(watched) and isValid(event)
        assert guard.eventFilter(watched, event) is True
        assert not isValid(watched)
    finally:
        if isValid(watched):
            watched.deleteLater()
            QCoreApplication.sendPostedEvents(watched, QEvent.Type.DeferredDelete)
