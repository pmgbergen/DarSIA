"""Keyboard/mouse navigation uses the original SidebarRow signal."""
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QVBoxLayout, QWidget

from darsia.gui.ui.sidebar import SidebarRow


def test_tab_enter_space_and_disabled_rows(qtbot):
    panel = QWidget()
    layout = QVBoxLayout(panel)
    first = SidebarRow('setup', 'rig', 'Rig', 'fa5s.circle', 'Open rig settings')
    second = SidebarRow('calibration', 'mass', 'Mass', 'fa5s.circle', 'Open mass settings')
    layout.addWidget(first)
    layout.addWidget(second)
    selected = []
    first.clicked.connect(lambda: selected.append(('setup', 'rig')))
    second.clicked.connect(lambda: selected.append(('calibration', 'mass')))
    qtbot.addWidget(panel)
    panel.show()
    panel.activateWindow()
    first.setFocus()
    qtbot.waitUntil(first.hasFocus)
    QTest.keyClick(first, Qt.Key.Key_Return)
    QTest.keyClick(first, Qt.Key.Key_Tab)
    qtbot.waitUntil(second.hasFocus)
    QTest.keyClick(second, Qt.Key.Key_Space)
    assert selected == [('setup', 'rig'), ('calibration', 'mass')]
    QTest.mouseClick(second, Qt.MouseButton.RightButton)
    second.setEnabled(False)
    QTest.keyClick(second, Qt.Key.Key_Return)
    QTest.mouseClick(second, Qt.MouseButton.LeftButton)
    assert selected == [('setup', 'rig'), ('calibration', 'mass')]
