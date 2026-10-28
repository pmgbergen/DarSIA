"""Real theme delivery after an original main window is destroyed."""

from collections import Counter
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEvent, QSettings
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

from darsia.gui.ui import theme
from darsia.gui.ui.main_window import MainWindow
from darsia.gui.ui.toolbar import ToolbarBuilder


def _actions(window):
    menu = window.menu_builder
    return {
        "new": menu.new_action,
        "open": menu.open_action,
        "save": menu.save_action,
        "settings": menu.open_full_config_action,
        "wizard": menu.current_wizard_action,
        "play": menu.play_action,
        "stop": menu.stop_action,
        "stream": menu.streaming_toggle_action,
        "log": menu.show_logging_action,
    }


def test_theme_refresh_retires_deleted_window_and_refreshes_surviving_toolbar(qapp, qtbot, monkeypatch, tmp_path):
    preferences = (QSettings.defaultFormat(), QCoreApplication.organizationName(),
                   QCoreApplication.applicationName())
    # An explicit, non-writing probe records the old Ini/UserScope base path,
    # including a caller's existing test isolation, before temporarily changing it.
    probe = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      "DarSIA-toolbar-path-probe", "path-probe")
    ini_root = Path(probe.fileName()).parent.parent
    old_style = qapp.style().objectName()
    old_palette = QPalette(qapp.palette())
    old_scheme = qapp.styleHints().colorScheme()
    original_theme_state = (theme._original_style, theme._original_palette)
    previous_windows = set(QApplication.topLevelWidgets())
    original_load_icon = ToolbarBuilder._load_icon
    loads = []

    def observe_load_icon(builder, key):
        icon = original_load_icon(builder, key)
        loads.append((builder, key, icon))
        return icon

    # Observe only the original icon loader. Theme emission, refresh_icons,
    # setIcon, action storage, and pytest-qt's exception capture stay unchanged.
    monkeypatch.setattr(ToolbarBuilder, "_load_icon", observe_load_icon)
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      str(tmp_path / "settings"))
    QCoreApplication.setOrganizationName("DarSIA-toolbar-lifetime-tests")
    QCoreApplication.setApplicationName(tmp_path.name)

    def check_refresh(windows, builders, actions):
        expected = Counter((id(builder), key) for builder in builders for key in actions[0])
        assert Counter((id(builder), key) for builder, key, _icon in loads) == expected
        icons = {(id(builder), key): icon for builder, key, icon in loads}
        for window, builder, captured in zip(windows, builders, actions):
            current = _actions(window)
            assert len(current) == 9 and current.keys() == captured.keys()
            for key, action in captured.items():
                assert current[key] is action and isValid(action)
                icon = icons[(id(builder), key)]
                assert not icon.isNull() and not action.icon().isNull()
                assert action.icon().cacheKey() == icon.cacheKey()

    try:
        first = MainWindow()
        survivor = MainWindow()
        first.show()
        survivor.show()
        qtbot.waitUntil(lambda: first.isVisible() and survivor.isVisible())
        # Keep both the Python builder and all QAction wrappers alive after
        # their owner's C++ destruction; garbage collection cannot mask the bug.
        first_builder, survivor_builder = first.toolbar_builder, survivor.toolbar_builder
        first_actions, survivor_actions = _actions(first), _actions(survivor)
        assert len(first_actions) == len(survivor_actions) == 9
        assert all(isValid(action) for action in first_actions.values())

        # Closing is not destruction: a hidden, reusable main window must still
        # receive the normal theme update, including all nine original actions.
        assert survivor.close() and isValid(survivor) and survivor.isHidden()
        loads.clear()
        theme.apply_theme(qapp, "Light")
        check_refresh((first, survivor), (first_builder, survivor_builder),
                      (first_actions, survivor_actions))
        survivor.show()
        qtbot.waitUntil(survivor.isVisible)

        assert first.close()
        first.deleteLater()
        QCoreApplication.sendPostedEvents(first, QEvent.Type.DeferredDelete)
        qtbot.waitUntil(lambda: not isValid(first))
        assert not isValid(first_actions["new"])
        assert all(not isValid(action) for action in first_actions.values())

        loads.clear()
        theme.apply_theme(qapp, "Dark")
        # The old plain-Python receiver reaches deleted QActions here. Both
        # observable loader calls and the normal Qt exception tripwire remain.
        assert not any(builder is first_builder for builder, _key, _icon in loads)
        check_refresh((survivor,), (survivor_builder,), (survivor_actions,))
        assert survivor.isVisible()
    finally:
        try:
            # Intentionally deleted wrappers are not registered with qtbot:
            # its addWidget teardown calls close() on retained wrappers. Own
            # both real windows here, including a partial constructor failure.
            for window in QApplication.topLevelWidgets():
                if type(window) is MainWindow and window not in previous_windows and isValid(window):
                    window.close()  # Original streaming-cache cleanup.
                    window.deleteLater()
                    QCoreApplication.sendPostedEvents(window, QEvent.Type.DeferredDelete)
        finally:
            # Restore actual application appearance and the theme helper's
            # saved baseline without replacing its signal or refresh behavior.
            qapp.setStyle(old_style)
            qapp.setPalette(old_palette)
            qapp.styleHints().setColorScheme(old_scheme)
            theme._original_style, theme._original_palette = original_theme_state
            QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(ini_root))
            QCoreApplication.setOrganizationName(preferences[1])
            QCoreApplication.setApplicationName(preferences[2])
            QSettings.setDefaultFormat(preferences[0])
