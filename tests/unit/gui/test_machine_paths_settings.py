"""The per-user setting that tells a GUI where this computer opens each shared root.

Every test reads and writes a temporary INI file, never the user's settings.
"""

import json
import os

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMainWindow, QMessageBox

from darsia.fluidflower import machine_paths, storage_paths

SHARE = r"\\nas-host\Lab"
MOUNT_OUTPUT = """/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)
devfs on /dev (devfs, local, nobrowse)
//someone@NAS-Host._smb._tcp.local/Lab on /Volumes/Lab (smbfs, nodev, nosuid, mounted by someone)
//GUEST:@10.0.0.5/Shared%20Data on /Volumes/Shared Data (smbfs, nodev, nosuid, mounted by someone)
map auto_home on /System/Volumes/Data/home (autofs, automounted, nobrowse)
server:/export on /Volumes/export (nfs, nodev, nosuid)
"""


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """A temporary settings file, an empty registry and no installed map."""
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "absent.toml"))
    monkeypatch.setenv("DARSIA_REVIEW_ALIAS_FILES", str(tmp_path / "no_review_aliases.json"))
    storage_paths.set_machine_path_map(())
    yield QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    storage_paths.set_machine_path_map(())


def test_saved_rows_are_read_back_and_installed(settings, tmp_path):
    local = str(tmp_path / "Volumes" / "Lab")
    assert machine_paths.read_machine_path_map(settings) == ()
    assert machine_paths.save_machine_path_map([(SHARE, local)], settings) == ((SHARE, local),)
    assert machine_paths.read_machine_path_map(settings) == ((SHARE, local),)
    assert storage_paths.machine_path_map() == ((SHARE, local),)
    storage_paths.set_machine_path_map(())
    assert machine_paths.install_machine_path_map(settings) == ((SHARE, local),)
    assert storage_paths.resolve_storage_path(SHARE + r"\FF\x") == tmp_path / "Volumes" / "Lab" / "FF" / "x"
    # Saving no rows removes the map.
    assert machine_paths.save_machine_path_map([], settings) == ()
    assert storage_paths.machine_path_map() == ()


def test_invalid_rows_are_refused_and_not_saved(settings, tmp_path):
    with pytest.raises(ValueError, match="full path"):
        machine_paths.save_machine_path_map([(SHARE, "relative/folder")], settings)
    assert machine_paths.read_machine_path_map(settings) == ()


def test_a_damaged_setting_installs_nothing(settings):
    settings.setValue(machine_paths.KEY, "{not json")
    assert machine_paths.read_machine_path_map(settings) == ()
    settings.setValue(machine_paths.KEY, '[["relative", "also relative"]]')
    assert machine_paths.install_machine_path_map(settings) == ()
    assert storage_paths.machine_path_map() == ()


def test_mounted_smb_shares_are_proposed_with_their_server():
    assert machine_paths.mounted_shares(MOUNT_OUTPUT) == [
        (r"\\NAS-Host\Lab", "/Volumes/Lab"),
        (r"\\10.0.0.5\Shared Data", "/Volumes/Shared Data"),
    ]
    assert machine_paths.mounted_shares("") == []


def test_dialog_saves_valid_rows_and_explains_invalid_ones(qapp, settings, tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, text: warnings.append(text))
    dialog = machine_paths.MachinePathMapDialog(settings=settings)
    assert dialog.rows() == []
    dialog.add_row(SHARE, "relative")
    dialog.add_row("", "")
    dialog.save()
    assert warnings and "full path" in warnings[0]
    assert dialog.result() == 0 and machine_paths.read_machine_path_map(settings) == ()
    local = str(tmp_path / "Volumes" / "Lab")
    dialog.table.item(0, 1).setText(local)
    dialog.save()
    assert dialog.result() == dialog.DialogCode.Accepted
    assert machine_paths.read_machine_path_map(settings) == ((SHARE, local),)
    reopened = machine_paths.MachinePathMapDialog(settings=settings)
    assert reopened.rows() == [(SHARE, local)]


def test_dialog_adds_mounted_shares_once(qapp, settings, monkeypatch):
    monkeypatch.setattr(machine_paths, "_system_mount_output", lambda: MOUNT_OUTPUT)
    dialog = machine_paths.MachinePathMapDialog(settings=settings)
    dialog.add_row(r"\\nas-host\lab", "/Volumes/Lab")
    dialog._suggest()
    assert dialog.rows() == [(r"\\nas-host\lab", "/Volumes/Lab"), (r"\\10.0.0.5\Shared Data", "/Volumes/Shared Data")]


def test_the_action_joins_the_settings_menu(qapp):
    window = QMainWindow()
    window.menuBar().addMenu("&File")
    settings_menu = window.menuBar().addMenu("&Settings")
    action = machine_paths.add_machine_path_action(window)
    assert action in settings_menu.actions()
    bare = QMainWindow()
    assert machine_paths.add_machine_path_action(bare) in bare.menuBar().actions()


def test_the_native_gui_installs_the_map_before_its_window(qapp, settings, monkeypatch):
    from darsia.fluidflower import native_darsia_app

    calls = []

    class Window(QMainWindow):
        def __init__(self, *, initial_series=None):
            super().__init__()
            calls.append(("window", storage_paths.machine_path_map()))

    local = os.path.abspath("mounted-share")
    monkeypatch.setattr(machine_paths, "QSettings", lambda *args: settings)
    settings.setValue(machine_paths.KEY, json.dumps([[SHARE, local]]))
    monkeypatch.setattr(native_darsia_app, "NativeFluidFlowerMainWindow", Window)
    assert native_darsia_app.main([]) == 0
    assert calls == [("window", ((SHARE, local),))]
    window = qapp._native_fluidflower_window
    assert machine_paths.ACTION_TEXT in [action.text() for action in window.menuBar().actions()]
    window.close()
