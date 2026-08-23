"""Test desktop integration module: path resolution, file operations, and uninstall logic.

These tests use tmp_path and monkeypatching to avoid real filesystem/registry writes
in CI environments. Platform-specific logic is tested separately for Linux and Windows.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch


class TestIconResolution:
    """Test icon file discovery."""

    def test_get_icon_path_returns_pathlib_path(self):
        """Test that _get_icon_path returns a Path object."""
        from darsia.gui.desktop_integration import _get_icon_path

        icon = _get_icon_path()
        assert isinstance(icon, Path)

    def test_icon_file_exists(self):
        """Test that the icon file exists in the installed package."""
        from darsia.gui.desktop_integration import _ensure_icon_exists

        assert _ensure_icon_exists(), "DarSIA logo PNG should exist in the package"


class TestExecutableResolution:
    """Test resolution of the darsia console script."""

    def test_get_darsia_executable_returns_path(self):
        """Test that _get_darsia_executable returns a Path object."""
        from darsia.gui.desktop_integration import _get_darsia_executable

        exe = _get_darsia_executable()
        assert isinstance(exe, Path)

    def test_darsia_executable_in_bin_directory(self):
        """Test that the resolved executable is in the same bin/Scripts directory as sys.executable."""
        from darsia.gui.desktop_integration import _get_darsia_executable

        exe = _get_darsia_executable()
        sys_bin = Path(sys.executable).parent
        assert exe.parent == sys_bin, "darsia exe should be in the same bin directory as sys.executable"

    def test_darsia_executable_name_by_platform(self):
        """Test that the executable name matches the platform."""
        from darsia.gui.desktop_integration import _get_darsia_executable

        exe = _get_darsia_executable()
        if sys.platform == "win32":
            assert exe.name == "darsia.exe", "Windows should resolve to darsia.exe"
        else:
            assert exe.name == "darsia", "Unix should resolve to darsia (no extension)"


class TestLinuxDesktopIntegration:
    """Test Linux .desktop file installation/uninstall."""

    @pytest.mark.skipif(sys.platform == "win32", reason="Linux-only test")
    def test_install_linux_creates_desktop_file(self, tmp_path, monkeypatch, capsys):
        """Test that _install_linux creates a .desktop file in the mocked XDG_DATA_HOME."""
        from darsia.gui.desktop_integration import _install_linux

        xdg_data = tmp_path / "data"
        monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data))

        _install_linux()

        desktop_path = xdg_data / "applications" / "darsia.desktop"
        assert desktop_path.exists(), ".desktop file should be created"

        content = desktop_path.read_text()
        assert "[Desktop Entry]" in content
        assert "Type=Application" in content
        assert "Name=DarSIA GUI" in content
        assert "Icon=darsia" in content

    @pytest.mark.skipif(sys.platform == "win32", reason="Linux-only test")
    def test_install_linux_creates_icon_file(self, tmp_path, monkeypatch, capsys):
        """Test that _install_linux creates a persistent icon file that survives function return."""
        from darsia.gui.desktop_integration import _install_linux

        xdg_data = tmp_path / "data"
        monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data))

        _install_linux()

        icon_path = xdg_data / "icons" / "hicolor" / "256x256" / "apps" / "darsia.png"
        assert icon_path.exists(), "Icon file should be created and persist after _install_linux returns"

    @pytest.mark.skipif(sys.platform == "win32", reason="Linux-only test")
    def test_uninstall_linux_removes_desktop_file(self, tmp_path, monkeypatch, capsys):
        """Test that _uninstall_linux removes a previously-installed .desktop file."""
        from darsia.gui.desktop_integration import _install_linux, _uninstall_linux

        xdg_data = tmp_path / "data"
        monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data))

        _install_linux()
        desktop_path = xdg_data / "applications" / "darsia.desktop"
        assert desktop_path.exists()

        _uninstall_linux()
        assert not desktop_path.exists(), ".desktop file should be removed after uninstall"

    @pytest.mark.skipif(sys.platform == "win32", reason="Linux-only test")
    def test_uninstall_linux_no_error_if_nothing_installed(self, tmp_path, monkeypatch, capsys):
        """Test that _uninstall_linux gracefully handles missing .desktop file."""
        from darsia.gui.desktop_integration import _uninstall_linux

        xdg_data = tmp_path / "data"
        monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data))

        _uninstall_linux()

        captured = capsys.readouterr()
        assert "nothing to uninstall" in captured.out.lower()

    @pytest.mark.skipif(sys.platform == "win32", reason="Linux-only test")
    def test_install_linux_xdg_data_home_default(self, tmp_path, monkeypatch, capsys):
        """Test that _install_linux uses ~/.local/share as default XDG_DATA_HOME."""
        from darsia.gui.desktop_integration import _install_linux

        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)

        _install_linux()

        desktop_path = home / ".local" / "share" / "applications" / "darsia.desktop"
        assert desktop_path.exists(), ".desktop file should use ~/.local/share as default"


class TestWindowsDesktopIntegration:
    """Test Windows Start Menu shortcut installation/uninstall."""

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
    def test_install_windows_exists_when_platform_is_win32(self):
        """Test that _install_windows exists and is callable on Windows."""
        from darsia.gui.desktop_integration import _install_windows

        assert callable(_install_windows), "_install_windows should be callable"

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
    def test_install_windows_creates_persistent_ico(self, tmp_path, monkeypatch):
        """Test that _install_windows creates an .ico file that persists after function returns.

        This is the critical test that would catch the bug where the icon was stored in
        a TemporaryDirectory and deleted when the function exited, leaving the .lnk
        shortcut pointing at a nonexistent file.
        """
        from darsia.gui.desktop_integration import _install_windows

        appdata = tmp_path / "AppData" / "Roaming"
        localappdata = tmp_path / "AppData" / "Local"
        monkeypatch.setenv("APPDATA", str(appdata))
        monkeypatch.setenv("LOCALAPPDATA", str(localappdata))

        _install_windows()

        ico_path = localappdata / "DarSIA" / "darsia.ico"
        assert ico_path.exists(), "Icon file should be persisted and exist after _install_windows returns"

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
    def test_uninstall_windows_removes_persistent_ico(self, tmp_path, monkeypatch):
        """Test that _uninstall_windows removes the persisted icon file."""
        from darsia.gui.desktop_integration import _install_windows, _uninstall_windows

        appdata = tmp_path / "AppData" / "Roaming"
        localappdata = tmp_path / "AppData" / "Local"
        monkeypatch.setenv("APPDATA", str(appdata))
        monkeypatch.setenv("LOCALAPPDATA", str(localappdata))

        _install_windows()
        ico_path = localappdata / "DarSIA" / "darsia.ico"
        assert ico_path.exists(), "Icon should exist after install"

        _uninstall_windows()
        assert not ico_path.exists(), "Icon should be removed after uninstall"

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
    def test_uninstall_windows_no_error_if_nothing_installed(self, tmp_path, monkeypatch):
        """Test that _uninstall_windows gracefully handles missing shortcut and icon."""
        from darsia.gui.desktop_integration import _uninstall_windows

        appdata = tmp_path / "AppData" / "Roaming"
        localappdata = tmp_path / "AppData" / "Local"
        monkeypatch.setenv("APPDATA", str(appdata))
        monkeypatch.setenv("LOCALAPPDATA", str(localappdata))

        _uninstall_windows()

        # Should complete without error even when nothing is installed


class TestMainDispatch:
    """Test the main() dispatcher function."""

    def test_main_install_calls_platform_handler(self):
        """Test that main() calls the correct platform-specific install handler."""
        from darsia.gui.desktop_integration import main

        install_called = {}

        def mock_install_linux():
            install_called["linux"] = True

        def mock_install_windows():
            install_called["windows"] = True

        with patch("darsia.gui.desktop_integration._install_linux", mock_install_linux):
            with patch("darsia.gui.desktop_integration._install_windows", mock_install_windows):
                with patch("sys.platform", "linux"):
                    with patch("sys.argv", ["darsia-install-desktop"]):
                        main()
                        assert install_called.get("linux"), "Linux install handler should be called"

    def test_main_uninstall_calls_platform_handler(self):
        """Test that main() --uninstall calls the correct platform-specific uninstall handler."""
        from darsia.gui.desktop_integration import main

        uninstall_called = {}

        def mock_uninstall_linux():
            uninstall_called["linux"] = True

        def mock_uninstall_windows():
            uninstall_called["windows"] = True

        with patch("darsia.gui.desktop_integration._uninstall_linux", mock_uninstall_linux):
            with patch("darsia.gui.desktop_integration._uninstall_windows", mock_uninstall_windows):
                with patch("sys.platform", "linux"):
                    with patch("sys.argv", ["darsia-install-desktop", "--uninstall"]):
                        main()
                        assert uninstall_called.get("linux"), "Linux uninstall handler should be called"

    def test_main_unsupported_platform_exits(self):
        """Test that main() exits with an error on unsupported platforms."""
        from darsia.gui.desktop_integration import main

        with patch("sys.platform", "darwin"):
            with patch("sys.argv", ["darsia-install-desktop"]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

    def test_main_missing_executable_exits(self):
        """Test that main() exits if the darsia executable is not found."""
        from darsia.gui.desktop_integration import main

        with patch("darsia.gui.desktop_integration._get_darsia_executable", return_value=Path("/nonexistent/darsia")):
            with patch("darsia.gui.desktop_integration._install_linux", side_effect=FileNotFoundError("darsia not found")):
                with patch("sys.platform", "linux"):
                    with patch("sys.argv", ["darsia-install-desktop"]):
                        with pytest.raises(SystemExit) as exc_info:
                            main()
                        assert exc_info.value.code == 1
