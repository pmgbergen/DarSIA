r"""Desktop integration for DarSIA GUI — platform-specific shortcut/menu entry installation.

Provides:
- Linux: .desktop file installation in ~/.local/share/applications/
- Windows: .lnk shortcut installation in %APPDATA%\Microsoft\Windows\Start Menu\Programs\
- Both: icon copying/generation, uninstall support, no-admin-required per-user setup

Uses sys.executable-relative resolution to locate the 'darsia' command, avoiding
any custom virtualenv discovery logic.
"""

import argparse
import os
import sys
from pathlib import Path


def _get_darsia_executable() -> Path:
    """Resolve the absolute path to the 'darsia' console script.

    Returns the sibling script in the same bin/Scripts directory as the current
    interpreter. This is guaranteed to exist after 'uv sync' picks up the
    [project.gui-scripts] entry point.

    On Windows: sys.executable.parent / "darsia.exe"
    On Linux/macOS: sys.executable.parent / "darsia"
    """
    bin_dir = Path(sys.executable).parent
    script_name = "darsia.exe" if sys.platform == "win32" else "darsia"
    return bin_dir / script_name


def _get_icon_path() -> Path:
    """Get the path to the DarSIA logo PNG within the darsia package.

    Returns the absolute path to the installed package's logo, relative to
    the desktop_integration.py module location (darsia/gui/desktop_integration.py).
    The icon is at darsia/presets/workflows/interface/DarSIA_Horisontal_Positiv_part.png.
    """
    module_dir = Path(__file__).parent
    return (
        module_dir / "../presets/workflows/interface/DarSIA_Horisontal_Positiv_part.png"
    ).resolve()


def _ensure_icon_exists() -> bool:
    """Verify the icon file exists in the installed package.

    Returns True if found, False otherwise.
    """
    icon = _get_icon_path()
    return icon.exists()


def _get_icon_cache_dir() -> Path:
    r"""Get the per-user cache directory for persisting desktop integration icons.

    On Windows: %LOCALAPPDATA%\DarSIA
    On Linux/macOS: $XDG_CACHE_HOME/darsia or ~/.cache/darsia
    """
    if sys.platform == "win32":
        localappdata = os.environ.get(
            "LOCALAPPDATA", str(Path.home() / "AppData" / "Local")
        )
        return Path(localappdata) / "DarSIA"
    else:
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        return Path(xdg_cache_home) / "darsia"


def _pad_image_to_square(img):
    """Pad an image to a square canvas, preserving aspect ratio.

    Center the image on a transparent square canvas with dimensions equal to
    the larger of the image's width or height. Returns a PIL Image object.
    """
    from PIL import Image

    width, height = img.size
    size = max(width, height)
    square = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    offset_x = (size - width) // 2
    offset_y = (size - height) // 2
    square.paste(img, (offset_x, offset_y), img if img.mode == "RGBA" else None)
    return square


def _install_linux() -> None:
    """Install a .desktop entry for DarSIA in the Linux application menu.

    Creates ~/.local/share/applications/darsia.desktop and copies the icon to
    ~/.local/share/icons/hicolor/256x256/apps/darsia.png.

    Uses $XDG_DATA_HOME if set, otherwise defaults to ~/.local/share.
    Requires no admin privileges.
    """
    from PIL import Image

    xdg_data_home = os.environ.get(
        "XDG_DATA_HOME", str(Path.home() / ".local" / "share")
    )
    applications_dir = Path(xdg_data_home) / "applications"
    icons_dir = Path(xdg_data_home) / "icons" / "hicolor" / "256x256" / "apps"

    applications_dir.mkdir(parents=True, exist_ok=True)
    icons_dir.mkdir(parents=True, exist_ok=True)

    darsia_exe = _get_darsia_executable()
    desktop_path = applications_dir / "darsia.desktop"
    icon_path = icons_dir / "darsia.png"

    desktop_content = f"""[Desktop Entry]
Type=Application
Name=DarSIA GUI
Comment=Darcy scale image analysis toolbox
Exec={darsia_exe}
Icon=darsia
Categories=Science;Graphics;
Terminal=false
"""
    desktop_path.write_text(desktop_content)
    print(f"✓ Created .desktop entry: {desktop_path}")

    if _ensure_icon_exists():
        src_icon = _get_icon_path()
        img = Image.open(src_icon)
        img_square = _pad_image_to_square(img)
        img_resized = img_square.resize((256, 256), Image.Resampling.LANCZOS)
        img_resized.save(icon_path)
        print(f"✓ Copied icon: {icon_path}")
    else:
        print("! Warning: icon file not found; .desktop entry created without icon")


def _uninstall_linux() -> None:
    """Remove the DarSIA .desktop entry and icon from Linux application menu."""
    xdg_data_home = os.environ.get(
        "XDG_DATA_HOME", str(Path.home() / ".local" / "share")
    )
    desktop_path = Path(xdg_data_home) / "applications" / "darsia.desktop"
    icon_path = (
        Path(xdg_data_home) / "icons" / "hicolor" / "256x256" / "apps" / "darsia.png"
    )

    removed = False
    if desktop_path.exists():
        desktop_path.unlink()
        print(f"✓ Removed .desktop entry: {desktop_path}")
        removed = True

    if icon_path.exists():
        icon_path.unlink()
        print(f"✓ Removed icon: {icon_path}")
        removed = True

    if not removed:
        print("ℹ No desktop entry found; nothing to uninstall")


def _install_windows() -> None:
    r"""Install a Start Menu shortcut for DarSIA on Windows.

    Creates %APPDATA%\Microsoft\Windows\Start Menu\Programs\DarSIA GUI.lnk
    pointing to the 'darsia.exe' console script, with a converted .ico icon.

    The icon is persisted at %LOCALAPPDATA%\DarSIA\darsia.ico so the shortcut
    can reference it persistently (Windows shortcuts store path references,
    not embedded icon data).

    Requires pywin32; uses only per-user Start Menu (no admin required).
    """
    import win32com.client  # noqa: F401; imported to trigger ImportError if unavailable
    from PIL import Image

    appdata = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    start_menu_dir = appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    start_menu_dir.mkdir(parents=True, exist_ok=True)

    icon_cache_dir = _get_icon_cache_dir()
    icon_cache_dir.mkdir(parents=True, exist_ok=True)
    ico_path = icon_cache_dir / "darsia.ico"

    darsia_exe = _get_darsia_executable()
    shortcut_path = start_menu_dir / "DarSIA GUI.lnk"

    if _ensure_icon_exists():
        src_icon = _get_icon_path()
        img = Image.open(src_icon)
        img_square = _pad_image_to_square(img)
        img_resized = img_square.resize((256, 256), Image.Resampling.LANCZOS)
        img_resized.save(ico_path, "ICO")
        print(f"✓ Generated icon: {ico_path}")
    else:
        print("! Warning: icon file not found; shortcut created without icon")

    shell = win32com.client.Dispatch("WScript.Shell")  # noqa: F821
    shortcut = shell.CreateShortcut(str(shortcut_path))
    shortcut.TargetPath = str(darsia_exe)
    shortcut.WorkingDirectory = str(Path.home())
    shortcut.WindowStyle = 1
    if ico_path.exists():
        shortcut.IconLocation = str(ico_path)
    shortcut.Save()

    print(f"✓ Created Start Menu shortcut: {shortcut_path}")


def _uninstall_windows() -> None:
    """Remove the DarSIA Start Menu shortcut and icon cache from Windows."""
    appdata = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    shortcut_path = (
        appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "DarSIA GUI.lnk"
    )

    icon_cache_dir = _get_icon_cache_dir()
    ico_path = icon_cache_dir / "darsia.ico"

    removed = False
    if shortcut_path.exists():
        shortcut_path.unlink()
        print(f"✓ Removed Start Menu shortcut: {shortcut_path}")
        removed = True

    if ico_path.exists():
        ico_path.unlink()
        print(f"✓ Removed icon cache: {ico_path}")
        removed = True

    if icon_cache_dir.exists():
        try:
            icon_cache_dir.rmdir()
        except OSError:
            pass

    if not removed:
        print("ℹ No Start Menu shortcut found; nothing to uninstall")


def main() -> None:
    """Entry point for the darsia-install-desktop command.

    Installs or uninstalls a desktop application entry for the DarSIA GUI.
    Dispatches to platform-specific handlers based on sys.platform.
    """
    parser = argparse.ArgumentParser(
        description="Install or uninstall DarSIA GUI desktop/application menu entry.",
        prog="darsia-install-desktop",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove the desktop entry instead of installing it.",
    )

    args = parser.parse_args()

    try:
        if sys.platform == "linux" or sys.platform.startswith("linux"):
            if args.uninstall:
                _uninstall_linux()
            else:
                _install_linux()
        elif sys.platform == "win32":
            if args.uninstall:
                _uninstall_windows()
            else:
                _install_windows()
        else:
            print(f"✗ Unsupported platform: {sys.platform}")
            sys.exit(1)
    except FileNotFoundError:
        print(f"✗ Error: darsia executable not found at {_get_darsia_executable()}")
        print("  Run 'uv sync' to install the 'darsia' entry point first.")
        sys.exit(1)
    except ImportError as e:
        if "win32com" in str(e):
            print(f"✗ Error: pywin32 is required on Windows; install it with 'uv sync'")
            sys.exit(1)
        raise
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
