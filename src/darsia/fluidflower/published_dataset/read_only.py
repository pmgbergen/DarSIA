"""Keep published dataset folders read-only inside a DarSIA process.

A published dataset opens as a series without ever being changed. When the app
opens one, its folder is protected: a Python audit hook refuses every
write-mode open, file or folder creation, rename, replacement, removal and
metadata change of a path inside it, whichever route asks for it. Reading is
unaffected, and nothing is checked while no folder is protected.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from .layout import MANIFEST

_PROTECTED: set[str] = set()
_INSTALLED = False
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
# Audit events that change the file system, and the positions of their paths.
_PATH_EVENTS = {
    "os.mkdir": (0,), "os.remove": (0,), "os.rmdir": (0,), "os.truncate": (0,), "os.chmod": (0,),
    "os.utime": (0,), "os.chflags": (0,), "os.chown": (0,), "os.setxattr": (0,), "os.removexattr": (0,),
    "os.rename": (0, 1), "os.replace": (0, 1), "os.link": (1,), "os.symlink": (1,),
    "shutil.rmtree": (0,), "shutil.move": (0, 1), "shutil.copytree": (1,), "shutil.copyfile": (1,),
    "shutil.make_archive": (0,),
}


class ReadOnlyDatasetError(PermissionError):
    """A write inside a published dataset folder was refused."""


def _key(path) -> str | None:
    """A comparable absolute form of a path, or None for file descriptors and others."""
    if isinstance(path, int):
        return None
    try:
        value = os.fsdecode(os.fspath(path))
    except TypeError:
        return None
    return os.path.normcase(os.path.abspath(value))


def _inside(path) -> bool:
    """True when ``path`` lies in a protected folder."""
    key = _key(path)
    if key is None:
        return False
    return any(key == root or key.startswith(root.rstrip(os.sep) + os.sep) for root in _PROTECTED)


def _is_write(mode, flags) -> bool:
    """True for an open that may write: a w, a, x or + mode, or write flags."""
    if isinstance(mode, str):
        return any(character in mode for character in "wax+")
    return isinstance(flags, int) and bool(flags & _WRITE_FLAGS)


def _hook(event: str, args) -> None:
    """The audit hook: refuse writes inside a protected folder."""
    if not _PROTECTED:
        return
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        if _is_write(mode, flags) and _inside(path):
            raise ReadOnlyDatasetError(f"Published dataset folders are read-only: {os.fspath(path)}")
        return
    positions = _PATH_EVENTS.get(event)
    if positions is None:
        return
    if event == "os.mkdir" and args and os.path.lexists(_key(args[0]) or ""):
        return  # The folder exists: the call creates nothing (mkdir with exist_ok stays harmless).
    for position in positions:
        if position < len(args) and _inside(args[position]):
            raise ReadOnlyDatasetError(f"Published dataset folders are read-only: {os.fspath(args[position])}")


def protect(root) -> None:
    """Protect a dataset folder for the rest of this process (installs the hook once)."""
    global _INSTALLED
    if not _INSTALLED:
        sys.addaudithook(_hook)
        _INSTALLED = True
    _PROTECTED.add(_key(Path(root).resolve()))


def unprotect(root) -> None:
    """Stop protecting a dataset folder (used by tests and when a series is removed)."""
    _PROTECTED.discard(_key(Path(root).resolve()))


def protected_roots() -> tuple[str, ...]:
    """The protected folders, as compared."""
    return tuple(sorted(_PROTECTED))


def dataset_root(path) -> Path | None:
    """The published dataset folder that holds ``path`` (it or a parent has a manifest), or None."""
    current = Path(path).resolve()
    for folder in (current, *current.parents):
        if (folder / MANIFEST).is_file():
            return folder
    return None
