"""Persisted list of recently-opened config files, via QSettings.

Storage location is OS-managed (registry / plist / ini) and keyed by the
organization/application name registered on QApplication in __main__.py —
independent of the current working directory, so the list is consistent no
matter where `python -m darsia.gui` is launched from.
"""

from PySide6.QtCore import QSettings

_KEY = "recentConfigFiles"
_MAX_RECENT = 10


def get_recent_configs() -> list[str]:
    """Return recent config paths, most-recently-used first."""
    value = QSettings().value(_KEY, [])
    # QSettings' native (registry) backend returns a bare str, not a
    # single-item list, when exactly one value was stored - normalize.
    if isinstance(value, str):
        value = [value]
    return list(value)


def add_recent_config(path: str) -> None:
    """Move `path` to the front of the recent list, deduplicated, capped."""
    recents = [p for p in get_recent_configs() if p != path]
    recents.insert(0, path)
    QSettings().setValue(_KEY, recents[:_MAX_RECENT])


def remove_recent_config(path: str) -> None:
    """Drop `path` from the recent list (e.g. it no longer exists on disk)."""
    recents = [p for p in get_recent_configs() if p != path]
    QSettings().setValue(_KEY, recents)


def clear_recent_configs() -> None:
    QSettings().setValue(_KEY, [])
