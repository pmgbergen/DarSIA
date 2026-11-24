"""Migrations of former config formats, in one place.

Each entry of `MIGRATIONS` rewrites one former format of a section into the current
one, in place, and is applied wherever config sections are read: by
`utils._get_section_from_toml` for workflow runs and by the GUI when it loads a config
(it writes its config back on save, so a migrated key disappears with the next save).
Each applied migration is logged and issues a `DeprecationWarning`. Section loaders only
know the current format. To drop a former format, delete its entry and function.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Migration:
    """Migration of one former format of a section."""

    section: str
    """Section (dotted path, e.g. `"image_porosity"`)."""
    key: str
    """Former key; the migration applies if the section contains it."""
    since: str
    """Date the former format was replaced."""
    apply: Callable[[dict], str]
    """Rewrite the section in place; returns a message for the log."""


def _image_porosity_active(sec: dict) -> str:
    """`active = false` becomes `method = "none"`; `active = true` keeps the given
    method (default `"kmeans"`)."""
    active = bool(sec.pop("active"))
    if active:
        sec.setdefault("method", "kmeans")
    else:
        sec["method"] = "none"
    return (
        f"[image_porosity] active = {str(active).lower()} is deprecated and was read "
        f"as method = {sec['method']!r}; remove the key 'active'."
    )


MIGRATIONS: list[Migration] = [
    Migration("image_porosity", "active", "2026-11-23", _image_porosity_active),
]
"""Former formats still read."""


def migrate_section(name: str, sec: dict) -> None:
    """Apply the migrations of section `name` to `sec`, in place."""
    for migration in MIGRATIONS:
        if migration.section == name and migration.key in sec:
            message = migration.apply(sec)
            logger.warning(message)
            warnings.warn(message, DeprecationWarning, stacklevel=2)


def migrate_config(config: dict) -> None:
    """Apply all migrations to a whole parsed config, in place."""
    for migration in MIGRATIONS:
        sec = config
        for part in migration.section.split("."):
            sec = sec.get(part) if isinstance(sec, dict) else None
        if isinstance(sec, dict) and migration.key in sec:
            migrate_section(migration.section, sec)
