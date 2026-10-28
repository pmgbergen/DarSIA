"""Per-user publication settings: the name deny-list of the builder's name scan.

The list belongs to one user on one machine. It lives next to the per-user
series registry (``publication.toml``; ``DARSIA_PUBLICATION_SETTINGS``
overrides the location), never in code, a series folder or a published
record::

    schema_version = 1

    [name_scan]
    deny = ["<a name>", "<another name>"]
"""

from __future__ import annotations

import os
from pathlib import Path
import tomllib

SETTINGS_FILE = "publication.toml"
ENVIRONMENT = "DARSIA_PUBLICATION_SETTINGS"


def publication_settings_path() -> Path:
    """Where this user's publication settings live (the file may not exist)."""
    explicit = os.environ.get(ENVIRONMENT)
    if explicit:
        return Path(os.path.abspath(os.path.expanduser(explicit)))
    from ..series_paths import series_registry_path

    return series_registry_path().parent / SETTINGS_FILE


def load_deny_list(path: str | Path | None = None, *, forbidden_roots=()) -> tuple[str, ...]:
    """Return the deny-listed names of the per-user settings (empty without a file).

    ``forbidden_roots`` are folders the settings must not lie in (the series
    being published, the DarSIA source checkout), so that the list itself can
    never be packaged.
    """
    selected = publication_settings_path() if path is None else Path(path)
    resolved = selected.expanduser().resolve()
    from ..series_paths import darsia_source_checkout

    checkout = darsia_source_checkout()
    for root in (*forbidden_roots, *(() if checkout is None else (checkout,))):
        if resolved.is_relative_to(Path(root).expanduser().resolve()):
            raise ValueError(
                "Publication settings must stay outside series folders and the DarSIA "
                f"source checkout: {resolved}"
            )
    if not resolved.is_file():
        return ()
    with resolved.open("rb") as stream:
        document = tomllib.load(stream)
    if document.get("schema_version") != 1:
        raise ValueError(f"Unsupported publication settings: {resolved}")
    values = (document.get("name_scan") or {}).get("deny", [])
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"[name_scan] deny must be a list of nonempty names: {resolved}")
    return tuple(dict.fromkeys(value.strip() for value in values))
