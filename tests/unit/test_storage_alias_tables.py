"""A series manifest can keep its storage aliases in pinned table files."""
import hashlib
from pathlib import Path

import pytest

from darsia.fluidflower import storage_paths
from darsia.fluidflower.storage_paths import series_path_aliases


def _table(folder, name, rows):
    lines = ["schema_version = 1", "[path_aliases]"] + [f'"{source}" = "{target}"' for source, target in rows.items()]
    content = ("\n".join(lines) + "\n").encode("utf-8")
    (Path(folder) / name).write_bytes(content)
    return dict(path=name, sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content))


def test_inline_and_table_rows_form_one_table(tmp_path):
    table = _table(tmp_path, "relocations.toml", {"//nas/share/old": "../moved"})
    document = dict(storage=dict(path_aliases={"//nas/share/other": "kept"}, path_alias_tables=[table]))
    assert series_path_aliases(document, tmp_path) == {"//nas/share/other": "kept", "//nas/share/old": "../moved"}


def test_a_changed_or_missing_table_is_refused(tmp_path):
    table = _table(tmp_path, "relocations.toml", {"//nas/share/old": "../moved"})
    (tmp_path / "relocations.toml").write_bytes(b"schema_version = 1\n[path_aliases]\n")
    with pytest.raises(ValueError, match="pinned bytes"):
        series_path_aliases(dict(storage=dict(path_alias_tables=[table])), tmp_path)
    with pytest.raises(OSError):
        series_path_aliases(dict(storage=dict(path_alias_tables=[dict(table, path="absent.toml")])), tmp_path)
    with pytest.raises(ValueError, match="path, SHA-256 and size"):
        series_path_aliases(dict(storage=dict(path_alias_tables=[{"path": "relocations.toml"}])), tmp_path)


def test_two_destinations_for_one_source_are_refused(tmp_path):
    table = _table(tmp_path, "relocations.toml", {"//NAS/share/old": "../moved"})
    document = dict(storage=dict(path_aliases={"//nas/share/old": "elsewhere"}, path_alias_tables=[table]))
    with pytest.raises(ValueError, match="two destinations"):
        series_path_aliases(document, tmp_path)


def test_table_rows_relocate_paths_like_inline_rows(tmp_path, monkeypatch):
    series = tmp_path / "series"
    series.mkdir()
    table = _table(series, "relocations.toml", {"//nas/share/old": "../moved"})
    (series / "series.toml").write_text(
        'schema_version = 1\nseries_id = "S"\n[storage]\npath_alias_tables = [{ path = "relocations.toml", '
        f'sha256 = "{table["sha256"]}", size_bytes = {table["size_bytes"]} }}]\n', encoding="utf-8")
    monkeypatch.setattr(storage_paths, "series_registry_path", lambda: tmp_path / "registry.toml")
    storage_paths.refresh_storage_paths()
    resolved = storage_paths.resolve_storage_path("//nas/share/old/file.json", series_roots=(str(series),))
    assert Path(resolved) == (tmp_path / "moved" / "file.json")
