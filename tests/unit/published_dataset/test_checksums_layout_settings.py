"""Checksum list, record names and the per-user deny-list settings."""

import hashlib

import pytest

from darsia.fluidflower.published_dataset import checksums, layout, settings


def test_checksum_list_round_trip_and_problems(tmp_path):
    files = {"a.zip": b"a", "README.md": b"readme"}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    digests = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    path = checksums.write_checksums(tmp_path, digests)
    assert path.read_text(encoding="ascii").splitlines()[0] == f"{digests['README.md']}  README.md"
    assert checksums.read_checksums(tmp_path) == digests
    assert checksums.verify_checksums(tmp_path) == []
    assert checksums.verify_checksums(tmp_path, sizes={"a.zip": 2}, full=False) == [
        "Size differs from the manifest: a.zip"]
    (tmp_path / "a.zip").write_bytes(b"b")
    assert checksums.verify_checksums(tmp_path) == ["Checksum differs: a.zip"]
    assert checksums.verify_checksums(tmp_path, full=False) == []
    (tmp_path / "a.zip").unlink()
    assert checksums.verify_checksums(tmp_path) == ["Missing file: a.zip"]
    with pytest.raises(ValueError):
        checksums.write_checksums(tmp_path / "x", {layout.CHECKSUMS: "0" * 64})
    (tmp_path / layout.CHECKSUMS).write_text("not a line\n", encoding="ascii")
    with pytest.raises(ValueError, match="line 1"):
        checksums.read_checksums(tmp_path)


def test_record_names_types_and_tiers(tmp_path):
    assert layout.require_record_type("R") == "R" and layout.require_tier("S100") == "S100"
    for function, value in ((layout.require_record_type, "X"), (layout.require_tier, "S70")):
        with pytest.raises(ValueError):
            function(value)
    assert layout.raw_archive_name("r1") == "raw_r1.zip"
    assert layout.run_of_raw_archive("raw_r1.zip") == "r1" and layout.run_of_raw_archive("series.zip") is None
    with pytest.raises(ValueError):
        layout.raw_archive_name("../x")
    for value in ("a/b", "x.json"):
        assert layout.record_relative(value) == value
    for value in ("", "/a", "a\\b", "C:/a", "a/../b", "a//b", "./a"):
        with pytest.raises(ValueError):
            layout.record_relative(value)
    (tmp_path / "a").mkdir()
    assert layout.inside(tmp_path, "a/b.txt") == tmp_path / "a" / "b.txt"


def test_deny_list_comes_from_the_users_settings_only(tmp_path, monkeypatch):
    settings_file = tmp_path / "user" / "publication.toml"
    monkeypatch.setenv(settings.ENVIRONMENT, str(settings_file))
    assert settings.publication_settings_path() == settings_file
    assert settings.load_deny_list() == ()
    settings_file.parent.mkdir()
    settings_file.write_text('schema_version = 1\n[name_scan]\ndeny = ["Fictional Person", "sample-host", '
                             '"Fictional Person"]\n', encoding="utf-8")
    assert settings.load_deny_list() == ("Fictional Person", "sample-host")
    with pytest.raises(ValueError, match="outside series folders"):
        settings.load_deny_list(forbidden_roots=(tmp_path,))
    settings_file.write_text('schema_version = 2\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        settings.load_deny_list()
    settings_file.write_text('schema_version = 1\n[name_scan]\ndeny = [""]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty"):
        settings.load_deny_list()
    monkeypatch.delenv(settings.ENVIRONMENT)
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry" / "series_registry.toml"))
    assert settings.publication_settings_path() == tmp_path / "registry" / settings.SETTINGS_FILE
