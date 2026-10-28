"""A machine path map reads shared roots at this computer's folder and writes them back.

Temporary folders stand in for a share mounted elsewhere (for example
/Volumes/<share> on a Mac). The non-Windows branches are exercised with a
module-local stand-in for ``os`` whose ``name`` is "posix"; nothing global
is changed.
"""

import json
import os
import posixpath
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from darsia.fluidflower import review_storage, storage_paths
from darsia.presets.workflows.config import utils

SHARE = r"\\nas-host\Lab"


class _PosixOs:
    """``os`` as seen on a computer other than Windows (only ``name`` differs)."""

    name = "posix"

    def __getattr__(self, attribute):
        return getattr(os, attribute)


class _PosixPath(PurePosixPath):
    """POSIX path parsing with the two file reads the review alias reader uses."""

    def is_file(self):
        return os.path.isfile(str(self))

    def read_text(self, **options):
        return Path(str(self)).read_text(**options)


@pytest.fixture
def clean(tmp_path, monkeypatch):
    """No per-user registry, no project roots and no installed map."""
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "absent.toml"))
    for key in [key for key in os.environ if key.startswith("DARSIA_") and key.endswith("_PROJECT_ROOT")]:
        monkeypatch.delenv(key)
    monkeypatch.delenv("DARSIA_FLUIDFLOWER_DATA_ROOT", raising=False)
    monkeypatch.delenv("DARSIA_REVIEW_ALIAS_FILES", raising=False)
    for name in ("_cached_key", "_cached_stamp", "_cached_aliases", "_cached_lookup"):
        monkeypatch.setattr(review_storage, name, getattr(review_storage, name))
    storage_paths.set_machine_path_map(())
    yield tmp_path
    storage_paths.set_machine_path_map(())


@pytest.fixture
def mounted(clean):
    """The share ``SHARE`` opened at a local folder, installed as this computer's map."""
    local = clean / "Volumes" / "Lab"
    (local / "FF" / "DarSIA").mkdir(parents=True)
    storage_paths.set_machine_path_map([(SHARE, str(local))])
    return local


def test_without_a_map_nothing_changes(clean):
    assert storage_paths.machine_path_map() == ()
    for value in (SHARE + r"\FF\a.toml", "X:/FF/a", "data/x", str(clean / "local")):
        assert storage_paths.machine_stored_text(value) == value
        assert storage_paths.machine_stored_text(Path(value)) == str(Path(value))
        assert storage_paths.machine_local_path(value) == Path(value)
        assert storage_paths.machine_local_root(value) is None
        assert storage_paths.machine_local_text(value) is value
        assert storage_paths.resolve_storage_path(value) == Path(value).expanduser()
        assert storage_paths.storage_path_remapper()(value) is value


def test_reads_go_to_this_computers_folder(mounted):
    stored = SHARE + r"\FF\DarSIA\series.toml"
    expected = mounted / "FF" / "DarSIA" / "series.toml"
    assert storage_paths.resolve_storage_path(stored) == expected
    assert storage_paths.machine_local_path(stored) == expected
    assert storage_paths.machine_local_text(stored) == str(expected)
    assert storage_paths.machine_local_text("not a path") == "not a path"
    # Forward slashes and another letter case name the same shared file.
    assert storage_paths.resolve_storage_path("//NAS-HOST/lab/FF/DarSIA/series.toml") == expected
    assert storage_paths.storage_path_remapper()(stored) == str(expected)
    assert storage_paths.storage_path_remapper()("not a path") == "not a path"
    assert storage_paths.same_storage_path(stored, expected)
    assert storage_paths.resolve_storage_path(r"\\other\share\x") == Path(r"\\other\share\x")


def test_writes_use_the_shared_spelling(mounted, clean):
    assert storage_paths.machine_stored_text(mounted / "FF" / "Results" / "ac1") == SHARE + r"\FF\Results\ac1"
    assert storage_paths.machine_stored_text(str(mounted).upper()) == SHARE + "\\"
    outside = clean / "elsewhere" / "x"
    assert storage_paths.machine_stored_text(outside) == str(outside)
    assert storage_paths.machine_stored_text("relative/x") == "relative/x"


def test_first_row_of_a_folder_and_longest_folder_win(clean):
    local = clean / "Volumes" / "Lab"
    nested = local / "Archive"
    storage_paths.set_machine_path_map([
        (SHARE, str(local)), ("Y:", str(local)), (r"\\nas-host\Archive", str(nested)),
        ("/srv/lab", str(clean / "Volumes" / "Posix")),
    ])
    assert storage_paths.machine_stored_text(local / "a") == SHARE + r"\a"
    assert storage_paths.machine_stored_text(nested / "b") == r"\\nas-host\Archive\b"
    assert storage_paths.machine_stored_text(clean / "Volumes" / "Posix" / "c") == "/srv/lab/c"
    assert storage_paths.resolve_storage_path(r"Y:\FF\x") == local / "FF" / "x"
    assert storage_paths.resolve_storage_path("/srv/lab/d") == clean / "Volumes" / "Posix" / "d"


def test_storage_aliases_apply_before_the_map(clean, monkeypatch):
    registry = clean / "registry.toml"
    registry.write_text(
        "schema_version = 1\n[storage.path_aliases]\n"
        + json.dumps(SHARE + r"\Old") + " = " + json.dumps(SHARE + r"\New") + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(registry))
    local = clean / "Volumes" / "Lab"
    storage_paths.set_machine_path_map([(SHARE, str(local))])
    assert storage_paths.resolve_storage_path(SHARE + r"\Old\x") == local / "New" / "x"
    assert storage_paths.storage_path_remapper()(SHARE + r"\Old\x") == str(local / "New" / "x")
    # The same aliases apply to this computer's own spelling of the old place.
    assert storage_paths.resolve_storage_path(local / "Old" / "x") == local / "New" / "x"
    assert storage_paths.storage_path_remapper()(str(local / "Old" / "x")) == str(local / "New" / "x")
    assert storage_paths.storage_path_remapper()(str(local / "Other")) == str(local / "Other")
    assert storage_paths.same_storage_path(local / "Old" / "x", SHARE + r"\New\x")
    assert storage_paths.machine_local_root(local / "Old") == storage_paths.machine_local_root(str(local).upper())
    assert storage_paths.machine_local_root(clean / "elsewhere") is None
    # Job-side alias rows never carry this computer's map.
    assert all("volumes" not in destination.casefold() for _, destination in storage_paths._alias_rows())


def test_a_job_input_scope_reads_as_recorded(mounted, monkeypatch):
    from darsia.fluidflower import preset_wasserstein_inputs

    monkeypatch.setattr(preset_wasserstein_inputs, "current_scope", lambda: SimpleNamespace(path_aliases=[]))
    stored = SHARE + r"\FF\x"
    assert storage_paths.resolve_storage_path(stored) == Path(stored)
    assert storage_paths.machine_local_path(stored) == Path(stored)


@pytest.mark.parametrize("rows, message", [
    ([("relative", "C:/x")], "full path"),
    ([(SHARE, "relative")], "full path"),
    ([(SHARE, "")], "Each row"),
    ([(SHARE, "C:/a"), ("//NAS-HOST/LAB", "C:/b")], "twice"),
    ([(SHARE, SHARE + r"\inside")], "inside a shared root"),
])
def test_inconsistent_rows_are_refused(clean, rows, message):
    if os.name != "nt":
        rows = [(canonical, local.replace("C:", "")) for canonical, local in rows]
    with pytest.raises(ValueError, match=message):
        storage_paths.set_machine_path_map(rows)
    assert storage_paths.machine_path_map() == ()


def test_a_bare_drive_and_blanks_are_accepted(clean):
    local = str(clean / "drive")
    assert storage_paths.validate_machine_path_map([("  X:  ", f" {local} ")]) == (("X:", local),)


def test_windows_spellings_are_read_with_forward_slashes_elsewhere(clean, monkeypatch):
    monkeypatch.setattr(storage_paths, "os", _PosixOs())
    assert storage_paths.machine_local_path(r"protocols\workbooks\injection.xlsx") == Path(
        "protocols/workbooks/injection.xlsx")
    assert str(storage_paths.machine_local_path(SHARE + r"\FF\x")).replace("\\", "/") == "//nas-host/Lab/FF/x"
    assert storage_paths.machine_local_path("plain/relative") == Path("plain/relative")


def test_configuration_paths_with_backslashes_resolve_in_the_series(clean, monkeypatch):
    root = clean / "series"
    (root / "config").mkdir(parents=True)
    (root / "series.toml").write_text("schema_version = 1\n", encoding="utf-8")
    config = root / "config" / "run.toml"
    config.write_text("", encoding="utf-8")
    value = r"protocols\workbooks\injection.xlsx"
    # Windows reads the authored value exactly as before.
    assert utils._config_path(value, config) == root / Path(value)
    monkeypatch.setattr(storage_paths, "os", _PosixOs())
    assert utils._config_path(value, config) == root / "protocols" / "workbooks" / "injection.xlsx"


def test_windows_review_aliases_resolve_on_another_system(clean, monkeypatch):
    """Review alias files hold Windows paths; elsewhere they are read lexically."""
    local = clean / "Volumes" / "Lab"
    storage_paths.set_machine_path_map([(SHARE, str(local))])
    aliases = clean / "storage_aliases.json"
    aliases.write_text(json.dumps({"format": review_storage.FORMAT, "aliases": [{
        "source": r"C:\Users\someone\Reviews\s1\manual_pairwise_review",
        "destination": SHARE + r"\FF\reviews\s1\manual_pairwise_review",
        "original_save_root": r"C:\Users\someone\Reviews",
        "save_root": SHARE + r"\FF\reviews",
    }]}), encoding="utf-8")
    monkeypatch.setenv("DARSIA_REVIEW_ALIAS_FILES", str(aliases))
    monkeypatch.setattr(review_storage, "os", SimpleNamespace(
        name="posix", path=posixpath, fspath=os.fspath, environ=os.environ, stat=os.stat,
        pathsep=os.pathsep))
    monkeypatch.setattr(review_storage, "Path", _PosixPath)
    review_storage.review_aliases(refresh=True)
    resolved = review_storage.resolve_review_path(
        r"C:\Users\someone\Reviews\s1\manual_pairwise_review\decisions\a.json")
    expected = local / "FF" / "reviews" / "s1" / "manual_pairwise_review" / "decisions" / "a.json"
    assert str(resolved) == storage_paths._normalized(expected)
    assert str(review_storage._absolute(SHARE + r"\FF")) == "//nas-host/Lab/FF"
    # The session's old local save root is resumed at the shared root.
    context = r"C:\Users\someone\Reviews\s1\manual_pairwise_review"
    assert str(review_storage.resolve_review_path(r"C:\Users\someone\Reviews", context=context)) == (
        storage_paths._normalized(local / "FF" / "reviews"))
