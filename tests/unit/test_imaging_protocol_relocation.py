"""Imaging and exclusion lookups follow explicit image-root relocation."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import toml

from darsia.experiment.protocols import ImagingProtocol
from darsia.fluidflower import preset_wasserstein_inputs as preset_io
from darsia.presets.workflows.config import utils


@pytest.fixture
def relocated(tmp_path, monkeypatch):
    old = tmp_path / "old"
    old.mkdir()
    for name in ("first.ARW", "excluded.ARW"):
        (old / name).write_bytes(b"lookup must not read image bytes")
    protocol = tmp_path / "imaging.csv"
    stamps = ["2022-10-05T15:56:12.144000+02:00", "2022-10-05T16:01:12.144000+02:00"]
    pd.DataFrame(
        {
            "path": [str(old / name) for name in ("first.ARW", "excluded.ARW")],
            "datetime": stamps,
        }
    ).to_csv(protocol, index=False)
    blacklist = tmp_path / "blacklist.csv"
    pd.DataFrame({"path": [str(old / "excluded.ARW")]}).to_csv(blacklist, index=False)
    current = tmp_path / "current"
    old.rename(current)
    registry = tmp_path / "registry.toml"
    registry.write_text(
        toml.dumps(
            dict(schema_version=1, storage=dict(path_aliases={str(old): str(current)}))
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(registry))
    return old, current, protocol, blacklist, stamps


def test_moved_csv_paths_keep_times_blacklist_and_original_bytes(relocated, monkeypatch):
    old, current, path, blacklist, stamps = relocated
    original = path.read_bytes(), blacklist.read_bytes()
    opened = Path.open

    def metadata_only(file, *args, **kwargs):
        if file.suffix == ".ARW":
            raise AssertionError("Imaging lookups must not decode or hash a RAW image")
        return opened(file, *args, **kwargs)

    monkeypatch.setattr(Path, "open", metadata_only)
    protocol = ImagingProtocol(path, blacklist=blacklist)
    first, excluded = current / "first.ARW", current / "excluded.ARW"
    assert protocol.get_datetime(first) == pd.Timestamp(stamps[0])
    assert protocol.get_datetime(excluded) == pd.Timestamp(stamps[1])
    assert not protocol.is_blacklisted(first)
    assert protocol.is_blacklisted(excluded)
    assert protocol.find_images_for_paths([first, excluded]) == [first]
    assert protocol.find_images_for_datetimes(
        [first, excluded], [pd.Timestamp(stamps[0])]
    ) == [first]
    with pytest.raises(ValueError, match="not found in protocol"):
        protocol.get_datetime(current.parent / "unrelated" / first.name)
    assert (path.read_bytes(), blacklist.read_bytes()) == original


def test_relative_names_keep_original_matching_rules(tmp_path, monkeypatch):
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "no-registry.toml"))
    path = tmp_path / "imaging.csv"
    stamps = ["2022-10-05T15:56:12+02:00", "2022-10-05T16:01:12+02:00"]
    pd.DataFrame(
        {"path": ["first.JPG", "sub/second.JPG"], "datetime": stamps}
    ).to_csv(path, index=False)
    protocol = ImagingProtocol(path)
    assert protocol.get_datetime(tmp_path / "first.JPG") == pd.Timestamp(stamps[0])
    assert protocol.get_datetime(tmp_path / "sub/second.JPG") == pd.Timestamp(stamps[1])
    assert protocol.df["path"].tolist() == ["first.JPG", "sub/second.JPG"]


@pytest.mark.parametrize("scoped", [False, True])
def test_alias_worker_uses_existing_configuration_scope_rules(
    relocated, monkeypatch, scoped
):
    old, current, path, blacklist, stamps = relocated
    from darsia.fluidflower.storage_paths import _alias_rows

    scope = SimpleNamespace(
        path_aliases=[
            dict(source=source, destination=destination)
            for source, destination in _alias_rows()
        ]
    )
    monkeypatch.setattr(utils, "_in_alias_job_worker", lambda: True)
    monkeypatch.setattr(preset_io, "current_scope", lambda: scope if scoped else None)
    monkeypatch.setattr(preset_io, "table_source", lambda path, sheet=None: path)
    protocol = ImagingProtocol(path, blacklist=blacklist)
    expected = current if scoped else old
    other = old if scoped else current
    assert protocol.get_datetime(expected / "first.ARW") == pd.Timestamp(stamps[0])
    assert protocol.is_blacklisted(expected / "excluded.ARW")
    with pytest.raises(ValueError, match="not found in protocol"):
        protocol.get_datetime(other / "first.ARW")
