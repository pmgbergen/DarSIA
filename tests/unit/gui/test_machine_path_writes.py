"""Setup records written on a computer with a machine path map name shared paths.

A temporary folder stands in for a share opened at /Volumes/<share>; its
shared root is ``SHARE``. Records must hold the shared spelling (or a path
relative to the series), and this computer must read them back locally.
Without a map every writer behaves as before.
"""

import json
import os
from pathlib import Path
import tomllib

import pytest
import toml

from darsia.fluidflower import series_creation as creation, series_locations, storage_paths
from darsia.fluidflower.protocol_workbook_archive import archive_protocol_sources
from darsia.fluidflower.protocol_workbooks import _assert_unchanged, _descriptor, _qualified
from darsia.fluidflower.series_folders_ui import read_image_locations, save_image_locations
from darsia.fluidflower.series_paths import register_series
from darsia.fluidflower.settings_storage_aliases import _store_shared_paths
from darsia.presets.workflows.config.data import DataConfig

SHARE = r"\\nas-host\Lab"


def shared(*parts):
    """The shared spelling of a path below ``SHARE``."""
    return "\\".join((SHARE, *parts))


@pytest.fixture
def share(tmp_path, monkeypatch):
    """The local folder of ``SHARE``; the map is installed for the test only."""
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    for key in [key for key in os.environ if key.startswith("DARSIA_") and key.endswith("_PROJECT_ROOT")]:
        monkeypatch.delenv(key)
    local = tmp_path / "Volumes" / "Lab"
    local.mkdir(parents=True)
    storage_paths.set_machine_path_map([(SHARE, str(local))])
    yield local
    storage_paths.set_machine_path_map(())


def _draft(darsia, images):
    darsia.parent.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    (images / "DSC1.JPG").write_bytes(b"never decoded")
    return creation.SeriesDraft("Study", "Study", "medium", darsia,
                                [creation.RunDraft("ac1", [images], baseline="DSC1.JPG")])


def test_new_series_names_shared_paths_and_reads_them_here(share):
    draft = _draft(share / "FF_NEW" / "DarSIA", share / "NEW" / "Raw" / "AC1" / "inj")
    files = creation.series_files(draft, created="TESTONLY")
    run = tomllib.loads(files["config/runs/ac1.toml"])
    assert run["data"]["folders"] == [shared("NEW", "Raw", "AC1", "inj")]
    assert run["data"]["results"] == shared("FF_NEW", "Results", "ac1")
    assert run["protocols"]["injection"] == shared("FF_NEW", "DarSIA", "protocols", "ac1", "injection_protocol.csv")
    assert run["protocols"]["imaging"] == {
        shared("NEW", "Raw", "AC1", "inj"): shared("FF_NEW", "DarSIA", "protocols", "ac1", "imaging_protocol_inj.csv")}
    manifest = tomllib.loads(files["series.toml"])
    assert manifest["roots"]["images_1"] == "../../NEW/Raw/AC1/inj"
    assert manifest["paths"]["results"] == "../Results"
    root = creation.create_series_folder(draft)
    data = DataConfig().load(root / "config/runs/ac1.toml", require_data=False, require_results=False)
    assert data.folders == [share / "NEW" / "Raw" / "AC1" / "inj"]
    assert data.results == share / "FF_NEW" / "Results" / "ac1"


def test_a_folder_on_another_disk_is_never_reached_through_the_share(share, tmp_path):
    local_images = tmp_path / "Local" / "Raw"
    draft = _draft(share / "FF_NEW" / "DarSIA", local_images)
    manifest = tomllib.loads(creation.series_files(draft, created="TESTONLY")["series.toml"])
    assert manifest["roots"]["images_1"] == str(local_images)
    assert creation._relative_folder(share / "FF_NEW" / "Results", share / "FF_NEW" / "DarSIA") == "../Results"


def test_without_a_map_new_series_files_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    images = tmp_path / "NEW" / "Raw"
    draft = _draft(tmp_path / "FF_NEW" / "DarSIA", images)
    files = creation.series_files(draft, created="TESTONLY")
    run = tomllib.loads(files["config/runs/ac1.toml"])
    assert run["data"]["folders"] == [str(images)]
    assert run["data"]["results"] == str(tmp_path / "FF_NEW" / "Results" / "ac1")
    assert tomllib.loads(files["series.toml"])["roots"]["images_1"] == Path(
        os.path.relpath(images, draft.darsia_folder)).as_posix()


def test_series_locations_store_shared_results_and_read_them_here(share):
    draft = _draft(share / "FF" / "DarSIA", share / "Raw")
    root = creation.create_series_folder(draft)
    new_results = share / "FF" / "Chosen"
    new_results.mkdir()
    series_locations.save_series_locations(root, share / "FF", new_results, preserve_inputs=False,
                                           replace_custom_results=True,
                                           path_aliases={share / "FF" / "Old": new_results})
    run = tomllib.loads((root / "config/runs/ac1.toml").read_text(encoding="utf-8"))
    assert run["data"]["results"] == shared("FF", "Chosen", "ac1")
    manifest = tomllib.loads((root / "series.toml").read_text(encoding="utf-8"))
    assert manifest["paths"]["results"] == "../Chosen"
    assert manifest["storage"]["path_aliases"] == {shared("FF", "Old"): shared("FF", "Chosen")}
    assert series_locations.read_series_locations(root)["results_folder"] == new_results
    assert storage_paths.resolve_storage_path(share / "FF" / "Old" / "x") == new_results / "x"


def test_image_folder_aliases_name_shared_paths(share):
    root = share / "project" / "DarSIA"
    root.mkdir(parents=True)
    images, moved = share / "images", share / "moved images"
    images.mkdir()
    moved.mkdir()
    (root / "series.toml").write_text(toml.dumps(dict(
        schema_version=1, series_id="study", name="Study", roots=dict(series=".", raw=str(images)),
        paths=dict(project="..", results="../Results"), runs=dict(ids=[]))), encoding="utf-8")
    register_series(root)
    save_image_locations(read_image_locations(root), {"raw": str(moved)})
    document = tomllib.loads((root / "series.toml").read_text(encoding="utf-8"))
    assert document["storage"]["path_aliases"] == {shared("images"): shared("moved images")}
    assert document["roots"]["raw"] == "../../moved images"
    assert read_image_locations(root)["images"][0]["folder"] == str(moved)


def test_protocol_cells_and_bindings_name_shared_paths(share):
    folder = share / "Raw" / "AC1"
    folder.mkdir(parents=True)
    assert _qualified("DSC1.ARW", folder) == shared("Raw", "AC1", "DSC1.ARW")
    assert _qualified(shared("Raw", "AC1", "DSC2.ARW"), folder) == shared("Raw", "AC1", "DSC2.ARW")
    table = folder / "imaging.csv"
    table.write_bytes(b"path,datetime\n")
    bound = _descriptor(table)
    assert bound["path"] == shared("Raw", "AC1", "imaging.csv")
    assert _descriptor(bound["path"]) == bound
    _assert_unchanged([bound])
    table.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        _assert_unchanged([bound])


def test_archived_protocol_folders_are_aliased_by_shared_paths(share):
    root = share / "project" / "DarSIA"
    protocols = root / "protocols"
    (protocols / "r1").mkdir(parents=True)
    (protocols / "r1" / "injection.csv").write_bytes(b"original protocol bytes")
    package = protocols / "workbooks"
    package.mkdir()
    (package / "protocol_workbook_migration.json").write_text(json.dumps(dict(
        format="darsia-protocol-workbook-migration-v1", status="active")), encoding="utf-8")
    (root / "series.toml").write_text(toml.dumps(dict(
        schema_version=1, series_id="test", paths=dict(project="..", protocols="protocols"),
        runs=dict(ids=["r1"]))), encoding="utf-8")
    (share / "project" / "Archive").mkdir()
    result = archive_protocol_sources(type("Plan", (), {"root": root})(), dict(folder=package, activated=True))
    archived = Path(result["folder"])
    manifest = tomllib.loads((root / "series.toml").read_text(encoding="utf-8"))
    assert manifest["storage"]["path_aliases"] == {
        shared("project", "DarSIA", "protocols", "r1"): shared("project", "Archive", archived.name, "r1")}
    relocated = storage_paths.resolve_storage_path(protocols / "r1" / "injection.csv", series_roots=(root,))
    assert relocated.read_bytes() == b"original protocol bytes"
    record = json.loads((package / "protocol_workbook_migration.json").read_text(encoding="utf-8"))
    assert record["archive"]["folder"] == shared("project", "Archive", archived.name)
    assert record["archive"]["manifest_after"]["path"] == shared("project", "DarSIA", "series.toml")


def test_settings_form_values_below_the_share_are_saved_shared(share, tmp_path):
    config = {
        "data": {"folders": [str(share / "Raw" / "AC1")], "results": str(share / "Results" / "ac1"),
                 "format": "JPG"},
        "protocols": {"imaging": {str(share / "Raw" / "AC1"): str(share / "protocols" / "imaging.csv")},
                      "injection": [str(tmp_path / "local.xlsx"), "ac1"]},
    }
    _store_shared_paths(config)
    assert config == {
        "data": {"folders": [shared("Raw", "AC1")], "results": shared("Results", "ac1"), "format": "JPG"},
        "protocols": {"imaging": {shared("Raw", "AC1"): shared("protocols", "imaging.csv")},
                      "injection": [str(tmp_path / "local.xlsx"), "ac1"]},
    }
    storage_paths.set_machine_path_map(())
    unchanged = {"data": {"results": str(share / "Results")}}
    section = unchanged["data"]
    _store_shared_paths(unchanged)
    assert unchanged["data"] is section and section == {"results": str(share / "Results")}
