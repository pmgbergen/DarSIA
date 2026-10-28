"""Archive only old protocol directories; retain references and active workbooks."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import toml

from darsia.fluidflower.protocol_workbook_archive import archive_protocol_sources
from darsia.fluidflower.storage_paths import resolve_storage_path


@pytest.fixture
def sources(tmp_path, monkeypatch):
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    project = tmp_path / "project"
    root = project / "DarSIA"
    root.mkdir(parents=True)
    manifest = root / "series.toml"
    manifest.write_bytes(b"# retain original manifest bytes\r\n" + toml.dumps(dict(
        schema_version=1, series_id="test", paths=dict(project="..", protocols="protocols"),
        runs=dict(ids=["r1"]))).encode())
    protocols = root / "protocols"
    files = {}
    for relative in ("r1/injection.csv", "r0/excluded.csv", "native/r1/imaging.csv", "timing/r1/injection.csv"):
        path = protocols / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((relative + "\noriginal protocol bytes\n").encode())
        files[path] = path.read_bytes()
    (protocols / "notes.txt").write_bytes(b"root-level file stays")
    hidden = protocols / ".reserved"
    hidden.mkdir()
    (hidden / "hold").write_bytes(b"hidden folder stays")
    package_folder = protocols / "workbooks"
    package_folder.mkdir()
    (package_folder / "imaging_protocol.xlsx").write_bytes(b"active workbook bytes")
    (package_folder / "protocol_workbook_migration.json").write_text(json.dumps(dict(
        format="darsia-protocol-workbook-migration-v1", status="active")), encoding="utf-8")
    external = project / "Archive" / "original_overflow_review.json"
    external.parent.mkdir()
    external.write_bytes(b"external overflow review stays")
    return SimpleNamespace(root=root, project=project, protocols=protocols, files=files,
        manifest=manifest, original_manifest=manifest.read_bytes(), external=external,
        plan=SimpleNamespace(root=root), package=dict(folder=package_folder, activated=True))


def test_archive_preserves_old_reference_resolution_and_excludes_active_inputs(sources):
    result = archive_protocol_sources(sources.plan, sources.package)
    assert result["archived"] and sources.package["archived"]
    assert len(result["moves"]) == 4
    for path, original in sources.files.items():
        assert not path.exists()
        relocated = resolve_storage_path(path, series_roots=(sources.root,))
        assert relocated.is_relative_to(result["folder"])
        assert relocated.read_bytes() == original
    assert (sources.protocols / "notes.txt").read_bytes() == b"root-level file stays"
    assert (sources.protocols / ".reserved/hold").is_file()
    assert (sources.package["folder"] / "imaging_protocol.xlsx").read_bytes() == b"active workbook bytes"
    assert sources.external.read_bytes() == b"external overflow review stays"
    record = json.loads((sources.package["folder"] / "protocol_workbook_migration.json").read_text())
    assert record["status"] == "active"
    assert record["archive"]["status"] == "complete"
    assert result["folder"].parent == sources.project / "Archive"


def test_occupied_target_rolls_back_moved_folders_without_replacing_other_data(sources, monkeypatch):
    target = sources.project / "Archive" / "ProtocolMigration_test"
    rename = Path.rename
    def collide(path, destination):
        if path == sources.protocols / "timing":
            occupied = target / "timing"
            occupied.mkdir()
            (occupied / "foreign.txt").write_bytes(b"other writer's file")
            raise FileExistsError("occupied archive target")
        return rename(path, destination)
    monkeypatch.setattr(Path, "rename", collide)
    with pytest.raises(RuntimeError, match="rolled back; active workbooks are unchanged"):
        archive_protocol_sources(sources.plan, sources.package, destination=target)
    assert all(path.read_bytes() == original for path, original in sources.files.items())
    assert sources.manifest.read_bytes() == sources.original_manifest
    assert (target / "timing/foreign.txt").read_bytes() == b"other writer's file"
    assert sources.package["activated"] and not sources.package.get("archived")


def test_manifest_save_failure_restores_exact_manifest_and_folders(sources):
    from darsia.fluidflower.series_method import _write_manifest
    target = sources.project / "Archive" / "ProtocolMigration_test"
    def failed_save(path, original, updated):
        _write_manifest(path, original, updated)
        raise OSError("save failed after replacement")
    with pytest.raises(RuntimeError, match="rolled back; active workbooks are unchanged"):
        archive_protocol_sources(sources.plan, sources.package, destination=target, writer=failed_save)
    assert sources.manifest.read_bytes() == sources.original_manifest
    assert all(path.read_bytes() == original for path, original in sources.files.items())
    assert not target.exists()
    assert sources.package["activated"] and not sources.package.get("archived")


def test_misconfigured_protocol_root_cannot_archive_configs_or_original_images(sources):
    document = toml.loads(sources.manifest.read_text())
    document["paths"]["protocols"] = "."
    sources.manifest.write_text(toml.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot contain the DarSIA configuration root"):
        archive_protocol_sources(sources.plan, sources.package)
    assert all(path.read_bytes() == original for path, original in sources.files.items())
    document["paths"]["protocols"] = "protocols"
    document["roots"] = {"raw": str(sources.protocols / "native")}
    sources.manifest.write_text(toml.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="contains configuration or original image data"):
        archive_protocol_sources(sources.plan, sources.package)
    assert all(path.read_bytes() == original for path, original in sources.files.items())
