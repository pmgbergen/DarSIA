"""Saved-export UI status stays bound to the current series and revision."""

import copy
import json
from pathlib import Path

import pytest

from darsia.fluidflower.setup_display import (
    completed_current_native_setup_export,
)

REVISION = "a" * 32


@pytest.fixture
def exported_package(tmp_path):
    current = {
        "schema": 1,
        "format": "darsia-fluidflower-geometry-facies-v1",
        "revision_id": REVISION,
        "recorded_utc": "2026-10-15T00:00:00Z",
        "configuration": {},
    }
    pointer = tmp_path / "overrides" / "geometry_facies" / "current.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_bytes(json.dumps(current).encode("utf-8"))
    destination = tmp_path / "exports" / "native_setup" / REVISION
    destination.mkdir(parents=True)
    files = {}
    for name in (
        "colored_labels.png",
        "native_labels.npz",
        "facies_properties.csv",
        "native_setup.toml",
    ):
        # Deliberately not valid image/NPZ/TOML: the UI metadata reader must not
        # parse or hash these bytes. Real output checks belong to the exporter.
        (destination / name).write_bytes(b"synthetic artifact")
        files[name] = {"sha256": "b" * 64}
    receipt = {
        "schema": 1,
        "format": "darsia-fluidflower-native-setup-export-v1",
        "revision_id": REVISION,
        "created_utc": "2026-10-15T00:00:01Z",
        "reference_run": "AC52",
        "files": files,
    }
    manifest = destination / "export.json"
    manifest.write_bytes(json.dumps(receipt).encode("utf-8"))
    return tmp_path, pointer, manifest, receipt


def test_matching_current_export_returns_actual_unmodified_receipt(exported_package):
    package, _pointer, _manifest, receipt = exported_package
    assert completed_current_native_setup_export(package) == receipt


def test_old_export_does_not_mark_new_current_revision(exported_package):
    package, pointer, _manifest, _receipt = exported_package
    current = json.loads(pointer.read_bytes())
    current["revision_id"] = "c" * 32
    pointer.write_bytes(json.dumps(current).encode("utf-8"))
    assert completed_current_native_setup_export(package) is None


def test_mismatching_receipt_revision_is_rejected(exported_package):
    package, _pointer, manifest, receipt = exported_package
    stale = copy.deepcopy(receipt)
    stale["revision_id"] = "c" * 32
    manifest.write_bytes(json.dumps(stale).encode("utf-8"))
    assert completed_current_native_setup_export(package) is None


@pytest.mark.parametrize(
    "name",
    [
        "colored_labels.png",
        "native_labels.npz",
        "facies_properties.csv",
        "native_setup.toml",
    ],
)
def test_missing_required_artifact_does_not_mark_applied(exported_package, name):
    package, _pointer, manifest, _receipt = exported_package
    (manifest.parent / name).unlink()
    assert completed_current_native_setup_export(package) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", 2),
        ("format", "other-export"),
        ("files", []),
    ],
)
def test_malformed_export_schema_is_rejected(exported_package, field, value):
    package, _pointer, manifest, receipt = exported_package
    malformed = copy.deepcopy(receipt)
    malformed[field] = value
    manifest.write_bytes(json.dumps(malformed).encode("utf-8"))
    assert completed_current_native_setup_export(package) is None


def test_one_series_never_inherits_another_series_export(exported_package, tmp_path):
    package, _pointer, _manifest, _receipt = exported_package
    other = tmp_path / "unexported_other_series"
    other.mkdir()
    assert completed_current_native_setup_export(package) is not None
    assert completed_current_native_setup_export(other) is None


def test_revision_cannot_escape_selected_geometry_package(exported_package):
    package, pointer, _manifest, _receipt = exported_package
    current = json.loads(pointer.read_bytes())
    current["revision_id"] = "../../other_series"
    pointer.write_bytes(json.dumps(current).encode("utf-8"))
    assert completed_current_native_setup_export(package) is None


def test_artifacts_are_not_opened_or_hashed(exported_package, monkeypatch):
    package, _pointer, _manifest, receipt = exported_package
    original = Path.read_bytes

    def metadata_only(path):
        assert path.name in {"current.json", "export.json"}
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", metadata_only)
    assert completed_current_native_setup_export(package) == receipt
