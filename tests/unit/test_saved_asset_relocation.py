"""Archived metadata stays SHA-bound; no RAW, arrays or scientific reader changes."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import toml

from darsia.fluidflower import phase_saved_assets as assets
from darsia.fluidflower.storage_paths import refresh_storage_paths


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def aliases(root, values, monkeypatch):
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(root.parent / "registry.toml"))
    (root / "series.toml").write_text(toml.dumps(dict(schema_version=1, series_id="Study",
        storage=dict(path_aliases={str(source): str(target) for source, target in values.items()}))))
    refresh_storage_paths(root)


def test_default_verified_reader_follows_archived_asset_and_keeps_original_descriptor(tmp_path, monkeypatch):
    root = tmp_path / "series"
    source = write(root / "Models/version/display_preparations/receipt.json", {"saved": True})
    descriptor = assets.asset(source)
    original = deepcopy(descriptor)
    archived = root / "Archive/display_preparations"
    archived.parent.mkdir()
    source.parent.rename(archived)
    aliases(root, {source.parent: archived}, monkeypatch)
    reader = assets.VerifiedAssetReader()
    assert reader.path(descriptor) == archived / source.name
    assert reader.json(descriptor) == {"saved": True} and descriptor == original
    (archived / source.name).write_text('{"saved":false}')
    with pytest.raises(ValueError, match="SHA differs"):
        reader.json(descriptor)


def test_selected_inspector_resolves_archived_children_before_original_reader_call(tmp_path, monkeypatch):
    from darsia.fluidflower import phase_saved_selected_registration as registration
    from darsia.fluidflower.phase_model_library import source_identity

    root = tmp_path / "series"
    model = root / "Models/phase-model-saved"
    write(model / "summary.json", {"model_id": model.name})
    for head in ("co2_presence", "gas_within_co2"):
        write(model / "models/generic" / (head + "_signal.json"), {"head": head})
    def identity(folder):
        value = source_identity(folder)
        return value["model_id"], value["version_sha256"]
    model_identity = list(identity(model))
    source = model / "source_alias"
    native = write(source / "native_frames/r1/manifest.json", dict(frame_count=1, frame_sources=[
        dict(frame_index=0, source_sha256="1" * 64, source_work_index=0,
             maps={"combined": {"sha256": "2" * 64}})]))
    native_descriptor = assets.asset(native)
    display = model / "display_preparations"
    gate = write(display / "gate.json", dict(runs={"r1": {}}, selected_frames=1, baseline_frames=0))
    gate_descriptor = assets.asset(gate)
    flags = {key: False for key in ("maps_changed", "scientific_inputs_changed", "labels_changed",
        "sessions_changed", "bottom_mass_allowed", "mass_eligible", "automatic_activation", "round3_allowed",
        "mass_admission", "legacy_strict_gate_claimed")}
    seal = dict(contract="saved-selected-fixture", complete=True, model_identity=model_identity,
        accepted_phase_gate=gate_descriptor, run_count=1, selected_frame_count=1, baseline_frame_count=0,
        timeline_frame_count=1, runs={"r1": dict(native_manifest=native_descriptor, frame_count=1)}, **flags)
    write(model / "seal.json", seal)
    write(model / "studio_preparation.json", dict(model_identity=model_identity, status="ready", partial=False, frame_count=1))
    exports = dict(CONTRACT=seal["contract"], FILES=("seal.json", "studio_preparation.json"), SEAL="seal.json")
    monkeypatch.setattr(registration, "configured_export", lambda section, name, **kwargs: exports[name])
    monkeypatch.setattr(registration, "_model_identity", identity)
    monkeypatch.setattr(registration.review, "load_profile", lambda folder, **kwargs: {"model_identity": list(identity(folder))})
    archived_source, archived_display = root / "Archive/source_alias", root / "Archive/display_preparations"
    archived_source.parent.mkdir()
    before = {path.relative_to(model): path.read_bytes() for path in model.rglob("*") if path.is_file()}
    source.rename(archived_source)
    display.rename(archived_display)
    aliases(root, {source: archived_source, display: archived_display}, monkeypatch)
    calls = []
    def original_reader_adapter(folder, run, *, reader, verify_geometry):
        # Only the unchanged scientific-reader boundary is doubled. The original
        # inspector and exact descriptor/SHA reads execute against moved files.
        calls.append((folder, run, verify_geometry))
        assert Path(folder) == archived_source
        return dict(native_path=reader.path(native_descriptor), native=reader.json(native_descriptor))
    monkeypatch.setattr(registration.review, "load_native_source", original_reader_adapter)
    result = registration.inspect_candidate(model, reader=assets.VerifiedAssetReader())
    assert calls == [(archived_source, "r1", False)]
    assert result["native"] == {"r1": native_descriptor["sha256"]}
    assert result["identity"] == model_identity
    assert not source.exists() and not display.exists()
    assert (model / "models/generic/co2_presence_signal.json").read_bytes() == before[Path("models/generic/co2_presence_signal.json")]
    assert (archived_display / "gate.json").read_bytes() == before[Path("display_preparations/gate.json")]
    assert (model / "seal.json").read_bytes() == before[Path("seal.json")]
