"""Ordinary saved-review and collection metadata survives a moved validation folder."""
import json
from pathlib import Path

import pytest
import toml

from darsia.fluidflower import phase_evidence_audit, phase_runtime, phase_validation
from darsia.fluidflower.storage_paths import refresh_storage_paths
from tests.unit.test_phase_model_folder import review_module  # noqa: F401


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_catalogue_review_jobs_and_collection_reuse_archived_validation(tmp_path, monkeypatch, review_module):
    from darsia.fluidflower.tools import phase_studio_mass_inputs as inputs, phase_review_collections as collections

    root = tmp_path / "series"
    model = root / "Models/phase-model-saved"
    write(model / "summary.json", {"model_id": model.name})
    for head in ("co2_presence", "gas_within_co2"):
        write(model / "models/generic" / (head + "_signal.json"), {"head": head})
    model_id, model_sha = phase_validation._model_identity(model)
    validation = model / "validation"
    folders = {name: validation / name for name in ("candidate", "review", "maps")}
    for folder in folders.values():
        folder.mkdir(parents=True)
    job = dict(run="r1", dataset="study", candidate_id="saved", params_sha256="a" * 64,
        optical_model_sha256=model_sha, physics_runtime=None, output_mode="production", role="candidate",
        family="phase", variant="saved", candidate_folder=str(folders["candidate"]),
        review_output_dir=str(folders["review"]), phase_map_output_dir=str(folders["maps"]))
    write(folders["candidate"] / "summary.json", dict(output_mode="production", selection_candidate_id="saved",
        params_sha256=job["params_sha256"], frame_count=1))
    common = dict(model_id=model_id, model_sha256=model_sha, physics_runtime=None, physics_profile=None,
                  pipeline_sha256=job["params_sha256"])
    binding = dict(common, completed_utc="2026-01-01T00:00:00Z", combined_root=str(validation),
                   selected_candidates={"r1": "saved"}, run_count=1)
    write(validation / "validation_binding.json", binding)
    write(validation / "postflight_manifest.json", dict(common, jobs=[job]))
    original = {path.relative_to(validation): path.read_bytes() for path in validation.rglob("*") if path.is_file()}
    archived = root / "Archive/validation"
    archived.parent.mkdir()
    validation.rename(archived)
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    (root / "series.toml").write_text(toml.dumps(dict(schema_version=1, series_id="Study",
        storage=dict(path_aliases={str(validation): str(archived)}))))
    refresh_storage_paths(root)
    # No scientific profile is installed in this finite saved-metadata fixture.
    # The real model identity, binding, job adaptation and collection guards run.
    monkeypatch.setattr(phase_runtime, "load_physics_profile", lambda folder: None)
    monkeypatch.setattr(collections, "load_physics_profile", lambda folder: None)
    assert phase_evidence_audit.verified_review_binding(model, saved_comparison=True) == binding
    actual_id, dataset, jobs = inputs._prepared_jobs(model, saved_comparison=True)
    assert actual_id == model_id and dataset == "study" and len(jobs) == 1
    assert Path(jobs[0]["combined_root"]) == archived
    assert Path(jobs[0]["review_root"]) == archived / "review"
    assert Path(jobs[0]["candidate"]["phase_map_output_dir"]) == archived / "maps"
    collection = collections.build_collection(model, [], output_root=root / "collections", base_root=validation)
    validated = collections.validated_collection(Path(collection["combined_root"]) / "postflight_manifest.json", model)
    assert validated["binding"]["selected_candidates"] == {"r1": "saved"}
    assert not validation.exists()
    assert {path.relative_to(archived): path.read_bytes() for path in archived.rglob("*") if path.is_file()} == original
    write(archived / "validation_binding.json", dict(binding, model_sha256="0" * 64))
    with pytest.raises(ValueError, match="different model"):
        phase_evidence_audit.verified_review_binding(model, saved_comparison=True)
