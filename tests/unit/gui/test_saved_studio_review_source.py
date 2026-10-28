"""The GUI follows the model's saved selected-camera presentation package."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

from darsia.fluidflower import phase_model_catalog as catalog
from darsia.fluidflower.phase_validation import _model_identity
from darsia.fluidflower.ui.model_training import SharedPhaseModelTrainingPage as Page
from .test_phase_model_layout_copy import originals, write  # noqa: F401


@pytest.fixture
def prepared(originals, monkeypatch):
    from darsia.fluidflower import phase_runtime
    # This tiny optical fixture has no physical model or native configuration.
    monkeypatch.setattr(phase_runtime, "load_physics_profile", lambda root: None)
    root = originals.model
    identity = list(_model_identity(root))
    source = root / "selected_studio/validation/prepared_sets/exact-package"
    preparation = dict(status="ready", partial=False, selected_camera_review=True,
                       model_identity=identity, validation=str(source))
    binding = dict(model_id=identity[0], model_sha256=identity[1], completed_utc="2026-01-01T00:00:00Z",
                   physics_profile=None, physics_runtime=None, pipeline_sha256=identity[1],
                   ffac_root=str(source), ffum_root=str(source), combined_root=str(source),
                   selected_candidates={"ac17": "exact-candidate"}, run_count=1, output_mode="review")
    manifest = dict(binding, jobs=[dict(run="ac17", candidate_id="exact-candidate", params_sha256=identity[1])])
    write(root / "studio_preparation.json", preparation)
    write(source / "validation_binding.json", binding)
    write(source / "postflight_manifest.json", manifest)
    entry = dict(root=str(root), model_id=identity[0])
    def no_fallback(*_args):
        pytest.fail("A matching saved package must not fall back to other campaigns")
    page = NS(_read_manifest=Page._read_manifest, _normal_review_collection=no_fallback)
    return NS(root=root, source=source, preparation=preparation, binding=binding,
              manifest=manifest, entry=entry, page=page)


def test_gui_uses_exact_saved_package_with_original_binding_and_pipeline_readers(prepared):
    data = prepared
    before = {p: p.read_bytes() for p in data.root.rglob("*.json")}
    assert not (data.root / "validation/validation_binding.json").exists()
    assert catalog.saved_studio_review_source(data.root) == data.source
    result = Page._validation_binding(data.page, data.entry, review_sessions=[])
    assert result == data.binding
    assert {p: p.read_bytes() for p in data.root.rglob("*.json")} == before


@pytest.mark.parametrize("problem, expected", [
    ("identity", "another model"), ("partial", "incomplete"),
    ("outside", "outside"), ("missing", "postflight_manifest"),
    ("binding", "different model"), ("pipeline", "another model revision"),
])
def test_broken_saved_package_is_reported_without_fallback(prepared, problem, expected):
    data = prepared
    if problem in {"identity", "partial", "outside"}:
        preparation = deepcopy(data.preparation)
        if problem == "identity":
            preparation["model_identity"][0] = "another-model"
        elif problem == "partial":
            preparation["partial"] = True
        else:
            preparation["validation"] = str(data.root.parent / "another-model/validation")
        write(data.root / "studio_preparation.json", preparation)
    elif problem == "missing":
        (data.source / "postflight_manifest.json").unlink()
    else:
        binding = dict(data.binding)
        binding["model_sha256" if problem == "binding" else "pipeline_sha256"] = "0" * 64
        write(data.source / "validation_binding.json", binding)
    assert Page._validation_binding(data.page, data.entry, review_sessions=[]) is None
    assert expected in data.entry["validation_collection_error"]


def test_selected_camera_registration_is_pending_review_not_an_incomplete_attempt(originals, monkeypatch):
    path = originals.model / "summary.json"
    summary = json.loads(path.read_text())
    summary.update(p1_final3=True, display_name="Saved selected-camera review")
    write(path, summary)
    monkeypatch.setattr(catalog, "_registration", lambda root: {
        "registration_kind": "ffac_selected_camera_policy_a_p1_final3_review"})
    entry = catalog.inspect_catalog_entry(originals.model)
    assert entry["state"] == "catalog_verification_pending"
    assert entry["selected_camera_catalog_registered"]
    assert not entry["p1_final3_registered_complete"]
    assert Page._validation_model_label(entry) == "Saved selected-camera review"
