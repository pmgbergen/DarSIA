"""New RAW cohorts preserve photometry without inheriting frozen review IDs."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import raw_photographic_review as review
from darsia.fluidflower.native_training_photometry import _fragment
from darsia.fluidflower.phase_saved_assets import asset
from darsia.fluidflower.phase_training import raw_fresh_request as fresh
from .test_raw_frame_context import CONFIGURATION, _reference_context
from .test_raw_photographic_review import NATIVE_HW, inputs  # noqa: F401


@pytest.fixture
def source(tmp_path, inputs):
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return asset(path)
    config = write("configuration.json", CONFIGURATION)
    context = _reference_context(tmp_path / "physical.npz")
    binding = dict(reference_context_path=context["path"], reference_context_sha256=context["sha256"],
                   optical_geometry_qa_pass=True, physical_scale_qa_pass=True)
    context_source = write("context.json", dict(binding=binding))
    photo = tmp_path / "photo.npz"
    np.savez(photo, photo_certified_sand_domain=np.ones(NATIVE_HW, bool),
             photo_boundary_uncertain_domain=np.zeros(NATIVE_HW, bool))
    preparation = json.loads(Path(inputs["reference"]["acquisition_manifest"]["path"]).read_text())
    run = preparation["records"][inputs["reference"]["record_index"]]["source"]["run"]
    manifest = review.write_raw_review_root(tmp_path / "review", run=run,
        native_configuration=inputs["configuration"], raw_binding=inputs["binding"],
        reference=inputs["reference"], frames=inputs["frames"], training_context=dict(
            photo_sand_qa_pass=True, photo_sand_qa_scope="certified-photo-sand-mask",
            photo_certified_sand_mask_path=str(photo), photo_certified_sand_mask_sha256=asset(photo)["sha256"]))
    run = json.loads(manifest.read_text(encoding="utf-8"))["run"]
    decisions = write("decisions.json", dict(decisions={run:dict(annotations=[
        dict(id="mark-a", reviewer_id="reviewer-a", category="confirmed_no_co2"),
        dict(id="mark-b", reviewer_id="reviewer-b", category="confirmed_only_free_gas")])}))
    combined = write("combined.json", dict(jobs=[dict(run=run, dataset="campaign",
        review_output_dir=str(tmp_path / "review"), native_manifest_sha256=asset(manifest)["sha256"])]))
    parent = tmp_path / "parent.json"
    fragment = b'{ "full_native_resolution" : true,\n "display_jpeg_is_training_input":false, "gains":[1.0,1.2,0.8] }'
    parent.write_bytes(b'{"model_id":"saved-version", "settings": { "v5_raw_photometry":' + fragment +
        b', "v5_review_fit_partition":{"old_ids":["frozen-mark"]}, "v5_chemistry_fit_scope":{"excluded_runs":["old-run"]}}}')
    project = tmp_path / "project"
    project.mkdir()
    return dict(output_root=tmp_path / "prepared", parent_request=asset(parent),
        bundles=[dict(decisions=decisions, combined_manifest=combined)],
        series="campaign", reviewers=["reviewer-a"], settings=dict(review_project_roots=dict(campaign=str(project))),
        native_configuration=config, context_source=context_source, actor="reviewer-a")


def test_full_context_request_preserves_exact_photometry_and_selected_reviewers(source):
    parent = Path(source["parent_request"]["path"])
    decisions = Path(source["bundles"][0]["decisions"]["path"])
    before = parent.read_bytes(), decisions.read_bytes()
    request = fresh.prepare_request(**source)
    persisted = Path(request["request_path"]).read_bytes()
    assert request["execution_ready"] is True and request["deployment_allowed"] is False
    assert request["execution_entrypoint"] == "darsia.fluidflower.phase_training.raw_fresh_training"
    assert request["frozen_trainer_compatible"] is False
    assert request["generalization_validated"] is False and request["context_mode"] == "full"
    assert request["settings"]["reviewers"] == ["reviewer-a"]
    assert all(key not in request["settings"] for key in fresh.FROZEN_ANNOTATION_SET_SETTINGS)
    assert set(request["provenance"]["excluded_inherited_settings"]) == set(fresh.FROZEN_ANNOTATION_SET_SETTINGS)
    assert _fragment(_fragment(persisted, "settings"), fresh.PHOTOMETRY_KEY) == _fragment(_fragment(before[0], "settings"), fresh.PHOTOMETRY_KEY)
    snapshot = json.loads(Path(request["bundles"][0]["decisions"]["path"]).read_text(encoding="utf-8"))
    assert [item["id"] for row in snapshot["decisions"].values() for item in row["annotations"]] == ["mark-a"]
    assert (parent.read_bytes(), decisions.read_bytes()) == before
    assert not any("command" in key for key in request)


def test_missing_physical_fields_report_the_actual_gap_before_output(source, tmp_path):
    partial = tmp_path / "partial.npz"
    np.savez(partial, domain=np.ones(NATIVE_HW, bool), facies=np.ones(NATIVE_HW, np.uint8),
             permeability=np.ones(NATIVE_HW), x_m=np.arange(NATIVE_HW[1]), y_m=np.arange(NATIVE_HW[0]))
    original = json.loads(Path(source["context_source"]["path"]).read_text())
    original["binding"].update(reference_context_path=str(partial), reference_context_sha256=asset(partial)["sha256"])
    path = tmp_path / "partial_context.json"
    path.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields:.*physics_supported_domain.*pixel_area_m2"):
        fresh.prepare_request(**dict(source, context_source=asset(path)))
    assert not source["output_root"].exists()


def test_full_context_requires_real_source_and_approved_optical_qa(source, tmp_path):
    with pytest.raises(ValueError, match="complete SHA-bound physical-context"):
        fresh.prepare_request(**dict(source, context_source=None))
    document = json.loads(Path(source["context_source"]["path"]).read_text())
    document["binding"]["optical_geometry_qa_pass"] = False
    path = tmp_path / "unapproved.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="approved QA: optical_geometry_qa_pass"):
        fresh.prepare_request(**dict(source, context_source=asset(path)))
    assert not source["output_root"].exists()


@pytest.mark.parametrize("scale_qa", ["absent", False, None], ids=["absent", "false", "null"])
def test_non_mass_raw_context_preserves_optional_scale_qa(source, tmp_path, scale_qa):
    document = json.loads(Path(source["context_source"]["path"]).read_text())
    if scale_qa == "absent":
        document["binding"].pop("physical_scale_qa_pass")
    else:
        document["binding"]["physical_scale_qa_pass"] = scale_qa
    path = tmp_path / "non_mass_context.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    request = fresh.prepare_request(**dict(source, context_source=asset(path)))
    assert request["context_mode"] == "full" and request["execution_ready"] is True
    assert request["deployment_allowed"] is False and request["generalization_validated"] is False
    assert request["native_configuration"] == source["native_configuration"]
    binding = request["physical_context"]["binding"]
    assert binding["optical_geometry_qa_pass"] is True
    if scale_qa == "absent":
        assert "physical_scale_qa_pass" not in binding
    else:
        assert binding["physical_scale_qa_pass"] is scale_qa
    saved = json.loads(Path(request["request_path"]).read_bytes())
    assert saved["physical_context"]["binding"] == binding
    assert saved["execution_blocker"] == "" and saved["frozen_trainer_compatible"] is False


def test_frozen_ids_empty_reviewers_and_changed_photometry_cannot_enter_new_cohort(source):
    with pytest.raises(ValueError, match="at least one reviewer"):
        fresh.prepare_request(**dict(source, reviewers=[]))
    settings = deepcopy(source["settings"])
    settings["v5_review_fit_partition"] = {"old_ids": ["frozen-mark"]}
    with pytest.raises(ValueError, match="cannot inherit frozen"):
        fresh.prepare_request(**dict(source, settings=settings))
    settings = dict(source["settings"], v5_raw_photometry=dict(changed=True))
    with pytest.raises(ValueError, match="photometry differs"):
        fresh.prepare_request(**dict(source, settings=settings))
    assert not source["output_root"].exists()


def test_photo_only_preparation_explicitly_selects_the_fresh_optical_trainer(source):
    request = fresh.prepare_request(**dict(source, context_mode="photo-only", context_source=None))
    assert request["physical_context"] is None and request["execution_ready"] is True
    assert request["execution_blocker"] == "" and request["frozen_trainer_compatible"] is False
    before = Path(request["request_path"]).read_bytes()
    with pytest.raises(FileExistsError):
        fresh.prepare_request(**dict(source, context_mode="photo-only", context_source=None))
    assert Path(request["request_path"]).read_bytes() == before


def test_no_selected_evidence_refuses_before_output(source):
    with pytest.raises(ValueError, match="No saved RAW review evidence"):
        fresh.prepare_request(**dict(source, reviewers=["reviewer-c"]))
    assert not source["output_root"].exists()


def test_another_declared_series_cannot_supply_input_photometry(source, tmp_path):
    parent = json.loads(Path(source["parent_request"]["path"]).read_bytes())
    parent["settings"]["series"] = ["othercampaign"]
    path = tmp_path / "other_parent.json"
    path.write_text(json.dumps(parent), encoding="utf-8")
    with pytest.raises(ValueError, match="belongs to another series"):
        fresh.prepare_request(**dict(source, parent_request=asset(path)))
    assert not source["output_root"].exists()


def test_preparation_cli_has_no_training_dispatch(source, tmp_path, capsys):
    spec = deepcopy(source)
    spec["output_root"] = str(spec["output_root"])
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert fresh.main(["--spec", str(path), "--spec-sha256", asset(path)["sha256"]]) == 0
    assert json.loads(capsys.readouterr().out)["execution_ready"] is True
    assert (source["output_root"] / "raw_training_preparation.json").is_file()
