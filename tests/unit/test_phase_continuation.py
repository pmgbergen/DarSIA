"""Genuine saved-weight initialization, immutable parents and strict lineage."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower.phase_features import OpticalFeatureContract
from darsia.fluidflower.phase_training import continuation, logistic
from darsia.fluidflower.phase_training.logistic import ReviewedRegion
from darsia.fluidflower.phase_training.model import DualPhaseSignalModel, TRANSFERABLE_MODEL_ALGORITHM
from .test_raw_photographic_review import inputs  # noqa: F401
from .test_trainer_raw_review import TRAINING_CONFIGURATION, trainer  # noqa: F401


@pytest.fixture
def samples():
    contract = OpticalFeatureContract()
    dimension = len(contract.feature_names)
    parent = DualPhaseSignalModel(feature_mean=tuple(np.full(dimension, 0.25)),
        feature_scale=tuple(np.full(dimension, 2.)), coefficients=tuple(np.zeros(dimension)),
        intercept=0.7, feature_names=contract.feature_names, feature_contract=contract.to_dict(),
        schema=2, algorithm=TRANSFERABLE_MODEL_ALGORITHM, training_provenance=dict(
            series_scope="campaign", campaign_id="saved-v1", head_training_runs=["run1"],
            target_kind="visible_co2_presence_phase_probability"))
    regions = [ReviewedRegion(region_id=f"region-{index}", run="run1", dataset="campaign",
        frame_index=index, elapsed_h=float(index), phase_class=phase, confidence="high",
        source="manual", reviewer="reviewer-a", reviewer_id="reviewer-a", payload=dict(id=f"r{index}"))
        for index, phase in enumerate(("water", "aqueous", "mixed_gas_aqueous"))]
    features = np.zeros((30, dimension), np.float32)
    features[:, 0] = np.repeat([-1., 1., 0.8], 10)
    return parent, contract, features, np.repeat(np.arange(3, dtype=np.int32), 10), regions


def test_saved_parameters_are_the_actual_start_and_parent_scaling_stays_exact(samples, monkeypatch):
    parent, contract, features, owners, regions = samples
    before = parent.to_dict()
    def no_fresh_fit(*args, **kwargs):
        raise AssertionError("A continuation must not initialize from a new pure fit")
    monkeypatch.setattr(logistic, "fit_weighted_logistic", no_fresh_fit)
    fit, diagnostics = continuation.continue_head(parent, features, owners, regions,
        head="co2_presence", feature_contract=contract, regularization=1e-4, max_iter=1)
    other, _ = continuation.continue_head(replace(parent, intercept=-0.7), features, owners, regions,
        head="co2_presence", feature_contract=contract, regularization=1e-4, max_iter=1)
    assert np.array_equal(fit.mean, parent.feature_mean) and np.array_equal(fit.scale, parent.feature_scale)
    assert abs(fit.intercept - parent.intercept) < 0.04
    assert abs(other.intercept - fit.intercept) > 1.3
    assert diagnostics["fresh_fit_performed"] is False and diagnostics["weak_bags"] == 1
    assert parent.to_dict() == before


def test_real_continuation_reduces_the_existing_weighted_objective(samples):
    parent, contract, features, owners, regions = samples
    first, _ = continuation.continue_head(parent, features, owners, regions,
        head="co2_presence", feature_contract=contract, regularization=1e-4, max_iter=1)
    final, diagnostics = continuation.continue_head(parent, features, owners, regions,
        head="co2_presence", feature_contract=contract, regularization=1e-4, max_iter=300)
    assert final.objective < first.objective
    assert final.predict(features[:10]).mean() < final.predict(features[10:20]).mean()
    assert diagnostics["parent_model_sha256"] == parent.to_dict()["model_sha256"]


def test_features_series_and_head_cannot_change_in_continuation(samples):
    parent, contract, features, owners, regions = samples
    arguments = dict(head="co2_presence", feature_contract=contract, regularization=1e-4, max_iter=1)
    with pytest.raises(ValueError, match="exact optical"):
        continuation.continue_head(parent, features, owners, regions,
            **dict(arguments, feature_contract=OpticalFeatureContract(include_dynamic=False)))
    with pytest.raises(ValueError, match="another series"):
        continuation.continue_head(parent, features, owners, [replace(row, dataset="othercampaign") for row in regions], **arguments)
    with pytest.raises(ValueError, match="compatible parent"):
        continuation.continue_head(parent, features, owners, regions, **dict(arguments, head="gas_within_co2"))
    with pytest.raises(ValueError, match="sample"):
        continuation.continue_head(parent, features[:, :-1], owners, regions, **arguments)


def test_each_head_uses_only_its_own_parent_training_runs(samples):
    parent, contract, features, owners, regions = samples
    other_regions = [replace(row, region_id="other-" + row.region_id, run="run2") for row in regions]
    all_regions = regions + other_regions
    all_features = np.concatenate((features, features * -5), axis=0)
    all_owners = np.concatenate((owners, owners + len(regions)))
    arguments = dict(head="co2_presence", feature_contract=contract, regularization=1e-4, max_iter=12)
    before, _ = continuation.continue_head(parent, all_features, all_owners, all_regions, **arguments)
    all_features[len(features):] = 500
    after, _ = continuation.continue_head(parent, all_features, all_owners, all_regions, **arguments)
    assert np.array_equal(before.coefficients, after.coefficients)
    assert before.intercept == after.intercept and before.objective == after.objective
    gas_parent = replace(parent, training_provenance=dict(parent.training_provenance,
        head_training_runs=["run2"], target_kind="visible_gas_within_co2_phase_probability"))
    gas_arguments = dict(arguments, head="gas_within_co2")
    # Supply both direct targets for the conditional head on its own run.
    gas_regions = [replace(row, phase_class="gas" if index % 3 == 0 else "aqueous")
                   for index, row in enumerate(all_regions)]
    gas_before, _ = continuation.continue_head(gas_parent, all_features, all_owners, gas_regions, **gas_arguments)
    all_features[:len(features)] = -500
    gas_after, _ = continuation.continue_head(gas_parent, all_features, all_owners, gas_regions, **gas_arguments)
    assert np.array_equal(gas_before.coefficients, gas_after.coefficients)
    assert gas_before.intercept == gas_after.intercept and gas_before.objective == gas_after.objective


def test_historical_evidence_is_retained_whole_and_unchanged(samples):
    _parent, _contract, _features, _owners, regions = samples
    continuation.require_retained_evidence(regions[:2], regions)
    with pytest.raises(ValueError, match="Historical evidence changed"):
        continuation.require_retained_evidence(regions[:2], [regions[0], replace(regions[1], phase_class="gas"), regions[2]])
    with pytest.raises(ValueError, match="Save new"):
        continuation.require_retained_evidence(regions, regions)
    with pytest.raises(ValueError, match="unique"):
        continuation.require_retained_evidence(regions[:2], regions + [regions[0]])


def test_relocated_same_sha_dependencies_keep_their_evidence_identity(samples):
    _parent, _contract, _features, _owners, regions = samples
    earlier = replace(regions[0], payload=dict(input=dict(path="old/location.npy", sha256="a" * 64)))
    relocated = replace(earlier, payload=dict(input=dict(path="new/location.npy", sha256="a" * 64)))
    continuation.require_retained_evidence([earlier], [relocated, regions[1]])
    different = replace(relocated, payload=dict(input=dict(path="new/location.npy", sha256="b" * 64)))
    with pytest.raises(ValueError, match="Historical evidence changed"):
        continuation.require_retained_evidence([earlier], [different, regions[1]])


def test_bad_optimizer_settings_are_refused(samples):
    parent, contract, features, owners, regions = samples
    with pytest.raises(ValueError, match="settings"):
        continuation.continue_head(parent, features, owners, regions, head="co2_presence",
            feature_contract=contract, regularization=float("nan"))


def test_unindexed_parent_is_refused_before_preparing_a_version(tmp_path):
    with pytest.raises(ValueError, match="indexed parent"):
        continuation.prepare_request(parent_entry=dict(root=str(tmp_path), model_id="saved-v1"),
            output_parent=tmp_path / "models", cache_parent=tmp_path / "cache", bundles=[], actor="reviewer-a")
    assert list(tmp_path.iterdir()) == []


def test_original_evaluation_roles_and_runs_never_enter_continuation_fit(tmp_path, samples):
    import json
    from darsia.fluidflower.phase_saved_assets import asset
    parent, _contract, _features, _owners, regions = samples
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps(dict(rows=[dict(region_id=row.region_id, run=row.run,
        role="evaluation-only" if index else "fit") for index, row in enumerate(regions[:2])])), encoding="utf-8")
    settings = dict(v5_review_fit_partition=asset(partition))
    allowed = continuation.fit_region_ids(regions[:2], regions, settings, dict(co2=parent))
    assert allowed == {regions[0].region_id, regions[2].region_id}
    with pytest.raises(ValueError, match="evaluation runs remain held out"):
        continuation.fit_region_ids(regions[:2], [*regions[:2], replace(regions[2], run="run2")],
                                    settings, dict(co2=parent))


def test_complete_synthetic_raw_producer_keeps_parent_and_publishes_real_heads(tmp_path, trainer, samples, monkeypatch):
    import json
    from darsia.fluidflower.phase_saved_assets import asset
    _trainer, _manifest = trainer
    run = json.loads(_manifest.read_text(encoding="utf-8"))["run"]
    parent_model, _contract, _matrix, _owners, _regions = samples
    configuration = tmp_path / "training_configuration.json"
    configuration.write_text(json.dumps(TRAINING_CONFIGURATION), encoding="utf-8")
    monkeypatch.setenv("DARSIA_NATIVE_CONFIGURATION", str(configuration))
    monkeypatch.setenv("DARSIA_NATIVE_CONFIGURATION_SHA256", asset(configuration)["sha256"])
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path
    def annotation(index, category):
        return dict(id=f"mark-{index}", frame_index=1, elapsed_h=0.5,
            x=(7 + index * 3) / 24, y=6 / 14, width=2 / 24, height=2 / 14,
            confidence="high", category=category)
    earlier = [annotation(index, category) for index, category in enumerate((
        "confirmed_no_co2", "confirmed_only_aqueous_co2", "confirmed_only_free_gas"))]
    decisions = write("earlier.json", dict(decisions={run:dict(reviewer="reviewer-a", annotations=earlier)}))
    combined = write("combined.json", dict(jobs=[dict(run=run, dataset="campaign", review_output_dir=str(tmp_path / "review"),
        native_manifest_sha256=asset(_manifest)["sha256"])]))
    (tmp_path / "project").mkdir()
    settings = dict(series=["campaign"], model_scope="single-series", radiometry_profile="legacy",
        pixels_per_region=8, stage_workers=1, extraction_workers=1, opencv_threads=1,
        mil_max_iters=2, regularization=1e-4, review_project_roots=dict(campaign=str(tmp_path / "project")))
    parent_root = tmp_path / "parent"
    parent_request = write("parent/training_request.json", dict(model_id="saved-v1", settings=settings,
        bundles=[dict(decisions=dict(asset(decisions), snapshot_path=str(decisions)),
                      combined_manifest=dict(asset(combined), snapshot_path=str(combined)))]))
    metadata = {"training_request.json": asset(parent_request)["sha256"]}
    parent_paths = [parent_request]
    for head in continuation.HEADS:
        model = replace(parent_model, training_provenance=dict(parent_model.training_provenance,
            head_training_runs=[run], target_kind=f"visible_{head}_phase_probability"))
        path = model.save(parent_root / "models/generic" / f"{head}_signal.json")
        metadata[f"models/generic/{head}_signal.json"] = asset(path)["sha256"]
        parent_paths.append(path)
    before = {str(path): path.read_bytes() for path in parent_paths}
    added = write("new.json", dict(decisions={run:dict(reviewer="reviewer-a",
        annotations=[annotation(3, "confirmed_only_aqueous_co2")])}))
    request = continuation.prepare_request(parent_entry=dict(root=str(parent_root), model_id="saved-v1",
        library_source_identity=dict(model_id="saved-v1", metadata_sha256=metadata)),
        output_parent=tmp_path / "models", cache_parent=tmp_path / "cache",
        bundles=[(added, combined)], actor="reviewer-a")
    summary = continuation.train(Path(request["request_path"]))
    assert summary["evaluation"]["deployment_allowed"] is False
    assert len(summary["continuation"]["additional_fit_region_ids"]) == 1
    for head, report in summary["models"]["generic"]["heads"].items():
        path = Path(report["model_path"])
        assert path.is_file() and path.is_relative_to(request["output_root"])
        model = DualPhaseSignalModel.load(path)
        assert model.training_provenance["continuation"]["historical_evidence_retained"]
        assert model.training_provenance["cross_validation"]["generalization_validated"] is False
        assert model.feature_mean == parent_model.feature_mean
        assert report["final_fit"]["fresh_fit_performed"] is False
    assert {str(path): path.read_bytes() for path in parent_paths} == before
