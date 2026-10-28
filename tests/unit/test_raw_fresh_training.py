"""Fresh RAW training reuses real extraction, MIL and complete-run validation."""

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import raw_photographic_review as review
from darsia.fluidflower.native.config import configure_parameters, reset_parameters
from darsia.fluidflower.native_training_photometry import _fragment
from darsia.fluidflower.phase_saved_assets import asset
from darsia.fluidflower.phase_training import raw_fresh_request as preparation
from darsia.fluidflower.phase_training import raw_fresh_training as fresh
from .test_raw_frame_context import NATIVE_HW, _reference_context
from .test_raw_photographic_review import _record, _write_json, inputs  # noqa: F401
from .test_trainer_raw_review import TRAINING_CONFIGURATION


def _annotation(identifier, category, *, column, row=4):
    """Describe a small native rectangle on the fixture's preserved review canvas."""
    return dict(id=identifier, reviewer_id="reviewer", reviewer="reviewer",
                category=category, confidence="high", frame_index=1, elapsed_h=0.5,
                x=(5 + column) / 24, y=(3 + row) / 14,
                width=2 / 24, height=2 / 14, shape="rectangle")


@pytest.fixture
def tiny_cohort(tmp_path, inputs, monkeypatch, request):
    """Freeze two local RAW experiments with actual float pixels and full context."""
    frame_offset = getattr(request, "param", 0.0)
    configuration = dict(json.loads(Path(inputs["configuration"]["path"]).read_bytes()),
                         **deepcopy(TRAINING_CONFIGURATION))
    configured = _write_json(tmp_path / "training_configuration.json", configuration)
    # The existing display-only fixture captures this descriptor object.
    inputs["configuration"].update(configured)
    context = _reference_context(tmp_path / "physical_context.npz")
    physical = _write_json(tmp_path / "context_source.json", dict(binding=dict(
        reference_context_path=context["path"], reference_context_sha256=context["sha256"],
        optical_geometry_qa_pass=True)))
    certified = np.zeros(NATIVE_HW, dtype=bool)
    certified[2:9, 3:14] = True
    photo_path = tmp_path / "photo_sand.npz"
    np.savez(photo_path, photo_certified_sand_domain=certified,
             photo_boundary_uncertain_domain=~certified)
    photo = asset(photo_path)
    photographic = dict(photo_sand_qa_pass=True, photo_sand_qa_scope="certified-photo-sand-mask",
                        photo_certified_sand_mask_path=photo["path"],
                        photo_certified_sand_mask_sha256=photo["sha256"])
    series, runs = "synthetic", ("samplea", "sampleb")
    root = tmp_path / "reviews"
    manifests, decisions, jobs = [], {}, []
    for run in runs:
        baseline = _record(tmp_path, run + "_baseline", raw_sha="", binding=inputs["binding"])
        native_row = _write_json(baseline["folder"] / "native_row.json",
                                 dict(raw_producer_audit=baseline["audit"]))
        acquisitions = _write_json(tmp_path / (run + "_acquisitions.json"), dict(records=[dict(
            source=dict(run=run), full=baseline["full"], native_row=native_row)]))
        frames = []
        for index in range(2):
            case = run + "_frame_" + str(index)
            shift_x = index + frame_offset if frame_offset else index
            record = _record(tmp_path, case, raw_sha="", binding=inputs["binding"], shift_x=shift_x)
            native = _write_json(record["folder"] / "native.json", dict(
                complete=True, processor_kind="raw_rig", audit=record["audit"], full=record["full"],
                source_sha256=record["camera"]["sha256"]))
            frames.append(dict(case_id=case, frame_index=index, elapsed_h=0.5 * index,
                               native_record=native,
                               native_geometry=_write_json(record["folder"] / "native_geometry.json", {}),
                               rig_geometry=_write_json(record["folder"] / "rig_geometry.json", {}),
                               photo_support=record["valid"]))
        manifest = review.write_raw_review_root(root, run=run, native_configuration=configured,
            raw_binding=inputs["binding"], reference=dict(acquisition_manifest=acquisitions,
                record_index=0, photo_support=baseline["valid"]), frames=frames,
            training_context=photographic)
        manifests.append(asset(manifest))
        decisions[run] = dict(reviewer_id="reviewer", annotations=[
            _annotation("water", "confirmed_no_co2", column=4),
            _annotation("aqueous", "confirmed_aqueous_co2", column=7),
            _annotation("gas", "confirmed_only_free_gas", column=10),
            _annotation("mixed", "confirmed_mixed_free_aqueous_co2", column=8, row=6)])
        jobs.append(dict(run=run, dataset=series, review_output_dir=str(root),
                         native_manifest_sha256=asset(manifest)["sha256"]))
    evidence = _write_json(tmp_path / "decisions.json", dict(decisions=decisions))
    combined = _write_json(tmp_path / "combined.json", dict(jobs=jobs))
    parent = tmp_path / "parent_request.json"
    fragment = b'{ "full_native_resolution" : true,\n "display_jpeg_is_training_input":false, "gains":[1.0,1.2,0.8] }'
    parent.write_bytes(b'{"model_id":"saved-input", "settings":{"v5_raw_photometry":' + fragment + b'}}')
    project = tmp_path / "project"
    project.mkdir()
    settings = dict(model_scope="single-series", series=[series], reviewers=["reviewer"],
                    review_project_roots={series: str(project)}, backend="cpu", fit_mode="mil",
                    regularization_grid=[1.0], weak_strength=0.35, positive_top_fraction=0.1,
                    mil_max_iters=3, pixels_per_region=8, stage_workers=1, extraction_workers=1,
                    opencv_threads=1, texture_sigma=2.0, evaluation_mode="whole-run",
                    change_evidence_ablation=False)
    prepare_kwargs = dict(
        parent_request=asset(parent), bundles=[dict(decisions=evidence, combined_manifest=combined)],
        series=series, reviewers=["reviewer"], settings=settings,
        native_configuration=configured, context_source=physical)
    request = preparation.prepare_request(output_root=tmp_path / "prepared", **prepare_kwargs)
    monkeypatch.setenv("DARSIA_NATIVE_CONFIGURATION", configured["path"])
    monkeypatch.setenv("DARSIA_NATIVE_CONFIGURATION_SHA256", configured["sha256"])
    names = ("darsia.fluidflower.phase_training.frame_context", "darsia.fluidflower.phase_training.trainer")
    loaded = {name: name in sys.modules for name in names}
    token = configure_parameters(configuration)
    yield SimpleNamespace(request=request, preparation=asset(request["request_path"]),
                          manifests=manifests, parent=parent, runs=runs, configuration=configured,
                          output=tmp_path / "trained", cache=tmp_path / "cache",
                          prepare_kwargs=prepare_kwargs)
    reset_parameters(token)
    for name in names:
        if not loaded[name]:
            sys.modules.pop(name, None)
            vars(sys.modules["darsia.fluidflower.phase_training"]).pop(name.rsplit(".", 1)[1], None)


def test_fit_view_remaps_owners_without_evaluation_only_labels():
    """An unsampled evaluation label must not create an empty validation fold."""
    matrix = np.arange(9, dtype=np.float32).reshape(3, 3)
    owners = np.asarray([0, 2, 2], dtype=np.int32)
    regions = [SimpleNamespace(region_id="fit-first"), SimpleNamespace(region_id="evaluation"),
               SimpleNamespace(region_id="fit-last")]
    selected, remapped, fitted = fresh._fit_view(matrix, owners, regions)
    assert np.array_equal(selected, matrix)
    assert remapped.dtype == np.int32 and remapped.tolist() == [0, 1, 1]
    assert [region.region_id for region in fitted] == ["fit-first", "fit-last"]
    assert owners.tolist() == [0, 2, 2] and len(regions) == 3


def test_automatic_cpu_selection_uses_cpu_numerical_validation(tiny_cohort, monkeypatch):
    """Resolve automatic CPU execution before selecting the existing parity helper."""
    from darsia.fluidflower.phase_training import trainer

    monkeypatch.setattr(trainer, "cuda_available", lambda: False)
    effective = fresh._effective_settings(dict(tiny_cohort.request["settings"], backend="auto"), trainer)
    assert effective["backend"] == "cpu"
    assert effective["regularization_grid"] == [1.0]
    assert not set(preparation.FROZEN_ANNOTATION_SET_SETTINGS) & set(effective)


def test_two_raw_experiments_extract_fit_validate_and_serialize(tiny_cohort):
    """Exercise real RAW features, mixed-region MIL, two run folds and saved heads."""
    from darsia.fluidflower.phase_training.model import DualPhaseSignalModel
    from darsia.fluidflower.phase_training import trainer

    source_files = dict(trainer=Path(trainer.__file__),
        training_primitives=trainer.PACKAGE_ROOT / "logistic.py",
        optical_features=trainer.FLUIDFLOWER_ROOT / "phase_features.py",
        fresh_runner=Path(fresh.__file__),
        reviewed_raw_basis=Path(fresh.__file__).with_name("raw_review_training.py"))
    source_sha256 = {key: sha256(path.read_bytes()).hexdigest() for key, path in source_files.items()}

    originals = {Path(bound["path"]): Path(bound["path"]).read_bytes()
                 for bound in tiny_cohort.manifests}
    originals[tiny_cohort.parent] = tiny_cohort.parent.read_bytes()
    originals[Path(tiny_cohort.preparation["path"])] = Path(tiny_cohort.preparation["path"]).read_bytes()
    summary = fresh.train(tiny_cohort.preparation, output_root=tiny_cohort.output, cache_root=tiny_cohort.cache)
    assert summary["deployment_allowed"] is False
    assert summary["provenance"]["source_sha256"] == source_sha256
    reports = summary["models"]["generic"]["heads"]
    assert set(reports) == {"co2_presence", "gas_within_co2"}
    for report in reports.values():
        assert report["fit_mode"] == "mil" and report["eligible_runs"] == list(tiny_cohort.runs)
        assert len(report["regularization_grid"]) == 1
        assert {fold["held_out_run"] for fold in report["regularization_grid"][0]["folds"]} == set(tiny_cohort.runs)
        assert report["final_fit"]["backend"] == "cpu"
        assert report["final_fit"]["diagnostics"]["weak_bags"] > 0
        assert report["numerical_validation"]["passed"] is True
        model_path = Path(report["model_path"])
        saved = json.loads(model_path.read_bytes())
        model = DualPhaseSignalModel.from_dict(saved)
        assert saved["model_sha256"] == report["model_sha256"]
        assert model.training_provenance["deployment_allowed"] is False
        assert model.training_provenance["source_sha256"] == source_sha256
        backend = model.training_provenance["backend_contract"]
        assert backend["production_backend"] == "cpu" and backend["production_precision"] == "float64"
        assert backend["authoritative_reference"] == "cpu-float64"
        assert backend["validated_backends"] == ["cpu"]
        assert backend["parity"] == report["numerical_validation"]
    assert summary["feature_extraction"]
    assert all(report["pixel_rows"] > 0 for report in summary["feature_extraction"])
    views = summary["provenance"]["original_to_context_views"]
    assert len(views) == len(tiny_cohort.runs)
    original_shas = {bound["sha256"] for bound in tiny_cohort.manifests}
    assert {view["original_manifest"]["sha256"] for view in views} == original_shas
    for view in views:
        original = json.loads(originals[Path(view["original_manifest"]["path"])])
        derivative = json.loads(Path(view["extraction_manifest"]["path"]).read_bytes())
        assert {key: value for key, value in derivative.items() if key != "training_context"} == {
            key: value for key, value in original.items() if key != "training_context"}
        assert derivative["training_context"] == dict(original["training_context"],
            **tiny_cohort.request["physical_context"]["binding"])
        assert view["original_manifest"]["sha256"] != view["extraction_manifest"]["sha256"]
        assert asset(view["extraction_manifest"]["path"])["sha256"] == view["extraction_manifest"]["sha256"]
        derivations = view["full_context_derivations"]
        assert len(derivations) == 1 and derivations[0]["frame_index"] == 1
        assert derivations[0]["receipt_sha256"] == original["frames"][1]["audit"]["sha256"]
        exports = TRAINING_CONFIGURATION["saved_review_metadata"]["training_frame_context"]["exports"]
        assert set(derivations[0]["fields"]) == {
            "observable_phase_domain", *exports["PHOTO_FIELDS"]["value"]}
    saved_request = (tiny_cohort.output / "training_request.json").read_bytes()
    assert json.loads(saved_request)["provenance"]["source_sha256"] == source_sha256
    expected_fragment = _fragment(_fragment(tiny_cohort.parent.read_bytes(), "settings"), preparation.PHOTOMETRY_KEY)
    assert _fragment(_fragment(saved_request, "settings"), preparation.PHOTOMETRY_KEY) == expected_fragment
    assert all(path.read_bytes() == before for path, before in originals.items())
    before = {path: path.read_bytes() for path in tiny_cohort.output.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        fresh.train(tiny_cohort.preparation, output_root=tiny_cohort.output, cache_root=tiny_cohort.cache)
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize("tiny_cohort", [0.25], indirect=True)
def test_full_context_footprint_failure_precedes_extraction(tiny_cohort, tmp_path, monkeypatch):
    """A valid reference archive cannot bypass strict per-frame physical admission."""
    from darsia.fluidflower.phase_training import trainer
    from darsia.fluidflower.phase_training.raw_frame_context import RawFrameGeometryCohort

    originals = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    binding = json.loads(Path(tiny_cohort.prepare_kwargs["context_source"]["path"]).read_bytes())["binding"]
    with np.load(binding["reference_context_path"], allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    edge = np.zeros(NATIVE_HW, dtype=bool)
    edge[3:7, :4] = True
    for key in ("domain", "physics_supported_domain", "physical_sand_domain", "facies_safe_interior"):
        arrays[key] = edge
    arrays["facies"] = edge.astype(np.uint8) * 2
    edge_path = tmp_path / "edge_context.npz"
    np.savez(edge_path, **arrays)
    edge_asset = asset(edge_path)
    edge_source = _write_json(tmp_path / "edge_context_source.json", dict(binding=dict(binding,
        reference_context_path=edge_asset["path"], reference_context_sha256=edge_asset["sha256"])))
    kwargs = dict(tiny_cohort.prepare_kwargs, context_source=edge_source)
    prepared = preparation.prepare_request(output_root=tmp_path / "edge_prepared", **kwargs)
    assert prepared["physical_context"]["archive"] == {
        key: edge_asset[key] for key in ("path", "sha256")}
    assert prepared["execution_ready"] is True
    manifest = json.loads(Path(tiny_cohort.manifests[0]["path"]).read_bytes())
    cohort = RawFrameGeometryCohort.from_manifest(tiny_cohort.manifests[0]["path"],
        context_binding=dict(manifest["training_context"], **prepared["physical_context"]["binding"]))
    fields = ("observable_phase_domain", "photo_certified_sand_domain", "photo_boundary_uncertain_domain")
    photo, _ = cohort.context(1, fields=fields)
    assert photo["observable_phase_domain"].all()
    assert cohort.frame(1)["H_ac28_to_frame_native"][0, 2] == 2.25

    def no_extraction(*args, **kwargs):
        pytest.fail("Feature extraction started before full-context admission")

    monkeypatch.setattr(trainer, "_load_feature_caches", no_extraction)
    with pytest.raises(ValueError, match="porous domain escapes exact reference support"):
        fresh.train(asset(prepared["request_path"]), output_root=tiny_cohort.output,
                    cache_root=tiny_cohort.cache)
    assert tiny_cohort.output.is_dir()
    assert (tiny_cohort.output / "input_snapshot/native_context_views/001.json").is_file()
    assert not (tiny_cohort.output / "summary.json").exists()
    assert not (tiny_cohort.output / "models").exists()
    assert not tiny_cohort.cache.exists()
    assert all(path.read_bytes() == before for path, before in originals.items())


def test_train_restores_active_configuration_and_environment_after_failure(tiny_cohort, monkeypatch):
    """Explicit bound configuration overrides prior state only for this execution."""
    from darsia.fluidflower.native.config import require_configuration

    previous = {"prior_configuration": True}
    monkeypatch.setenv("DARSIA_NATIVE_CONFIGURATION", "prior-configuration.json")
    monkeypatch.delenv("DARSIA_NATIVE_CONFIGURATION_SHA256", raising=False)
    token = configure_parameters(previous)
    expected = json.loads(Path(tiny_cohort.configuration["path"]).read_bytes())
    entered = []

    def fail_execution(prepared, *, output_root, cache_root):
        entered.append(True)
        assert prepared == tiny_cohort.preparation
        assert (output_root, cache_root) == (tiny_cohort.output, tiny_cohort.cache)
        assert require_configuration() == expected
        assert require_configuration() is not previous
        assert os.environ["DARSIA_NATIVE_CONFIGURATION"] == str(Path(tiny_cohort.configuration["path"]).resolve())
        assert os.environ["DARSIA_NATIVE_CONFIGURATION_SHA256"] == tiny_cohort.configuration["sha256"]
        raise RuntimeError("Synthetic execution failure")

    monkeypatch.setattr(fresh, "_train", fail_execution)
    try:
        with pytest.raises(RuntimeError, match="Synthetic execution failure"):
            fresh.train(tiny_cohort.preparation, output_root=tiny_cohort.output, cache_root=tiny_cohort.cache)
        assert entered == [True]
        assert require_configuration() is previous
        assert os.environ["DARSIA_NATIVE_CONFIGURATION"] == "prior-configuration.json"
        assert "DARSIA_NATIVE_CONFIGURATION_SHA256" not in os.environ
        assert not tiny_cohort.output.exists() and not tiny_cohort.cache.exists()
    finally:
        reset_parameters(token)
