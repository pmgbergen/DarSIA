"""Finite saved metadata, original model identities and real manifest CAS.

These specimens exercise the choice/lock contract. They do not qualify a
campaign model, fit, numerical result, production input or scientific approval.
"""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest
import toml

from darsia.fluidflower import phase_model_library as library
from darsia.fluidflower import series_calibration_approval as approval
from darsia.fluidflower import series_method as methods
from darsia.fluidflower.pipeline.refit_runner import frozen_calibration


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(document, indent=2) + "\n").encode("utf-8"))


def bound_products(root):
    """Original metadata producers; no reader/validator substitutions."""
    root = Path(root)
    root.mkdir(exist_ok=True)
    manifest = root / "series.toml"
    if not manifest.exists():
        manifest.write_bytes(
            b'schema_version = 1\nseries_id = "finite-series"\n'
            b'[future]\nkeep = [1, 2, 3]\n'
        )
    model = root / "models" / "phase-model-finite"
    _write(model / "summary.json", dict(model_id=model.name))
    _write(model / "training_request.json", dict(model_id=model.name))
    for head in ("co2_presence_signal.json", "gas_within_co2_signal.json"):
        _write(model / "models" / "finite" / head, dict(finite_metadata=True))
    pointer = library.index_model(root, model)
    plan = dict(runs=dict(heldout=["heldout"], descriptive=[], excluded=[]),
                rules=dict(finite_metadata=True), baseline=dict(finite_metadata=True),
                chemistry_model="titration")
    fit = dict(parameters=dict(gain=2.0, beta=0.5), calibration_runs=["fit"],
               anchors_by_run={})
    document = frozen_calibration(
        plan, fit, task=dict(configuration=dict(path="finite-plan.json", sha256="a" * 64,
                                              size_bytes=1), commit="b" * 40), files={},
    )
    calibration = root / "named-calibration.json"
    _write(calibration, document)
    return SimpleNamespace(
        root=root, manifest=manifest, model=model, pointer=pointer,
        calibration=calibration, calibration_document=document,
        selection=methods.MethodSelection(active="phase_model", model_version=str(pointer),
                                          mass_calibration=calibration.name),
    )


@pytest.fixture
def saved_choice(tmp_path):
    return bound_products(tmp_path)


def test_explicit_use_binds_files_locks_and_keeps_all_source_bytes(saved_choice):
    case = saved_choice
    before = {p: p.read_bytes() for p in case.root.rglob("*") if p.is_file()}
    original = tomllib.loads(case.manifest.read_text(encoding="utf-8"))
    binding = approval.selection_binding(case.root, case.selection)
    assert approval.read_approval(case.root) is None
    assert {p: p.read_bytes() for p in before} == before  # Passive read only.
    result = approval.approve_series_method(
        case.root, case.selection, expected_bytes=before[case.manifest]
    )
    assert result == case.selection
    record = approval.read_approval(case.root)
    assert record["locked"] is True and record["binding"] == binding
    assert record["events"][-1]["binding"] == binding
    assert record["events"][-1]["actor"]
    after = tomllib.loads(case.manifest.read_text(encoding="utf-8"))
    assert {k: v for k, v in after.items() if k != "method"} == original
    assert all(p.read_bytes() == raw for p, raw in before.items() if p != case.manifest)
    with pytest.raises(ValueError, match="locked"):
        methods.save_series_method(
            case.root, methods.MethodSelection(active="colour_path", colour_calibration="other")
        )


def test_unlock_requires_reason_preserves_prior_identity_and_reapproval_history(saved_choice):
    case = saved_choice
    approval.approve_series_method(case.root, case.selection)
    first = approval.read_approval(case.root)
    original = case.manifest.read_bytes()
    with pytest.raises(ValueError, match="reason"):
        approval.unlock_series_method(case.root, " \t ")
    assert case.manifest.read_bytes() == original
    approval.unlock_series_method(case.root, "Compare a newly fitted calibration.")
    unlocked = approval.read_approval(case.root)
    assert not unlocked["locked"] and unlocked["binding"] == first["binding"]
    assert unlocked["events"][:-1] == first["events"]
    other = case.root / "other-named-calibration.json"
    _write(other, dict(case.calibration_document,
                       parameters=dict(gain=3.0, beta=0.5)))
    changed = replace(case.selection, mass_calibration=other.name)
    approval.approve_series_method(case.root, changed)
    final = approval.read_approval(case.root)
    assert final["locked"] and final["binding"] != first["binding"]
    assert final["events"][:2] == unlocked["events"]
    assert final["events"][0]["binding"] == first["binding"]


@pytest.mark.parametrize("changed", ["calibration", "model", "pointer"])
def test_exact_saved_source_drift_is_refused_and_does_not_unlock(saved_choice, changed):
    case = saved_choice
    approval.approve_series_method(case.root, case.selection)
    manifest = case.manifest.read_bytes()
    path = dict(calibration=case.calibration,
                model=case.model / "models/finite/co2_presence_signal.json",
                pointer=case.pointer)[changed]
    document = json.loads(path.read_bytes())
    document["changed_after_approval"] = True
    _write(path, document)
    with pytest.raises(ValueError, match="changed|identity"):
        approval.read_approval(case.root)
    assert approval.read_approval(case.root, verify=False)["locked"]
    with pytest.raises(ValueError, match="locked"):
        approval.approve_series_method(case.root, case.selection)
    assert case.manifest.read_bytes() == manifest


@pytest.mark.parametrize("chemistry", [None, "simple_flash", "both"])
def test_new_approval_never_infers_or_admits_unsupported_chemistry(saved_choice, chemistry):
    case = saved_choice
    document = dict(case.calibration_document)
    if chemistry is None:
        document.pop("chemistry_model")
    else:
        document["chemistry_model"] = chemistry
    _write(case.calibration, document)
    original = case.manifest.read_bytes()
    with pytest.raises(ValueError, match="explicitly record Titration"):
        approval.approve_series_method(case.root, case.selection)
    assert case.manifest.read_bytes() == original
    # Absence still has its original legacy meaning; no source relabelling or
    # new optional approval is introduced by the old save/read API.
    methods.save_series_method(case.root, case.selection)
    assert methods.read_series_method(case.root) == case.selection
    assert approval.read_approval(case.root) is None
    assert json.loads(case.calibration.read_bytes()) == document


def test_original_manifest_cas_refuses_external_edit_and_incomplete_model(saved_choice):
    case = saved_choice
    original = case.manifest.read_bytes()
    case.manifest.write_bytes(original + b'\nnewer = "retain"\n')
    newer = case.manifest.read_bytes()
    with pytest.raises(methods.ManifestChangedError, match="changed"):
        approval.approve_series_method(case.root, case.selection, expected_bytes=original)
    assert case.manifest.read_bytes() == newer
    # Genuine incomplete model identity, no fake changed SHA or validator seam.
    head = case.model / "models/finite/gas_within_co2_signal.json"
    head.rename(head.with_suffix(".retained"))
    with pytest.raises(ValueError, match="identity changed"):
        approval.approve_series_method(case.root, case.selection)
    assert case.manifest.read_bytes() == newer


def test_use_rechecks_the_exact_files_shown_by_cached_verification(saved_choice):
    case = saved_choice
    shown = approval.selection_binding(case.root, case.selection)
    original = case.manifest.read_bytes()
    case.calibration.write_bytes(case.calibration.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed since they were shown"):
        approval.approve_series_method(case.root, case.selection,
                                      expected_bytes=original, expected_binding=shown)
    assert case.manifest.read_bytes() == original
    assert approval.read_approval(case.root) is None


@pytest.mark.parametrize("model_reference", ["model", "library_pointer"])
def test_relative_saved_references_and_locked_approval_survive_results_move(tmp_path, monkeypatch, model_reference):
    from darsia.fluidflower import series_paths, series_storage_migration
    from darsia.fluidflower.storage_paths import refresh_storage_paths

    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    project = tmp_path / "project"
    project.mkdir()
    case = bound_products(project / "DarSIA")
    monkeypatch.setenv("DARSIA_FINITESERIES_PROJECT_ROOT", str(case.root))
    document = tomllib.loads(case.manifest.read_text())
    document["paths"] = {"results": "results", "phase_model_library": "results/library"}
    case.manifest.write_text(toml.dumps(document))
    source, target = case.root / "results", project / "Results"
    model = source / "models" / case.model.name
    model.parent.mkdir(parents=True)
    case.model.rename(model)
    calibration = source / "calibration" / case.calibration.name
    calibration.parent.mkdir()
    case.calibration.rename(calibration)
    pointer = library.index_model(case.root, model)
    review = source / "calibration/reviews/saved/review.json"
    _write(review, {"review_id": "saved"})
    selected_model = model if model_reference == "model" else pointer
    selection = replace(case.selection, model_version=selected_model.relative_to(case.root).as_posix(),
                        mass_calibration=calibration.relative_to(case.root).as_posix())
    approval.approve_series_method(case.root, selection)
    approved = approval.read_approval(case.root)
    method = tomllib.loads(case.manifest.read_text())["method"]
    payloads = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    plan = series_storage_migration.preview_results_migration(case.root, project, target)
    series_storage_migration.apply_results_migration(plan, writers_quiescent=True)
    refresh_storage_paths(case.root)
    # Actual locations changed; the recorded event, approval and source hashes did not.
    current = approval.selection_binding(case.root, selection)
    assert current != approved["binding"]
    assert current["calibration"]["path"] == str(target / calibration.relative_to(source))
    assert current["phase_model"]["root"] == str(target / model.relative_to(source))
    assert approval.read_approval(case.root) == approved and approved["locked"]
    assert tomllib.loads(case.manifest.read_text())["method"] == method
    assert methods.read_series_method(case.root) == selection
    assert {path.relative_to(target): path.read_bytes() for path in target.rglob("*") if path.is_file()} == payloads
    assert series_paths.series_folder_path(case.root, "phase_model_library") == target / "library"
    assert series_paths.series_path("finite-series", "phase_model_library") == target / "library"
    assert series_paths.calibration_artifact_path(case.root, "calibration_reviews", "saved", "review.json") == target / review.relative_to(source)
    with pytest.raises(ValueError, match="locked"):
        methods.save_series_method(case.root, selection)
    moved_calibration = target / calibration.relative_to(source)
    moved_calibration.write_bytes(moved_calibration.read_bytes() + b" ")
    with pytest.raises(ValueError, match="approved calibration or model changed"):
        approval.read_approval(case.root)
    assert approval.read_approval(case.root, verify=False) == approved


def test_overflow_calibration_with_every_phase_two_field_can_be_approved(saved_choice):
    from darsia.fluidflower.pipeline.local_overflow_refit import FROZEN_FORMAT
    case = saved_choice
    document = dict(case.calibration_document, format=FROZEN_FORMAT, variant_f=True,
        parameters=dict(candidate="N1", gain=2.0, beta=.5), calibration_runs=["fit"],
        anchors_by_run=dict(fit=dict(elapsed_h=2.25, affected_pore_volume_m3=.001)),
        rules=dict(anchor_target_h=2.25, anchor_tolerance_h=.13))
    for field, reason in (("anchors_by_run", "anchor for every fit run"),
                          ("chemistry_model", "explicitly record Titration")):
        _write(case.calibration, {key: value for key, value in document.items() if key != field})
        with pytest.raises(ValueError, match=reason):
            approval.approve_series_method(case.root, case.selection)
    _write(case.calibration, document)
    approval.approve_series_method(case.root, case.selection)
    assert approval.read_approval(case.root)["binding"]["chemistry_model"] == "titration"
