"""Exercise deferred saved-registration routes without Qt startup or NAS reads.

Compile the actual repository entrypoints and their DTOs into isolated modules;
only bound metadata readers are stubbed. This executes deferred ImportFrom and
original admission/return logic, rather than checking source spellings.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def _entrypoints(monkeypatch, relative_path, names):
    path = Path(__file__).resolve().parents[2] / relative_path
    parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = []
    for node in parsed.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_BLIND_SEED"
            for target in node.targets
        ):
            selected.append(node)
    assert {node.name for node in selected if hasattr(node, "name")} == set(names)
    module = ModuleType("_saved_registration_route_" + Path(relative_path).stem)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(
        Path=Path, Mapping=Mapping, Sequence=Sequence, Any=Any,
        dataclass=dataclass, replace=replace, hashlib=hashlib, json=json, re=re,
        resolve_review_path=Path,
    )
    unit = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
              *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), module.__dict__)
    return module


@pytest.fixture
def saved_inputs(tmp_path, monkeypatch):
    model = (tmp_path / "model").resolve()
    source = model / "source_alias"
    source.mkdir(parents=True)
    native = source / "ac1.json"
    native.write_text(json.dumps({"frame_count": 2, "frames": [{"index": 0}, {"index": 1}]}),
                      encoding="utf-8")
    native_sha = hashlib.sha256(native.read_bytes()).hexdigest()
    profile = {"format": "ffac-selected-policy-a-review-physics-v1"}
    runtime = {"reader_contract": "local-test-profile", "revision": 1}
    admission = {
        "pipeline_sha256": "a" * 64,
        "model_identity": ["unit-model", "b" * 64],
        "registration": {"native_run_manifest_sha256": {"ac1": native_sha}},
    }
    calls = {"selected": [], "native": [], "profile": [], "registration": []}

    def selected_saved_review(root, *, results=None):
        assert Path(root).resolve() == model
        calls["selected"].append((Path(root), results))
        return admission

    selected = ModuleType("darsia.fluidflower.phase_saved_selected_registration")
    selected.selected_saved_review = selected_saved_review
    monkeypatch.setitem(sys.modules, selected.__name__, selected)

    def load_native_source(root, run, *, verify_geometry=True):
        assert Path(root).resolve() == source and run == "ac1"
        assert verify_geometry is False
        calls["native"].append((Path(root), run, verify_geometry))
        return {"native_path": native, "native": json.loads(native.read_text(encoding="utf-8"))}

    def load_profile(root):
        assert Path(root).resolve() == model
        calls["profile"].append(Path(root))
        return profile

    def runtime_identity(value):
        assert value is profile
        return runtime

    policy = ModuleType("darsia.fluidflower.phase_selected_policy_review")
    policy.load_native_source = load_native_source
    policy.load_profile = load_profile
    policy.runtime_identity = runtime_identity
    monkeypatch.setitem(sys.modules, policy.__name__, policy)

    registration_file = tmp_path / "registration.json"
    registration_file.write_text(json.dumps({
        "registration_kind": "ffac_selected_camera_policy_a_p1_final3_review",
    }), encoding="utf-8")

    def registration_path(model_id):
        assert model_id == "unit-model"
        calls["registration"].append(model_id)
        return registration_file

    registration = ModuleType("darsia.fluidflower.phase_saved_registration")
    registration.registration_path = registration_path
    registration.registration_warning = lambda root: "unused selected-branch warning"
    monkeypatch.setitem(sys.modules, registration.__name__, registration)

    job = {
        "run": "AC1", "candidate_folder": "model", "review_output_dir": "model/source_alias",
        "phase_map_output_dir": "saved_maps", "candidate_id": "selected-unit",
        "params_sha256": admission["pipeline_sha256"], "physics_runtime": runtime,
        "native_manifest_sha256": native_sha, "frame_count": 2, "dataset": "ffac",
        "quality_flags": {"selected_camera_review": True},
    }
    return SimpleNamespace(selection=tmp_path, model=model, native=native, profile=profile,
                           admission=admission, job=job, calls=calls)


def _pairwise(monkeypatch):
    return _entrypoints(
        monkeypatch, "src/darsia/fluidflower/tools/calibration_pairwise_review.py",
        {"Candidate", "ReviewPair", "ReviewInventory", "_run_number", "_sha256_file", "_read_json",
         "_manifest_relative_path", "_selected_policy_a_review_candidate", "_candidate_from_job",
         "discover_review_inventory", "_t4b_saved_review_candidate",
         "_admitted_saved_review_candidate"},
    )


def test_selected_candidate_dispatches_saved_reader_and_keeps_timeline_guard(saved_inputs, monkeypatch):
    pairwise = _pairwise(monkeypatch)
    data = saved_inputs
    original = data.native.read_bytes()
    candidate = pairwise._selected_policy_a_review_candidate(data.selection, data.job, data.model)
    assert data.calls["selected"] == [(data.model, None)]
    assert data.calls["native"] == [(data.model / "source_alias", "ac1", False)]
    assert candidate.run == "ac1" and candidate.frame_count == 2
    assert candidate.phase_model_only is True
    assert candidate.params_sha256 == data.admission["pipeline_sha256"]
    assert candidate.optical_model_sha256 == data.admission["model_identity"][1]
    assert candidate.per_frame_csv == data.native
    assert candidate.review_output_dir == data.model / "source_alias"
    assert candidate.phase_support_format == data.profile["format"]
    assert candidate.provenance()["phase_model_only"] is True
    with pytest.raises(ValueError, match="registered camera timeline"):
        pairwise._selected_policy_a_review_candidate(
            data.selection, dict(data.job, frame_count=3), data.model,
        )
    assert data.native.read_bytes() == original
    assert not (data.model / "no_mass_plot").exists()
    assert not (data.model / "no_movie").exists()


def test_selected_candidate_accepts_archived_source_child_via_shared_alias(saved_inputs, monkeypatch, tmp_path):
    import toml
    from darsia.fluidflower.review_storage import resolve_review_path
    from darsia.fluidflower.storage_paths import refresh_storage_paths

    data = saved_inputs
    original = data.native.read_bytes()
    source = data.model / "source_alias"
    archived = tmp_path / "Archive" / "source_alias"
    archived.parent.mkdir()
    source.rename(archived)
    native = archived / data.native.name
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    (tmp_path / "series.toml").write_text(toml.dumps(dict(schema_version=1, series_id="Study",
        storage=dict(path_aliases={str(source): str(archived)}))))
    refresh_storage_paths(tmp_path)
    policy = sys.modules["darsia.fluidflower.phase_selected_policy_review"]
    def load_archived_native(root, run, *, verify_geometry=True):
        assert Path(root) == archived and verify_geometry is False
        return dict(native_path=native, native=json.loads(native.read_bytes()))
    monkeypatch.setattr(policy, "load_native_source", load_archived_native)
    pairwise = _pairwise(monkeypatch)
    pairwise.resolve_review_path = resolve_review_path
    candidate = pairwise._selected_policy_a_review_candidate(data.selection, data.job, data.model)
    assert candidate.review_output_dir == archived and candidate.per_frame_csv == native
    assert candidate.params_sha256 == data.job["params_sha256"]
    assert not source.exists() and native.read_bytes() == original
    with pytest.raises(ValueError, match="registered camera timeline"):
        pairwise._selected_policy_a_review_candidate(data.selection, dict(data.job, frame_count=3), data.model)


def test_inventory_dispatches_once_and_returns_saved_single_candidate(saved_inputs, monkeypatch):
    pairwise = _pairwise(monkeypatch)
    data = saved_inputs
    manifest = data.selection / "postflight_manifest.json"
    manifest.write_text(json.dumps({"physics_profile": data.profile, "model_root": "model",
                                    "jobs": [data.job]}), encoding="utf-8")
    inventory = pairwise.discover_review_inventory(data.selection, blind_seed="unit-seed")
    assert data.calls["selected"] == [(data.model, None)]
    assert data.calls["native"] == [(data.model / "source_alias", "ac1", False)]
    assert inventory.manifest_path == manifest.resolve()
    assert inventory.unpaired == {} and inventory.pending_runs == ()
    assert len(inventory.pairs) == 1
    pair = inventory.pairs[0]
    assert pair.run == "ac1" and pair.comparison_mode == "single_candidate"
    assert pair.candidate_a is pair.candidate_b
    assert pair.candidate_a.params_sha256 == data.admission["pipeline_sha256"]
    assert pair.candidate_a.phase_support_format == data.profile["format"]
    assert pair.candidate_a.quality_flags == data.job["quality_flags"]
    assert not (data.model / "no_movie").exists()


def test_registered_pipeline_dispatches_reader_and_rejects_other_identity(saved_inputs, monkeypatch):
    training = _entrypoints(monkeypatch, "src/darsia/fluidflower/ui/model_training.py",
                            {"_registered_saved_phase_pipeline"})
    data = saved_inputs
    identity = tuple(data.admission["model_identity"])
    assert training._registered_saved_phase_pipeline(data.model, data.profile, identity) == "a" * 64
    assert data.calls["selected"] == [(data.model, None)]
    assert data.calls["registration"] == ["unit-model"]
    with pytest.raises(ValueError, match="another model revision"):
        training._registered_saved_phase_pipeline(data.model, data.profile, (identity[0], "c" * 64))
    assert len(data.calls["selected"]) == 2
    assert training._registered_saved_phase_pipeline(data.model, {"format": "unrelated"}, identity) is None
    assert len(data.calls["selected"]) == 2


@pytest.fixture
def saved_revision_inputs(tmp_path, monkeypatch):
    model = (tmp_path / "revision_model").resolve()
    selection = model / "validation"
    selection.mkdir(parents=True)
    review = model / "review_source"
    native = review / "native_frames/ac1/manifest.json"
    native.parent.mkdir(parents=True)
    native.write_text(json.dumps({
        "dataset": "ffac", "run": "ac1", "frame_count": 2,
        "frame_sources": [{"index": 0}, {"index": 1}],
    }), encoding="utf-8")
    maps = selection / "phase_maps/ac1"
    maps.mkdir(parents=True)
    checked = {
        "pipeline_sha256": "e" * 64,
        "model_identity": ["joint-unit", "d" * 64],
        "registration": {
            "registration_kind": "ffac_g3_t4b_floor_review",
            "native_run_manifest_sha256": {"ac1": hashlib.sha256(native.read_bytes()).hexdigest()},
        },
    }
    calls = []
    readers = ModuleType("darsia.fluidflower.phase_saved_revision_review")

    def reader(name):
        def saved_review(root):
            assert Path(root).resolve() == model
            calls.append((name, Path(root)))
            return checked
        return saved_review

    for name in ("t4b_saved_review", "t4_saved_review", "l1_neighbourhood_saved_review",
                 "t3_saved_review", "combined_saved_review"):
        setattr(readers, name, reader(name))
    monkeypatch.setitem(sys.modules, readers.__name__, readers)

    registration_file = tmp_path / "revision_registration.json"
    registration_file.write_text(json.dumps({
        "registration_kind": "ffac_g3_t4b_floor_review",
    }), encoding="utf-8")
    registration = ModuleType("darsia.fluidflower.phase_saved_registration")
    registration.registration_path = lambda model_id: registration_file
    registration.registration_warning = lambda root: "Saved revision; no legacy approval warning"
    monkeypatch.setitem(sys.modules, registration.__name__, registration)

    job = {
        "run": "AC1", "role": "candidate", "variant": "joint-head", "family": "phase",
        "candidate_id": "joint-unit-candidate", "grid_order": 1,
        "params_sha256": checked["pipeline_sha256"],
        "optical_model_sha256": checked["model_identity"][1],
        "review_output_dir": str(review), "phase_map_output_dir": str(maps),
        "quality_flags": {"approved_g3_downstream": True,
                          "mass_analysis_allowed": False, "scientific_acceptance": False},
        "output_mode": "review", "phase_model_only": True, "dataset": "ffac",
        "phase_support_format": "saved-joint-head-test",
    }
    identity = dict(model_id=checked["model_identity"][0],
                    model_sha256=checked["model_identity"][1],
                    pipeline_sha256=checked["pipeline_sha256"])
    binding = dict(identity, combined_root=str(selection),
                   selected_candidates={"ac1": job["candidate_id"]})
    (selection / "validation_binding.json").write_text(json.dumps(binding), encoding="utf-8")
    (selection / "postflight_manifest.json").write_text(
        json.dumps(dict(identity, jobs=[job])), encoding="utf-8",
    )
    return SimpleNamespace(model=model, selection=selection, native=native, maps=maps,
                           job=job, checked=checked, calls=calls, registration=registration_file)


def test_joint_head_pairwise_dispatches_saved_reader_and_preserves_review_restrictions(
    saved_revision_inputs, monkeypatch,
):
    pairwise = _pairwise(monkeypatch)
    data = saved_revision_inputs
    before = {path: path.read_bytes() for path in data.model.rglob("*.json")}
    candidate = pairwise._t4b_saved_review_candidate(data.selection, data.job, data.model)
    assert data.calls == [("t4b_saved_review", data.model)]
    assert candidate.params_sha256 == data.checked["pipeline_sha256"]
    assert candidate.optical_model_sha256 == data.checked["model_identity"][1]
    assert candidate.phase_model_only is True and candidate.frame_count == 2
    assert candidate.per_frame_csv == data.native and candidate.phase_map_output_dir == data.maps
    assert candidate.quality_flags["mass_analysis_allowed"] is False
    assert candidate.quality_flags["scientific_acceptance"] is False
    invalid = dict(data.job, quality_flags=dict(data.job["quality_flags"], mass_analysis_allowed=True))
    with pytest.raises(ValueError, match="intact review-only mass restrictions"):
        pairwise._t4b_saved_review_candidate(data.selection, invalid, data.model)
    assert before == {path: path.read_bytes() for path in data.model.rglob("*.json")}


@pytest.mark.parametrize(
    "registration_kind,reader_name,error_prefix",
    [
        ("ffac_g3_t4b_floor_review", "t4b_saved_review", "T4b"),
        ("ffac_g3_t4_floor_review", "t4_saved_review", "T4"),
        ("ffac_l1_neighbourhood_floor_review", "l1_neighbourhood_saved_review", "L1-neighbourhood"),
        ("ffac_g3_t3_floor_review", "t3_saved_review", "T3"),
        ("ffac_g3_t2_l1zone_review", "combined_saved_review", "Combined"),
    ],
)
def test_registered_revision_pipeline_dispatches_and_preserves_model_identity(
    saved_revision_inputs, monkeypatch, registration_kind, reader_name, error_prefix,
):
    training = _entrypoints(monkeypatch, "src/darsia/fluidflower/ui/model_training.py",
                            {"_registered_saved_phase_pipeline"})
    data = saved_revision_inputs
    data.registration.write_text(json.dumps({"registration_kind": registration_kind}), encoding="utf-8")
    identity = tuple(data.checked["model_identity"])
    profile = {"format": "ffac-rig-camera-physics-v1"}
    assert training._registered_saved_phase_pipeline(data.model, profile, identity) == "e" * 64
    assert data.calls == [(reader_name, data.model)]
    with pytest.raises(ValueError, match=error_prefix + " saved maps identify another model revision"):
        training._registered_saved_phase_pipeline(data.model, profile, (identity[0], "f" * 64))
    assert data.calls == [(reader_name, data.model), (reader_name, data.model)]
