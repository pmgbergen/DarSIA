"""One PhaseModel folder per series: [paths] phase_model and photograph identity.

A temporary series.toml decides every location. Without [paths] phase_model each
location must be exactly the historical one; with it, everything used to train a
phase model lands below PhaseModel/Annotations and PhaseModel/Model.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from darsia.fluidflower import series_paths

SERIES = "FF_TEST"
KEY = "fftest"


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Only the temporary registry and its series are visible."""
    for name in tuple(os.environ):
        if (name.startswith("DARSIA_") and name.endswith("_PROJECT_ROOT")) or name in {
            "DARSIA_FLUIDFLOWER_DATA_ROOT", "DARSIA_FLUIDFLOWER_REVIEW_ROOT",
            "DARSIA_REVIEW_ALIAS_FILES", "FLUIDFLOWER_PHASE_COLLECTION_ROOT",
            "DARSIA_FLUIDFLOWER_PHASE_MODEL_DEFAULTS",
        }:
            monkeypatch.delenv(name)
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    monkeypatch.setenv("DARSIA_FLUIDFLOWER_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("DARSIA_FLUIDFLOWER_PHASE_MODEL_DEFAULTS", str(tmp_path / "no-defaults.json"))
    return tmp_path


def make_series(tmp_path: Path, paths: dict[str, str], name: str = "series", *, register=True) -> Path:
    """Write (and register) a series.toml with the given [paths] table."""
    root = tmp_path / name
    root.mkdir()
    lines = ["schema_version = 1", f'series_id = "{SERIES}"', 'name = "Test series"', "", "[paths]"]
    lines += [f"{key} = {json.dumps(value)}" for key, value in paths.items()]
    (root / "series.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if register:
        series_paths.register_series(root)
    return root.resolve()


def phase_model_layout(root: Path, *, canonical=False, expert=False) -> dict[str, Path]:
    phase_root = root / "ExpertKnowledge" / "PhaseModels" if expert else root / "PhaseModels"
    annotations, model = (phase_root / "TrainingInput", phase_root / "Models") if canonical else (
        root / "PhaseModel" / "Annotations", root / "PhaseModel" / "Model")
    return {
        "phase_annotations": annotations,
        "reviews": annotations,
        "review_collections": annotations / "review_collections",
        "evidence_actions": annotations / "evidence_actions",
        "evidence_audits": annotations / "evidence_audits",
        "training_enrollments": annotations / "training_enrollments",
        "phase_models": model,
        "phase_model_library": model / "library",
        "models": root / "models",
        "results": root / "results",
    }


def historical_layout(root: Path) -> dict[str, Path]:
    """The locations of the base revision, before [paths] phase_model existed."""
    return {
        "reviews": root / "reviews",
        "review_collections": root / "phase_review_collections",
        "models": root / "models",
        "training": root / "training",
        "evidence_actions": root / "reviews" / "evidence_actions",
        "evidence_audits": root / "reviews" / "evidence_audits",
        "phase_model_library": root / "models" / "phase",
        "phase_annotations": root / "reviews",
        "phase_models": root / "models",
    }


def test_phase_model_moves_every_training_folder(isolated):
    root = make_series(isolated, {"phase_model": "PhaseModel"})
    for key, expected in phase_model_layout(root).items():
        assert series_paths.series_path(SERIES, key) == expected, key
        assert series_paths.series_folder_path(root, key) == expected, key
    assert series_paths.phase_model_root(root) == root / "PhaseModel"
    assert series_paths.model_versions_folder(root) == root / "PhaseModel" / "Model"
    # Before the layout, the series used the historical folders.
    assert series_paths.series_folder_path(root, "evidence_actions", before_phase_model=True) == (
        root / "reviews" / "evidence_actions")


def test_without_phase_model_every_location_is_unchanged(isolated):
    root = make_series(isolated, {})
    for key, expected in historical_layout(root).items():
        assert series_paths.series_path(SERIES, key) == expected, key
        assert series_paths.series_folder_path(root, key) == expected, key
    assert series_paths.phase_model_root(root) is None
    with pytest.raises(KeyError, match="Unknown series path"):
        series_paths.series_path(SERIES, "phase_model")


def test_customised_historical_paths_keep_their_old_meaning(isolated):
    root = make_series(isolated, {"reviews": "custom reviews", "models": "custom models"})
    assert series_paths.series_path(SERIES, "reviews") == root / "custom reviews"
    assert series_paths.series_path(SERIES, "models") == root / "custom models"
    # The trained versions followed [paths] models before; the ledger, audits
    # and library index were fixed folders and stay so.
    assert series_paths.series_path(SERIES, "phase_models") == root / "custom models"
    assert series_paths.series_path(SERIES, "evidence_actions") == root / "reviews" / "evidence_actions"
    assert series_paths.series_path(SERIES, "evidence_audits") == root / "reviews" / "evidence_audits"
    assert series_paths.series_path(SERIES, "phase_model_library") == root / "models" / "phase"
    # Callers holding only the folder always used its fixed models folder.
    assert series_paths.model_versions_folder(root) == root / "models"


def test_explicit_paths_win_over_phase_model(isolated):
    root = make_series(isolated, {
        "phase_model": "PhaseModel", "reviews": "reviews", "evidence_actions": "ledger",
        "phase_annotations": "Labels",
    })
    assert series_paths.series_path(SERIES, "reviews") == root / "reviews"
    assert series_paths.series_path(SERIES, "evidence_actions") == root / "ledger"
    assert series_paths.series_path(SERIES, "review_collections") == root / "Labels" / "review_collections"
    assert series_paths.series_path(SERIES, "phase_models") == root / "PhaseModel" / "Model"
    absolute = isolated / "elsewhere"
    root = make_series(isolated, {"phase_model": str(absolute)}, name="absolute", register=False)
    assert series_paths.series_folder_path(root, "reviews") == absolute / "Annotations"


def test_new_series_from_the_wizard_uses_the_phase_model_folder(isolated):
    from darsia.fluidflower.series_creation import PHASE_MODEL_FOLDER, _series_toml

    draft = NS(runs=[NS(run_id="r1")], series_id=SERIES, name="Test series",
               rig_class="medium", created_by="test")
    paths = tomllib.loads(_series_toml(draft, "2026-01-01T00:00:00Z"))["paths"]
    assert paths["phase_model"] == "../Results/calibration/" + PHASE_MODEL_FOLDER
    # Folders it replaces are derived, so they are not written explicitly.
    assert not {"reviews", "review_collections", "training"} & set(paths)
    root = make_series(isolated, {key: value for key, value in paths.items() if key != "root"})
    expected_layout = phase_model_layout(root.parent / "Results/calibration", canonical=True, expert=True)
    expected_layout["models"] = expected_layout["phase_models"]
    expected_layout["results"] = (root.parent / "Results").resolve()
    for key, expected in expected_layout.items():
        assert series_paths.series_path(SERIES, key) == expected, key


@pytest.mark.parametrize("phase_root", ["PhaseModels", "ExpertKnowledge/PhaseModels"])
def test_canonical_phase_models_keeps_training_and_validation_together_with_separate_models(isolated, phase_root):
    from darsia.fluidflower.phase_model_library import index_model, records_for_location
    from darsia.fluidflower.review_storage import series_review_root
    from darsia.fluidflower.tools.phase_review_collections import collection_root
    from darsia.fluidflower.ui.model_training import SharedPhaseModelTrainingPage

    root = make_series(isolated, {"phase_model": phase_root})
    layout = phase_model_layout(root, canonical=True, expert=phase_root.startswith("ExpertKnowledge/"))
    assert series_review_root(SERIES) == layout["phase_annotations"]
    assert collection_root(SERIES) == layout["review_collections"]
    for key, expected in layout.items():
        assert series_paths.series_folder_path(root, key) == expected
    training = dict(include_in_training=True, editing_locked=False)
    validation = dict(include_in_training=False, editing_locked=True)
    for name, settings in (("classification", training), ("validation", validation)):
        path = _write(layout["reviews"] / name / "review_session.json", settings)
        assert json.loads(path.read_text()) == settings
    model = _model(layout["phase_models"])
    index_model(root, model)
    assert SharedPhaseModelTrainingPage._models_folder_for_series(
        NS(_configured_series_roots={KEY: root}), KEY) == layout["phase_models"]
    for selected in (root / phase_root, layout["phase_models"], layout["phase_model_library"]):
        assert [record["root"] for record in records_for_location(selected)] == [str(model)]
    assert {path.name for path in (root / phase_root).iterdir()} == {"TrainingInput", "Models"}


def test_explicit_legacy_children_remain_authoritative_with_a_canonical_parent(isolated):
    root = make_series(isolated, {"phase_model": "PhaseModels",
        "reviews": "PhaseModel/Annotations", "phase_models": "PhaseModel/Model",
        "phase_model_library": "PhaseModel/Model/library"})
    assert series_paths.series_path(SERIES, "reviews") == root / "PhaseModel" / "Annotations"
    assert series_paths.model_versions_folder(root) == root / "PhaseModel" / "Model"


def _event(event_id: str, actor: str) -> dict:
    from darsia.fluidflower import phase_review_retractions as withdrawals

    return {"format": withdrawals.EVENT_FORMAT, "event_id": event_id, "action": "retract",
            "created_utc": "2026-01-01T00:00:00+00:00", "actor": actor,
            "target": {"series": KEY, "run": "r1", "kind": "manual", "identifier": f"box-{event_id}"}}


def _model(folder: Path, model_id: str = "phase-model-20260101T000000Z-00000000") -> Path:
    root = folder / model_id
    _write(root / "summary.json", {"model_id": model_id, "created_utc": "2026-01-01T00:00:00+00:00",
                                   "inventory": {"datasets": [KEY]}})
    for head in ("co2_presence", "gas_within_co2"):
        _write(root / "models" / "generic" / f"{head}_signal.json", {"head": head})
    return root


def test_sessions_collections_ledger_audits_library_and_models_follow_phase_model(isolated):
    from darsia.fluidflower import phase_review_retractions as withdrawals
    from darsia.fluidflower.phase_evidence_audit import queue_directory
    from darsia.fluidflower.phase_model_history import project_phase_model_roots
    from darsia.fluidflower.phase_model_library import (
        discover_series_sources, index_model, index_series, library_records, library_root,
        records_for_location, resolve_reference,
    )
    from darsia.fluidflower.phase_workflow import default_phase_review_roots
    from darsia.fluidflower.review_storage import series_review_root
    from darsia.fluidflower.tools.phase_review_collections import collection_root
    from darsia.fluidflower.ui.model_training import SharedPhaseModelTrainingPage

    root = make_series(isolated, {"phase_model": "PhaseModel"})
    layout = phase_model_layout(root)
    assert series_review_root(SERIES) == layout["reviews"]
    assert layout["reviews"] in default_phase_review_roots()
    assert collection_root(SERIES) == layout["review_collections"]

    assert withdrawals.ledger_folder(root) == layout["evidence_actions"] / "events"
    withdrawals._append_event(root, _event("e1", "Ada"))
    assert (layout["evidence_actions"] / "events" / "e1.json").is_file()
    assert [event["event_id"] for event in withdrawals.load_retractions(series=SERIES)["events"]] == ["e1"]

    model = _model(layout["phase_models"])
    assert queue_directory(model) == layout["evidence_audits"]

    pointer = index_model(root, model)
    assert pointer.parent == library_root(root) == layout["phase_model_library"]
    assert [record["root"] for record in library_records(root)] == [str(model)]
    assert [record["root"] for record in records_for_location(layout["phase_models"])] == [str(model)]
    assert resolve_reference(pointer) == model
    assert model in project_phase_model_roots()
    assert discover_series_sources(root) == [model]
    assert index_series(root)["library"] == str(layout["phase_model_library"])

    studio = NS(_configured_series_roots={KEY: root})
    assert SharedPhaseModelTrainingPage._models_folder_for_series(studio, KEY) == layout["phase_models"]
    unregistered = NS(_configured_series_roots={KEY: isolated / "copy"})
    (isolated / "copy").mkdir()
    (isolated / "copy" / "series.toml").write_bytes((root / "series.toml").read_bytes())
    assert SharedPhaseModelTrainingPage._models_folder_for_series(unregistered, KEY) == (
        isolated / "copy" / "PhaseModel" / "Model")


def test_new_model_versions_and_their_index_land_in_the_model_folder(isolated):
    from darsia.fluidflower.phase_model_library import index_registered_model
    from darsia.fluidflower.phase_workflow import prepare_phase_training_request
    from darsia.fluidflower.ui.model_training import SharedPhaseModelTrainingPage

    root = make_series(isolated, {"phase_model": "PhaseModel"})
    layout = phase_model_layout(root)
    decisions = _write(isolated / "inputs" / "decisions.json", {"decisions": {"r1": {"annotations": []}}})
    manifest = _write(isolated / "inputs" / "manifest.json", {"jobs": [{"run": "r1", "dataset": SERIES}]})
    output_parent = SharedPhaseModelTrainingPage._models_folder_for_series(
        NS(_configured_series_roots={KEY: root}), KEY)
    request = prepare_phase_training_request(
        output_parent=output_parent, cache_parent=isolated / "cache" / "training",
        bundles=[(decisions, manifest)], actor="Ada", settings={"series": [SERIES]})
    output = Path(request["output_root"])
    assert output.parent == layout["phase_models"]
    assert (output / "training_request.json").is_file()
    assert index_registered_model(output).parent == layout["phase_model_library"]


def test_historical_model_registration_survives_results_relocation(isolated):
    from darsia.fluidflower import phase_model_library as library, series_storage_migration

    root = make_series(isolated, {})
    model = _model(isolated / "separate saved models")
    source, target = root / "results", isolated / "Results"
    registration = _write(source / "phase_model_review_registrations" / "saved.json", {"model_root": str(model)})
    original = registration.read_bytes()
    assert library.registration_sources(source) == [model]
    plan = series_storage_migration.preview_results_migration(root, isolated, target)
    series_storage_migration.apply_results_migration(plan, writers_quiescent=True)
    assert library.registration_sources(source) == [model]
    assert model in library.discover_series_sources(root)
    result = library.index_series(root)
    assert not result["errors"]
    assert [record["root"] for record in library.library_records(root)] == [str(model)]
    assert (target / registration.relative_to(source)).read_bytes() == original


def test_without_phase_model_consumers_use_the_historical_folders(isolated, monkeypatch):
    from darsia.fluidflower import phase_review_retractions as withdrawals
    from darsia.fluidflower import phase_saved_promotions as promotions
    from darsia.fluidflower.phase_evidence_audit import queue_directory
    from darsia.fluidflower.phase_model_library import index_model, library_root
    from darsia.fluidflower.review_storage import series_review_root
    from darsia.fluidflower.series_convert import inspect_review_registration
    from darsia.fluidflower.tools.phase_review_collections import collection_root
    from darsia.fluidflower.ui.model_training import SharedPhaseModelTrainingPage

    root = make_series(isolated, {})
    assert series_review_root(SERIES) == root / "reviews"
    assert collection_root(SERIES) == root / "phase_review_collections"
    assert withdrawals.ledger_folder(root) == root / "reviews" / "evidence_actions" / "events"
    withdrawals._append_event(root, _event("e1", "Ada"))
    assert (root / "reviews" / "evidence_actions" / "events" / "e1.json").is_file()
    model = _model(root / "models")
    assert queue_directory(model) == root / "reviews" / "evidence_audits"
    assert library_root(root) == root / "models" / "phase"
    assert index_model(root, model).parent == root / "models" / "phase"
    studio = NS(_configured_series_roots={KEY: root})
    assert SharedPhaseModelTrainingPage._models_folder_for_series(studio, KEY) == root / "models"
    copy = isolated / "copy"
    copy.mkdir()
    (copy / "series.toml").write_text('[paths]\nmodels = "custom"\n', encoding="utf-8")
    studio = NS(_configured_series_roots={KEY: copy})
    assert SharedPhaseModelTrainingPage._models_folder_for_series(studio, KEY) == copy / "models"
    _write(root / "phase_review_collections" / "index.json",
           {"format": "fluidflower-phase-review-collection-v1", "collections": []})
    assert inspect_review_registration(root)["state"] != "not_registered"
    host = isolated / "host-cache"
    monkeypatch.setattr(promotions, "default_execution_cache_root", lambda: host)
    monkeypatch.setattr(promotions, "_literal", lambda key: "enrollments")
    assert promotions.promotion_root() == promotions.promotion_root(SERIES) == host.resolve() / "enrollments"


def test_enrollments_and_collection_index_follow_phase_model(isolated, monkeypatch):
    from darsia.fluidflower import phase_saved_promotions as promotions
    from darsia.fluidflower.series_convert import inspect_review_registration

    root = make_series(isolated, {"phase_model": "PhaseModel"})
    layout = phase_model_layout(root)
    monkeypatch.setattr(promotions, "default_execution_cache_root", lambda: isolated / "host-cache")
    monkeypatch.setattr(promotions, "_literal", lambda key: "enrollments")
    assert promotions.promotion_root(SERIES) == layout["training_enrollments"]
    assert promotions.promotion_root() == (isolated / "host-cache").resolve() / "enrollments"
    assert promotions._promotion_roots() == [promotions.promotion_root(), layout["training_enrollments"]]
    _write(layout["review_collections"] / "index.json",
           {"format": "fluidflower-phase-review-collection-v1", "collections": []})
    assert inspect_review_registration(root)["state"] != "not_registered"


def test_ledger_left_in_the_previous_folder_is_refused(isolated):
    from darsia.fluidflower import phase_review_retractions as withdrawals

    root = make_series(isolated, {})
    withdrawals._append_event(root, _event("e1", "Ada"))
    text = (root / "series.toml").read_text(encoding="utf-8")
    (root / "series.toml").write_text(text + 'phase_model = "PhaseModel"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="move the events or set \\[paths\\] evidence_actions"):
        withdrawals.load_retractions(series=SERIES)
    (root / "series.toml").write_text(
        text + 'phase_model = "PhaseModel"\nevidence_actions = "reviews/evidence_actions"\n', encoding="utf-8")
    assert [event["event_id"] for event in withdrawals.load_retractions(series=SERIES)["events"]] == ["e1"]


FRAME = {"frame_index": 2, "elapsed_h": 1.5, "source_name": "IMG_0002.JPG", "source_sha256": "AB" * 32}


def test_photo_identity_is_stamped_from_the_frame_record():
    from darsia.fluidflower.phase_annotation_binding import (
        PHOTO_IDENTITY_KEY, photo_identity, stamp_camera_identity,
    )

    payload = {"id": "box1", "frame_index": 2}
    stamped = stamp_camera_identity(payload, FRAME, "R1", dataset=SERIES)
    assert payload == {"id": "box1", "frame_index": 2}
    assert stamped[PHOTO_IDENTITY_KEY] == {"dataset": KEY, "run": "r1",
                                           "source_name": "IMG_0002.JPG", "source_sha256": "ab" * 32}
    assert {key: value for key, value in stamped.items() if key != PHOTO_IDENTITY_KEY} == payload
    assert photo_identity({"source_path": "C:\\images\\IMG_7.JPG"}, dataset=SERIES, run="r1") == {
        "dataset": KEY, "run": "r1", "source_name": "IMG_7.JPG", "source_sha256": None}
    # A frame record that does not name its photograph leaves the payload as it was.
    assert photo_identity({"elapsed_h": 1.0}, dataset=SERIES, run="r1") is None
    assert stamp_camera_identity(payload, {"elapsed_h": 1.0}, "r1", dataset=SERIES) == payload


def test_selected_camera_contract_keeps_its_original_fields():
    from darsia.fluidflower.phase_annotation_binding import CONTRACT, stamp_camera_identity

    stamped = stamp_camera_identity({"id": "box1"}, FRAME, "R1", dataset="ffac", selected_camera=True)
    assert stamped == {"id": "box1", "source_label_identity": {
        "dataset": "ffac", "run": "r1", "camera_source_sha256": "ab" * 32},
        "source_sha256": "ab" * 32, "source_binding_contract": CONTRACT}


@pytest.fixture
def review_module(tmp_path, monkeypatch):
    """The review-store module, importable without a registered default series."""
    name = "darsia.fluidflower.tools.calibration_pairwise_review"
    if name in sys.modules:
        yield sys.modules[name]
        return
    from darsia.fluidflower.tools import _paths
    # The module resolves one registered default series at import time.
    monkeypatch.setattr(_paths, "series_root", lambda _dataset: tmp_path / "import-default")
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)
    package = sys.modules["darsia.fluidflower.tools"]
    if getattr(package, "calibration_pairwise_review", None) is module:
        delattr(package, "calibration_pairwise_review")


def _review_pair(review, tmp_path: Path):
    native = tmp_path / "prepared" / "native_frames" / "r1" / "manifest.json"
    frames = [dict(FRAME, frame_index=index, elapsed_h=float(index),
                   source_name=f"IMG_{index:04d}.JPG", source_sha256=f"{index:064x}")
              for index in range(3)]
    _write(native, {"frame_count": 3, "frame_sources": frames})
    candidate = review.Candidate(
        run="r1", role="candidate", family="phase", variant="unit", candidate_id="c1", grid_order=1,
        params_sha256="p" * 64, normalized_mae=None, mass_plot=tmp_path / "none.png",
        contour_movie=tmp_path / "none.mp4", per_frame_csv=native, summary_path=tmp_path / "summary.json",
        plot_seconds=0.0, image_seconds=0.0, fps=8.0, frame_count=3, dataset=SERIES,
        review_output_dir=tmp_path / "prepared", phase_model_only=True)
    return review.ReviewPair(run="r1", candidate_a=candidate, candidate_b=candidate,
                             comparison_mode="single_candidate")


def _box(identifier: str, frame_index: int) -> dict:
    return {"id": identifier, "candidate": "A", "frame_index": frame_index, "category": "confirmed_free_gas",
            "shape": "rectangle", "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}


def test_review_store_stamps_new_boxes_and_keeps_legacy_boxes(review_module, tmp_path):
    from darsia.fluidflower.phase_annotation_binding import PHOTO_IDENTITY_KEY

    pair = _review_pair(review_module, tmp_path)
    session = tmp_path / "session"
    legacy = dict(_box("legacy", 0), reviewer="Ada", reviewer_id="ada")
    _write(session / "pairwise_review_decisions.json",
           {"schema": 2, "decisions": {"r1": {"run": "r1", "annotations": [legacy]}}})
    store = review_module.ReviewStore(session, [pair])
    record = store.save({"run": "r1", "reviewer": "Ada", "choice": "unresolved",
                         "annotations": [legacy, _box("new", 1)]})
    saved = {item["id"]: item for item in record["annotations"]}
    assert PHOTO_IDENTITY_KEY not in saved["legacy"]
    assert saved["new"][PHOTO_IDENTITY_KEY] == {"dataset": KEY, "run": "r1",
                                                "source_name": "IMG_0001.JPG", "source_sha256": f"{1:064x}"}
    # A stamped box moved to another photograph is stamped again; legacy stays untouched.
    moved = dict(saved["new"], frame_index=2)
    record = store.save({"run": "r1", "reviewer": "Ada", "choice": "unresolved",
                         "annotations": [saved["legacy"], moved]})
    saved = {item["id"]: item for item in record["annotations"]}
    assert saved["new"][PHOTO_IDENTITY_KEY]["source_name"] == "IMG_0002.JPG"
    assert PHOTO_IDENTITY_KEY not in saved["legacy"]


def test_legacy_annotation_without_identity_loads_for_training(tmp_path):
    from darsia.fluidflower.phase_training import trainer

    native = tmp_path / "prepared" / "native_frames" / "r1" / "manifest.json"
    _write(native, {"frame_count": 2, "frame_sources": [
        {"frame_index": 0, "elapsed_h": 0.0, "source_name": "IMG_0000.JPG"},
        {"frame_index": 1, "elapsed_h": 1.0, "source_name": "IMG_0001.JPG"}]})
    legacy = dict(_box("legacy", 1), phase_class="gas", training_eligible=True, confidence="high",
                  reviewer="Ada", elapsed_h=1.0)
    stamped = dict(_box("stamped", 0), phase_class="gas", training_eligible=True, confidence="high",
                   reviewer="Ada", elapsed_h=0.0, photo_identity={
                       "dataset": KEY, "run": "r1", "source_name": "IMG_0000.JPG", "source_sha256": None})
    decisions = _write(tmp_path / "decisions.json", {"decisions": {"r1": {"reviewer": "Ada",
                                                                           "annotations": [legacy, stamped]}}})
    manifest = _write(tmp_path / "manifest.json", {"jobs": [{"run": "r1", "dataset": KEY,
                                                             "review_output_dir": str(tmp_path / "prepared")}]})
    from darsia.fluidflower import phase_review_retractions as withdrawals

    before = hashlib.sha256(decisions.read_bytes()).hexdigest()
    _sources, regions, _conflicts = trainer._load_sources_and_regions(
        [trainer.ReviewBundle(decisions, manifest)], withdrawal_snapshot=withdrawals.empty_snapshot(KEY))
    assert sorted(region.region_id for region in regions) == ["r1:manual:legacy", "r1:manual:stamped"]
    assert "photo_identity" not in next(r for r in regions if r.region_id.endswith("legacy")).payload
    assert hashlib.sha256(decisions.read_bytes()).hexdigest() == before
