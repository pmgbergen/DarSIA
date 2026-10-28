"""The genuine preparation binding drives categorical and drawn-fixture replay."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import geometry_raster_sources as sources
from darsia.fluidflower import layer_painting_fixtures as fixtures
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower import map_placement as placement
from darsia.fluidflower import painted_adoption_table as table
from darsia.fluidflower import rig_point_alignment as alignment
from darsia.fluidflower import rig_point_review as review
from darsia.fluidflower.painting_frame_models import FrameModelSelection

from .test_layer_painting_fixtures import _version, FRAME
from .test_map_adoption import _native_index


@pytest.fixture
def case(tmp_path):
    record, underlay = _version(tmp_path / "versions")
    base_path = _native_index(tmp_path / "geometry")
    base = json.loads(base_path.read_text())
    del base["runs"]["run4"]
    runs = []
    for run, row in base["runs"].items():
        model_path = Path(row["reference_model"]["path"])
        model = json.loads(model_path.read_text())
        photo = tmp_path / f"{run}.ARW"
        photo.write_bytes(f"private {run} RAW".encode())
        model["reference_image"] = sources.descriptor(photo)
        model_path.write_text(json.dumps(model), encoding="utf-8")
        row["reference_model"] = sources.descriptor(model_path)
        runs.append(dict(run_id=run, path=str(photo), image_size=[60, 40]))
    base_path.write_text(json.dumps(base), encoding="utf-8")
    original_models = placement.read_run_geometry(base_path)
    frame = replace(FRAME, run_id="run1", image=runs[0]["path"], image_size=(60, 40))
    document = review.prepare_review(frame, runs, original_models)
    shifted = {name: [row, col+4.] for name, (row, col) in document["runs"]["run2"]["current_rc"].items()}
    document = review.update_run(document, "run2", shifted)
    for run in ("run1", "run2"):
        document = review.approve_run(document, run, viewed_identity=document["runs"][run]["photo"]["identity"],
                                      actor="Reviewer")
    project = SimpleNamespace(root=tmp_path / "project")
    selection = FrameModelSelection(str(base_path), "user override", str(project.root), "private", "private", "private")
    derived = alignment.capture_reviewed_selection(project, selection, document, frame, ["run1", "run2"])
    prepared = adoption.prepare_adoption(tmp_path / "adoption", underlay.analysis_layers,
        reference_run="run1", geometry_index=derived.index_path, runs=["run1", "run2"],
        roles={1: "porous_medium", 2: "porous_medium", 3: "colorchecker"},
        version=record, fixture_underlay=underlay, fixture_frame=FRAME)
    folder = Path(prepared["folder"])
    row = next(row for row in prepared["runs"] if row["run_id"] == "run2")
    with np.load(folder / row["labels"]["path"]) as saved:
        labels = saved["array"].copy()
    facies = tmp_path / "facies.csv"
    facies.write_text("id,name,porosity,permeability\n1,Upper,.4,10\n2,Lower,.3,20\n3,Fixture,.7,90\n")
    current = {"configuration": {"source_mode": sources.CANONICAL_EDIT,
        "canonical_import_parent": {"revision_id": "parent"}, "facies_properties_source_path": str(facies),
        "frame_binding": {"display_reference_run": "run1", "runs": {
            run: {"fields": {"shape_corrected_baseline": {}}} for run in ("run1", "run2")}}}}
    config, _ = adoption.adoption_candidate(current, folder, {1: 1, 2: 2, 3: 3})
    return SimpleNamespace(record=record, underlay=underlay, base=base_path, derived=derived,
        models=placement.read_run_geometry(derived.index_path), original_models=original_models,
        preparation=prepared, folder=folder, row=row, labels=labels, config=config, runs=runs)


def test_manual_point_change_moves_categorical_transport_and_inside_replay(case):
    original_labels = case.underlay.analysis_layers.copy()
    reference, target = (case.models[run].model for run in ("run1", "run2"))
    left, right, shape = sources.painted_map_composition(reference, target,
        geometry_index=case.preparation["geometry_index"])
    actual = sources.carry_label_reference(original_labels, reference_matrix=left.tolist(),
        target_matrix=right.tolist(), target_shape_hw=list(shape),
        reference_filled=np.zeros(original_labels.shape, bool), source_fill_status="KNOWN",
        reference_support=original_labels != 0)
    np.testing.assert_array_equal(actual["labels"], case.labels)
    legacy, _, _, _ = adoption.native_labels(original_labels,
        case.original_models["run1"], case.original_models["run2"])
    assert case.labels[10, 25] == 1 and legacy[10, 25] == 3
    assert not np.array_equal(case.labels, legacy)
    inside = np.zeros(original_labels.shape, bool)
    inside[12:22, 20:30] = True
    effective = table._native_inside(case.row, inside, case.preparation)
    expected, _, _, _ = adoption.native_labels(inside.astype(np.uint8), case.models["run1"], case.models["run2"])
    np.testing.assert_array_equal(effective["labels"], expected)
    old = table._native_inside(case.row, inside)
    assert not np.array_equal(old["labels"], expected)
    np.testing.assert_array_equal(case.underlay.analysis_layers, original_labels)


def test_table_and_adopted_fixture_replay_use_same_reviewed_transform(case):
    drawn_ids, _ = table._checked_native_underlay(case.folder, case.row, case.underlay,
                                                 case.labels, case.preparation)
    assert drawn_ids == case.row["fixture_underlay_layer_ids"]
    actual = fixtures.read_underlay(table._local(case.folder, case.row["fixture_underlay"]))
    result = fixtures.adopted_underlay(case.config, "run2", case.labels)
    np.testing.assert_array_equal(result["fixture_mask"], actual.fixture_mask)
    assert result["drawn_mask"].any()
    with pytest.raises(ValueError, match="frozen placement replay"):
        table._checked_native_underlay(case.folder, case.row, case.underlay, case.labels)


def test_registered_native_reader_uses_genuine_preparation_before_composition(case, tmp_path):
    labels_path = case.folder / case.row["labels"]["path"]
    photo = tmp_path / "corrected-baseline.npz"
    np.savez_compressed(photo, array=np.zeros((*case.labels.shape, 3), np.float32), metadata={})
    image = sources.descriptor(photo)
    registration = dict(kind=sources.PAINTED, parent_revision_id="private-parent", status="painted",
        authority=sources.descriptor(case.folder / "approval.json"),
        measurement=sources.descriptor(case.folder / "index.json"), labels=sources.descriptor(labels_path),
        source_image=image, source_map=sources.descriptor(case.folder / "source_map.npz"),
        reference_model=case.row["reference_model"], target_model=case.row["target_model"])
    row = dict(run_id="run2", labels_path=str(labels_path), source_image_path=str(photo),
               facies_path="", registration=registration)
    config = dict(frame_binding=dict(runs=dict(run2=dict(fields=dict(shape_corrected_baseline=image)))))
    payload = dict(labels=case.labels, actual=dict(labels=sources.descriptor(labels_path)))
    sources._validate_painted_run(config, payload, dict(revision_id="private-parent"), row)
    # A genuine but differently bound preparation cannot use the approved H.
    changed = deepcopy(case.preparation)
    changed.pop("geometry_index")
    wrong = tmp_path / "other-preparation.json"
    wrong.write_text(json.dumps(changed), encoding="utf-8")
    row["registration"]["measurement"] = sources.descriptor(wrong)
    with pytest.raises(ValueError, match="exact replay of their placement"):
        sources._validate_painted_run(config, payload, dict(revision_id="private-parent"), row)


def test_corrupt_or_stale_point_sidecar_is_rejected_by_native_and_fixture_replay(case):
    index = json.loads(Path(case.derived.index_path).read_text())
    path = Path(index[alignment.KEY]["path"])
    original = path.read_bytes()
    readers = [
        lambda: table._native_inside(case.row, case.underlay.fixture_mask, case.preparation),
        lambda: table._checked_native_underlay(case.folder, case.row, case.underlay, case.labels, case.preparation),
        lambda: fixtures.adopted_underlay(case.config, "run2", case.labels),
    ]
    path.write_bytes(original + b" ")
    for reader in readers:
        with pytest.raises(ValueError, match="metadata changed"):
            reader()
    path.write_bytes(original)
    photo = Path(case.runs[1]["path"])
    photo.write_bytes(photo.read_bytes() + b"changed")
    for reader in readers:
        with pytest.raises(ValueError, match="photograph changed"):
            reader()
