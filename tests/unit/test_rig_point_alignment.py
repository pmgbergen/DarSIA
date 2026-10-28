"""Approved point corrections alter transport without changing measured evidence."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import map_placement as placement
from darsia.fluidflower import photo_edge_continuation as policy
from darsia.fluidflower import rig_point_alignment as alignment
from darsia.fluidflower import rig_point_review as review
from darsia.fluidflower.painting_frame_models import FrameModelSelection
from darsia.fluidflower.rig_reference_quadrilateral import reference_frame_geometry
from darsia.fluidflower.series_geometry import RigFrame

from .test_rig_point_review import context, prepare, approve  # noqa: F401
from .test_painted_review_comparison import source as display_source  # noqa: F401


def descriptor(path):
    raw = Path(path).read_bytes()
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def selection(project, path):
    return FrameModelSelection(str(path), "user override", str(project.root), "private", "private-series", "private")


@pytest.fixture
def case(context, tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    entries, original = {}, {}
    for run in ("source", "target"):
        path = folder / f"{run}.json"
        photo = next(item["path"] for item in context.runs if item["run_id"] == run)
        model = dict(format=placement.GEOMETRY_FORMAT, run_id=run, shape_hw=[700, 1000],
            reference_image=descriptor(photo), H_metric_mm_to_camera_u=context.models[run].metric_to_grid.tolist())
        path.write_text(json.dumps(model), encoding="utf-8")
        original[run] = path.read_bytes()
        entries[run] = dict(reference_model=dict(descriptor(path), path=path.name))
    index = folder / "index.json"
    base = dict(format=placement.GEOMETRY_FORMAT, runs=entries, run_order=["source", "target"],
        run_count=2, reference_models_complete=True, pending_runs=[],
        excluded_runs={"manual": "Outside selected adoption cohort"})
    index.write_text(json.dumps(base), encoding="utf-8")
    document = prepare(context)
    points = {name: [row+3., col+9.] for name, (row, col) in document["runs"]["target"]["current_rc"].items()}
    document = review.update_run(document, "target", points)
    document = approve(approve(document, "source"), "target")
    document["required_runs"] = ["source", "target"]
    project = SimpleNamespace(root=tmp_path / "project")
    project.root.mkdir()
    return SimpleNamespace(context=context, project=project, index=index, base=base, original=original,
                           document=document, selection=selection(project, index))


def capture(case):
    return alignment.capture_reviewed_selection(case.project, case.selection, case.document,
        case.context.frame, [row["run_id"] for row in case.context.runs])


def test_derived_index_keeps_cohort_and_original_evidence_but_uses_approved_matrices(case):
    original_index = case.index.read_bytes()
    derived = capture(case)
    again = capture(case)
    assert derived == again
    document = json.loads(Path(derived.index_path).read_text())
    for name in ("run_order", "run_count", "excluded_runs", "pending_runs", "reference_models_complete"):
        assert document[name] == case.base[name]
    assert set(document["runs"]) == {"source", "target"}
    models = placement.read_run_geometry(derived.index_path)
    for run, model in models.items():
        np.testing.assert_array_equal(model.metric_to_grid, review.metric_to_grid(case.document, run))
        assert model.entry[alignment.ENTRY_KEY] == document[alignment.KEY]
        assert Path(model.source).read_bytes() == case.original[run]
        assert model.model == json.loads(case.original[run])
        assert Path(document["runs"][run]["reference_model"]["path"]).is_absolute()
    assert not np.allclose(models["target"].metric_to_grid, case.context.models["target"].metric_to_grid)
    assert case.index.read_bytes() == original_index
    assert derived.sha256 == descriptor(derived.index_path)["sha256"]
    assert derived.size_bytes == descriptor(derived.index_path)["size_bytes"]
    assert "reference_point_reviews" in Path(derived.index_path).parts


def test_pending_selected_run_blocks_but_explicitly_excluded_registered_run_does_not(case):
    case.document["runs"]["target"]["status"] = "pending"
    with pytest.raises(ValueError, match="approve"):
        capture(case)


def test_changed_base_index_or_bound_photograph_invalidates_effective_alignment(case):
    derived = capture(case)
    doc = json.loads(Path(derived.index_path).read_text())
    case.index.write_bytes(case.index.read_bytes() + b" ")
    with pytest.raises(ValueError, match="metadata changed"):
        placement.read_run_geometry(derived.index_path)
    case.index.write_text(json.dumps(case.base), encoding="utf-8")
    photo = Path(case.context.runs[1]["path"])
    photo.write_bytes(photo.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="photograph changed"):
        alignment.effective_matrix(doc[alignment.KEY], "target")


def test_continuation_closure_uses_reviewed_transform_and_refuses_foreign_sidecar(case, monkeypatch):
    derived = capture(case)
    models = placement.read_run_geometry(derived.index_path)
    source, target = (policy._model(models[run]) for run in ("source", "target"))
    assert "point_review" in source and "point_review" in target
    matrix, shape = policy._checked_model(target, verify_files=True)
    np.testing.assert_array_equal(matrix, models["target"].metric_to_grid)
    assert shape == (700, 1000)
    alignment.validate_contract_binding(descriptor(derived.index_path), source, target)
    frame = review.reviewed_frame(case.document, "target")
    monkeypatch.setattr(policy, "_own_frame", lambda authority, run: frame)
    contract = dict(format=policy.FORMAT, policy=policy.POLICY, target_grid="camera_u",
        source_run="source", target_run="target", source_model=source, target_model=target,
        geometry_index=descriptor(derived.index_path), own_frame=frame.to_dict(),
        frame_sha256=policy._json_sha(frame.to_dict()), frame_authority={}, known_ids=[1, 2],
        source_fill="no-inherited-photo-fill", adopted=False)
    checked = policy.validate_contract(contract, verify_files=True)
    np.testing.assert_array_equal(checked[1], models["target"].metric_to_grid)
    bad = deepcopy(contract)
    del bad["source_model"]["point_review"]
    with pytest.raises(ValueError, match="captured geometry index"):
        policy.validate_contract(bad, verify_files=True)
    second = deepcopy(case.document)
    point = second["runs"]["target"]["current_rc"]["top_left"]
    second = approve(review.set_point(second, "target", "top_left", point[0]+1., point[1]), "target")
    other = alignment.capture_reviewed_selection(case.project, case.selection, second, case.context.frame,
                                                ["source", "target", "manual"])
    other_binding = json.loads(Path(other.index_path).read_text())[alignment.KEY]
    bad["source_model"]["point_review"] = other_binding
    bad["target_model"]["point_review"] = other_binding
    with pytest.raises(ValueError, match="captured geometry index"):
        policy.validate_contract(bad, verify_files=False)


def test_legacy_index_and_model_closure_are_unchanged_without_review(case):
    models = placement.read_run_geometry(case.index)
    for run, model in models.items():
        np.testing.assert_array_equal(model.metric_to_grid, case.context.models[run].metric_to_grid)
        assert alignment.ENTRY_KEY not in model.entry
        closed = policy._model(model)
        assert "point_review" not in closed
        matrix, _ = policy._checked_model(closed, verify_files=True)
        np.testing.assert_array_equal(matrix, case.context.models[run].metric_to_grid)


def test_effective_model_matrix_replays_only_the_exact_index_bound_model(case):
    model = json.loads(case.original["target"])
    expected_original = np.array(model["H_metric_mm_to_camera_u"])
    np.testing.assert_array_equal(alignment.effective_model_matrix(model), expected_original)
    np.testing.assert_array_equal(alignment.effective_model_matrix(model, descriptor(case.index)), expected_original)
    inverted = deepcopy(model)
    inverted["H_camera_u_to_metric_mm"] = np.linalg.inv(inverted.pop("H_metric_mm_to_camera_u")).tolist()
    np.testing.assert_allclose(alignment.effective_model_matrix(inverted), expected_original, atol=1e-10)
    derived = capture(case)
    np.testing.assert_array_equal(alignment.effective_model_matrix(model, descriptor(derived.index_path)),
                                  review.metric_to_grid(case.document, "target"))
    changed = deepcopy(model)
    changed["H_metric_mm_to_camera_u"][0][2] += 1.
    with pytest.raises(ValueError, match="differs from its reviewed index"):
        alignment.effective_model_matrix(changed, descriptor(derived.index_path))


def test_painted_comparison_uses_reviewed_matrix_and_preserves_raw_native_certificate(display_source, tmp_path):
    from darsia.fluidflower import painted_review_comparison as display
    source = display_source
    for text, (_bound, document) in source.documents.items():
        path = Path(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(document, sort_keys=True).encode())
    runs, entries = [], {}
    for row in source.review["runs"]:
        run = row["run_id"]
        photo = Path(source.models[run]["reference_image"]["path"])
        photo.parent.mkdir(parents=True, exist_ok=True)
        photo.write_bytes(run.encode())
        runs.append(dict(run_id=run, path=str(photo), image_size=[40, 30]))
        entries[run] = dict(reference_model=deepcopy(row["target_model"]))
    index = tmp_path / "display-index.json"
    index.write_text(json.dumps(dict(format=placement.GEOMETRY_FORMAT, runs=entries)), encoding="utf-8")
    points = dict(top_left=[5., 5.], top_right=[5., 35.], bottom_right=[25., 35.], bottom_left=[25., 5.])
    frame = RigFrame(**reference_frame_geometry(points, measurements_m=dict(width=.030, height=.020)), run_id="ac17")
    models = placement.read_run_geometry(index)
    document = review.prepare_review(frame, runs, models)
    target_points = {name: [row+1., col+2.] for name, (row, col) in document["runs"]["ac18"]["current_rc"].items()}
    document = review.update_run(document, "ac18", target_points)
    document = approve(approve(document, "ac17"), "ac18")
    project = SimpleNamespace(root=tmp_path / "display-project")
    derived = alignment.capture_reviewed_selection(project, selection(project, index), document, frame, ["ac17", "ac18"])
    effective = placement.read_run_geometry(derived.index_path)
    row = deepcopy(source.review["runs"][1])
    row["contract"] = dict(geometry_index=descriptor(derived.index_path),
        source_model=policy._model(effective["ac17"]), target_model=policy._model(effective["ac18"]))
    geometry, native, certificate = display._model(row)
    np.testing.assert_array_equal(geometry.metric_to_grid, review.metric_to_grid(document, "ac18"))
    assert geometry.model == source.models["ac18"]
    assert certificate == source.certificates["ac18"]
    np.testing.assert_array_equal(native, source.models["ac18"]["baseline_native_grid"]["H_baseline_camera_u_to_rig_corrected_native"])
