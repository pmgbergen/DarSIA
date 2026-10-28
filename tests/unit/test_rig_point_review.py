"""Point review is explicit, immutable in origin and bound to the viewed file."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import rig_point_review as review
from darsia.fluidflower.rig_reference_quadrilateral import reference_frame_geometry
from darsia.fluidflower.series_geometry import RigFrame


PHOTO = dict(top_left=[10., 20.], top_right=[12., 920.],
             bottom_right=[502., 918.], bottom_left=[500., 18.])


@pytest.fixture
def context(tmp_path):
    runs = []
    for run_id in ("source", "target", "manual"):
        path = tmp_path / f"{run_id}.ARW"
        path.write_bytes(f"private {run_id} photo".encode())
        runs.append(dict(run_id=run_id, path=str(path), image_size=[1000, 700]))
    frame = RigFrame(**reference_frame_geometry(PHOTO, measurements_m=dict(width=.899, height=.490)),
                     run_id="source", image=runs[0]["path"], image_size=(1000, 700))
    frame.reference_geometry.update(estimated=True, uncertainty_mm=5., point_descriptions={
        "top_left": "top_of_free_water", "top_right": "top_of_free_water",
        "bottom_right": "rig_frame", "bottom_left": "rig_frame"})
    source_matrix = np.array([[1., .02, 4.], [.01, -1.1, 800.], [.00003, .00001, 1.]])
    composition = np.array([[1.04, .015, 13.], [-.01, .98, -7.], [.00001, -.00002, 1.]])
    models = dict(source=SimpleNamespace(metric_to_grid=source_matrix),
                  target=SimpleNamespace(metric_to_grid=composition @ source_matrix))
    return SimpleNamespace(frame=frame, runs=runs, models=models, composition=composition)


def prepare(context, **kwargs):
    return review.prepare_review(context.frame, context.runs, kwargs.get("models", context.models))


def test_previously_approved_source_can_be_corrected_and_approved_again(context):
    document=review.approve_run(prepare(context),'source',
        viewed_identity=review.photo_binding(context.runs[0]['path'],context.runs[0]['image_size'])['identity'],
        actor='Reviewer')
    saved=review.reviewed_frame(document,'source')
    changed=review.set_point(document,'source','top_left',11.,20.)
    assert review.reference_matches(changed,saved)
    assert review.review_blocker(changed,saved,['source'],bindings(context))
    changed=review.approve_run(changed,'source',viewed_identity=changed['runs']['source']['photo']['identity'],actor='Reviewer')
    assert review.review_blocker(changed,review.reviewed_frame(changed,'source'),['source'],bindings(context)) is None


def approve(document, run_id):
    return review.approve_run(document, run_id, viewed_identity=document["runs"][run_id]["photo"]["identity"],
                              actor="Reviewer")


def bindings(context):
    return {run["run_id"]: review.photo_binding(run["path"], run["image_size"]) for run in context.runs}


def project_points(points, matrix):
    source = np.array([points[name][::-1] for name in review.POINT_NAMES])
    moved = np.column_stack((source, np.ones(len(source)))) @ matrix.T
    return moved[:, :2] / moved[:, 2, None]


def test_prepare_uses_exact_reference_and_model_composition_without_approval(context):
    before = deepcopy(context.frame.to_dict())
    document = prepare(context)
    assert document["reference_identity"] == review.reference_signature(context.frame)
    assert document["reference"]["frame"] == before
    assert document["runs"]["source"]["proposed_rc"] == PHOTO
    target = document["runs"]["target"]
    expected = project_points(PHOTO, context.composition)
    np.testing.assert_allclose([target["proposed_rc"][name][::-1] for name in review.POINT_NAMES], expected, atol=1e-10)
    assert all(row["status"] == "pending" and row["approval"] is None for row in document["runs"].values())
    assert target["current_rc"] == target["proposed_rc"] and target["current_rc"] is not target["proposed_rc"]
    assert target["current_rc"]["top_left"] is not target["proposed_rc"]["top_left"]
    assert context.frame.to_dict() == before
    json.dumps(document, allow_nan=False)
    assert "approve" in review.review_blocker(document, context.frame, ["source"], bindings(context))


def test_missing_or_invalid_models_require_manual_placement_without_invented_points(context):
    context.models["target"] = SimpleNamespace(metric_to_grid=np.zeros((3, 3)))
    document = prepare(context)
    for run_id in ("target", "manual"):
        row = document["runs"][run_id]
        assert row["proposed_rc"] is None and row["current_rc"] is None
        assert row["proposal_binding"]["method"] == "manual_required"
    original = deepcopy(document)
    for name, (row, col) in PHOTO.items():
        document = review.set_point(document, "manual", name, row, col)
    assert original["runs"]["manual"]["current_rc"] is None
    assert document["runs"]["manual"]["proposed_rc"] is None
    assert document["runs"]["manual"]["current_rc"] == PHOTO
    assert document["runs"]["manual"]["status"] == "corrected"
    document = approve(document, "manual")
    assert review.review_blocker(document, context.frame, ["manual"], bindings(context)) is None


def test_corrected_approved_points_drive_frame_and_keep_immutable_before_after(context):
    document = prepare(context)
    proposal = deepcopy(document["runs"]["target"]["proposed_rc"])
    corrected = {name: [row + 8., col - 3.] for name, (row, col) in proposal.items()}
    pending = review.update_run(document, "target", corrected)
    assert document["runs"]["target"]["current_rc"] == proposal
    row = pending["runs"]["target"]
    assert row["proposed_rc"] == row["before_rc"] == proposal
    assert row["current_rc"] == row["after_rc"] == corrected
    assert row["changes"][-1]["before_rc"] == proposal
    approved = approve(pending, "target")
    assert review.review_blocker(approved, context.frame, ["target"], bindings(context)) is None
    frame = review.reviewed_frame(approved, "target")
    assert frame.reference_geometry["photo_corners_rc"] == corrected
    assert frame.reference_geometry["estimated"] and frame.reference_geometry["uncertainty_mm"] == 5.
    assert frame.reference_geometry["point_descriptions"] == context.frame.reference_geometry["point_descriptions"]
    assert frame.width_m == .899 and frame.height_m == .490
    assert frame.run_id == "target" and frame.image == bindings(context)["target"]["path"]
    assert frame.image_size == (1000, 700)
    assert approved["runs"]["target"]["proposed_rc"] == proposal
    updated = review.set_point(approved, "target", "top_left", *np.array(corrected["top_left"]) + [1., 1.])
    assert updated["runs"]["target"]["status"] == "corrected" and updated["runs"]["target"]["approval"] is None
    assert "approve" in review.review_blocker(updated, context.frame, ["target"], bindings(context))


@pytest.mark.parametrize("points", [
    {**PHOTO, "top_left": [float("nan"), 0.]},
    {**PHOTO, "top_right": PHOTO["bottom_left"]},
    dict(top_left=[0., 0.], top_right=[1., 1.], bottom_right=[2., 2.], bottom_left=[3., 3.]),
    {name: point for name, point in PHOTO.items() if name != "bottom_left"},
])
def test_invalid_incomplete_crossed_or_degenerate_point_sets_are_rejected(context, points):
    document = prepare(context)
    with pytest.raises(ValueError):
        review.update_run(document, "target", points)


def test_approval_requires_viewed_photo_and_current_path_stamp(context):
    document = prepare(context)
    with pytest.raises(ValueError, match="Load and view"):
        review.approve_run(document, "source", viewed_identity="not-viewed", actor="Reviewer")
    with pytest.raises(ValueError, match="Name who approves"):
        review.approve_run(document, "source", viewed_identity="not-viewed", actor=" ")
    path = context.runs[0]["path"]
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="changed after loading"):
        approve(document, "source")
    fresh = prepare(context)
    fresh = approve(fresh, "source")
    with open(path, "ab") as stream:
        stream.write(b" changed")
    assert "photograph changed" in review.review_blocker(fresh, context.frame, ["source"], bindings(context))
    with pytest.raises(ValueError, match="photograph changed"):
        review.reviewed_frame(fresh, "source")


@pytest.mark.parametrize("change", ["description", "dimension", "reference_point", "proposal", "correction", "snapshot"])
def test_changed_reference_or_review_data_invalidates_approval(context, change):
    document = approve(prepare(context), "source")
    frame = deepcopy(context.frame)
    if change == "description":
        frame.reference_geometry["point_descriptions"]["top_left"] = "rig_frame"
    elif change == "dimension":
        frame.height_m += .001
    elif change == "reference_point":
        frame.reference_geometry["photo_corners_rc"]["top_left"][0] += 1.
    elif change == "proposal":
        document["runs"]["source"]["proposed_rc"]["top_left"][0] += 1.
    elif change == "correction":
        document["runs"]["source"]["current_rc"]["top_left"][0] += 1.
    else:
        document["reference"]["measurements_m"]["top"] += .001
    assert review.review_blocker(document, frame, ["source"], bindings(context)) is not None


def test_persistence_preserves_other_metadata_and_uses_existing_writer_with_actor(context, tmp_path):
    # The actor of the save is the decider the linked series names in its configuration.
    (tmp_path / "series.toml").write_text(
        'schema_version = 1\n[decisions]\ndecider = "Test decider"\n', encoding="utf-8")

    class Project:
        def __init__(self):
            self.manifest = dict(tools=dict(other_tool={"value": 7}, native_series=dict(
                reference_run="source", keep=[1, 2], series_root=str(tmp_path))))
            self.calls = []

        def update_metadata(self, section, values, **kwargs):
            self.calls.append((section, deepcopy(values), kwargs))
            self.manifest[section].update(deepcopy(values))

    project = Project()
    assert review.load_review(project) is None
    document = approve(prepare(context), "source")
    review.save_review(project, document)
    loaded = review.load_review(project)
    assert loaded == document and loaded is not document
    loaded["runs"]["source"]["current_rc"]["top_left"][0] += 1.
    assert review.load_review(project) == document
    assert project.manifest["tools"]["other_tool"] == {"value": 7}
    assert project.manifest["tools"]["native_series"]["keep"] == [1, 2]
    assert project.calls[0][0] == "tools" and project.calls[0][2]["updated_by"] == "Test decider"


@pytest.mark.parametrize("malformed", [None, {}, {"format": review.FORMAT}, [], "bad"])
def test_malformed_reviews_produce_status_reason_instead_of_raising(context, malformed):
    assert isinstance(review.review_blocker(malformed, context.frame, ["source"], bindings(context)), str)


def test_missing_required_run_and_photo_binding_block_transfer(context):
    document = approve(prepare(context), "source")
    assert "prepare and review" in review.review_blocker(document, context.frame, ["unknown"], bindings(context))
    assert "binding is missing" in review.review_blocker(document, context.frame, ["source"], {})


def test_exact_approved_source_frame_can_be_saved_and_reused_but_later_edits_cannot(context):
    document = prepare(context)
    adjusted = {name: [row+3., col+2.] for name, (row, col) in PHOTO.items()}
    document = approve(review.update_run(document, "source", adjusted), "source")
    replacement = review.reviewed_frame(document, "source")
    assert replacement.to_dict() != context.frame.to_dict()
    assert review.reference_matches(document, context.frame)
    assert review.reference_matches(document, replacement)
    assert review.reference_matches(document, RigFrame.from_dict(replacement.to_dict()))
    assert review.review_blocker(document, replacement, ["source"], bindings(context)) is None
    for field in ("source", "width", "point", "description"):
        edited = deepcopy(replacement)
        if field == "source":
            edited.source = "marked in geometry workspace"
        elif field == "width":
            edited.width_m += .001
        elif field == "point":
            edited.reference_geometry["photo_corners_rc"]["top_left"][0] += 1.
        else:
            edited.reference_geometry["point_descriptions"]["top_left"] = "rig_frame"
        assert not review.reference_matches(document, edited)
        assert review.review_blocker(document, edited, ["source"], bindings(context)) is not None


@pytest.mark.parametrize("quadrilateral", [False, True])
def test_approved_metric_transform_uses_same_physical_coordinates_for_source_and_target(context, quadrilateral):
    if quadrilateral:
        metric = np.array([[-.08, .46], [.84, .54], [.90, 0.], [0., 0.]])
        sides = np.linalg.norm(np.roll(metric, -1, axis=0) - metric, axis=1)
        measures = dict(zip(("top", "right", "bottom", "left"), sides))
        measures["diagonal_ac"] = float(np.linalg.norm(metric[0] - metric[2]))
        context.frame = RigFrame(**reference_frame_geometry(PHOTO, shape="quadrilateral", measurements_m=measures), run_id="source")
    document = approve(approve(prepare(context), "source"), "target")
    source_h = review.metric_to_grid(document, "source")
    target_h = review.metric_to_grid(document, "target")
    before = deepcopy(document)
    for run_id, matrix in (("source", source_h), ("target", target_h)):
        frame = review.reviewed_frame(document, run_id)
        metric = frame.reference_geometry["metric_corners_xy"]
        physical = np.array([metric[name] for name in review.POINT_NAMES]) * 1000.
        mapped = np.column_stack((physical, np.ones(4))) @ matrix.T
        np.testing.assert_allclose(mapped[:, :2] / mapped[:, 2, None],
            [document["runs"][run_id]["current_rc"][name][::-1] for name in review.POINT_NAMES], atol=1e-8)
    relative = target_h @ np.linalg.inv(source_h)
    np.testing.assert_allclose(project_points(PHOTO, relative), project_points(PHOTO, context.composition), atol=1e-8)
    assert document == before
    with pytest.raises(ValueError, match="approve"):
        review.metric_to_grid(document, "manual")
