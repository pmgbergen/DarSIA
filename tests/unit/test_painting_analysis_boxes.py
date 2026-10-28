"""Pure measured-box drafts; no GUI, image, file save or scientific work."""

from copy import deepcopy

import numpy as np
import pytest

from darsia.fluidflower import metric_analysis_domains as domains
from darsia.fluidflower.painting_analysis_boxes import (
    AnalysisBoxesDraft, NonRectangularBox, metric_to_raw, raw_to_metric, raw_point_to_metric,
)


def configuration():
    return dict(format=domains.FORMAT, units="mm", x_axis="right", y_axis="up",
        origin="inner left-wall and bottom intersection", geometry_revision_id="saved-geometry",
        geometry_revision=dict(path="unchanged.json", sha256="a" * 64, size_bytes=123),
        geometry_index=dict(path="models.json", sha256="b" * 64, size_bytes=456),
        image_correction_revision_id="saved-correction", run_order=["run-one", "run-two"],
        viewport_mm=[0., 900., 0., 490.], domains=[domains.box("Lower", [1.234567890123456, 400., 20., 200.],
            domain_id="lower", role="box1"), domains.box("Upper", [10., 450., 180., 420.],
            domain_id="upper", role="box2")])


def test_metadata_and_unchanged_canvas_edits_preserve_float_values_and_revision_identity():
    original = configuration()
    before = deepcopy(original)
    draft = AnalysisBoxesDraft(original)
    matrix = [[2., .1, 40.], [.05, -2., 1000.], [.0001, .0002, 1.]]
    projections = draft.canvas_outlines(matrix, first_index=3)
    assert draft.apply_canvas_edits({row.canvas_index: row.points_rc for row in projections}, projections) == ()
    draft.update_metadata("lower", name="Renamed", enabled=False)
    changed = draft.configuration()
    assert changed["domains"][0]["bounds_mm"][0].hex() == original["domains"][0]["bounds_mm"][0].hex()
    assert changed["geometry_revision"] == original["geometry_revision"]
    assert changed["geometry_index"] == original["geometry_index"]
    assert original == before
    changed["geometry_revision"]["sha256"] = "0" * 64
    assert draft.configuration()["geometry_revision"] == before["geometry_revision"]


def test_per_run_projective_mapping_uses_xy_mm_and_raw_row_column_without_nominal_scaling():
    points = ((10., 20.), (30., 20.), (30., 40.), (10., 40.))
    first = [[2., 0., 100.], [0., -3., 500.], [0., 0., 1.]]
    second = [[1.5, .2, -12.], [.1, -2.1, 700.], [.0001, .0002, 1.]]
    assert metric_to_raw(points, first)[0] == (440., 120.)
    for matrix in (first, second):
        projected = metric_to_raw(points, matrix)
        np.testing.assert_allclose(raw_to_metric(projected, matrix), points, rtol=0, atol=1e-12)
    assert metric_to_raw(points, first) != metric_to_raw(points, second)


def test_layer_guides_are_ignored_and_box_edits_do_not_mutate_supplied_changes():
    draft = AnalysisBoxesDraft(configuration())
    before = deepcopy(configuration())
    projections = draft.canvas_outlines(np.eye(3), first_index=2)
    changed = {0: [(0., 0.), (0., 100.), (100., 100.)],
               2: [(30., 2.), (30., 300.), (150., 300.), (150., 2.)]}
    snapshot = deepcopy(changed)
    assert draft.apply_canvas_edits(changed, projections) == ("lower",)
    assert changed == snapshot
    assert draft.configuration()["domains"][0]["bounds_mm"] == [2., 300., 30., 150.]
    assert draft.configuration()["domains"][1] == before["domains"][1]


def test_irregular_followed_outline_is_refused_without_replacing_the_rectangle_by_its_bounds():
    draft = AnalysisBoxesDraft(configuration())
    before = draft.configuration()
    token = draft.canvas_outlines(np.eye(3))[0]
    irregular = [(20., 1.), (20., 400.), (120., 350.), (200., 400.), (200., 1.)]
    with pytest.raises(NonRectangularBox, match="must stay rectangles"):
        draft.apply_canvas_edits({0: irregular}, (token,))
    with pytest.raises(NonRectangularBox, match="must stay rectangles"):
        draft.add_outline("Curved", tuple((col, row) for row, col in irregular))
    assert draft.configuration() == before
    clone = draft.clone()
    draft.update_metadata("lower", name="Renamed")
    assert clone.configuration() == before


def adjoining():
    config = configuration()
    config["domains"] = [domains.box("Left", [0., 100., 0., 100.], domain_id="left", role="box1"),
        domains.box("Right", [100., 200., 0., 100.], domain_id="right", role="box2"),
        domains.box("Point contact", [100., 200., 100., 200.], domain_id="point")]
    return AnalysisBoxesDraft(config)


def test_moving_a_shared_side_is_one_atomic_change_without_moving_a_point_contact():
    draft = adjoining()
    assert draft.shared_side_group("left", "x_max") == (("left", "x_max"), ("right", "x_min"))
    undo = draft.clone()
    assert draft.move_side("left", "x_max", 110.) == ("left", "right")
    values = {row["domain_id"]: row["bounds_mm"] for row in draft.configuration()["domains"]}
    assert values == dict(left=[0., 110., 0., 100.], right=[110., 200., 0., 100.],
                          point=[100., 200., 100., 200.])
    assert undo.configuration()["domains"][0]["bounds_mm"] == [0., 100., 0., 100.]


def test_shared_side_toggle_and_invalid_neighbour_width_do_not_partially_change_boxes():
    draft = adjoining()
    before = draft.configuration()
    with pytest.raises(ValueError, match="finite increasing"):
        draft.move_side("left", "x_max", 250.)
    assert draft.configuration() == before
    assert draft.move_side("left", "x_max", 110., move_shared=False) == ("left",)
    assert draft.configuration()["domains"][1] == before["domains"][1]


def test_conflicting_shared_side_requests_are_refused_atomically():
    draft = adjoining()
    before = draft.configuration()
    with pytest.raises(ValueError, match="conflicting positions"):
        draft.edit_sides({("left", "x_max"): 110., ("right", "x_min"): 120.})
    assert draft.configuration() == before


def test_snap_to_box_side_and_layer_edge_point_keeps_rectangles_and_never_edits_a_layer():
    draft = adjoining()
    layer_point = (125.12345678901234, 87.65432109876543)
    before_point = tuple(layer_point)
    draft.snap_side_to_layer_point("left", "x_max", layer_point)
    assert layer_point == before_point
    values = draft.configuration()["domains"]
    assert values[0]["bounds_mm"][1].hex() == layer_point[0].hex()
    assert values[1]["bounds_mm"][0].hex() == layer_point[0].hex()
    draft.move_side("left", "x_max", 120., move_shared=False)
    draft.snap_side_to_box("left", "x_max", "right", "x_min", move_shared=False)
    assert draft.configuration()["domains"][0]["bounds_mm"][1] == layer_point[0]
    matrix = [[2., 0., 100.], [0., -3., 500.], [0., 0., 1.]]
    assert raw_point_to_metric((440., 120.), matrix) == (10., 20.)
    with pytest.raises(ValueError, match="same x or y axis"):
        draft.snap_side_to_box("left", "x_max", "right", "y_min")


def test_whole_box_translation_and_side_no_op_keep_an_independent_undo_snapshot():
    draft = adjoining()
    before = draft.clone()
    assert draft.move_side("left", "x_max", 100.) == ()
    assert draft.move_box("left", 2., 3.) == ("left",)
    assert draft.configuration()["domains"][0]["bounds_mm"] == [2., 102., 3., 103.]
    assert draft.configuration()["domains"][1] == before.configuration()["domains"][1]


def test_overlapping_nested_boxes_are_independent_and_not_a_partition():
    draft = AnalysisBoxesDraft(configuration())
    nested = draft.add_rectangle("Nested", [20., 30., 30., 40.], domain_id="nested")
    overlap = draft.add_outline("Overlap", [(25., 35.), (500., 35.), (500., 300.), (25., 300.)], domain_id="overlap")
    assert len(draft.configuration()["domains"]) == 4
    draft.remove(nested)
    assert [row.domain_id for row in draft.outlines()] == ["lower", "upper", overlap]
    assert set(draft.mass_boxes()) == {"box1", "box2"}


def test_auto_import_only_adds_missing_legacy_roles_without_replacing_saved_or_disabled_rows():
    config = configuration()
    config["domains"] = [dict(config["domains"][0], enabled=False),
                         domains.box("box2", [1., 2., 1., 2.], domain_id="unassigned")]
    draft = AnalysisBoxesDraft(config)
    values = dict(box1=[[.4, .2], [.1, .01]], box2=[[.35000000000000003, .45], [.2, .25]])
    source = dict(path="legacy.toml", sha256="c" * 64, size_bytes=789)
    snapshot = deepcopy(values)
    added = draft.import_missing_legacy(values, source=source)
    assert len(added) == 1
    saved = draft.configuration()
    assert saved["domains"][0] == config["domains"][0]
    imported = saved["domains"][-1]
    assert imported["role"] == "box2" and imported["name"] != "box2"
    assert imported["legacy_import"]["original_value"] == values["box2"]
    assert imported["legacy_import"]["decision"] == "pending"
    assert draft.import_missing_legacy(None, source={}) == ()
    assert values == snapshot


def test_legacy_original_corner_order_and_float_values_stay_exact_until_bounds_change():
    original = dict(box1=[[.4000000000000001, .2], [.1, .01]], box2=[[.35, .45], [.2, .25]])
    config = configuration()
    config["domains"] = domains.import_legacy_boxes(original, source={"binding": "unchanged"})
    for row in config["domains"]:
        row["legacy_import"]["decision"] = "use_coordinates_in_measured_frame"
    draft = AnalysisBoxesDraft(config)
    first = draft.outlines()[0].domain_id
    draft.update_metadata(first, name="Kept")
    assert draft.mass_boxes() == original
    assert draft.mass_boxes()["box1"][0][0].hex() == original["box1"][0][0].hex()
    tokens = draft.canvas_outlines(np.eye(3))
    draft.apply_canvas_edits({0: [(10., 110.), (10., 410.), (200., 410.), (200., 110.)]}, tokens)
    assert draft.configuration()["domains"][0]["legacy_import"]["unmodified"] is False
    assert draft.mass_boxes()["box1"] == [[.11, .01], [.41, .2]]


def test_batch_changes_are_atomic_when_another_box_outline_is_invalid():
    draft = AnalysisBoxesDraft(configuration())
    before = draft.outlines()
    tokens = draft.canvas_outlines(np.eye(3))
    changes = {0: [(10., 10.), (10., 20.), (20., 20.), (20., 10.)],
               1: [(1., 1.), (2., 2.), (3., 3.)]}
    with pytest.raises(ValueError, match="positive area"):
        draft.apply_canvas_edits(changes, tokens)
    assert draft.outlines() == before


def test_stale_projection_token_does_not_overwrite_a_newer_draft():
    draft = AnalysisBoxesDraft(configuration())
    tokens = draft.canvas_outlines(np.eye(3))
    changes = {0: [(10., 10.), (10., 20.), (20., 20.), (20., 10.)]}
    draft.apply_canvas_edits(changes, tokens)
    with pytest.raises(ValueError, match="redraw before editing"):
        draft.apply_canvas_edits(changes, tokens)


@pytest.mark.parametrize("matrix", [np.zeros((3, 3)), [[1, 0], [0, 1]],
    [[1, 0, 0], [0, 1, 0], [float("nan"), 0, 1]]])
def test_missing_or_noninvertible_frame_mapping_is_refused(matrix):
    with pytest.raises(ValueError, match="homography"):
        AnalysisBoxesDraft(configuration()).canvas_outlines(matrix)


def test_projective_horizon_is_refused_without_clipping_or_fallback():
    matrix = [[1., 0., 0.], [0., 1., 0.], [1., 0., -2.]]
    with pytest.raises(ValueError, match="projective horizon"):
        metric_to_raw(((1., 1.), (3., 1.), (3., 4.), (1., 4.)), matrix)


@pytest.mark.parametrize("points", [[], [(1., 1.), (2., 2.), (3., 3.)],
    [(0., 0.), (2., 2.), (0., 2.), (2., 0.)], [(True, 0.), (2., 0.), (2., 2.)]])
def test_invalid_or_self_intersecting_outline_is_not_added(points):
    draft = AnalysisBoxesDraft(configuration())
    before = draft.outlines()
    with pytest.raises(ValueError):
        draft.add_outline("Invalid", points)
    assert draft.outlines() == before


def test_duplicate_names_or_ids_do_not_replace_existing_boxes():
    draft = AnalysisBoxesDraft(configuration())
    before = draft.outlines()
    with pytest.raises(ValueError, match="unique"):
        draft.add_rectangle("lower", [1., 2., 1., 2.])
    with pytest.raises(ValueError, match="unique"):
        draft.add_rectangle("New", [1., 2., 1., 2.], domain_id="lower")
    assert draft.outlines() == before
