"""Hand corrections on top of a layer proposal.

Corrections are brush strokes, restore strokes, lasso outlines and region
fills drawn on top of a proposal. They are kept in their own file, applied in
drawing order and never change the proposal. Synthetic grids only; no real
series is read.
"""

import json

import numpy as np
import pytest

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower.series_geometry import SeriesRunInputs

OUT, LEFT, RIGHT, LENS = lp.OUTSIDE, 2, 3, 4


def _classes():
    """20 x 30 grid: a 2 px border outside, LEFT on the left and RIGHT on the right."""

    classes = np.full((20, 30), OUT, dtype=np.uint8)
    classes[2:18, 2:15] = LEFT
    classes[2:18, 15:28] = RIGHT
    return classes


def _proposal(classes, image_size=(30, 20)):
    return lp.LayerProposal(
        classes=classes,
        confidence=np.full(classes.shape, 255, dtype=np.uint8),
        image_size=image_size,
        digest="d",
    )


class _Project:
    def __init__(self, root):
        self.root = root


def test_a_correction_keeps_the_saved_precision_and_round_trips():
    points = [(10.04, 20.06), (11.0, 21.0)]
    brush = lp.Correction("brush", LEFT, points, radius_px=3.456)
    assert brush.points == [(10.0, 20.1), (11.0, 21.0)] and brush.radius_px == 3.46
    assert lp.Correction.from_dict(brush.to_dict()) == brush
    restore = lp.Correction("restore", 7, [(1.0, 2.0)], radius_px=2.0)
    assert restore.layer_id == lp.ERASER
    fill = lp.Correction("fill", RIGHT, [(5.0, 5.0)], replaced_layer=LEFT)
    assert lp.Correction.from_dict(json.loads(json.dumps(fill.to_dict()))) == fill
    for bad in (
        lambda: lp.Correction("smudge", LEFT, [(1.0, 1.0)]),
        lambda: lp.Correction("lasso", LEFT, [(1.0, 1.0), (2.0, 2.0)]),
        lambda: lp.Correction("brush", LEFT, [(1.0, 1.0)]),
        lambda: lp.Correction("brush", OUT, [(1.0, 1.0)], radius_px=1.0),
        lambda: lp.Correction("fill", LEFT, [(1.0, 1.0)]),
    ):
        with pytest.raises(ValueError):
            bad()


def test_corrections_get_their_own_file_only_once_there_are_any(tmp_path):
    project = _Project(tmp_path)
    path = lp.corrections_path(project, "AC17")
    assert path.name == "AC17.corrections.json"
    assert path.parent == lp.session_path(project, "AC17").parent
    assert lp.save_corrections(project, "AC17", []) is None and not path.exists()
    assert lp.load_corrections(project, "AC17") == []
    rows = [
        lp.Correction("brush", LEFT, [(1.0, 2.0), (3.0, 4.0)], radius_px=2.0,
                      proposal_digest="p1"),
        lp.Correction("fill", RIGHT, [(5.0, 6.0)], replaced_layer=LEFT,
                      proposal_digest="p1"),
    ]
    assert lp.save_corrections(project, "AC17", rows) == path
    text = path.read_text(encoding="utf-8")
    assert text.count('{"kind":') == 2  # One correction per line.
    assert json.loads(text)["format"] == lp.LAYER_CORRECTIONS_FORMAT
    assert lp.load_corrections(project, "AC17") == rows
    assert not lp.session_path(project, "AC17").exists()
    assert lp.corrections_digest(rows) and lp.corrections_digest([]) == ""
    lp.save_corrections(project, "AC17", [])
    assert path.exists() and lp.load_corrections(project, "AC17") == []


def test_corrections_apply_in_drawing_order_inside_the_frame_only():
    classes = _classes()
    inside = classes != OUT
    corrections = [
        # A lasso of LENS over both layers that reaches out of the frame.
        lp.Correction(
            "lasso", LENS, [(0.0, 10.0), (0.0, 20.0), (6.0, 20.0), (6.0, 10.0)]
        ),
        # A brush of RIGHT into the left layer.
        lp.Correction("brush", RIGHT, [(12.0, 6.0), (12.0, 10.0)], radius_px=1.0),
        # A restore stroke over part of the lasso: the proposal shows again there.
        lp.Correction("restore", 0, [(3.0, 12.0)], radius_px=1.0),
    ]
    result, decided, skipped = lp.apply_corrections(
        classes, corrections, (1.0, 1.0), inside=inside
    )
    assert skipped == []
    assert np.array_equal(classes, _classes())  # The proposal is never changed.
    assert np.all(result[~inside] == OUT)
    assert result[5, 18] == LENS and result[5, 11] == LENS and decided[5, 18]
    assert result[3, 12] == LEFT and not decided[3, 12]
    assert result[12, 8] == RIGHT and decided[12, 8]
    assert result[15, 5] == LEFT and not decided[15, 5]


def test_a_fill_takes_its_connected_region_and_is_skipped_once_it_no_longer_fits():
    classes = _classes()
    classes[8:11, 6:9] = RIGHT  # An island of RIGHT inside LEFT.
    fill = lp.Correction("fill", LEFT, [(9.0, 7.0)], replaced_layer=RIGHT)
    result, decided, skipped = lp.apply_corrections(
        classes, [fill], (1.0, 1.0), inside=classes != OUT
    )
    assert skipped == [] and np.all(result[8:11, 6:9] == LEFT)
    assert np.all(decided[8:11, 6:9]) and decided.sum() == 9
    assert np.all(result[2:18, 15:28] == RIGHT)  # The other RIGHT region stays.
    # On a proposal without the island the fill does not apply.
    result, _decided, skipped = lp.apply_corrections(
        _classes(), [fill], (1.0, 1.0), inside=_classes() != OUT
    )
    assert skipped == [0] and np.array_equal(result, _classes())
    # Nor does a correction of a layer that is no longer in the session.
    brush = lp.Correction("brush", 9, [(9.0, 7.0)], radius_px=1.0)
    _result, _decided, skipped = lp.apply_corrections(
        _classes(), [brush], (1.0, 1.0), layer_ids={LEFT, RIGHT}
    )
    assert skipped == [0]


def test_corrections_are_drawn_in_raw_pixels_on_any_grid():
    coarse = _classes()[::2, ::2].copy()  # The proposal on a half-resolution grid.
    fill = lp.Correction("fill", LEFT, [(4.0, 24.0)], replaced_layer=RIGHT)
    result, _decided, skipped = lp.apply_corrections(
        coarse, [fill], (0.5, 0.5), inside=coarse != OUT
    )
    assert skipped == [] and not np.any(result == RIGHT)


def test_the_corrected_map_counts_its_own_regions():
    classes = _classes()
    classes[8:11, 6:9] = RIGHT
    proposal = _proposal(classes)
    proposal.layers = [
        {"layer_id": LEFT, "pixels": 1, "regions": 1},
        {"layer_id": RIGHT, "pixels": 1, "regions": 2},
    ]
    assert lp.proposal_region_count(proposal) == lp.class_region_count(classes) == 4
    fill = lp.Correction("fill", LEFT, [(9.0, 7.0)], replaced_layer=RIGHT)
    corrected, _decided, skipped = lp.corrected_classes(proposal, [fill])
    assert skipped == [] and lp.class_region_count(corrected) == 3
    assert lp.corrections_not_applied(proposal, [fill]) == []
    stale = lp.Correction("fill", LEFT, [(9.0, 20.0)], replaced_layer=LEFT)
    assert lp.corrections_not_applied(proposal, [stale]) == [0]
    assert lp.corrections_not_applied(None, [stale]) == []


def test_readiness_asks_for_the_properties_of_a_layer_a_correction_adds(tmp_path):
    session = lp.PaintingSession(run_id="run", image_size=(30, 20))
    for layer_id in (LEFT, RIGHT, LENS):
        session.layers.append(
            lp.PaintedLayer(layer_id, f"L{layer_id}", "#888888", f"F{layer_id}")
        )
    session.facies[f"F{LEFT}"] = lp.FaciesEntry("porous_medium", 0.4, 10.0)
    session.facies[f"F{RIGHT}"] = lp.FaciesEntry("porous_medium", 0.4, 20.0)
    session.facies[f"F{LENS}"] = lp.FaciesEntry("porous_medium", None, None)
    baseline = tmp_path / "raw.png"
    baseline.write_bytes(b"photo")
    reference = SeriesRunInputs(
        "run", tmp_path / "run.toml", baseline_image=baseline, width_m=0.3, height_m=0.2
    )
    proposal = _proposal(_classes())
    proposal.digest = lp.proposal_digest(session, None)
    proposal.layers = [
        {"layer_id": LEFT, "pixels": 10, "regions": 1},
        {"layer_id": RIGHT, "pixels": 10, "regions": 1},
    ]
    assert lp.check_ready_for_revision(session, proposal, None, reference) == []
    square = [(4.0, 4.0), (4.0, 8.0), (8.0, 8.0), (8.0, 4.0)]
    lens = lp.Correction("lasso", LENS, square)
    problems = lp.check_ready_for_revision(session, proposal, None, reference, [lens])
    assert len(problems) == 1 and f"F{LENS}" in problems[0]


def test_the_saved_map_draws_corrections_at_full_resolution_and_keeps_them():
    session = lp.PaintingSession(run_id="run", image_size=(60, 40))
    for layer_id in (LEFT, RIGHT, LENS):
        session.layers.append(
            lp.PaintedLayer(layer_id, f"L{layer_id}", "#888888", f"F{layer_id}")
        )
    proposal = _proposal(_classes(), image_size=(60, 40))  # A half-resolution grid.
    plain, plain_regions = lp.painted_label_map(session, proposal, frame=None)
    again, again_regions = lp.painted_label_map(
        session, proposal, frame=None, corrections=()
    )
    assert np.array_equal(again, plain) and again_regions == plain_regions
    assert not any(row["layer_id"] == LENS for row in plain_regions)
    # A dot of LENS below the sliver size (8 px here) is kept: a correction decided it.
    dot = lp.Correction("brush", LENS, [(20.0, 10.0)], radius_px=1.0)
    _labels, regions = lp.painted_label_map(
        session, proposal, frame=None, corrections=[dot]
    )
    lens = [row for row in regions if row["layer_id"] == LENS]
    assert len(lens) == 1 and 0 < lens[0]["pixels"] < 8


def test_corrections_added_one_at_a_time_give_the_same_map():
    classes = _classes()
    inside = classes != OUT
    corrections = [
        lp.Correction("brush", RIGHT, [(5.0, 5.0), (5.0, 12.0)], radius_px=1.5),
        lp.Correction("fill", LENS, [(10.0, 20.0)], replaced_layer=RIGHT),
        lp.Correction("restore", 0, [(10.0, 20.0)], radius_px=2.0),
        lp.Correction("lasso", LEFT, [(12.0, 18.0), (12.0, 26.0), (16.0, 22.0)]),
        # Partly off the grid: only the part on it counts.
        lp.Correction("brush", LENS, [(-3.0, 29.0), (3.0, 31.0)], radius_px=2.0),
    ]
    whole, whole_decided, _skipped = lp.apply_corrections(
        classes, corrections, (1.0, 1.0), inside=inside
    )
    state = None
    for correction in corrections:
        result, decided, skipped = lp.apply_corrections(
            classes, [correction], (1.0, 1.0), inside=inside, start=state
        )
        assert skipped == []
        state = (result, decided)
    assert np.array_equal(state[0], whole) and np.array_equal(state[1], whole_decided)
    assert lp.layer_at(whole, (10.0, 20.0), (1.0, 1.0)) == RIGHT  # Restored.
    assert lp.layer_at(whole, (13.0, 22.0), (1.0, 1.0)) == LEFT
    assert lp.layer_at(whole, (25.0, 0.0), (1.0, 1.0)) is None
    assert lp.layer_at(whole[::2, ::2], (10.0, 7.0), (0.5, 0.5)) == LEFT
