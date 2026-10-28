"""The manual map: the polygons exactly as drawn, as a version of its own kind.

No classifier is used. Where polygons overlap the smaller one wins, the
proposal's frame-edge fill reaches only along the rig-frame edge, and every
other pixel no polygon covers stays uncovered; such a map can be saved but not
approved. Approved, it is placed and adopted exactly as saved. Temporary
folders and synthetic data only.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_manual as lpm
from darsia.fluidflower.series_geometry import RigFrame

SHAPE = (60, 80)  # rows, cols
SIZE = (80, 60)  # width, height
# A frame of 80 x 60 px over a 0.08 x 0.06 m rig: one pixel per millimetre.
FRAME = RigFrame(
    corners={"top_left": (-0.5, -0.5), "top_right": (-0.5, 79.5),
             "bottom_right": (59.5, 79.5), "bottom_left": (59.5, -0.5)},
    width_m=0.08, height_m=0.06,
)
# A frame 5 px inside the photograph on every side, of the same scale.
INNER = RigFrame(
    corners={"top_left": (4.5, 4.5), "top_right": (4.5, 74.5),
             "bottom_right": (54.5, 74.5), "bottom_left": (54.5, 4.5)},
    width_m=0.07, height_m=0.05,
)
TOP, BAND, BOTTOM = 1, 2, 3


class _Project:
    def __init__(self, root):
        self.root = root


def _layers(top=28.0, left=0.0):
    """A top layer down to ``top``, a band from row 28 to 32 and a bottom layer,
    from column ``left`` to the right border."""

    right = 80.0
    return [
        lp.Polygon(TOP, [(0.0, left), (0.0, right), (top, right), (top, left)]),
        lp.Polygon(BAND, [(28.0, left), (28.0, right), (32.0, right), (32.0, left)]),
        lp.Polygon(BOTTOM, [(32.0, left), (32.0, right), (60.0, right), (60.0, left)]),
    ]


def _session(polygons, *, edge_fill_mm=5.0, run_id="run"):
    session = lp.PaintingSession(run_id=run_id, image_size=SIZE)
    for layer_id in (TOP, BAND, BOTTOM):
        session.layers.append(
            lp.PaintedLayer(layer_id, f"L{layer_id}", "#888888", f"F{layer_id}")
        )
    session.polygons = list(polygons)
    session.edge_fill_mm = edge_fill_mm
    # As a session file reads back.
    return lp.PaintingSession.from_dict(json.loads(json.dumps(session.to_dict())))


def _no_classifier(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a manual map uses no classifier")

    monkeypatch.setattr(lp, "_classify", refuse)
    monkeypatch.setattr(lp, "propose_layers", refuse)


def _files(folder):
    return {path.name: path.read_bytes() for path in Path(folder).iterdir()}


def _settings(session):
    payload = session.to_dict()
    payload.pop("updated_utc", None)
    payload.pop("updated_by", None)
    return payload


def test_the_polygons_are_drawn_exactly_and_the_smaller_one_wins(monkeypatch):
    _no_classifier(monkeypatch)
    # A lens of the band layer inside the top layer.
    lens = lp.Polygon(BAND, [(10.0, 30.0), (10.0, 50.0), (20.0, 50.0), (20.0, 30.0)])
    polygons = [*_layers(), lens]
    manual = lpm.manual_map(_session(polygons), SIZE, frame=FRAME)
    assert manual.classes.shape == SHAPE and manual.image_size == SIZE
    assert manual.inside.all()
    assert np.array_equal(manual.classes, lp.rasterize_polygons(polygons, SHAPE, (1.0, 1.0)))
    assert manual.classes[15, 40] == BAND and manual.classes[5, 5] == TOP
    assert manual.classes[30, 60] == BAND and manual.classes[45, 5] == BOTTOM
    assert manual.uncovered_pixels() == 0 and manual.edge_filled_pixels() == 0
    assert manual.edge_fill_mm == 5.0
    # Strokes guide the proposal only; they are not part of the manual map.
    session = _session(polygons)
    session.strokes.append(lp.Stroke(BOTTOM, 3.0, [(15.0, 40.0)]))
    assert np.array_equal(lpm.manual_map(session, SIZE, frame=FRAME).classes, manual.classes)
    with pytest.raises(ValueError, match="another size"):
        lpm.manual_map(_session(polygons), (40, 30), frame=FRAME)


def test_only_the_frame_edge_strip_is_filled_and_other_gaps_stay_uncovered(monkeypatch):
    _no_classifier(monkeypatch)
    # The polygons stop 3 px before the left edge, and rows 20-27 are left
    # open between the top layer and the band, across the whole rig.
    polygons = _layers(top=20.0, left=3.0)
    manual = lpm.manual_map(_session(polygons), SIZE, frame=FRAME)
    # The proposal's own rule on the same grid.
    inside = lp.frame_mask(FRAME, SHAPE, (1.0, 1.0))
    guide = lp.polygon_guide(polygons, SHAPE, (1.0, 1.0), inside, 6.0)
    fill = lp.frame_edge_fill(guide, inside, 5.0)
    assert np.array_equal(manual.edge_filled, fill > 0)
    expected = guide.owner.copy()
    expected[fill > 0] = fill[fill > 0]
    assert np.array_equal(manual.classes, expected)
    # The left strip and both ends of the open rows are filled; the open rows
    # farther than 5 px from the frame edge stay uncovered.
    assert np.all(manual.classes[:18, :3] == TOP) and np.all(manual.classes[34:, :3] == BOTTOM)
    assert manual.classes[24, 77] != lp.OUTSIDE and manual.edge_filled[24, 77]
    gap = (manual.classes == lp.OUTSIDE) & inside
    assert gap[21:28, 10:70].all() and np.array_equal(manual.uncovered(), gap)
    assert manual.uncovered_pixels() == int(gap.sum()) > 0
    assert manual.edge_filled_pixels() == int((fill > 0).sum()) > 0
    # Without the edge fill nothing is filled.
    bare = lpm.manual_map(_session(polygons, edge_fill_mm=0.0), SIZE, frame=FRAME)
    assert bare.edge_filled_pixels() == 0
    assert bare.uncovered_pixels() == manual.uncovered_pixels() + manual.edge_filled_pixels()


def test_pixels_outside_the_rig_frame_are_outside_and_never_uncovered():
    manual = lpm.manual_map(_session(_layers(top=20.0)), SIZE, frame=INNER)
    assert not manual.inside[:4].any() and manual.inside[10:50, 10:70].all()
    assert np.all(manual.classes[~manual.inside] == lp.OUTSIDE)
    assert not manual.uncovered()[~manual.inside].any()
    assert manual.uncovered()[24, 40]


@pytest.mark.parametrize("seed", range(12))
def test_the_edge_strip_is_the_frame_edge_fill_of_the_proposal(seed):
    # The manual map's strip, one layer at a time, against the proposal's own
    # frame_edge_fill on random grids, with ties and layers outside the frame.
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(5, 40)), int(rng.integers(5, 40)))
    inside = rng.random(shape) > 0.15
    owner = np.where(rng.random(shape) > 0.6, rng.integers(1, 6, shape), 0).astype(np.uint8)
    layers = sorted(int(value) for value in rng.choice(np.arange(1, 9), int(rng.integers(1, 6)), replace=False))
    distance = {
        layer: np.where(
            rng.random(shape) > 0.1, rng.integers(0, 8, shape).astype(np.float32), np.inf
        ).astype(np.float32)
        for layer in layers
    }
    guide = lp.PolygonGuide(
        owner=owner,
        fixed=np.zeros(shape, dtype=bool),
        restricted=np.zeros(shape, dtype=bool),
        distance=distance,
        training=np.zeros(shape, dtype=np.uint8),
        snap_px=0.0,
    )
    for edge_px in (0.0, 1.0, 2.5, 5.0, 30.0):
        strip = lpm._frame_edge_strip(
            owner, inside, edge_px, ((layer, distance[layer]) for layer in layers)
        )
        expected = lp.frame_edge_fill(guide, inside, edge_px)
        assert strip.dtype == expected.dtype and np.array_equal(strip, expected)
    assert not lpm._frame_edge_strip(owner, inside, 5.0, iter(())).any()


def test_a_manual_map_is_saved_as_a_version_and_read_back_exactly(tmp_path):
    project = _Project(tmp_path)
    session = _session(_layers(top=20.0, left=3.0), run_id="ac17")
    manual = lpm.manual_map(session, SIZE, frame=FRAME)
    record = lpm.save_manual_version(project, session, manual, name=" Drawn ")
    folder = Path(record["folder"])
    assert record["kind"] == lpm.MANUAL_MAP_KIND == "manual map" and lpm.is_manual_version(record)
    assert record["name"] == "Drawn" and record["run_id"] == "ac17"
    assert record["sha256"] == lpm.manual_map_digest(session, manual)
    assert record["uncovered_pixels"] == manual.uncovered_pixels() > 0
    assert record["edge_filled_pixels"] == manual.edge_filled_pixels() > 0
    # The session and the map; no proposal and no corrections.
    assert sorted(path.name for path in folder.iterdir()) == [
        "manual_map.npz", "session.json", "version.json"
    ]
    assert record["files"] == {
        name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
        for name in lpm.MANUAL_VERSION_FILES
    }
    assert json.loads((folder / "version.json").read_text(encoding="utf-8")) == {
        key: value for key, value in record.items() if key != "folder"
    }
    assert [row["folder"] for row in lp.list_versions(project, "ac17")] == [str(folder)]
    loaded_session, loaded = lpm.load_manual_version(record)
    assert isinstance(loaded, lpm.ManualMap)
    assert loaded_session.to_dict() == session.to_dict()
    for name in ("classes", "inside", "edge_filled"):
        assert np.array_equal(getattr(loaded, name), getattr(manual, name))
    assert loaded.classes.dtype == np.uint8 and loaded.inside.dtype == bool
    assert loaded.image_size == SIZE and loaded.edge_fill_mm == 5.0
    # The same drawing gives the same SHA-256 in a new folder; another one does not.
    again = lpm.save_manual_version(project, session, lpm.manual_map(session, SIZE, frame=FRAME), name="Again")
    assert again["sha256"] == record["sha256"] and again["folder"] != record["folder"]
    other = _session(_layers(top=24.0, left=3.0), run_id="ac17")
    assert lpm.manual_map_digest(other, lpm.manual_map(other, SIZE, frame=FRAME)) != record["sha256"]
    with pytest.raises(ValueError, match="needs a name"):
        lpm.save_manual_version(project, session, manual, name="  ")
    with pytest.raises(ValueError, match="another size"):
        lpm.save_manual_version(project, lp.PaintingSession(run_id="ac17", image_size=(40, 30)), manual, name="X")


def test_a_changed_manual_map_file_or_record_is_refused(tmp_path):
    project = _Project(tmp_path)
    session = _session(_layers(), run_id="ac17")
    record = lpm.save_manual_version(project, session, lpm.manual_map(session, SIZE, frame=FRAME), name="Map")
    path = Path(record["folder"]) / "manual_map.npz"
    original = path.read_bytes()
    path.write_bytes(original + b"\0")
    with pytest.raises(ValueError, match="manual_map.npz has changed"):
        lpm.load_manual_version(record)
    with pytest.raises(ValueError, match="manual_map.npz has changed"):
        lpm.version_layers(record, frame=FRAME)
    path.write_bytes(original)
    partial = dict(record, files={"session.json": record["files"]["session.json"]})
    with pytest.raises(ValueError, match="does not list its files"):
        lpm.load_manual_version(partial)
    with pytest.raises(ValueError, match="not a manual map"):
        lpm.load_manual_version(dict(record, kind="proposal"))
    with pytest.raises(ValueError, match="Unsupported manual map"):
        np.savez_compressed(tmp_path / "other.npz", meta=np.asarray(json.dumps({"format": "x"})))
        lpm.ManualMap.load(tmp_path / "other.npz")


def test_approval_needs_every_pixel_inside_the_frame_and_names_the_sha(tmp_path):
    project = _Project(tmp_path)
    open_session = _session(_layers(top=20.0, left=3.0), run_id="ac17")
    gaps = lpm.save_manual_version(
        project, open_session, lpm.manual_map(open_session, SIZE, frame=FRAME), name="Gaps"
    )
    with pytest.raises(ValueError, match="pixels inside the rig frame are covered by no polygon"):
        lpm.approve_manual_version(gaps, reviewer="Test reviewer")
    assert not lp.approvals_path(project, "ac17").exists()
    # The open rows covered, and the strip left at the frame edge filled.
    session = _session(_layers(left=3.0), run_id="ac17")
    manual = lpm.manual_map(session, SIZE, frame=FRAME)
    assert manual.uncovered_pixels() == 0 and manual.edge_filled_pixels() > 0
    record = lpm.save_manual_version(project, session, manual, name="Complete")
    before = _files(record["folder"])
    approval = lpm.approve_manual_version(
        record, reviewer=" Test reviewer ", now="2026-10-21T09:00:00Z"
    )
    assert approval == {
        "format": lp.LAYER_APPROVAL_FORMAT,
        "run_id": "ac17",
        "version": Path(record["folder"]).name,
        "name": "Complete",
        "kind": "manual map",
        "sha256": record["sha256"],
        "files": record["files"],
        "reviewer": "Test reviewer",
        "approved_utc": "2026-10-21T09:00:00Z",
    }
    assert _files(record["folder"]) == before
    # The record and file of a proposal's approval: one approvals file for both.
    classes = np.full(SHAPE, TOP, dtype=np.uint8)
    proposal = lp.LayerProposal(classes, np.full(SHAPE, 255, np.uint8), SIZE, "inputs")
    edited = lp.save_version(project, session, proposal, [], name="Proposal")
    other = lp.approve_version(edited, reviewer=" Test reviewer ", now="2026-10-21T09:01:00Z")
    assert sorted(other) == sorted(approval)
    lines = lp.approvals_path(project, "ac17").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [approval, other]
    assert lines[0] == json.dumps(approval, ensure_ascii=False, sort_keys=True)
    assert lp.read_map_approvals(project, "ac17") == [approval, other]
    assert lp.map_approval([approval, other], record["sha256"]) == approval
    # A record whose SHA-256 is not that of its content is refused.
    with pytest.raises(ValueError, match="does not match its content"):
        lpm.approve_manual_version(dict(record, sha256="0" * 64))


def test_restoring_a_manual_map_brings_back_the_session_only(tmp_path):
    project = _Project(tmp_path)
    session = _session(_layers(), run_id="ac17")
    record = lpm.save_manual_version(project, session, lpm.manual_map(session, SIZE, frame=FRAME), name="Map")
    # The current state: other polygons, a proposal and a correction.
    current = _session(_layers(top=20.0), run_id="ac17")
    lp.save_painting_session(project, current)
    classes = np.full(SHAPE, TOP, dtype=np.uint8)
    lp.LayerProposal(classes, np.full(SHAPE, 255, np.uint8), SIZE, "inputs").save(
        lp.proposal_path(project, "ac17")
    )
    lp.save_corrections(project, "ac17", [lp.Correction("brush", BAND, [(5.0, 5.0)], radius_px=2.0)])
    proposal_before = lp.proposal_path(project, "ac17").read_bytes()
    corrections_before = lp.corrections_path(project, "ac17").read_bytes()
    restored = lpm.restore_manual_version(project, record, actor="test")
    assert _settings(restored) == _settings(session)
    saved = json.loads(lp.session_path(project, "ac17").read_text(encoding="utf-8"))
    assert _settings(lp.PaintingSession.from_dict(saved)) == _settings(session)
    assert lp.proposal_path(project, "ac17").read_bytes() == proposal_before
    assert lp.corrections_path(project, "ac17").read_bytes() == corrections_before


def _old_version_layers(record, frame):
    """The layers of a saved version as placement and adoption drew them
    before the manual map (frozen copy)."""

    session, proposal, corrections = lp.load_version(record)
    labels, regions = lp.painted_label_map(session, proposal, frame=frame, corrections=corrections)
    table = np.zeros(int(labels.max()) + 1, dtype=np.uint8)
    for row in regions:
        table[int(row["label_id"])] = int(row["layer_id"])
    return session, table[labels]


def test_placement_and_adoption_layers_of_a_version(tmp_path):
    project = _Project(tmp_path)
    session = _session(_layers(left=3.0), run_id="ac17")
    manual = lpm.manual_map(session, SIZE, frame=FRAME)
    record = lpm.save_manual_version(project, session, manual, name="Manual")
    loaded_session, layers = lpm.version_layers(record, frame=None)
    assert loaded_session.to_dict() == session.to_dict()
    assert layers.dtype == np.uint8 and np.array_equal(layers, manual.classes)
    open_session = _session(_layers(top=20.0), run_id="ac17")
    gaps = lpm.save_manual_version(
        project, open_session, lpm.manual_map(open_session, SIZE, frame=FRAME), name="Gaps"
    )
    with pytest.raises(ValueError, match="no polygon covers"):
        lpm.version_layers(gaps, frame=FRAME)
    # A proposal with a correction is drawn exactly as before.
    classes = np.full(SHAPE, TOP, dtype=np.uint8)
    classes[30:] = BOTTOM
    proposal = lp.LayerProposal(classes, np.full(SHAPE, 255, np.uint8), SIZE, "inputs")
    corrections = [lp.Correction("brush", BAND, [(10.0, 10.0)], radius_px=3.0)]
    edited = lp.save_version(project, session, proposal, corrections, name="Edited")
    old_session, old = _old_version_layers(edited, FRAME)
    new_session, new = lpm.version_layers(edited, frame=FRAME)
    assert np.array_equal(new, old) and new_session.to_dict() == old_session.to_dict()
    assert np.array_equal(
        lpm.painted_layers(session, proposal, frame=FRAME, corrections=corrections), old
    )


def test_differences_and_the_area_of_a_pixel():
    session = _session(_layers(top=20.0))
    manual = lpm.manual_map(session, SIZE, frame=INNER)
    other = manual.classes.copy()
    other[40:45, 10:20] = TOP  # Bottom layer in the manual map.
    other[24, 40] = BAND  # Uncovered in the manual map: not a difference.
    other[0, 0] = BAND  # Outside the rig frame: not a difference.
    differ = lpm.manual_map_differences(manual, other)
    assert differ.dtype == bool and int(differ.sum()) == 50 and differ[40:45, 10:20].all()
    assert not lpm.manual_map_differences(manual, manual.classes).any()
    with pytest.raises(ValueError, match="same grid"):
        lpm.manual_map_differences(manual, other[:-1])
    # One pixel per millimetre: 0.01 cm². Without a rig size the photograph
    # is taken to be 1 m wide: 80 px per metre.
    assert lpm.pixel_area_cm2(FRAME, SIZE) == pytest.approx(0.01)
    assert lpm.pixel_area_cm2(None, SIZE) == pytest.approx(1.0e4 / 80.0**2)
