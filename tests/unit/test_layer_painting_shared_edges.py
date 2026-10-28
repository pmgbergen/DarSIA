"""Shared points and edges of layer painting polygons, and the choice whether
the classifier may move shared edges.

Points of several polygons within half a raw pixel are one shared point; an
edge between two shared points that polygons of different layers both have
is a shared edge. By default the classifier places a shared edge within the
snap distance, like any other edge; when it may not, the boundary lies on the
shared edge, drawn at full resolution in the exported map. Synthetic
photographs only; no real series is read.
"""

import json

import numpy as np

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower.series_geometry import RigFrame

TOP, BOTTOM, EXTRA = 1, 2, 3
# A 160 x 120 px photograph of a 0.16 x 0.12 m rig: one raw pixel per mm.
WIDTH, HEIGHT = 160, 120
FRAME = RigFrame(
    corners={"top_left": (-0.5, -0.5), "top_right": (-0.5, 159.5),
             "bottom_right": (119.5, 159.5), "bottom_left": (119.5, -0.5)},
    width_m=0.16, height_m=0.12,
)
# The two layers meet at row 56 in the photograph, but their polygons share
# an edge at row 50. A third layer lies well below, with edges of its own.
TOP_POLYGON = [(0.0, 0.0), (0.0, 160.0), (50.0, 160.0), (50.0, 0.0)]
BOTTOM_POLYGON = [(50.0, 0.0), (50.0, 160.0), (120.0, 160.0), (120.0, 0.0)]
EXTRA_POLYGON = [(90.0, 100.0), (90.0, 150.0), (115.0, 150.0), (115.0, 100.0)]


def _polygons():
    return [
        lp.Polygon(TOP, list(TOP_POLYGON)),
        lp.Polygon(BOTTOM, list(BOTTOM_POLYGON)),
        lp.Polygon(EXTRA, list(EXTRA_POLYGON)),
    ]


def _image():
    """Raw photograph and an analysis grid of half its size."""

    rng = np.random.default_rng(5)
    raw = np.empty((HEIGHT, WIDTH, 3), dtype=np.float32)
    raw[:56] = (0.15, 0.30, 0.70)
    raw[56:] = (0.70, 0.45, 0.15)
    raw[94:111, 104:146] = (0.20, 0.65, 0.25)
    raw = np.clip(raw + rng.normal(0.0, 0.02, raw.shape).astype(np.float32), 0.0, 1.0)
    return lp.PaintingImage("photo.png", (WIDTH, HEIGHT), raw, raw[::2, ::2].copy())


def _session(*, move_shared_edges=True):
    session = lp.PaintingSession(run_id="run", image_size=(WIDTH, HEIGHT))
    for layer_id in (TOP, BOTTOM, EXTRA):
        session.layers.append(
            lp.PaintedLayer(layer_id, f"L{layer_id}", "#888888", f"F{layer_id}")
        )
    session.polygons = _polygons()
    session.snap_mm = 12.0
    session.smoothing = "low"
    session.min_region_cm2 = 0.0
    session.move_shared_edges = move_shared_edges
    return session


def _layers(labels, regions):
    """The layer of each pixel of an exported label map."""

    table = np.zeros(int(labels.max()) + 1, dtype=np.uint8)
    for row in regions:
        table[row["label_id"]] = row["layer_id"]
    return table[labels]


# ---------------------------------------------------------------------------
# Shared points and edges


def test_points_of_several_polygons_within_half_a_pixel_are_one_shared_point():
    points = {
        0: [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0)],
        1: [(0.5, 0.0), (10.0, 10.3), (20.0, 20.0)],
        2: [(30.0, 30.0), (30.0, 30.4), (40.0, 30.0)],  # Two of its own points.
        3: [(0.0, 10.6), (5.0, 5.0), (8.0, 1.0)],  # 0.6 px from a point of 0.
    }
    assert lp.shared_points(points) == [[(0, 0), (1, 0)], [(0, 2), (1, 1)]]
    # A chain of near points is one shared point.
    chain = {
        0: [(0.0, 0.0), (9.0, 9.0), (9.0, 0.0)],
        1: [(0.0, 0.4), (5.0, 5.0), (7.0, 0.0)],
        2: [(0.0, 0.8), (6.0, 9.0), (8.0, 3.0)],
    }
    assert lp.shared_points(chain) == [[(0, 0), (1, 0), (2, 0)]]
    assert lp.shared_points(points, tolerance=0.7)[1] == [(0, 1), (3, 0)]
    assert lp.shared_points({}) == []


def test_a_shared_edge_needs_both_end_points_and_two_layers():
    top = lp.Polygon(
        TOP, [(0.0, 0.0), (0.0, 40.0), (10.0, 40.0), (10.0, 20.0), (10.0, 0.0)]
    )
    bottom = lp.Polygon(BOTTOM, [(10.0, 0.0), (10.0, 20.2), (20.0, 20.0), (20.0, 0.0)])
    # The same layer on the other side shares no layer boundary.
    same = lp.Polygon(TOP, [(10.0, 20.0), (10.0, 40.0), (20.0, 40.0)])
    assert lp.shared_edges([top, bottom]) == [((10.0, 20.0), (10.0, 0.0))]
    assert lp.shared_edges([top, same]) == []
    # One shared end point is not enough.
    corner = lp.Polygon(BOTTOM, [(10.0, 40.0), (20.0, 40.0), (20.0, 30.0)])
    assert lp.shared_edges([top, corner]) == []
    # Each shared edge is given once, also when three polygons have it.
    third = lp.Polygon(EXTRA, [(10.0, 0.0), (10.0, 20.0), (5.0, 10.0)])
    assert lp.shared_edges([top, bottom, third]) == [((10.0, 20.0), (10.0, 0.0))]


# ---------------------------------------------------------------------------
# The polygon guide


def test_kept_shared_edges_get_no_snap_band_and_other_edges_keep_theirs():
    # The shared edge runs along row 30 from column 0 to 40. From 40 to 80 the
    # top polygon's edge is its own, with nothing drawn below it.
    top = lp.Polygon(
        TOP, [(0.0, 0.0), (0.0, 80.0), (30.0, 80.0), (30.0, 40.0), (30.0, 0.0)]
    )
    bottom = lp.Polygon(BOTTOM, [(30.0, 0.0), (30.0, 40.0), (60.0, 40.0), (60.0, 0.0)])
    polygons = [top, bottom]
    shape, scale = (60, 80), (1.0, 1.0)
    inside = np.ones(shape, dtype=bool)
    moved = lp.polygon_guide(polygons, shape, scale, inside, 6.0)
    kept = lp.polygon_guide(polygons, shape, scale, inside, 6.0, keep_shared_edges=True)
    assert np.all(kept.fixed[moved.fixed])
    gained = kept.fixed & ~moved.fixed
    rows, cols = np.nonzero(gained)
    # Only along the shared edge, up to the photograph's border, and not
    # within the snap distance of the other edges, which start at column 40.
    assert gained[24:37, 0:30].all()
    assert rows.min() >= 24 and rows.max() <= 36 and cols.max() <= 40
    assert np.array_equal(kept.fixed[:, 41:], moved.fixed[:, 41:])
    # Training samples and the rest of the guide are unchanged.
    assert np.array_equal(kept.training, moved.training)
    assert np.array_equal(kept.owner, moved.owner)
    # Without shared edges the choice changes nothing.
    apart = [(31.0, 0.0), (31.0, 40.0), (60.0, 40.0), (60.0, 0.0)]
    alone = [top, lp.Polygon(BOTTOM, apart)]
    plain = lp.polygon_guide(alone, shape, scale, inside, 6.0)
    same = lp.polygon_guide(alone, shape, scale, inside, 6.0, keep_shared_edges=True)
    assert np.array_equal(plain.fixed, same.fixed)
    assert np.array_equal(plain.restricted, same.restricted)


# ---------------------------------------------------------------------------
# Proposal and export


def test_no_keeps_the_boundary_on_the_shared_edge_at_full_resolution():
    image = _image()
    moved_session, kept_session = _session(), _session(move_shared_edges=False)
    moved = lp.propose_layers(image, moved_session, frame=FRAME)
    kept = lp.propose_layers(image, kept_session, frame=FRAME)
    assert moved.digest != kept.digest

    moved_map = _layers(*lp.painted_label_map(moved_session, moved, frame=FRAME))
    kept_map = _layers(*lp.painted_label_map(kept_session, kept, frame=FRAME))
    owner = lp.rasterize_polygons(_polygons(), (HEIGHT, WIDTH), (1.0, 1.0))
    # Yes: the classifier moves the shared edge towards the colour boundary.
    assert np.mean(moved_map[52:55] == TOP) > 0.9
    # No: the boundary lies exactly on the shared edge, at full resolution.
    assert np.array_equal(kept_map[40:62], owner[40:62])
    assert np.all(kept_map[45:51] == TOP) and np.all(kept_map[51:56] == BOTTOM)
    # The third layer's own edges are placed as before.
    assert np.array_equal(kept_map[80:], moved_map[80:])
    assert np.array_equal(kept.classes[40:], moved.classes[40:])
    assert np.mean(kept_map[91:93, 110:140] == BOTTOM) > 0.9  # Moved in.


def test_the_choice_is_saved_and_digested_only_for_no():
    session = _session()
    settings = session.to_dict()["proposal_settings"]
    assert "move_shared_edges" not in settings
    assert "move_shared_edges" not in lp._session_text(session)
    digest = lp.proposal_digest(session, FRAME)

    session.move_shared_edges = False
    payload = json.loads(json.dumps(session.to_dict()))
    assert payload["proposal_settings"]["move_shared_edges"] is False
    assert lp.proposal_digest(session, FRAME) != digest
    back = lp.PaintingSession.from_dict(payload)
    assert back.move_shared_edges is False
    assert lp.proposal_digest(back, FRAME) == lp.proposal_digest(session, FRAME)

    session.move_shared_edges = True
    assert lp.proposal_digest(session, FRAME) == digest
    # A session saved before the choice existed may move shared edges.
    payload["proposal_settings"].pop("move_shared_edges")
    assert lp.PaintingSession.from_dict(payload).move_shared_edges is True
    # Without polygons there are no shared edges, and the choice is not digested.
    session.polygons = []
    strokes_only = lp.proposal_digest(session, FRAME)
    session.move_shared_edges = False
    assert lp.proposal_digest(session, FRAME) == strokes_only
