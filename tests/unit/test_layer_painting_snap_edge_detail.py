"""Proposal options for thin layers and frame edges in the layer painting tool.

A layer may keep its own snap distance (0 keeps a thin layer's polygons
exactly), uncovered strips along the rig-frame edge take the nearest
polygon's layer, and a proposal can use a finer analysis grid. Synthetic
photographs only; no real series is read.
"""

import json

import numpy as np

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower.series_geometry import RigFrame

SHAPE = (60, 80)  # rows, cols; one grid pixel is one raw pixel.
# A frame of 80 x 60 px over a 0.08 x 0.06 m rig: one pixel per millimetre.
FRAME = RigFrame(
    corners={"top_left": (-0.5, -0.5), "top_right": (-0.5, 79.5),
             "bottom_right": (59.5, 79.5), "bottom_left": (59.5, -0.5)},
    width_m=0.08, height_m=0.06,
)
TOP, BAND, BOTTOM = 1, 2, 3


def _band_polygons(left=0.0):
    """A thick top layer, a 4 px thin band and a thick bottom layer."""

    right = 80.0
    return [
        lp.Polygon(TOP, [(0.0, left), (0.0, right), (28.0, right), (28.0, left)]),
        lp.Polygon(BAND, [(28.0, left), (28.0, right), (32.0, right), (32.0, left)]),
        lp.Polygon(BOTTOM, [(32.0, left), (32.0, right), (60.0, right), (60.0, left)]),
    ]


def _image(seed=1, strip=None):
    """Blue-ish layers, a band of nearly the same colour, and pixel noise."""

    rng = np.random.default_rng(seed)
    rgb = np.empty((*SHAPE, 3), dtype=np.float32)
    rgb[:28] = (0.10, 0.20, 0.55)
    rgb[28:32] = (0.13, 0.22, 0.52)
    rgb[32:] = (0.08, 0.18, 0.58)
    if strip is not None:
        rgb[:, :strip] = (0.13, 0.22, 0.52)  # A bright frame edge, like the band.
    rgb += rng.normal(0.0, 0.03, rgb.shape).astype(np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)
    return lp.PaintingImage("synthetic.png", (SHAPE[1], SHAPE[0]), rgb, rgb.copy())


def _session(polygons, *, snap_mm=6.0, band_snap=None, edge_fill_mm=None):
    session = lp.PaintingSession(run_id="run", image_size=(SHAPE[1], SHAPE[0]))
    for layer_id in (TOP, BAND, BOTTOM):
        session.layers.append(lp.PaintedLayer(layer_id, f"L{layer_id}", "#888888", f"F{layer_id}"))
    session.layer(BAND).snap_mm = band_snap
    session.polygons = polygons
    session.snap_mm = snap_mm
    session.min_region_cm2 = 0.0
    if edge_fill_mm is not None:
        session.edge_fill_mm = edge_fill_mm
    return session


def test_a_layer_snap_of_zero_fixes_a_thin_layer_and_keeps_it_in_its_polygons():
    polygons = _band_polygons()
    inside = np.ones(SHAPE, dtype=bool)
    common = lp.polygon_guide(polygons, SHAPE, (1.0, 1.0), inside, 6.0)
    band = common.owner == BAND
    assert band.any() and not (common.fixed & band).any()  # 4 px < 2 x 6 px.
    own = lp.polygon_guide(polygons, SHAPE, (1.0, 1.0), inside, 6.0, layer_snap_px={BAND: 0.0})
    assert (own.fixed & band).sum() == band.sum()
    assert np.array_equal(own.near(BAND), common.distance[BAND] <= 0.0)  # Only its own pixels.
    assert own.layer_snap(BAND) == 0.0 and own.layer_snap(TOP) == 6.0
    # The thick layers keep the common snap distance.
    assert np.array_equal(own.fixed & ~band, common.fixed & ~band)


def test_a_proposal_keeps_a_thin_layer_with_snap_zero_exactly():
    image = _image()
    proposal = lp.propose_layers(image, _session(_band_polygons(), band_snap=0.0), frame=FRAME)
    owner = lp.rasterize_polygons(_band_polygons(), SHAPE, (1.0, 1.0))
    assert np.all(proposal.classes[owner == BAND] == BAND)
    assert not np.any(proposal.classes[owner != BAND] == BAND)


def test_the_frame_edge_strip_takes_the_nearest_polygon_layer():
    # The frame reaches beyond the photograph on the left, as for AC17, and
    # the polygons stop 3 px before its border.
    polygons = _band_polygons(left=3.0)
    inside = np.ones(SHAPE, dtype=bool)
    guide = lp.polygon_guide(polygons, SHAPE, (1.0, 1.0), inside, 6.0)
    fill = lp.frame_edge_fill(guide, inside, 5.0)
    strip = (guide.owner == 0) & inside
    assert strip[:, :3].all() and not strip[:, 4:].any()
    assert np.all(fill[:20, :3] == TOP) and np.all(fill[40:, :3] == BOTTOM)
    assert np.all(fill[~strip] == 0)
    assert not lp.frame_edge_fill(guide, inside, 0.0).any()
    assert not lp.frame_edge_fill(guide, inside, 2.0)[:, :1].any()  # Beyond 2 px of a polygon.


def test_a_proposal_fills_the_frame_edge_strip_from_the_polygons():
    image = _image(strip=3)
    polygons = _band_polygons(left=3.0)
    filled = lp.propose_layers(image, _session(polygons), frame=FRAME)
    assert np.all(filled.classes[:20, :3] == TOP) and np.all(filled.classes[40:, :3] == BOTTOM)
    unfilled = lp.propose_layers(image, _session(polygons, edge_fill_mm=0.0), frame=FRAME)
    assert unfilled.digest != filled.digest


def test_a_finer_analysis_grid_is_read_from_the_display_image():
    rng = np.random.default_rng(2)
    display = rng.random((60, 80, 3)).astype(np.float32)
    analysis = display[::2, ::2].copy()
    image = lp.PaintingImage("photo.png", (160, 120), display, analysis)
    fine = lp.with_analysis_width(image, 80)
    assert fine.analysis_shape == (60, 80) and fine.display is display
    assert fine.analysis_scale() == (0.5, 0.5) and fine.image_size == (160, 120)
    assert lp.with_analysis_width(image, 10_000).analysis_shape == (60, 80)  # At most the display.
    assert lp.with_analysis_width(image, 40) is image  # Already that grid.
    half = lp.with_analysis_width(fine, 40)
    assert half.analysis_shape == (30, 40) and half._features is None


def test_layer_snap_edge_fill_and_detail_round_trip_and_enter_the_digest():
    session = _session(_band_polygons(), band_snap=0.0, edge_fill_mm=3.0)
    session.detail = "fine"
    payload = json.loads(json.dumps(session.to_dict()))
    rows = {row["layer_id"]: row for row in payload["layers"]}
    assert rows[BAND]["snap_mm"] == 0.0 and "snap_mm" not in rows[TOP]
    assert payload["proposal_settings"]["edge_fill_mm"] == 3.0
    assert payload["proposal_settings"]["detail"] == "fine"
    payload["format"] = lp.LAYER_PAINTING_FORMAT
    back = lp.PaintingSession.from_dict(payload)
    assert back.layer(BAND).snap_mm == 0.0 and back.layer(TOP).snap_mm is None
    assert back.layer_snap_mm(BAND) == 0.0 and back.layer_snap_mm(TOP) == 6.0
    assert back.edge_fill_mm == 3.0 and back.detail == "fine"
    # A session saved before these settings existed reads the defaults.
    for row in payload["layers"]:
        row.pop("snap_mm", None)
    for key in ("edge_fill_mm", "detail"):
        payload["proposal_settings"].pop(key)
    old = lp.PaintingSession.from_dict(payload)
    assert old.edge_fill_mm == lp.DEFAULT_EDGE_FILL_MM and old.detail == "normal"
    assert all(row.snap_mm is None for row in old.layers)
    payload["proposal_settings"]["detail"] = "unknown"
    assert lp.PaintingSession.from_dict(payload).detail == "normal"
    digest = lp.proposal_digest(session, FRAME)
    session.layer(BAND).snap_mm = 1.0
    assert lp.proposal_digest(session, FRAME) != digest
    session.layer(BAND).snap_mm = 0.0
    session.detail = "normal"
    assert lp.proposal_digest(session, FRAME) != digest


def test_the_exported_label_map_draws_layers_with_snap_zero_exactly():
    # A coarse analysis grid (half the raw size) and one exact layer.
    rng = np.random.default_rng(3)
    raw = np.empty((120, 160, 3), dtype=np.float32)
    raw[:56], raw[56:64], raw[64:] = (0.10, 0.20, 0.55), (0.13, 0.22, 0.52), (0.08, 0.18, 0.58)
    raw = np.clip(raw + rng.normal(0.0, 0.03, raw.shape).astype(np.float32), 0.0, 1.0)
    image = lp.PaintingImage("photo.png", (160, 120), raw, raw[::2, ::2].copy())
    polygons = [
        lp.Polygon(TOP, [(0.0, 0.0), (0.0, 160.0), (56.0, 160.0), (56.0, 0.0)]),
        lp.Polygon(BAND, [(56.0, 0.0), (56.0, 160.0), (63.0, 160.0), (63.0, 0.0)]),
        lp.Polygon(BOTTOM, [(63.0, 0.0), (63.0, 160.0), (120.0, 160.0), (120.0, 0.0)]),
    ]
    frame = RigFrame(
        corners={"top_left": (-0.5, -0.5), "top_right": (-0.5, 159.5),
                 "bottom_right": (119.5, 159.5), "bottom_left": (119.5, -0.5)},
        width_m=0.16, height_m=0.12,
    )
    session = _session(polygons, band_snap=0.0)
    session.image_size = (160, 120)
    proposal = lp.propose_layers(image, session, frame=frame)
    classes, _rows = lp.painted_label_map(session, proposal, frame=frame)
    owner = lp.rasterize_polygons(polygons, (120, 160), (1.0, 1.0))
    assert np.all(classes[owner == BAND] == BAND)  # Its exact outline at full resolution.
