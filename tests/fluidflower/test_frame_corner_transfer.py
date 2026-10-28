"""Frame corner transfer between camera positions on a synthetic rig."""
import math

import cv2
import numpy as np
import pytest

from darsia.fluidflower.tools.frame_corner_transfer import (
    edge_magnitude,
    fit_similarity,
    transfer_corners,
)

CORNERS = {"top_left": (120.0, 90.0), "top_right": (112.0, 1510.0),
           "bottom_right": (980.0, 1500.0), "bottom_left": (985.0, 100.0)}


def _rig_photo(seed=0):
    """A dark interior with sand texture, a bright metal frame and some fittings."""
    rng = np.random.default_rng(seed)
    img = np.full((1100, 1600), 40.0, np.float32)
    pts = np.array([[c, r] for r, c in (CORNERS["top_left"], CORNERS["top_right"],
                                       CORNERS["bottom_right"], CORNERS["bottom_left"])], np.int32)
    cv2.polylines(img, [pts], True, 220.0, 14)
    cv2.fillPoly(img, [pts], 60.0)
    cv2.polylines(img, [pts], True, 220.0, 14)
    inner = img.copy()
    inner[200:900, 200:1400] += rng.normal(0, 12, (700, 1200)).astype(np.float32)
    cv2.circle(inner, (400, 300), 25, 150.0, -1)
    cv2.rectangle(inner, (1200, 180), (1380, 300), 180.0, -1)
    return np.clip(inner, 0, 255)


def _warp(img, scale, rot_deg, dx, dy):
    """Similarity warp about the origin, matching ``SimilarityFit.apply``."""
    th = math.radians(rot_deg)
    M = np.array([[scale * math.cos(th), -scale * math.sin(th), dx],
                  [scale * math.sin(th), scale * math.cos(th), dy]], np.float32)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=40.0)


def _expected(scale, rot_deg, dx, dy):
    th = math.radians(rot_deg)
    out = {}
    for k, (r, c) in CORNERS.items():
        x = scale * (math.cos(th) * c - math.sin(th) * r) + dx
        y = scale * (math.sin(th) * c + math.cos(th) * r) + dy
        out[k] = (y, x)
    return out


def test_fit_similarity_recovers_parameters():
    target = _expected(1.004, -0.15, 12.0, -7.0)
    fit = fit_similarity(CORNERS, target)
    assert fit.scale == pytest.approx(1.004, abs=1e-6)
    assert fit.rotation_deg == pytest.approx(-0.15, abs=1e-6)
    assert fit.translation_col == pytest.approx(12.0, abs=1e-6)
    assert fit.translation_row == pytest.approx(-7.0, abs=1e-6)
    assert max(fit.residual_px.values()) < 1e-6


def test_same_camera_position_reproduces_corners():
    ref = edge_magnitude(_rig_photo(0))
    tgt = edge_magnitude(_rig_photo(1) + np.float32(5.0))  # other noise, brighter lamps
    res = transfer_corners(ref, tgt, CORNERS, inward=120, outward=20, search=40, refine_search=15)
    for k, (r, c) in CORNERS.items():
        m = res.corners[k]
        assert m.status == "matched"
        assert abs(m.row - r) < 1.0 and abs(m.col - c) < 1.0


def test_shifted_rotated_camera_is_recovered():
    ref_img = _rig_photo(0)
    scale, rot, dx, dy = 1.003, -0.2, -18.0, -6.0
    tgt_img = _warp(_rig_photo(2) * 0.8, scale, rot, dx, dy)
    res = transfer_corners(edge_magnitude(ref_img), edge_magnitude(tgt_img), CORNERS,
                           inward=120, outward=20, search=60, refine_search=15)
    expected = _expected(scale, rot, dx, dy)
    for k in CORNERS:
        m = res.corners[k]
        assert m.status == "matched", k
        assert abs(m.row - expected[k][0]) < 1.5, k
        assert abs(m.col - expected[k][1]) < 1.5, k
    assert res.fit.rotation_deg == pytest.approx(rot, abs=0.02)
    assert res.fit.scale == pytest.approx(scale, abs=5e-4)
    assert max(res.fit.residual_px.values()) < 1.5


def test_hidden_corner_is_predicted_and_flagged():
    ref_img = _rig_photo(0)
    tgt_img = _warp(_rig_photo(3), 1.0, 0.0, 9.0, -4.0)
    tgt_img[0:260, 0:260] = 40.0  # top-left corner hidden (a cable, a lamp reflection)
    res = transfer_corners(edge_magnitude(ref_img), edge_magnitude(tgt_img), CORNERS,
                           inward=120, outward=20, search=60, refine_search=15)
    tl = res.corners["top_left"]
    assert tl.status == "predicted"
    expected = _expected(1.0, 0.0, 9.0, -4.0)["top_left"]
    assert abs(tl.row - expected[0]) < 2.0 and abs(tl.col - expected[1]) < 2.0
    assert all(res.corners[k].status == "matched" for k in ("top_right", "bottom_right", "bottom_left"))
    d = res.as_dict()
    assert d["corners"]["top_left"]["status"] == "predicted" and "fit" in d


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError):
        transfer_corners(np.zeros((10, 10), np.float32), np.zeros((10, 12), np.float32), CORNERS)


def test_degenerate_case_is_flagged_not_fitted_on_bad_matches():
    ref_img = _rig_photo(0)
    tgt_img = np.full_like(ref_img, 40.0)  # nothing to match
    res = transfer_corners(edge_magnitude(ref_img), edge_magnitude(tgt_img), CORNERS,
                           inward=120, outward=20, search=40, refine_search=15)
    assert res.fit.degenerate
    assert not res.usable
    assert res.confidence == -1.0
    assert all(c.status == "unresolved" for c in res.corners.values())
    assert res.fit.scale == 1.0 and res.fit.rotation_deg == 0.0


def test_usable_requires_three_matched_corners():
    ref_img = _rig_photo(0)
    tgt_img = _warp(_rig_photo(3), 1.0, 0.0, 9.0, -4.0)
    tgt_img[0:260, 0:260] = 40.0
    res = transfer_corners(edge_magnitude(ref_img), edge_magnitude(tgt_img), CORNERS,
                           inward=120, outward=20, search=60, refine_search=15)
    assert res.usable and res.confidence >= 0.9
    assert set(res.fit.fitted_on) == {"top_right", "bottom_right", "bottom_left"}


def test_fit_similarity_rejects_degenerate_points():
    with pytest.raises(ValueError):
        fit_similarity({"a": (0.0, 0.0)}, {"a": (1.0, 1.0)})
    with pytest.raises(ValueError):
        fit_similarity({"a": (5.0, 5.0), "b": (5.0, 5.0)}, {"a": (1.0, 1.0), "b": (2.0, 2.0)})
    with pytest.raises(ValueError):
        fit_similarity({"a": (0.0, 0.0), "b": (10.0, 0.0)}, {"a": (1.0, 1.0), "b": (1.0, 1.0)})
    with pytest.raises(ValueError):
        fit_similarity({"a": (0.0, 0.0), "b": (10.0, 0.0)}, {"a": (float("nan"), 1.0), "b": (2.0, 2.0)})


def test_usable_requires_a_plausible_scale():
    from darsia.fluidflower.tools.frame_corner_transfer import CornerMatch, SimilarityFit, TransferResult
    corners = {k: CornerMatch(k, 0.0, 0.0, 0.95, "matched", 0.0, 0.0) for k in ("a", "b", "c")}
    good = TransferResult(corners, SimilarityFit(1.002, -0.1, 3.0, 2.0, {}, ("a", "b", "c"), False))
    assert good.usable
    shrunk = TransferResult(corners, SimilarityFit(0.1, 0.0, 0.0, 0.0, {}, ("a", "b", "c"), False))
    assert not shrunk.usable
    nan_fit = TransferResult(corners, SimilarityFit(float("nan"), 0.0, 0.0, 0.0, {}, ("a", "b", "c"), False))
    assert not nan_fit.usable
