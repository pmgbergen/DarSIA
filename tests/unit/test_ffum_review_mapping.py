"""Exact legacy coordinates, inverse support and physical-scale diagnostics."""
import numpy as np
import pytest

from darsia.fluidflower.phase.ffum_review_mapping import (
    UmReviewMapping,
    annotation_content_outline,
)


def mapping(*, nonlinear=False):
    y, x = np.mgrid[:60, :80].astype(float)
    px = 2*x + (.003*y*y if nonlinear else 0)
    py = 1.5*y + (.001*x*x if nonlinear else 0)
    return UmReviewMapping(np.stack((py, px)).reshape(2, -1), (60, 80),
        np.array([[1., .03, 4.], [0., 1., -3.]]), (640, 480), (320, 240), (-19, -11))


def test_pixel_centres_inverse_drift_resize_and_canvas_origin():
    m = mapping()
    q = np.array([[0., 0.], [12.25, 21.75], [79., 59.]])
    registered = q * [2, 1.5]
    resized_y = registered[:, 1] + 3
    resized_x = registered[:, 0] - .03*resized_y - 4
    expected = (np.column_stack((resized_x, resized_y))+.5)*2-.5+[19, 11]
    np.testing.assert_allclose(m.content_to_display(q), expected, atol=1e-12)
    np.testing.assert_allclose(m.display_to_content(expected), q, atol=1e-10)


def test_nonlinear_roundtrip_and_outside_not_assumed_affine():
    m = mapping(nonlinear=True)
    rng = np.random.default_rng(1729)
    q = rng.uniform([0, 0], [79, 59], (1500, 2))
    image = m.content_to_display(q)
    recovered = m.display_to_content(image, tolerance=1e-7, chunk_size=127)
    np.testing.assert_allclose(recovered, q, atol=1e-7)
    assert np.isnan(m.display_to_content([[1e6, 1e6], [np.nan, 0.]])).all()
    assert np.isnan(m.content_to_display([[-1, 10], [10, 60]])).all()


def test_jacobian_retains_measured_anisotropy_and_units():
    m = mapping()
    q = np.array([[12.25, 21.75], [79., 59.]])
    _, jac = m.content_to_display(q, jacobian=True)
    expected = np.array([[4., -.09], [0., 3.]])
    np.testing.assert_allclose(jac, np.broadcast_to(expected, jac.shape), atol=1e-12)
    metric = m.physical_jacobian(q, content_spacing_yx_m=(.002, .001))
    np.testing.assert_allclose(metric[0], expected/[.001, .002], atol=1e-9)
    with pytest.raises(ValueError, match="physical axes"):
        m.physical_jacobian(q, content_spacing_yx_m=(0, .001))


def test_folded_grid_is_rejected_before_inverse():
    y, x = np.mgrid[:10, :10].astype(float)
    m = UmReviewMapping(np.stack((y, -x)).reshape(2, -1), (10, 10),
        np.eye(3)[:2], (10, 10), (10, 10), (0, 0))
    with pytest.raises(ValueError, match="folds"):
        m.display_to_content([[1, 1]])


def test_original_box_and_polygon_use_canvas_padding_before_mapping():
    payload = dict(id="unchanged", x=.25, y=.2, width=.1, height=.4, points=[])
    outline = annotation_content_outline(payload, canvas_wh=(200, 100),
                                         content_box_xywh=(50, 10, 100, 80), step=3)
    np.testing.assert_allclose(outline.min(0), [0, 10])
    np.testing.assert_allclose(outline.max(0), [20, 50])
    assert payload["id"] == "unchanged"
    polygon = dict(payload, points=[[.25,.2],[.35,.2],[.25,.6]])
    outline = annotation_content_outline(polygon, canvas_wh=(200, 100),
                                         content_box_xywh=(50, 10, 100, 80), step=3)
    assert np.all(outline[:, 0]/20+(outline[:, 1]-10)/40 <= 1+1e-12)
