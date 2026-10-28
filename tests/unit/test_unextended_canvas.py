"""An independent full-photograph source keeps its centre and leaves the ring unsupported."""
import numpy as np
import pytest

from darsia.fluidflower.pipeline.series import unextended_canvas

PARAMETERS = {"extension": dict(native_shape_hw=[3, 4], minimum_canvas_hw=[5, 7],
                                native_origin_xy=[-2, -1], contract="ring-contract")}


def centre():
    return np.arange(36, dtype=np.float32).reshape(3, 3, 4) / 10


def test_centre_bytes_stay_and_ring_pixels_are_unsupported():
    eligible = np.zeros((5, 7), bool)
    eligible[1:4, 2:6] = True
    eligible[2, 0] = True  # one rig pixel left of the native box
    canvas, valid, audit = unextended_canvas(centre(), eligible, parameters=PARAMETERS)
    assert canvas.dtype == np.float32 and canvas.shape == (3, 5, 7)
    assert canvas[:, 1:4, 2:6].tobytes() == centre().tobytes()
    assert np.isnan(canvas[:, 2, 0]).all()
    assert valid[1:4, 2:6].all() and valid.sum() == 12
    assert audit["requested_ring_pixels"] == 1
    assert audit["finite_ring_pixels"] == 0 and audit["unsupported_ring_pixels"] == 1
    assert audit["windows"] == [] and audit["native_centre_bytes_preserved"] is True
    assert audit["contract"] == "ring-contract" and audit["canvas_hw"] == [5, 7]


def test_without_ring_pixels_nothing_is_requested():
    eligible = np.zeros((5, 7), bool)
    eligible[1:4, 2:6] = True
    _, _, audit = unextended_canvas(centre(), eligible, parameters=PARAMETERS)
    assert audit["requested_ring_pixels"] == 0 and audit["unsupported_ring_pixels"] == 0


def test_wrong_centre_or_canvas_is_refused():
    eligible = np.zeros((5, 7), bool)
    with pytest.raises(TypeError):
        unextended_canvas(centre().astype(np.float64), eligible, parameters=PARAMETERS)
    with pytest.raises(ValueError):
        unextended_canvas(centre(), np.zeros((4, 7), bool), parameters=PARAMETERS)
