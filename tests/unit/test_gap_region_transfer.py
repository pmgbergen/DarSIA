"""Categorical placement suggestions, never outside-analysis approval fixtures."""
from types import MappingProxyType

import numpy as np
import pytest

from darsia.fluidflower.painted_gap_region_transfer import transfer_region_masks
from darsia.fluidflower.painted_gap_regions import REGION_CLASSIFICATIONS


def _case():
    component = np.zeros((6, 8), bool)
    component[1:3, 1:5] = True
    masks = {name: np.zeros_like(component) for name in REGION_CLASSIFICATIONS}
    masks["air"][1:3, 1:3] = True
    masks["free_water"][1:3, 3:5] = True
    masks["free_water"][2, 4] = False
    masks["sand"][2, 4] = True
    target = np.zeros((8, 12), bool)
    target[2:4, 3:7] = True
    target[6, 11] = True
    return masks, component, target


def test_own_placements_move_each_class_and_leave_unmatched_target_pixels_unmarked():
    masks, component, target = _case()
    originals = {key: value.copy() for key, value in masks.items()}
    source_before, target_before = component.copy(), target.copy()
    source_h = np.eye(3).tolist()
    target_h = [[1., 0., -2.], [0., 1., -1.], [0., 0., 1.]]
    result = transfer_region_masks(MappingProxyType(masks), component, source_h, target, target_h)
    for name, source in masks.items():
        expected = np.zeros(target.shape, bool)
        expected[2:4, 3:7] = source[1:3, 1:5]
        np.testing.assert_array_equal(result["class_masks"][name], expected)
        np.testing.assert_array_equal(source, originals[name])
        with pytest.raises(ValueError):
            result["class_masks"][name].setflags(write=True)
    assert result["matched_pixels"] == 8 and result["remaining_pixels"] == 1
    assert np.argwhere(result["unmarked_mask"]).tolist() == [[6, 11]]
    assert result["class_masks"]["sand"][3, 6]
    np.testing.assert_array_equal(component, source_before)
    np.testing.assert_array_equal(target, target_before)


def test_carrier_edge_replication_never_invents_classifications_outside_source_support():
    source = np.ones((2, 2), bool)
    target = np.ones((4, 7), bool)
    masks = {name: np.zeros_like(source) for name in REGION_CLASSIFICATIONS}
    masks["air"][:] = True
    result = transfer_region_masks(masks, source, np.eye(3), target, np.eye(3))
    expected = np.zeros(target.shape, bool)
    expected[:2, :2] = True
    np.testing.assert_array_equal(result["class_masks"]["air"], expected)
    assert result["matched_pixels"] == 4 and result["remaining_pixels"] == 24


def test_noncommuting_placements_use_destination_inverse_times_source():
    masks, component, _ = _case()
    target = np.ones((15, 19), bool)
    source_h = [[2., 0., 1.], [0., 2., 1.], [0., 0., 1.]]
    target_h = [[1., 0., -3.], [0., 1., -2.], [0., 0., 1.]]
    result = transfer_region_masks(masks, component, source_h, target, target_h)
    np.testing.assert_allclose(result["source_to_target_h"],
        [[2., 0., 4.], [0., 2., 3.], [0., 0., 1.]], rtol=0, atol=0)
    assert result["class_masks"]["air"][5, 6]
    assert result["class_masks"]["free_water"][5, 10]
    assert result["class_masks"]["sand"][7, 12]
    assert not any(mask[0, 0] for mask in result["class_masks"].values())


@pytest.mark.parametrize("fault", ["overlap", "outside", "dtype", "shape", "palette", "empty", "singular", "nonfinite"])
def test_invalid_source_classes_or_placements_refuse_without_changing_the_target(fault):
    masks, component, target = _case()
    before = target.copy()
    source_h = np.eye(3)
    if fault == "overlap": masks["sand"][1, 1] = True
    elif fault == "outside": masks["air"][0, 0] = True
    elif fault == "dtype": masks["air"] = masks["air"].astype(np.uint8)
    elif fault == "shape": masks["air"] = np.zeros((1, 1), bool)
    elif fault == "palette": masks.pop("other")
    elif fault == "empty": masks = {name: np.zeros_like(component) for name in REGION_CLASSIFICATIONS}
    elif fault == "singular": source_h[:] = 0
    elif fault == "nonfinite": source_h[0, 0] = np.nan
    with pytest.raises(ValueError):
        transfer_region_masks(masks, component, source_h, target, np.eye(3))
    np.testing.assert_array_equal(target, before)
