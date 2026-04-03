"""Unit tests for roi_to_mask()."""

import numpy as np
import pytest

import darsia
from darsia import make_coordinate
from darsia.presets.workflows.config.roi import RoiConfig
from darsia.utils.standard_images import roi_to_mask


def _make_scalar_image(rows: int = 10, cols: int = 20) -> darsia.ScalarImage:
    """Helper: create a scalar image with known dimensions.

    The image has default ij-indexing with origin [0, 1.0], so physical
    coordinates are x in [0, 2] (columns) and y in [0, 1] (rows, reversed:
    y=1.0 is at row 0, y=0.0 is at row *rows*).
    """
    arr = np.ones((rows, cols), dtype=np.float32)
    return darsia.ScalarImage(arr, space_dim=2, dimensions=[1.0, 2.0])


def _make_roi(corner_1: list, corner_2: list, name: str = "roi") -> RoiConfig:
    roi = RoiConfig()
    roi.roi = make_coordinate([corner_1, corner_2])
    roi.name = name
    return roi


class TestRoiConfigToMaskBasic:
    def test_returns_boolean_scalar_image(self):
        img = _make_scalar_image()
        roi = _make_roi([0.5, 0.3], [1.5, 0.7])
        mask = roi_to_mask(roi.roi, img)
        assert isinstance(mask, darsia.ScalarImage)
        assert mask.img.dtype == np.bool_

    def test_same_shape_as_reference(self):
        img = _make_scalar_image(rows=8, cols=16)
        roi = _make_roi([0.0, 0.0], [2.0, 1.0])
        mask = roi_to_mask(roi.roi, img)
        assert mask.img.shape == img.img.shape

    def test_interior_roi_sets_correct_pixels(self):
        """Pixels inside the bounding box are True; outside are False."""
        img = _make_scalar_image()
        # corner_1 = (x=0.5, y=0.3), corner_2 = (x=1.5, y=0.7)
        roi = _make_roi([0.5, 0.3], [1.5, 0.7])
        mask = roi_to_mask(roi.roi, img)

        # Determine expected pixel ranges via the coordinate system
        cs = img.coordinatesystem
        v = cs.voxel(roi.roi)
        r_min = max(0, int(np.min(v[:, 0])))
        r_max = min(int(np.max(v[:, 0])), img.num_voxels[0])
        c_min = max(0, int(np.min(v[:, 1])))
        c_max = min(int(np.max(v[:, 1])), img.num_voxels[1])

        expected_count = (r_max - r_min) * (c_max - c_min)
        assert np.sum(mask.img) == expected_count

        # All True pixels must be inside [r_min:r_max, c_min:c_max]
        true_rows, true_cols = np.where(mask.img)
        assert np.all(true_rows >= r_min)
        assert np.all(true_rows < r_max)
        assert np.all(true_cols >= c_min)
        assert np.all(true_cols < c_max)

    def test_full_image_roi_sets_all_pixels(self):
        """An ROI covering the entire image should produce an all-True mask."""
        img = _make_scalar_image()
        # corners at the image's physical extent
        roi = _make_roi(
            img.coordinatesystem._coordinate_of_origin_voxel.tolist(),
            img.coordinatesystem._coordinate_of_opposite_voxel.tolist(),
        )
        mask = roi_to_mask(roi.roi, img)
        assert np.all(mask.img)


class TestRoiConfigToMaskClipping:
    def test_roi_partially_outside_clipped_to_image_bounds(self):
        """A partially out-of-bounds ROI should still produce a valid mask."""
        img = _make_scalar_image()
        # corner_1 and corner_2 extend beyond the image boundary on all sides
        roi = _make_roi([-1.0, -0.5], [3.0, 1.5])
        mask = roi_to_mask(roi.roi, img)

        # The mask should have the same shape and contain at least some True pixels
        assert mask.img.shape == img.img.shape
        assert np.any(mask.img)
        # No pixels outside image bounds can be True
        assert mask.img.shape[0] == img.num_voxels[0]
        assert mask.img.shape[1] == img.num_voxels[1]

    def test_roi_completely_outside_produces_no_true_pixels(self):
        """An ROI entirely outside the image should yield an all-False mask
        (or at most clip to zero pixels).
        """
        img = _make_scalar_image()
        # Both corners have y > 1, which is above the top of the image (row < 0)
        # The coordinate system: y=1.0 is top (row=0), y>1.0 is above image
        # x is always in [0, 2] so cols are fine
        # Use y > 1.0 for both corners to go entirely above the image
        roi = _make_roi([0.0, 1.1], [2.0, 1.5])
        mask = roi_to_mask(roi.roi, img)
        # All voxel rows will be negative -> clipped to [0:0] or [0:negative]
        assert mask.img.shape == img.img.shape
        # At worst, the slice is mask[0:0, ...] which sets nothing
        assert not np.any(mask.img)


class TestRoiConfigToMaskUnion:
    def test_union_of_two_rois(self):
        """OR-combining two masks covers both ROI areas."""
        img = _make_scalar_image()
        roi_a = _make_roi([0.0, 0.5], [0.9, 1.0], name="left")
        roi_b = _make_roi([1.1, 0.0], [2.0, 0.5], name="right")

        mask_a = roi_to_mask(roi_a.roi, img)
        mask_b = roi_to_mask(roi_b.roi, img)
        union = darsia.zeros_like(img, dtype=np.bool_)
        union.img |= mask_a.img
        union.img |= mask_b.img

        assert np.sum(union.img) == np.sum(mask_a.img) + np.sum(mask_b.img)

    def test_overlapping_union(self):
        """Overlapping ROIs do not double-count pixels in the union."""
        img = _make_scalar_image()
        roi_a = _make_roi([0.0, 0.3], [1.5, 0.7], name="a")
        roi_b = _make_roi([0.5, 0.3], [2.0, 0.7], name="b")

        mask_a = roi_to_mask(roi_a.roi, img)
        mask_b = roi_to_mask(roi_b.roi, img)
        union = darsia.zeros_like(img, dtype=np.bool_)
        union.img |= mask_a.img
        union.img |= mask_b.img

        # union pixels <= sum of individual masks (overlap counted once)
        assert np.sum(union.img) <= np.sum(mask_a.img) + np.sum(mask_b.img)
        # union pixels >= each individual mask
        assert np.sum(union.img) >= np.sum(mask_a.img)
        assert np.sum(union.img) >= np.sum(mask_b.img)
