"""Synthetic metadata edge cases; no images, jobs or series are opened."""
import copy
import numpy as np
import pytest

from darsia.fluidflower.storage_grid_metadata import build_origin_grid
from darsia.fluidflower.storage_resolution import make_grid, metric_grid_shape


@pytest.mark.parametrize("resolution", [0.25, 0.5, 1, 2])
def test_exact_maximum_adds_one_cell_on_both_axes(resolution):
    step = resolution / 1000
    grid = build_origin_grid([[0, step, 0, step]], resolution)
    assert grid["integer_edge_indices_xy"] == [0, 2, 0, 2]
    assert grid["shape_hw"] == [2, 2]
    assert grid["upper_edge_ties_xy"] == [True, True]
    assert metric_grid_shape(grid["extent_m"], resolution) == (2, 2)


@pytest.mark.parametrize("extent,indices,shape,ties", [
    ([-0.002, -0.001, -0.003, -0.002], [-2, 0, -3, -1], [2, 2], [True, True]),
    ([-0.0011, -0.0001, 0.0002, 0.0019], [-2, 0, 0, 2], [2, 2], [False, False]),
    ([0.3, 0.3, -0.3, -0.3], [300, 301, -300, -299], [1, 1], [True, True]),
    ([0, 0, 0, 0], [0, 1, 0, 1], [1, 1], [True, True]),
])
def test_negative_decimal_and_single_point_bounds(extent, indices, shape, ties):
    actual = build_origin_grid([extent], 1)
    assert actual["integer_edge_indices_xy"] == indices
    assert actual["shape_hw"] == shape
    assert actual["upper_edge_ties_xy"] == ties


def test_union_is_fixed_and_input_metadata_is_unchanged():
    extents = [[-0.0021, 0.002, 0, 0.001], [-0.001, 0.001, -0.002, 0.0021]]
    before = copy.deepcopy(extents)
    actual = build_origin_grid(extents, 1)
    assert extents == before
    assert actual["source_extent_count"] == 2
    assert actual["integer_edge_indices_xy"] == [-3, 3, -2, 3]
    assert actual["upper_edge_ties_xy"] == [True, False]


@pytest.mark.parametrize("resolution,indices,shape", [
    (0.25, [-337, 3913, -45, 2400], [2445, 4250]),
    (0.5, [-169, 1957, -23, 1200], [1223, 2126]),
    (1, [-85, 979, -12, 600], [612, 1064]),
    (2, [-43, 490, -6, 300], [306, 533]),
])
def test_recorded_non_tied_catalogue_envelope_keeps_existing_extent(resolution, indices, shape):
    # Literal previously read metadata, not a new per-pixel coverage proof.
    envelope = [-0.08413322568019735, 0.9781983127173667,
                -0.011016017632171022, 0.5998311927693274]
    actual = build_origin_grid([envelope], resolution)
    assert actual["integer_edge_indices_xy"] == indices
    assert actual["shape_hw"] == shape
    assert actual["upper_edge_ties_xy"] == [False, False]


def test_native_does_not_inspect_extents():
    class MustNotRead:
        def __iter__(self):
            raise AssertionError("Native mode inspected extent data")
    assert build_origin_grid(MustNotRead(), "native") is None


@pytest.mark.parametrize("extent", [None, [], [[0, 1, 0]], [[1, 0, 0, 1]],
    [[0, 1, 1, 0]], [[True, 1, 0, 1]], [["0", 1, 0, 1]],
    [[0, float("nan"), 0, 1]], [[0, 1, 0, float("inf")]],
    [[0, 1, 0, None]], [[0j, 1, 0, 1]], {"bounds": [0, 1, 0, 1]}])
def test_malformed_finite_extents_stop(extent):
    with pytest.raises(ValueError):
        build_origin_grid(extent, 1)


@pytest.mark.parametrize("resolution", [None, True, "1", 0, 0.1, float("inf")])
def test_unsupported_resolution_stops(resolution):
    with pytest.raises(ValueError):
        build_origin_grid([[0, 0.1, 0, 0.1]], resolution)


def test_declared_grid_keeps_original_writer_stop_without_runtime_expansion():
    actual = build_origin_grid([[0, 0.001, 0, 0.001]], 1)
    x = np.array([[0.001]], dtype=np.float64)
    y = np.array([[0.001]], dtype=np.float64)
    kwargs = dict(include=np.array([[True]]), storage_resolution_mm=1)
    with pytest.raises(ValueError, match="outside declared half-open"):
        make_grid(x, y, np.ones((1, 1)), extent_m=[0, 0.001, 0, 0.001], **kwargs)
    result = make_grid(x, y, np.ones((1, 1)), extent_m=actual["extent_m"], **kwargs)
    assert result.shape == (2, 2)
    with pytest.raises(ValueError, match="outside declared half-open"):
        make_grid(np.array([[0.002]]), y, np.ones((1, 1)),
                  extent_m=actual["extent_m"], **kwargs)


def test_original_cell_count_bound_remains_mandatory():
    with pytest.raises(ValueError, match="bounded cell count"):
        build_origin_grid([[0, 100, 0, 100]], 0.25)
