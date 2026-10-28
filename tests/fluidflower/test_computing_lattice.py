"""The computing lattice: storage's cells, image orientation and per-cell reductions."""
import math

import numpy as np
import pytest

from darsia.fluidflower import computing_lattice as cl
from darsia.fluidflower.storage_resolution import make_grid, metric_grid_shape

EXTENT = (-0.002, 0.012, -0.003, 0.009)


def rotated_canvas(shape=(40, 52), pitch=0.00025, angle=0.02, origin=(-0.0015, 0.0085)):
    """Pixel-centre metre coordinates of a slightly rotated synthetic canvas; rows run down."""
    rows, columns = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    c, s = math.cos(angle), math.sin(angle)
    x = origin[0] + pitch * (c * columns + s * rows)
    y = origin[1] - pitch * (-s * columns + c * rows)
    return x.astype(np.float32), y.astype(np.float32)


@pytest.fixture
def lattice():
    return cl.computing_lattice(EXTENT, 1.0)


@pytest.fixture
def assignment(lattice):
    x, y = rotated_canvas()
    return cl.CellAssignment(lattice, x, y, np.full(x.shape, 0.0625))


@pytest.mark.parametrize("good", [0.5, 1, 1.0])
def test_computing_resolution_admits_half_and_one_millimetre(good):
    assert cl.computing_resolution_mm(good) == float(good)


@pytest.mark.parametrize("bad", ["native", 0.25, 2.0, True, float("nan"), None])
def test_other_computing_resolutions_are_refused(bad):
    with pytest.raises(ValueError, match="0.5 or 1 mm"):
        cl.computing_resolution_mm(bad)


def test_the_lattice_has_the_storage_shape_and_refuses_an_off_lattice_extent(lattice):
    assert lattice.shape == metric_grid_shape(EXTENT, 1.0) == (12, 14)
    assert cl.computing_lattice(EXTENT, 0.5).shape == (24, 28)
    with pytest.raises(ValueError, match="origin-zero metric lattice"):
        cl.computing_lattice((-0.0025, 0.012, -0.003, 0.009), 1.0)
    with pytest.raises(ValueError, match="finite metric extent"):
        cl.computing_lattice((0, 1, 2), 1.0)


def test_cell_coordinates_put_integer_cells_at_their_centres_top_row_first(lattice):
    forward, backward = lattice.metric_to_cell(), lattice.cell_to_metric()
    np.testing.assert_allclose(forward @ backward, np.eye(3), atol=1e-12)
    x, y = lattice.cell_centres_m()
    assert x[0, 0] == pytest.approx(-0.0015) and y[0, 0] == pytest.approx(0.0085)
    assert y[-1, 0] == pytest.approx(-0.0025) and x[0, -1] == pytest.approx(0.0115)
    cells = forward @ np.array([x[3, 5], y[3, 5], 1.0])
    np.testing.assert_allclose(cells[:2] / cells[2], [5, 3], atol=1e-9)
    assert lattice.spacing_m == 0.001
    flipped = lattice.to_storage(np.arange(12 * 14).reshape(12, 14))
    assert flipped[0, 0] == 11 * 14
    with pytest.raises(ValueError, match="does not lie"):
        lattice.to_storage(np.zeros((3, 3)))
    meta = lattice.metadata()
    assert meta["resolution_mm"] == 1.0 and meta["shape"] == [12, 14]
    assert meta["mask_fraction"] == cl.MASK_FRACTION == 0.5


def test_cells_are_storage_cells_flipped_to_image_orientation(lattice, assignment):
    x, y = rotated_canvas()
    grid = make_grid(x, y, np.full(x.shape, 0.0625), include=np.ones(x.shape, bool), extent_m=EXTENT,
                     storage_resolution_mm=1.0, precision="float64")
    np.testing.assert_array_equal(assignment.cells, grid.probability_cell_index)
    storage_count = grid.cell_probability_pixel_count.reshape(grid.shape)
    np.testing.assert_array_equal(assignment.pixel_count, storage_count[::-1])
    assert assignment.pixel_count.sum() == x.size


def test_sums_fractions_and_masks_keep_area(lattice, assignment):
    values = np.arange(40 * 52, dtype=np.float64).reshape(40, 52)
    total = assignment.total(values)
    assert total.sum() == pytest.approx(values.sum())
    mask = np.zeros((40, 52), bool)
    mask[:, :26] = True
    fraction = assignment.fraction(mask)
    assert ((fraction >= 0) & (fraction <= 1)).all()
    np.testing.assert_allclose((fraction * assignment.cell_area).sum(), mask.sum() * 0.0625)
    np.testing.assert_array_equal(assignment.mask(mask), fraction >= 0.5)
    with pytest.raises(ValueError, match="boolean"):
        assignment.fraction(mask.astype(np.uint8))
    with pytest.raises(ValueError, match="finite"):
        assignment.total(np.full((40, 52), np.nan))


def test_means_weight_valid_area_and_reject_cells_below_half_valid(lattice, assignment):
    values = np.full((40, 52), 3.0)
    valid = np.ones((40, 52), bool)
    valid[:, 40:] = False
    values[:, 40:] = np.nan
    mean, fraction = assignment.mean(values, valid)
    covered = fraction >= 0.5
    np.testing.assert_array_equal(np.isfinite(mean), covered & (assignment.cell_area > 0))
    np.testing.assert_allclose(mean[np.isfinite(mean)], 3.0)
    weighted = np.where(valid, np.linspace(1, 2, 52)[None, :], 0.0)
    mean, _ = assignment.mean(weighted, valid)
    inside = (fraction == 1.0)
    expected = assignment.total(weighted)[inside] / assignment.total(valid.astype(float))[inside]
    np.testing.assert_allclose(mean[inside], expected)
    with pytest.raises(ValueError, match="finite"):
        assignment.mean(values, np.ones((40, 52), bool))


def test_majority_takes_the_largest_area_and_the_lowest_id_on_ties(lattice):
    x, y = np.meshgrid(np.arange(4) * 0.00025 + 0.000125, 0.000875 - np.arange(4) * 0.00025)
    one_cell = cl.CellAssignment(cl.computing_lattice((0.0, 0.001, 0.0, 0.001), 1.0),
                                 x.astype(np.float32), y.astype(np.float32), np.ones((4, 4)))
    categories = np.array([[7, 7, 3, 3]] * 4, np.uint8)
    assert one_cell.majority(categories)[0, 0] == 3
    categories[0, 2] = 7
    assert one_cell.majority(categories)[0, 0] == 7
    where = np.zeros((4, 4), bool)
    assert one_cell.majority(categories, where=where)[0, 0] == 0
    where[0, 0] = True
    assert one_cell.majority(categories, where=where)[0, 0] == 7
    with pytest.raises(ValueError, match="integer"):
        one_cell.majority(categories.astype(float))


def test_log_rgb_averages_linear_values_and_marks_invalid_cells_nan(lattice, assignment):
    linear = np.stack([np.full((40, 52), value) for value in (100.0, 400.0, 900.0)])
    linear[0, :, ::2] = 300.0
    log = np.log(linear).astype(np.float32)
    valid = np.ones((40, 52), bool)
    valid[:, 44:] = False
    log[:, ~valid] = np.nan
    result, good, fraction = assignment.log_rgb(log, valid)
    assert result.dtype == np.float32 and result.shape == (3, 12, 14) and fraction.dtype == np.float32
    np.testing.assert_array_equal(good, np.isfinite(result).all(0))
    np.testing.assert_array_equal(good, (fraction >= 0.5) & (assignment.cell_area > 0))
    np.testing.assert_allclose(np.exp(result[1][good]), 400.0, rtol=1e-6)
    # Channel 0 averages 100 and 300 in linear values, not in logarithms.
    interior = good & (assignment.pixel_count >= 16)
    assert np.exp(result[0][interior]).mean() == pytest.approx(200.0, rel=0.05)
    assert not np.isclose(np.exp(result[0][interior]).mean(), math.sqrt(100 * 300), rtol=0.01)
    with pytest.raises(ValueError, match="CHW float32"):
        assignment.log_rgb(log.astype(np.float64), valid)
    with pytest.raises(ValueError, match="finite pixels"):
        assignment.log_rgb(log, np.ones((40, 52), bool))


def test_a_pixel_outside_the_extent_fails_closed_as_for_storage(lattice):
    x, y = rotated_canvas(origin=(-0.0035, 0.0085))
    with pytest.raises(ValueError, match="outside declared half-open metric extent"):
        cl.CellAssignment(lattice, x, y, np.ones(x.shape))
    x, y = rotated_canvas()
    with pytest.raises(ValueError, match="positive and finite"):
        cl.CellAssignment(lattice, x, y, np.zeros(x.shape))


def test_canvas_pitch_is_the_square_root_of_the_mapped_pixel_area():
    affine = np.array([[0.0002, 0.0, 0.01], [0.0, -0.0003, 0.5], [0.0, 0.0, 1.0]])
    assert cl.canvas_pitch_m(affine, (30, 40)) == pytest.approx(math.sqrt(0.0002 * 0.0003))
    with pytest.raises(ValueError, match="no area"):
        cl.canvas_pitch_m(np.diag([0.0, 1.0, 1.0]), (3, 3))
