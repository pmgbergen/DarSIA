"""The simulator geometry export (N18) on small rig-corrected canvases with 1 mm cells."""

import io
import json

import numpy as np
import pytest

from darsia.fluidflower.published_dataset import simulator_geometry as geometry
from darsia.fluidflower import storage_resolution

ROLES = {0: "free_water", 1: "porous_medium", 2: "porous_medium", 3: "porous_medium"}
WIDTH, HEIGHT = 0.06, 0.04  # 60 x 40 cells of 1 mm
FIXTURE = {"chart": [(0.020, 0.010), (0.030, 0.010), (0.030, 0.020), (0.020, 0.020)]}
TABLE = [{"facies_id": key, "name": name, "analysis_role": "porous_medium", "porosity": phi, "permeability": 1.0,
          "permeability_unit": "D", "swi": None, "grain_size": None, "source": None}
         for key, name, phi in ((1, "Upper sand", 0.4), (2, "Middle sand", 0.3), (3, "Lower sand", 0.2))]
FLAT_DEPTH = [[-1.0, -1.0, 0.01], [2.0, -1.0, 0.01], [0.5, 2.0, 0.01]]
DECLARED = {"inner_frame": "canvas", "sand_surface": "free_water"}


def layered(*, dip=True):
    """Water above row 5; facies 1, then 2 (a boundary dipping to the right), then 3 from row 30."""
    facies = np.zeros((40, 60), np.int64)
    facies[5:, :] = 1
    for column in range(60):
        facies[15 + (column // 10 if dip else 0):, column] = 2
    facies[30:, :] = 3
    return facies


def test_canvas_coordinates_put_the_origin_at_the_lower_left():
    x, y, area = geometry.canvas_coordinates((40, 60), WIDTH, HEIGHT)
    assert x.shape == y.shape == (40, 60) and area == pytest.approx(1e-6)
    assert x[0, 0] == pytest.approx(0.0005) and x[0, -1] == pytest.approx(0.0595)
    assert y[0, 0] == pytest.approx(0.0395) and y[-1, 0] == pytest.approx(0.0005)
    with pytest.raises(ValueError):
        geometry.canvas_coordinates((0, 60), WIDTH, HEIGHT)


def test_fixture_polygons_in_metres_use_the_production_rasterizer():
    mask = geometry.fixture_mask((40, 60), WIDTH, HEIGHT, FIXTURE)
    rows, columns = np.nonzero(mask)
    assert mask[21:29, 21:29].all() and not mask[:19].any() and not mask[:, :19].any()
    assert 20 <= rows.min() and rows.max() <= 30 and 20 <= columns.min() and columns.max() <= 30
    assert not geometry.fixture_mask((40, 60), WIDTH, HEIGHT, {}).any()
    with pytest.raises(ValueError):
        geometry.fixture_mask((40, 60), WIDTH, HEIGHT, {"line": [(0, 0), (0.01, 0.01)]})


def test_the_domain_runs_from_the_sand_surface_to_the_floor():
    porous = np.zeros((10, 6), bool)
    porous[3:8, :4] = True
    porous[6:8, 5] = True
    water = np.zeros((10, 6), bool)
    water[:3, :3] = True
    frame = np.zeros((10, 6), bool)
    frame[8:, :] = True
    domain, counts = geometry.porous_domain(porous, np.zeros((10, 6), bool), water, frame,
                                            sand_surface="free_water")
    assert counts == {"surface_columns_below_free_water": 3, "surface_columns_at_highest_sand": 2,
                      "surface_columns_from_neighbour": 1}
    assert domain[3:8, :3].all() and not domain[:3, :3].any()  # Below the free water, down to the floor.
    assert domain[3:8, 3].all() and not domain[:3, 3].any()  # No water: from its highest sand cell.
    assert domain[3:8, 4].all()  # No water and no sand: the nearest column's surface.
    assert domain[6:8, 5].all() and not domain[:6, 5].any()
    assert not domain[8:].any()  # The frame is outside.


def test_flat_layers_continue_unchanged_across_a_fixture():
    facies = layered(dip=False)
    completed = geometry.complete_geometry(facies, ROLES, width_m=WIDTH, height_m=HEIGHT, **DECLARED,
                                           fixture_polygons_m=FIXTURE)
    hidden = geometry.fixture_mask((40, 60), WIDTH, HEIGHT, FIXTURE)
    assert np.array_equal(completed.facies, facies)
    assert np.array_equal(completed.fixture_filled, hidden) and not completed.edge_filled.any()
    assert not completed.unfilled.any() and not (completed.original & hidden).any()


def test_a_dipping_boundary_is_interpolated_between_its_two_sides():
    facies = layered()
    completed = geometry.complete_geometry(facies, ROLES, width_m=WIDTH, height_m=HEIGHT, **DECLARED,
                                           fixture_polygons_m=FIXTURE)
    hidden = completed.fixture_filled
    columns = np.flatnonzero(hidden.any(axis=0))
    left, right = columns.min() - 1, columns.max() + 1
    _x, y, _area = geometry.canvas_coordinates((40, 60), WIDTH, HEIGHT)
    side = {column: 0.5 * (y[15 + column // 10 - 1, column] + y[15 + column // 10, column]) for column in (left, right)}
    for row, column in zip(*np.nonzero(hidden)):
        boundary = side[left] + (column - left) / (right - left) * (side[right] - side[left])
        expected = 3 if row >= 30 else (1 if y[row, column] > boundary else 2)  # 2|3 is flat at row 30.
        assert completed.facies[row, column] == expected, (row, column)
    assert np.array_equal(completed.facies[~hidden], facies[~hidden])  # Original cells never change.


def test_a_boundary_seen_on_one_side_only_keeps_its_height():
    facies = layered(dip=False)
    facies[25:30, 40:] = 4  # A lens on the right side only: boundaries 2|4 and 4|3 exist there alone.
    hidden = {"chart": [(0.030, 0.005), (0.040, 0.005), (0.040, 0.030), (0.030, 0.030)]}
    completed = geometry.complete_geometry(facies, {**ROLES, 4: "porous_medium"}, width_m=WIDTH, height_m=HEIGHT,
                                           **DECLARED,
                                           fixture_polygons_m=hidden)
    filled = completed.fixture_filled
    rows = {row: set(completed.facies[row][filled[row]]) for row in range(40) if filled[row].any()}
    assert rows
    for row, values in rows.items():
        expected = 1 if row < 15 else (2 if row < 25 else (4 if row < 30 else 3))
        assert values == {expected}, row


def test_a_fixture_at_the_wall_is_continued_from_its_one_side():
    facies = layered()
    wall = {"chart": [(0.050, 0.010), (0.0605, 0.010), (0.0605, 0.025), (0.050, 0.025)]}
    completed = geometry.complete_geometry(facies, ROLES, width_m=WIDTH, height_m=HEIGHT, **DECLARED,
                                           fixture_polygons_m=wall)
    filled = completed.fixture_filled
    columns = np.flatnonzero(filled.any(axis=0))
    side = columns.min() - 1
    for row, column in zip(*np.nonzero(filled)):
        assert completed.facies[row, column] == facies[row, side]
    assert columns.max() == 59 and not completed.unfilled.any()


def test_edges_are_replicated_to_the_walls_the_floor_and_the_sand_surface():
    truth = layered(dip=False)
    facies = truth.copy()
    unknown = 9  # A facies id without a role: nothing is known there.
    facies[5:7, :] = unknown  # A strip below the free water (the sand surface).
    facies[37:, :] = unknown  # A strip above the floor.
    facies[5:, :3] = unknown  # A strip at the left wall.
    roles = dict(ROLES)
    completed = geometry.complete_geometry(facies, roles, width_m=WIDTH, height_m=HEIGHT, **DECLARED,
                                           fixture_polygons_m={})
    assert not completed.fixture_filled.any() and not completed.unfilled.any()
    assert np.array_equal(completed.edge_filled, facies == unknown)
    assert np.array_equal(completed.facies[5:7, 3:], np.ones((2, 57), np.int64))  # From the sand below.
    assert np.array_equal(completed.facies[37:, 3:], np.full((3, 57), 3))  # From the cell above.
    assert np.array_equal(completed.facies[7:37, :3], truth[7:37, :3])  # From the nearest cell in the row.
    assert set(np.unique(completed.facies[37:, :3])) == {3} and set(np.unique(completed.facies[5:7, :3])) == {1}


def test_a_fixture_role_in_the_map_is_removed_and_filled():
    facies = layered(dip=False)
    facies[20:28, 20:28] = 7
    completed = geometry.complete_geometry(facies, {**ROLES, 7: "colorchecker"}, width_m=WIDTH, height_m=HEIGHT,
                                           **DECLARED)
    assert completed.fixture_filled.sum() == 64 and np.array_equal(completed.facies, layered(dip=False))


def test_grid_products_report_fractions_porosity_depth_and_volumes():
    facies = layered(dip=False)
    npz, text, counts, warnings = geometry.export_geometry(facies, ROLES, TABLE, width_m=WIDTH, height_m=HEIGHT,
                                                           depth_points=FLAT_DEPTH, resolution_mm=2.0,
                                                           fixture_polygons_m=FIXTURE, depth_source="request",
                                                           **DECLARED)
    arrays = np.load(io.BytesIO(npz))
    assert arrays["porosity"].shape == (20, 30) and arrays["facies_ids"].tolist() == [1, 2, 3]
    cell = 0.002 ** 2
    # Grid row 0 is the lowest (y increasing): facies 3 there, facies 1 near the top of the sand.
    assert arrays["porosity"][0, 0] == pytest.approx(0.2) and arrays["facies_majority"][0, 0] == 3
    assert arrays["depth_m"][0, 0] == pytest.approx(0.01)
    assert arrays["pore_volume_m3"][0, 0] == pytest.approx(0.2 * 0.01 * cell, rel=1e-5)
    assert arrays["bulk_volume_m3"][0, 0] == pytest.approx(0.01 * cell, rel=1e-5)
    assert arrays["facies_area_fractions"][:, 0, 0].tolist() == [0.0, 0.0, 1.0]
    sand = arrays["domain_area_fraction"] > 0
    assert np.allclose(arrays["facies_area_fractions"].sum(axis=0)[sand], 1.0)
    top = arrays["domain_area_fraction"][-1]  # The water rows lie above the sand: outside the domain.
    assert np.all(top == 0) and np.isnan(arrays["porosity"][-1]).all()
    porosity = arrays["porosity"][sand]
    assert np.all((porosity >= 0.2 - 1e-6) & (porosity <= 0.4 + 1e-6))
    assert arrays["fixture_filled"].any() and not arrays["edge_filled"].any() and not arrays["unfilled"].any()
    assert np.nanmax(arrays["fixture_filled_area_fraction"]) == pytest.approx(1.0)
    document = json.loads(text)
    assert document["format"] == "darsia-simulator-geometry-v1" and document["counts"] == counts
    assert document["grid"]["shape"] == [20, 30] and document["grid"]["resolution_mm"] == 2.0
    assert [row["porosity"] for row in document["facies_table"]] == [0.4, 0.3, 0.2]
    assert document["depth_points"] == FLAT_DEPTH
    assert document["method"]["fixture_polygons"] == ["chart"]
    assert document["method"]["fixtures_declared"] == "polygons in metres"
    assert document["method"]["depth_points_source"] == "request" and document["warnings"] == warnings
    assert warnings == [f"Facies {key} has no swi, grain_size, source in the series table; published as null"
                        for key in (1, 2, 3)]
    assert set(document["arrays"]) == set(arrays.files)


def test_porosity_is_pore_over_bulk_volume_where_depth_varies():
    facies = layered(dip=False)
    sloped = [[0.0, -1.0, 0.01], [0.0, 2.0, 0.01], [0.06, 0.5, 0.03], [0.06, -1.0, 0.03], [0.06, 2.0, 0.03]]
    npz, _text, _counts, _warnings = geometry.export_geometry(facies, ROLES, TABLE, width_m=WIDTH, height_m=HEIGHT,
                                                              depth_points=sloped, resolution_mm=2.0,
                                                              fixture_polygons_m={}, **DECLARED)
    arrays = np.load(io.BytesIO(npz))
    sand = arrays["domain_area_fraction"] > 0
    ratio = arrays["pore_volume_m3"][sand] / arrays["bulk_volume_m3"][sand]
    assert np.allclose(arrays["porosity"][sand], ratio)
    assert arrays["depth_m"][0, -1] > arrays["depth_m"][0, 0]


@pytest.mark.parametrize("spacing, shape, extent", [
    (5.0, (8, 12), [0.0, 0.06, 0.0, 0.04]),
    (7.5, (6, 8), [0.0, 0.06, 0.0, 0.045]),
])
def test_independent_simulator_spacings_conserve_measured_geometry(spacing, shape, extent):
    facies = layered(dip=False)
    original = facies.copy()
    npz, text, _counts, _warnings = geometry.export_geometry(
        facies, ROLES, TABLE, width_m=WIDTH, height_m=HEIGHT,
        depth_points=FLAT_DEPTH, resolution_mm=spacing, fixture_polygons_m={}, **DECLARED,
    )
    document = json.loads(text)
    assert document["grid"]["shape"] == list(shape)
    assert document["grid"]["resolution_mm"] == spacing
    assert document["grid"]["extent_m"] == extent
    np.testing.assert_array_equal(facies, original)
    with np.load(io.BytesIO(npz), allow_pickle=False) as arrays:
        assert arrays["porosity"].shape == shape
        assert arrays["facies_ids"].tolist() == [1, 2, 3]
        domain_area = arrays["domain_area_m2"]
        sand = domain_area > 0
        # These totals come from the 35 x 60 native sand cells, their three
        # declared layer porosities and the measured constant 1 cm depth.
        assert domain_area.sum(dtype=np.float64) == pytest.approx(0.0021, rel=1e-5)
        assert arrays["bulk_volume_m3"].sum(dtype=np.float64) == pytest.approx(2.1e-5, rel=1e-5)
        assert arrays["pore_volume_m3"].sum(dtype=np.float64) == pytest.approx(6.3e-6, rel=1e-5)
        np.testing.assert_allclose(arrays["depth_m"][sand], 0.01, rtol=1e-5)
        np.testing.assert_allclose(
            arrays["porosity"][sand],
            arrays["pore_volume_m3"][sand] / arrays["bulk_volume_m3"][sand], rtol=1e-5,
        )
        assert arrays["facies_majority"][0, 0] == 3
        assert arrays["porosity"][0, 0] == pytest.approx(0.2)
        fractions = arrays["facies_area_fractions"]
        np.testing.assert_allclose(fractions[:, sand].sum(axis=0), 1.0, rtol=1e-5)
        for index, facies_id in enumerate(arrays["facies_ids"]):
            area = np.sum(fractions[index][sand] * domain_area[sand], dtype=np.float64)
            assert area == pytest.approx(np.count_nonzero(original == facies_id) * 1e-6, rel=1e-5)
        assert not sand[-1].any()
        assert np.isnan(arrays["porosity"][~sand]).all()
        assert np.isnan(arrays["depth_m"][~sand]).all()
        for name in ("fixture_filled", "edge_filled", "unfilled"):
            assert not arrays[name].any()

    # The independent simulator request must not widen saved-map admission.
    x, y, area = geometry.canvas_coordinates(original.shape, WIDTH, HEIGHT)
    with pytest.raises(ValueError):
        storage_resolution.make_grid(
            x, y, np.full(original.shape, area), include=original > 0,
            extent_m=extent, storage_resolution_mm=spacing,
        )


@pytest.mark.parametrize("spacing", storage_resolution.RESOLUTIONS_MM)
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_simulator_saved_spacings_preserve_the_original_grid(monkeypatch, spacing, precision):
    x, y, _area = geometry.canvas_coordinates((4, 6), 0.006, 0.004)
    area = np.arange(1, 25, dtype=np.float64).reshape(4, 6) * 1e-7
    include = np.ones(x.shape, bool)
    include[0, ::2] = False
    originals = [value.copy() for value in (x, y, area, include)]
    options = {"include": include, "extent_m": (0.0, 0.006, 0.0, 0.004), "precision": precision}
    expected = storage_resolution.make_grid(x, y, area, storage_resolution_mm=spacing, **options)
    actual = geometry._simulator_grid(x, y, area, resolution_mm=spacing, **options)
    assert type(actual) is type(expected) is storage_resolution.MetricStorageGrid
    assert actual.metadata() == expected.metadata()
    assert vars(actual).keys() == vars(expected).keys()
    for name, value in vars(expected).items():
        if isinstance(value, np.ndarray):
            assert vars(actual)[name].dtype == value.dtype
            np.testing.assert_array_equal(vars(actual)[name], value)
        else:
            assert vars(actual)[name] == value
    for value, original in zip((x, y, area, include), originals):
        np.testing.assert_array_equal(value, original)

    export_options = {
        "width_m": WIDTH, "height_m": HEIGHT, "depth_points": FLAT_DEPTH,
        "resolution_mm": spacing, "precision": precision, "fixture_polygons_m": {}, **DECLARED,
    }
    actual_export = geometry.export_geometry(layered(dip=False), ROLES, TABLE, **export_options)

    def original_grid(x_m, y_m, pixel_area_m2, *, resolution_mm, **options):
        return storage_resolution.make_grid(
            x_m, y_m, pixel_area_m2, storage_resolution_mm=resolution_mm, **options,
        )

    # The oracle changes only the constructor route back to the original.
    # Measured geometry, aggregation, precision casts and export remain real.
    with monkeypatch.context() as reference:
        reference.setattr(geometry, "_simulator_grid", original_grid)
        expected_export = geometry.export_geometry(layered(dip=False), ROLES, TABLE, **export_options)
    assert actual_export[1:] == expected_export[1:]
    with np.load(io.BytesIO(actual_export[0]), allow_pickle=False) as actual_arrays, \
            np.load(io.BytesIO(expected_export[0]), allow_pickle=False) as expected_arrays:
        assert actual_arrays.files == expected_arrays.files
        for name in actual_arrays.files:
            actual_array, expected_array = actual_arrays[name], expected_arrays[name]
            assert actual_array.dtype == expected_array.dtype, name
            assert actual_array.shape == expected_array.shape, name
            np.testing.assert_array_equal(np.isnan(actual_array), np.isnan(expected_array))
            assert actual_array.tobytes() == expected_array.tobytes(), name


@pytest.mark.parametrize("spacing", [None, "native", "5", True, False, 0, -1, np.nan, np.inf, -np.inf, 5e-324])
def test_simulator_resolution_is_explicit_finite_and_positive(spacing):
    x, y, area = geometry.canvas_coordinates((2, 2), 0.01, 0.01)
    options = {"include": np.ones(x.shape, bool), "extent_m": (0.0, 0.01, 0.0, 0.01)}
    with pytest.raises(ValueError):
        geometry._simulator_grid(x, y, np.full(x.shape, area), resolution_mm=spacing, **options)
    with pytest.raises(ValueError):
        geometry.covering_extent(0.01, 0.01, spacing)
    if spacing is None:
        with pytest.raises(TypeError):
            geometry._simulator_grid(x, y, np.full(x.shape, area), **options)
        with pytest.raises(TypeError):
            geometry.covering_extent(0.01, 0.01)


@pytest.mark.parametrize("width, height", [
    (0, 0.01), (-0.01, 0.01), (np.nan, 0.01), (np.inf, 0.01), (True, 0.01),
    (0.01, 0), (0.01, -0.01), (0.01, np.nan), (0.01, np.inf), (0.01, False),
])
def test_simulator_covering_extent_refuses_invalid_canvas_dimensions(width, height):
    with pytest.raises(ValueError):
        geometry.covering_extent(width, height, 5.0)


@pytest.mark.parametrize("invalid", [
    "shape", "empty_include", "integer_include", "negative_area", "nonfinite_area",
    "excluded_nan", "excluded_upper_edge", "excluded_below_extent", "bad_precision",
    "nonfinite_extent", "zero_extent", "reversed_extent", "off_lattice_extent", "too_many_cells",
])
def test_simulator_grid_refuses_invalid_admission_before_reduction(monkeypatch, invalid):
    x, y, area = geometry.canvas_coordinates((2, 2), 0.01, 0.01)
    area = np.full(x.shape, area)
    include = np.array([[True, False], [True, True]])
    extent = (0.0, 0.01, 0.0, 0.01)
    precision = "float32"
    if invalid == "shape":
        area = area[:, :1]
    elif invalid == "empty_include":
        include[:] = False
    elif invalid == "integer_include":
        include = include.astype(np.int64)
    elif invalid == "negative_area":
        area[0, 0] = -1
    elif invalid == "nonfinite_area":
        area[0, 0] = np.inf
    elif invalid == "excluded_nan":
        x[0, 1] = np.nan
    elif invalid == "excluded_upper_edge":
        x[0, 1] = extent[1]
    elif invalid == "excluded_below_extent":
        y[0, 1] = -0.001
    elif invalid == "bad_precision":
        precision = "float16"
    elif invalid == "nonfinite_extent":
        extent = (0.0, np.inf, 0.0, 0.01)
    elif invalid == "zero_extent":
        extent = (0.0, 0.0, 0.0, 0.01)
    elif invalid == "reversed_extent":
        extent = (0.01, 0.0, 0.0, 0.01)
    elif invalid == "off_lattice_extent":
        extent = (0.0, 0.011, 0.0, 0.01)
    elif invalid == "too_many_cells":
        extent = (0.0, 100.005, 0.0, 100.005)  # 20,001 squared cells, but only four native inputs.

    def forbidden_reduction(*args, **kwargs):
        pytest.fail("Invalid simulator grid reached reduction/allocation")

    # Keep even a broken cell-bound guard finite: no oversized bincount may run.
    monkeypatch.setattr(geometry.np, "bincount", forbidden_reduction)
    with pytest.raises(ValueError):
        geometry._simulator_grid(x, y, area, include=include, extent_m=extent,
                                 resolution_mm=5.0, precision=precision)


def test_the_covering_extent_lies_on_the_storage_lattice():
    assert geometry.covering_extent(0.06, 0.04, 2.0) == (0.0, 0.06, 0.0, 0.04)
    assert geometry.covering_extent(0.061, 0.0405, 2.0) == (0.0, 0.062, 0.0, 0.042)
    assert geometry.covering_extent(0.9, 0.49, 1.0) == (0.0, 0.9, 0.0, 0.49)


def test_the_facies_table_merges_the_optional_columns(tmp_path):
    path = tmp_path / "properties.csv"
    path.write_text("facies_id,name,swi,grain_size,source\n1,Upper,0.1,0.8,sieve\n2,Middle,,coarse,\n",
                    encoding="utf-8")
    extra = geometry.read_property_columns(path)
    assert extra == [{"facies_id": 1, "swi": 0.1, "grain_size": 0.8, "source": "sieve"},
                     {"facies_id": 2, "swi": None, "grain_size": "coarse", "source": None}]
    rows = geometry.facies_table([{"id": 1, "name": "Upper", "porosity": 0.4, "permeability": 10.0,
                                   "analysis_role": "porous_medium", "permeability_unit": "D"},
                                  {"id": 3, "name": "Lower", "porosity": 0.2, "permeability": None,
                                   "analysis_role": "seal", "permeability_unit": "D"}], extra)
    assert rows[0]["swi"] == 0.1 and rows[0]["source"] == "sieve" and rows[1]["swi"] is None
    assert geometry.read_property_columns(tmp_path / "missing.csv") == []


def write_revision(folder, facies, *, run_labels=None):
    """A geometry revision: labels (one label per facies here), regions with roles and a properties table."""
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "labels.npz", array=facies.astype(np.int32))
    (folder / "facies.csv").write_text("id,swi,grain_size,source\n1,0.1,0.5,sieve\n", encoding="utf-8")
    regions = [{"label_id": key, "facies_id": key, "name": f"Layer {key}", "analysis_role": role,
                "porosity": None if role == "free_water" else 0.4, "permeability": 5.0, "enabled": True}
               for key, role in ROLES.items()]
    configuration = {"format": "darsia-fluidflower-geometry-config-v1", "width_m": WIDTH, "height_m": HEIGHT,
                     "coordinate_convention": "origin_lower_left_x_right_y_up", "grouping_mode": "shared_facies",
                     "labels_path": str(folder / "labels.npz"), "regions": regions, "permeability_unit": "mD"}
    artifacts = {"facies_properties_path": str(folder / "facies.csv"), "run_geometry": {}}
    if run_labels is not None:
        np.savez(folder / "labels_r2.npz", array=run_labels.astype(np.int32))
        artifacts["run_geometry"]["r2"] = {"labels_path": str(folder / "labels_r2.npz")}
    revision = {"format": "darsia-fluidflower-geometry-facies-v1", "revision_id": "rev-1",
                "configuration": configuration, "artifacts": artifacts}
    (folder / "geometry.json").write_text(json.dumps(revision), encoding="utf-8")
    return revision


def test_a_run_reads_its_own_or_the_series_geometry(tmp_path):
    other = layered(dip=False)
    other[20:, :] = 3
    revision = write_revision(tmp_path / "geometry", layered(), run_labels=other)
    facies, roles, table, source = geometry.run_geometry(revision, "r1")
    assert np.array_equal(facies, layered()) and roles[0] == "free_water" and roles[2] == "porous_medium"
    unlabelled = layered()
    unlabelled[35:, :3] = 42  # A label without a region: undefined, never facies 0 (free water here).
    revision = write_revision(tmp_path / "geometry-2", unlabelled)
    facies, roles, _table, _source = geometry.run_geometry(revision, "r1")
    assert np.all(facies[35:, :3] == 4) and 4 not in roles
    completed = geometry.complete_geometry(facies, roles, width_m=WIDTH, height_m=HEIGHT, **DECLARED,
                                           fixture_polygons_m={})
    assert completed.counts["edge_filled_cells_by_role"] == {"undefined": 15}
    assert np.array_equal(completed.facies, layered())
    revision = write_revision(tmp_path / "geometry", layered(), run_labels=other)
    assert source == {"scope": "series_reference", "labels": "labels.npz", "revision": "rev-1"}
    upper = next(row for row in table if row["facies_id"] == 1)
    assert upper["swi"] == 0.1 and upper["source"] == "sieve" and upper["permeability_unit"] == "mD"
    facies, _roles, _table, source = geometry.run_geometry(revision, "R2")
    assert np.array_equal(facies, other) and source["scope"] == "run_override" and source["labels"] == "labels_r2.npz"
    with pytest.raises(ValueError, match="rig-corrected"):
        geometry.run_geometry(dict(revision, configuration=dict(revision["configuration"],
                                                                coordinate_convention="pixels")), "r1")


def test_depth_points_come_from_the_request_or_the_run_configuration(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "depth.csv").write_text("x,y,depth\n0,0,0.01\n1,0,0.02\n0,1,0.03\n", encoding="utf-8")
    inline = geometry.depth_points_for({"depth_points": FLAT_DEPTH}, tmp_path, {})
    assert inline.tolist() == FLAT_DEPTH
    from_file = geometry.depth_points_for({"depth_points": "config/depth.csv"}, tmp_path, {})
    assert from_file.tolist() == [[0, 0, 0.01], [1, 0, 0.02], [0, 1, 0.03]]
    config = {"mass": {"template": {"shared": {"depth_points": {"path": "config/depth.csv", "sha256": "0" * 64}}}}}
    assert geometry.depth_points_for({}, tmp_path, config).shape == (3, 3)
    with pytest.raises(ValueError, match="No measured depth points"):
        geometry.depth_points_for({}, tmp_path, {})
    with pytest.raises(ValueError):
        geometry.depth_points_for({"depth_points": "../outside.csv"}, tmp_path, {})


def test_the_request_names_its_geometry_and_known_fields_only(tmp_path):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"format": geometry.REQUEST_FORMAT, "geometry": "geometry/geometry.json",
                                "fixtures_m": FIXTURE, **DECLARED}), encoding="utf-8")
    assert geometry.load_request(path)["geometry"] == "geometry/geometry.json"
    path.write_text(json.dumps({"format": geometry.REQUEST_FORMAT, "geometry": "g.json", "colour": 1}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="known fields"):
        geometry.load_request(path)
    path.write_text(json.dumps({"format": "other", "geometry": "g.json"}), encoding="utf-8")
    with pytest.raises(ValueError, match="request"):
        geometry.load_request(path)


def test_continue_layers_leaves_a_region_without_sides_unfilled():
    facies = np.ones((6, 4), np.int64)
    hidden = np.zeros((6, 4), bool)
    hidden[:, :] = True
    result, filled = geometry.continue_layers(facies, ~hidden, hidden, np.linspace(0.006, 0.001, 6)[:, None]
                                              * np.ones((1, 4)))
    assert not filled.any() and np.array_equal(result, facies)


def test_replicate_edges_needs_donors_and_targets():
    facies = np.ones((4, 4), np.int64)
    x, y, _area = geometry.canvas_coordinates((4, 4), 0.004, 0.004)
    none = np.zeros((4, 4), bool)
    result, filled = geometry.replicate_edges(facies, none, ~none, x, y)
    assert not filled.any() and np.array_equal(result, facies)


def test_every_missing_declaration_refuses_with_its_reason():
    facies = layered(dip=False)
    with pytest.raises(ValueError, match="inner frame must be declared"):
        geometry.complete_geometry(facies, ROLES, width_m=WIDTH, height_m=HEIGHT, inner_frame="",
                                   sand_surface="free_water", fixture_polygons_m={})
    with pytest.raises(ValueError, match="no frame cell"):
        geometry.complete_geometry(facies, ROLES, width_m=WIDTH, height_m=HEIGHT, inner_frame="frame_role",
                                   sand_surface="free_water", fixture_polygons_m={})
    framed = facies.copy()
    framed[-1, :] = 8
    with pytest.raises(ValueError, match="frame cells"):
        geometry.complete_geometry(framed, {**ROLES, 8: "frame"}, width_m=WIDTH, height_m=HEIGHT, **DECLARED,
                                   fixture_polygons_m={})
    dry = facies.copy()
    dry[:5] = 1
    with pytest.raises(ValueError, match="no free-water cell"):
        geometry.complete_geometry(dry, ROLES, width_m=WIDTH, height_m=HEIGHT, **DECLARED, fixture_polygons_m={})
    with pytest.raises(ValueError, match="sand surface must be one of"):
        geometry.complete_geometry(facies, ROLES, width_m=WIDTH, height_m=HEIGHT, inner_frame="canvas",
                                   sand_surface="guess", fixture_polygons_m={})
    with pytest.raises(ValueError, match="No fixtures are declared"):
        geometry.complete_geometry(facies, ROLES, width_m=WIDTH, height_m=HEIGHT, **DECLARED)
    outside = {"chart": [(0.001, 0.0395), (0.005, 0.0395), (0.005, 0.0385), (0.001, 0.0385)]}  # In the water.
    with pytest.raises(ValueError, match="covers no cell of the porous domain"):
        geometry.complete_geometry(facies, ROLES, width_m=WIDTH, height_m=HEIGHT, **DECLARED,
                                   fixture_polygons_m=outside)
    zero = np.where(facies == 0, 5, facies)
    zero[10:12, :] = 0
    with pytest.raises(ValueError, match="Facies id 0 cannot be porous"):
        geometry.complete_geometry(zero, {0: "porous_medium", 1: "porous_medium", 2: "porous_medium",
                                          3: "porous_medium", 5: "free_water"}, width_m=WIDTH, height_m=HEIGHT,
                                   **DECLARED, fixture_polygons_m={})
    explicit = geometry.complete_geometry(facies, ROLES, width_m=WIDTH, height_m=HEIGHT, **DECLARED,
                                          fixture_polygons_m={})
    assert explicit.counts["fixture_filled_cells"] == 0 and explicit.counts["fixture_polygons"] == []


def test_the_facies_table_must_carry_porosity_permeability_and_a_stated_unit():
    facies = layered(dip=False)
    options = dict(width_m=WIDTH, height_m=HEIGHT, depth_points=FLAT_DEPTH, resolution_mm=2.0,
                   fixture_polygons_m={}, **DECLARED)
    for change, message in ((dict(porosity=None), "no porosity"), (dict(porosity=1.5), "no porosity"),
                            (dict(permeability=None), "no positive permeability"),
                            (dict(permeability_unit="source_unit"), "not stated")):
        table = [dict(row, **change) if row["facies_id"] == 2 else row for row in TABLE]
        with pytest.raises(ValueError, match=message):
            geometry.export_geometry(facies, ROLES, table, **options)
    unstated = [dict(row, permeability_unit="source_unit") for row in TABLE]
    _npz, text, _counts, _warnings = geometry.export_geometry(facies, ROLES, unstated, permeability_unit="D",
                                                              **options)
    assert {row["permeability_unit"] for row in json.loads(text)["facies_table"]} == {"D"}
    with pytest.raises(ValueError, match="the table states D, the request mD"):
        geometry.export_geometry(facies, ROLES, TABLE, permeability_unit="mD", **options)
    unused = TABLE + [{"facies_id": 9, "name": "Unused", "analysis_role": "porous_medium", "porosity": None,
                       "permeability": None, "permeability_unit": "", "swi": None, "grain_size": None,
                       "source": None}]
    geometry.export_geometry(facies, ROLES, unused, **options)  # A facies absent from the domain is not checked.


def test_too_few_depth_points_and_unfilled_cells_refuse():
    facies = layered(dip=False)
    options = dict(width_m=WIDTH, height_m=HEIGHT, resolution_mm=2.0, fixture_polygons_m={}, **DECLARED)
    for points in (FLAT_DEPTH[:2], [[0, 0, 0.01], [1, 0, 0.0], [0, 1, 0.01]]):
        with pytest.raises(ValueError, match="three measured depth points"):
            geometry.export_geometry(facies, ROLES, TABLE, depth_points=points, **options)
    wide = {"band": [(-0.001, 0.012), (0.0615, 0.012), (0.0615, 0.022), (-0.001, 0.022)]}  # Wall to wall.
    with pytest.raises(ValueError, match="have no geometry"):
        geometry.export_geometry(facies, ROLES, TABLE, depth_points=FLAT_DEPTH,
                                 **dict(options, fixture_polygons_m=wide))


def test_the_request_must_declare_the_frame_and_the_surface(tmp_path):
    path = tmp_path / "request.json"
    base = {"format": geometry.REQUEST_FORMAT, "geometry": "g.json"}
    path.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="inner_frame"):
        geometry.load_request(path)
    path.write_text(json.dumps(dict(base, inner_frame="canvas")), encoding="utf-8")
    with pytest.raises(ValueError, match="sand_surface"):
        geometry.load_request(path)
    path.write_text(json.dumps(dict(base, inner_frame="canvas", sand_surface="free_water",
                                    permeability_unit="D")), encoding="utf-8")
    assert geometry.load_request(path)["permeability_unit"] == "D"
